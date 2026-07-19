#!/usr/bin/env python3
"""Low-storage full-video renderer for the accepted canonical surprise map.

The model is DINOv3 at 384px / 24x24 patches.  It streams a video through the
encoder and keeps only the 300-frame motion-aligned history on GPU; it stores
the small raw-NLL, patch-flow, and RoC caches needed for a second render pass.
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

from surprise_normalization import DEFAULT_DISPLAY_EMA, DEFAULT_FLOW_ALPHA, raw_global_display_scale

SIZE, PATCH, GRID, DIM = 384, 16, 24, 768
LOG_2PI = math.log(2.0 * math.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("raw_maps", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("roc", type=Path)
    parser.add_argument("raw_video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=DEFAULT_FLOW_ALPHA)
    parser.add_argument("--display-ema", type=float, default=DEFAULT_DISPLAY_EMA)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-only", action="store_true", help="Write caches but skip the render pass")
    return parser.parse_args()


def model_input(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb = np.stack([
        cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for frame in frames
    ])
    values = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((values - mean) / std).to(device)


def patch_flow(current: np.ndarray, previous: np.ndarray, model: cv2.DISOpticalFlow) -> np.ndarray:
    dense = np.nan_to_num(model.calc(current, previous, None))  # current -> previous
    return dense.reshape(GRID, PATCH, GRID, PATCH, 2).mean(axis=(1, 3)).astype(np.float32) / PATCH


def sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.arange(GRID, device=flow.device, dtype=torch.float32),
        torch.arange(GRID, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    x, y = cols + flow[..., 0], rows + flow[..., 1]
    return torch.stack((x / (GRID - 1) * 2.0 - 1.0, y / (GRID - 1) * 2.0 - 1.0), dim=-1)


def warp_history(history: deque[torch.Tensor], flow: torch.Tensor) -> deque[torch.Tensor]:
    if not history:
        return history
    stack = torch.stack(tuple(history)).permute(0, 3, 1, 2)
    grid = sampling_grid(flow).unsqueeze(0).expand(len(history), -1, -1, -1)
    warped = F.grid_sample(stack, grid, mode="bilinear", padding_mode="border", align_corners=True).permute(0, 2, 3, 1).contiguous()
    return deque((warped[index] for index in range(len(warped))), maxlen=history.maxlen)


def rate_of_change(current: torch.Tensor, aligned_previous: torch.Tensor) -> np.ndarray:
    current = current / (current.norm(dim=-1, keepdim=True) + 1e-8)
    aligned_previous = aligned_previous / (aligned_previous.norm(dim=-1, keepdim=True) + 1e-8)
    return (1.0 - (current * aligned_previous).sum(-1)).clamp(0, 2).float().cpu().numpy()


def compute_caches(args: argparse.Namespace, total: int, device: torch.device) -> None:
    if args.raw_maps.exists() and args.flow.exists() and args.roc.exists():
        print("all canonical caches already exist", flush=True)
        return
    for path in (args.raw_maps, args.flow, args.roc):
        path.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp = args.raw_maps.with_suffix(".partial.npy")
    flow_tmp = args.flow.with_suffix(".partial.npy")
    roc_tmp = args.roc.with_suffix(".partial.npy")
    for path in (raw_tmp, flow_tmp, roc_tmp):
        path.unlink(missing_ok=True)
    raw_cache = np.lib.format.open_memmap(raw_tmp, mode="w+", dtype=np.float16, shape=(total, GRID, GRID))
    flow_cache = np.lib.format.open_memmap(flow_tmp, mode="w+", dtype=np.float16, shape=(total - 1, GRID, GRID, 2))
    roc_cache = np.lib.format.open_memmap(roc_tmp, mode="w+", dtype=np.float16, shape=(total, GRID, GRID))
    cap = cv2.VideoCapture(str(args.video))
    model = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0).eval().to(device)
    flow_model = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    history: deque[torch.Tensor] = deque(maxlen=args.history)
    prior_gray = None
    frame_index = 0
    with torch.inference_mode():
        while frame_index < total:
            frames = []
            for _ in range(min(args.batch_size, total - frame_index)):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Source video ended early")
                frames.append(frame)
            features = model.forward_features(model_input(frames, device))
            if isinstance(features, dict):
                features = features.get("x_norm_patchtokens", features.get("patch_tokens"))
            if features is not None:
                features = features[:, getattr(model, "num_prefix_tokens", 0):]
            if features is None or features.shape[1:] != (GRID * GRID, DIM):
                raise RuntimeError(f"Unexpected DINOv3 patch-token shape: {None if features is None else tuple(features.shape)}")
            tokens = features.reshape(len(frames), GRID, GRID, DIM)
            for frame, current in zip(frames, tokens):
                gray = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                aligned_previous = None
                if prior_gray is not None:
                    current_flow = patch_flow(gray, prior_gray, flow_model)
                    flow_cache[frame_index - 1] = current_flow.astype(np.float16)
                    history = warp_history(history, torch.from_numpy(current_flow).to(device))
                    aligned_previous = history[-1]
                if len(history) == args.history:
                    stacked = torch.stack(tuple(history))
                    mean = stacked.mean(0)
                    variance = stacked.var(0, unbiased=False).clamp_min(1e-4)
                    raw = (0.5 * ((current - mean).square() / variance + torch.log(variance) + LOG_2PI)).mean(-1)
                    raw_cache[frame_index] = raw.float().cpu().numpy().astype(np.float16)
                else:
                    raw_cache[frame_index] = np.nan
                roc_cache[frame_index] = (rate_of_change(current, aligned_previous) if aligned_previous is not None else np.zeros((GRID, GRID), np.float32)).astype(np.float16)
                history.append(current)
                prior_gray = gray
                frame_index += 1
                if frame_index % 50 == 0 or frame_index == total:
                    print(f"computed {frame_index}/{total}", flush=True)
    cap.release()
    del raw_cache, flow_cache, roc_cache, model
    torch.cuda.empty_cache()
    os.replace(raw_tmp, args.raw_maps)
    os.replace(flow_tmp, args.flow)
    os.replace(roc_tmp, args.roc)


def overlay(frame: np.ndarray, score: np.ndarray, lo: float, hi: float, warming: bool) -> np.ndarray:
    if warming:
        output = frame.copy()
        label = "Canonical surprise: warming up 300-frame history"
    else:
        value = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        value = cv2.GaussianBlur(value, (3, 3), 0)
        heat = cv2.applyColorMap(
            cv2.resize((value * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC),
            cv2.COLORMAP_TURBO,
        )
        output = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
        label = "Canonical surprise: flow + patch RoC + display EMA"
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def render(args: argparse.Namespace, total: int, fps: float) -> None:
    raw = np.load(args.raw_maps, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    roc = np.load(args.roc, mmap_mode="r")
    valid = np.asarray(raw[args.history:], dtype=np.float32)
    lo, hi = raw_global_display_scale(valid[np.isfinite(valid)])
    roc_reference = float(np.mean(roc[args.history:], dtype=np.float64))
    print(f"render scale raw p5/p95={lo:.4f}/{hi:.4f}; RoC reference={roc_reference:.5f}", flush=True)
    cap = cv2.VideoCapture(str(args.video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.raw_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    displayed = None
    for index in range(total):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Source video ended during render")
        warming = index < args.history or not np.isfinite(raw[index]).all()
        if warming:
            score = np.zeros((GRID, GRID), np.float32)
        else:
            score = np.asarray(raw[index], dtype=np.float32) / (1.0 + args.alpha * np.linalg.norm(np.asarray(flow[index - 1], dtype=np.float32), axis=-1))
            score *= np.asarray(roc[index], dtype=np.float32) / max(roc_reference, 1e-8)
            displayed = score if displayed is None else args.display_ema * score + (1.0 - args.display_ema) * displayed
            score = displayed
        writer.write(overlay(frame, score, lo, hi, warming))
        if (index + 1) % 150 == 0 or index + 1 == total:
            print(f"rendered {index + 1}/{total}", flush=True)
    cap.release()
    writer.release()


def main() -> None:
    args = parse_args()
    if args.history < 2 or not 0 < args.display_ema <= 1:
        raise ValueError("history must be >=2 and display EMA must be in (0,1]")
    cap = cv2.VideoCapture(str(args.video))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    if total <= args.history:
        raise ValueError("Video is shorter than the requested history")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    compute_caches(args, total, device)
    if args.compute_only:
        return
    render(args, total, fps)


if __name__ == "__main__":
    main()
