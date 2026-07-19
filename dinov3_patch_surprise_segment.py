#!/usr/bin/env python3
"""Reference-style DINOv3 patch surprise for a selected video segment.

Computes DINOv3 384px (24x24) patch tokens, current->previous DIS optical
flow, and an exact motion-aligned per-patch rolling diagonal-Gaussian NLL.
The overlay is composed at the source video resolution; a caller may downscale
only after this program has written the raw overlay video.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from surprise_normalization import DEFAULT_FLOW_ALPHA, motion_normalize

SIZE, PATCH = 384, 16
GRID = SIZE // PATCH
DIM = 768
LOG_2PI = math.log(2.0 * math.pi)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--history", type=int, default=300)
    p.add_argument("--patches", type=Path, required=True)
    p.add_argument("--flow", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--global-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flow-alpha", type=float, default=DEFAULT_FLOW_ALPHA, help="Default motion-normalization strength for the rendered heatmap.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def model_input(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb = np.stack([cv2.cvtColor(cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for x in frames])
    values = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((values - mean) / std).to(device)


def decode(cap: cv2.VideoCapture, count: int) -> list[np.ndarray]:
    values = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Source video ended before requested segment")
        values.append(frame)
    return values


def cache_patches(a: argparse.Namespace, n: int, device: torch.device) -> None:
    if a.patches.exists():
        return
    a.patches.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.patches.with_suffix(".partial.npy")
    tmp.unlink(missing_ok=True)
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, GRID, GRID, DIM))
    model = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0).eval().to(device)
    cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    index, batch = 0, 12
    while index < n:
        frames = decode(cap, min(batch, n - index))
        with torch.inference_mode():
            tokens = model.forward_features(model_input(frames, device))
            if isinstance(tokens, dict):
                tokens = tokens.get("x_norm_patchtokens", tokens.get("patch_tokens"))
                if tokens is None:
                    raise RuntimeError("DINOv3 did not return patch tokens")
            tokens = tokens[:, getattr(model, "num_prefix_tokens", 0):]
            if tokens.shape[1:] != (GRID * GRID, DIM):
                raise RuntimeError(f"Unexpected DINOv3 token geometry: {tuple(tokens.shape)}")
        cache[index:index + len(frames)] = tokens.reshape(len(frames), GRID, GRID, DIM).float().cpu().numpy().astype(np.float16)
        index += len(frames)
        print(f"DINOv3 patches {index}/{n}", flush=True)
    cap.release(); del cache
    os.replace(tmp, a.patches)


def cache_flow(a: argparse.Namespace, n: int) -> None:
    if a.flow.exists():
        return
    a.flow.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.flow.with_suffix(".partial.npy")
    tmp.unlink(missing_ok=True)
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n - 1, GRID, GRID, 2))
    cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    flow_model = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prior = None
    for index in range(n):
        frame = decode(cap, 1)[0]
        current = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prior is not None:
            dense = np.nan_to_num(flow_model.calc(current, prior, None))  # current -> previous
            pooled = dense.reshape(GRID, PATCH, GRID, PATCH, 2).mean(axis=(1, 3)) / PATCH
            cache[index - 1] = pooled.astype(np.float16)
        prior = current
        if index % 100 == 0 or index == n - 1:
            print(f"DIS flow {index + 1}/{n}", flush=True)
    cap.release(); del cache
    os.replace(tmp, a.flow)


def sample_grid(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(values.permute(2, 0, 1).unsqueeze(0), grid.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=True)[0].permute(1, 2, 0)


def sampling_grid(patch_flow: torch.Tensor) -> torch.Tensor:
    rows, cols = torch.meshgrid(torch.arange(GRID, device=patch_flow.device, dtype=torch.float32), torch.arange(GRID, device=patch_flow.device, dtype=torch.float32), indexing="ij")
    x, y = cols + patch_flow[..., 0], rows + patch_flow[..., 1]
    return torch.stack((x / (GRID - 1) * 2.0 - 1.0, y / (GRID - 1) * 2.0 - 1.0), dim=-1)


def border(score: float) -> tuple[int, int, int]:
    if not np.isfinite(score): return (90, 90, 90)
    if score >= 2.0: return (0, 0, 255)
    if score >= 1.35: return (0, 220, 255)
    return (0, 210, 0)


def compose(frame: np.ndarray, patch_map: np.ndarray, score: float, timestamp: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(patch_map.astype(np.float32), (3, 3), 0)
    lo, hi = np.percentile(blurred, (5, 95))
    heat = np.zeros_like(blurred, np.uint8) if hi <= lo else (np.clip((blurred - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_TURBO)
    output = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), border(score), max(6, output.shape[1] // 160))
    text = "Global surprise: warming up" if not np.isfinite(score) else f"Global surprise: {score:.2f}"
    cv2.putText(output, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output, f"DINOv3 patch Gaussian surprise | {timestamp:.1f}s", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, f"DINOv3 patch Gaussian surprise | {timestamp:.1f}s", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def render(a: argparse.Namespace, n: int, fps: float, device: torch.device) -> None:
    patches, flow = np.load(a.patches, mmap_mode="r"), np.load(a.flow, mmap_mode="r")
    global_scores = np.load(a.global_cache)["scores"][a.start_frame:a.end_frame]
    cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    maps = np.lib.format.open_memmap(a.maps, mode="w+", dtype=np.float16, shape=(n, GRID, GRID))
    history: deque[torch.Tensor] = deque(maxlen=a.history)
    for index in range(n):
        frame = decode(cap, 1)[0]
        current = torch.from_numpy(np.array(patches[index], np.float32, copy=True)).to(device)
        if index:
            grid = sampling_grid(torch.from_numpy(np.array(flow[index - 1], np.float32, copy=True)).to(device))
            if history:
                stack = torch.stack(tuple(history)).permute(0, 3, 1, 2)
                aligned = F.grid_sample(stack, grid.unsqueeze(0).expand(len(history), -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True).permute(0, 2, 3, 1).contiguous()
                history = deque((aligned[i] for i in range(len(aligned))), maxlen=a.history)
        if len(history) == a.history:
            stacked = torch.stack(tuple(history))
            mean, var = stacked.mean(0), stacked.var(0, unbiased=False).clamp_min(1e-4)
            local = (0.5 * ((current - mean).square() / var + torch.log(var) + LOG_2PI)).mean(-1).cpu().numpy()
        else:
            local = np.zeros((GRID, GRID), np.float32)
        history.append(current)
        maps[index] = local.astype(np.float16)  # Preserve raw NLL as a diagnostic cache.
        display_local = motion_normalize(local, np.asarray(flow[index - 1], np.float32) if index else None, a.flow_alpha)
        writer.write(compose(frame, display_local, float(global_scores[index]), (a.start_frame + index) / fps))
        if index % 50 == 0 or index == n - 1: print(f"render {index + 1}/{n}", flush=True)
    cap.release(); writer.release(); del maps


def main() -> None:
    a = args()
    if not 0 <= a.start_frame < a.end_frame: raise ValueError("Invalid segment")
    cap = cv2.VideoCapture(str(a.video)); total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 15.0; cap.release()
    if a.end_frame > total: raise ValueError(f"Video has {total} frames")
    n = a.end_frame - a.start_frame
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    cache_patches(a, n, device); cache_flow(a, n); render(a, n, fps, device)
    a.output.with_suffix(".json").write_text(json.dumps({"model":"vit_base_patch16_dinov3", "input_size":SIZE, "patch_grid":GRID, "flow":"DIS current_to_previous", "score":"per-patch diagonal Gaussian NLL", "history":a.history, "start_frame":a.start_frame, "end_frame":a.end_frame, "overlay":"source-resolution; final encoding may downscale"}, indent=2) + "\n")

if __name__ == "__main__": main()
