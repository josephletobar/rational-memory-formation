#!/usr/bin/env python3
"""Compare motion-aligned DINO patch surprise across history lengths.

Uses cached patch tokens and cached current-to-previous flow only: no encoder
inference.  Each panel has its own rolling diagonal Gaussian, all rendered
with the established flow attenuation and display-only temporal EMA.
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from surprise_normalization import DEFAULT_FLOW_ALPHA, raw_global_display_scale

LOG_2PI = math.log(2.0 * math.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("patches", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("raw_maps", type=Path, help="Output .npy cache with shape (windows, frames, H, W).")
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment-start", type=int, required=True)
    parser.add_argument("--output-start", type=int, required=True)
    parser.add_argument("--output-end", type=int, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=(32, 128, 300, 600))
    parser.add_argument("--alpha", type=float, default=DEFAULT_FLOW_ALPHA)
    parser.add_argument("--display-ema", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sampling_grid(patch_flow: torch.Tensor) -> torch.Tensor:
    height, width = patch_flow.shape[:2]
    rows, cols = torch.meshgrid(
        torch.arange(height, device=patch_flow.device, dtype=torch.float32),
        torch.arange(width, device=patch_flow.device, dtype=torch.float32),
        indexing="ij",
    )
    x, y = cols + patch_flow[..., 0], rows + patch_flow[..., 1]
    return torch.stack((x / (width - 1) * 2.0 - 1.0, y / (height - 1) * 2.0 - 1.0), dim=-1)


def aligned_history(history: deque[torch.Tensor], flow: torch.Tensor) -> deque[torch.Tensor]:
    if not history:
        return history
    stack = torch.stack(tuple(history)).permute(0, 3, 1, 2)
    grid = sampling_grid(flow).unsqueeze(0).expand(len(history), -1, -1, -1)
    warped = F.grid_sample(stack, grid, mode="bilinear", padding_mode="border", align_corners=True).permute(0, 2, 3, 1).contiguous()
    return deque((warped[i] for i in range(len(warped))), maxlen=history.maxlen)


def nll(current: torch.Tensor, history: deque[torch.Tensor], window: int) -> np.ndarray | None:
    if len(history) < window:
        return None
    stack = torch.stack(tuple(history)[-window:])
    mean = stack.mean(0)
    variance = stack.var(0, unbiased=False).clamp_min(1e-4)
    score = (0.5 * ((current - mean).square() / variance + torch.log(variance) + LOG_2PI)).mean(-1)
    return score.float().cpu().numpy()


def compute_raw_maps(args: argparse.Namespace, device: torch.device) -> np.ndarray:
    patches = np.load(args.patches, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    windows = tuple(args.windows)
    count = args.output_end - args.segment_start
    if patches.shape[0] < count or flow.shape[0] < count - 1:
        raise ValueError("Cached patches/flow do not cover requested segment")
    args.raw_maps.parent.mkdir(parents=True, exist_ok=True)
    maps = np.lib.format.open_memmap(args.raw_maps, mode="w+", dtype=np.float16, shape=(len(windows), count, *patches.shape[1:3]))
    history: deque[torch.Tensor] = deque(maxlen=max(windows))
    for index in range(count):
        current = torch.from_numpy(np.array(patches[index], dtype=np.float32, copy=True)).to(device)
        if index:
            history = aligned_history(history, torch.from_numpy(np.array(flow[index - 1], dtype=np.float32, copy=True)).to(device))
        for panel, window in enumerate(windows):
            score = nll(current, history, window)
            if score is not None:
                maps[panel, index] = score.astype(np.float16)
            else:
                maps[panel, index] = np.nan
        history.append(current)
        if index % 50 == 0 or index == count - 1:
            print(f"scored {index + 1}/{count}", flush=True)
    del maps
    return np.load(args.raw_maps, mmap_mode="r")


def panel(frame: np.ndarray, score: np.ndarray, lo: float, hi: float, label: str) -> np.ndarray:
    value = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    value = cv2.GaussianBlur(value, (3, 3), 0)
    heat = cv2.applyColorMap(
        cv2.resize((value * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC),
        cv2.COLORMAP_TURBO,
    )
    output = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def render(args: argparse.Namespace, maps: np.ndarray) -> None:
    windows = tuple(args.windows)
    offset = args.output_start - args.segment_start
    count = args.output_end - args.output_start
    raw_for_scale = maps[:, offset : offset + count]
    valid = raw_for_scale[np.isfinite(raw_for_scale)]
    lo, hi = raw_global_display_scale(valid)
    print(f"shared raw-score display p5/p95={lo:.5f}/{hi:.5f}", flush=True)
    flow = np.load(args.flow, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.output_start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(windows), height))
    ema: list[np.ndarray | None] = [None] * len(windows)
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("video ended early")
        j = offset + index
        flow_magnitude = np.linalg.norm(np.asarray(flow[j - 1], dtype=np.float32), axis=-1)
        outputs = []
        for panel_index, window in enumerate(windows):
            raw = np.asarray(maps[panel_index, j], dtype=np.float32)
            if not np.isfinite(raw).all():
                outputs.append(panel(frame, np.zeros_like(flow_magnitude), lo, hi, f"history {window}: warming up"))
                continue
            score = raw / (1.0 + args.alpha * flow_magnitude)
            ema[panel_index] = score if ema[panel_index] is None else args.display_ema * score + (1.0 - args.display_ema) * ema[panel_index]
            outputs.append(panel(frame, ema[panel_index], lo, hi, f"history {window} frames"))
        writer.write(cv2.hconcat(outputs))
        if index % 150 == 0 or index == count - 1:
            print(f"rendered {index + 1}/{count}", flush=True)
    cap.release()
    writer.release()


def main() -> None:
    args = parse_args()
    if not 0 < args.display_ema <= 1:
        raise ValueError("--display-ema must be in (0, 1]")
    if args.segment_start > args.output_start or args.output_start >= args.output_end:
        raise ValueError("Invalid segment/output bounds")
    if any(window < 2 for window in args.windows):
        raise ValueError("Every history window must be at least two frames")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    maps = compute_raw_maps(args, device)
    render(args, maps)


if __name__ == "__main__":
    main()
