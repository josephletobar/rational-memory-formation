#!/usr/bin/env python3
"""Cyberpunk display treatment for the canonical RoC-enhanced surprise map.

The score is unchanged: 300-frame patch Gaussian, alpha-2 flow attenuation,
and motion-aligned patch RoC multiplier.  Styling is display-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from surprise_normalization import DEFAULT_DISPLAY_EMA, DEFAULT_FLOW_ALPHA, raw_global_display_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("maps", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("roc", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment-start", type=int, required=True)
    parser.add_argument("--output-start", type=int, required=True)
    parser.add_argument("--output-end", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_FLOW_ALPHA)
    parser.add_argument("--display-ema", type=float, default=DEFAULT_DISPLAY_EMA)
    parser.add_argument("--trail-decay", type=float, default=0.94)
    return parser.parse_args()


def warp_scalar(previous: np.ndarray, current_to_previous_flow: np.ndarray) -> np.ndarray:
    """Bring a prior patch-grid scalar field into current-frame coordinates."""
    height, width = previous.shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = np.clip(x + current_to_previous_flow[..., 0], 0, width - 1)
    y = np.clip(y + current_to_previous_flow[..., 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    return ((1 - wx) * (1 - wy) * previous[y0, x0] + wx * (1 - wy) * previous[y0, x1]
            + (1 - wx) * wy * previous[y1, x0] + wx * wy * previous[y1, x1])


def upsample(field: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.GaussianBlur(cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC), (0, 0), 2.2)


def style(frame: np.ndarray, current: np.ndarray, trail: np.ndarray, arrival: np.ndarray, width: int, height: int) -> np.ndarray:
    current, trail, arrival = (upsample(value, width, height)[..., None] for value in (current, trail, arrival))
    base = frame.astype(np.float32) * 0.31
    # BGR neon field: cyan persistence plus magenta at newly arriving novelty.
    glow = np.concatenate((255 * trail, 220 * trail + 35 * current, 55 * trail + 255 * arrival), axis=-1)
    output = np.clip(base + 0.84 * glow, 0, 255).astype(np.uint8)
    # Scanlines keep the image textured without changing the underlying field.
    output[::4] = (output[::4].astype(np.float32) * 0.72).astype(np.uint8)
    # A faint, pulsing DINO patch lattice makes movement between semantic cells legible.
    lattice_alpha = int(25 + 55 * float(np.clip(current.mean() * 3.0, 0, 1)))
    overlay = output.copy()
    for x in np.linspace(0, width - 1, 25, dtype=int): cv2.line(overlay, (x, 0), (x, height - 1), (255, 230, 40), 1)
    for y in np.linspace(0, height - 1, 25, dtype=int): cv2.line(overlay, (0, y), (width - 1, y), (255, 230, 40), 1)
    output = cv2.addWeighted(overlay, lattice_alpha / 255.0, output, 1 - lattice_alpha / 255.0, 0)
    border = (255, int(100 + 155 * float(np.clip(trail.mean() * 3, 0, 1))), 255)
    cv2.rectangle(output, (0, 0), (width - 1, height - 1), border, max(4, width // 260))
    label = "SEMANTIC FIELD // FLOW-ALIGNED RoC"
    cv2.putText(output, label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (40, 10, 20), 1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    if not 0 < args.display_ema <= 1 or not 0 < args.trail_decay < 1:
        raise ValueError("EMA must be in (0,1] and trail decay in (0,1)")
    maps = np.load(args.maps, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    roc = np.load(args.roc, mmap_mode="r")
    offset, count = args.output_start - args.segment_start, args.output_end - args.output_start
    if roc.shape[0] != count:
        raise ValueError("RoC cache must match the output interval")
    lo, hi = raw_global_display_scale(np.asarray(maps[offset:offset + count], dtype=np.float32))
    roc_reference = float(np.mean(roc, dtype=np.float64))
    print(f"raw display p5/p95={lo:.4f}/{hi:.4f}; RoC reference={roc_reference:.5f}", flush=True)
    cap = cv2.VideoCapture(str(args.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, args.output_start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    score_ema = trail = None
    for index in range(count):
        ok, frame = cap.read()
        if not ok: raise RuntimeError("video ended early")
        j = offset + index
        score = np.asarray(maps[j], dtype=np.float32) / (1 + args.alpha * np.linalg.norm(np.asarray(flow[j - 1], dtype=np.float32), axis=-1))
        score *= np.asarray(roc[index], dtype=np.float32) / max(roc_reference, 1e-8)
        score_ema = score if score_ema is None else args.display_ema * score + (1 - args.display_ema) * score_ema
        current = np.clip((score_ema - lo) / max(hi - lo, 1e-8), 0, 1)
        previous_trail = np.zeros_like(current) if trail is None else warp_scalar(trail, np.asarray(flow[j - 1], dtype=np.float32))
        arrival = np.maximum(current - previous_trail, 0)
        trail = np.maximum(current, args.trail_decay * previous_trail)
        writer.write(style(frame, current, trail, arrival, width, height))
        if index % 150 == 0: print(f"rendered {index + 1}/{count}", flush=True)
    cap.release(); writer.release()


if __name__ == "__main__":
    main()
