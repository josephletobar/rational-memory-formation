#!/usr/bin/env python3
"""Visualize motion-aligned fading trails for the accepted RoC default."""
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
    parser.add_argument("--trail-decay", type=float, default=0.96, help="Per-frame retained trail energy after advection.")
    return parser.parse_args()


def warp_grid(previous: np.ndarray, current_to_previous_flow: np.ndarray) -> np.ndarray:
    """Bilinearly bring a prior 24x24 heat grid into current coordinates."""
    height, width = previous.shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = np.clip(x + current_to_previous_flow[..., 0], 0, width - 1)
    y = np.clip(y + current_to_previous_flow[..., 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    return ((1 - wx) * (1 - wy) * previous[y0, x0] + wx * (1 - wy) * previous[y0, x1]
            + (1 - wx) * wy * previous[y1, x0] + wx * wy * previous[y1, x1])


def overlay(frame: np.ndarray, score: np.ndarray, lo: float, hi: float, label: str) -> np.ndarray:
    value = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    value = cv2.GaussianBlur(value, (3, 3), 0)
    heat = cv2.applyColorMap(
        cv2.resize((value * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC),
        cv2.COLORMAP_TURBO,
    )
    output = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    if not 0 < args.display_ema <= 1 or not 0 <= args.trail_decay <= 1:
        raise ValueError("--display-ema must be in (0, 1] and --trail-decay in [0, 1]")
    maps = np.load(args.maps, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    roc = np.load(args.roc, mmap_mode="r")
    offset, count = args.output_start - args.segment_start, args.output_end - args.output_start
    if roc.shape[0] != count:
        raise ValueError("RoC cache does not match requested output range")
    raw_all = np.asarray(maps[offset:offset + count], dtype=np.float32)
    lo, hi = raw_global_display_scale(raw_all)
    roc_reference = float(np.mean(roc, dtype=np.float64))
    print(f"raw p5/p95={lo:.4f}/{hi:.4f}; RoC reference={roc_reference:.5f}", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.output_start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    displayed = trail = None
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("video ended early")
        j = offset + index
        score = np.asarray(maps[j], dtype=np.float32) / (1.0 + args.alpha * np.linalg.norm(np.asarray(flow[j - 1], dtype=np.float32), axis=-1))
        score *= np.asarray(roc[index], dtype=np.float32) / max(roc_reference, 1e-8)
        displayed = score if displayed is None else args.display_ema * score + (1.0 - args.display_ema) * displayed
        if trail is None:
            trail = displayed.copy()
        else:
            carried = args.trail_decay * warp_grid(trail, np.asarray(flow[j - 1], dtype=np.float32))
            trail = np.maximum(carried, displayed)
        writer.write(cv2.hconcat((
            overlay(frame, displayed, lo, hi, "Default: RoC + display EMA"),
            overlay(frame, trail, lo, hi, f"Motion-aligned fading trails (decay={args.trail_decay:g})"),
        )))
        if index % 150 == 0:
            print(f"rendered {index + 1}/{count}", flush=True)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
