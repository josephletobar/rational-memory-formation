#!/usr/bin/env python3
"""Render the canonical 24x24 surprise maps with RoC deliberately disabled."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from surprise_normalization import DEFAULT_DISPLAY_EMA, DEFAULT_FLOW_ALPHA, raw_global_display_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("raw_maps", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("raw_video", type=Path)
    parser.add_argument("--seconds", type=float, default=0, help="0 renders the complete source video.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_FLOW_ALPHA)
    parser.add_argument("--display-ema", type=float, default=DEFAULT_DISPLAY_EMA)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--global-lo", type=float, default=None, help="Shared fixed lower display bound.")
    parser.add_argument("--global-hi", type=float, default=None, help="Shared fixed upper display bound.")
    return parser.parse_args()


def overlay(frame: np.ndarray, score: np.ndarray, lo: float, hi: float, warming: bool) -> np.ndarray:
    if warming:
        out = frame.copy()
        text = "24x24 canonical (no RoC): warming up 300-frame history"
    else:
        value = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        value = cv2.GaussianBlur(value, (3, 3), 0)
        heat = cv2.applyColorMap(
            cv2.resize((value * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC),
            cv2.COLORMAP_TURBO,
        )
        out = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
        text = f"Base canonical: flow + display EMA (no RoC) | shared scale {lo:g}..{hi:g}"
    cv2.putText(out, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    raw, flow = np.load(args.raw_maps, mmap_mode="r"), np.load(args.flow, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    count = min(int(round(fps * args.seconds)), len(raw)) if args.seconds > 0 else len(raw)
    valid = np.asarray(raw[args.history:count], dtype=np.float32)
    local_lo, local_hi = raw_global_display_scale(valid[np.isfinite(valid)])
    if (args.global_lo is None) != (args.global_hi is None):
        raise ValueError("--global-lo and --global-hi must be provided together")
    lo, hi = (local_lo, local_hi) if args.global_lo is None else (args.global_lo, args.global_hi)
    print(f"display scale={lo:.6g}..{hi:.6g} (local p5/p95={local_lo:.6g}..{local_hi:.6g})", flush=True)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.raw_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ema = None
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Source video ended early")
        warm = index < args.history or not np.isfinite(raw[index]).all()
        if warm:
            score = np.zeros(raw.shape[1:], dtype=np.float32)
        else:
            score = np.asarray(raw[index], dtype=np.float32) / (1.0 + args.alpha * np.linalg.norm(np.asarray(flow[index - 1], dtype=np.float32), axis=-1))
            ema = score if ema is None else args.display_ema * score + (1.0 - args.display_ema) * ema
            score = ema
        writer.write(overlay(frame, score, lo, hi, warm))
        if (index + 1) % 150 == 0 or index + 1 == count:
            print(f"rendered {index + 1}/{count}", flush=True)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
