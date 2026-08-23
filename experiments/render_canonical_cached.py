#!/usr/bin/env python3
"""Re-render cached canonical surprise maps without recomputing DINO or flow."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

GRID = 24
FLOW_ALPHA = 2.0
DISPLAY_EMA = 0.25


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("maps", type=Path)
    p.add_argument("flow", type=Path)
    p.add_argument("roc", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--heat-max", type=float, default=3e-5)
    return p.parse_args()


def overlay(frame: np.ndarray, score: np.ndarray, warming: bool, heat_max: float) -> np.ndarray:
    if warming:
        out = frame.copy()
        text = "Canonical flow-aligned W_t vs W_t-1: warming up 300 frames"
    else:
        value = cv2.GaussianBlur(
            np.clip(score / heat_max, 0, 1).astype(np.float32), (3, 3), 0
        )
        heat = cv2.applyColorMap(
            cv2.resize((value * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC),
            cv2.COLORMAP_TURBO,
        )
        out = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
        text = f"Canonical flow + RoC W_t vs W_t-1 | fixed scale 0..{heat_max:.0e}"
    cv2.putText(out, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .54, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .54, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    a = args()
    maps = np.load(a.maps, mmap_mode="r")
    flows = np.load(a.flow, mmap_mode="r")
    rocs = np.load(a.roc, mmap_mode="r")
    cap = cv2.VideoCapture(str(a.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total != len(maps):
        raise ValueError(f"video has {total} frames but cache has {len(maps)}")
    ref = float(np.mean(rocs[a.window:], dtype=np.float64))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ema = None
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("source ended during render")
        warming = i < a.window or not np.isfinite(maps[i]).all()
        if warming:
            score = np.zeros((GRID, GRID), np.float32)
        else:
            score = np.asarray(maps[i], np.float32) / (1 + FLOW_ALPHA * np.linalg.norm(np.asarray(flows[i - 1], np.float32), axis=-1))
            score *= np.asarray(rocs[i], np.float32) / max(ref, 1e-8)
            ema = score if ema is None else DISPLAY_EMA * score + (1 - DISPLAY_EMA) * ema
            score = ema
        writer.write(overlay(frame, score, warming, a.heat_max))
        if (i + 1) % 150 == 0 or i + 1 == total:
            print(f"rendered {i + 1}/{total}", flush=True)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
