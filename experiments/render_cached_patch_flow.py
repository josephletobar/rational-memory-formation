#!/usr/bin/env python3
"""Render cached 24x24 patch-cell optical flow beside the source frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("flow", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--segment-start", type=int, required=True, help="Video frame matching flow[0]'s preceding frame.")
    p.add_argument("--output-start", type=int, required=True)
    p.add_argument("--output-end", type=int, required=True)
    return p.parse_args()


def flow_panel(patch_flow: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    grid_h, grid_w = patch_flow.shape[:2]
    dense = cv2.resize(patch_flow, (width, height), interpolation=cv2.INTER_CUBIC)
    # Values are patch-cell displacements; convert to source-pixel motion.
    dense[..., 0] *= width / grid_w
    dense[..., 1] *= height / grid_h
    mag, angle = cv2.cartToPolar(dense[..., 0], dense[..., 1], angleInDegrees=True)
    hsv = np.zeros((height, width, 3), np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / scale * 255, 0, 255).astype(np.uint8)
    panel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Arrows make direction legible without having to remember the color wheel.
    for row in range(1, grid_h, 2):
        for col in range(1, grid_w, 2):
            x, y = int((col + .5) * width / grid_w), int((row + .5) * height / grid_h)
            dx = float(patch_flow[row, col, 0] * width / grid_w)
            dy = float(patch_flow[row, col, 1] * height / grid_h)
            cv2.arrowedLine(panel, (x, y), (round(x + dx * 3), round(y + dy * 3)), (255, 255, 255), 1, cv2.LINE_AA, tipLength=.25)
    return panel


def main() -> None:
    a = args(); flow = np.load(a.flow, mmap_mode="r")
    if a.output_start <= a.segment_start or a.output_end > a.segment_start + len(flow) + 1:
        raise ValueError("Requested output range is outside cached flow")
    magnitude = np.linalg.norm(np.asarray(flow, np.float32), axis=-1)
    # Pixel displacement corresponding to the 99th-percentile patch motion.
    scale_cells = max(float(np.percentile(magnitude, 99)), 1e-3)
    cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.output_start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    for frame_index in range(a.output_start, a.output_end):
        ok, frame = cap.read()
        if not ok: raise RuntimeError(f"Video ended at {frame_index}")
        patch_flow = np.asarray(flow[frame_index - a.segment_start - 1], np.float32)
        panel = flow_panel(patch_flow, w, h, scale_cells * w / flow.shape[2])
        joined = cv2.hconcat((frame, panel))
        cv2.putText(joined, "Source", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(joined, "Source", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(joined, "Cached DIS optical flow (color: direction/magnitude; arrows: direction)", (w + 18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(joined, "Cached DIS optical flow (color: direction/magnitude; arrows: direction)", (w + 18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(joined)
        if (frame_index - a.output_start) % 150 == 0: print(f"rendered {frame_index - a.output_start + 1}/{a.output_end-a.output_start}", flush=True)
    cap.release(); writer.release()


if __name__ == "__main__": main()
