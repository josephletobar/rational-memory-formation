#!/usr/bin/env python3
"""Show whether cached patch-grid flow reconstructs the next video frame."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("flow", type=Path, help="Current-to-previous 24x24 flow in patch-cell units.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=60.0)
    return parser.parse_args()


def warp_previous(previous: np.ndarray, patch_flow: np.ndarray) -> np.ndarray:
    height, width = previous.shape[:2]
    dense = cv2.resize(np.asarray(patch_flow, np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    dense[..., 0] *= width / patch_flow.shape[1]
    dense[..., 1] *= height / patch_flow.shape[0]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    # Cached flow maps current coordinates back into the previous frame.
    return cv2.remap(previous, x + dense[..., 0], y + dense[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .68, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    args = parse_args()
    flow = np.load(args.flow, mmap_mode="r")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = min(int(round(args.seconds * fps)), len(flow) + 1)
    ok, previous = cap.read()
    if not ok:
        raise RuntimeError("Could not read first video frame")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    for index in range(1, frames):
        ok, current = cap.read()
        if not ok:
            raise RuntimeError("Video ended early")
        warped = warp_previous(previous, flow[index - 1])
        writer.write(cv2.hconcat((label(current.copy(), "Actual current frame"), label(warped, "Previous frame warped by cached flow"))))
        previous = current
        if index % 150 == 0:
            print(f"rendered {index}/{frames - 1}", flush=True)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
