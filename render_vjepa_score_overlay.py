#!/usr/bin/env python3
"""Render a staged video with causal V-JEPA engagement and N=2 surprise scores."""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("probe", type=Path)
    p.add_argument("normalization", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-width", type=int, default=480)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--duration-sec", type=float, default=None)
    args = p.parse_args()
    saved = np.load(args.cache)
    features, ends = saved["features"].astype(np.float32), saved["ends"].astype(np.float64)
    norm = np.load(args.normalization)
    x = (features - norm["mean"]) / norm["std"]
    engagement = joblib.load(args.probe).predict_proba(x)[:, 1].astype(np.float32)
    surprise = np.zeros(len(x), dtype=np.float32)
    for i in range(2, len(x)):
        prior = x[i - 2:i]
        surprise[i] = 0.5 * np.mean((x[i] - prior.mean(0)) ** 2 / (prior.var(0) + 1e-5))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width0, height0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, args.max_width / width0)
    width, height = int(width0 * scale) // 2 * 2, int(height0 * scale) // 2 * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".part.mp4")
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)
    frame_i = int(round(args.start_sec * fps))
    final_frame = (
        frame_i + int(round(args.duration_sec * fps))
        if args.duration_sec is not None else None
    )
    while True:
        if final_frame is not None and frame_i >= final_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (width, height) != (width0, height0):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        t = frame_i / fps
        idx = int(np.searchsorted(ends, t, side="right") - 1)
        e = float(engagement[idx]) if idx >= 0 else 0.0
        s = float(surprise[idx]) if idx >= 0 else 0.0
        cv2.rectangle(frame, (0, 0), (width, 68), (0, 0, 0), -1)
        text1 = f"V-JEPA engagement: {e:.3f}"
        text2 = f"V-JEPA surprise N=2: {s:.3f}"
        for y, text in ((27, text1), (55, text2)):
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_i += 1
        if frame_i % 900 == 0:
            print(f"{frame_i / fps:.1f}s source time rendered", flush=True)
    cap.release(); writer.release()
    partial.replace(args.output)


if __name__ == "__main__":
    main()
