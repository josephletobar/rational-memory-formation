#!/usr/bin/env python3
"""Render clip-level V-JEPA surprise against the preceding 300 source frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path); p.add_argument("cache", type=Path); p.add_argument("raw_video", type=Path)
    p.add_argument("--history-frames", type=int, default=300)
    return p.parse_args()


def score_windows(features: np.ndarray, starts: np.ndarray, ends: np.ndarray, fps: float, history_frames: int) -> np.ndarray:
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    history_seconds = history_frames / fps
    scores = np.full(len(unit), np.nan, dtype=np.float32)
    for index in range(len(unit)):
        previous = np.flatnonzero((ends[:index] <= starts[index]) & (starts[:index] >= starts[index] - history_seconds))
        if len(previous) >= 3:
            reference = unit[previous].mean(axis=0); reference /= max(np.linalg.norm(reference), 1e-8)
            scores[index] = 1.0 - float(unit[index] @ reference)
    return scores


def color(value: float) -> tuple[int, int, int]:
    # green -> yellow -> red in OpenCV BGR
    value = float(np.clip(value, 0, 1))
    if value < .5:
        return (0, 255, int(510 * value))
    return (0, int(510 * (1 - value)), 255)


def main() -> None:
    a = parse_args(); data = np.load(a.cache)
    features, starts, ends = np.asarray(data["features"], np.float32), np.asarray(data["starts"], np.float32), np.asarray(data["ends"], np.float32)
    cap = cv2.VideoCapture(str(a.video)); fps = cap.get(cv2.CAP_PROP_FPS) or 15.0; count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scores = score_windows(features, starts, ends, fps, a.history_frames)
    valid = scores[np.isfinite(scores)]
    lo, hi = np.percentile(valid, (5, 95))
    print(f"windows={len(scores)}; history={a.history_frames} frames; display p5/p95={lo:.5f}/{hi:.5f}", flush=True)
    a.raw_video.parent.mkdir(parents=True, exist_ok=True); writer = cv2.VideoWriter(str(a.raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    current = np.nan; next_window = 0
    for frame_index in range(count):
        ok, frame = cap.read()
        if not ok: raise RuntimeError("source ended early")
        moment = frame_index / fps
        while next_window < len(scores) and ends[next_window] <= moment:
            if np.isfinite(scores[next_window]): current = scores[next_window]
            next_window += 1
        if np.isfinite(current):
            normalized = float(np.clip((current - lo) / max(hi-lo, 1e-8), 0, 1)); border = color(normalized)
            thickness = max(12, min(width, height) // 30); frame[:thickness] = border; frame[-thickness:] = border; frame[:, :thickness] = border; frame[:, -thickness:] = border
            text = f"V-JEPA clip surprise {current:.4f} | prior 300 frames"
        else:
            text = "V-JEPA: warming up prior 300-frame history"
        cv2.putText(frame, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,0,0), 1, cv2.LINE_AA)
        writer.write(frame)
        if (frame_index + 1) % 150 == 0 or frame_index + 1 == count: print(f"rendered {frame_index+1}/{count}", flush=True)
    cap.release(); writer.release(); np.save(a.raw_video.with_suffix(".scores.npy"), scores)


if __name__ == "__main__": main()
