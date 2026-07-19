#!/usr/bin/env python3
"""Cache the smoothed per-frame optical-flow descriptors for reuse.

This is the exact pre-window portion of ``extract_video(..., optical_flow)``:
Farnebäck flow on the same prepared 256px frames, then the same temporal
Gaussian smoothing.  Window size and stride are intentionally absent, so one
raw cache can materialize many controlled temporal configurations exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from pathlib import Path

import cv2
import numpy as np

from train_vjepa_probes import flow_cell_descriptor, prepare_frame, smooth_temporal_features


RAW_SUFFIX = ".optflow_raw.h16x12.ts2.npz"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def raw_path(cache_dir: Path, video: Path) -> Path:
    return cache_dir / f"{video.stem}{RAW_SUFFIX}"


def raw_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            return "flow_sequence" in data and "fps" in data and data["flow_sequence"].ndim == 2
    except (OSError, ValueError):
        return False


def atomic_save(path: Path, flow_sequence: np.ndarray, fps: float) -> None:
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    partial.unlink(missing_ok=True)
    np.savez_compressed(partial, flow_sequence=flow_sequence, fps=np.float64(fps))
    partial.replace(path)


def extract_one(task):
    index, total, video, cache = task
    cv2.setNumThreads(1)
    if raw_valid(cache):
        print(f"[{index}/{total}] raw cached: {video.name}", flush=True)
        return
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError(f"Invalid frame rate for {video}")
    previous = None
    sequence = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(prepare_frame(frame), cv2.COLOR_RGB2GRAY)
        if previous is not None:
            sequence.append(flow_cell_descriptor(previous, gray, grid_h=16, grid_w=12))
        previous = gray
    cap.release()
    if not sequence:
        raise RuntimeError(f"No flow pairs found in {video}")
    sequence = smooth_temporal_features(
        np.asarray(sequence, dtype=np.float32), fps=fps, smooth_seconds=2.0
    )
    atomic_save(cache, sequence, fps)
    print(f"[{index}/{total}] raw cached: {video.name} ({time.perf_counter() - started:.1f}s)", flush=True)


def main():
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    videos = [
        args.input_root / line.strip()
        for line in args.list_file.read_text().splitlines()
        if line.strip()
    ]
    missing = [video for video in videos if not video.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source video: {missing[:3]}")
    tasks = [(i, len(videos), video, raw_path(args.cache_dir, video)) for i, video in enumerate(videos, 1)]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks), os.cpu_count() or 1)) as pool:
        futures = [pool.submit(extract_one, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
