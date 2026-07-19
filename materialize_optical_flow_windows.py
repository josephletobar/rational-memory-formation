#!/usr/bin/env python3
"""Materialize exact optical-flow window features from cached raw flow."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cache_optical_flow_raw import RAW_SUFFIX, raw_valid
from train_vjepa_probes import flow_window_feature, save_feature_cache


SLUG = "optflow.h16x12.lvl3.ts2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--raw-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, required=True)
    parser.add_argument("--stride-frames", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def output_path(output_dir: Path, video: Path, window: int, stride: int) -> Path:
    return output_dir / f"{video.stem}.{SLUG}.w{window}.s{stride}.full.npz"


def worker(task):
    index, total, video, raw_cache, output, window, stride = task
    if output.is_file():
        print(f"[{index}/{total}] features cached: {video.name}", flush=True)
        return
    if not raw_valid(raw_cache):
        raise RuntimeError(f"Missing/invalid raw flow cache for {video}")
    started = time.perf_counter()
    with np.load(raw_cache) as data:
        sequence = data["flow_sequence"]
        fps = float(data["fps"])
    total_frames = len(sequence) + 1
    if total_frames < window:
        raise RuntimeError(f"No full W{window} windows in {video}")
    args = SimpleNamespace(flow_pyramid_levels=3, flow_grid_h=16, flow_grid_w=12)
    starts, ends, features = [], [], []
    max_start = total_frames - window
    for start in range(0, max_start + 1, stride):
        features.append(flow_window_feature(sequence, start, window, args))
        starts.append(start / fps)
        ends.append((start + window) / fps)
    result = (np.stack(features), np.asarray(starts), np.asarray(ends))
    save_feature_cache(output, result)
    output.with_name(output.name + ".stats.json").write_text(json.dumps({
        "encoder": "optical_flow", "window_frames": window, "stride_frames": stride,
        "feature_dimensionality": int(result[0].shape[1]), "num_windows": int(result[0].shape[0]),
        "feature_extraction_seconds": time.perf_counter() - started,
        "raw_flow_reused": True,
    }, indent=2) + "\n")
    print(f"[{index}/{total}] materialized: {video.name}", flush=True)


def main():
    args = parse_args()
    if args.window_frames < 2 or args.stride_frames < 1:
        raise ValueError("window must be >=2 and stride >=1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = [args.input_root / line.strip() for line in args.list_file.read_text().splitlines() if line.strip()]
    tasks = [
        (i, len(videos), video, args.raw_cache_dir / f"{video.stem}{RAW_SUFFIX}",
         output_path(args.output_dir, video, args.window_frames, args.stride_frames),
         args.window_frames, args.stride_frames)
        for i, video in enumerate(videos, 1)
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks), os.cpu_count() or 1)) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
