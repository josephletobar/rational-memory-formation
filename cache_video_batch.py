#!/usr/bin/env python3
"""Cache one encoder's window features for an explicit batch of videos."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

from train_vjepa_probes import (
    ENCODER_PRESETS,
    AutoModel,
    choose_device,
    extract_video,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder", choices=("vjepa2", "videomae", "optical_flow"), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-frames", type=int, default=None)
    parser.add_argument("--stride-frames", type=int, default=None)
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=None,
        help="Windows per encoder forward pass (uses an encoder-safe default).",
    )
    return parser.parse_args()


def encoder_config(name, window_frames=None, stride_frames=None):
    if name == "vjepa2":
        default_window, default_stride, slug = 32, 16, "facebook-vjepa2-vitl-fpc32-256-diving48"
    elif name == "videomae":
        default_window, default_stride, slug = 16, 16, "MCG-NJU-videomae-base"
    else:
        default_window, default_stride, slug = 32, 32, "optflow.h16x12.lvl3.ts2"
    return window_frames or default_window, stride_frames or default_stride, slug


def write_stats(cache: Path, encoder: str, window_frames: int, stride_frames: int,
                features, elapsed_seconds: float) -> None:
    cache.with_name(cache.name + ".stats.json").write_text(json.dumps({
        "encoder": encoder,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "feature_dimensionality": int(features.shape[1]),
        "num_windows": int(features.shape[0]),
        "feature_extraction_seconds": elapsed_seconds,
    }, indent=2) + "\n")


def load_encoder(name, device):
    if name == "optical_flow":
        return None
    model = AutoModel.from_pretrained(ENCODER_PRESETS[name])
    encoder = model.vjepa2 if name == "vjepa2" and hasattr(model, "vjepa2") else model
    encoder.requires_grad_(False).eval().to(device)
    return encoder


def extract_flow_video(task):
    """Extract one independent optical-flow cache in a child process."""
    index, total, video, cache, extract_args = task
    # The outer process pool supplies the parallelism.  One OpenCV thread per
    # child avoids an unnecessary nested thread pool; it does not alter flow
    # values.
    cv2.setNumThreads(1)
    if cache.exists():
        print(f"[{index}/{total}] cached: {video.name}", flush=True)
        return
    print(f"[{index}/{total}] extracting: {video.name}", flush=True)
    started = time.perf_counter()
    features, _, _ = extract_video(
        video,
        cache,
        encoder=None,
        device=None,
        args=extract_args,
        encoder_name="optical_flow",
    )
    write_stats(
        cache, "optical_flow", extract_args.window_frames,
        extract_args.stride_frames, features, time.perf_counter() - started,
    )


def main():
    args = parse_args()
    # On this A40, V-JEPA's long-token attention is throughput-optimal at two
    # clips; larger batches fit but take longer per window.  VideoMAE benefits
    # from a larger batch, while optical flow does not use this setting.
    if args.feature_batch_size is None:
        default_batch_sizes = {"videomae": 8, "vjepa2": 2, "optical_flow": 2}
        args.feature_batch_size = default_batch_sizes[args.encoder]
    if args.encoder == "optical_flow":
        # Each source video is independent.  A small pool uses otherwise-idle
        # pod CPU cores while VideoMAE and V-JEPA occupy the GPU.  The feature
        # calculation itself is unchanged; OpenCV threads stay low to avoid
        # oversubscribing the pod inside that outer pool.
        cv2.setNumThreads(2)
    window_frames, stride_frames, slug = encoder_config(
        args.encoder, args.window_frames, args.stride_frames
    )
    if window_frames < 2 or stride_frames < 1:
        raise ValueError("--window-frames must be >=2 and --stride-frames >=1")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    extract_args = SimpleNamespace(
        window_frames=window_frames,
        stride_frames=stride_frames,
        max_seconds=None,
        feature_batch_size=args.feature_batch_size,
        flow_grid_h=16,
        flow_grid_w=12,
        flow_pyramid_levels=3,
        flow_temporal_smooth=2.0,
    )
    videos = [
        args.input_root / line.strip()
        for line in args.list_file.read_text().splitlines()
        if line.strip()
    ]
    missing = [video for video in videos if not video.exists()]
    if missing:
        raise FileNotFoundError(f"Missing staged videos: {missing[:3]}")

    print(f"loading {args.encoder} for {len(videos)} videos", flush=True)
    encoder = load_encoder(args.encoder, device)
    limit = "full"
    def extract_one(index, video):
        cache = args.cache_dir / (
            f"{video.stem}.{slug}.w{window_frames}.s{stride_frames}.{limit}.npz"
        )
        if cache.exists():
            print(f"[{index}/{len(videos)}] cached: {video.name}", flush=True)
            return
        print(f"[{index}/{len(videos)}] extracting: {video.name}", flush=True)
        started = time.perf_counter()
        features, _, _ = extract_video(video, cache, encoder, device, extract_args, args.encoder)
        write_stats(
            cache, args.encoder, window_frames, stride_frames,
            features, time.perf_counter() - started,
        )

    if args.encoder == "optical_flow" and len(videos) > 1:
        workers = min(4, len(videos))
        tasks = [
            (index, len(videos), video, args.cache_dir / (
                f"{video.stem}.{slug}.w{window_frames}.s{stride_frames}.{limit}.npz"
            ), extract_args)
            for index, video in enumerate(videos, start=1)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(extract_flow_video, task)
                for task in tasks
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for index, video in enumerate(videos, start=1):
            extract_one(index, video)


if __name__ == "__main__":
    main()
