#!/usr/bin/env python3
"""Cache forward/backward dense optical flow in the DINO 16x16 patch grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from transformers import AutoImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path, help="Output .npy flow cache")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def grayscale_batch(processor: AutoImageProcessor, frames: list[np.ndarray]) -> list[np.ndarray]:
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    pixels = processor(images=rgb, return_tensors="pt")["pixel_values"].numpy()
    mean = np.asarray(processor.image_mean, dtype=np.float32)[None, :, None, None]
    std = np.asarray(processor.image_std, dtype=np.float32)[None, :, None, None]
    images = np.clip((pixels * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return [cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY) for image in images]


def patch_flow(previous: np.ndarray, current: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    forward = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = previous.shape
    if height % grid or width % grid:
        raise RuntimeError(f"DINO image {width}x{height} is not divisible by grid {grid}")
    cell_h, cell_w = height // grid, width // grid
    def pool(flow: np.ndarray) -> np.ndarray:
        pooled = flow.reshape(grid, cell_h, grid, cell_w, 2).mean(axis=(1, 3))
        pooled[..., 0] /= cell_w
        pooled[..., 1] /= cell_h
        return pooled.astype(np.float16)
    return pool(forward), pool(backward)


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    processor = AutoImageProcessor.from_pretrained(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".partial.npy")
    cache = None
    prior = None
    frames: list[np.ndarray] = []
    pair_index = 0
    resume_pairs = 0
    if temporary.exists():
        cache = np.lib.format.open_memmap(temporary, mode="r+")
        if cache.dtype != np.float16 or cache.shape[0] != frame_count - 1:
            raise RuntimeError(f"Partial flow cache has {cache.shape}/{cache.dtype}; it does not match this video")
        written = np.any(cache != 0, axis=(1, 2, 3, 4))
        resume_pairs = int(np.count_nonzero(written))
        if not np.all(written[:resume_pairs]) or np.any(written[resume_pairs:]):
            raise RuntimeError("Partial flow cache is not contiguous; remove it before retrying")
        print(f"resuming dense flow at {resume_pairs}/{frame_count - 1}", flush=True)

    def flush() -> None:
        nonlocal cache, prior, pair_index
        if not frames:
            return
        for current in grayscale_batch(processor, frames):
            if prior is not None:
                if cache is None:
                    grid = current.shape[0] // 14
                    cache = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=np.float16,
                        shape=(frame_count - 1, 2, grid, grid, 2),
                    )
                elif cache.shape != (frame_count - 1, 2, current.shape[0] // 14, current.shape[0] // 14, 2):
                    raise RuntimeError(f"Partial flow cache has incompatible shape {cache.shape}")
                # Decode all earlier frames to preserve the exact predecessor,
                # but skip recomputing already durable partial entries.
                if pair_index < resume_pairs:
                    pair_index += 1
                    prior = current
                    continue
                forward, backward = patch_flow(prior, current, cache.shape[2])
                cache[pair_index, 0] = forward
                cache[pair_index, 1] = backward
                pair_index += 1
            prior = current
        frames.clear()
        print(f"cached dense flow: {pair_index}/{frame_count - 1} frame pairs", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) >= args.batch_size:
            flush()
    flush()
    cap.release()
    if pair_index != frame_count - 1:
        raise RuntimeError(f"Read {pair_index} pairs but expected {frame_count - 1}")
    del cache
    os.replace(temporary, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "video": args.video.name, "model_preprocessing": args.model,
        "fps": fps, "pairs": pair_index, "grid": 16,
        "directions": ["previous_to_current", "current_to_previous"],
        "units": "DINO_patch_cells", "dtype": "float16",
    }, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
