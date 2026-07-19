#!/usr/bin/env python3
"""Cache DINOv2 CLS and 16x16 patch tokens for every frame of one video."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path, help="Output .npy patch-token cache")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    if frame_count < 1:
        raise RuntimeError(f"No frames in {args.video}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".partial.npy")

    frames: list[np.ndarray] = []
    cache = None
    index = 0
    if temporary.exists():
        cache = np.lib.format.open_memmap(temporary, mode="r+")
        if cache.dtype != np.float16 or cache.shape[0] != frame_count:
            raise RuntimeError(f"Partial cache has {cache.shape}/{cache.dtype}; it does not match this video")
        written = np.any(cache != 0, axis=(1, 2, 3))
        index = int(np.count_nonzero(written))
        if not np.all(written[:index]) or np.any(written[index:]):
            raise RuntimeError("Partial cache is not contiguous; remove it before retrying")
        print(f"resuming patch tokens at {index}/{frame_count}", flush=True)

    def flush() -> None:
        nonlocal cache, index
        if not frames:
            return
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        pixels = processor(images=rgb, return_tensors="pt").to(device)
        with torch.inference_mode():
            tokens = model(**pixels).last_hidden_state
            patch_tokens = torch.nn.functional.normalize(tokens[:, 1:].float(), dim=-1)
        patch_count, dimension = patch_tokens.shape[1:]
        grid = int(round(patch_count**0.5))
        if grid * grid != patch_count:
            raise RuntimeError(f"Expected square patch grid, got {patch_count} tokens")
        if cache is None:
            cache = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float16,
                shape=(frame_count, grid, grid, dimension),
            )
        elif cache.shape != (frame_count, grid, grid, dimension):
            raise RuntimeError(f"Partial cache has {cache.shape}; expected {(frame_count, grid, grid, dimension)}")
        count = len(frames)
        cache[index : index + count] = patch_tokens.reshape(count, grid, grid, dimension).cpu().numpy().astype(np.float16)
        index += count
        frames.clear()
        print(f"cached patch tokens: {index}/{frame_count}", flush=True)

    # Resume directly at the first missing frame.  Seeking avoids repeatedly
    # decoding thousands of already durable frames after an interrupted GPU job.
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        print(f"seeked decoder to frame {index}", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            # Some MP4s transiently stop OpenCV's decoder mid-stream after a
            # long GPU run.  Reopen and seek once before treating it as EOF.
            if index < frame_count:
                cap.release()
                cap = cv2.VideoCapture(str(args.video))
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
            if not ok:
                break
        frames.append(frame)
        if len(frames) >= args.batch_size:
            flush()
    flush()
    cap.release()
    if index != frame_count:
        raise RuntimeError(f"Read {index} frames but metadata reported {frame_count}")
    del cache
    os.replace(temporary, args.output)
    metadata = {
        "video": args.video.name,
        "model": args.model,
        "fps": fps,
        "frames": frame_count,
        "grid": grid,
        "dimension": dimension,
        "normalized": True,
        "dtype": "float16",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
