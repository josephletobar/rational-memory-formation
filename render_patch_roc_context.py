#!/usr/bin/env python3
"""Default surprise beside a simple motion-aligned patch rate-of-change test."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from surprise_normalization import (
    DEFAULT_CONTEXT_BETA,
    DEFAULT_DISPLAY_EMA,
    DEFAULT_FLOW_ALPHA,
    global_context_adjustment,
    raw_global_display_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("patches", type=Path)
    parser.add_argument("maps", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("roc_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment-start", type=int, required=True)
    parser.add_argument("--output-start", type=int, required=True)
    parser.add_argument("--output-end", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_FLOW_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_CONTEXT_BETA)
    parser.add_argument("--display-ema", type=float, default=DEFAULT_DISPLAY_EMA)
    parser.add_argument("--ablate-global-context", action="store_true", help="Build the RoC panel from the flow-normalized base, without patch/global cosine boosting.")
    return parser.parse_args()


def warp_previous(previous: np.ndarray, current_to_previous_flow: np.ndarray) -> np.ndarray:
    """Bilinearly sample the previous patch grid at current-frame locations."""
    height, width = previous.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = np.clip(x + current_to_previous_flow[..., 0], 0, width - 1)
    y = np.clip(y + current_to_previous_flow[..., 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    return ((1 - wx) * (1 - wy) * previous[y0, x0] + wx * (1 - wy) * previous[y0, x1]
            + (1 - wx) * wy * previous[y1, x0] + wx * wy * previous[y1, x1])


def patch_rate_of_change(current: np.ndarray, previous: np.ndarray, flow: np.ndarray) -> np.ndarray:
    aligned_previous = warp_previous(previous, flow)
    current = np.asarray(current, dtype=np.float32)
    aligned_previous = np.asarray(aligned_previous, dtype=np.float32)
    current /= np.linalg.norm(current, axis=-1, keepdims=True) + 1e-8
    aligned_previous /= np.linalg.norm(aligned_previous, axis=-1, keepdims=True) + 1e-8
    return np.clip(1.0 - np.sum(current * aligned_previous, axis=-1), 0.0, 2.0)


def overlay(frame: np.ndarray, score: np.ndarray, lo: float, hi: float, label: str) -> np.ndarray:
    value = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    value = cv2.GaussianBlur(value, (3, 3), 0)
    heat = cv2.applyColorMap(
        cv2.resize((value * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC),
        cv2.COLORMAP_TURBO,
    )
    output = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    if not 0 < args.display_ema <= 1:
        raise ValueError("--display-ema must be in (0, 1]")
    patches = np.load(args.patches, mmap_mode="r")
    maps = np.load(args.maps, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    offset, count = args.output_start - args.segment_start, args.output_end - args.output_start
    expected_shape = (count, *patches.shape[1:3])
    if args.roc_cache.exists():
        roc = np.load(args.roc_cache, mmap_mode="r")
        if roc.shape != expected_shape:
            raise ValueError(f"Existing RoC cache has shape {roc.shape}, expected {expected_shape}")
        print("using cached patch RoC", flush=True)
    else:
        args.roc_cache.parent.mkdir(parents=True, exist_ok=True)
        roc = np.lib.format.open_memmap(args.roc_cache, mode="w+", dtype=np.float16, shape=expected_shape)
        for index in range(count):
            j = offset + index
            roc[index] = patch_rate_of_change(patches[j], patches[j - 1], flow[j - 1]).astype(np.float16)
        del roc
        roc = np.load(args.roc_cache, mmap_mode="r")
    # A clip-wide reference preserves the original score's scale.  A patch at
    # average RoC is unchanged; higher/lower RoC changes heat linearly.
    roc_reference = float(np.mean(roc, dtype=np.float64))
    raw_all = np.asarray(maps[offset:offset + count], dtype=np.float32)
    lo, hi = raw_global_display_scale(raw_all)
    print(f"raw display p5/p95={lo:.4f}/{hi:.4f}; clip-wide patch RoC mean={roc_reference:.5f}", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.output_start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    default_ema = roc_ema = None
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("video ended early")
        j = offset + index
        base = np.asarray(maps[j], dtype=np.float32) / (1.0 + args.alpha * np.linalg.norm(np.asarray(flow[j - 1], dtype=np.float32), axis=-1))
        default = global_context_adjustment(base, np.asarray(patches[j], dtype=np.float32), args.beta)
        roc_source = base if args.ablate_global_context else default
        roc_adjusted = roc_source * (np.asarray(roc[index], dtype=np.float32) / max(roc_reference, 1e-8))
        default_ema = default if default_ema is None else args.display_ema * default + (1 - args.display_ema) * default_ema
        roc_ema = roc_adjusted if roc_ema is None else args.display_ema * roc_adjusted + (1 - args.display_ema) * roc_ema
        writer.write(cv2.hconcat((
            overlay(frame, default_ema, lo, hi, "Default: context boost + display EMA"),
            overlay(frame, roc_ema, lo, hi, "+ RoC; no global context" if args.ablate_global_context else "+ motion-aligned patch RoC (linear)"),
        )))
        if index % 150 == 0:
            print(f"rendered {index + 1}/{count}", flush=True)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
