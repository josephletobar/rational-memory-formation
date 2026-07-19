#!/usr/bin/env python3
"""Render motion-aligned local semantic surprise from cached DINO patch tokens.

Each stored DINO patch token has its own rolling diagonal Gaussian.  At every
frame, the previous patch-token grids are warped into the current frame using
the cached *backward* optical flow before the Gaussian is evaluated.  Thus a
patch is compared with its own recent physical region, rather than with a
fixed image coordinate or another patch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--patches", required=True, type=Path)
    p.add_argument("--flow", required=True, type=Path)
    p.add_argument("--global-cache", required=True, type=Path,
                   help="Existing global DINO score .npz; its score is preserved exactly.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--map-output", type=Path,
                   help="Optional .npy cache of the per-frame 16x16 surprise maps.")
    p.add_argument("--history", type=int, default=300)
    p.add_argument("--max-frames", type=int,
                   help="Render an initial cached prefix for a fast intermediate preview.")
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def global_scores(path: Path, frames: int) -> np.ndarray:
    with np.load(path) as archive:
        for key in ("scores", "surprise", "distances"):
            if key in archive:
                values = np.asarray(archive[key], dtype=np.float32)
                break
        else:
            raise RuntimeError(f"No global score array in {path}; found {archive.files}")
    if len(values) != frames:
        raise RuntimeError(f"Global score has {len(values)} frames; patch cache has {frames}")
    return values


def border_color(score: float) -> tuple[int, int, int]:
    if not np.isfinite(score):
        return (90, 90, 90)
    if score >= 2.0:
        return (0, 0, 255)
    if score >= 1.35:
        return (0, 220, 255)
    return (0, 210, 0)


def backward_grid(backward: torch.Tensor) -> torch.Tensor:
    """Return grid_sample coordinates for current->previous flow in patch cells."""
    grid_h, grid_w = backward.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=backward.device, dtype=torch.float32),
        torch.arange(grid_w, device=backward.device, dtype=torch.float32),
        indexing="ij",
    )
    source_x = xx + backward[..., 0]
    source_y = yy + backward[..., 1]
    return torch.stack((2.0 * source_x / max(grid_w - 1, 1) - 1.0,
                        2.0 * source_y / max(grid_h - 1, 1) - 1.0), dim=-1)


def overlay(frame: np.ndarray, local_map: np.ndarray, global_score: float, index: int, fps: float) -> np.ndarray:
    height, width = frame.shape[:2]
    # Local standardized distances: 1 is ordinary; clip the visualization at 3.
    intensity = np.clip((local_map - 1.0) / 2.0, 0.0, 1.0)
    heat = cv2.resize((intensity * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    alpha = cv2.resize((0.58 * intensity).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    result = (frame.astype(np.float32) * (1.0 - alpha[..., None]) + heat.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    color = border_color(global_score)
    cv2.rectangle(result, (0, 0), (width - 1, height - 1), color, max(5, width // 110))
    global_text = "warming up" if not np.isfinite(global_score) else f"Global surprise: {global_score:.2f}"
    cv2.putText(result, global_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(result, global_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(result, f"Patch novelty (motion-aligned) | {index / fps:0.1f}s", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(result, f"Patch novelty (motion-aligned) | {index / fps:0.1f}s", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    if args.history < 2:
        raise ValueError("--history must be at least 2")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    patches = np.load(args.patches, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    if patches.ndim != 4:
        raise RuntimeError(f"Patch cache must be T×H×W×D, got {patches.shape}")
    stored_frames, gh, gw, dim = patches.shape
    frames = min(stored_frames, args.max_frames) if args.max_frames else stored_frames
    if flow.ndim != 5 or flow.shape[0] < frames - 1 or flow.shape[1:] != (2, gh, gw, 2):
        raise RuntimeError(f"Incompatible caches: patches {patches.shape}, flow {flow.shape}")
    scores = global_scores(args.global_cache, stored_frames)[:frames]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = min(source_w, args.max_width) if args.max_width else source_w
    out_h = int(round(source_h * out_w / source_w))
    out_h -= out_h % 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {args.output}")
    maps = np.lib.format.open_memmap(args.map_output, mode="w+", dtype=np.float16, shape=(frames, gh, gw)) if args.map_output else None

    # A literal tensor stack of 300 x 768 x 16 x 16 samples at every frame is
    # needlessly expensive.  The diagonal Gaussian only needs sum and sum of
    # squares, so keep those rolling sufficient statistics on the GPU.  We keep
    # the small (16x16x2) coordinate maps for the individual historic frames so
    # that the outgoing frame can be subtracted in its current physical place.
    token_indices: list[int] = []
    coordinate_maps: list[torch.Tensor] = []  # current coordinates -> token's original coordinates
    token_sum = None
    token_sq_sum = None
    identity_y, identity_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, gh, device=device),
        torch.linspace(-1.0, 1.0, gw, device=device), indexing="ij")
    identity = torch.stack((identity_x, identity_y), dim=-1)

    def sample_grid(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample HxWxD values using normalized HxWx2 coordinates."""
        return F.grid_sample(values.permute(2, 0, 1).unsqueeze(0), coordinates.unsqueeze(0),
                             mode="bilinear", padding_mode="border", align_corners=True)[0].permute(1, 2, 0)

    for index in range(frames):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended at frame {index}/{frames}")
        current = torch.from_numpy(np.array(patches[index], dtype=np.float32, copy=True)).to(device)
        # Carry all sufficient statistics and coordinate maps one frame forward
        # using the cached current->previous patch-cell optical flow.
        if index:
            step = backward_grid(torch.from_numpy(np.array(flow[index - 1, 1], dtype=np.float32, copy=True)).to(device))
            token_sum = sample_grid(token_sum, step)
            token_sq_sum = sample_grid(token_sq_sum, step)
            if coordinate_maps:
                map_batch = torch.stack(coordinate_maps, dim=0).permute(0, 3, 1, 2)
                grid_batch = step.unsqueeze(0).expand(len(coordinate_maps), -1, -1, -1)
                coordinate_maps = [item.permute(1, 2, 0).contiguous() for item in F.grid_sample(
                    map_batch, grid_batch, mode="bilinear", padding_mode="border", align_corners=True)]
        if len(token_indices) >= args.history:
            count = float(len(token_indices))
            mean = token_sum / count
            variance = (token_sq_sum / count - mean.square()).clamp_min(1e-4)
            local = torch.sqrt(((current - mean).square() / variance).mean(dim=-1))
            local_np = local.detach().cpu().numpy().astype(np.float32)
        else:
            local_np = np.zeros((gh, gw), dtype=np.float32)
        # Retain exactly the prior --history frames for the next score.  The
        # old contribution is sampled from its original cached patch tokens at
        # the composed, current physical coordinates before it leaves the pool.
        if len(token_indices) == args.history:
            outgoing_index = token_indices.pop(0)
            outgoing_coordinates = coordinate_maps.pop(0)
            outgoing = torch.from_numpy(np.array(patches[outgoing_index], dtype=np.float32, copy=True)).to(device)
            outgoing = sample_grid(outgoing, outgoing_coordinates)
            outgoing_sq = torch.from_numpy(np.array(patches[outgoing_index], dtype=np.float32, copy=True)).to(device).square()
            outgoing_sq = sample_grid(outgoing_sq, outgoing_coordinates)
            token_sum -= outgoing
            token_sq_sum -= outgoing_sq
        token_indices.append(index)
        coordinate_maps.append(identity)
        if token_sum is None:
            token_sum = current.clone()
            token_sq_sum = current.square()
        else:
            token_sum += current
            token_sq_sum += current.square()
        if maps is not None:
            maps[index] = local_np.astype(np.float16)
        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(overlay(frame, local_np, float(scores[index]), index, fps))
        if index % 100 == 0 or index == frames - 1:
            print(f"rendered {index + 1}/{frames}", flush=True)
    cap.release()
    writer.release()
    if maps is not None:
        del maps
    metadata = {
        "video": args.video.name, "frames": frames, "fps": fps, "history": args.history,
        "patch_grid": [gh, gw], "patch_dimensions": dim,
        "patch_comparison": "motion-aligned own-patch rolling diagonal Gaussian",
        "flow_direction": "current_to_previous", "global_scores": str(args.global_cache),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
