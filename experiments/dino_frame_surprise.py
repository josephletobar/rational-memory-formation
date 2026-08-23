#!/usr/bin/env python3
"""Render frame-level DINO standardized-distance surprise for one video.

For every frame t, DINO embeds that frame independently.  Once 64 prior
embeddings are available, the script fits a diagonal Gaussian to embeddings
t-64..t-1 and reports the RMS diagonal-Mahalanobis distance of embedding t.
The RMS form makes scores comparable across DINO embedding dimensions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--history", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--max-width",
        type=int,
        help="Downscale rendered output to this maximum width (aspect ratio preserved).",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Existing .npz from this script; reuse its per-frame DINO embeddings.",
    )
    return parser.parse_args()


def video_metadata(video: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, count, width, height


def embed_frames(
    video: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def flush() -> None:
        if not frames:
            return
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs)
            vectors = output.last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
        embeddings.append(vectors.cpu().numpy())
        frames.clear()

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        index += 1
        if len(frames) >= batch_size:
            flush()
            print(f"embedded {index}/{frame_count} frames", flush=True)
    flush()
    cap.release()
    return np.concatenate(embeddings, axis=0)


def standardized_distances(embeddings: np.ndarray, history: int, epsilon: float) -> np.ndarray:
    scores = np.full(len(embeddings), np.nan, dtype=np.float32)
    for t in range(history, len(embeddings)):
        prior = embeddings[t - history : t]
        mean = prior.mean(axis=0)
        variance = prior.var(axis=0) + epsilon
        z = (embeddings[t] - mean) / np.sqrt(variance)
        scores[t] = np.sqrt(np.mean(np.square(z)))
    return scores


def score_color(score: float) -> tuple[int, int, int]:
    if not np.isfinite(score):
        return (100, 100, 100)
    if score >= 2.0:
        return (0, 0, 255)
    if score >= 1.35:
        return (0, 200, 255)
    return (0, 220, 0)


def draw_overlay(frame: np.ndarray, scores: np.ndarray, index: int, history: int) -> np.ndarray:
    height, width = frame.shape[:2]
    score = float(scores[index])
    color = score_color(score)
    panel_height = 74
    cv2.rectangle(frame, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 8)

    if np.isfinite(score):
        label = f"DINO standardized distance: {score:.2f}  (previous {history} frames)"
        bar_ratio = min(score / 3.0, 1.0)
        cv2.rectangle(frame, (16, 42), (width - 16, 62), (50, 50, 50), -1)
        cv2.rectangle(frame, (16, 42), (16 + int((width - 32) * bar_ratio), 62), color, -1)
    else:
        label = f"DINO standardized distance: warming up ({index}/{history} prior frames)"
    cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # A compact rolling plot makes the temporal trace visible while scrubbing.
    recent = scores[max(history, index - 63) : index + 1]
    recent = recent[np.isfinite(recent)]
    if len(recent) > 1:
        plot_width, plot_height = min(260, width // 3), 42
        left, bottom = width - plot_width - 14, panel_height - 8
        cv2.rectangle(frame, (left, bottom - plot_height), (left + plot_width, bottom), (20, 20, 20), -1)
        ceiling = max(2.5, float(np.nanpercentile(recent, 95)) * 1.1)
        xs = np.linspace(left, left + plot_width - 1, len(recent)).astype(np.int32)
        ys = (bottom - np.clip(recent / ceiling, 0, 1) * (plot_height - 4)).astype(np.int32)
        cv2.polylines(frame, [np.column_stack((xs, ys))], False, color, 1, cv2.LINE_AA)
    return frame


def render(
    video: Path,
    output: Path,
    scores: np.ndarray,
    fps: float,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    history: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {output}")
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (source_width, source_height) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(draw_overlay(frame, scores, index, history))
        index += 1
    cap.release()
    writer.release()


def save_plot(output: Path, scores: np.ndarray, fps: float, history: int) -> None:
    """Write a full-video distance plot without requiring a plotting package."""
    width, height = 1800, 480
    left, right, top, bottom = 110, 40, 52, 72
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    plot_width, plot_height = width - left - right, height - top - bottom
    finite = scores[np.isfinite(scores)]
    ceiling = max(2.5, float(np.percentile(finite, 99)) * 1.05) if len(finite) else 2.5
    for threshold, color in ((1.35, (0, 200, 255)), (2.0, (0, 0, 255))):
        y = int(top + plot_height * (1.0 - min(threshold / ceiling, 1.0)))
        cv2.line(image, (left, y), (width - right, y), color, 1, cv2.LINE_AA)
        cv2.putText(image, f"{threshold:.2f}", (16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    valid_indices = np.flatnonzero(np.isfinite(scores))
    if len(valid_indices) > 1:
        xs = left + (valid_indices / (len(scores) - 1) * plot_width).astype(np.int32)
        values = np.clip(scores[valid_indices] / ceiling, 0, 1)
        ys = (top + (1.0 - values) * plot_height).astype(np.int32)
        cv2.polylines(image, [np.column_stack((xs, ys))], False, (0, 190, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (left, top), (width - right, height - bottom), (180, 180, 180), 1)
    duration = (len(scores) - 1) / fps if len(scores) else 0.0
    cv2.putText(image, f"DINO frame surprise - RMS standardized distance (previous {history} embeddings)", (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "0s", (left, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, f"{duration:.1f}s", (width - right - 82, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(output.with_suffix(".png")), image)


def main() -> None:
    args = parse_args()
    if args.history < 2:
        raise ValueError("--history must be at least 2")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    fps, _, source_width, source_height = video_metadata(args.video)
    width, height = source_width, source_height
    if args.max_width and source_width > args.max_width:
        width = args.max_width - (args.max_width % 2)
        height = int(round(source_height * width / source_width))
        height -= height % 2
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=False) as saved:
            embeddings = saved["embeddings"].astype(np.float32)
        print(f"reused {len(embeddings)} cached DINO frame embeddings", flush=True)
    else:
        processor = AutoImageProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).eval().to(device)
        embeddings = embed_frames(args.video, model, processor, device, args.batch_size)
    scores = standardized_distances(embeddings, args.history, args.epsilon)
    render(
        args.video, args.output, scores, fps, source_width, source_height,
        width, height, args.history,
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        embeddings=embeddings.astype(np.float16), scores=scores, fps=fps,
        history=args.history, model=args.model, distance="rms_diagonal_mahalanobis",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
