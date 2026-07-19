#!/usr/bin/env python3
"""Render saved probe predictions into a video without opening a GUI window."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def put(frame, text: str, row: int, color: tuple[int, int, int]) -> None:
    y = 42 + row * 42
    cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                color, 2, cv2.LINE_AA)


def render_predictions(
    predictions_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    border_model: str | None = None,
    max_width: int | None = None,
) -> int:
    saved = np.load(predictions_path)
    video = Path(str(saved["video"]))
    if not video.exists():
        candidate = predictions_path.parent / video.name
        video = candidate if candidate.exists() else video
    if not video.exists():
        raise FileNotFoundError(video)

    starts, ends = saved["starts"], saved["ends"]
    centers = (starts + ends) / 2
    model_names = [name for name in ("linear", "mlp", "rf") if name in saved.files]
    if not model_names:
        raise ValueError("Prediction file contains no supported probe outputs")
    if border_model is None:
        border_model = "rf" if "rf" in model_names else model_names[0]
    if border_model not in model_names:
        raise ValueError(f"Border model {border_model!r} is not available")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata for {video}")
    frame_limit = max_frames
    if max_seconds is not None:
        seconds_limit = round(max_seconds * fps)
        frame_limit = seconds_limit if frame_limit is None else min(frame_limit, seconds_limit)

    if max_width is not None and max_width < 1:
        raise ValueError("max_width must be positive")
    scale = 1.0 if max_width is None else min(1.0, max_width / width)
    output_width = max(2, round(width * scale))
    output_height = max(2, round(height * scale))
    output_width += output_width % 2
    output_height += output_height % 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create {temporary}")

    frame_index = 0
    try:
        while frame_limit is None or frame_index < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            time_sec = frame_index / fps
            for row, name in enumerate(model_names):
                probability = float(np.interp(time_sec, centers, saved[name]))
                predicted = probability >= threshold
                color = (0, 210, 0) if predicted else (50, 120, 255)
                put(
                    frame,
                    f"{name.upper()}: {str(predicted).upper()}  p={probability:.3f}",
                    row,
                    color,
                )
            border_probability = float(np.interp(time_sec, centers, saved[border_model]))
            border_color = (0, 210, 0) if border_probability >= threshold else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (output_width - 1, output_height - 1), border_color, max(6, round(16 * scale)))
            put(frame, f"t={time_sec:.2f}s", len(model_names) + 1, (255, 255, 255))
            writer.write(frame)
            frame_index += 1
            if frame_index % max(1, round(fps * 60)) == 0:
                print(f"  rendered {frame_index / fps:.0f}s", flush=True)
    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError("No frames were rendered")
    temporary.replace(output_path)
    print(f"Rendered {frame_index} frames to {output_path}", flush=True)
    return frame_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--border-model", choices=("linear", "mlp", "rf"), default=None)
    parser.add_argument("--max-width", type=int, default=None)
    args = parser.parse_args()
    render_predictions(
        args.predictions, args.output, args.threshold, max_seconds=args.max_seconds,
        border_model=args.border_model, max_width=args.max_width,
    )


if __name__ == "__main__":
    main()
