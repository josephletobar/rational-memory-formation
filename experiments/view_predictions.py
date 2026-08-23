#!/usr/bin/env python3
"""Play a held-out video with frame-by-frame probe predictions.

  conda run -n samworld python view_predictions.py results/predictions_VIDEO.npz

Press space to pause/resume and q or escape to quit.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fast", action="store_true", help="Decode as fast as possible")
    return p.parse_args()


def put(frame, text, row, color):
    y = 42 + row * 42
    cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    saved = np.load(args.predictions)
    video = Path(str(saved["video"]))
    if not video.exists():
        # Also handle moving the results directory with the source video nearby.
        video = Path.cwd() / video.name
    if not video.exists():
        raise FileNotFoundError(video)

    starts, ends = saved["starts"], saved["ends"]
    centers = (starts + ends) / 2
    intervals = saved["intervals"]
    model_names = [name for name in ("linear", "mlp") if name in saved.files]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 if args.fast else max(1, round(1000 / fps))
    paused, frame_index = False, -1

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
        time_sec = frame_index / fps
        first_prediction_row = 0
        if len(intervals):
            ground_truth = any(left <= time_sec < right for left, right in intervals)
            put(frame, f"GROUND TRUTH: {str(ground_truth).upper()}", 0,
                (0, 210, 0) if ground_truth else (180, 180, 180))
            first_prediction_row = 1
        for row, name in enumerate(model_names, start=first_prediction_row):
            # np.interp uses the nearest endpoint for the first/last half-window.
            probability = float(np.interp(time_sec, centers, saved[name]))
            predicted = probability >= args.threshold
            color = (0, 210, 0) if predicted else (50, 120, 255)
            put(frame, f"{name.upper()}: {str(predicted).upper()}  p={probability:.3f}",
                row, color)
        put(frame, f"t={time_sec:.2f}s   space: pause   q: quit", len(model_names) + 2,
            (255, 255, 255))

        cv2.imshow("V-JEPA held-out predictions", frame)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
