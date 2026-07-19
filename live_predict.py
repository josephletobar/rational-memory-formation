#!/usr/bin/env python3
"""Run encoder + probes online and display predictions with OpenCV.

No features or predictions are written to disk. A prediction is refreshed each
time a new sliding window is ready and is displayed on every intervening frame.
"""

import argparse
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from joblib import load as joblib_load

import cv2
import numpy as np
import torch
from transformers import AutoModel

from train_vjepa_probes import (
    ENCODER_PRESETS,
    LinearProbe,
    TinyMLP,
    choose_device,
    encode_batch,
    prepare_frame,
)


# The current canonical baseline is the 19/9 optical-flow RF.  Keep a portable
# fallback for use on machines where the Crucial X9 is not mounted.
CANONICAL_OPTICAL_FLOW_PROBES = Path(
    "/Volumes/Crucial X9/theory-of-mind/trained_models/optical_flow_28cached_w32s32_rf"
)
DEFAULT_PROBES = (
    CANONICAL_OPTICAL_FLOW_PROBES
    if CANONICAL_OPTICAL_FLOW_PROBES.is_dir()
    else Path("results_rf")
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument(
        "--probes",
        type=Path,
        default=DEFAULT_PROBES,
        help="Probe directory. Defaults to the canonical optical-flow RF results.",
    )
    p.add_argument(
        "--encoder",
        choices=("vjepa2", "videomae", "optical_flow"),
        default="optical_flow",
        help="Feature family to use for window embeddings (default: canonical optical-flow baseline).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the HF model id (defaults from --encoder preset).",
    )
    p.add_argument("--window-frames", type=int, default=32)
    p.add_argument("--stride-frames", type=int, default=4)
    p.add_argument(
        "--flow-grid-h",
        type=int,
        default=16,
        help="Hough-like grid rows for optical-flow descriptor pooling.",
    )
    p.add_argument(
        "--flow-grid-w",
        type=int,
        default=12,
        help="Hough-like grid cols for optical-flow descriptor pooling.",
    )
    p.add_argument(
        "--flow-temporal-smooth",
        type=float,
        default=2.0,
        help="Temporal Gaussian smoothing window in seconds for optical-flow descriptors.",
    )
    p.add_argument(
        "--flow-pyramid-levels",
        type=int,
        default=3,
        help="Temporal pyramid levels for optical-flow descriptors (1-3 supported).",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--allow-download", action="store_true")
    return p.parse_args()


def draw(frame, text, row, color):
    y = 42 + row * 42
    cv2.putText(
        frame,
        text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if args.window_frames < 2 or args.stride_frames < 1:
        raise ValueError("window-frames must be >= 2 and stride-frames >= 1")
    if args.encoder == "optical_flow":
        if args.flow_grid_h < 1 or args.flow_grid_w < 1:
            raise ValueError("flow-grid-h and flow-grid-w must be >= 1")
        if args.flow_pyramid_levels < 1 or args.flow_pyramid_levels > 3:
            raise ValueError("flow-pyramid-levels must be 1, 2, or 3")

    device = choose_device(args.device)
    if args.encoder == "optical_flow":
        print("Using optical-flow features; no HF encoder loaded.")
        encoder = None
        model = None
    else:
        model_id = args.model if args.model is not None else ENCODER_PRESETS[args.encoder]
        print(f"Loading frozen encoder on {device}")
        model = AutoModel.from_pretrained(
            model_id, local_files_only=not args.allow_download
        )
        encoder = model.vjepa2 if hasattr(model, "vjepa2") else model
        encoder.requires_grad_(False).eval().to(device)
    del model

    normalization = np.load(args.probes / "normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]
    dim = len(mean)
    probes = {}
    linear_path = args.probes / "linear_probe.pt"
    mlp_path = args.probes / "mlp_probe.pt"
    rf_path = args.probes / "rf_probe.joblib"

    if linear_path.exists():
        probe = LinearProbe(dim).to(device)
        state = torch.load(linear_path, map_location="cpu", weights_only=True)
        probe.load_state_dict(state)
        probe.eval()
        probes["linear"] = ("torch", probe)
    if mlp_path.exists():
        probe = TinyMLP(dim).to(device)
        state = torch.load(mlp_path, map_location="cpu", weights_only=True)
        probe.load_state_dict(state)
        probe.eval()
        probes["mlp"] = ("torch", probe)
    if rf_path.exists():
        probes["rf"] = ("rf", joblib_load(rf_path))
    if not probes:
        raise FileNotFoundError(f"No probe files found in {args.probes}")
    del mlp_path, linear_path, rf_path

    for name, (probe_type, probe) in probes.items():
        if probe_type == "torch":
            probe.to(device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps)) if fps > 0 else 1

    window = deque(maxlen=args.window_frames)
    probabilities = None
    frame_index = -1
    paused = False
    flow_args = None
    if args.encoder == "optical_flow":
        flow_args = SimpleNamespace(
            flow_grid_h=args.flow_grid_h,
            flow_grid_w=args.flow_grid_w,
            flow_temporal_smooth=args.flow_temporal_smooth,
            flow_pyramid_levels=args.flow_pyramid_levels,
            source_fps=fps if fps > 0 else 15.0,
        )

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            window.append(prepare_frame(frame))

            window_start = frame_index - args.window_frames + 1
            should_predict = (
                len(window) == args.window_frames
                and window_start % args.stride_frames == 0
            )
            if should_predict:
                embedding = encode_batch(
                    [np.stack(window)],
                    encoder,
                    device,
                    args.encoder,
                    flow_args=flow_args,
                )
                embedding = (embedding - mean) / std
                x = torch.from_numpy(embedding).float().to(device)
                probabilities = {}
                with torch.inference_mode():
                    for name, (probe_type, probe) in probes.items():
                        if probe_type == "torch":
                            probabilities[name] = torch.sigmoid(probe(x)).item()
                        else:
                            probabilities[name] = float(
                                probe.predict_proba(embedding)[0, 1]
                            )

        if probabilities is None:
            draw(frame, f"WAITING FOR {args.window_frames}-FRAME WINDOW", 0, (180, 180, 180))
        else:
            for row, (name, probability) in enumerate(probabilities.items()):
                predicted = probability >= args.threshold
                color = (0, 210, 0) if predicted else (50, 120, 255)
                draw(
                    frame,
                    f"{name.upper()}: {str(predicted).upper()}  p={probability:.3f}",
                    row,
                    color,
                )

        time_sec = frame_index / fps if fps > 0 else 0.0
        draw(frame, f"t={time_sec:.2f}s  space: pause  q: quit", 3, (255, 255, 255))
        cv2.imshow("Live encoder predictions", frame)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
