#!/usr/bin/env python3
"""Apply saved full-video probes to every window of one complete video."""

import argparse
import re
from pathlib import Path
from types import SimpleNamespace
from joblib import load as joblib_load

import numpy as np
import torch
from transformers import AutoModel

from train_vjepa_probes import (
    ENCODER_PRESETS,
    LinearProbe,
    TinyMLP,
    choose_device,
    extract_video,
    predict,
)
from render_predictions import render_predictions


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument(
        "--probes",
        type=Path,
        default=DEFAULT_PROBES,
        help="Probe directory. Defaults to the canonical optical-flow RF results.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--render-video",
        type=Path,
        default=None,
        help="Write an annotated MP4 after prediction extraction; no GUI is opened",
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
    p.add_argument("--stride-frames", type=int, default=32)
    p.add_argument("--flow-grid-h", type=int, default=16)
    p.add_argument("--flow-grid-w", type=int, default=12)
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
    p.add_argument("--feature-batch-size", type=int, default=2)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Process only this source prefix; useful for saving a stopped partial run",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--feature-cache-dir", type=Path, default=None)
    p.add_argument("--cached-only", action="store_true", help="Require an existing feature cache and skip encoder inference.")
    p.add_argument("--border-model", choices=("linear", "mlp", "rf"), default=None)
    p.add_argument("--render-max-width", type=int, default=None,
                   help="Optional width cap for saved render videos (preserves aspect ratio).")
    args = p.parse_args()

    if args.encoder == "optical_flow":
        if args.flow_grid_h < 1 or args.flow_grid_w < 1:
            raise ValueError("flow-grid-h and flow-grid-w must be >= 1")
        if args.flow_pyramid_levels < 1 or args.flow_pyramid_levels > 3:
            raise ValueError("flow-pyramid-levels must be 1, 2, or 3")
    if not args.video.exists():
        raise FileNotFoundError(args.video)

    output = args.output or args.probes / f"predictions_{args.video.stem}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or args.probes / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    if args.encoder == "optical_flow":
        model = None
        encoder = None
        slug = f"optflow.h{args.flow_grid_h}x{args.flow_grid_w}.lvl{args.flow_pyramid_levels}.ts{args.flow_temporal_smooth:g}"
    else:
        model_id = args.model if args.model is not None else ENCODER_PRESETS[args.encoder]
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-")

    limit = "full" if args.max_seconds is None else f"t{args.max_seconds:g}"
    cache = cache_dir / (
        f"{args.video.stem}.{slug}.w{args.window_frames}.s{args.stride_frames}.{limit}.npz"
    )
    if args.cached_only and not cache.exists():
        raise FileNotFoundError(f"Missing required cached features: {cache}")
    if args.encoder != "optical_flow":
        if args.cached_only:
            print("Cached-only mode: encoder inference disabled")
            model = encoder = None
        else:
            print(f"Loading frozen encoder on {device}")
            model = AutoModel.from_pretrained(model_id, local_files_only=not args.allow_download)
            encoder = model.vjepa2 if hasattr(model, "vjepa2") else model
            encoder.requires_grad_(False).eval().to(device)

    extraction_args = SimpleNamespace(
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        feature_batch_size=args.feature_batch_size,
        max_seconds=args.max_seconds,
        flow_grid_h=args.flow_grid_h,
        flow_grid_w=args.flow_grid_w,
        flow_temporal_smooth=args.flow_temporal_smooth,
        flow_pyramid_levels=args.flow_pyramid_levels,
    )
    features, starts, ends = extract_video(
        args.video, cache, encoder, device, extraction_args, args.encoder
    )
    del model, encoder

    normalization = np.load(args.probes / "normalization.npz")
    features = (features - normalization["mean"]) / normalization["std"]
    probabilities = {}
    available_probe_types = {}
    if (args.probes / "linear_probe.pt").exists():
        available_probe_types["linear"] = LinearProbe
    if (args.probes / "mlp_probe.pt").exists():
        available_probe_types["mlp"] = TinyMLP
    if (args.probes / "rf_probe.joblib").exists():
        available_probe_types["rf"] = "rf"
    if not available_probe_types:
        raise FileNotFoundError(f"No probe files found in {args.probes}")
    for name, probe_type in available_probe_types.items():
        if name == "rf":
            rf_probe = joblib_load(args.probes / f"{name}_probe.joblib")
            probabilities[name] = rf_probe.predict_proba(features)[:, 1]
            continue
        probe = probe_type(features.shape[1]).to(device)
        probe.load_state_dict(
            torch.load(args.probes / f"{name}_probe.pt", map_location="cpu", weights_only=True)
        )
        probabilities[name] = predict(probe, features, device)

    temporary_output = output.with_name(f"{output.stem}.part{output.suffix}")
    np.savez_compressed(
        temporary_output,
        video=str(args.video.resolve()),
        starts=starts,
        ends=ends,
        intervals=np.empty((0, 2), dtype=np.float64),
        labels=np.empty(0, dtype=np.float32),
        **probabilities,
    )
    temporary_output.replace(output)
    print(f"Saved {len(starts)} full-video window predictions to {output}")
    if args.render_video is not None:
        render_predictions(
            output, args.render_video, max_seconds=args.max_seconds,
            border_model=args.border_model, max_width=args.render_max_width,
        )


if __name__ == "__main__":
    main()
