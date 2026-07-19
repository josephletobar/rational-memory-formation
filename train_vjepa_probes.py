#!/usr/bin/env python3
"""Full-video frozen V-JEPA/VideoMAE or optical-flow pyramid features + linear/MLP/RF probes.

Uses the existing local Hugging Face cache by default (no downloads):

  conda run -n samworld python train_vjepa_probes.py

Annotations are adjacent ``VIDEO.mp4.clipme.json`` files in the data directory.
"""

import argparse
import json
import random
import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from joblib import dump as joblib_dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_DATA_DIR = Path("/Volumes/Crucial X9/theory-of-mind")
ENCODER_PRESETS = {
    "vjepa2": "facebook/vjepa2-vitl-fpc32-256-diving48",
    "videomae": "MCG-NJU/videomae-base",
    "optical_flow": "optical-flow",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path("."),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_rf"),
        help="Output directory (default: canonical optical-flow RF results).",
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
    p.add_argument(
        "--stride-frames",
        type=int,
        default=32,
        help="Window spacing; 32 covers the full video without redundant overlap",
    )
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
    p.add_argument("--overlap-threshold", type=float, default=0.5)
    p.add_argument(
        "--positive-label",
        default="goal_directed_activity,positive",
        help="Comma-separated annotation labels treated as positive.",
    )
    p.add_argument("--feature-batch-size", type=int, default=2)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--probe-types",
        default="rf",
        help="Comma-separated probe types to train (default: canonical RF): linear,mlp,rf.",
    )
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-n-jobs", type=int, default=-1)
    p.add_argument("--val-fraction", type=float, default=0.33)
    p.add_argument(
        "--fixed-train-videos",
        type=Path,
        default=None,
        help="Optional newline-delimited train-video basenames; requires --fixed-val-videos.",
    )
    p.add_argument(
        "--fixed-val-videos",
        type=Path,
        default=None,
        help="Optional newline-delimited held-out-video basenames; requires --fixed-train-videos.",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Stop after this many eligible annotated videos",
    )
    p.add_argument(
        "--include-video",
        action="append",
        default=None,
        metavar="FILENAME",
        help="Train only on these video basenames; repeatable",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Use this existing feature-cache directory instead of OUTPUT_DIR/feature_cache.",
    )
    p.add_argument(
        "--cached-only",
        action="store_true",
        help="Use only videos with an exact existing feature cache; never load an encoder or compute features.",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional cap in source seconds (quick smoke run).",
    )
    args = p.parse_args()
    return args


def gaussian_kernel_1d(ksize):
    if ksize <= 1:
        return np.array([1.0], dtype=np.float32)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    sigma = max(1e-6, ksize / 6.0)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def smooth_temporal_features(feature_sequence, fps, smooth_seconds):
    if feature_sequence.size == 0:
        return feature_sequence
    ksize = int(round(max(0.0, smooth_seconds) * max(1.0, fps)))
    if ksize <= 1:
        return feature_sequence
    if ksize % 2 == 0:
        ksize += 1
    kernel = gaussian_kernel_1d(ksize)
    x = torch.from_numpy(feature_sequence.astype(np.float32)).T.unsqueeze(0)
    pad = len(kernel) // 2
    x = F.pad(x, (pad, pad), mode="reflect")
    weight = torch.from_numpy(kernel).view(1, 1, -1)
    out = F.conv1d(x, weight.expand(x.shape[1], 1, -1), groups=x.shape[1])
    return out.squeeze(0).T.numpy()


def flow_cell_descriptor(prev_gray, next_gray, grid_h, grid_w):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    rows = np.linspace(0, flow.shape[0], grid_h + 1, dtype=np.int32)
    cols = np.linspace(0, flow.shape[1], grid_w + 1, dtype=np.int32)
    rows[-1] = flow.shape[0]
    cols[-1] = flow.shape[1]
    descriptor = []
    for r0, r1 in zip(rows[:-1], rows[1:]):
        for c0, c1 in zip(cols[:-1], cols[1:]):
            if r1 <= r0 or c1 <= c0:
                descriptor.extend((0.0, 0.0, 0.0, 0.0))
                continue
            patch = flow[r0:r1, c0:c1]
            dx = patch[:, :, 0]
            dy = patch[:, :, 1]
            descriptor.extend(
                (
                    float(dx.mean()),
                    float(dx.std()),
                    float(dy.mean()),
                    float(dy.std()),
                )
            )
    return np.array(descriptor, dtype=np.float32)


def temporal_pyramid_features(flow_seq, levels, grid_h, grid_w):
    feature_dim = grid_h * grid_w * 4
    if flow_seq.size == 0:
        total_segments = sum(2 ** (lvl - 1) for lvl in range(1, levels + 1))
        return np.zeros((feature_dim * 2 * total_segments,), dtype=np.float32)

    L = len(flow_seq)
    segments = []
    for level in range(levels):
        num_segments = 2**level
        seg_size = max(1.0, L / float(num_segments))
        for i in range(num_segments):
            start = int(round(i * seg_size))
            end = min(int(round((i + 1) * seg_size)), L)
            if start < end:
                segments.append((start, end))

    pieces = []
    for start, end in segments:
        chunk = flow_seq[start:end]
        if len(chunk) == 0:
            mean = np.zeros(feature_dim, dtype=np.float32)
            std = np.zeros(feature_dim, dtype=np.float32)
        else:
            mean = chunk.mean(axis=0).astype(np.float32)
            std = chunk.std(axis=0).astype(np.float32)
        pieces.append(mean)
        pieces.append(std)
    return np.concatenate(pieces).astype(np.float32)


def flow_window_feature(flow_sequence, window_start, window_frames, args):
    pair_start = max(0, int(window_start))
    pair_end = min(len(flow_sequence), pair_start + window_frames - 1)
    chunk = flow_sequence[pair_start:pair_end]
    return temporal_pyramid_features(
        chunk,
        levels=max(1, min(3, int(args.flow_pyramid_levels))),
        grid_h=args.flow_grid_h,
        grid_w=args.flow_grid_w,
    )


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_annotation(path, positive_labels):
    data = json.loads(path.read_text())
    video = path.parent / data["video"]
    intervals = [
        (float(a["start_sec"]), float(a["end_sec"]))
        for a in data["annotations"]
        if a.get("label") in positive_labels
    ]
    return video, intervals


def positive_fraction(start, end, intervals):
    """Fraction of a window covered by the union of positive intervals."""
    pieces = []
    for left, right in intervals:
        left, right = max(start, left), min(end, right)
        if right > left:
            pieces.append((left, right))
    if not pieces:
        return 0.0
    pieces.sort()
    covered = 0.0
    left, right = pieces[0]
    for next_left, next_right in pieces[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            covered += right - left
            left, right = next_left, next_right
    return (covered + right - left) / (end - start)


def prepare_frame(frame, size=256):
    """Official eval preprocessing: resize short side, center crop, RGB."""
    height, width = frame.shape[:2]
    short_side = round(size * 256 / 224)
    scale = short_side / min(height, width)
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    y = (resized.shape[0] - size) // 2
    x = (resized.shape[1] - size) // 2
    return cv2.cvtColor(resized[y : y + size, x : x + size], cv2.COLOR_BGR2RGB)


def encode_batch(clips, encoder, device, encoder_name, flow_args=None):
    if encoder_name == "optical_flow":
        if len(clips) != 1:
            raise ValueError("optical_flow encoding expects one window at a time")
        if flow_args is None:
            raise ValueError("flow_args required for optical_flow encoding")
        frames_rgb = clips[0]
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames_rgb]
        descriptors = []
        for previous, next_frame in zip(gray[:-1], gray[1:]):
            descriptors.append(
                flow_cell_descriptor(
                    previous,
                    next_frame,
                    grid_h=flow_args.flow_grid_h,
                    grid_w=flow_args.flow_grid_w,
                )
            )
        flow_seq = np.asarray(descriptors, dtype=np.float32)
        flow_seq = smooth_temporal_features(
            flow_seq,
            fps=getattr(flow_args, "source_fps", 15.0),
            smooth_seconds=flow_args.flow_temporal_smooth,
        )
        return flow_window_feature(flow_seq, 0, len(frames_rgb), flow_args)[None, :]

    pixels = np.stack(clips).astype(np.float32) / 255.0  # B,T,H,W,C
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    pixels = torch.from_numpy(pixels).permute(0, 1, 4, 2, 3).to(device)
    with torch.inference_mode():
        if encoder_name == "vjepa2":
            output = encoder(pixel_values_videos=pixels, skip_predictor=True)
            # V-JEPA has no CLS token. The official transfer setup learns an
            # attentive pool over all patch tokens; mean pooling is its
            # parameter-free analogue, preserving a genuinely single-layer linear
            # probe.
            embedding = output.last_hidden_state.mean(dim=1)
        else:
            output = encoder(pixel_values=pixels)
            embedding = (
                output.pooler_output
                if getattr(output, "pooler_output", None) is not None
                else output.last_hidden_state.mean(dim=1)
            )
    return embedding.float().cpu().numpy()


def save_feature_cache(cache, result):
    """Atomically publish a completed cache file.

    An interrupted extraction must never leave a truncated ``.npz`` that a
    later ``--cached-only`` study run mistakes for a completed video.
    """
    partial = cache.with_name(f"{cache.stem}.partial{cache.suffix}")
    partial.unlink(missing_ok=True)
    np.savez_compressed(partial, features=result[0], starts=result[1], ends=result[2])
    partial.replace(cache)


def extract_video(video, cache, encoder, device, args, encoder_name):
    if cache.exists():
        saved = np.load(cache)
        return saved["features"], saved["starts"], saved["ends"]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid frame rate for {video}")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_seconds = total_frames / fps if total_frames > 0 else 0.0

    # Preserve each checkpoint's native spatial evaluation geometry.  V-JEPA2
    # was cached at 256px, while this VideoMAE checkpoint has a fixed 224px
    # positional grid and rejects 256px inputs.
    input_size = 224 if encoder_name == "videomae" else 256
    ring, pending = deque(maxlen=args.window_frames), []
    features, starts, ends = [], [], []
    frame_index = -1

    if encoder_name == "optical_flow":
        prev_gray = None
        flow_sequence = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if args.max_seconds is not None and frame_index / fps >= args.max_seconds:
                break
            frame_rgb = prepare_frame(frame)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                flow_sequence.append(
                    flow_cell_descriptor(
                        prev_gray,
                        gray,
                        grid_h=args.flow_grid_h,
                        grid_w=args.flow_grid_w,
                    )
                )
            prev_gray = gray

        if not flow_sequence:
            cap.release()
            raise RuntimeError(f"No full windows found in {video}")

        flow_sequence = np.asarray(flow_sequence, dtype=np.float32)
        flow_sequence = smooth_temporal_features(
            flow_sequence,
            fps,
            smooth_seconds=args.flow_temporal_smooth,
        )
        total_frames_seen = frame_index + 1
        window_pairs = args.window_frames - 1
        max_start = total_frames_seen - args.window_frames
        for start_frame in range(0, max_start + 1):
            if start_frame % args.stride_frames != 0:
                continue
            if start_frame + window_pairs > len(flow_sequence):
                break
            feature = flow_window_feature(
                flow_sequence,
                start_frame,
                args.window_frames,
                args,
            )
            features.append(feature)
            starts.append(start_frame / fps)
            ends.append((start_frame + args.window_frames) / fps)
            if len(starts) % 50 == 0:
                progress = min(total_seconds, ends[-1]) if total_frames > 0 else 0.0
                print(
                    f"  {video.name}: {ends[-1]:.1f}s / {progress:.1f}s "
                    f"({len(starts)} windows)",
                    flush=True,
                )

        cap.release()
        if not features:
            raise RuntimeError(f"No full windows found in {video}")
        result = (np.stack(features), np.asarray(starts), np.asarray(ends))
        save_feature_cache(cache, result)
        return result

    def flush():
        if pending:
            features.append(encode_batch(pending, encoder, device, encoder_name))
            pending.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if args.max_seconds is not None and frame_index / fps >= args.max_seconds:
            break
        ring.append(prepare_frame(frame, size=input_size))
        start_frame = frame_index - args.window_frames + 1
        if len(ring) == args.window_frames and start_frame % args.stride_frames == 0:
            pending.append(np.stack(ring))
            starts.append(start_frame / fps)
            ends.append((start_frame + args.window_frames) / fps)
            if len(starts) % 50 == 0:
                print(
                    f"  {video.name}: {ends[-1]:.1f}s / {total_seconds:.1f}s "
                    f"({len(starts)} windows)",
                    flush=True,
                )
            if len(pending) == args.feature_batch_size:
                flush()
    flush()
    cap.release()
    if not features:
        raise RuntimeError(f"No full windows found in {video}")
    result = (np.concatenate(features), np.asarray(starts), np.asarray(ends))
    save_feature_cache(cache, result)
    return result


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


class TinyMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit(model, x, y, args, device):
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float()),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
    return model


def predict(model, x, device):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(x).float().to(device))
    return torch.sigmoid(logits).cpu().numpy()


def fit_random_forest(train_x, train_y, args):
    max_depth = args.rf_max_depth
    if max_depth is not None and max_depth <= 0:
        max_depth = None
    return RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=max_depth,
        random_state=args.seed,
        n_jobs=args.rf_n_jobs,
    ).fit(train_x, train_y)


def predict_random_forest(model, x):
    return model.predict_proba(x)[:, 1]


def metrics(y, probability):
    predicted = probability >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y, probability)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def svg_timeline(path, title, starts, ends, intervals, probabilities):
    """Write a timeline without a plotting dependency."""
    width, height, margin = 1000, 360, 60
    duration = max(float(ends[-1]), max((x[1] for x in intervals), default=0.0))
    sx = lambda t: margin + (width - 2 * margin) * t / duration
    sy = lambda p: height - margin - (height - 2 * margin) * p
    colors = {"linear": "#1464f4", "mlp": "#e04b36", "rf": "#8a2be2"}
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="sans-serif" font-size="18">{safe_title}</text>',
    ]
    for left, right in intervals:
        parts.append(
            f'<rect x="{sx(left):.1f}" y="{margin}" width="{sx(right)-sx(left):.1f}" '
            f'height="{height-2*margin}" fill="#ffd966" opacity="0.35"/>'
        )
    for value in (0.0, 0.5, 1.0):
        parts.append(f'<line x1="{margin}" y1="{sy(value):.1f}" x2="{width-margin}" y2="{sy(value):.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="18" y="{sy(value)+5:.1f}" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    centers = (starts + ends) / 2
    for name, values in probabilities.items():
        color = colors.get(name, "#444444")
        points = " ".join(f"{sx(t):.1f},{sy(p):.1f}" for t, p in zip(centers, values))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    legend = []
    if "linear" in probabilities:
        legend.append(('linear', colors["linear"]))
    if "mlp" in probabilities:
        legend.append(('MLP', colors["mlp"]))
    if "rf" in probabilities:
        legend.append(('RF', colors["rf"]))
    for i, (label, color) in enumerate(legend):
        parts.append(
            f'<text x="{width-190}" y="{28 + 20 * i}" '
            f'fill="{color}" font-family="sans-serif" font-size="12">{label}</text>'
        )
    parts.append(
        f'<text x="{margin}" y="{height-18}" font-family="sans-serif" font-size="12">0 s</text>'
    )
    parts.append(
        f'<text x="{width-margin-50}" y="{height-18}" font-family="sans-serif" font-size="12">{duration:.0f} s</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    args = parse_args()
    if args.window_frames < 2 or args.stride_frames < 1:
        raise ValueError("window-frames must be >= 2 and stride-frames >= 1")
    if not 0 <= args.overlap_threshold <= 1:
        raise ValueError("overlap-threshold must be between 0 and 1")
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("max-videos must be at least 1")
    if bool(args.fixed_train_videos) != bool(args.fixed_val_videos):
        raise ValueError("--fixed-train-videos and --fixed-val-videos must be supplied together")
    probe_types = [item.strip().lower() for item in args.probe_types.split(",") if item.strip()]
    if not probe_types:
        raise ValueError("--probe-types must include at least one of linear, mlp, rf")
    invalid_probes = [name for name in probe_types if name not in {"linear", "mlp", "rf"}]
    if invalid_probes:
        raise ValueError(
            f"Unsupported --probe-types value(s): {invalid_probes}; use linear, mlp, rf."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or args.output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    included = set(args.include_video) if args.include_video else None
    found_included = set()
    positive_labels = {label.strip() for label in args.positive_label.split(",") if label.strip()}
    if not positive_labels:
        raise ValueError("--positive-label must contain at least one label")
    annotations = (
        path for path in args.data_dir.rglob("*.mp4.clipme.json")
        if not path.name.startswith("._")
        and not any(marker in path.name for marker in (".OLD", ".PARTIAL-OLD", ".WHOLE_EPISODE_OLD"))
    )
    for annotation in sorted(annotations):
        video, intervals = read_annotation(annotation, positive_labels)
        if included is not None and video.name not in included:
            continue
        if not video.exists():
            raise FileNotFoundError(video)
        records.append((video, intervals))
        found_included.add(video.name)
        if args.max_videos is not None and len(records) >= args.max_videos:
            break
    if not records:
        raise ValueError("At least one eligible annotated video is required")
    if included is not None and found_included != included:
        missing = sorted(included - found_included)
        raise ValueError(f"Requested training videos were not found/eligible: {missing}")
    if args.encoder == "optical_flow":
        if args.flow_grid_h < 1 or args.flow_grid_w < 1:
            raise ValueError("flow-grid-h and flow-grid-w must be >= 1")
        if args.flow_pyramid_levels < 1 or args.flow_pyramid_levels > 3:
            raise ValueError("flow-pyramid-levels must be 1, 2, or 3")

    device = choose_device(args.device)
    if args.encoder == "optical_flow":
        print("Using optical-flow features; no HF encoder loaded")
        model_id = ENCODER_PRESETS[args.encoder]
        model = None
        encoder = None
        slug = f"optflow.h{args.flow_grid_h}x{args.flow_grid_w}.lvl{args.flow_pyramid_levels}.ts{args.flow_temporal_smooth:g}"
    else:
        model_id = args.model if args.model is not None else ENCODER_PRESETS[args.encoder]
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-")
        if args.cached_only:
            print("Cached-only mode: encoder inference disabled")
            model = encoder = None
        else:
            print(f"Loading frozen encoder on {device} (downloads allowed: {args.allow_download})")
            model = AutoModel.from_pretrained(
                model_id, local_files_only=not args.allow_download
            )
            encoder = model.vjepa2 if hasattr(model, "vjepa2") else model
            encoder.requires_grad_(False).eval().to(device)
    if args.cached_only:
        limit = "full" if args.max_seconds is None else f"t{args.max_seconds:g}"
        uncached = []
        cached_records = []
        for video, intervals in records:
            cache = cache_dir / f"{video.stem}.{slug}.w{args.window_frames}.s{args.stride_frames}.{limit}.npz"
            if cache.exists():
                cached_records.append((video, intervals))
            else:
                uncached.append(video.name)
        records = cached_records
        print(f"Cached-only: using {len(records)} videos; skipping {len(uncached)} without exact caches")
        if uncached:
            print("Skipped uncached videos: " + ", ".join(sorted(uncached)))
        if not records:
            raise ValueError("No eligible videos have matching feature caches")
    data = {}
    feature_loading_started = time.perf_counter()
    for video, intervals in records:
        print(f"Extracting/loading windows: {video.name}")
        limit = "full" if args.max_seconds is None else f"t{args.max_seconds:g}"
        cache = cache_dir / f"{video.stem}.{slug}.w{args.window_frames}.s{args.stride_frames}.{limit}.npz"
        features, starts, ends = extract_video(
            video, cache, encoder, device, args, args.encoder
        )
        labels = np.asarray(
            [positive_fraction(a, b, intervals) >= args.overlap_threshold for a, b in zip(starts, ends)],
            dtype=np.float32,
        )
        data[video.name] = dict(
            features=features, starts=starts, ends=ends, labels=labels, intervals=intervals
        )
    feature_loading_seconds = time.perf_counter() - feature_loading_started
    del model, encoder

    videos = sorted(data)
    validation_parts = []
    if args.fixed_train_videos is not None:
        train_videos = [
            line.strip() for line in args.fixed_train_videos.read_text().splitlines() if line.strip()
        ]
        val_videos = [
            line.strip() for line in args.fixed_val_videos.read_text().splitlines() if line.strip()
        ]
        if not train_videos or not val_videos:
            raise ValueError("fixed train and validation splits must both be non-empty")
        if set(train_videos) & set(val_videos):
            raise ValueError("fixed train and validation splits overlap")
        if set(train_videos) | set(val_videos) != set(videos):
            missing = sorted(set(videos) - (set(train_videos) | set(val_videos)))
            unexpected = sorted((set(train_videos) | set(val_videos)) - set(videos))
            raise ValueError(f"fixed split does not exactly match loaded data; missing={missing}, unexpected={unexpected}")
        train_x = np.concatenate([data[v]["features"] for v in train_videos])
        train_y = np.concatenate([data[v]["labels"] for v in train_videos])
        val_x = np.concatenate([data[v]["features"] for v in val_videos])
        val_y = np.concatenate([data[v]["labels"] for v in val_videos])
        offset = 0
        for video in val_videos:
            indices = np.arange(len(data[video]["labels"]))
            validation_parts.append((video, indices, offset))
            offset += len(indices)
        split_scope = "fixed_held-out_videos"
    elif len(videos) == 1:
        video = videos[0]
        labels = data[video]["labels"]
        indices = np.arange(len(labels))
        counts = np.bincount(labels.astype(np.int64), minlength=2)
        stratify = labels if counts.min() >= 2 else None
        train_indices, val_indices = train_test_split(
            indices,
            test_size=args.val_fraction,
            random_state=args.seed,
            stratify=stratify,
        )
        train_videos = val_videos = [video]
        train_x = data[video]["features"][train_indices]
        train_y = labels[train_indices]
        val_x = data[video]["features"][val_indices]
        val_y = labels[val_indices]
        validation_parts.append((video, np.sort(val_indices), 0))
        split_scope = "held-out_windows_within_one_full_video"
    else:
        random.Random(args.seed).shuffle(videos)
        n_val = min(len(videos) - 1, max(1, round(len(videos) * args.val_fraction)))
        val_videos, train_videos = videos[:n_val], videos[n_val:]
        train_x = np.concatenate([data[v]["features"] for v in train_videos])
        train_y = np.concatenate([data[v]["labels"] for v in train_videos])
        val_x = np.concatenate([data[v]["features"] for v in val_videos])
        val_y = np.concatenate([data[v]["labels"] for v in val_videos])
        offset = 0
        for video in val_videos:
            indices = np.arange(len(data[video]["labels"]))
            validation_parts.append((video, indices, offset))
            offset += len(indices)
        split_scope = "held-out_videos"

    # Fit normalization on training videos only.
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    probes_to_train = []
    for name in probe_types:
        if name == "linear":
            probes_to_train.append((name, LinearProbe(train_x.shape[1])))
        elif name == "mlp":
            probes_to_train.append((name, TinyMLP(train_x.shape[1])))
        else:
            probes_to_train.append((name, None))
    probabilities, report = {}, {
        "full_video_training": True,
        "split_scope": split_scope,
        "included_videos": sorted(included) if included is not None else None,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "positive_labels": sorted(positive_labels),
        "cached_only": args.cached_only,
        "feature_loading_seconds": feature_loading_seconds,
    }
    for name, probe in probes_to_train:
        training_started = time.perf_counter()
        if name == "rf":
            rf_model = fit_random_forest(train_x, train_y, args)
            probabilities[name] = predict_random_forest(rf_model, val_x)
            report[name] = metrics(val_y, probabilities[name])
            report[name]["training_seconds"] = time.perf_counter() - training_started
            joblib_dump(rf_model, args.output_dir / f"{name}_probe.joblib")
            continue

        fit(probe, train_x, train_y, args, device)
        probabilities[name] = predict(probe, val_x, device)
        report[name] = metrics(val_y, probabilities[name])
        report[name]["training_seconds"] = time.perf_counter() - training_started
        torch.save(probe.cpu().state_dict(), args.output_dir / f"{name}_probe.pt")

    for video, indices, offset in validation_parts:
        n = len(indices)
        video_probabilities = {
            name: values[offset : offset + n] for name, values in probabilities.items()
        }
        svg_timeline(
            args.output_dir / f"timeline_{Path(video).stem}.svg",
            video,
            data[video]["starts"][indices],
            data[video]["ends"][indices],
            data[video]["intervals"],
            video_probabilities,
        )
        np.savez_compressed(
            args.output_dir / f"predictions_{Path(video).stem}.npz",
            video=str(args.data_dir / video),
            starts=data[video]["starts"][indices],
            ends=data[video]["ends"][indices],
            intervals=np.asarray(data[video]["intervals"], dtype=np.float64).reshape(-1, 2),
            labels=data[video]["labels"][indices],
            **video_probabilities,
        )

    np.savez(args.output_dir / "normalization.npz", mean=mean, std=std)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
