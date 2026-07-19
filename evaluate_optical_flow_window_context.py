#!/usr/bin/env python3
"""Evaluate longer optical-flow windows under the locked robustness protocol.

Only the temporal window and matched no-overlap stride vary.  Feature caches
must already exist; extraction is deliberately kept outside this evaluator.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("OPTICAL_FLOW_STUDY_ROOT", "/Volumes/Crucial X9/theory-of-mind/robustness_study"))
DATA = Path(os.environ.get("OPTICAL_FLOW_DATA_DIR", "/Volumes/Crucial X9/theory-of-mind"))
SPLIT_METRICS = Path(os.environ.get(
    "OPTICAL_FLOW_SPLIT_METRICS",
    DATA / "trained_models/optical_flow_28cached_w32s32_rf/metrics.json",
))
FEATURE_ROOT = Path(os.environ.get("OPTICAL_FLOW_FEATURE_ROOT", ROOT / "feature_cache/optical_flow"))
RESULTS_ROOT = Path(os.environ.get("OPTICAL_FLOW_RESULTS_ROOT", ROOT / "optical_flow_window_context"))
PROBE = Path(__file__).with_name("train_vjepa_probes.py")
SLUG = "optflow.h16x12.lvl3.ts2"
# Four context lengths, each evaluated at no overlap, 50% overlap, and 75%
# overlap.  This is a sampling-sensitivity study rather than a test-set tune:
# every row uses the same locked split and RF probe.
CONFIGS = (
    (16, 16), (16, 8), (16, 4),
    (32, 32), (32, 16), (32, 8),
    (64, 64), (64, 32), (64, 16),
    (128, 128), (128, 64), (128, 32),
)


def selected_configs() -> tuple[tuple[int, int], ...]:
    """Optionally run a resumable subset without changing the protocol."""
    requested = os.environ.get("OPTICAL_FLOW_CONFIGS", "").strip()
    if not requested:
        return CONFIGS
    configs = []
    for entry in requested.split(","):
        values = re.findall(r"\d+", entry)
        if len(values) != 2:
            raise ValueError(f"expected window/stride pair, got: {entry}")
        pair = tuple(map(int, values))
        if pair not in CONFIGS:
            raise ValueError(f"unsupported optical-flow configuration: {entry}")
        configs.append(pair)
    return tuple(configs)


def cache_dir(window: int, stride: int) -> Path:
    return FEATURE_ROOT / f"w{window}_s{stride}"


def cache_stats(window: int, stride: int, videos: list[str]) -> dict[str, float | int]:
    entries = []
    missing = []
    for video in videos:
        cache = cache_dir(window, stride) / f"{Path(video).stem}.{SLUG}.w{window}.s{stride}.full.npz"
        stats = cache.with_name(cache.name + ".stats.json")
        if not cache.is_file() or not stats.is_file():
            missing.append(video)
            continue
        entries.append(json.loads(stats.read_text()))
    if missing:
        raise RuntimeError(f"W{window}/S{stride} is missing {len(missing)} locked caches: {missing[:3]}")
    dims = {entry["feature_dimensionality"] for entry in entries}
    if len(dims) != 1:
        raise RuntimeError(f"W{window}/S{stride} inconsistent feature dimensions: {dims}")
    return {
        "feature_dimensionality": dims.pop(),
        "feature_extraction_seconds": sum(entry["feature_extraction_seconds"] for entry in entries),
        "total_windows": sum(entry["num_windows"] for entry in entries),
    }


def main() -> None:
    split = json.loads(SPLIT_METRICS.read_text())
    train, heldout = split["train_videos"], split["val_videos"]
    videos = train + heldout
    if len(train) != 19 or len(heldout) != 9 or len(set(videos)) != 28:
        raise RuntimeError("expected the locked 19-train / 9-held-out split")

    train_file = ROOT / "locked_train_videos.txt"
    heldout_file = ROOT / "locked_heldout_videos.txt"
    train_file.write_text("\n".join(train) + "\n")
    heldout_file.write_text("\n".join(heldout) + "\n")

    rows = []
    for window, stride in selected_configs():
        stats = cache_stats(window, stride, videos)
        output = RESULTS_ROOT / f"w{window}_s{stride}"
        command = [
            sys.executable, str(PROBE), "--data-dir", str(DATA),
            "--output-dir", str(output), "--encoder", "optical_flow",
            "--window-frames", str(window), "--stride-frames", str(stride),
            "--feature-cache-dir", str(cache_dir(window, stride)), "--cached-only",
            "--probe-types", "rf", "--rf-n-estimators", "200", "--rf-n-jobs",
            os.environ.get("OPTICAL_FLOW_RF_N_JOBS", "-1"),
            "--seed", "0", "--device", "cpu",
            "--positive-label", "goal_directed_activity,positive",
            "--fixed-train-videos", str(train_file), "--fixed-val-videos", str(heldout_file),
        ]
        for video in videos:
            command.extend(("--include-video", video))
        subprocess.run(command, check=True)
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics["train_videos"] != train or metrics["val_videos"] != heldout:
            raise RuntimeError(f"W{window}/S{stride} did not use the locked split")
        rows.append({"window": window, "stride": stride, **stats, **metrics["rf"]})

    out_dir = RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Optical-flow temporal-context analysis", "",
        "Protocol held fixed: the locked 19/9 video split, seed 0, 200-tree Random Forest, labels, positive class, flow grid/pyramid/smoothing, and 0.5 decision threshold. The only varied factors are temporal window and stride: no overlap (S=W), 50% overlap (S=W/2), and 75% overlap (S=W/4).", "",
        "| Window | Stride | F1 | Precision | Recall | ROC AUC | Feature dim | Extraction time (s) | RF time (s) |", 
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['window']} | {row['stride']} | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['roc_auc']:.4f} | {row['feature_dimensionality']} | "
            f"{row['feature_extraction_seconds']:.1f} | {row['training_seconds']:.3f} |"
        )
    best = max(rows, key=lambda row: row["f1"])
    lines += [
        "",
        f"Best fixed-split configuration: W{best['window']}/S{best['stride']} (F1 {best['f1']:.4f}, AUC {best['roc_auc']:.4f}).",
        "Interpret this as a fixed-split temporal-context sensitivity check, not a general claim about optical flow.",
    ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "results.md")


if __name__ == "__main__":
    main()
