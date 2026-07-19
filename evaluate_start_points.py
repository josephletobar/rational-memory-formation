#!/usr/bin/env python3
"""Reproduce Su and Grauman Figure 6-style start-point evaluation locally.

The paper defines F1 against the nearest ground-truth start within an allowed
temporal error but does not specify duplicate-match or tie handling.  We make
that ambiguity explicit: within each video and tolerance, candidate
prediction/ground-truth pairs are sorted by absolute temporal error, then by
their timestamps, and greedily accepted only when both starts are unmatched.
This is one-to-one nearest-neighbor matching.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


X9 = Path("/Volumes/Crucial X9/theory-of-mind")
STUDY = X9 / "robustness_study"
POSITIVE_LABELS = {"goal_directed_activity", "positive"}
TOLERANCES = tuple(range(11))


@dataclass(frozen=True)
class Run:
    name: str
    encoder: str
    window: int
    stride: int
    result_dir: Path


def parse_window_stride(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"w(\d+)_s(\d+)", name)
    if not match:
        raise ValueError(f"cannot parse window/stride from {name}")
    return int(match.group(1)), int(match.group(2))


def discover_runs() -> list[Run]:
    specs = (
        (
            "Optical flow",
            "optical_flow",
            STUDY / "optical_flow_window_context",
            STUDY / "feature_cache/optical_flow",
            "optflow.h16x12.lvl3.ts2",
        ),
        (
            "VideoMAE",
            "videomae",
            STUDY,
            STUDY / "feature_cache/videomae",
            "MCG-NJU-videomae-base",
        ),
        (
            "V-JEPA 2",
            "vjepa2",
            STUDY,
            STUDY / "feature_cache/vjepa2",
            "facebook-vjepa2-vitl-fpc32-256-diving48",
        ),
    )
    runs: list[Run] = []
    for display, encoder, results_root, cache_root, slug in specs:
        if encoder == "optical_flow":
            candidate_dirs = sorted(results_root.glob("w*_s*"))
        else:
            candidate_dirs = sorted(results_root.glob(f"quick_{encoder}_w*_s*_rf"))
        for result_dir in candidate_dirs:
            if not (result_dir / "metrics.json").is_file() or not (result_dir / "rf_probe.joblib").is_file():
                continue
            config_name = result_dir.name if encoder == "optical_flow" else result_dir.name.split("_")[2] + "_" + result_dir.name.split("_")[3]
            window, stride = parse_window_stride(config_name)
            cache_dir = cache_root / config_name
            if not cache_dir.is_dir():
                continue
            runs.append(Run(f"{display} W{window}/S{stride}", encoder, window, stride, result_dir))
    return runs


def ground_truth_starts(annotation: Path, positive_labels: set[str]) -> np.ndarray:
    data = json.loads(annotation.read_text())
    starts = sorted(
        float(item["start_sec"])
        for item in data["annotations"]
        if item.get("label") in positive_labels
    )
    # Annotation intervals should be distinct; collapse exact duplicated starts
    # so a duplicate label cannot create two targets for one onset.
    return np.asarray(sorted(set(starts)), dtype=np.float64)


def predicted_starts(predictions: Path, threshold: float) -> np.ndarray:
    """Use the original saved held-out RF probabilities, not a reloaded model."""
    with np.load(predictions, allow_pickle=False) as values:
        starts = values["starts"].astype(np.float64)
        positive = values["rf"] >= threshold
    onset = positive & np.r_[True, ~positive[:-1]]
    return starts[onset]


def confusion_counts(predictions: Path, threshold: float) -> dict[str, int]:
    """Window-level binary counts from the original saved held-out outputs."""
    with np.load(predictions, allow_pickle=False) as values:
        truth = values["labels"].astype(bool)
        predicted = values["rf"] >= threshold
    return {
        "tp": int(np.sum(predicted & truth)),
        "tn": int(np.sum(~predicted & ~truth)),
        "fp": int(np.sum(predicted & ~truth)),
        "fn": int(np.sum(~predicted & truth)),
    }


def one_to_one_matches(predicted: np.ndarray, truth: np.ndarray, tolerance: float) -> int:
    """Globally nearest-first matching within a single video.

    Sorting all eligible pairs by distance ensures each accepted pair is the
    nearest available counterpart when it is accepted.  Timestamp ordering is
    only a deterministic tie breaker.
    """
    pairs = [
        (abs(float(p - g)), float(p), float(g), p_index, g_index)
        for p_index, p in enumerate(predicted)
        for g_index, g in enumerate(truth)
        if abs(float(p - g)) <= tolerance
    ]
    pairs.sort()
    used_pred, used_truth, matches = set(), set(), 0
    for _, _, _, p_index, g_index in pairs:
        if p_index not in used_pred and g_index not in used_truth:
            used_pred.add(p_index)
            used_truth.add(g_index)
            matches += 1
    return matches


def evaluate(run: Run, threshold: float) -> dict:
    metrics = json.loads((run.result_dir / "metrics.json").read_text())
    val_videos = metrics["val_videos"]
    positive_labels = set(metrics.get("positive_labels", POSITIVE_LABELS))

    predicted_by_video: dict[str, np.ndarray] = {}
    truth_by_video: dict[str, np.ndarray] = {}
    classification = defaultdict(int)
    for video_name in val_videos:
        annotation = X9 / f"{video_name}.clipme.json"
        if not annotation.is_file():
            raise FileNotFoundError(f"missing annotation: {annotation}")
        prediction_file = run.result_dir / f"predictions_{Path(video_name).stem}.npz"
        if not prediction_file.is_file():
            raise FileNotFoundError(f"missing saved held-out prediction: {prediction_file}")
        predicted_by_video[video_name] = predicted_starts(prediction_file, threshold)
        truth_by_video[video_name] = ground_truth_starts(annotation, positive_labels)
        for key, value in confusion_counts(prediction_file, threshold).items():
            classification[key] += value

    rows = []
    total_pred = sum(len(x) for x in predicted_by_video.values())
    total_truth = sum(len(x) for x in truth_by_video.values())
    for tolerance in TOLERANCES:
        matches = sum(
            one_to_one_matches(predicted_by_video[video], truth_by_video[video], tolerance)
            for video in val_videos
        )
        precision = matches / total_pred if total_pred else 0.0
        recall = matches / total_truth if total_truth else 0.0
        f1 = 2 * matches / (total_pred + total_truth) if (total_pred + total_truth) else 0.0
        rows.append({"tolerance_seconds": tolerance, "matches": matches, "precision": precision, "recall": recall, "f1": f1})
    return {
        "run": run.name,
        "encoder": run.encoder,
        "window_frames": run.window,
        "stride_frames": run.stride,
        "threshold": threshold,
        "heldout_videos": val_videos,
        "num_predicted_starts": total_pred,
        "num_ground_truth_starts": total_truth,
        "confusion_matrix": dict(classification),
        "rows": rows,
    }


def write_report(results: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "start_point_results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Start-point evaluation (Su and Grauman Figure 6 style)",
        "",
        "Evaluation uses only the locked nine held-out videos for each saved RF run. Predicted starts are the start timestamps of contiguous positive (probability >= 0.5) window runs; ground-truth starts are positive annotation interval start timestamps.",
        "",
        "**Implementation assumption - one-to-one nearest-neighbor matching:** Su and Grauman specify the nearest ground-truth start within tolerance but do not state how duplicate matches or ties are handled. Within each video and tolerance, this implementation forms every eligible predicted/ground-truth pair, sorts by absolute error (then timestamps as deterministic tie breakers), and greedily accepts a pair only if both starts remain unmatched.",
        "",
        "| Model | Predicted starts | Ground-truth starts | " + " | ".join(f"F1 @ {t}s" for t in TOLERANCES) + " |",
        "|---|---:|---:|" + "|".join("---:" for _ in TOLERANCES) + "|",
    ]
    for result in results:
        cells = [
            result["run"],
            str(result["num_predicted_starts"]),
            str(result["num_ground_truth_starts"]),
            *(f"{row['f1']:.4f}" for row in result["rows"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Held-out window-level binary classification",
        "",
        "These counts answer ordinary engaged/non-engaged classification. They are computed from the exact saved held-out window labels and RF probabilities at the same 0.5 threshold; they are distinct from the start-point matching above.",
        "",
        "| Model | TP | TN | FP | FN | Accuracy | Sensitivity / recall | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        counts = result["confusion_matrix"]
        tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        lines.append(
            f"| {result['run']} | {tp} | {tn} | {fp} | {fn} | {accuracy:.4f} | "
            f"{sensitivity:.4f} | {specificity:.4f} |"
        )
    (output / "start_point_results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=STUDY / "start_point_evaluation")
    args = parser.parse_args()
    runs = discover_runs()
    if not runs:
        raise RuntimeError("No compatible saved RF runs found")
    results = [evaluate(run, args.threshold) for run in runs]
    write_report(results, args.output)
    print(args.output / "start_point_results.md")


if __name__ == "__main__":
    main()
