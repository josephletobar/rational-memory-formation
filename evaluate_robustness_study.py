#!/usr/bin/env python3
"""Evaluate the pre-registered temporal robustness configurations.

This script never extracts features.  It requires all 28 exact caches for a
row, fixes the historical 19/9 split explicitly, and records the requested
metrics and timings in one reproducible study summary.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Volumes/Crucial X9/theory-of-mind/robustness_study")
DATA = Path("/Volumes/Crucial X9/theory-of-mind")
SPLIT_SOURCE = DATA / "trained_models/optical_flow_28cached_w32s32_rf/metrics.json"
PROBE = Path(__file__).with_name("train_vjepa_probes.py")

# (display encoder, requested W/S, actual W/S, cache encoder, note)
ROWS = [
    ("Optical Flow", 32, 32, 32, 32, "optical_flow", "Existing flow implementation, unchanged."),
    ("VideoMAE", 16, 8, 16, 8, "videomae", "Native 16-frame configuration."),
    ("VideoMAE", 32, 16, 16, 16, "videomae", "Requested W32 unsupported; closest native W16 with requested stride S16."),
    ("VideoMAE", 64, 32, 16, 32, "videomae", "Requested W64 unsupported; closest native W16 with requested stride S32."),
    ("VideoMAE", 16, 16, 16, 16, "videomae", "Native 16-frame configuration."),
    ("VideoMAE", 32, 32, 16, 32, "videomae", "Requested W32 unsupported; closest native W16 with requested stride S32."),
    ("VideoMAE", 64, 64, 16, 64, "videomae", "Requested W64 unsupported; closest native W16 with requested stride S64."),
    ("VideoMAE", 16, 4, 16, 4, "videomae", "Native 16-frame 75%-overlap configuration."),
    ("VideoMAE", 32, 8, 16, 8, "videomae", "Requested W32 unsupported; closest native W16 with requested stride S8."),
    ("VideoMAE", 64, 16, 16, 16, "videomae", "Requested W64 unsupported; closest native W16 with requested stride S16."),
    ("V-JEPA", 16, 8, 16, 8, "vjepa2", "Native supported configuration."),
    ("V-JEPA", 32, 16, 32, 16, "vjepa2", "Native supported configuration."),
    ("V-JEPA", 64, 32, 64, 32, "vjepa2", "Native supported configuration."),
    ("V-JEPA", 16, 16, 16, 16, "vjepa2", "Native supported configuration."),
    ("V-JEPA", 32, 32, 32, 32, "vjepa2", "Native supported configuration."),
    ("V-JEPA", 64, 64, 64, 64, "vjepa2", "Native supported configuration."),
    ("V-JEPA", 16, 4, 16, 4, "vjepa2", "Native supported 75%-overlap configuration."),
    ("V-JEPA", 32, 8, 32, 8, "vjepa2", "Native supported 75%-overlap configuration."),
    ("V-JEPA", 64, 16, 64, 16, "vjepa2", "Native supported 75%-overlap configuration."),
]
SLUGS = {
    "optical_flow": "optflow.h16x12.lvl3.ts2",
    "videomae": "MCG-NJU-videomae-base",
    "vjepa2": "facebook-vjepa2-vitl-fpc32-256-diving48",
}


def cache_dir(encoder, window, stride):
    return ROOT / "feature_cache" / encoder / f"w{window}_s{stride}"


def cache_stats(encoder, window, stride, videos):
    directory = cache_dir(encoder, window, stride)
    slug = SLUGS[encoder]
    stats = []
    missing = []
    for video in videos:
        cache = directory / f"{Path(video).stem}.{slug}.w{window}.s{stride}.full.npz"
        sidecar = cache.with_name(cache.name + ".stats.json")
        if not cache.exists() or not sidecar.exists():
            missing.append(video)
            continue
        stats.append(json.loads(sidecar.read_text()))
    if missing:
        raise RuntimeError(f"{encoder} W{window} S{stride}: missing exact cache/stats for {missing}")
    dims = {entry["feature_dimensionality"] for entry in stats}
    if len(dims) != 1:
        raise RuntimeError(f"inconsistent feature dimensions: {dims}")
    return {
        "feature_dimensionality": dims.pop(),
        "feature_extraction_seconds": sum(entry["feature_extraction_seconds"] for entry in stats),
        "total_windows": sum(entry["num_windows"] for entry in stats),
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_SOURCE.read_text())
    train = split["train_videos"]
    heldout = split["val_videos"]
    videos = train + heldout
    if len(train) != 19 or len(heldout) != 9 or len(set(videos)) != 28:
        raise RuntimeError("the locked split is not the required 19 train / 9 held-out videos")
    train_file = ROOT / "locked_train_videos.txt"
    heldout_file = ROOT / "locked_heldout_videos.txt"
    train_file.write_text("\n".join(train) + "\n")
    heldout_file.write_text("\n".join(heldout) + "\n")

    unique = {}
    for display, req_w, req_s, actual_w, actual_s, encoder, note in ROWS:
        unique[(encoder, actual_w, actual_s)] = (display, actual_w, actual_s)

    results = {}
    for (encoder, window, stride), _ in unique.items():
        stat = cache_stats(encoder, window, stride, videos)
        output = ROOT / "probe_results" / encoder / f"w{window}_s{stride}"
        command = [
            sys.executable, str(PROBE), "--data-dir", str(DATA), "--output-dir", str(output),
            "--encoder", encoder, "--window-frames", str(window), "--stride-frames", str(stride),
            "--feature-cache-dir", str(cache_dir(encoder, window, stride)), "--cached-only",
            "--probe-types", "rf", "--rf-n-estimators", "200", "--rf-n-jobs", "-1",
            "--seed", "0", "--device", "cpu", "--positive-label", "goal_directed_activity,positive",
            "--fixed-train-videos", str(train_file), "--fixed-val-videos", str(heldout_file),
        ]
        for video in videos:
            command.extend(["--include-video", video])
        subprocess.run(command, check=True)
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics["train_videos"] != train or metrics["val_videos"] != heldout:
            raise RuntimeError(f"{encoder} W{window} S{stride}: fixed split verification failed")
        rf = metrics["rf"]
        results[(encoder, window, stride)] = {**stat, **rf, "probe_result_dir": str(output)}

    rendered = []
    for display, req_w, req_s, actual_w, actual_s, encoder, note in ROWS:
        data = results[(encoder, actual_w, actual_s)]
        rendered.append({
            "encoder": display, "requested_window": req_w, "requested_stride": req_s,
            "actual_window": actual_w, "actual_stride": actual_s, "cache_encoder": encoder,
            "f1": data["f1"], "precision": data["precision"], "recall": data["recall"],
            "roc_auc": data["roc_auc"], "feature_dimensionality": data["feature_dimensionality"],
            "feature_extraction_seconds": data["feature_extraction_seconds"],
            "rf_training_seconds": data["training_seconds"], "total_windows": data["total_windows"],
            "note": note,
        })
    (ROOT / "robustness_results.json").write_text(json.dumps(rendered, indent=2) + "\n")
    lines = [
        "# Temporal robustness study", "",
        "Locked protocol: 19 training videos, 9 held-out videos, seed 0, 200-tree Random Forest, identical labels/preprocessing, and a 0.5 decision threshold.", "",
        "The matrix includes no-overlap (S=W), 50%-overlap (S=W/2), and 75%-overlap (S=W/4) conditions. Dense S=1 is intentionally excluded: it is appropriate only for short clips when compute permits and would not be a comparable full-corpus condition here.", "",
        "| Encoder | Window | Stride | F1 | Precision | Recall | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rendered:
        lines.append(f"| {row['encoder']} | {row['requested_window']} | {row['requested_stride']} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['roc_auc']:.4f} |")
    lines += ["", "## Actual encoder configurations and runtimes", ""]
    for row in rendered:
        lines.append(
            f"- {row['encoder']} requested W{row['requested_window']}/S{row['requested_stride']} → actual W{row['actual_window']}/S{row['actual_stride']}; "
            f"dim {row['feature_dimensionality']}; extraction {row['feature_extraction_seconds']:.1f}s over {row['total_windows']} windows; "
            f"RF {row['rf_training_seconds']:.3f}s. {row['note']}"
        )
    native_videomae = [
        row for row in rendered
        if row["encoder"] == "VideoMAE"
        and row["requested_window"] == row["actual_window"]
        and row["requested_stride"] == row["actual_stride"]
    ]
    vjepa = [row for row in rendered if row["encoder"] == "V-JEPA"]
    flow = next(row for row in rendered if row["encoder"] == "Optical Flow")
    best_videomae = max(native_videomae, key=lambda row: row["f1"])
    best_vjepa = max(vjepa, key=lambda row: row["f1"])
    vjepa_by_window = {
        window: [row for row in vjepa if row["requested_window"] == window]
        for window in sorted({row["requested_window"] for row in vjepa})
    }
    vjepa_window_best = {
        window: max(rows, key=lambda row: row["f1"])
        for window, rows in vjepa_by_window.items()
    }
    vjepa_overlap_ranges = {
        window: max(row["f1"] for row in rows) - min(row["f1"] for row in rows)
        for window, rows in vjepa_by_window.items()
    }
    mae_range = max(row["f1"] for row in native_videomae) - min(row["f1"] for row in native_videomae)
    latent_best = max(best_videomae, best_vjepa, key=lambda row: row["f1"])
    lines += ["", "## Best configurations", ""]
    lines.append(
        f"- Best native VideoMAE: W{best_videomae['requested_window']}/S{best_videomae['requested_stride']} "
        f"(F1 {best_videomae['f1']:.4f}, AUC {best_videomae['roc_auc']:.4f})."
    )
    lines.append(
        f"- Best V-JEPA: W{best_vjepa['requested_window']}/S{best_vjepa['requested_stride']} "
        f"(F1 {best_vjepa['f1']:.4f}, AUC {best_vjepa['roc_auc']:.4f})."
    )
    lines.append(
        f"- Optical-flow baseline: W32/S32 (F1 {flow['f1']:.4f}, AUC {flow['roc_auc']:.4f})."
    )
    lines += ["", "## Scientific analysis", ""]
    lines.append(
        "1. **Does increasing temporal context improve VideoMAE?** This checkpoint only natively accepts 16 frames; "
        "therefore this study cannot make a native W16-versus-W32/W64 VideoMAE context claim. Mapped W32/W64 rows are reported for protocol transparency, not interpreted as longer-context VideoMAE."
    )
    context_summary = "; ".join(
        f"W{window}: best F1 {row['f1']:.4f} at S{row['requested_stride']}"
        for window, row in vjepa_window_best.items()
    )
    lines.append(
        "2. **Does increasing temporal context improve V-JEPA?** The observed best-per-window results are "
        + context_summary + ". These fixed-split results describe this dataset only."
    )
    overlap_summary = "; ".join(
        f"V-JEPA W{window} range {value:.4f}" for window, value in vjepa_overlap_ranges.items()
    )
    lines.append(
        "3. **Does overlap materially affect performance?** Within a fixed V-JEPA window, the F1 ranges across no, 50%, and 75% overlap are "
        + overlap_summary + f". Native VideoMAE W16 spans {mae_range:.4f} F1 across S16, S8, and S4."
    )
    if latent_best["f1"] > flow["f1"]:
        lines.append(
            f"4. **Does any latent representation outperform optical flow?** Yes on this held-out split: {latent_best['encoder']} "
            f"W{latent_best['requested_window']}/S{latent_best['requested_stride']} has F1 {latent_best['f1']:.4f}, versus flow {flow['f1']:.4f}."
        )
        lines.append("5. **Is flow superiority robust?** No: at least one tested frozen latent configuration exceeded flow on held-out F1 in this study.")
    else:
        lines.append(
            f"4. **Does any latent representation outperform optical flow?** No: the best latent F1 is {latent_best['f1']:.4f}, "
            f"below the optical-flow F1 of {flow['f1']:.4f}."
        )
        lines.append(
            "5. **Is flow superiority robust?** Across these representative temporal window and overlap configurations, optical flow consistently outperformed frozen latent video representations for egocentric engagement estimation on this dataset. This does not claim generalization to other datasets or encoders."
        )
    (ROOT / "robustness_results.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "robustness_results.md")


if __name__ == "__main__":
    main()
