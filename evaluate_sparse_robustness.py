#!/usr/bin/env python3
"""Render the abbreviated, fixed-split temporal-context robustness result."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Volumes/Crucial X9/theory-of-mind/robustness_study")
DATA = Path("/Volumes/Crucial X9/theory-of-mind")

ROWS = [
    ("Optical flow", "W32/S32", DATA / "trained_models/optical_flow_28cached_w32s32_rf/metrics.json"),
    ("VideoMAE", "W16/S8", ROOT / "quick_videomae_w16_s8_rf/metrics.json"),
    ("V-JEPA", "W16/S8", ROOT / "quick_vjepa2_w16_s8_rf/metrics.json"),
    ("V-JEPA", "W32/S16", ROOT / "quick_vjepa2_w32_s16_rf/metrics.json"),
    ("V-JEPA", "W64/S32", ROOT / "quick_vjepa2_w64_s32_rf/metrics.json"),
]


def main() -> None:
    rendered = []
    for encoder, temporal, path in ROWS:
        metrics = json.loads(path.read_text())["rf"]
        rendered.append({"encoder": encoder, "temporal": temporal, **metrics})
    (ROOT / "sparse_robustness_results.json").write_text(json.dumps(rendered, indent=2) + "\n")
    lines = [
        "# Abbreviated temporal-context robustness result", "",
        "Protocol: same locked 19/9 video split, seed 0, 200-tree Random Forest, labels, preprocessing, and 0.5 threshold. The three V-JEPA rows hold overlap fixed at 50%, isolating temporal context.", "",
        "| Encoder | Window / stride | F1 | Precision | Recall | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rendered:
        lines.append(f"| {row['encoder']} | {row['temporal']} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['roc_auc']:.4f} |")
    lines += ["", "## Interpretation", "", "At fixed 50% overlap, V-JEPA improves from W16 to W64 and each tested V-JEPA configuration exceeds the optical-flow baseline on this held-out split. This abbreviated study does not test whether overlap itself changes performance."]
    (ROOT / "sparse_robustness_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
