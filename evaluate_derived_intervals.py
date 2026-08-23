#!/usr/bin/env python3
"""Evaluate causal V-JEPA engagement/surprise scores against derived QA intervals.

Scores are emitted at the end of each W64/S32 window and held until the next
completed window.  This intentionally uses no future interpolation.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

FUSION_WEIGHTS = (0.2842623927566942, 0.43147521448661164, 0.2842623927566942)
FUSION_ENGAGEMENT_EXPONENT = 0.46908670308515854
FUSION_SURPRISE_EXPONENT = 2.1464652623796985


def percentile_score(values):
    """Map finite scores to stable empirical percentile ranks in (0, 1)."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.zeros(len(values), dtype=np.float64)
    finite_values = values[finite]
    if not len(finite_values):
        return result
    order = np.argsort(finite_values, kind="stable")
    ranks = np.empty(len(finite_values), dtype=np.float64)
    ranks[order] = (np.arange(len(finite_values)) + 0.5) / len(finite_values)
    result[finite] = ranks
    return result


def targets_for_video(path: Path, video_uid: str):
    data = json.loads(path.read_text())
    targets = []
    for video in data.get("videos", []):
        if video.get("video_uid") != video_uid:
            continue
        for clip in video.get("clips", []):
            for ann in clip.get("annotations", []):
                for query in ann.get("language_queries", []):
                    targets.append((
                        float(query["video_start_sec"]),
                        float(query["video_end_sec"]),
                        query.get("query", ""),
                    ))
    return targets


def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def overlap_length(interval, regions):
    start, end = interval
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in regions)


def score_method(name, scores, ends, video_duration, targets, budgets):
    stride = float(np.median(np.diff(ends))) if len(ends) > 1 else 0.0
    segments = [(float(end), min(video_duration, float(end) + stride)) for end in ends]
    rows = []
    for budget in budgets:
        count = max(1, int(np.ceil(len(scores) * budget / 100.0)))
        chosen = np.argpartition(scores, -count)[-count:]
        regions = merge([segments[i] for i in chosen])
        retained = sum(b - a for a, b in regions)
        hits = sum(overlap_length((a, b), regions) > 0 for a, b, _ in targets)
        target_total = sum(b - a for a, b, _ in targets)
        target_covered = sum(overlap_length((a, b), regions) for a, b, _ in targets)
        rows.append({
            "method": name,
            "budget_percent": budget,
            "selected_windows": count,
            "retained_seconds": retained,
            "retained_percent": 100.0 * retained / video_duration,
            "intervals_captured": hits,
            "intervals_total": len(targets),
            "interval_recall": hits / len(targets) if targets else None,
            "target_seconds_covered": target_covered,
            "target_seconds_total": target_total,
            "target_time_recall": target_covered / target_total if target_total else None,
        })
    return rows


def score_selected(name, selected, ends, video_duration, targets, budget, **metadata):
    """Score an explicitly selected causal-window mask."""
    stride = float(np.median(np.diff(ends))) if len(ends) > 1 else 0.0
    segments = [(float(end), min(video_duration, float(end) + stride)) for end in ends]
    chosen = np.flatnonzero(selected)
    regions = merge([segments[i] for i in chosen])
    retained = sum(b - a for a, b in regions)
    hits = sum(overlap_length((a, b), regions) > 0 for a, b, _ in targets)
    target_total = sum(b - a for a, b, _ in targets)
    target_covered = sum(overlap_length((a, b), regions) for a, b, _ in targets)
    return {
        "method": name,
        "budget_percent": budget,
        "selected_windows": int(len(chosen)),
        "retained_seconds": retained,
        "retained_percent": 100.0 * retained / video_duration,
        "intervals_captured": hits,
        "intervals_total": len(targets),
        "interval_recall": hits / len(targets) if targets else None,
        "target_seconds_covered": target_covered,
        "target_seconds_total": target_total,
        "target_time_recall": target_covered / target_total if target_total else None,
        **metadata,
    }


def hard_onset_mask(surprise, engagement, tau_surprise, tau_engagement=0.5):
    """Surprise opens an episode; engagement above threshold sustains it."""
    selected = np.zeros(len(surprise), dtype=bool)
    memory_open = False
    for i, (s_value, e_value) in enumerate(zip(surprise, engagement)):
        onset = s_value > tau_surprise
        memory_open = bool(onset or (memory_open and e_value > tau_engagement))
        selected[i] = memory_open
    return selected


def score_hard_onset_persistence(
        engagement, surprise, ends, video_duration, targets, budgets,
        tau_engagement=0.5):
    """Budget-match tau_S without consulting targets; tau_E remains fixed."""
    rows = []
    for budget in budgets:
        target_ratio = budget / 100.0
        low, high = 0.0, 1.0
        candidates = []
        for _ in range(32):
            tau_surprise = (low + high) / 2.0
            selected = hard_onset_mask(
                surprise, engagement, tau_surprise, tau_engagement)
            ratio = float(selected.mean())
            candidates.append((abs(ratio - target_ratio), tau_surprise, selected))
            if ratio > target_ratio:
                low = tau_surprise
            else:
                high = tau_surprise
        _, tau_surprise, selected = min(candidates, key=lambda item: item[0])
        rows.append(score_selected(
            "hard_onset_persistence",
            selected,
            ends,
            video_duration,
            targets,
            budget,
            tau_surprise=float(tau_surprise),
            tau_engagement=float(tau_engagement),
        ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("probe", type=Path)
    p.add_argument("normalization", type=Path)
    p.add_argument("intervals", type=Path)
    p.add_argument("video_uid")
    p.add_argument("output", type=Path)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--budgets", default="25,20,15,10")
    p.add_argument("--time-offset", type=float, default=0.0,
                   help="Source-video seconds at the start of a staged subclip.")
    args = p.parse_args()
    saved = np.load(args.cache)
    features = saved["features"].astype(np.float32)
    ends = saved["ends"].astype(np.float64)
    norm = np.load(args.normalization)
    x = (features - norm["mean"]) / norm["std"]
    engagement = joblib.load(args.probe).predict_proba(x)[:, 1].astype(np.float32)

    # Canonical N=2 causal diagonal-Gaussian surprise: fit the prior only to
    # the two preceding overlapping W64 embeddings, then score the current.
    surprise = np.zeros(len(x), dtype=np.float32)
    for i in range(2, len(x)):
        prior = x[i - 2:i]
        mean = prior.mean(axis=0)
        variance = prior.var(axis=0) + 1e-5
        surprise[i] = 0.5 * np.mean((x[i] - mean) ** 2 / variance)
    engagement_percentile = percentile_score(engagement)
    surprise_percentile = percentile_score(surprise)
    engagement_term = engagement_percentile ** FUSION_ENGAGEMENT_EXPONENT
    surprise_term = surprise_percentile ** FUSION_SURPRISE_EXPONENT
    w_engagement, w_surprise, w_interaction = FUSION_WEIGHTS
    old_fusion = np.clip(
        w_engagement * engagement_term
        + w_surprise * surprise_term
        + w_interaction * engagement_term * surprise_term,
        0.0,
        1.0,
    )
    # Stateful soft memory: surprise opens memory continuously and
    # engagement controls how much of the previous memory persists.
    soft_memory = np.zeros(len(engagement_percentile), dtype=np.float64)
    for i, (s_value, e_value) in enumerate(
            zip(surprise_percentile, engagement_percentile)):
        previous = soft_memory[i - 1] if i else 0.0
        soft_memory[i] = s_value + (1.0 - s_value) * e_value * previous
    targets = [
        (start - args.time_offset, end - args.time_offset, query)
        for start, end, query in targets_for_video(args.intervals, args.video_uid)
        if end > args.time_offset and start < args.time_offset + args.duration
    ]
    budgets = [float(v) for v in args.budgets.split(",")]
    rows = score_method("vjepa_engagement", engagement, ends, args.duration, targets, budgets)
    rows += score_method("vjepa_surprise_n2", surprise, ends, args.duration, targets, budgets)
    rows += score_method("old_canonical_fusion", old_fusion, ends, args.duration, targets, budgets)
    rows += score_hard_onset_persistence(
        engagement_percentile,
        surprise_percentile,
        ends,
        args.duration,
        targets,
        budgets,
    )
    rows += score_method(
        "soft_recurrent_fusion", soft_memory, ends, args.duration, targets, budgets)
    out = {
        "video_uid": args.video_uid,
        "window_frames": 64,
        "stride_frames": 32,
        "fps": 15,
        "surprise": "causal N=2 diagonal Gaussian; no future interpolation",
        "fusion": {
            "formula": "wE*E^0.469087 + wS*S^2.146465 + wES*E^0.469087*S^2.146465",
            "calibration": "stable empirical percentile ranks computed per video",
            "weights": {
                "engagement": w_engagement,
                "surprise": w_surprise,
                "interaction": w_interaction,
            },
        },
        "recurrent_fusions": {
            "hard": {
                "formula": "g_t=1[S_t>tau_S]; m_t=g_t+(1-g_t)m_(t-1)1[E_t>tau_E]",
                "tau_engagement": 0.5,
                "tau_surprise": "chosen per requested retention budget without using annotations",
                "patience_windows": 0,
            },
            "soft": {
                "formula": "M_t=S_t+(1-S_t)E_tM_(t-1)",
            },
            "calibration": "stable empirical percentile ranks computed per video",
        },
        "targets": [{"start_sec": a, "end_sec": b, "query": q} for a, b, q in targets],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for row in rows:
        print("{method:20s} {budget_percent:>4.0f}% kept={retained_percent:5.2f}% "
              "intervals={intervals_captured}/{intervals_total} "
              "time={target_time_recall:.3f}".format(**row))


if __name__ == "__main__":
    main()
