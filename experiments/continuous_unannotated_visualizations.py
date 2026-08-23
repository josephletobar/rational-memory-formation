#!/usr/bin/env python3
"""Render RF predictions for every newly cacheable, unannotated source video.

The queue only reads VideoMAE W16/S16 features already stored on the Crucial
X9.  It neither invokes an encoder nor recomputes features, so it can run in
parallel with the GPU cache jobs.  Completed MP4s and prediction arrays are
atomic outputs and are skipped on subsequent passes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


X9 = Path("/Volumes/Crucial X9")
DATA = X9 / "theory-of-mind"
QUICKTIME = X9 / "ego4d_download" / "quicktime"
PROBES = DATA / "trained_models" / "optical_flow_28cached_w32s32_rf"
CACHE = DATA / "pod_cache_backup_current" / "results_optical_flow_cache" / "feature_cache"
OUTPUT = DATA / "unannotated_predictions" / "optical_flow_w32s32_rf"
SLUG = "optflow.h16x12.lvl3.ts2"


def source_videos() -> list[Path]:
    videos = [
        p for p in DATA.glob("*.mp4")
        if not p.name.startswith("._")
        and (re.fullmatch(r"[0-9a-f]{16}\.mp4", p.name) or p.name == "egocentric_video.mp4")
    ]
    videos.extend(p for p in (DATA / "new_qanda_downloads").rglob("*.mp4") if not p.name.startswith("._"))
    videos.extend(p for p in QUICKTIME.glob("*_qt.mp4") if not p.name.startswith("._"))
    return sorted(videos)


def cache_for(video: Path) -> Path:
    return CACHE / f"{video.stem}.{SLUG}.w32.s32.full.npz"


def is_unannotated(video: Path) -> bool:
    return not video.with_name(video.name + ".clipme.json").exists()


def next_video() -> Path | None:
    for video in source_videos():
        if not is_unannotated(video) or not cache_for(video).exists():
            continue
        output = OUTPUT / f"{video.stem}_optical_flow_rf_predictions.mp4"
        if not output.exists() or output.stat().st_size == 0:
            return video
    return None


def output_bytes() -> int:
    """Account for completed and temporary renders before scheduling more work."""
    return sum(path.stat().st_size for path in OUTPUT.glob("*") if path.is_file())


def render(video: Path, max_width: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred = OUTPUT / f"{video.stem}_predictions.npz"
    render_video = OUTPUT / f"{video.stem}_optical_flow_rf_predictions.mp4"
    raw_video = OUTPUT / f"{video.stem}_optical_flow_rf_predictions.render_raw.mp4"
    command = [
        sys.executable, str(Path(__file__).with_name("predict_video.py")), str(video),
        "--probes", str(PROBES),
        "--encoder", "optical_flow", "--window-frames", "32", "--stride-frames", "32",
        "--feature-cache-dir", str(CACHE), "--cached-only", "--border-model", "rf",
        "--output", str(pred), "--render-video", str(raw_video),
        "--render-max-width", str(max_width), "--device", "cpu",
    ]
    print(f"Rendering {video}", flush=True)
    subprocess.run(command, check=True)
    compressed = render_video.with_name(f"{render_video.stem}.part{render_video.suffix}")
    subprocess.run([
        "avconvert", "--source", str(raw_video), "--preset", "Preset640x480",
        "--output", str(compressed), "--replace",
    ], check=True)
    compressed.replace(render_video)
    raw_video.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument(
        "--max-output-gib", type=float, default=5.0,
        help="Keep visualization artifacts bounded so feature-cache space is reserved.",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.max_width < 1 or args.max_output_gib <= 0:
        raise ValueError("poll-seconds, max-width, and max-output-gib must be positive")
    if not (PROBES / "rf_probe.joblib").exists():
        raise FileNotFoundError(f"Missing RF probe: {PROBES / 'rf_probe.joblib'}")
    while True:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if output_bytes() >= args.max_output_gib * 1024**3:
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
            continue
        video = next_video()
        if video is None:
            if args.once:
                return 0
            print("No cacheable unannotated videos yet; waiting", flush=True)
            time.sleep(args.poll_seconds)
            continue
        try:
            render(video, args.max_width)
        except Exception as exc:  # Keep later videos moving; retain failure evidence.
            print(f"FAILED {video}: {exc}", flush=True)
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
