#!/usr/bin/env python3
"""Verify that every source video has local VideoMAE, V-JEPA, and flow caches.

This uses the same deliberate source-corpus definition as cache_missing_on_pod.sh:
canonical root recordings, every original new_qanda download (recursively), and
the canonical QuickTime copies.  It is a completion gate, not a cache creator.
"""

from __future__ import annotations

import re
import sys
import zipfile
import argparse
from pathlib import Path


SOURCE_ROOT = Path("/Volumes/Crucial X9")
DATA_ROOT = SOURCE_ROOT / "theory-of-mind"
QUICKTIME_ROOT = SOURCE_ROOT / "ego4d_download" / "quicktime"


def source_videos() -> list[Path]:
    videos = [
        path
        for path in DATA_ROOT.glob("*.mp4")
        if not path.name.startswith("._")
        and (
            re.fullmatch(r"[0-9a-f]{16}\.mp4", path.name)
            or path.name == "egocentric_video.mp4"
        )
    ]
    videos.extend(
        path
        for path in (DATA_ROOT / "new_qanda_downloads").rglob("*.mp4")
        if not path.name.startswith("._")
    )
    videos.extend(
        path
        for path in QUICKTIME_ROOT.glob("*_qt.mp4")
        if not path.name.startswith("._")
    )
    return sorted(videos)


def valid_feature_cache(path: Path) -> bool:
    """Return whether ``path`` is an NPZ with the three required arrays.

    Extraction is published atomically, but this gate also protects against a
    damaged transfer or an unrelated nonempty file taking a cache filename.
    Inspecting the ZIP directory is intentionally lightweight: it does not
    load every full feature matrix into RAM during the inventory check.
    """
    if not path.is_file() or path.stat().st_size == 0 or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return {"features.npy", "starts.npy", "ends.npy"}.issubset(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def cached_stems(marker: str) -> set[str]:
    return {
        path.name.split(marker, 1)[0]
        for path in DATA_ROOT.rglob("*.npz")
        # macOS writes ``._*`` metadata sidecars on the external X9.  A
        # sidecar can survive after the actual archive is removed, so it must
        # never be accepted as evidence that a feature cache exists.
        if marker in path.name
        and not path.name.startswith("._")
        and valid_feature_cache(path)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    # Kept solely for backward-compatible invocation of an older runner.  The
    # all-video completion contract always requires VideoMAE, so it no longer
    # changes the result.
    parser.add_argument("--skip-videomae", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.skip_videomae:
        print("NOTE: --skip-videomae is ignored; VideoMAE is required for completion.")
    videos = source_videos()
    if len({video.stem for video in videos}) != len(videos):
        print("ERROR: source corpus contains duplicate video stems", file=sys.stderr)
        return 2

    checks = {
        "VideoMAE W16/S16": ".MCG-NJU-videomae-base.w16.s16.full.npz",
        "V-JEPA W32/S16": ".facebook-vjepa2-vitl-fpc32-256-diving48.w32.s16.full.npz",
        "optical flow W32/S32": ".optflow.h16x12.lvl3.ts2.w32.s32.full.npz",
    }
    failed = False
    print(f"Source videos: {len(videos)}")
    for label, marker in checks.items():
        cached = cached_stems(marker)
        missing = [video for video in videos if video.stem not in cached]
        print(f"{label}: {len(videos) - len(missing)}/{len(videos)} cached")
        if missing:
            failed = True
            for video in missing[:20]:
                print(f"  MISSING {video}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
