#!/usr/bin/env bash
# Stage the exact all-video cache corpus once on the pod.  This is resumable
# and intentionally bandwidth-capped so it can run beside GPU extraction.
set -euo pipefail

SRC="/Volumes/Crucial X9"
POD="root@69.30.85.137"
PORT="22096"
KEY="/Users/jleto/.ssh/id_ed25519"
REMOTE="/workspace/rational-memory-formation/full_cache_source"
LOG="/Volumes/Crucial X9/theory-of-mind/robustness_study/full_source_stage.log"
MANIFEST="$(mktemp -t theory_full_source_manifest.XXXXXX)"
trap 'rm -f "$MANIFEST"' EXIT

python3 - "$SRC" > "$MANIFEST" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
data = root / "theory-of-mind"
quicktime = root / "ego4d_download" / "quicktime"
videos = [
    path for path in data.glob("*.mp4")
    if not path.name.startswith("._")
    and (re.fullmatch(r"[0-9a-f]{16}\.mp4", path.name) or path.name == "egocentric_video.mp4")
]
videos.extend(
    path for path in (data / "new_qanda_downloads").rglob("*.mp4")
    if not path.name.startswith("._")
)
videos.extend(
    path for path in quicktime.glob("*_qt.mp4")
    if not path.name.startswith("._")
)
for path in sorted(videos):
    print(path.relative_to(root).as_posix())
PY

[[ "$(wc -l < "$MANIFEST" | tr -d ' ')" == "716" ]]
echo "[$(date '+%F %T')] staging 716 source videos to pod" | tee -a "$LOG"
ssh -i "$KEY" -p "$PORT" "$POD" "mkdir -p '$REMOTE'"
# macOS ships an older rsync; ``--partial`` is the portable resumable mode.
rsync -rlpt --no-owner --no-group --partial --bwlimit=20000 --files-from="$MANIFEST" \
  -e "ssh -i $KEY -p $PORT" "$SRC/" "$POD:$REMOTE/" \
  2>&1 | tee -a "$LOG"
ssh -i "$KEY" -p "$PORT" "$POD" "touch '$REMOTE/STAGED_COMPLETE'"
echo "[$(date '+%F %T')] full source staging complete" | tee -a "$LOG"
