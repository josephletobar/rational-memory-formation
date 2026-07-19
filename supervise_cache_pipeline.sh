#!/usr/bin/env bash
# Keep the long cache queue recoverable overnight.  Each worker rebuilds its
# missing manifest from the X9 cache, so restarting only ever processes items
# that did not make it safely back to local storage.
set -euo pipefail

REPO="/Users/jleto/LocalProjects/theory-of-mind"
DATA="/Volumes/Crucial X9/theory-of-mind"
# Any active cache queue, including the separately tagged parallel shards,
# must finish before a recovery pass derives a new manifest.
PARENT_PATTERN="cache_missing_on_pod.sh (videomae|vjepa2|optical_flow)"
remaining_count() {
  python3 - "$DATA" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
videos = [
    p for p in root.glob("*.mp4")
    if not p.name.startswith("._")
    and (re.fullmatch(r"[0-9a-f]{16}\.mp4", p.name) or p.name == "egocentric_video.mp4")
]
videos += [p for p in (root / "new_qanda_downloads").glob("*.mp4") if not p.name.startswith("._")]
markers = (".MCG-NJU-videomae-base.", ".facebook-vjepa2-", ".optflow.")
total = 0
for marker in markers:
    cached = {p.name.split(marker, 1)[0] for p in root.rglob("*.npz") if marker in p.name}
    total += sum(p.stem not in cached for p in videos)
print(total)
PY
}

while true; do
  if pgrep -f "$PARENT_PATTERN" >/dev/null; then
    sleep 60
    continue
  fi

  remaining="$(remaining_count)"
  if [[ "$remaining" == "0" ]]; then
    exit 0
  fi

  # The prior screen may have been removed after a failed child.  Use a fresh
  # name for the restart and let each pass skip the now-local completed caches.
  screen -dmS "theory_cache_recovery_$(date +%s)" bash -lc \
    "cd '$REPO' && bash cache_missing_on_pod.sh videomae && bash cache_missing_on_pod.sh vjepa2 && bash cache_missing_on_pod.sh optical_flow"
  sleep 90
done
