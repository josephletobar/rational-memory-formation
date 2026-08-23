#!/usr/bin/env bash
# Local completion watcher for the active DINOv3 last-three-minute run.
set -euo pipefail
KEY=/Users/jleto/.ssh/id_ed25519
HOST=root@69.30.85.137
PORT=22096
REMOTE=/tmp/dino_patch_work
ROOT='/Volumes/Crucial X9/theory-of-mind'
CACHE="$ROOT/dino_patch_cache"
RESULTS="$ROOT/dino_patch_surprise"
mkdir -p "$CACHE" "$RESULTS"
exists() { ssh -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p "$PORT" "$HOST" "test -f '$1'"; }
pull() { rsync -ah --partial -e "ssh -i $KEY -p $PORT" "$HOST:$1" "$2/"; }
VIDEO="$REMOTE/output/09ea_dinov3_384_last3m_patch_surprise.mp4"
until exists "$VIDEO"; do sleep 30; done
for f in \
  "$REMOTE/cache/09ea_dinov3_384_last3m_with_context.patches.fp16.npy" \
  "$REMOTE/cache/09ea_dinov3_384_last3m_with_context.dis_flow.fp16.npy" \
  "$REMOTE/cache/09ea_dinov3_384_last3m_with_context.nll_maps.fp16.npy"; do
  pull "$f" "$CACHE"
done
pull "$VIDEO" "$RESULTS"
pull "${VIDEO%.mp4}.json" "$RESULTS"
python3 - "$CACHE" "$RESULTS" <<'PY'
from pathlib import Path
import sys
for root in map(Path, sys.argv[1:]):
    for path in root.glob('09ea_dinov3_384_last3m*'):
        if path.stat().st_size == 0:
            raise SystemExit(f'empty copy: {path}')
print('DINOv3 last-3m copy verified')
PY
