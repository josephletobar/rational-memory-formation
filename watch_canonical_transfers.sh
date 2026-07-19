#!/usr/bin/env bash
# Local watcher: pulls finished canonical artifacts from the pod to Crucial X9.
set -euo pipefail

HOST=root@69.30.85.137
PORT=22096
KEY="$HOME/.ssh/id_ed25519"
REMOTE=/tmp/dino_patch_work
DRIVE='/Volumes/Crucial X9/theory-of-mind'
mkdir -p "$DRIVE/dino_patch_surprise/canonical_full" "$DRIVE/dino_patch_cache/canonical_full"

for stem in 09ea3872eb883ec1 0f5a78b48827083d a65dec1048bd5e15; do
  until ssh -i "$KEY" -p "$PORT" "$HOST" "test -f $REMOTE/output/canonical/$stem.done"; do
    sleep 30
  done
  scp -P "$PORT" -i "$KEY" "$HOST:$REMOTE/output/canonical/${stem}_canonical.mp4" "$DRIVE/dino_patch_surprise/canonical_full/"
  scp -P "$PORT" -i "$KEY" "$HOST:$REMOTE/cache/canonical/${stem}_raw_nll.fp16.npy" "$DRIVE/dino_patch_cache/canonical_full/"
  scp -P "$PORT" -i "$KEY" "$HOST:$REMOTE/cache/canonical/${stem}_dis_flow.fp16.npy" "$DRIVE/dino_patch_cache/canonical_full/"
  scp -P "$PORT" -i "$KEY" "$HOST:$REMOTE/cache/canonical/${stem}_patch_roc.fp16.npy" "$DRIVE/dino_patch_cache/canonical_full/"
  touch "$DRIVE/dino_patch_surprise/canonical_full/${stem}.transferred"
done
