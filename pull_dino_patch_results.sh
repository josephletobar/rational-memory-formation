#!/usr/bin/env bash
# Runs locally and copies each durable pod result to the Crucial X9 as soon as
# it exists, so the pod's ephemeral container disk is never the only copy.
set -euo pipefail
KEY=/Users/jleto/.ssh/id_ed25519
HOST=root@69.30.85.137
PORT=22096
REMOTE=/tmp/dino_patch_work
ROOT='/Volumes/Crucial X9/theory-of-mind'
CACHE="$ROOT/dino_patch_cache"
RESULTS="$ROOT/dino_patch_surprise"
mkdir -p "$CACHE" "$RESULTS"

available() {
  ssh -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p "$PORT" "$HOST" "test -f '$1'"
}
pull() {
  local remote_path="$1" local_dir="$2"
  rsync -ah --partial -e "ssh -i $KEY -p $PORT" "$HOST:$remote_path" "$local_dir/"
}
for stem in 09ea3872eb883ec1 0f5a78b48827083d a65dec1048bd5e15; do
  patch="$REMOTE/cache/$stem.dinov2-base.patches16x16.fp16.npy"
  flow="$REMOTE/cache/$stem.dinov2-base.farneback16x16.forward_backward.fp16.npy"
  result="$REMOTE/output/${stem}_dino_patch_semantic_surprise_h300.mp4"
  until available "$patch" && available "$flow"; do sleep 45; done
  pull "$patch" "$CACHE"
  pull "${patch%.npy}.json" "$CACHE"
  pull "$flow" "$CACHE"
  pull "${flow%.npy}.json" "$CACHE"
  until available "$result"; do sleep 45; done
  pull "$result" "$RESULTS"
  pull "${result%.mp4}.json" "$RESULTS"
  pull "$REMOTE/cache/$stem.dinov2-base.patch_surprise_h300.fp16.npy" "$CACHE"
done
