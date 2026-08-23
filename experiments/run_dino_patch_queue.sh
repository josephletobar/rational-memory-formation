#!/usr/bin/env bash
# Runs on the pod's local container disk.  GPU work is intentionally serial;
# each video's Farneback cache runs on CPU in parallel with DINO token caching.
set -euo pipefail
cd /tmp/dino_patch_work
mkdir -p output

cache_one() {
  local stem="$1"
  local source="input/${stem}.mp4"
  local patches="cache/${stem}.dinov2-base.patches16x16.fp16.npy"
  local flow="cache/${stem}.dinov2-base.farneback16x16.forward_backward.fp16.npy"
  local global="input/${stem}_dino_standardized_distance_h300.npz"
  local result="output/${stem}_dino_patch_semantic_surprise_h300.mp4"
  local raw="output/${stem}_dino_patch_semantic_surprise_h300.raw.mp4"
  local map="cache/${stem}.dinov2-base.patch_surprise_h300.fp16.npy"

  # The first video may already be actively caching when this queue begins.
  # Wait for those detached workers instead of duplicating GPU work.
  while pgrep -f "[c]ache_dino_patch_tokens.py input/${stem}.mp4|[c]ache_dino_dense_flow.py input/${stem}.mp4" >/dev/null; do
    sleep 20
  done
  if [[ ! -f "$patches" || ! -f "$flow" ]]; then
    echo "[$(date -Is)] cache $stem"
    python3 code/cache_dino_patch_tokens.py "$source" "$patches" --batch-size 64 --device cuda > "cache/${stem}.patch.log" 2>&1 &
    local patch_pid=$!
    python3 code/cache_dino_dense_flow.py "$source" "$flow" --batch-size 64 > "cache/${stem}.flow.log" 2>&1 &
    local flow_pid=$!
    wait "$patch_pid"
    wait "$flow_pid"
  fi
  if [[ ! -f "$result" ]]; then
    echo "[$(date -Is)] render $stem"
    python3 code/render_patch_semantic_surprise.py "$source" --patches "$patches" --flow "$flow" --global-cache "$global" --output "$raw" --map-output "$map" --history 300 --max-width 640 --device cuda > "output/${stem}.render.log" 2>&1
    ffmpeg -y -loglevel error -i "$raw" -c:v h264_nvenc -preset p4 -rc vbr -b:v 4M -maxrate 5M -movflags +faststart "$result"
    rm -f "$raw"
  fi
  echo "[$(date -Is)] complete $stem"
}

cache_one 09ea3872eb883ec1
cache_one 0f5a78b48827083d
cache_one a65dec1048bd5e15
echo "[$(date -Is)] queue complete"
