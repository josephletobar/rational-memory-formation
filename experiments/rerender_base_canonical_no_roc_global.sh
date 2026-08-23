#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/dino_patch_work
# Shared range across all three full-video, motion-normalized base-score caches.
# It spans the combined 99th percentile (~2.73) with a small amount of headroom.
GLOBAL_LO=0
GLOBAL_HI=3

for stem in 09ea3872eb883ec1 0f5a78b48827083d a65dec1048bd5e15; do
  base="$ROOT/cache/canonical/$stem"
  raw="$ROOT/output/canonical/${stem}_canonical_no_roc_global.raw.mp4"
  temp="$ROOT/output/canonical/${stem}_canonical_no_roc_global.mp4"
  final="$ROOT/output/canonical/${stem}_canonical.mp4"
  rm -f "$raw" "$temp"
  python3 "$ROOT/code/render_24grid_no_roc.py" "$ROOT/input/$stem.mp4" \
    "${base}_raw_nll.fp16.npy" "${base}_dis_flow.fp16.npy" "$raw" \
    --history 300 --alpha 2 --display-ema .25 --global-lo "$GLOBAL_LO" --global-hi "$GLOBAL_HI"
  ffmpeg -y -i "$raw" -vf scale=640:-2 -c:v h264_nvenc -preset p4 -rc vbr -b:v 4M -maxrate 5M -an "$temp"
  mv -f "$temp" "$final"
  rm -f "$raw"
  echo "rerendered base $stem"
done
