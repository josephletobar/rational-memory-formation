#!/usr/bin/env bash
# Pod-side serial queue.  The first 09ea task is launched separately, then
# this waits for it and runs the other two full-video canonical passes.
set -euo pipefail

ROOT=/tmp/dino_patch_work
CODE="$ROOT/code/canonical_full_video.py"

wait_for_first() {
  while pgrep -f "[c]anonical_full_video.py $ROOT/input/09ea3872eb883ec1.mp4" >/dev/null; do
    sleep 30
  done
  test -s "$ROOT/output/canonical/09ea3872eb883ec1_canonical.mp4"
  touch "$ROOT/output/canonical/09ea3872eb883ec1.done"
}

run_one() {
  local stem="$1"
  python3 "$CODE" "$ROOT/input/$stem.mp4" \
    "$ROOT/cache/canonical/${stem}_raw_nll.fp16.npy" \
    "$ROOT/cache/canonical/${stem}_dis_flow.fp16.npy" \
    "$ROOT/cache/canonical/${stem}_patch_roc.fp16.npy" \
    "$ROOT/output/canonical/${stem}_canonical.raw.mp4" \
    "$ROOT/output/canonical/${stem}_canonical.mp4" \
    --history 300 --alpha 2 --display-ema .25 --batch-size 12 --device cuda \
    > "$ROOT/output/canonical/${stem}.log" 2>&1
  ffmpeg -y -i "$ROOT/output/canonical/${stem}_canonical.raw.mp4" \
    -vf scale=640:-2 -c:v h264_nvenc -preset p4 -rc vbr -b:v 4M -maxrate 5M -an \
    "$ROOT/output/canonical/${stem}_canonical.mp4" \
    >> "$ROOT/output/canonical/${stem}.log" 2>&1
  touch "$ROOT/output/canonical/${stem}.done"
}

wait_for_first
run_one 0f5a78b48827083d
run_one a65dec1048bd5e15
