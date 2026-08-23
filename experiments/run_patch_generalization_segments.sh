#!/usr/bin/env bash
set -euo pipefail

WORK=/tmp/dino_patch_work
for entry in \
  '09ea3872eb883ec1:245.633' \
  '0f5a78b48827083d:244.067' \
  'a65dec1048bd5e15:243.767'
do
  stem=${entry%%:*}
  start=${entry##*:}
  clip="$WORK/input/${stem}_middle2m_15fps.mp4"
  maps="$WORK/cache/canonical/${stem}_middle2m_dino_patch_w300_s1_raw_cosine.fp16.npy"
  raw="$WORK/output/canonical/${stem}_middle2m_dino_patch_w300_s1.raw.mp4"
  final="$WORK/output/canonical/${stem}_middle2m_dino_patch_w300_s1.mp4"
  rm -f "$clip" "$maps" "${maps%.npy}.partial.npy" "$raw" "$final"
  ffmpeg -y -ss "$start" -i "$WORK/input/${stem}.mp4" -t 120 -vf fps=15 -an -c:v h264_nvenc -preset p4 -cq 20 "$clip"
  python3 "$WORK/code/dino_patch_window_stride1.py" "$clip" "$maps" "$raw" --window 300 --batch-size 12
  ffmpeg -y -i "$raw" -vf scale=640:-2 -c:v h264_nvenc -preset p4 -rc vbr -b:v 4M -maxrate 5M -an "$final"
  rm -f "$raw" "$clip"
  echo "completed $stem"
done
