#!/usr/bin/env bash
# Run the controlled W128/S64 V-JEPA2 cache + RF probe on the Runpod pod.
set -euo pipefail

export HF_HOME=/workspace/hf_cache
PY=/workspace/venv_jepa/bin/python
REPO=/workspace/rational-memory-formation
DATA="$REPO/robust_study_source/theory-of-mind"
CACHE="$REPO/robust_vjepa2_w128_s64.out"
RESULT="$REPO/robustness_results/vjepa2_w128_s64_rf"
TRAIN_LIST="$REPO/robustness_results/locked_train_videos.txt"
HELDOUT_LIST="$REPO/robustness_results/locked_heldout_videos.txt"
WINDOW_LIST="$REPO/robust_vjepa2_w128_s64.list"

cat "$TRAIN_LIST" "$HELDOUT_LIST" | sed 's#^#theory-of-mind/#' > "$WINDOW_LIST"

"$PY" "$REPO/cache_video_batch.py" \
  --input-root "$REPO/robust_study_source" \
  --list-file "$WINDOW_LIST" \
  --cache-dir "$CACHE" \
  --encoder vjepa2 --window-frames 128 --stride-frames 64 --feature-batch-size 2

INCLUDES=()
while IFS= read -r video; do
  INCLUDES+=(--include-video "$video")
done < <(cat "$TRAIN_LIST" "$HELDOUT_LIST")

"$PY" "$REPO/train_vjepa_probes.py" \
  --data-dir "$DATA" --output-dir "$RESULT" \
  --encoder vjepa2 --window-frames 128 --stride-frames 64 \
  --feature-cache-dir "$CACHE" --cached-only --probe-types rf \
  --rf-n-estimators 200 --rf-n-jobs -1 --seed 0 --device cpu \
  --positive-label goal_directed_activity,positive \
  --fixed-train-videos "$TRAIN_LIST" --fixed-val-videos "$HELDOUT_LIST" \
  "${INCLUDES[@]}"
