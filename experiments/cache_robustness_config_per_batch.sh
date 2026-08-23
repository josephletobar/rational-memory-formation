#!/usr/bin/env bash
# Cache one controlled robustness-study configuration on the Runpod A40.
# Completed files are pulled after each batch; rerunning resumes from the X9.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 {optical_flow|videomae|vjepa2} WINDOW STRIDE LOCAL_CACHE_DIR" >&2
  exit 2
fi
ENCODER="$1"; WINDOW="$2"; STRIDE="$3"; LOCAL_CACHE_DIR="$4"
LOCAL_SOURCE_ROOT="/Volumes/Crucial X9"
LOCAL_DATA_ROOT="$LOCAL_SOURCE_ROOT/theory-of-mind"
SPLIT_METRICS="$LOCAL_DATA_ROOT/trained_models/optical_flow_28cached_w32s32_rf/metrics.json"
POD="root@69.30.85.137"; POD_PORT="22096"; POD_KEY="$HOME/.ssh/id_ed25519"
REMOTE_REPO="/workspace/rational-memory-formation"
TAG="robust_${ENCODER}_w${WINDOW}_s${STRIDE}"
REMOTE_STAGE="$REMOTE_REPO/${TAG}.stage"
REMOTE_OUTPUT="$REMOTE_REPO/${TAG}.out"
REMOTE_LIST="$REMOTE_REPO/${TAG}.list"
BATCH_SIZE=2

case "$ENCODER" in
  vjepa2) SLUG="facebook-vjepa2-vitl-fpc32-256-diving48"; REMOTE_PY="/workspace/venvs/vjepa/bin/python"; FEATURE_BATCH=2; [[ "$WINDOW" == 64 ]] && FEATURE_BATCH=1 ;;
  videomae) SLUG="MCG-NJU-videomae-base"; REMOTE_PY="python3"; FEATURE_BATCH=8 ;;
  optical_flow) SLUG="optflow.h16x12.lvl3.ts2"; REMOTE_PY="python3"; FEATURE_BATCH=2 ;;
  *) echo "unknown encoder: $ENCODER" >&2; exit 2 ;;
esac

mkdir -p "$LOCAL_CACHE_DIR"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${TAG}.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
MANIFEST="$WORK_DIR/needed.txt"

python3 - "$SPLIT_METRICS" "$LOCAL_DATA_ROOT" "$LOCAL_CACHE_DIR" "$SLUG" "$WINDOW" "$STRIDE" "$LOCAL_SOURCE_ROOT" > "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
metrics = Path(sys.argv[1]); video_root = Path(sys.argv[2]); cache_dir = Path(sys.argv[3])
slug = sys.argv[4]; window = int(sys.argv[5]); stride = int(sys.argv[6]); source_root = Path(sys.argv[7])
data = json.loads(metrics.read_text()); videos = data["train_videos"] + data["val_videos"]
if len(videos) != 28 or len(set(videos)) != 28:
    raise SystemExit("locked study split must contain exactly 28 distinct videos")
for name in videos:
    video = video_root / name
    cache = cache_dir / f"{video.stem}.{slug}.w{window}.s{stride}.full.npz"
    if not video.exists(): raise SystemExit(f"missing locked-split video: {video}")
    if not cache.exists(): print(video.relative_to(source_root))
PY

TOTAL="$(wc -l < "$MANIFEST" | tr -d ' ')"
if [[ "$TOTAL" == 0 ]]; then echo "$TAG: all 28 caches already present"; exit 0; fi
echo "$TAG: caching $TOTAL missing locked-split videos"
rsync -rlptz -e "ssh -i $POD_KEY -p $POD_PORT" cache_video_batch.py train_vjepa_probes.py "$POD:$REMOTE_REPO/"
split -l "$BATCH_SIZE" -d -a 3 "$MANIFEST" "$WORK_DIR/batch_"
INDEX=0
for batch in "$WORK_DIR"/batch_*; do
  [[ -f "$batch" ]] || continue
  INDEX=$((INDEX + 1)); COUNT="$(wc -l < "$batch" | tr -d ' ')"
  echo "$TAG: batch $INDEX ($(head -n 1 "$batch"))"
  ssh -i "$POD_KEY" -p "$POD_PORT" "$POD" "mkdir -p '$REMOTE_STAGE' '$REMOTE_OUTPUT'"
  rsync -rlptz --files-from="$batch" -e "ssh -i $POD_KEY -p $POD_PORT" "$LOCAL_SOURCE_ROOT/" "$POD:$REMOTE_STAGE/"
  scp -i "$POD_KEY" -P "$POD_PORT" "$batch" "$POD:$REMOTE_LIST"
  ssh -i "$POD_KEY" -p "$POD_PORT" "$POD" "cd '$REMOTE_REPO' && $REMOTE_PY cache_video_batch.py --input-root '$REMOTE_STAGE' --list-file '$REMOTE_LIST' --cache-dir '$REMOTE_OUTPUT' --encoder '$ENCODER' --window-frames '$WINDOW' --stride-frames '$STRIDE' --feature-batch-size '$FEATURE_BATCH'"
  rsync -rlptz -e "ssh -i $POD_KEY -p $POD_PORT" "$POD:$REMOTE_OUTPUT/" "$LOCAL_CACHE_DIR/"
  ssh -i "$POD_KEY" -p "$POD_PORT" "$POD" "rm -rf '$REMOTE_STAGE' '$REMOTE_OUTPUT' '$REMOTE_LIST'"
  echo "$TAG: batch $INDEX complete ($COUNT videos pulled)"
done
echo "$TAG: complete"
