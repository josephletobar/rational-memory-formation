#!/usr/bin/env bash
# Cache one controlled robustness-study configuration while reusing an
# immutable, pod-local copy of the locked 28-video study set.  It preserves
# the exact extractor, cache names, and local outputs used by the regular
# helper; it only removes redundant X9 -> pod transfers between configurations.
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
REMOTE_SOURCE="$REMOTE_REPO/robust_study_source"
TAG="robust_${ENCODER}_w${WINDOW}_s${STRIDE}"
# The network volume has an opaque per-directory write quota on this pod even
# when df reports ample capacity.  Extracted feature batches are transient and
# are immediately copied back to the X9, so keep them on the container's local
# disk instead.  Inputs and durable source staging remain on /workspace.
REMOTE_TEMP_ROOT="/tmp/rmf_feature_cache"
REMOTE_OUTPUT="$REMOTE_TEMP_ROOT/${TAG}.out"
REMOTE_LIST="$REMOTE_TEMP_ROOT/${TAG}.list"
BATCH_SIZE=2

case "$ENCODER" in
  vjepa2) SLUG="facebook-vjepa2-vitl-fpc32-256-diving48"; REMOTE_PY="/workspace/venvs/vjepa/bin/python"; FEATURE_BATCH=2; [[ "$WINDOW" == 64 ]] && FEATURE_BATCH=1 ;;
  videomae) SLUG="MCG-NJU-videomae-base"; REMOTE_PY="python3"; FEATURE_BATCH=8 ;;
  optical_flow) SLUG="optflow.h16x12.lvl3.ts2"; REMOTE_PY="python3"; FEATURE_BATCH=2; BATCH_SIZE=4 ;;
  *) echo "unknown encoder: $ENCODER" >&2; exit 2 ;;
esac

mkdir -p "$LOCAL_CACHE_DIR"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${TAG}.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
MANIFEST="$WORK_DIR/needed.txt"
ALL_MANIFEST="$WORK_DIR/locked_split.txt"

python3 - "$SPLIT_METRICS" "$LOCAL_DATA_ROOT" "$LOCAL_CACHE_DIR" "$SLUG" "$WINDOW" "$STRIDE" "$LOCAL_SOURCE_ROOT" "$MANIFEST" "$ALL_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
metrics, video_root, cache_dir = map(Path, sys.argv[1:4])
slug, window, stride = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
source_root, needed_file, all_file = map(Path, sys.argv[7:10])
data = json.loads(metrics.read_text())
videos = data["train_videos"] + data["val_videos"]
if len(videos) != 28 or len(set(videos)) != 28:
    raise SystemExit("locked study split must contain exactly 28 distinct videos")
all_lines, needed_lines = [], []
for name in videos:
    video = video_root / name
    cache = cache_dir / f"{video.stem}.{slug}.w{window}.s{stride}.full.npz"
    if not video.exists():
        raise SystemExit(f"missing locked-split video: {video}")
    relative = str(video.relative_to(source_root))
    all_lines.append(relative)
    if not cache.exists():
        needed_lines.append(relative)
all_file.write_text("\n".join(all_lines) + "\n")
needed_file.write_text("\n".join(needed_lines) + ("\n" if needed_lines else ""))
PY

TOTAL="$(wc -l < "$MANIFEST" | tr -d ' ')"
if [[ "$TOTAL" == 0 ]]; then echo "$TAG: all 28 caches already present"; exit 0; fi
echo "$TAG: caching $TOTAL missing locked-split videos (reusing pod study source)"
rsync -rlptz -e "ssh -i $POD_KEY -p $POD_PORT" cache_video_batch.py train_vjepa_probes.py "$POD:$REMOTE_REPO/"
ssh -i "$POD_KEY" -p "$POD_PORT" "$POD" "mkdir -p '$REMOTE_SOURCE' '$REMOTE_OUTPUT' '$REMOTE_TEMP_ROOT'"
# --ignore-existing makes this idempotent and never changes a previously
# staged input; standard rsync publishes a complete file atomically.
rsync -rlptz --ignore-existing --files-from="$ALL_MANIFEST" -e "ssh -i $POD_KEY -p $POD_PORT" "$LOCAL_SOURCE_ROOT/" "$POD:$REMOTE_SOURCE/"
split -l "$BATCH_SIZE" -d -a 3 "$MANIFEST" "$WORK_DIR/batch_"
INDEX=0
for batch in "$WORK_DIR"/batch_*; do
  [[ -f "$batch" ]] || continue
  INDEX=$((INDEX + 1)); COUNT="$(wc -l < "$batch" | tr -d ' ')"
  echo "$TAG: batch $INDEX ($(head -n 1 "$batch"))"
  scp -i "$POD_KEY" -P "$POD_PORT" "$batch" "$POD:$REMOTE_LIST"
  ssh -i "$POD_KEY" -p "$POD_PORT" "$POD" "cd '$REMOTE_REPO' && $REMOTE_PY cache_video_batch.py --input-root '$REMOTE_SOURCE' --list-file '$REMOTE_LIST' --cache-dir '$REMOTE_OUTPUT' --encoder '$ENCODER' --window-frames '$WINDOW' --stride-frames '$STRIDE' --feature-batch-size '$FEATURE_BATCH'"
  rsync -rlptz -e "ssh -i $POD_KEY -p $POD_PORT" "$POD:$REMOTE_OUTPUT/" "$LOCAL_CACHE_DIR/"
  ssh -i "$POD_KEY" -p "$POD_PORT" "$POD" "rm -rf '$REMOTE_OUTPUT' '$REMOTE_LIST'; mkdir -p '$REMOTE_OUTPUT'"
  echo "$TAG: batch $INDEX complete ($COUNT videos pulled)"
done
echo "$TAG: complete"
