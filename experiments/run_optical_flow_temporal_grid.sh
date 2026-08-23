#!/usr/bin/env bash
# Controlled optical-flow-only temporal sampling study. Raw flow is cached
# once per locked-split video, then reused to materialize all W/S conditions.
set -euo pipefail

REPO="/Users/jleto/LocalProjects/theory-of-mind"
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
DATA="/Volumes/Crucial X9/theory-of-mind"
SPLIT="$DATA/trained_models/optical_flow_28cached_w32s32_rf/metrics.json"
POD="root@69.30.85.137"; PORT="22096"; KEY="$HOME/.ssh/id_ed25519"
REMOTE_REPO="/workspace/rational-memory-formation"
REMOTE_SOURCE="$REMOTE_REPO/robust_study_source"
REMOTE_ROOT="/tmp/rmf_feature_cache/optical_flow_temporal_grid"
REMOTE_RAW="$REMOTE_ROOT/raw"; REMOTE_LIST="$REMOTE_ROOT/locked_28.list"
LOCAL_RAW="$ROOT/raw_optical_flow_h16x12_ts2"
LOG="$ROOT/optical_flow_window_context/run.log"

mkdir -p "$ROOT/optical_flow_window_context" "$LOCAL_RAW"
MANIFEST="$(mktemp -t optical_flow_grid.XXXXXX)"
trap 'rm -f "$MANIFEST"' EXIT
python3 - "$SPLIT" "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
split = json.loads(Path(sys.argv[1]).read_text())
videos = split["train_videos"] + split["val_videos"]
if len(videos) != 28 or len(set(videos)) != 28:
    raise SystemExit("expected exactly the locked 19/9 split")
Path(sys.argv[2]).write_text("".join(f"theory-of-mind/{v}\n" for v in videos))
PY

cd "$REPO"
rsync -rlptz -e "ssh -i $KEY -p $PORT" cache_optical_flow_raw.py materialize_optical_flow_windows.py train_vjepa_probes.py "$POD:$REMOTE_REPO/"
ssh -i "$KEY" -p "$PORT" "$POD" "mkdir -p '$REMOTE_SOURCE' '$REMOTE_RAW' '$REMOTE_ROOT'"
rsync -rlptz --ignore-existing --files-from="$MANIFEST" -e "ssh -i $KEY -p $PORT" '/Volumes/Crucial X9/' "$POD:$REMOTE_SOURCE/"
scp -i "$KEY" -P "$PORT" "$MANIFEST" "$POD:$REMOTE_LIST"
ssh -i "$KEY" -p "$PORT" "$POD" "cd '$REMOTE_REPO' && python3 cache_optical_flow_raw.py --input-root '$REMOTE_SOURCE' --list-file '$REMOTE_LIST' --cache-dir '$REMOTE_RAW' --workers 4"
rsync -rlptz -e "ssh -i $KEY -p $PORT" "$POD:$REMOTE_RAW/" "$LOCAL_RAW/"

# Four windows × no overlap, 50% overlap, and 75% overlap.
for spec in "16 16" "16 8" "16 4" "32 32" "32 16" "32 8" "64 64" "64 32" "64 16" "128 128" "128 64" "128 32"; do
  set -- $spec; window="$1"; stride="$2"
  local_out="$ROOT/feature_cache/optical_flow/w${window}_s${stride}"
  remote_out="$REMOTE_ROOT/w${window}_s${stride}"
  mkdir -p "$local_out"
  # Preserve the existing canonical W32/S32 baseline verbatim.
  if [[ "$window" == "32" && "$stride" == "32" ]] && [[ "$(find "$local_out" -maxdepth 1 -type f -name '*.npz' ! -name '._*' | wc -l | tr -d ' ')" == "28" ]]; then
    echo "[$(date '+%F %T')] reused canonical W32/S32" >> "$LOG"; continue
  fi
  ssh -i "$KEY" -p "$PORT" "$POD" "rm -rf '$remote_out'; mkdir -p '$remote_out'"
  ssh -i "$KEY" -p "$PORT" "$POD" "cd '$REMOTE_REPO' && python3 materialize_optical_flow_windows.py --input-root '$REMOTE_SOURCE' --list-file '$REMOTE_LIST' --raw-cache-dir '$REMOTE_RAW' --output-dir '$remote_out' --window-frames '$window' --stride-frames '$stride' --workers 8"
  rsync -rlptz -e "ssh -i $KEY -p $PORT" "$POD:$remote_out/" "$local_out/"
  # The X9 copy is the durable feature cache; reclaim the pod's small /tmp
  # before materializing the next (potentially denser) condition.
  ssh -i "$KEY" -p "$PORT" "$POD" "rm -rf '$remote_out'"
  echo "[$(date '+%F %T')] materialized W${window}/S${stride}" >> "$LOG"
done

python3 evaluate_optical_flow_window_context.py >> "$LOG" 2>&1
echo "[$(date '+%F %T')] optical-flow temporal grid complete" >> "$LOG"
