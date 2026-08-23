#!/usr/bin/env bash
# Copy atomically published cache files from the pod back to the Crucial X9
# while a multi-video extraction batch is still running.  The normal batch
# helper performs a final sync too; this is a durability layer for the pod's
# known instability, not an alternative cache writer.
set -euo pipefail

POD_HOST="root@69.30.85.137"
POD_PORT="22096"
POD_KEY="/Users/jleto/.ssh/id_ed25519"
REMOTE_ROOT="/tmp/rmf_feature_cache"
LOCAL_ROOT="/Volumes/Crucial X9/theory-of-mind/pod_cache_backup_current"
STATUS_ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"

sync_one() {
  local tag="$1" dest="$2"
  mkdir -p "$dest"
  # Cache archives are saved through an atomic rename, so --ignore-existing
  # cannot copy a partially written feature NPZ.
  rsync -a --ignore-existing --no-owner --no-group \
    -e "ssh -p $POD_PORT -i $POD_KEY" \
    "$POD_HOST:$REMOTE_ROOT/cache_out_$tag/" "$dest/" || true
}

while [[ ! -f "$STATUS_ROOT/ALL_LOCAL_CACHES_VERIFIED" ]]; do
  sync_one "vjepa2" "$LOCAL_ROOT/results_vjepa2_cache_stride16/feature_cache"
  sync_one "videomae_parallel" "$LOCAL_ROOT/results_videomae_cache_stride16/feature_cache"
  sync_one "flow_parallel" "$LOCAL_ROOT/results_optical_flow_cache/feature_cache"
  sleep 90
done
