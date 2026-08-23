#!/usr/bin/env bash
# Preserve the controlled study's GPU order, then resume the broader cache
# objective (main corpus + new_qanda_downloads + canonical QuickTime sources).
set -euo pipefail
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
LOG="$ROOT/full_cache_after_robustness.log"
while [[ ! -f "$ROOT/STUDY_COMPLETE" ]]; do
  sleep 60
done
cd "/Users/jleto/LocalProjects/theory-of-mind"

run_cache_with_retries() {
  local encoder="$1"
  local attempt
  for attempt in 1 2 3; do
    if bash cache_missing_on_pod.sh "$encoder"; then
      return 0
    fi
    if [[ "$attempt" -lt 3 ]]; then
      echo "[$(date '+%F %T')] $encoder cache pass failed (attempt $attempt/3); resuming in 60s"
      sleep 60
    fi
  done
  echo "[$(date '+%F %T')] $encoder cache pass failed after 3 resumable attempts" >&2
  return 1
}

{
  echo "[$(date '+%F %T')] controlled study metrics complete; starting all-video cache completion"
  # VideoMAE was deliberately removed from the active corpus-cache scope.
  # Each remaining helper resumes from durable local NPZ files, so a later
  # pass only handles genuinely missing or invalid caches.
  run_cache_with_retries vjepa2
  run_cache_with_retries optical_flow
  python3 verify_all_feature_caches.py --skip-videomae
  touch "$ROOT/ALL_LOCAL_CACHES_VERIFIED"
  echo "[$(date '+%F %T')] all-video cache completion verified"
} 2>&1 | tee -a "$LOG"
