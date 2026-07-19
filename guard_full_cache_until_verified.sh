#!/usr/bin/env bash
# Last-resort durable supervisor for the all-video cache objective.  The
# encoder helpers resume from local archives, so restarting the corrected
# runner after a transient pod/transfer failure never recomputes valid work.
set -euo pipefail

ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
cd "/Users/jleto/LocalProjects/theory-of-mind"

while [[ ! -f "$ROOT/ALL_LOCAL_CACHES_VERIFIED" ]]; do
  # Never overlap a legacy run or the corrected finalizer it hands off to.
  if pgrep -f 'run_full_cache_after_robustness\.sh|finalize_full_cache_after_legacy_runner\.sh' >/dev/null; then
    sleep 90
    continue
  fi
  ./run_full_cache_after_robustness.sh >> "$ROOT/full_cache_guard.log" 2>&1 || true
  sleep 90
done
