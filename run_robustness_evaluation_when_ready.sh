#!/usr/bin/env bash
# Wait for both resumable cache sweeps, then run the fixed-split RF evaluation.
set -euo pipefail
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
LOG="$ROOT/evaluation.log"
while [[ ! -f "$ROOT/OVERLAP75_CACHES_COMPLETE" || ! -f "$ROOT/FLOW_CACHE_COMPLETE" ]]; do
  sleep 60
done
cd "/Users/jleto/LocalProjects/theory-of-mind"
python3 evaluate_robustness_study.py 2>&1 | tee "$LOG"
touch "$ROOT/STUDY_COMPLETE"
