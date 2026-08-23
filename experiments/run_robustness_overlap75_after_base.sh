#!/usr/bin/env bash
# Add the predeclared 75%-overlap rows only after the original GPU sweep.
set -euo pipefail
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
while [[ ! -f "$ROOT/LATENT_CACHES_COMPLETE" ]]; do
  sleep 60
done
cd "/Users/jleto/LocalProjects/theory-of-mind"

run() {
  local encoder="$1" window="$2" stride="$3"
  local target="$ROOT/feature_cache/$encoder/w${window}_s${stride}"
  local log="$ROOT/cache_logs/${encoder}_w${window}_s${stride}.log"
  local attempt
  for attempt in 1 2 3; do
    echo "[$(date '+%F %T')] START $encoder W$window S$stride (attempt $attempt/3)" | tee -a "$log"
    if ./cache_robustness_config_on_pod.sh "$encoder" "$window" "$stride" "$target" 2>&1 | tee -a "$log"; then
      echo "[$(date '+%F %T')] DONE $encoder W$window S$stride" | tee -a "$log"
      return 0
    fi
    echo "[$(date '+%F %T')] RETRY $encoder W$window S$stride after transient failure" | tee -a "$log"
    sleep 20
  done
  return 1
}

run videomae 16 4
run vjepa2 16 4
run vjepa2 32 8
run vjepa2 64 16
touch "$ROOT/OVERLAP75_CACHES_COMPLETE"
