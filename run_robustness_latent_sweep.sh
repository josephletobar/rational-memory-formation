#!/usr/bin/env bash
# Sequential GPU sweep.  VideoMAE's W32/W64 requested rows are represented by
# the documented closest native W16 configurations in the evaluation manifest.
set -euo pipefail

ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
LOG_DIR="$ROOT/cache_logs"
mkdir -p "$LOG_DIR"

run() {
  local encoder="$1" window="$2" stride="$3"
  local target="$ROOT/feature_cache/$encoder/w${window}_s${stride}"
  local log="$LOG_DIR/${encoder}_w${window}_s${stride}.log"
  # The cache helper is idempotent: a retry only transfers/extracts files that
  # have not already been copied to the X9.  Retrying here prevents a transient
  # SSH or pod hiccup from abandoning the remaining controlled configurations.
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
  echo "[$(date '+%F %T')] FAILED $encoder W$window S$stride after 3 attempts" | tee -a "$log" >&2
  return 1
}

# Native VideoMAE has 16 temporal positions.  These are its four unique,
# closest-supported W16 configurations needed by the six requested rows.
run videomae 16 8
run videomae 16 16
run videomae 16 32
run videomae 16 64
run videomae 16 4

# V-JEPA supports all six requested configurations directly.
run vjepa2 16 8
run vjepa2 32 16
run vjepa2 64 32
run vjepa2 16 16
run vjepa2 32 32
run vjepa2 64 64
run vjepa2 16 4
run vjepa2 32 8
run vjepa2 64 16

touch "$ROOT/LATENT_CACHES_COMPLETE"
