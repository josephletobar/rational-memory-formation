#!/usr/bin/env bash
set -euo pipefail
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
mkdir -p "$ROOT/cache_logs"
./cache_robustness_config_on_pod.sh optical_flow 32 32 \
  "$ROOT/feature_cache/optical_flow/w32_s32" \
  2>&1 | tee -a "$ROOT/cache_logs/optical_flow_w32_s32.log"
touch "$ROOT/FLOW_CACHE_COMPLETE"
