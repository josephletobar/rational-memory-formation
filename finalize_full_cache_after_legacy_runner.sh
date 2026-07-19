#!/usr/bin/env bash
# Let the legacy all-video job keep its in-flight V-JEPA and flow work, then
# run the corrected three-encoder completion sequence.  The legacy verifier
# is now strict, so it cannot publish a false-complete marker first.
set -euo pipefail

LEGACY_PID="${1:?usage: $0 LEGACY_RUNNER_PID}"
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
while kill -0 "$LEGACY_PID" 2>/dev/null; do
  sleep 60
done

rm -f "$ROOT/ALL_LOCAL_CACHES_VERIFIED"
cd "/Users/jleto/LocalProjects/theory-of-mind"
exec ./run_full_cache_after_robustness.sh
