#!/usr/bin/env bash
# Let the in-flight V-JEPA/flow run finish, then restart its matching
# V-JEPA/flow-only completion sequence if recovery is needed.
set -euo pipefail

OLD_PID="${1:?usage: $0 EXISTING_RUNNER_PID}"
ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"

while kill -0 "$OLD_PID" 2>/dev/null; do
  sleep 60
done

# The verifier's marker is valid only after the V-JEPA/flow-only completion
# pass succeeds, so clear any stale version before recovery.
rm -f "$ROOT/ALL_LOCAL_CACHES_VERIFIED"
cd "/Users/jleto/LocalProjects/theory-of-mind"
exec ./run_full_cache_after_robustness.sh
