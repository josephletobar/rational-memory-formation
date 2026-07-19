#!/usr/bin/env bash
# Silent watchdog for the long-running cache pipeline.  It never starts or
# stops extraction itself; it writes a durable terminal-state file so an
# interrupted local Codex session does not hide a completed/failed pipeline.
set -euo pipefail

ROOT="/Volumes/Crucial X9/theory-of-mind/robustness_study"
STATUS="$ROOT/CACHE_PIPELINE_STATUS"
LOG_DIR="$ROOT/cache_logs"

require_runner() {
  local completion_marker="$1" process_pattern="$2" label="$3"
  # A waiting stage is normal, but only while the corresponding detached
  # launcher still exists.  This catches an outer `set -e` exit that did not
  # get far enough to append a FAILED marker to its log.
  if [[ ! -f "$completion_marker" ]] && ! pgrep -f "$process_pattern" >/dev/null; then
    printf '%s FAILURE — %s launcher exited before its completion marker\n' \
      "$(date '+%F %T')" "$label" > "$STATUS"
    exit 2
  fi
}

while true; do
  if [[ -f "$ROOT/ALL_LOCAL_CACHES_VERIFIED" ]]; then
    printf '%s COMPLETE\n' "$(date '+%F %T')" > "$STATUS"
    exit 0
  fi
  # Logs are append-only: a runner can log a terminal-looking failed attempt,
  # then be restarted by its outer screen.  Treat a failure as terminal only
  # when it is the most recent runner marker across the logs.
  # Every runner marker begins with an ISO timestamp in brackets.  Sort those
  # lines before selecting the latest one: filesystem traversal order from
  # grep is not chronological when several append-only logs coexist.
  last_marker="$(grep -RhE '\] (START|FAILED|RETRY)|cache pass failed after 3 resumable attempts' "$LOG_DIR" 2>/dev/null | LC_ALL=C sort | tail -n 1 || true)"
  if [[ "$last_marker" == *"FAILED "* || "$last_marker" == *"cache pass failed after 3 resumable attempts"* ]]; then
    printf '%s FAILURE — inspect %s\n' "$(date '+%F %T')" "$LOG_DIR" > "$STATUS"
    exit 2
  fi
  require_runner "$ROOT/LATENT_CACHES_COMPLETE" \
    'run_robustness_latent_sweep\.sh' 'latent cache sweep'
  require_runner "$ROOT/OVERLAP75_CACHES_COMPLETE" \
    'run_robustness_overlap75_after_base\.sh' '75%-overlap sweep'
  require_runner "$ROOT/STUDY_COMPLETE" \
    'run_robustness_evaluation_when_ready\.sh' 'evaluation gate'
  require_runner "$ROOT/ALL_LOCAL_CACHES_VERIFIED" \
    'run_full_cache_after_robustness\.sh' 'all-video cache pass'
  printf '%s RUNNING\n' "$(date '+%F %T')" > "$STATUS"
  sleep 45
done
