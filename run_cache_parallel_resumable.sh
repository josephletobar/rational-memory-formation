#!/usr/bin/env bash
# Generic durable worker for a full-corpus cache family.  It resumes from
# locally synced cache archives after any pod/SSH interruption.
set -u

ENCODER="${1:?usage: $0 {videomae|optical_flow} remote_tag log_path}"
TAG="${2:?usage: $0 {videomae|optical_flow} remote_tag log_path}"
LOG="${3:?usage: $0 {videomae|optical_flow} remote_tag log_path}"
cd "/Users/jleto/LocalProjects/theory-of-mind"

attempt=0
while true; do
  attempt=$((attempt + 1))
  printf '[%s] %s cache attempt %d\n' "$(date '+%F %T')" "$ENCODER" "$attempt" >> "$LOG"
  REMOTE_TAG="$TAG" bash cache_missing_on_pod.sh "$ENCODER" >> "$LOG" 2>&1
  rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf '[%s] %s cache queue complete\n' "$(date '+%F %T')" "$ENCODER" >> "$LOG"
    exit 0
  fi
  printf '[%s] %s cache attempt %d failed (exit %d); retrying in 60s\n' \
    "$(date '+%F %T')" "$ENCODER" "$attempt" "$rc" >> "$LOG"
  sleep 60
done
