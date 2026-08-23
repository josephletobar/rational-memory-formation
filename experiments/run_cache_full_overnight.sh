#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/jleto/LocalProjects/rational-memory-formation"
POD="root@69.30.85.238"
PORT="22111"
DATA_DIR="/Volumes/Crucial X9/theory-of-mind"
REMOTE_BASE="/workspace/rational-memory-formation"
WINDOW="32"
STRIDE="16"
DEVICE="cuda"
LABELED_LIST="/tmp/pod_labeled_videos.txt"
RAW_LIST="/tmp/pod_raw_videos.txt"

run_encoder() {
  local encoder="$1"

  ./stream_cached_labeled_videos.sh \
    --pod "${POD}" \
    --port "${PORT}" \
    --local-data-dir "${DATA_DIR}" \
    --remote-base "${REMOTE_BASE}" \
    --video-list "${LABELED_LIST}" \
    --window "${WINDOW}" \
    --stride "${STRIDE}" \
    --device "${DEVICE}" \
    --encoder "${encoder}"

  ./stream_cached_labeled_videos.sh \
    --pod "${POD}" \
    --port "${PORT}" \
    --local-data-dir "${DATA_DIR}" \
    --remote-base "${REMOTE_BASE}" \
    --video-list "${RAW_LIST}" \
    --window "${WINDOW}" \
    --stride "${STRIDE}" \
    --device "${DEVICE}" \
    --encoder "${encoder}"
}

cd "${PROJECT_DIR}"
run_encoder vjepa2
run_encoder videomae

