#!/usr/bin/env bash
set -euo pipefail

# Stream missing feature caches to the pod in small batches, then copy each
# completed batch back to the Crucial X9 before freeing pod disk space.

ENCODER="${1:-}"
if [[ -z "$ENCODER" ]]; then
  echo "usage: cache_missing_on_pod.sh {videomae|vjepa2|optical_flow}" >&2
  exit 2
fi
# VideoMAE has been explicitly removed from the active workload.  Keep this
# guard at the shared entry point so detached legacy launchers exit safely.
if [[ "$ENCODER" == "videomae" ]]; then
  echo "VideoMAE caching is disabled by current project scope." >&2
  exit 0
fi
# Cache destinations live with the main corpus, while sources span the main
# corpus, new_qanda, and the canonical QuickTime copies in ego4d_download.
LOCAL_SOURCE_ROOT="/Volumes/Crucial X9"
LOCAL_DATA_ROOT="$LOCAL_SOURCE_ROOT/theory-of-mind"
QUICKTIME_ROOT="$LOCAL_SOURCE_ROOT/ego4d_download/quicktime"
POD_HOST="root@69.30.85.137"
POD_PORT="22096"
POD_KEY="/Users/jleto/.ssh/id_ed25519"
REMOTE_REPO="/workspace/rational-memory-formation"
# V-JEPA2 was added after the pod image's bundled Transformers release.  Its
# isolated venv inherits the pod's CUDA/PyTorch packages but carries a recent
# Transformers build.  Other encoders remain on the image default.
REMOTE_PYTHON="${REMOTE_PYTHON:-}"
# A second encoder can use the same pod safely when it has its own staging
# directory, output directory, and list file.  The default keeps the existing
# single-encoder behavior intact.
REMOTE_TAG_INPUT="${REMOTE_TAG:-}"
REMOTE_TAG="${REMOTE_TAG_INPUT:-$ENCODER}"
REMOTE_STAGE="$REMOTE_REPO/cache_stage_$REMOTE_TAG"
# Feature batches are ephemeral: write them on local container storage and
# rsync them to the Crucial X9 before deletion.  This avoids a network-volume
# quota failure observed despite plentiful reported /workspace capacity.
REMOTE_TEMP_ROOT="/tmp/rmf_feature_cache"
REMOTE_OUTPUT="$REMOTE_TEMP_ROOT/cache_out_$REMOTE_TAG"
REMOTE_LIST="$REMOTE_TEMP_ROOT/cache_batch_$REMOTE_TAG.list"
# A low-priority, resumable pre-stage may populate this exact source tree
# while the controlled GPU study runs.  It is used only after its atomic
# completion marker exists; batch uploads remain the safe fallback.
REMOTE_FULL_SOURCE="$REMOTE_REPO/full_cache_source"
# Eight source videos comfortably fit on the pod volume while cutting the
# number of encoder reloads in half.  The already-running invocation has
# captured its batch size, so this affects the subsequent V-JEPA/flow passes.
BATCH_SIZE="${BATCH_SIZE:-8}"
# Optional disjoint shards let CPU-only optical flow use the pod's otherwise
# idle CPU capacity.  Each shard creates its own manifest, staging directory,
# and output directory via REMOTE_TAG, so no two workers write the same cache.
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || (( SHARD_INDEX >= SHARD_COUNT )); then
  echo "SHARD_COUNT must be positive and SHARD_INDEX must be in [0, SHARD_COUNT)" >&2
  exit 2
fi
MANIFEST="$(mktemp -t theory_of_mind_missing.XXXXXX)"
PART_DIR="$(mktemp -d -t theory_of_mind_cache_parts.XXXXXX)"

cleanup() {
  rm -f "$MANIFEST"
  rmdir "$PART_DIR" 2>/dev/null || true
}
trap cleanup EXIT

case "$ENCODER" in
  videomae)
    # The all-video cache contract is the downstream default W16/S16 cache.
    # Do not let an experimental robustness-study stride satisfy this pass.
    CACHE_MARKER=".MCG-NJU-videomae-base.w16.s16.full.npz"
    LOCAL_CACHE_DIR="$LOCAL_DATA_ROOT/pod_cache_backup_current/results_videomae_cache_stride16/feature_cache"
    ;;
  vjepa2)
    # V-JEPA's default downstream representation is W32/S16.
    CACHE_MARKER=".facebook-vjepa2-vitl-fpc32-256-diving48.w32.s16.full.npz"
    LOCAL_CACHE_DIR="$LOCAL_DATA_ROOT/pod_cache_backup_current/results_vjepa2_cache_stride16/feature_cache"
    REMOTE_PYTHON="${REMOTE_PYTHON:-/workspace/venvs/vjepa/bin/python}"
    ;;
  optical_flow)
    CACHE_MARKER=".optflow.h16x12.lvl3.ts2.w32.s32.full.npz"
    LOCAL_CACHE_DIR="$LOCAL_DATA_ROOT/pod_cache_backup_current/results_optical_flow_cache/feature_cache"
    ;;
  *)
    echo "unknown encoder: $ENCODER" >&2
    exit 2
    ;;
esac

REMOTE_PYTHON="${REMOTE_PYTHON:-python}"

mkdir -p "$LOCAL_CACHE_DIR"

# The primary queue runs VideoMAE first.  V-JEPA and optical flow also have
# dedicated parallel workers with separate pod staging/output directories.
# When this primary queue reaches either encoder, wait for its parallel worker
# to finish, then build a fresh manifest and only fill any residual misses.
if [[ -z "$REMOTE_TAG_INPUT" ]]; then
  case "$ENCODER" in
    videomae) PARALLEL_SCREEN="encoder_vmae_worker" ;;
    vjepa2) PARALLEL_SCREEN="theory_vjepa_parallel" ;;
    optical_flow) PARALLEL_SCREEN="encoder_flow_worker" ;;
    *) PARALLEL_SCREEN="" ;;
  esac
  if [[ -n "$PARALLEL_SCREEN" ]]; then
    while screen -ls 2>/dev/null | grep -q "[.]$PARALLEL_SCREEN"; do
      echo "[$ENCODER] waiting for $PARALLEL_SCREEN to finish"
      sleep 30
    done
  fi
fi

python3 - "$LOCAL_SOURCE_ROOT" "$LOCAL_DATA_ROOT" "$QUICKTIME_ROOT" "$CACHE_MARKER" "$MANIFEST" "$SHARD_COUNT" "$SHARD_INDEX" <<'PY'
from pathlib import Path
import re
import sys
import zipfile

source_root = Path(sys.argv[1])
root = Path(sys.argv[2])
quicktime_root = Path(sys.argv[3])
marker = sys.argv[4]
output = Path(sys.argv[5])
shard_count = int(sys.argv[6])
shard_index = int(sys.argv[7])
# The root has a handful of rendered/review outputs beside the actual raw
# recordings. Cache source recordings only: 16-hex-ID root videos, the one
# legacy egocentric source video, every original download in new_qanda, and
# one canonical QuickTime file per UUID. The _qt_mac/_quicktime/_qt_final
# files are conversion variants of the corresponding _qt source.
videos = [
    p for p in root.glob("*.mp4")
    if not p.name.startswith("._")
    and (re.fullmatch(r"[0-9a-f]{16}\.mp4", p.name) or p.name == "egocentric_video.mp4")
]
videos.extend(
    # new_qanda contains both direct downloads and nested collections.  The
    # full-cache contract is every source video in that corpus, not merely
    # the files at its top level.
    p for p in (root / "new_qanda_downloads").rglob("*.mp4")
    if not p.name.startswith("._")
)
videos.extend(
    p for p in quicktime_root.glob("*_qt.mp4")
    if not p.name.startswith("._")
)

def valid_feature_cache(path: Path) -> bool:
    """Lightweight archive check; do not accept a Finder sidecar/corrupt NPZ."""
    if not path.is_file() or path.stat().st_size == 0 or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return {"features.npy", "starts.npy", "ends.npy"}.issubset(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False

cached = {
    p.name.split(marker, 1)[0]
    for p in root.rglob("*.npz")
    # Ignore Finder/resource-fork sidecars.  They look like NPZ filenames but
    # are not feature archives and must not suppress a needed extraction.
    if (
        marker in p.name
        and not p.name.startswith("._")
        and valid_feature_cache(p)
    )
}
missing = sorted(p.relative_to(source_root).as_posix() for p in videos if p.stem not in cached)
if shard_count > 1:
    missing = [p for i, p in enumerate(missing) if i % shard_count == shard_index]
output.write_text("\n".join(missing) + ("\n" if missing else ""))
print(f"{len(missing)} videos missing {marker} (shard {shard_index + 1}/{shard_count})")
PY

if [[ ! -s "$MANIFEST" ]]; then
  echo "nothing missing for $ENCODER"
  exit 0
fi

rsync -a --no-owner --no-group -e "ssh -p $POD_PORT -i $POD_KEY" \
  "$(dirname "$0")/cache_video_batch.py" \
  "$POD_HOST:$REMOTE_REPO/cache_video_batch.py"

USE_FULL_SOURCE=0
if ssh -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST" \
  "test -f '$REMOTE_FULL_SOURCE/STAGED_COMPLETE'"; then
  USE_FULL_SOURCE=1
  echo "[$ENCODER] using completed pod-local full source stage"
fi

split -d -a 5 -l "$BATCH_SIZE" "$MANIFEST" "$PART_DIR/batch_"
for batch in "$PART_DIR"/batch_*; do
  [[ -s "$batch" ]] || continue
  count="$(wc -l < "$batch" | tr -d ' ')"
  echo "[$ENCODER] preparing batch of $count videos"
  ssh -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST" \
    "mkdir -p '$REMOTE_OUTPUT' '$REMOTE_TEMP_ROOT'"
  INPUT_ROOT="$REMOTE_STAGE"
  if [[ "$USE_FULL_SOURCE" == 1 ]]; then
    INPUT_ROOT="$REMOTE_FULL_SOURCE"
  else
    ssh -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST" "mkdir -p '$REMOTE_STAGE'"
    rsync -a --no-owner --no-group --partial --files-from="$batch" -e "ssh -p $POD_PORT -i $POD_KEY" \
      "$LOCAL_SOURCE_ROOT/" "$POD_HOST:$REMOTE_STAGE/"

    # This nominal QuickTime source has a damaged H.264 payload throughout
    # most of its 21:54 duration.  Its same-ID VP9 original is intact and has
    # the exact same duration/resolution.  Copy it only into the ephemeral
    # pod stage under the canonical `_qt` name: source recordings on the X9
    # remain untouched, while the resulting cache retains the canonical stem
    # required by the all-video contract.
    CORRUPT_QT_REL="ego4d_download/quicktime/3b47e163-76ea-4eb5-9a99-afacf6fb41ec_qt.mp4"
    REPAIR_SOURCE="$LOCAL_SOURCE_ROOT/ego4d_download/3b47e163-76ea-4eb5-9a99-afacf6fb41ec.mp4"
    if grep -Fqx "$CORRUPT_QT_REL" "$batch"; then
      echo "[$ENCODER] staging intact same-ID source for damaged 3b47 QuickTime copy"
      rsync -a --no-owner --no-group --partial -e "ssh -p $POD_PORT -i $POD_KEY" \
        "$REPAIR_SOURCE" "$POD_HOST:$REMOTE_STAGE/$CORRUPT_QT_REL"
    fi
  fi
  scp -P "$POD_PORT" -i "$POD_KEY" "$batch" "$POD_HOST:$REMOTE_LIST"
  ssh -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST" \
    "cd '$REMOTE_REPO' && '$REMOTE_PYTHON' cache_video_batch.py --input-root '$INPUT_ROOT' --list-file '$REMOTE_LIST' --cache-dir '$REMOTE_OUTPUT' --encoder '$ENCODER' --device cuda"
  rsync -a --no-owner --no-group -e "ssh -p $POD_PORT -i $POD_KEY" \
    "$POD_HOST:$REMOTE_OUTPUT/" "$LOCAL_CACHE_DIR/"
  if [[ "$USE_FULL_SOURCE" == 1 ]]; then
    ssh -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST" \
      "rm -rf '$REMOTE_OUTPUT' '$REMOTE_LIST'"
  else
    ssh -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST" \
      "rm -rf '$REMOTE_STAGE' '$REMOTE_OUTPUT' '$REMOTE_LIST'"
  fi
  echo "[$ENCODER] cached locally: $count videos"
done
