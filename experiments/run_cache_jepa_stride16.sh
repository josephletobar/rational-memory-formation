#!/usr/bin/env bash
set -euo pipefail

PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/samworld/bin/python"
PROJECT="/Users/jleto/LocalProjects/theory-of-mind"
DATA="/Volumes/Crucial X9/theory-of-mind"
RESULTS="$PROJECT/results_vjepa_cache_stride16"

mkdir -p "$RESULTS/feature_cache"

"$PYTHON" -u - <<'PY'
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from train_vjepa_probes import (
    ENCODER_PRESETS,
    AutoModel,
    choose_device,
    extract_video,
    re,
)


data_dir = Path('/Volumes/Crucial X9/theory-of-mind')
output = Path('/Users/jleto/LocalProjects/theory-of-mind/results_vjepa_cache_stride16')
cache_dir = output / 'feature_cache'
cache_dir.mkdir(parents=True, exist_ok=True)

window_frames = 32
stride_frames = 16
device = choose_device('mps')
model_id = ENCODER_PRESETS['vjepa2']
print(f'Loading encoder {model_id} on {device} ...', flush=True)
model = AutoModel.from_pretrained(model_id)
encoder = model.vjepa2 if hasattr(model, 'vjepa2') else model
encoder.requires_grad_(False).eval().to(device)

# JEPA feature cache key (matches train_vjepa_probes.py naming)
slug = 'facebook-vjepa2-vitl-fpc32-256-diving48'

phase = 1
labeled = []
seen = set()

for ann in sorted(data_dir.glob('*.clipme.json')):
    if ann.name.startswith('._'):
        continue
    if ann.name.endswith('.glance-over-1s.clipme.json'):
        continue
    try:
        data = json.loads(ann.read_text())
    except Exception:
        print(f'!! skip unreadable annotation: {ann.name}', flush=True)
        continue
    video = Path(data.get('video', ''))
    if not video.name:
        print(f'!! skip bad annotation (no video): {ann.name}', flush=True)
        continue
    if not video.exists():
        video = ann.parent / video
    if not video.exists():
        print(f'!! skip missing video {data.get("video", "")!r} from {ann.name}', flush=True)
        continue
    if video.name in seen:
        continue
    seen.add(video.name)
    labeled.append(video)

raw = []
for video in sorted(data_dir.glob('*.mp4')):
    if video.name.startswith('.'):
        continue
    if video.name.endswith('_annotation_review.mp4'):
        continue
    if video.name.endswith('_vjepa_predictions.mp4'):
        continue
    if video.name.endswith('_rf_render.mp4') or video.name.endswith('_jepa_rf_render.mp4'):
        continue
    if 'glance-over-1s' in video.name:
        continue
    if video.name.startswith('._'):
        continue
    raw.append(video)


def cache_for(video):
    return cache_dir / f"{video.stem}.{slug}.w{window_frames}.s{stride_frames}.full.npz"

def extract_list(videos, label):
    args = SimpleNamespace(
        window_frames=window_frames,
        stride_frames=stride_frames,
        feature_batch_size=2,
        max_seconds=None,
    )
    ok = 0
    fail = 0
    print(f'=== {label} phase: {len(videos)} videos ===', flush=True)

    for video in videos:
        if video.name not in seen and label == 'labeled':
            # not needed, but keeps list robust if duplicates
            pass
        cache = cache_for(video)
        if cache.exists():
            print(f'cached skip: {video.name}', flush=True)
            ok += 1
            continue
        try:
            print(f'processing {video.name}', flush=True)
            extract_video(video, cache, encoder, device, args, 'vjepa2')
            ok += 1
        except Exception:
            fail += 1
            print(f'ERROR: {video.name}', flush=True)
            traceback.print_exc()
        
    print(f'{label} done: ok={ok} fail={fail}', flush=True)
    return ok, fail

extract_list(labeled, 'labeled')

# Continue raw videos after labeled pass
remaining = [v for v in raw if v.name not in {x.name for x in labeled}]
extract_list(remaining, 'raw')

print('all-done', flush=True)
PY