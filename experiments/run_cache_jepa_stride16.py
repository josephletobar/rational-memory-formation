#!/usr/bin/env python3
import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from train_vjepa_probes import ENCODER_PRESETS, AutoModel, choose_device, extract_video


def collect_labeled_videos(data_dir: Path):
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
            print(f"skip unreadable annotation: {ann.name}", flush=True)
            continue
        video_field = data.get('video')
        if not video_field:
            print(f"skip bad annotation (no video): {ann.name}", flush=True)
            continue
        candidate = data_dir / video_field
        if not candidate.exists():
            candidate = Path(video_field)
        if not candidate.exists():
            print(f"skip missing labeled video: {video_field} (from {ann.name})", flush=True)
            continue
        if candidate.name in seen:
            continue
        labeled.append(candidate)
        seen.add(candidate.name)
    return labeled


def collect_raw_videos(data_dir: Path, labeled_videos):
    labeled_names = {v.name for v in labeled_videos}
    raw = []
    seen = set()
    for video in sorted(data_dir.glob('*.mp4')):
        name = video.name
        if name in labeled_names:
            continue
        if name.startswith('._') or name.startswith('.'):
            continue
        if name.endswith('_annotation_review.mp4'):
            continue
        if name.endswith('_vjepa_predictions.mp4'):
            continue
        if 'glance-over-1s' in name:
            continue
        if name not in seen:
            raw.append(video)
            seen.add(name)
    return raw


def cache_list(videos, label, encoder, device, cache_dir, window_frames=32, stride_frames=16):
    args = SimpleNamespace(
        window_frames=window_frames,
        stride_frames=stride_frames,
        feature_batch_size=2,
        max_seconds=None,
    )
    slug = 'facebook-vjepa2-vitl-fpc32-256-diving48'
    ok = 0
    fail = 0
    skipped = 0
    print(f"=== {label} phase: {len(videos)} videos ===", flush=True)

    for idx, video in enumerate(videos, 1):
        cache = cache_dir / f"{video.stem}.{slug}.w{window_frames}.s{stride_frames}.full.npz"
        if cache.exists():
            print(f"[{idx}/{len(videos)}] cached skip: {video.name}", flush=True)
            skipped += 1
            continue
        try:
            print(f"[{idx}/{len(videos)}] processing {video.name}", flush=True)
            extract_video(video, cache, encoder, device, args, 'vjepa2')
            ok += 1
        except Exception:
            fail += 1
            print(f"ERROR on {video.name}", flush=True)
            traceback.print_exc()
    print(f"{label} done: ok={ok}, skipped={skipped}, fail={fail}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu', choices=('auto', 'cpu', 'cuda', 'mps'))
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path('/Volumes/Crucial X9/theory-of-mind')
    output = Path('/Users/jleto/LocalProjects/theory-of-mind/results_vjepa_cache_stride16')
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / 'feature_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    labeled_videos = collect_labeled_videos(data_dir)
    print(f'labeled videos selected: {len(labeled_videos)}', flush=True)
    raw_videos = collect_raw_videos(data_dir, labeled_videos)
    print(f'raw videos selected: {len(raw_videos)}', flush=True)

    device = choose_device(args.device)
    model_id = ENCODER_PRESETS['vjepa2']
    print(f'Loading encoder: {model_id} on {device}', flush=True)
    model = AutoModel.from_pretrained(model_id)
    encoder = model.vjepa2 if hasattr(model, 'vjepa2') else model
    encoder.requires_grad_(False).eval().to(device)

    cache_list(labeled_videos, 'labeled', encoder, device, cache_dir)
    cache_list(raw_videos, 'raw', encoder, device, cache_dir)

    print('all-done', flush=True)


if __name__ == '__main__':
    main()
