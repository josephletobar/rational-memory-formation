#!/bin/zsh
set -e

PYTHON='/opt/homebrew/Caskroom/miniconda/base/envs/samworld/bin/python'
PROJECT='/Users/jleto/LocalProjects/theory-of-mind'
DATA='/Volumes/Crucial X9/theory-of-mind'
PROBES="$PROJECT/results_two_video"

cd "$PROJECT"
mkdir -p "$PROBES/feature_cache"

# Reuse the already verified full-video feature cache for the first training video.
FIRST_CACHE='0b941d85cf228741.facebook-vjepa2-vitl-fpc32-256-diving48.w32.s32.full.npz'
if [[ ! -f "$PROBES/feature_cache/$FIRST_CACHE" ]]; then
    cp "results/feature_cache/$FIRST_CACHE" "$PROBES/feature_cache/$FIRST_CACHE"
fi

if [[ -f "$PROBES/linear_probe.pt" && -f "$PROBES/mlp_probe.pt" && -f "$PROBES/normalization.npz" && -f "$PROBES/metrics.json" ]]; then
    echo '=== Reusing completed two-video probes ==='
else
    echo '=== Training full two-video probes ==='
    "$PYTHON" -u train_vjepa_probes.py \
        --device mps \
        --data-dir "$DATA" \
        --output-dir "$PROBES" \
        --include-video '0b941d85cf228741.mp4' \
        --include-video 'egocentric_video.mp4'
fi

echo '=== Rendering library videos ==='
for stem in \
    '03fb89df97cc9908' \
    '20b7e0d8cf3e158a' \
    '21a625ead630c23e'
do
    echo "=== Processing $stem.mp4 ==="
    "$PYTHON" -u predict_video.py "$DATA/$stem.mp4" \
        --probes "$PROBES" \
        --output "$DATA/predictions_$stem.npz" \
        --render-video "$DATA/${stem}_vjepa_predictions.mp4" \
        --device mps
done

echo '=== OVERNIGHT PIPELINE COMPLETE ==='
