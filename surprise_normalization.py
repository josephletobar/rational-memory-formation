"""Default, non-learned normalizers for patch-level semantic surprise."""
from __future__ import annotations

import numpy as np


# Locked baseline: this is the configuration that produced the accepted
# `09ea_last3m_flow_normalized_alpha2_raw_global.mp4` visualization.
DEFAULT_FLOW_ALPHA = 2.0
DEFAULT_DISPLAY_PERCENTILES = (5.0, 95.0)
# Accepted visualization default: patch-vs-frame-context cosine is a simple
# objectness accent, and EMA is display-only (never part of the score cache).
DEFAULT_CONTEXT_BETA = 2.0
DEFAULT_DISPLAY_EMA = 0.25


def flow_magnitude(patch_flow: np.ndarray) -> np.ndarray:
    """Return per-patch optical-flow magnitude in patch-cell units."""
    return np.linalg.norm(np.asarray(patch_flow, dtype=np.float32), axis=-1)


def motion_normalize(raw_surprise: np.ndarray, patch_flow: np.ndarray | None, alpha: float = DEFAULT_FLOW_ALPHA) -> np.ndarray:
    """Discount predictable fast motion: S_motion_free = S/(1 + alpha*|flow|).

    This is deliberately fixed, local, and non-learned.  `patch_flow=None`
    represents the first frame, for which no temporal motion exists yet.
    """
    raw = np.asarray(raw_surprise, dtype=np.float32)
    if patch_flow is None:
        return raw.copy()
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return raw / (1.0 + alpha * flow_magnitude(patch_flow))


def raw_global_display_scale(raw_scores: np.ndarray) -> tuple[float, float]:
    """Fixed clip-wide display scale; do not re-expand after flow correction."""
    lo, hi = np.percentile(np.asarray(raw_scores, dtype=np.float32), DEFAULT_DISPLAY_PERCENTILES)
    return float(lo), float(hi)


def global_context_adjustment(motion_normalized_surprise: np.ndarray, patch_tokens: np.ndarray, beta: float = DEFAULT_CONTEXT_BETA) -> np.ndarray:
    """Apply the simple no-threshold global-context cosine residual.

    Patches less similar than the frame's mean similarity to the global DINO
    context are boosted; more similar patches are attenuated.
    """
    x = np.asarray(patch_tokens, dtype=np.float32)
    global_embedding = x.mean(axis=(0, 1))
    global_embedding /= np.linalg.norm(global_embedding) + 1e-8
    local = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    cosine = np.einsum("hwd,d->hw", local, global_embedding)
    return np.asarray(motion_normalized_surprise, dtype=np.float32) * (1.0 + beta * (float(cosine.mean()) - cosine))


def frame_global_context_adjustment(motion_normalized_surprise: np.ndarray, patch_tokens: np.ndarray, frame_embedding: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Linearly scale patches by cosine similarity to the whole-frame token.

    The per-frame mean cosine is the neutral point: more frame-like patches
    attenuate, while less frame-like patches boost.  No threshold or learned
    head is involved.
    """
    x = np.asarray(patch_tokens, dtype=np.float32)
    global_token = np.asarray(frame_embedding, dtype=np.float32)
    global_token /= np.linalg.norm(global_token) + 1e-8
    local = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    cosine = np.einsum("hwd,d->hw", local, global_token)
    return np.asarray(motion_normalized_surprise, dtype=np.float32) * (1.0 + beta * (float(cosine.mean()) - cosine))
