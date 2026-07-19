#!/usr/bin/env python3
"""Side-by-side raw versus fixed flow-magnitude-corrected patch surprise."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from surprise_normalization import DEFAULT_FLOW_ALPHA


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path); p.add_argument("maps", type=Path); p.add_argument("flow", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--segment-start", type=int, required=True); p.add_argument("--output-start", type=int, required=True); p.add_argument("--output-end", type=int, required=True)
    p.add_argument("--alpha", type=float, default=DEFAULT_FLOW_ALPHA, help="Fixed attenuation: S/(1 + alpha*patch-flow magnitude).")
    return p.parse_args()


def overlay(frame: np.ndarray, values: np.ndarray, lo: float, hi: float, label: str) -> np.ndarray:
    heat = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = cv2.GaussianBlur(heat.astype(np.float32), (3, 3), 0)
    heat = cv2.resize((heat * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    blended = cv2.addWeighted(frame, .55, cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), .45, 0)
    cv2.putText(blended, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .76, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(blended, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .76, (0,0,0), 1, cv2.LINE_AA)
    return blended


def main() -> None:
    a=arguments(); maps=np.load(a.maps,mmap_mode='r'); flow=np.load(a.flow,mmap_mode='r')
    offset=a.output_start-a.segment_start; count=a.output_end-a.output_start
    if offset < 1 or offset+count > len(maps): raise ValueError("Output range is outside cached maps")
    # One stable scale for the entire requested clip: hot means the same score
    # at every timestamp, rather than merely being high relative to that frame.
    global_lo, global_hi = np.percentile(np.asarray(maps[offset:offset + count], np.float32), (5, 95))
    print(f'global raw-score scale: p5={global_lo:.5f}, p95={global_hi:.5f}', flush=True)
    cap=cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES,a.output_start)
    fps=cap.get(cv2.CAP_PROP_FPS) or 15.; w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True,exist_ok=True); writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h))
    for local_index in range(count):
        ok,frame=cap.read()
        if not ok: raise RuntimeError('Video ended early')
        cache_index=offset+local_index
        raw=np.asarray(maps[cache_index],np.float32)
        magnitude=np.linalg.norm(np.asarray(flow[cache_index-1],np.float32),axis=-1)
        corrected=raw/(1.0+a.alpha*magnitude)
        # Shared global raw-score scale preserves both time consistency and the
        # visible effect of the fixed flow correction.
        left=overlay(frame,raw,global_lo,global_hi,'Raw patch surprise (global scale)')
        right=overlay(frame,corrected,global_lo,global_hi,f'Flow-corrected (global scale): S / (1 + {a.alpha:g} |flow|)')
        writer.write(cv2.hconcat((left,right)))
        if local_index%150==0: print(f'rendered {local_index+1}/{count}',flush=True)
    cap.release();writer.release()

if __name__=='__main__': main()
