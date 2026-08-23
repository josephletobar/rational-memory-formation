#!/usr/bin/env python3
"""Global DINO window-change surprise: W_t (frames t-299..t) vs W_(t-1)."""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

SIZE, DIM = 384, 768


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path); p.add_argument("embeddings", type=Path); p.add_argument("scores", type=Path); p.add_argument("raw_video", type=Path)
    p.add_argument("--window", type=int, default=300); p.add_argument("--batch-size", type=int, default=12)
    return p.parse_args()


def inputs(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb = np.stack([cv2.cvtColor(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for f in frames])
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor((.485, .456, .406)).view(1,3,1,1); std = torch.tensor((.229,.224,.225)).view(1,3,1,1)
    return ((x-mean)/std).to(device)


def border(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0, 1))
    if value < .5:
        return (0, 255, int(510 * value))
    return (0, int(510 * (1 - value)), 255)


def main() -> None:
    a = parse_args(); cap = cv2.VideoCapture(str(a.video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 15.; cap.release()
    a.embeddings.parent.mkdir(parents=True, exist_ok=True); a.scores.parent.mkdir(parents=True, exist_ok=True)
    et = a.embeddings.with_suffix(".partial.npy"); et.unlink(missing_ok=True)
    values = np.lib.format.open_memmap(et, mode="w+", dtype=np.float16, shape=(total,DIM))
    device = torch.device("cuda"); model = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0).eval().to(device)
    cap = cv2.VideoCapture(str(a.video)); index = 0
    with torch.inference_mode():
        while index < total:
            frames = []
            for _ in range(min(a.batch_size,total-index)):
                ok, frame = cap.read()
                if not ok: raise RuntimeError("source ended early")
                frames.append(frame)
            out = model.forward_features(inputs(frames,device))
            if isinstance(out, dict): out = out.get("x_norm_clstoken", out.get("x_norm_regtokens", out.get("x_prenorm")))
            if out is None: raise RuntimeError("DINO returned no global token")
            if out.ndim == 3: out = out[:,0]
            if out.shape != (len(frames), DIM): raise RuntimeError(f"unexpected global-token shape {tuple(out.shape)}")
            values[index:index+len(frames)] = out.float().cpu().numpy().astype(np.float16); index += len(frames)
            if index % 100 == 0 or index == total: print(f"encoded {index}/{total}",flush=True)
    cap.release(); del values, model; torch.cuda.empty_cache(); os.replace(et,a.embeddings)
    z = np.asarray(np.load(a.embeddings,mmap_mode="r"),np.float32); z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    scores = np.full(total,np.nan,np.float32); running = z[:a.window].sum(axis=0); prior = running / max(np.linalg.norm(running),1e-8)
    for t in range(a.window,total):
        running += z[t] - z[t-a.window]
        current = running / max(np.linalg.norm(running),1e-8)
        scores[t] = 1.0-float(current@prior); prior=current
    np.save(a.scores,scores); valid=scores[np.isfinite(scores)]; lo,hi=np.percentile(valid,(5,95)); print(f"window={a.window}; score p5/p95={lo:.8f}/{hi:.8f}",flush=True)
    cap=cv2.VideoCapture(str(a.video)); w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); a.raw_video.parent.mkdir(parents=True,exist_ok=True); writer=cv2.VideoWriter(str(a.raw_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    for t in range(total):
        ok, frame=cap.read()
        if not ok: raise RuntimeError("source ended during render")
        if np.isfinite(scores[t]):
            v=float(np.clip((scores[t]-lo)/max(hi-lo,1e-12),0,1)); c=border(v); thick=max(12,min(w,h)//30); frame[:thick]=c;frame[-thick:]=c;frame[:,:thick]=c;frame[:,-thick:]=c; text=f"DINO global window change {scores[t]:.7f} | W_t vs W_t-1"
        else: text="DINO global 300-frame window: warming up"
        cv2.putText(frame,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),3,cv2.LINE_AA);cv2.putText(frame,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),1,cv2.LINE_AA);writer.write(frame)
        if (t+1)%150==0 or t+1==total: print(f"rendered {t+1}/{total}",flush=True)
    cap.release();writer.release()


if __name__ == "__main__": main()
