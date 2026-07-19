#!/usr/bin/env python3
"""Patchwise DINO window-change maps: W_t(p) vs W_(t-1)(p), stride one."""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

SIZE, GRID, DIM = 384, 24, 768
FIXED_HEAT_MAX = 1.0e-5  # absolute cosine-window-change display scale, shared across runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path); p.add_argument("maps", type=Path); p.add_argument("raw_video", type=Path)
    p.add_argument("--window", type=int, default=300); p.add_argument("--batch-size", type=int, default=12)
    return p.parse_args()


def inputs(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb=np.stack([cv2.cvtColor(cv2.resize(f,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for f in frames])
    x=torch.from_numpy(rgb).permute(0,3,1,2).float().div_(255); mean=torch.tensor((.485,.456,.406)).view(1,3,1,1); std=torch.tensor((.229,.224,.225)).view(1,3,1,1)
    return ((x-mean)/std).to(device)


def overlay(frame: np.ndarray, score: np.ndarray, warming: bool) -> np.ndarray:
    if warming:
        out, text = frame.copy(), "DINO patch W_t vs W_t-1: warming up 300 frames"
    else:
        value=cv2.GaussianBlur(np.clip(score/FIXED_HEAT_MAX,0,1).astype(np.float32),(3,3),0)
        heat=cv2.applyColorMap(cv2.resize((value*255).astype(np.uint8),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO)
        out, text=cv2.addWeighted(frame,.55,heat,.45,0), "DINO patch W_t vs W_t-1 | fixed scale 0..1e-5"
    cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),3,cv2.LINE_AA);cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.56,(0,0,0),1,cv2.LINE_AA)
    return out


def main() -> None:
    a=parse_args(); cap=cv2.VideoCapture(str(a.video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 15.; cap.release()
    a.maps.parent.mkdir(parents=True,exist_ok=True); temp=a.maps.with_suffix(".partial.npy"); temp.unlink(missing_ok=True); maps=np.lib.format.open_memmap(temp,mode="w+",dtype=np.float16,shape=(total,GRID,GRID))
    device=torch.device("cuda"); model=timm.create_model("vit_base_patch16_dinov3",pretrained=True,num_classes=0).eval().to(device); history:deque[torch.Tensor]=deque(); running=torch.zeros((GRID*GRID,DIM),device=device,dtype=torch.float32)
    cap=cv2.VideoCapture(str(a.video)); index=0
    with torch.inference_mode():
        while index<total:
            frames=[]
            for _ in range(min(a.batch_size,total-index)):
                ok,frame=cap.read()
                if not ok: raise RuntimeError("source ended early")
                frames.append(frame)
            out=model.forward_features(inputs(frames,device))
            if isinstance(out,dict): out=out.get("x_norm_patchtokens",out.get("patch_tokens"))
            if out is None: raise RuntimeError("DINO returned no patch tokens")
            if out.shape[1] != GRID*GRID:
                prefix = getattr(model, "num_prefix_tokens", out.shape[1] - GRID*GRID)
                out = out[:, prefix:]
            if out.shape[1:] != (GRID*GRID,DIM): raise RuntimeError(f"unexpected patch-token shape {tuple(out.shape)}")
            for current in out:
                current=current.float(); current=current/(current.norm(dim=-1,keepdim=True)+1e-8)
                if len(history)<a.window:
                    history.append(current.to(torch.float16).clone()); running.add_(current); maps[index]=np.nan
                else:
                    old=history.popleft().float(); prior=running/(running.norm(dim=-1,keepdim=True)+1e-8); running.add_(current-old); present=running/(running.norm(dim=-1,keepdim=True)+1e-8)
                    maps[index]=(1-(present*prior).sum(-1)).reshape(GRID,GRID).clamp_min(0).cpu().numpy().astype(np.float16); history.append(current.to(torch.float16).clone())
                index+=1
                if index%100==0 or index==total: print(f"encoded {index}/{total}",flush=True)
    cap.release();del maps,model,history,running;torch.cuda.empty_cache();os.replace(temp,a.maps)
    maps=np.load(a.maps,mmap_mode="r"); cap=cv2.VideoCapture(str(a.video));w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));a.raw_video.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.raw_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    for i in range(total):
        ok,frame=cap.read()
        if not ok: raise RuntimeError("source ended during render")
        warm=i<a.window or not np.isfinite(maps[i]).all(); score=np.zeros((GRID,GRID),np.float32) if warm else np.asarray(maps[i],np.float32);writer.write(overlay(frame,score,warm))
        if (i+1)%150==0 or i+1==total: print(f"rendered {i+1}/{total}",flush=True)
    cap.release();writer.release()


if __name__=="__main__": main()
