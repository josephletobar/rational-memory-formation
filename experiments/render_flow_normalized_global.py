#!/usr/bin/env python3
"""Minimal default: globally scaled, flow-magnitude-invariant surprise."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from surprise_normalization import DEFAULT_FLOW_ALPHA, raw_global_display_scale

def args():
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('maps',type=Path);p.add_argument('flow',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--segment-start',type=int,required=True);p.add_argument('--output-start',type=int,required=True);p.add_argument('--output-end',type=int,required=True);p.add_argument('--alpha',type=float,default=DEFAULT_FLOW_ALPHA)
    p.add_argument('--display-ema',type=float,default=.25,help='display-only EMA weight for the current heatmap; 1 disables temporal smoothing')
    return p.parse_args()

def main():
    a=args()
    if not 0 < a.display_ema <= 1: raise ValueError('--display-ema must be in (0, 1]')
    maps=np.load(a.maps,mmap_mode='r');flow=np.load(a.flow,mmap_mode='r');offset,count=a.output_start-a.segment_start,a.output_end-a.output_start
    raw=np.asarray(maps[offset:offset+count],np.float32);flows=np.asarray(flow[offset-1:offset+count-1],np.float32)
    normalized=raw/(1+a.alpha*np.linalg.norm(flows,axis=-1))
    # Preserve the useful attenuation visually: use the same raw-score
    # clip-wide p5/p95 scale as the proven raw-vs-flow-corrected comparison.
    lo,hi=raw_global_display_scale(raw);print(f'global raw-score display scale p5/p95={lo:.5f}/{hi:.5f}',flush=True)
    cap=cv2.VideoCapture(str(a.video));cap.set(cv2.CAP_PROP_POS_FRAMES,a.output_start);fps=cap.get(cv2.CAP_PROP_FPS) or 15.;w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    displayed_heat=None
    for i in range(count):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('video ended early')
        score=normalized[i];heat=np.clip((score-lo)/max(hi-lo,1e-6),0,1).astype(np.float32);heat=cv2.GaussianBlur(heat,(3,3),0)
        displayed_heat=heat if displayed_heat is None else a.display_ema*heat+(1-a.display_ema)*displayed_heat
        heat=displayed_heat
        heat=cv2.applyColorMap(cv2.resize((heat*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO)
        out=cv2.addWeighted(frame,.55,heat,.45,0)
        text=f'Flow-normalized surprise | raw global display scale | alpha={a.alpha:g} | display EMA={a.display_ema:g}'
        cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),3,cv2.LINE_AA);cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.68,(0,0,0),1,cv2.LINE_AA)
        writer.write(out)
        if i%150==0:print(f'rendered {i+1}/{count}',flush=True)
    cap.release();writer.release()
if __name__=='__main__':main()
