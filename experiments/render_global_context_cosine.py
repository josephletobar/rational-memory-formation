#!/usr/bin/env python3
"""No-threshold global-context cosine adjustment of cached DINO patch surprise."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from surprise_normalization import DEFAULT_FLOW_ALPHA, global_context_adjustment, motion_normalize, raw_global_display_scale

def args():
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('patches',type=Path);p.add_argument('maps',type=Path);p.add_argument('output',type=Path)
    p.add_argument('flow',type=Path)
    p.add_argument('--segment-start',type=int,required=True);p.add_argument('--output-start',type=int,required=True);p.add_argument('--output-end',type=int,required=True);p.add_argument('--alpha',type=float,default=DEFAULT_FLOW_ALPHA,help='Default flow-normalization strength.');p.add_argument('--beta',type=float,default=2.0,help='Global-context cosine adjustment strength.')
    return p.parse_args()

def overlay(frame,score,lo,hi,label):
    heat=np.clip((score-lo)/max(hi-lo,1e-6),0,1).astype(np.float32);heat=cv2.GaussianBlur(heat,(3,3),0)
    heat=cv2.applyColorMap(cv2.resize((heat*255).astype(np.uint8),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO)
    out=cv2.addWeighted(frame,.55,heat,.45,0)
    cv2.putText(out,label,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.66,(255,255,255),3,cv2.LINE_AA);cv2.putText(out,label,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.66,(0,0,0),1,cv2.LINE_AA)
    return out

def main():
    a=args();tokens=np.load(a.patches,mmap_mode='r');maps=np.load(a.maps,mmap_mode='r');flow=np.load(a.flow,mmap_mode='r');offset,count=a.output_start-a.segment_start,a.output_end-a.output_start
    # Preserve the exact display convention from the good flow-corrected
    # baseline: a raw-score clip-wide scale shared by raw and corrected maps.
    # Re-scaling after normalization would artificially re-expand fast-motion
    # regions and hide the intended attenuation.
    raw_all=np.asarray(maps[offset:offset+count],np.float32)
    lo,hi=raw_global_display_scale(raw_all);print(f'global raw-score scale p5/p95={lo:.4f}/{hi:.4f}',flush=True)
    cap=cv2.VideoCapture(str(a.video));cap.set(cv2.CAP_PROP_POS_FRAMES,a.output_start);fps=cap.get(cv2.CAP_PROP_FPS) or 15.;w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h))
    for i in range(count):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('video ended early')
        j=offset+i;raw=np.asarray(maps[j],np.float32);x=np.asarray(tokens[j],np.float32)
        motion_free=motion_normalize(raw, np.asarray(flow[j-1],np.float32), a.alpha)
        adjusted=global_context_adjustment(motion_free,x,a.beta)
        left=overlay(frame,motion_free,lo,hi,f'Flow-normalized surprise (alpha={a.alpha:g})')
        right=overlay(frame,adjusted,lo,hi,f'Flow-normalized + global context (beta={a.beta:g})')
        writer.write(cv2.hconcat((left,right)))
        if i%150==0:print(f'rendered {i+1}/{count}',flush=True)
    cap.release();writer.release()
if __name__=='__main__':main()
