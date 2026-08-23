#!/usr/bin/env python3
"""Test DINO's implicit context prior with hot-patch cluster boosting.

Candidate regions are connected components of the globally hot raw surprise
map.  Each component's pooled DINO token is compared with pooled low-surprise
background tokens in the same frame.  The component alone is multiplied by
1 + beta * (1 - cosine(object, background)).  No learned head or VLM is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument('video',type=Path);p.add_argument('patches',type=Path);p.add_argument('maps',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--segment-start',type=int,required=True);p.add_argument('--output-start',type=int,required=True);p.add_argument('--output-end',type=int,required=True)
    p.add_argument('--beta',type=float,default=1.0);p.add_argument('--min-cells',type=int,default=2)
    return p.parse_args()


def overlay(frame, values, lo, hi, label):
    value=np.clip((values-lo)/max(hi-lo,1e-6),0,1).astype(np.float32)
    value=cv2.GaussianBlur(value,(3,3),0)
    heat=cv2.applyColorMap(cv2.resize((value*255).astype(np.uint8),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO)
    out=cv2.addWeighted(frame,.55,heat,.45,0)
    cv2.putText(out,label,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(out,label,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.68,(0,0,0),1,cv2.LINE_AA)
    return out


def norm(v): return v/(np.linalg.norm(v)+1e-8)


def main():
    a=args();patches=np.load(a.patches,mmap_mode='r');maps=np.load(a.maps,mmap_mode='r')
    offset,count=a.output_start-a.segment_start,a.output_end-a.output_start
    if offset<0 or offset+count>len(maps):raise ValueError('range outside cache')
    # Fixed thresholds and display scale across the entire comparison segment.
    all_scores=np.asarray(maps[offset:offset+count],np.float32)
    display_lo,display_hi=np.percentile(all_scores,(5,95))
    hot_threshold=np.percentile(all_scores,90)
    background_threshold=np.percentile(all_scores,60)
    print(f'global display p5/p95={display_lo:.4f}/{display_hi:.4f}; hot>{hot_threshold:.4f}; background<{background_threshold:.4f}',flush=True)
    cap=cv2.VideoCapture(str(a.video));cap.set(cv2.CAP_PROP_POS_FRAMES,a.output_start)
    fps=cap.get(cv2.CAP_PROP_FPS) or 15.;w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h))
    for local_i in range(count):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('video ended early')
        cache_i=offset+local_i;raw=np.asarray(maps[cache_i],np.float32);tokens=np.asarray(patches[cache_i],np.float32)
        boosted=raw.copy();hot=(raw>=hot_threshold).astype(np.uint8)
        n_labels,labels,_,_=cv2.connectedComponentsWithStats(hot,connectivity=4)
        background=raw<background_threshold
        if background.any(): background_token=norm(tokens[background].mean(axis=0))
        else: background_token=norm(tokens.reshape(-1,tokens.shape[-1]).mean(axis=0))
        strongest=0.; component_count=0
        for label in range(1,n_labels):
            mask=labels==label
            if int(mask.sum())<a.min_cells:continue
            obj_token=norm(tokens[mask].mean(axis=0)); mismatch=max(0.,1.-float(obj_token@background_token))
            boosted[mask]=raw[mask]*(1.+a.beta*mismatch)
            strongest=max(strongest,mismatch);component_count+=1
        left=overlay(frame,raw,display_lo,display_hi,'Raw surprise (global scale)')
        right=overlay(frame,boosted,display_lo,display_hi,'Context-cluster boost (global scale)')
        cv2.putText(right,f'clusters={component_count}  max mismatch={strongest:.2f}',(18,66),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(right,f'clusters={component_count}  max mismatch={strongest:.2f}',(18,66),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA)
        writer.write(cv2.hconcat((left,right)))
        if local_i%150==0:print(f'rendered {local_i+1}/{count}',flush=True)
    cap.release();writer.release()

if __name__=='__main__':main()
