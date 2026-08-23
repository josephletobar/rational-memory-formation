#!/usr/bin/env python3
"""Canonical flow-aligned patch pipeline with stride-one W_t vs W_(t-1) change."""
from __future__ import annotations

import argparse, os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

SIZE, PATCH, GRID, DIM = 384, 16, 24, 768
FLOW_ALPHA, DISPLAY_EMA, FIXED_HEAT_MAX = 2.0, .25, 3.0e-5


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument('video',type=Path); p.add_argument('maps',type=Path); p.add_argument('flow',type=Path); p.add_argument('roc',type=Path); p.add_argument('raw_video',type=Path)
    p.add_argument('--window',type=int,default=300); p.add_argument('--batch-size',type=int,default=12); return p.parse_args()


def model_input(frames:list[np.ndarray],device:torch.device)->torch.Tensor:
    rgb=np.stack([cv2.cvtColor(cv2.resize(f,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for f in frames])
    x=torch.from_numpy(rgb).permute(0,3,1,2).float().div_(255); mean=torch.tensor((.485,.456,.406)).view(1,3,1,1); std=torch.tensor((.229,.224,.225)).view(1,3,1,1)
    return ((x-mean)/std).to(device)


def patch_flow(current:np.ndarray,previous:np.ndarray,dis:cv2.DISOpticalFlow)->np.ndarray:
    dense=np.nan_to_num(dis.calc(current,previous,None)); return dense.reshape(GRID,PATCH,GRID,PATCH,2).mean((1,3)).astype(np.float32)/PATCH


def warp(history:deque[torch.Tensor],flow:torch.Tensor)->deque[torch.Tensor]:
    if not history:return history
    rows,cols=torch.meshgrid(torch.arange(GRID,device=flow.device,dtype=torch.float32),torch.arange(GRID,device=flow.device,dtype=torch.float32),indexing='ij')
    x,y=cols+flow[...,0],rows+flow[...,1]; grid=torch.stack((x/(GRID-1)*2-1,y/(GRID-1)*2-1),-1).unsqueeze(0).expand(len(history),-1,-1,-1)
    stack=torch.stack(tuple(history)).permute(0,3,1,2); out=F.grid_sample(stack,grid,mode='bilinear',padding_mode='border',align_corners=True).permute(0,2,3,1).contiguous()
    return deque((out[i] for i in range(len(out))),maxlen=history.maxlen)


def overlay(frame:np.ndarray,score:np.ndarray,warming:bool)->np.ndarray:
    if warming: out,text=frame.copy(),'Canonical flow-aligned W_t vs W_t-1: warming up 300 frames'
    else:
        value=cv2.GaussianBlur(np.clip(score/FIXED_HEAT_MAX,0,1).astype(np.float32),(3,3),0); heat=cv2.applyColorMap(cv2.resize((value*255).astype(np.uint8),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO)
        out,text=cv2.addWeighted(frame,.55,heat,.45,0),f'Canonical flow + RoC W_t vs W_t-1 | fixed scale 0..{FIXED_HEAT_MAX:.0e}'
    cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,255,255),3,cv2.LINE_AA);cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.54,(0,0,0),1,cv2.LINE_AA);return out


def main()->None:
    a=parse_args(); cap=cv2.VideoCapture(str(a.video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 15.; cap.release()
    for p in (a.maps,a.flow,a.roc):p.parent.mkdir(parents=True,exist_ok=True)
    mt,ft,rt=(p.with_suffix('.partial.npy') for p in (a.maps,a.flow,a.roc))
    for p in (mt,ft,rt):p.unlink(missing_ok=True)
    maps=np.lib.format.open_memmap(mt,mode='w+',dtype=np.float16,shape=(total,GRID,GRID)); flows=np.lib.format.open_memmap(ft,mode='w+',dtype=np.float16,shape=(total-1,GRID,GRID,2)); rocs=np.lib.format.open_memmap(rt,mode='w+',dtype=np.float16,shape=(total,GRID,GRID))
    device=torch.device('cuda'); model=timm.create_model('vit_base_patch16_dinov3',pretrained=True,num_classes=0).eval().to(device); dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); history:deque[torch.Tensor]=deque(maxlen=a.window); prior=None; cap=cv2.VideoCapture(str(a.video)); index=0
    with torch.inference_mode():
        while index<total:
            frames=[]
            for _ in range(min(a.batch_size,total-index)):
                ok,frame=cap.read()
                if not ok:raise RuntimeError('source ended early')
                frames.append(frame)
            out=model.forward_features(model_input(frames,device))
            if isinstance(out,dict):out=out.get('x_norm_patchtokens',out.get('patch_tokens'))
            if out is None:raise RuntimeError('DINO returned no patch tokens')
            if out.shape[1]!=GRID*GRID:out=out[:,getattr(model,'num_prefix_tokens',out.shape[1]-GRID*GRID):]
            if out.shape[1:]!=(GRID*GRID,DIM):raise RuntimeError(f'unexpected patch-token shape {tuple(out.shape)}')
            for frame,current in zip(frames,out.reshape(len(frames),GRID,GRID,DIM)):
                current=current/(current.norm(dim=-1,keepdim=True)+1e-8); gray=cv2.cvtColor(cv2.resize(frame,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY); aligned=None
                if prior is not None:
                    f=patch_flow(gray,prior,dis); flows[index-1]=f.astype(np.float16); history=warp(history,torch.from_numpy(f).to(device)); aligned=history[-1]
                if len(history)==a.window:
                    stack=torch.stack(tuple(history)); previous=stack.mean(0); current_window=(stack.sum(0)-stack[0]+current)/a.window
                    maps[index]=(1-(current_window/(current_window.norm(dim=-1,keepdim=True)+1e-8)*previous/(previous.norm(dim=-1,keepdim=True)+1e-8)).sum(-1)).clamp_min(0).float().cpu().numpy().astype(np.float16)
                else: maps[index]=np.nan
                if aligned is None:rocs[index]=0
                else:rocs[index]=(1-(current*aligned/(aligned.norm(dim=-1,keepdim=True)+1e-8)).sum(-1)).clamp(0,2).float().cpu().numpy().astype(np.float16)
                history.append(current);prior=gray;index+=1
                if index%100==0 or index==total:print(f'encoded {index}/{total}',flush=True)
    cap.release();del maps,flows,rocs,model;torch.cuda.empty_cache();os.replace(mt,a.maps);os.replace(ft,a.flow);os.replace(rt,a.roc)
    maps=np.load(a.maps,mmap_mode='r');flows=np.load(a.flow,mmap_mode='r');rocs=np.load(a.roc,mmap_mode='r');ref=float(np.mean(rocs[a.window:],dtype=np.float64));cap=cv2.VideoCapture(str(a.video));w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));a.raw_video.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.raw_video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));ema=None
    for i in range(total):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('source ended during render')
        warm=i<a.window or not np.isfinite(maps[i]).all()
        if warm:score=np.zeros((GRID,GRID),np.float32)
        else:
            score=np.asarray(maps[i],np.float32)/(1+FLOW_ALPHA*np.linalg.norm(np.asarray(flows[i-1],np.float32),axis=-1));score*=np.asarray(rocs[i],np.float32)/max(ref,1e-8);ema=score if ema is None else DISPLAY_EMA*score+(1-DISPLAY_EMA)*ema;score=ema
        writer.write(overlay(frame,score,warm))
        if (i+1)%150==0 or i+1==total:print(f'rendered {i+1}/{total}',flush=True)
    cap.release();writer.release()


if __name__=='__main__':main()
