#!/usr/bin/env python3
"""Two-minute 48x48-patch canonical run; all scoring rules match the default."""
from __future__ import annotations
import argparse, math, os
from collections import deque
from pathlib import Path
import cv2, numpy as np, timm, torch
import torch.nn.functional as F
from surprise_normalization import DEFAULT_DISPLAY_EMA, DEFAULT_FLOW_ALPHA, raw_global_display_scale

SIZE, PATCH, GRID, DIM = 768, 16, 48, 768
LOG2PI = math.log(2 * math.pi)

def args():
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('maps',type=Path);p.add_argument('flow',type=Path);p.add_argument('roc',type=Path);p.add_argument('raw_video',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--seconds',type=float,default=120);p.add_argument('--history',type=int,default=300);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--alpha',type=float,default=DEFAULT_FLOW_ALPHA);p.add_argument('--display-ema',type=float,default=DEFAULT_DISPLAY_EMA);return p.parse_args()

def input_tensor(frames,device):
    rgb=np.stack([cv2.cvtColor(cv2.resize(f,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for f in frames])
    x=torch.from_numpy(rgb).permute(0,3,1,2).float().div_(255);mean=torch.tensor((.485,.456,.406)).view(1,3,1,1);std=torch.tensor((.229,.224,.225)).view(1,3,1,1);return ((x-mean)/std).to(device)

def grid(flow):
    r,c=torch.meshgrid(torch.arange(GRID,device=flow.device,dtype=torch.float32),torch.arange(GRID,device=flow.device,dtype=torch.float32),indexing='ij');x,y=c+flow[...,0],r+flow[...,1];return torch.stack((x/(GRID-1)*2-1,y/(GRID-1)*2-1),-1)

def warp(history,flow):
    if not history:return history
    stack=torch.stack(tuple(history)).permute(0,3,1,2);g=grid(flow).unsqueeze(0).expand(len(history),-1,-1,-1);w=F.grid_sample(stack,g,mode='bilinear',padding_mode='border',align_corners=True).permute(0,2,3,1).contiguous();return deque((w[i] for i in range(len(w))),maxlen=history.maxlen)

def overlay(frame,score,lo,hi,warming):
    if warming:out=frame.copy();text='48x48 canonical: warming up 300-frame history'
    else:
        v=cv2.GaussianBlur(np.clip((score-lo)/max(hi-lo,1e-6),0,1).astype(np.float32),(3,3),0);heat=cv2.applyColorMap(cv2.resize((v*255).astype(np.uint8),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO);out=cv2.addWeighted(frame,.55,heat,.45,0);text='48x48 canonical: flow + patch RoC + display EMA'
    cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),3,cv2.LINE_AA);cv2.putText(out,text,(18,36),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),1,cv2.LINE_AA);return out

def main():
    a=args();device=torch.device('cuda');cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS) or 15.;total=min(int(round(fps*a.seconds)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)));cap.release()
    for p in (a.maps,a.flow,a.roc):p.parent.mkdir(parents=True,exist_ok=True)
    mt,ft,rt=(p.with_suffix('.partial.npy') for p in (a.maps,a.flow,a.roc));[p.unlink(missing_ok=True) for p in (mt,ft,rt)]
    maps=np.lib.format.open_memmap(mt,mode='w+',dtype=np.float16,shape=(total,GRID,GRID));flows=np.lib.format.open_memmap(ft,mode='w+',dtype=np.float16,shape=(total-1,GRID,GRID,2));rocs=np.lib.format.open_memmap(rt,mode='w+',dtype=np.float16,shape=(total,GRID,GRID))
    model=timm.create_model('vit_base_patch16_dinov3',pretrained=True,num_classes=0).eval().to(device);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);history=deque(maxlen=a.history);prior=None;cap=cv2.VideoCapture(str(a.video));index=0
    with torch.inference_mode():
        while index<total:
            frames=[]
            for _ in range(min(a.batch_size,total-index)):
                ok,f=cap.read()
                if not ok:raise RuntimeError('video ended early')
                frames.append(f)
            tokens=model.forward_features(input_tensor(frames,device));
            if isinstance(tokens,dict):tokens=tokens.get('x_norm_patchtokens',tokens.get('patch_tokens'))
            tokens=tokens[:,getattr(model,'num_prefix_tokens',0):].reshape(len(frames),GRID,GRID,DIM)
            for frame,current in zip(frames,tokens):
                gray=cv2.cvtColor(cv2.resize(frame,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY);aligned=None
                if prior is not None:
                    dense=np.nan_to_num(dis.calc(gray,prior,None));pf=dense.reshape(GRID,PATCH,GRID,PATCH,2).mean((1,3)).astype(np.float32)/PATCH;flows[index-1]=pf.astype(np.float16);history=warp(history,torch.from_numpy(pf).to(device));aligned=history[-1]
                if len(history)==a.history:
                    s=torch.stack(tuple(history));mean,var=s.mean(0),s.var(0,unbiased=False).clamp_min(1e-4);maps[index]=(0.5*((current-mean).square()/var+torch.log(var)+LOG2PI)).mean(-1).float().cpu().numpy().astype(np.float16)
                else:maps[index]=np.nan
                if aligned is None:rocs[index]=0
                else:
                    c=current/(current.norm(dim=-1,keepdim=True)+1e-8);q=aligned/(aligned.norm(dim=-1,keepdim=True)+1e-8);rocs[index]=(1-(c*q).sum(-1)).clamp(0,2).float().cpu().numpy().astype(np.float16)
                history.append(current);prior=gray;index+=1
                if index%50==0 or index==total:print(f'computed {index}/{total}',flush=True)
    cap.release();del maps,flows,rocs,model;torch.cuda.empty_cache();os.replace(mt,a.maps);os.replace(ft,a.flow);os.replace(rt,a.roc)
    maps=np.load(a.maps,mmap_mode='r');flows=np.load(a.flow,mmap_mode='r');rocs=np.load(a.roc,mmap_mode='r');valid=np.asarray(maps[a.history:],np.float32);lo,hi=raw_global_display_scale(valid[np.isfinite(valid)]);ref=float(np.mean(rocs[a.history:],dtype=np.float64));print(f'scale={lo:.4f}/{hi:.4f}; roc_ref={ref:.5f}',flush=True)
    cap=cv2.VideoCapture(str(a.video));w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));a.raw_video.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.raw_video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));ema=None
    for i in range(total):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('render ended early')
        warm=i<a.history
        if warm:score=np.zeros((GRID,GRID),np.float32)
        else:
            score=np.asarray(maps[i],np.float32)/(1+a.alpha*np.linalg.norm(np.asarray(flows[i-1],np.float32),axis=-1));score*=np.asarray(rocs[i],np.float32)/max(ref,1e-8);ema=score if ema is None else a.display_ema*score+(1-a.display_ema)*ema;score=ema
        writer.write(overlay(frame,score,lo,hi,warm))
        if (i+1)%150==0 or i+1==total:print(f'rendered {i+1}/{total}',flush=True)
    cap.release();writer.release()
if __name__=='__main__':main()
