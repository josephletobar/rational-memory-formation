#!/usr/bin/env python3
"""Cache actual DINOv3 frame-global (CLS-style prefix) tokens for a segment."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import timm
import torch

SIZE=384; DIM=768
def args():
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('output',type=Path);p.add_argument('--start-frame',type=int,required=True);p.add_argument('--end-frame',type=int,required=True);p.add_argument('--batch-size',type=int,default=12);return p.parse_args()
def tensor(frames,device):
    rgb=np.stack([cv2.cvtColor(cv2.resize(x,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for x in frames])
    x=torch.from_numpy(rgb).permute(0,3,1,2).float().div_(255.)
    return ((x-torch.tensor((.485,.456,.406)).view(1,3,1,1))/torch.tensor((.229,.224,.225)).view(1,3,1,1)).to(device)
def main():
    a=args();n=a.end_frame-a.start_frame;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): print(f'exists: {a.output}');return
    tmp=a.output.with_suffix('.partial.npy');tmp.unlink(missing_ok=True);cache=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=(n,DIM))
    model=timm.create_model('vit_base_patch16_dinov3',pretrained=True,num_classes=0).eval().to(device)
    cap=cv2.VideoCapture(str(a.video));cap.set(cv2.CAP_PROP_POS_FRAMES,a.start_frame);i=0
    while i<n:
        frames=[]
        for _ in range(min(a.batch_size,n-i)):
            ok,f=cap.read()
            if not ok:raise RuntimeError('video ended early')
            frames.append(f)
        with torch.inference_mode():
            tokens=model.forward_features(tensor(frames,device))
            if isinstance(tokens,dict): tokens=tokens.get('x_norm_clstoken',tokens.get('cls_token'))
            else: tokens=tokens[:,0]
            if tokens.shape[1]!=DIM:raise RuntimeError(f'unexpected global-token shape {tuple(tokens.shape)}')
        cache[i:i+len(frames)]=tokens.float().cpu().numpy().astype(np.float16);i+=len(frames);print(f'global tokens {i}/{n}',flush=True)
    cap.release();del cache;os.replace(tmp,a.output)
if __name__=='__main__':main()
