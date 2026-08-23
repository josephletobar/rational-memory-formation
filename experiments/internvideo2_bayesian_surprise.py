"""Annotate a video with frame-wise InternVideo2 Bayesian surprise.

Each frame is embedded independently.  The score is the negative log density
under a diagonal Gaussian fit to the preceding 64 frame embeddings.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoModel
from torchvision import transforms

MODEL_ID = "OpenGVLab/InternVideo2_CLIP_S"

def embed(model, transform, frame, device):
    x = transform(frame).unsqueeze(0).unsqueeze(1).to(device)  # B,T,C,H,W; one frame
    with torch.inference_mode():
        if hasattr(model, "encode_vision"):
            z = model.encode_vision(x, test=True)
        else:
            out = model(x)
            z = getattr(out, "pooler_output", None) or out.last_hidden_state[:, 0]
        if isinstance(z, (tuple, list)): z = z[0]
        if z.ndim > 2: z = z.mean(dim=tuple(range(1, z.ndim - 1)))
    return torch.nn.functional.normalize(z.float(), dim=-1)[0].cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--history", type=int, default=64)
    p.add_argument("--model", default=MODEL_ID)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).eval().to(device)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    hist, scores = [], []
    while True:
        ok, frame = cap.read()
        if not ok: break
        z = embed(model, transform, frame, device)
        if len(hist) < 2:
            score = 0.0
        else:
            H = np.asarray(hist[-a.history:], dtype=np.float32)
            mu, var = H.mean(0), H.var(0) + 1e-4
            score = float(0.5 * np.mean((z - mu) ** 2 / var + np.log(var)))
        hist.append(z); scores.append(score)
        recent = np.asarray(scores[-min(len(scores), a.history):])
        baseline = float(np.median(recent)) + 1e-6
        scaled = np.clip(score / baseline, 0, 9.99)
        color = (0, 0, 255) if scaled > 2 else (0, 180, 255) if scaled > 1 else (0, 220, 0)
        cv2.rectangle(frame, (0, 0), (w, 72), (0, 0, 0), -1)
        cv2.putText(frame, f"InternVideo2 surprise: {score:.3f}  ({scaled:.2f}x recent median)", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
        cv2.putText(frame, f"history: {min(len(hist)-1, a.history)} frames", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        out.write(frame)
    cap.release(); out.release()
    np.savez_compressed(str(a.output.with_suffix(".npz")), scores=np.asarray(scores), fps=fps, history=a.history)
    print(f"wrote {a.output} ({len(scores)} frames)")

if __name__ == "__main__": main()
