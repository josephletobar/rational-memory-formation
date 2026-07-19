"""Validate that the selected V-JEPA2 checkpoint can encode a clip length."""

import argparse

import torch
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()

    model_id = "facebook/vjepa2-vitl-fpc32-256-diving48"
    print(f"loading {model_id}", flush=True)
    model = AutoModel.from_pretrained(model_id)
    encoder = model.vjepa2 if hasattr(model, "vjepa2") else model
    encoder.eval().cuda()
    print(f"testing {args.frames} frames", flush=True)
    pixels = torch.zeros((1, args.frames, 3, 256, 256), device="cuda")
    with torch.inference_mode():
        output = encoder(pixel_values_videos=pixels, skip_predictor=True)
    print(
        f"success shape={tuple(output.last_hidden_state.shape)} "
        f"peak_bytes={torch.cuda.max_memory_allocated()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
