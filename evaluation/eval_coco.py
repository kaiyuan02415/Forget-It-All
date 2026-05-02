"""MS-COCO 30K CLIP score for measuring generative-quality preservation
(paper Tables 1, 4, 16: ``CLIPcoco``, FID).

Generates images for the first ``--max_prompts`` lines of ``coco_prompts.txt`` and
reports the mean CLIP image-text similarity. Optionally writes images to disk so an
external FID tool (e.g. ``pytorch-fid``) can be run separately.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from evaluation._common import load_edited_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompts_file", default="datasets/coco_prompts.txt")
    p.add_argument("--out_dir", default="runs/coco/eval")
    p.add_argument("--max_prompts", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def clip_score(images: list[Image.Image], texts: list[str], clip, proc, device):
    inputs = proc(text=texts, images=images, return_tensors="pt",
                  padding=True, truncation=True).to(device)
    out = clip(**inputs)
    image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return (image_emb * text_emb).sum(dim=-1).cpu().tolist()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()][: args.max_prompts]

    pipe = load_edited_pipeline(args.ckpt, model_id=args.model_id, device=args.device)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    sims: list[float] = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        torch.manual_seed(i); np.random.seed(i)
        with torch.no_grad():
            images = pipe(batch, num_inference_steps=50, guidance_scale=7.5).images
        for j, img in enumerate(images):
            img.save(os.path.join(args.out_dir, f"img_{i+j:05d}.jpg"))
        sims.extend(clip_score(images, batch, clip, proc, args.device))

    print(f"\n=== MS-COCO CLIP score over {len(sims)} prompts ===")
    print(f"  CLIPcoco = {100 * np.mean(sims):.2f}")
    print(f"  Images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
