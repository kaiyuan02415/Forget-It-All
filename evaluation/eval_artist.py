"""Artist-style unlearning evaluation (paper §4.4, Tables 4-5).

For each (prompt, seed) in ``test_<artist>.csv``, generate paired images with the
original SD and the FIA-edited model. Compute:

* ``CLIPₐ``: CLIP similarity between the **edited** image and the artist's name.
  Lower means the style is more thoroughly removed.
* ``FSR`` (Forget-Success Rate): fraction of prompts where the edited image's CLIP
  similarity to the artist drops below the original's.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from evaluation._common import load_edited_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="FIA-edited UNet checkpoint")
    p.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--datasets_dir", default="datasets")
    p.add_argument("--artists", default="Van Gogh,Monet,Pablo Picasso,Leonardo Da Vinci,Salvador Dali")
    p.add_argument("--out_dir", default="runs/art/eval")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def clip_similarity(images: list[Image.Image], texts: list[str], clip, proc, device):
    inputs = proc(text=texts, images=images, return_tensors="pt",
                  padding=True, truncation=True).to(device)
    out = clip(**inputs)
    image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return (image_emb * text_emb).sum(dim=-1).cpu().tolist()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    artists = [a.strip() for a in args.artists.split(",") if a.strip()]
    edited = load_edited_pipeline(args.ckpt, model_id=args.model_id, device=args.device)
    original = load_edited_pipeline(None, model_id=args.model_id, device=args.device)

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("\n=== Artist-style CLIP scores (lower CLIPₐ = more forgotten) ===")
    for artist in artists:
        csv_path = os.path.join(args.datasets_dir, f"test_{artist}.csv")
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)

        edited_imgs, original_imgs, prompts = [], [], []
        for _, row in df.iterrows():
            prompt = row["prompt"]
            seed = int(row["evaluation_seed"])

            torch.manual_seed(seed); np.random.seed(seed)
            edited_imgs.append(edited(prompt, num_inference_steps=50,
                                       guidance_scale=7.5).images[0])
            torch.manual_seed(seed); np.random.seed(seed)
            original_imgs.append(original(prompt, num_inference_steps=50,
                                          guidance_scale=7.5).images[0])
            prompts.append(prompt)

        sims_edit = clip_similarity(edited_imgs, [artist] * len(prompts),
                                    clip, proc, args.device)
        sims_orig = clip_similarity(original_imgs, [artist] * len(prompts),
                                    clip, proc, args.device)
        clip_a = np.mean(sims_edit) * 100
        fsr = np.mean([e < o for e, o in zip(sims_edit, sims_orig)]) * 100
        print(f"  {artist:<20s} CLIPₐ={clip_a:6.2f}  FSR={fsr:6.2f}%")


if __name__ == "__main__":
    main()
