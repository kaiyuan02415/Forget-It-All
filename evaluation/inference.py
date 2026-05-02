"""Visual sanity check: generate paired (original, edited) images for a prompt.

    python evaluation/inference.py --ckpt runs/object_10/edited_unet.safetensors \
                                   --prompt "a photo of a french horn" -n 4
"""

from __future__ import annotations

import argparse
import os

import torch

from evaluation._common import load_edited_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--out_dir", default="runs/inference")
    p.add_argument("-n", "--num_images", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.join(args.out_dir, "base")
    edit_dir = os.path.join(args.out_dir, "edited")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(edit_dir, exist_ok=True)

    base = load_edited_pipeline(None, model_id=args.model_id, device=args.device)
    edit = load_edited_pipeline(args.ckpt, model_id=args.model_id, device=args.device)

    for i in range(args.num_images):
        seed = args.seed + i
        gen_b = torch.Generator(device=args.device).manual_seed(seed)
        gen_e = torch.Generator(device=args.device).manual_seed(seed)
        img_b = base(args.prompt, generator=gen_b, num_inference_steps=50,
                     guidance_scale=7.5).images[0]
        img_e = edit(args.prompt, generator=gen_e, num_inference_steps=50,
                     guidance_scale=7.5).images[0]
        img_b.save(os.path.join(base_dir, f"{i:02d}.png"))
        img_e.save(os.path.join(edit_dir, f"{i:02d}.png"))
        print(f"  [{i}] saved {base_dir}/{i:02d}.png & {edit_dir}/{i:02d}.png")


if __name__ == "__main__":
    main()
