"""ImageNette object-unlearning evaluation (paper §4.2, Table 1/2).

For each ImageNette class, generate ``--n_per_class`` images with the FIA-edited
model and classify them with a pre-trained ResNet-50. Report per-class **forgetting
accuracy** (lower is better for forgotten concepts) and **preserving accuracy**
(higher is better for retained concepts).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from evaluation._common import load_edited_pipeline


IMAGENETTE_CLASSES = [
    "garbage truck", "cassette player", "tench", "english springer", "chain saw",
    "parachute", "golf ball", "church", "french horn", "gas pump",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="FIA-edited UNet checkpoint")
    p.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--csv", default="datasets/imagenette.csv",
                   help="CSV with columns: prompt, evaluation_seed, class")
    p.add_argument("--out_dir", default="runs/object/eval")
    p.add_argument("--n_per_class", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--forget", default=",".join(IMAGENETTE_CLASSES),
                   help="comma-separated forget set (rest are preserved)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    forget_set = {c.strip().lower() for c in args.forget.split(",") if c.strip()}

    df = pd.read_csv(args.csv)
    label_col = "class" if "class" in df.columns else "label_str"
    df[label_col] = df[label_col].str.lower()

    pipe = load_edited_pipeline(args.ckpt, model_id=args.model_id, device=args.device)

    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights).to(args.device).eval()
    preprocess = weights.transforms()

    correct = defaultdict(int)
    total = defaultdict(int)
    for cls in IMAGENETTE_CLASSES:
        sub = df[df[label_col] == cls.lower()].head(args.n_per_class)
        for _, row in sub.iterrows():
            seed = int(row["evaluation_seed"])
            torch.manual_seed(seed)
            np.random.seed(seed)
            with torch.no_grad():
                image = pipe(row["prompt"], num_inference_steps=50,
                             guidance_scale=7.5).images[0]
            inp = preprocess(image).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred = classifier(inp).argmax(dim=1).item()
            pred_label = weights.meta["categories"][pred].lower()
            total[cls] += 1
            if cls in pred_label or pred_label in cls:
                correct[cls] += 1

    print("\n=== ImageNette per-class accuracy ===")
    forget_accs, preserve_accs = [], []
    for cls in IMAGENETTE_CLASSES:
        acc = correct[cls] / max(total[cls], 1) * 100
        bucket = "forget" if cls.lower() in forget_set else "preserve"
        print(f"  {cls:<20s} {bucket:>8s} acc={acc:6.2f}%  ({correct[cls]}/{total[cls]})")
        (forget_accs if bucket == "forget" else preserve_accs).append(acc)

    print()
    if forget_accs:
        print(f"  Avg forgetting accuracy ↓ : {np.mean(forget_accs):6.2f}%")
    if preserve_accs:
        print(f"  Avg preserving accuracy ↑ : {np.mean(preserve_accs):6.2f}%")
    if forget_accs and preserve_accs:
        F = np.mean(forget_accs) / 100
        P = np.mean(preserve_accs) / 100
        if (P + (1 - F)) > 0:
            harmonic = 2 * P * (1 - F) / (P + (1 - F)) * 100
            print(f"  Overall score (harmonic) ↑: {harmonic:6.2f}%")


if __name__ == "__main__":
    main()
