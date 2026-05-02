"""End-to-end FIA pipeline: saliency → mask → fusion → save."""

from __future__ import annotations

import json
import os
from typing import Sequence

from fia.mask_fusion import fuse_and_apply, save_unet
from fia.neurons import build_concept_mask, mask_density
from fia.prompts import build_pair_prompts
from fia.saliency import compute_concept_saliency, save_saliency
from fia.utils import FIAConfig, ensure_dir, load_pipeline


def run_fia(concepts: Sequence[str], cfg: FIAConfig,
            *, save_intermediate: bool = True) -> str:
    """Run the full FIA pipeline for a list of concepts.

    Args:
        concepts: target concepts to forget (e.g. ``["parachute", "golf ball", ...]``).
        cfg:      :class:`FIAConfig`.
        save_intermediate: if True, dump per-concept saliency tensors to disk under
            ``cfg.output_dir/saliency/<concept>/``.

    Returns:
        Path to the saved unlearned UNet (``cfg.output_dir/edited_unet.safetensors``).
    """

    out = ensure_dir(cfg.output_dir)
    cfg.to_yaml(os.path.join(out, "config.yaml"))

    pipe = load_pipeline(cfg)

    per_concept_masks = []
    for concept in concepts:
        print(f"\n=== Concept: {concept} (r2={cfg.r2_for(concept):.4f}) ===")
        c_prompts, b_prompts = build_pair_prompts(cfg.task, concept, seed=cfg.seed)
        saliency = compute_concept_saliency(pipe, c_prompts, b_prompts, cfg)

        if save_intermediate:
            save_saliency(saliency, os.path.join(out, "saliency", concept.replace(" ", "_")))

        # Build the concept-sensitive mask for this concept using its own r₂.
        cfg_concept = FIAConfig(**{**cfg.__dict__, "r2": cfg.r2_for(concept)})
        masks = build_concept_mask(saliency, cfg_concept)
        densities = [mask_density(m) for m in masks]
        print(f"   per-layer mask densities: {[f'{d:.4f}' for d in densities]}")
        per_concept_masks.append(masks)

    print("\n=== Fusion ===")
    stats = fuse_and_apply(pipe.unet, per_concept_masks, cfg)
    print(f"   pruned={stats['overall_pruned_ratio']:.4%}  "
          f"agnostic={stats['overall_agnostic_ratio']:.4%}  τ_ca={stats['tau_ca']}")

    with open(os.path.join(out, "fusion_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    ckpt = os.path.join(out, "edited_unet.safetensors")
    save_unet(pipe.unet, ckpt)
    print(f"\nSaved unlearned UNet to {ckpt}")
    return ckpt
