"""Stage 3: Multi-Concept Mask Fusion (paper §3.3, Eq.7-8).

Given a per-concept binary mask ``Q^{(c)}_ℓ`` for every concept *c* (from
:mod:`fia.neurons`), we fuse them into one mask per layer that we can multiply into
the UNet's FFN₂ weights.

* Per-concept mask (Eq.7)::

      Mask_ℓ^{(c)}(i, j) = 1 if (i, j) ∈ Q_ℓ^{(c)} else 0

* Concept-agnostic detection (Eq.8). Sum over concepts::

      s_{ℓ,i,j} = Σ_c Mask_ℓ^{(c)}(i, j)

  and define a **concept-agnostic threshold** ``τ_ca = ⌈α · C⌉`` where *C* is the
  number of target concepts and ``α ∈ (0, 1]`` is the *concept-agnostic ratio*.
  Neurons with ``s ≥ τ_ca`` are considered concept-agnostic — they support most
  concepts and therefore encode generic visual features. Such neurons are
  **preserved**. Truly concept-specific neurons (``0 < s < τ_ca``) are pruned.

The final pruning mask satisfies ``prune_ℓ(i, j) = 1`` iff that neuron should be
zeroed, and the unlearned UNet uses ``W ← W ⊙ (1 − prune)``.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import torch

from fia.utils import FIAConfig, ensure_dir, ffn2_layers


def fuse_masks(per_concept_masks: Sequence[Sequence[torch.Tensor]],
               alpha: float) -> tuple[list[torch.Tensor], dict]:
    """Combine per-concept masks into a single per-layer pruning mask.

    Args:
        per_concept_masks: outer length ``C``, inner length ``L``; each entry is a
            ``(C_out, C_in)`` bool tensor produced by :func:`fia.neurons.build_concept_mask`.
        alpha: concept-agnostic ratio (paper Eq.8).

    Returns:
        ``(prune_masks, stats)`` where ``prune_masks[l]`` is a bool tensor of shape
        ``(C_out, C_in)`` (True ⇒ prune), and ``stats`` reports per-layer densities.
    """

    C = len(per_concept_masks)
    if C == 0:
        raise ValueError("Need at least one concept mask")
    L = len(per_concept_masks[0])
    tau_ca = math.ceil(alpha * C)

    prune_masks: list[torch.Tensor] = []
    stats = {"n_concepts": C, "tau_ca": tau_ca, "alpha": alpha, "layers": []}

    for l in range(L):
        s = torch.zeros_like(per_concept_masks[0][l], dtype=torch.int32)
        for c in range(C):
            s = s + per_concept_masks[c][l].int()

        agnostic = s >= tau_ca           # preserved
        sensitive = (s > 0) & ~agnostic  # pruned
        prune_masks.append(sensitive)

        stats["layers"].append(
            dict(
                layer=l,
                shape=tuple(s.shape),
                agnostic=int(agnostic.sum()),
                pruned=int(sensitive.sum()),
                untouched=int((s == 0).sum()),
                pruned_ratio=float(sensitive.float().mean()),
            )
        )

    total = sum(L["shape"][0] * L["shape"][1] for L in stats["layers"])
    pruned = sum(L["pruned"] for L in stats["layers"])
    agnostic = sum(L["agnostic"] for L in stats["layers"])
    stats["overall_pruned_ratio"] = pruned / total
    stats["overall_agnostic_ratio"] = agnostic / total
    return prune_masks, stats


@torch.no_grad()
def apply_masks(unet, prune_masks: Sequence[torch.Tensor]) -> None:
    """Zero out neurons in FFN₂ weight matrices according to the fused mask.

    Implements ``W ← W ⊙ (1 − prune)`` from the paper. Modifies ``unet`` in place.
    """

    layers = ffn2_layers(unet)
    if len(layers) != len(prune_masks):
        raise ValueError(f"Got {len(prune_masks)} masks for {len(layers)} FFN₂ layers")
    for (name, module), mask in zip(layers, prune_masks):
        if mask.shape != module.weight.shape:
            raise ValueError(
                f"{name}: mask shape {tuple(mask.shape)} != weight shape "
                f"{tuple(module.weight.shape)}"
            )
        keep = (~mask).to(module.weight.dtype).to(module.weight.device)
        module.weight.mul_(keep)


def fuse_and_apply(unet, per_concept_masks: Sequence[Sequence[torch.Tensor]],
                   cfg: FIAConfig) -> dict:
    """One-shot Stage 3 helper. Returns the fusion statistics dict."""

    prune_masks, stats = fuse_masks(per_concept_masks, cfg.alpha)
    apply_masks(unet, prune_masks)
    return stats


def save_unet(unet, path: str) -> None:
    """Save the edited UNet weights to ``path``.

    Uses safetensors if the extension is ``.safetensors``, otherwise vanilla
    ``torch.save``. The state dict is sufficient to reload via
    ``UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')`` and
    ``unet.load_state_dict(state_dict)``.
    """

    ensure_dir(os.path.dirname(path) or ".")
    state = {k: v.detach().cpu() for k, v in unet.state_dict().items()}
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(state, path)
    else:
        torch.save(state, path)
