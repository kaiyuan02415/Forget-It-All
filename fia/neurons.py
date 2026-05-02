"""Stage 2: Concept-Sensitive Neuron selection (paper §3.2, Eq.3-6).

Given a saliency tensor ``S[t][l] ∈ ℝ^{C_out × C_in}`` from :mod:`fia.saliency`, we
fold it into a single per-concept binary mask ``Q ∈ {0,1}^{C_out × C_in}`` per layer:

1. **Time-Integrated Sensitivity** (Eq.3): combine *response strength* and *activation
   persistence*::

       A_{ℓ,i,j} = ½ · (1/T) Σ_t S_{ℓ,t,i,j}
                 + ½ · (1/T) Σ_t 1[S_{ℓ,t,i,j} > τ_{ℓ,t}]

   The threshold ``τ_{ℓ,t}`` is chosen adaptively as the top-``r₁`` quantile of the
   per-(layer, timestep) saliency map, so the indicator stays informative regardless
   of timestep-dependent magnitude.

2. **Spatial Sparsity Selection** (Eq.4-6): take the **intersection** of two top-k
   selections in ``A``:

   * **Channel-level** ``C_ℓ``: per output channel *i*, keep the top
     ``k = r₂ · C_in`` input neurons.
   * **Layer-level** ``G_ℓ``: across the whole layer, keep the top
     ``K_g = r₂ · C_out · C_in`` entries.

   The final mask is ``Q_ℓ = C_ℓ ∩ G_ℓ`` — neurons that are both locally relevant
   for some output channel AND globally salient in the layer.
"""

from __future__ import annotations

import torch

from fia.utils import FIAConfig


def _quantile_threshold(x: torch.Tensor, ratio: float) -> float:
    """Return the value τ s.t. ``ratio`` fraction of entries exceed it (descending)."""
    flat = x.flatten()
    if flat.numel() == 0:
        return 0.0
    k = max(1, int(ratio * flat.numel()))
    top = torch.topk(flat, k=k, largest=True).values
    return top[-1].item()


def _time_integrated(saliency_layer: list[torch.Tensor], r1: float) -> torch.Tensor:
    """Eq.3: combine response strength and activation frequency over T timesteps."""

    T = len(saliency_layer)
    avg = torch.zeros_like(saliency_layer[0])
    freq = torch.zeros_like(saliency_layer[0])
    for t in range(T):
        s = saliency_layer[t]
        avg += s
        tau = _quantile_threshold(s, r1)
        freq += (s > tau).float()
    avg /= T
    freq /= T

    # Normalize avg to roughly the same scale as freq before averaging — otherwise
    # the magnitudes dominate the indicator term. We scale avg by its own max so
    # both terms live in [0, 1].
    avg_scale = avg.max().clamp_min(1e-12)
    avg = avg / avg_scale
    return 0.5 * avg + 0.5 * freq


def _channel_topk_mask(A: torch.Tensor, r2: float) -> torch.Tensor:
    """Eq.4: per-row (output-channel) top-k of A → binary mask."""

    C_out, C_in = A.shape
    k = max(1, int(r2 * C_in))
    _, idx = torch.topk(A, k=k, dim=1, largest=True)
    mask = torch.zeros_like(A, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _layer_topk_mask(A: torch.Tensor, r2: float) -> torch.Tensor:
    """Eq.5: layer-wide top-K_g of A → binary mask."""

    flat = A.flatten()
    k = max(1, int(r2 * flat.numel()))
    _, flat_idx = torch.topk(flat, k=k, largest=True)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[flat_idx] = True
    return mask.view_as(A)


def build_concept_mask(saliency: list[list[torch.Tensor]], cfg: FIAConfig
                       ) -> list[torch.Tensor]:
    """Convert per-(t, l) saliency tensors into a per-layer binary mask.

    Args:
        saliency: nested list of shape ``[T][L]`` with tensors of shape ``(C_out, C_in)``.
        cfg:      :class:`FIAConfig` (uses ``r1`` and ``r2``).

    Returns:
        A list of length ``L`` with bool tensors ``Q_ℓ`` of shape ``(C_out, C_in)``.
        ``True`` marks a neuron as **concept-sensitive** (a candidate for pruning in
        the fusion stage).
    """

    n_layers = len(saliency[0])
    masks: list[torch.Tensor] = []
    for l in range(n_layers):
        A = _time_integrated([saliency[t][l] for t in range(len(saliency))], cfg.r1)
        channel_mask = _channel_topk_mask(A, cfg.r2)  # Eq.4
        layer_mask = _layer_topk_mask(A, cfg.r2)      # Eq.5
        masks.append(channel_mask & layer_mask)       # Eq.6
    return masks


def mask_density(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())
