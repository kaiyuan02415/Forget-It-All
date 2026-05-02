"""Stage 1: Contrastive Concept Saliency  (paper §3.1, Eq.1, Eq.2).

For each target concept *c* we run the diffusion model on:

* ``concept_prompts`` — ``"a golf ball on the table"`` style strings, and
* ``base_prompts``    — the same context with the concept removed (``"a table"``).

We hook each FFN₂ layer and accumulate activation column-norms across all prompts
and the first ``saliency_steps`` denoising timesteps. Then per (output channel *i*,
input neuron *j*) we form the **unified saliency**::

    U_{ℓ,t,i,j}  =  |W_{ℓ,i,j}| · ‖X_{ℓ,t,j}‖₂

(See :mod:`fia.hooks` for the simplification of Eq.1.) Finally, the **contrastive
concept saliency** combines concept-prompt and base-prompt statistics::

    S_{ℓ,t,i,j} = max(0, μ_c − μ_b − σ_b)              (paper Eq.2)

where ``μ_c`` and ``μ_b`` are the mean of ``U`` over the concept- and base-prompt
batches, and ``σ_b`` is the standard deviation under the base prompts. Subtracting
``σ_b`` filters out neurons whose baseline activation is just noisy.

Output: a saliency tensor ``S[t][l] ∈ ℝ^{C_out × C_in}`` for every (timestep, layer).
"""

from __future__ import annotations

from typing import Sequence

import torch
import tqdm

from fia.hooks import FFNActivationCollector
from fia.utils import FIAConfig, ensure_dir, ffn2_layers, set_seed


def _absolute_weights(unet) -> dict[str, torch.Tensor]:
    return {n: m.weight.detach().abs().cpu().float()
            for n, m in ffn2_layers(unet)}


@torch.no_grad()
def _collect_norms(pipe, prompts: Sequence[str], n_layers: int, cfg: FIAConfig):
    """Run the diffusion pipeline once per prompt, returning per-(t, l, j) norm stats.

    Returns ``mean[t][l][j]`` and ``std[t][l][j]`` — the running mean and std of
    ‖X_{ℓ,t,j}‖₂ across prompts. Each prompt contributes one norm per (t, ℓ, j).
    """

    means: list[list[torch.Tensor | None]] = [[None] * n_layers for _ in range(cfg.saliency_steps)]
    sq_means: list[list[torch.Tensor | None]] = [[None] * n_layers for _ in range(cfg.saliency_steps)]
    count = 0

    for prompt in tqdm.tqdm(prompts, desc="prompts", leave=False):
        collector = FFNActivationCollector(cfg.saliency_steps, n_layers)
        collector.attach(pipe.unet)

        set_seed(cfg.seed)
        try:
            pipe(
                prompt,
                num_inference_steps=cfg.inference_steps,
                guidance_scale=7.5,
                output_type="latent",
            )
        finally:
            collector.detach()

        norms = collector.column_norm_table()  # [T][L] → (C_in,)
        count += 1
        for t in range(cfg.saliency_steps):
            for l in range(n_layers):
                v = norms[t][l]
                means[t][l] = v.clone() if means[t][l] is None else means[t][l] + v
                sq = v * v
                sq_means[t][l] = sq.clone() if sq_means[t][l] is None else sq_means[t][l] + sq

    if count == 0:
        raise RuntimeError("No prompts processed — empty saliency stats.")

    out_mean: list[list[torch.Tensor]] = [[None] * n_layers for _ in range(cfg.saliency_steps)]
    out_std: list[list[torch.Tensor]] = [[None] * n_layers for _ in range(cfg.saliency_steps)]
    for t in range(cfg.saliency_steps):
        for l in range(n_layers):
            mu = means[t][l] / count
            var = (sq_means[t][l] / count) - mu * mu
            out_mean[t][l] = mu
            out_std[t][l] = var.clamp_min(0).sqrt()
    return out_mean, out_std  # per-(t, l) tensors of shape (C_in,)


def compute_concept_saliency(pipe, concept_prompts: Sequence[str],
                             base_prompts: Sequence[str], cfg: FIAConfig
                             ) -> list[list[torch.Tensor]]:
    """Compute Eq.1+Eq.2 saliency S[t][l] of shape ``(C_out, C_in)`` for one concept.

    Args:
        pipe:             a Stable Diffusion pipeline (UNet inside is hooked).
        concept_prompts:  prompts that contain the target concept.
        base_prompts:     matched prompts that describe only the surroundings.
        cfg:              :class:`FIAConfig`.

    Returns:
        A nested list ``S`` with shape ``[T][L]`` whose entries are CPU float tensors of
        shape ``(C_out, C_in)``. ``S[t][l][i, j] = max(0, μ_c − μ_b − σ_b)`` weighted by
        the absolute weight ``|W_{ℓ,i,j}|`` (Eq.1). All non-concept-specific neurons
        clip to zero.
    """

    abs_w = _absolute_weights(pipe.unet)  # {layer_name: (C_out, C_in) abs weights}
    layer_names = sorted(abs_w.keys())
    n_layers = len(layer_names)

    mu_c, _ = _collect_norms(pipe, concept_prompts, n_layers, cfg)
    mu_b, sigma_b = _collect_norms(pipe, base_prompts, n_layers, cfg)

    saliency: list[list[torch.Tensor]] = []
    for t in range(cfg.saliency_steps):
        row: list[torch.Tensor] = []
        for l in range(n_layers):
            # Eq.2 — contrastive: how much does the concept lift the activation
            # above the base mean, beyond the base's own noise floor?
            contrast = (mu_c[t][l] - mu_b[t][l] - sigma_b[t][l]).clamp_min(0.0)
            # Eq.1 — broadcast onto every output channel via |W|.
            U = abs_w[layer_names[l]] * contrast.unsqueeze(0)
            row.append(U)
        saliency.append(row)
    return saliency


def save_saliency(saliency: list[list[torch.Tensor]], path: str) -> None:
    ensure_dir(path)
    payload = {(t, l): saliency[t][l]
               for t in range(len(saliency)) for l in range(len(saliency[0]))}
    torch.save(payload, f"{path}/saliency.pt")


def load_saliency(path: str) -> list[list[torch.Tensor]]:
    payload = torch.load(f"{path}/saliency.pt", map_location="cpu")
    T = max(t for t, _ in payload) + 1
    L = max(l for _, l in payload) + 1
    return [[payload[(t, l)] for l in range(L)] for t in range(T)]
