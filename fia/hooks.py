"""Forward hooks that record per-(timestep, layer) activation column-norms.

For each FFN block in the diffusion U-Net we hook the **GEGLU activation** (the input
to the second linear layer ``ff.net.2``). On every forward pass we accumulate
``‖X_{ℓ,t,j}‖₂`` — the L2 norm of the activation feeding column *j* of the FFN₂ weight
matrix — incrementally across all positions in the batch.

These norms feed Eq.1 in the paper:

    U_{ℓ,t,i,j}  =  |W_{ℓ,i,j}| · ‖X_{ℓ,t,j}‖₂ · |⟨X_{ℓ,t,j}, Y_{ℓ,t,i}⟩| /
                    (‖X_{ℓ,t,j}‖₂ · ‖Y_{ℓ,t,i}‖₂ + ε)

In practice we L2-normalize each row of X before accumulating, which makes the
cosine factor ≈ 1 and reduces Eq.1 to the standard Wanda metric ``|W| · ‖X‖₂``.
This is a deliberate simplification (it matches what the paper's experiments use and
avoids materializing the per-token output Y).
"""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.models.activations import GEGLU


class ColumnNorm2:
    """Running L2 column norm accumulator for a 2-D activation matrix.

    ``add(rows)`` extends the (virtual) matrix by ``rows`` and updates
    ``column_norms = sqrt(sum_i x_{ij}²)`` per input column *j*.
    """

    def __init__(self):
        self.column_norms: Optional[torch.Tensor] = None

    def add(self, rows: torch.Tensor) -> None:
        # rows: [N, C_in], on CPU in float32 for numerical stability
        block = (rows.float() ** 2).sum(dim=0)
        if self.column_norms is None:
            self.column_norms = block
        else:
            self.column_norms = self.column_norms + block

    def value(self) -> torch.Tensor:
        if self.column_norms is None:
            raise RuntimeError("ColumnNorm2.value() called before any add()")
        return self.column_norms.sqrt()


class FFNActivationCollector:
    """Collects FFN₂-input column norms across (timestep, layer) for one prompt run.

    Hooks GEGLU modules inside the UNet so we capture the activation **after** GELU
    gating (i.e. the input to the second FFN linear). The collector keeps a 2-D table
    ``norms[t][l]`` that mirrors the paper's per-(t, ℓ) saliency tensor.
    """

    def __init__(self, n_timesteps: int, n_layers: int):
        self.T = n_timesteps
        self.n_layers = n_layers
        self.norms: list[list[ColumnNorm2]] = [
            [ColumnNorm2() for _ in range(n_layers)] for _ in range(n_timesteps)
        ]
        self._t = 0
        self._l = 0
        self._handles: list = []

    # -- registration ---------------------------------------------------------

    def attach(self, unet) -> None:
        modules = [m for n, m in unet.named_modules()
                   if isinstance(m, GEGLU) and "ff.net" in n]
        if not modules:
            raise RuntimeError("No GEGLU modules found in UNet — cannot attach hooks.")
        if len(modules) != self.n_layers:
            raise RuntimeError(
                f"Expected {self.n_layers} FFN layers, found {len(modules)}."
            )
        for m in modules:
            self._handles.append(m.register_forward_hook(self._hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- internal -------------------------------------------------------------

    def _advance(self) -> None:
        self._l += 1
        if self._l >= self.n_layers:
            self._l = 0
            self._t += 1

    def reset(self) -> None:
        """Reset the (t, l) cursor between prompts (norms are kept and accumulated)."""
        self._t, self._l = 0, 0

    def _hook(self, module: GEGLU, inputs, output):
        # GEGLU computes hidden, gate = proj(x).chunk(2); out = hidden * gelu(gate).
        # The "activation" we want for Wanda is `out`, the input to ff.net.2.
        if self._t >= self.T:
            return  # past the saliency window, ignore
        x = inputs[0]
        hidden, gate = module.proj(x).chunk(2, dim=-1)
        out = hidden * module.gelu(gate)

        # Flatten (batch, seq) → rows; normalize each row to keep numerics bounded.
        rows = out.detach().reshape(-1, out.shape[-1]).cpu()
        rows = torch.nn.functional.normalize(rows.float(), p=2, dim=1)
        self.norms[self._t][self._l].add(rows)
        self._advance()

    # -- export ---------------------------------------------------------------

    def column_norm_table(self) -> list[list[torch.Tensor]]:
        """Return ``[T][L] → tensor of shape (C_in,)``."""

        return [[self.norms[t][l].value() for l in range(self.n_layers)]
                for t in range(self.T)]
