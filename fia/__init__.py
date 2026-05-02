"""FIA: Forget-It-All — multi-concept machine unlearning for diffusion models.

The package mirrors the paper's three-stage pipeline:

* :mod:`fia.saliency`     — Stage 1: contrastive concept saliency (paper Eq.1, 2)
* :mod:`fia.neurons`      — Stage 2: time + spatial sparsity selection (Eq.3-6)
* :mod:`fia.mask_fusion`  — Stage 3: per-concept mask fusion + apply to UNet (Eq.7, 8)

The high-level driver is :func:`fia.run_fia`, which is also exposed via
``python -m fia`` / ``python main.py``.
"""

from fia.saliency import compute_concept_saliency
from fia.neurons import build_concept_mask
from fia.mask_fusion import fuse_and_apply, save_unet
from fia.pipeline import run_fia

__all__ = [
    "compute_concept_saliency",
    "build_concept_mask",
    "fuse_and_apply",
    "save_unet",
    "run_fia",
]
