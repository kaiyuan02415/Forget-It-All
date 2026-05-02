"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import os
import sys

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def _add_project_root_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)


_add_project_root_to_path()


def load_edited_pipeline(ckpt_path: str | None,
                         model_id: str = "runwayml/stable-diffusion-v1-5",
                         device: str = "cuda",
                         dtype: torch.dtype = torch.float16
                         ) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline with optional FIA-edited UNet weights.

    If ``ckpt_path`` is None, returns the unmodified base model (useful as the
    "original SD" reference).
    """

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)

    if ckpt_path is not None:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location="cpu")
        # Either we saved just the UNet (state has UNet keys) or a full pipeline.
        first_key = next(iter(state))
        if first_key.startswith("unet."):
            state = {k[len("unet."):]: v for k, v in state.items() if k.startswith("unet.")}
        pipe.unet.load_state_dict(state, strict=False)

    return pipe.to(device)
