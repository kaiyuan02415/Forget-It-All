"""Configuration loading, model loading, and small helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Iterable

import torch
import yaml
from diffusers import StableDiffusionPipeline


@dataclass
class FIAConfig:
    """Hyperparameters and runtime settings for an FIA run.

    Field names map to the paper's notation:

    * :attr:`r1`   — temporal sparsity ratio (paper §3.2, Eq.3)
    * :attr:`r2`   — spatial sparsity ratio (Eq.4-5)
    * :attr:`alpha`— concept-agnostic ratio (Eq.8)
    * :attr:`saliency_steps` — number of denoising steps used for saliency (paper §B.2
      recommends the first 10 of 50)
    """

    task: str = "object"  # one of {"object", "explicit", "art"}
    model_id: str = "runwayml/stable-diffusion-v1-5"
    seed: int = 1244
    dtype: str = "float16"

    # Diffusion sampling
    inference_steps: int = 50
    saliency_steps: int = 10

    # FIA hyperparameters
    r1: float = 0.05
    r2: float = 0.01
    alpha: float = 0.6
    # Optional per-concept r2 override, e.g. {"parachute": 0.03}
    r2_overrides: dict = field(default_factory=dict)

    # I/O
    output_dir: str = "runs/fia"
    device: str = "cuda"

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "FIAConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.dtype]

    def r2_for(self, concept: str) -> float:
        return float(self.r2_overrides.get(concept, self.r2))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_pipeline(cfg: FIAConfig) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline with the safety checker disabled.

    The safety checker is removed because FIA must observe activations on every prompt,
    including I2P prompts that the SD safety filter would otherwise blank.
    """

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id, torch_dtype=cfg.torch_dtype, safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(cfg.device)


def ffn2_layers(unet) -> list[tuple[str, torch.nn.Linear]]:
    """Return UNet FFN₂ linear layers in deterministic (sorted) order.

    These are the modules pruned by FIA: paper Table 13 / 10 shows FFN₂ is the optimal
    target. There are 16 such layers in SD v1.4 / v1.5 UNets.
    """

    layers: list[tuple[str, torch.nn.Linear]] = []
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear) and "ff.net" in name and "proj" not in name:
            layers.append((name, module))
    layers.sort(key=lambda kv: kv[0])
    return layers


def set_seed(seed: int) -> None:
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunked(it: Iterable, n: int):
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
