"""Top-level CLI for FIA.

Examples:

    # multi-object: simultaneously forget the 10 ImageNette classes
    python main.py --config configs/object.yaml \
                   --concepts "parachute,golf ball,garbage truck,cassette player,church,tench,french horn,gas pump,english springer,chain saw" \
                   --output_dir runs/object_10

    # explicit content: SD v1.4, single concept "naked"
    python main.py --config configs/explicit.yaml --concepts naked \
                   --output_dir runs/explicit

    # multi-artist style on SD v1.5
    python main.py --config configs/art.yaml \
                   --concepts "Van Gogh,Monet,Pablo Picasso,Leonardo Da Vinci,Salvador Dali" \
                   --output_dir runs/art_5
"""

from __future__ import annotations

import argparse

from fia.pipeline import run_fia
from fia.utils import FIAConfig


def parse_args():
    p = argparse.ArgumentParser(description="Forget-It-All multi-concept unlearning")
    p.add_argument("--config", required=True, help="YAML config (configs/*.yaml)")
    p.add_argument("--concepts", required=True,
                   help="comma-separated list of concepts to forget")
    p.add_argument("--output_dir", default=None,
                   help="override config's output_dir")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_save_intermediate", action="store_true",
                   help="skip dumping per-concept saliency tensors")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = FIAConfig.from_yaml(args.config)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if not concepts:
        raise SystemExit("No concepts provided")

    run_fia(concepts, cfg, save_intermediate=not args.no_save_intermediate)


if __name__ == "__main__":
    main()
