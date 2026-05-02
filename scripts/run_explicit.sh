#!/usr/bin/env bash
# Explicit-content (NSFW) unlearning pipeline (paper §4.3).
# Single concept "naked" on Stable Diffusion v1.4 + I2P benchmark.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="runs/explicit"

python main.py --config configs/explicit.yaml \
               --concepts "naked" \
               --output_dir "${RUN_DIR}"

python evaluation/eval_explicit.py --ckpt "${RUN_DIR}/edited_unet.safetensors" \
                                   --model_id CompVis/stable-diffusion-v1-4 \
                                   --out_dir "${RUN_DIR}/eval_i2p" \
                                   --max_prompts 300
