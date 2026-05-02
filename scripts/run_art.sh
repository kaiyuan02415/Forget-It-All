#!/usr/bin/env bash
# Multi-artist style unlearning pipeline (paper §4.4).

set -euo pipefail
cd "$(dirname "$0")/.."

ARTISTS="Van Gogh,Monet,Pablo Picasso,Leonardo Da Vinci,Salvador Dali"
RUN_DIR="runs/art_5"

python main.py --config configs/art.yaml \
               --concepts "${ARTISTS}" \
               --output_dir "${RUN_DIR}"

python evaluation/eval_artist.py --ckpt "${RUN_DIR}/edited_unet.safetensors" \
                                 --artists "${ARTISTS}" \
                                 --out_dir "${RUN_DIR}/eval_artist"

python evaluation/eval_coco.py   --ckpt "${RUN_DIR}/edited_unet.safetensors" \
                                 --prompts_file datasets/coco_prompts.txt \
                                 --out_dir "${RUN_DIR}/eval_coco" \
                                 --max_prompts 200
