#!/usr/bin/env bash
# Multi-object unlearning pipeline (paper §4.2).
# Forgets all 10 ImageNette classes simultaneously, then evaluates the resulting
# UNet on the ImageNette classifier and on MS-COCO 30K.

set -euo pipefail
cd "$(dirname "$0")/.."

CONCEPTS="parachute,golf ball,garbage truck,cassette player,church,tench,french horn,gas pump,english springer,chain saw"
RUN_DIR="runs/object_10"

python main.py --config configs/object.yaml \
               --concepts "${CONCEPTS}" \
               --output_dir "${RUN_DIR}"

python evaluation/eval_object.py --ckpt "${RUN_DIR}/edited_unet.safetensors" \
                                 --csv  datasets/imagenette.csv \
                                 --out_dir "${RUN_DIR}/eval_imagenette"

python evaluation/eval_coco.py   --ckpt "${RUN_DIR}/edited_unet.safetensors" \
                                 --prompts_file datasets/coco_prompts.txt \
                                 --out_dir "${RUN_DIR}/eval_coco" \
                                 --max_prompts 200
