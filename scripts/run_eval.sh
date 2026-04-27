#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
python -m src.main \
  --config configs/remix.yaml \
  --mode eval \
  --ckpt outputs/checkpoints/remix_best.pt

echo "finish remix evaluation"