#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
python -m src.main \
  --config configs/routing.yaml \
  --mode routing \
  --shortcut_ckpt outputs/checkpoints/shortcut_best.pt \
  --grounded_ckpt outputs/checkpoints/grounded_best.pt
