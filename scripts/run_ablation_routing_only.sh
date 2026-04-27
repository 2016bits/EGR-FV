#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
python -m src.main \
  --config configs/ablation_routing_only.yaml \
  --mode remix \
  --shortcut_ckpt outputs/checkpoints/shortcut_best.pt \
  --grounded_ckpt outputs/checkpoints/grounded_best.pt

echo "finish ablation routing only"