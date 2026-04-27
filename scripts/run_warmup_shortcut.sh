#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
python -m src.main \
  --config configs/shortcut.yaml \
  --mode warmup_shortcut

echo "finish warmup shortcut training"