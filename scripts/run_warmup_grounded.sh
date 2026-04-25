#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
python -m src.main \
  --config configs/grounded.yaml \
  --mode warmup_grounded
