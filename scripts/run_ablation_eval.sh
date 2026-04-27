#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

python -m src.main --config configs/remix.yaml --mode eval --ckpt outputs/checkpoints/remix_best.pt
python -m src.main --config configs/ablation_two_branch.yaml --mode eval --ckpt outputs/checkpoints/ablation_two_branch/remix_best.pt
python -m src.main --config configs/ablation_routing_only.yaml --mode eval --ckpt outputs/checkpoints/ablation_routing_only/remix_best.pt
python -m src.main --config configs/ablation_real_remix_no_weight.yaml --mode eval --ckpt outputs/checkpoints/ablation_real_remix_no_weight/remix_best.pt
python -m src.main --config configs/ablation_remix_random.yaml --mode eval --ckpt outputs/checkpoints/ablation_remix_random/remix_best.pt
python -m src.main --config configs/ablation_remix_heuristic.yaml --mode eval --ckpt outputs/checkpoints/ablation_remix_heuristic/remix_best.pt
python -m src.main --config configs/ablation_no_fusion.yaml --mode eval --ckpt outputs/checkpoints/ablation_no_fusion/remix_best.pt
python -m src.main --config configs/ablation_no_orth.yaml --mode eval --ckpt outputs/checkpoints/ablation_no_orth/remix_best.pt

python scripts/summarize_ablation_results.py
