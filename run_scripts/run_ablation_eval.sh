#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

python -m src.main --config configs/remix.yaml --mode eval
python -m src.main --config configs/ablation_two_branch.yaml --mode eval
python -m src.main --config configs/ablation_routing_only.yaml --mode eval
python -m src.main --config configs/ablation_real_remix_no_weight.yaml --mode eval
python -m src.main --config configs/ablation_remix_random.yaml --mode eval
python -m src.main --config configs/ablation_remix_heuristic.yaml --mode eval
python -m src.main --config configs/ablation_no_fusion.yaml --mode eval
python -m src.main --config configs/ablation_no_orth.yaml --mode eval

python scripts/summarize_ablation_results.py
