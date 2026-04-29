#!/bin/sh
# set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

"$PYTHON_BIN" -m src.main --config configs/remix.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_two_branch.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_routing_only.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_real_remix_no_weight.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_remix_random.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_remix_heuristic.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_no_fusion.yaml --mode eval
"$PYTHON_BIN" -m src.main --config configs/ablation_no_orth.yaml --mode eval

"$PYTHON_BIN" scripts/summarize_ablation_results.py
