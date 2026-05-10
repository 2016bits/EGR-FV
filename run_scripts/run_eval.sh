#!/bin/sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

for config in \
  configs/ablation_grounded_only.yaml \
  configs/ablation_two_branch.yaml \
  configs/ablation_routing_only.yaml \
  configs/ablation_random_remix_only.yaml \
  configs/ablation_full_wo_remix.yaml \
  configs/ablation_full_wo_evidence_contrast.yaml \
  configs/ablation_full_wo_grounded_dominant.yaml \
  configs/ablation_full_hard_routing.yaml \
  configs/ablation_full_in_sample_routing.yaml \
  configs/ablation_fusion_inference.yaml \
  configs/remix.yaml; do
  "$PYTHON_BIN" -m src.main --config "$config" --mode eval
done

"$PYTHON_BIN" scripts/summarize_ablation_results.py
