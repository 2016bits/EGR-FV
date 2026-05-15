#!/bin/sh
set -eu

# Train the two PolitiHop ablations needed to fill the evidence-perturbation
# analysis table: `full_wo_evidence_contrast` and `full_wo_remix`. The routing
# cache is assumed to be already produced by `run_politihop.sh`.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BASE_ROUTING="data/PolitiHop/routed/train_routing.jsonl"

checkpoint_for_config() {
  case "$1" in
    *politihop_ablation_full_wo_evidence_contrast*) echo "outputs/PolitiHop/checkpoints/full_wo_evidence_contrast/remix_best.pt" ;;
    *politihop_ablation_full_wo_remix*)             echo "outputs/PolitiHop/checkpoints/full_wo_remix/remix_best.pt" ;;
    *) echo "" ;;
  esac
}

train_config() {
  config=$1
  ckpt=$(checkpoint_for_config "$config")
  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
    echo "skip existing checkpoint: $ckpt"
    return 0
  fi
  CONFIG="$config" sh "$SCRIPT_DIR/run_remix.sh"
}

if [ ! -f "$BASE_ROUTING" ]; then
  echo "missing PolitiHop routing cache: $BASE_ROUTING" >&2
  echo "Run run_scripts/run_politihop.sh first (it produces train_routing.jsonl)." >&2
  exit 1
fi

TRAIN_CONFIGS="
configs/politihop_ablation_full_wo_evidence_contrast.yaml
configs/politihop_ablation_full_wo_remix.yaml
"

for config in $TRAIN_CONFIGS; do
  train_config "$config"
done

echo "finish PolitiHop ablation training"
