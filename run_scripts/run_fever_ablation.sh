#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
export RUN_ROUTING="${RUN_ROUTING:-1}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export RUN_SUMMARY="${RUN_SUMMARY:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

BASE_ROUTING="data/FEVER/routed/train_routing.jsonl"
HARD_ROUTING="data/FEVER/routed/train_routing_hard.jsonl"
IN_SAMPLE_ROUTING="data/FEVER/routed/train_routing_in_sample.jsonl"

checkpoint_for_config() {
  case "$1" in
    *fever_ablation_routing_only*) echo "outputs/FEVER/checkpoints/routing_only/remix_best.pt" ;;
    *fever_ablation_full_wo_evidence_contrast*) echo "outputs/FEVER/checkpoints/full_wo_evidence_contrast/remix_best.pt" ;;
    *fever_ablation_full_wo_grounded_dominant*) echo "outputs/FEVER/checkpoints/full_wo_grounded_dominant/remix_best.pt" ;;
    *fever_ablation_full_wo_remix*) echo "outputs/FEVER/checkpoints/full_wo_remix/remix_best.pt" ;;
    *fever_ablation_full_hard_routing*) echo "outputs/FEVER/checkpoints/full_hard_routing/remix_best.pt" ;;
    *fever_ablation_full_in_sample_routing*) echo "outputs/FEVER/checkpoints/full_in_sample_routing/remix_best.pt" ;;
    *) echo "" ;;
  esac
}

ensure_routing_variants() {
  if [ ! -f "$BASE_ROUTING" ]; then
    echo "missing base routing cache: $BASE_ROUTING" >&2
    exit 1
  fi

  if [ ! -f "$HARD_ROUTING" ]; then
    if ls data/FEVER/routed/train_routing.fold*.jsonl >/dev/null 2>&1; then
      "$PYTHON_BIN" scripts/materialize_routing_variant.py \
        --source-glob "data/FEVER/routed/train_routing.fold*.jsonl" \
        --output "$HARD_ROUTING" \
        --stats-output "data/FEVER/routed/train_routing_hard.stats.json" \
        --variant hard
    else
      "$PYTHON_BIN" scripts/materialize_routing_variant.py \
        --source "$BASE_ROUTING" \
        --output "$HARD_ROUTING" \
        --stats-output "data/FEVER/routed/train_routing_hard.stats.json" \
        --variant hard
    fi
  fi

  if [ "$RUN_ROUTING" = "1" ] && [ ! -f "$IN_SAMPLE_ROUTING" ]; then
    CONFIG=configs/fever_ablation_full_in_sample_routing.yaml sh "$SCRIPT_DIR/run_routing.sh"
  fi
}

train_config() {
  config=$1
  ckpt=$(checkpoint_for_config "$config")
  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
    echo "skip existing checkpoint: $ckpt"
    return 0
  fi
  if [ "$config" = "configs/fever_ablation_full_hard_routing.yaml" ] && [ ! -f "$HARD_ROUTING" ]; then
    echo "skip hard routing training because $HARD_ROUTING is missing" >&2
    return 0
  fi
  if [ "$config" = "configs/fever_ablation_full_in_sample_routing.yaml" ] && [ ! -f "$IN_SAMPLE_ROUTING" ]; then
    echo "skip in-sample routing training because $IN_SAMPLE_ROUTING is missing" >&2
    return 0
  fi
  CONFIG="$config" sh "$SCRIPT_DIR/run_remix.sh"
}

eval_config() {
  config=$1
  ckpt=$(checkpoint_for_config "$config")
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "skip eval for $config because $ckpt is missing" >&2
    return 0
  fi
  "$PYTHON_BIN" -m src.main --config "$config" --mode eval
}

TRAIN_CONFIGS="
configs/fever_ablation_routing_only.yaml
configs/fever_ablation_full_wo_evidence_contrast.yaml
configs/fever_ablation_full_wo_grounded_dominant.yaml
configs/fever_ablation_full_wo_remix.yaml
configs/fever_ablation_full_hard_routing.yaml
configs/fever_ablation_full_in_sample_routing.yaml
"

EVAL_CONFIGS="
configs/fever_ablation_routing_only.yaml
configs/fever_ablation_routing_only_symmetric.yaml
configs/fever_ablation_full_wo_evidence_contrast.yaml
configs/fever_ablation_full_wo_evidence_contrast_symmetric.yaml
configs/fever_ablation_full_wo_grounded_dominant.yaml
configs/fever_ablation_full_wo_grounded_dominant_symmetric.yaml
configs/fever_ablation_full_wo_remix.yaml
configs/fever_ablation_full_wo_remix_symmetric.yaml
configs/fever_ablation_full_hard_routing.yaml
configs/fever_ablation_full_hard_routing_symmetric.yaml
configs/fever_ablation_full_in_sample_routing.yaml
configs/fever_ablation_full_in_sample_routing_symmetric.yaml
"

if [ "$RUN_ROUTING" = "1" ] || [ "$RUN_TRAIN" = "1" ]; then
  ensure_routing_variants
fi

if [ "$RUN_TRAIN" = "1" ]; then
  for config in $TRAIN_CONFIGS; do
    train_config "$config"
  done
fi

if [ "$RUN_EVAL" = "1" ]; then
  for config in $EVAL_CONFIGS; do
    eval_config "$config"
  done
fi

if [ "$RUN_SUMMARY" = "1" ]; then
  "$PYTHON_BIN" scripts/summarize_fever_ablation_results.py
fi

echo "finish FEVER ablation workflow"
