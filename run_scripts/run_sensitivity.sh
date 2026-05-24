#!/bin/sh
# Sensitivity / ablation sweep driver for HOVER (Tier-Should EMNLP supplementary experiments).
#
# Covers four sweep groups:
#   - necessity:    necessity-score component weights (5 variants)
#   - threshold:    routing thresholds 0.6/0.4 and 0.8/0.2 (2 variants)
#   - remix_ratio:  remix sampler distribution_preserving_ratio (3 variants)
#   - loss_weight:  alpha / beta (routing-stage) + lambda_contrast (trainer-side) (6 variants)
#
# For variants that change necessity-score / threshold / alpha / beta the script
# first calls scripts/materialize_routing_variant.py to (re)build a
# data/HOVER/routed/sensitivity/<variant>.jsonl cache offline (no GPU),
# then runs `--mode remix` and `--mode eval`. Variants that only change the
# remix sampler or trainer loss weights reuse the baseline routing.jsonl.
#
# Usage:
#   sh run_scripts/run_sensitivity.sh
#   SWEEPS=remix_ratio sh run_scripts/run_sensitivity.sh
#   SWEEPS="necessity threshold" sh run_scripts/run_sensitivity.sh
#   SKIP_EXISTING=0 sh run_scripts/run_sensitivity.sh   # force re-run
#   RUN_SUMMARIZE=0 sh run_scripts/run_sensitivity.sh   # skip summarizer step

set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

SWEEPS="${SWEEPS:-necessity threshold remix_ratio loss_weight}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_SUMMARIZE="${RUN_SUMMARIZE:-1}"

SOURCE_ROUTING="${SOURCE_ROUTING:-data/HOVER/routed/train_routing.jsonl}"
SENSITIVITY_DIR="${SENSITIVITY_DIR:-data/HOVER/routed/sensitivity}"
mkdir -p "$SENSITIVITY_DIR"

# variant_name | sweep_group | materialize_extra_args
# materialize_extra_args is a single shell-safe token list. Empty means "no
# materialization required" (variant reuses the baseline routing.jsonl).
VARIANTS_SPEC='
necessity_only_grounded     | necessity   | --necessity-weights grounded_correct=1,shortcut_wrong=0,disagreement=0,evidence_length=0
necessity_only_shortcut     | necessity   | --necessity-weights grounded_correct=0,shortcut_wrong=1,disagreement=0,evidence_length=0
necessity_only_disagreement | necessity   | --necessity-weights grounded_correct=0,shortcut_wrong=0,disagreement=1,evidence_length=0
necessity_only_evlen        | necessity   | --necessity-weights grounded_correct=0,shortcut_wrong=0,disagreement=0,evidence_length=1
necessity_equal             | necessity   | --necessity-weights grounded_correct=0.25,shortcut_wrong=0.25,disagreement=0.25,evidence_length=0.25
threshold_60_40             | threshold   | --grounded-threshold 0.6 --bias-threshold 0.4
threshold_80_20             | threshold   | --grounded-threshold 0.8 --bias-threshold 0.2
remix_less_grounded         | remix_ratio |
remix_more_grounded         | remix_ratio |
remix_equal                 | remix_ratio |
alpha_025                   | loss_weight | --alpha 0.25
alpha_200                   | loss_weight | --alpha 2.0
beta_050                    | loss_weight | --beta 0.5
beta_200                    | loss_weight | --beta 2.0
lambda_contrast_010         | loss_weight |
lambda_contrast_040         | loss_weight |
'

sweep_requested() {
  target="$1"
  for sweep in $SWEEPS; do
    if [ "$sweep" = "$target" ]; then
      return 0
    fi
  done
  return 1
}

materialize_variant() {
  variant="$1"; shift
  extra_args="$*"
  output="$SENSITIVITY_DIR/${variant}.jsonl"

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$output" ]; then
    echo "[skip] materialize $variant — $output already exists"
    return 0
  fi

  echo "[run]  materialize $variant"
  # shellcheck disable=SC2086
  "$PYTHON_BIN" scripts/materialize_routing_variant.py \
    --variant necessity \
    --source "$SOURCE_ROUTING" \
    --output "$output" \
    --overwrite \
    $extra_args
}

run_stage() {
  config="$1"; mode="$2"; expected="$3"

  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$expected" ] && [ -f "$expected" ]; then
    echo "[skip] $config :: $mode — $expected already exists"
    return 0
  fi

  echo "[run]  $config :: $mode"
  "$PYTHON_BIN" -m src.main --config "$config" --mode "$mode"
}

run_variant() {
  variant="$1"; sweep_group="$2"; shift 2
  extra_args="$*"

  if ! sweep_requested "$sweep_group"; then
    return 0
  fi

  config="configs/ablations/sensitivity/${variant}.yaml"
  if [ ! -f "$config" ]; then
    echo "[error] missing config: $config" >&2
    return 1
  fi

  echo
  echo "=== sweep=$sweep_group variant=$variant ==="

  if [ -n "$extra_args" ]; then
    # shellcheck disable=SC2086
    materialize_variant "$variant" $extra_args
  fi

  ckpt_dir="outputs/HOVER/checkpoints/sensitivity/${variant}"
  pred_dir="outputs/HOVER/predictions/sensitivity/${variant}"
  run_stage "$config" remix "$ckpt_dir/remix_best.pt"
  run_stage "$config" eval  "$pred_dir/eval_report.json"
}

# Parse VARIANTS_SPEC and dispatch.
printf '%s\n' "$VARIANTS_SPEC" | while IFS='|' read -r variant sweep_group materialize_args; do
  variant=$(printf '%s' "$variant" | sed 's/^ *//;s/ *$//')
  sweep_group=$(printf '%s' "$sweep_group" | sed 's/^ *//;s/ *$//')
  materialize_args=$(printf '%s' "$materialize_args" | sed 's/^ *//;s/ *$//')
  [ -z "$variant" ] && continue
  # shellcheck disable=SC2086
  run_variant "$variant" "$sweep_group" $materialize_args
done

if [ "$RUN_SUMMARIZE" = "1" ]; then
  echo
  echo "=== summarizing sensitivity ==="
  "$PYTHON_BIN" scripts/summarize_sensitivity.py
fi

echo
echo "sensitivity sweep finished"
