#!/bin/sh
set -eu

# Run the three supplementary "is routing meaningful?" analyses on every
# dataset that has a full set of warm-up + EGR-FV checkpoints:
#
#   (1) per-group performance (bias_easy / hard / grounded_needed) for
#       claim-only / claim-evidence / EGR-FV with the gain column.
#   (2) Pearson / Spearman correlation between necessity score r_i and
#       evidence sensitivity Δ_i (indicator and NLL).
#   (3) Routing visualization: clean-correct / shuffled-wrong / null-wrong
#       rate vs. necessity-score bucket.
#
# Per-sample JSONL files are written under
#   outputs/analysis/routing_analysis/<dataset>.jsonl
# and the aggregated tables + figure under the same directory.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/routing_analysis}"
DATASETS="${DATASETS:-HOVER FEVER PolitiHop}"

resolve_paths() {
  dataset=$1
  case "$dataset" in
    HOVER)
      CFG="configs/default.yaml"
      CKPT_ROOT="outputs/HOVER/checkpoints"
      ;;
    FEVER)
      CFG="configs/fever.yaml"
      CKPT_ROOT="outputs/FEVER/checkpoints"
      ;;
    PolitiHop)
      CFG="configs/politihop.yaml"
      CKPT_ROOT="outputs/PolitiHop/checkpoints"
      ;;
    *)
      echo "unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
  SHORTCUT_CKPT="$CKPT_ROOT/shortcut_best.pt"
  GROUNDED_CKPT="$CKPT_ROOT/grounded_best.pt"
  EGRFV_CKPT="$CKPT_ROOT/remix_best.pt"
  export CFG SHORTCUT_CKPT GROUNDED_CKPT EGRFV_CKPT
}

run_one() {
  dataset=$1
  resolve_paths "$dataset"

  out_dir="$OUTPUT_ROOT"
  mkdir -p "$out_dir"
  records_out="$out_dir/$dataset.jsonl"
  summary_out="$out_dir/$dataset.summary.json"

  for ckpt in "$SHORTCUT_CKPT" "$GROUNDED_CKPT" "$EGRFV_CKPT"; do
    if [ ! -f "$ckpt" ]; then
      echo "[skip] $dataset: checkpoint missing at $ckpt" >&2
      return 0
    fi
  done

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$records_out" ]; then
    echo "[skip] $dataset: existing per-sample records at $records_out"
    return 0
  fi

  echo "[run]  $dataset -> $records_out"
  "$PYTHON_BIN" scripts/eval_routing_analysis.py \
    --config "$CFG" \
    --shortcut-ckpt "$SHORTCUT_CKPT" \
    --grounded-ckpt "$GROUNDED_CKPT" \
    --egrfv-ckpt "$EGRFV_CKPT" \
    --dataset-name "$dataset" \
    --output "$records_out" \
    --summary-output "$summary_out"
}

for dataset in $DATASETS; do
  run_one "$dataset"
done

echo
echo "Aggregating routing-analysis reports..."
"$PYTHON_BIN" scripts/summarize_routing_analysis.py \
  --root "$OUTPUT_ROOT" \
  --datasets $DATASETS \
  --output-prefix routing_analysis

echo "Done. See $OUTPUT_ROOT/routing_analysis_report.md"
