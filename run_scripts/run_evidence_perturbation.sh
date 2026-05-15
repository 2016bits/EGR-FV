#!/bin/sh
set -eu

# Run the evidence-perturbation evaluation for every (dataset, method) pair
# requested in the analysis table:
#
#   methods : claim-evidence | w/o contrast | w/o remix | EGR-FV
#   datasets: HOVER | FEVER | PolitiHop
#
# For each pair we load the corresponding checkpoint, evaluate the grounded
# branch under clean / shuffled / null / wrong_evidence conditions, and dump a
# JSON report. The aggregated markdown/CSV table is produced by
# `scripts/summarize_evidence_perturbation.py` at the end.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/evidence_perturbation}"
DATASETS="${DATASETS:-HOVER FEVER PolitiHop}"
METHODS="${METHODS:-claim-evidence wo-contrast wo-remix EGR-FV}"

# Map (dataset, method) -> (config_path, ckpt_kind, ckpt_path, method_display).
# We use the same configs that already exist in the repo where possible. For
# methods that don't have a per-dataset ablation config (e.g. PolitiHop), we
# fall back to the dataset's main config.
resolve_method() {
  dataset=$1
  method=$2

  case "$dataset" in
    HOVER)
      base_config="configs/default.yaml"
      ckpt_root="outputs/HOVER/checkpoints"
      ;;
    FEVER)
      base_config="configs/fever.yaml"
      ckpt_root="outputs/FEVER/checkpoints"
      ;;
    PolitiHop)
      base_config="configs/politihop.yaml"
      ckpt_root="outputs/PolitiHop/checkpoints"
      ;;
    *)
      echo "unknown dataset: $dataset" >&2
      return 1
      ;;
  esac

  case "$method" in
    claim-evidence)
      CFG="$base_config"
      KIND="grounded"
      CKPT="$ckpt_root/grounded_best.pt"
      DISPLAY="claim-evidence"
      ;;
    wo-contrast)
      case "$dataset" in
        HOVER)     CFG="configs/ablation_full_wo_evidence_contrast.yaml" ;;
        FEVER)     CFG="configs/fever_ablation_full_wo_evidence_contrast.yaml" ;;
        PolitiHop) CFG="configs/politihop_ablation_full_wo_evidence_contrast.yaml" ;;
        *)         CFG="$base_config" ;;
      esac
      KIND="remix"
      CKPT="$ckpt_root/full_wo_evidence_contrast/remix_best.pt"
      DISPLAY="w/o contrast"
      ;;
    wo-remix)
      case "$dataset" in
        HOVER)     CFG="configs/ablation_full_wo_remix.yaml" ;;
        FEVER)     CFG="configs/fever_ablation_full_wo_remix.yaml" ;;
        PolitiHop) CFG="configs/politihop_ablation_full_wo_remix.yaml" ;;
        *)         CFG="$base_config" ;;
      esac
      KIND="remix"
      CKPT="$ckpt_root/full_wo_remix/remix_best.pt"
      DISPLAY="w/o remix"
      ;;
    EGR-FV)
      CFG="$base_config"
      KIND="remix"
      CKPT="$ckpt_root/remix_best.pt"
      DISPLAY="EGR-FV"
      ;;
    *)
      echo "unknown method: $method" >&2
      return 1
      ;;
  esac
  if [ ! -f "$CFG" ]; then
    # Fall back to the dataset's main config if the ablation-specific one
    # does not exist for this dataset.
    CFG="$base_config"
  fi
  export CFG KIND CKPT DISPLAY
}

run_one() {
  dataset=$1
  method=$2
  resolve_method "$dataset" "$method"
  out_dir="$OUTPUT_ROOT/$dataset"
  mkdir -p "$out_dir"
  out_path="$out_dir/${method}.json"

  if [ ! -f "$CKPT" ]; then
    echo "[skip] $dataset / $method: checkpoint not found at $CKPT" >&2
    return 0
  fi
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_path" ]; then
    echo "[skip] $dataset / $method: existing report at $out_path"
    return 0
  fi

  echo "[run]  $dataset / $method -> $out_path"
  "$PYTHON_BIN" scripts/eval_evidence_perturbation.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --ckpt-kind "$KIND" \
    --dataset-name "$dataset" \
    --method-name "$DISPLAY" \
    --output "$out_path"
}

for dataset in $DATASETS; do
  for method in $METHODS; do
    run_one "$dataset" "$method"
  done
done

echo
echo "Aggregating reports..."
"$PYTHON_BIN" scripts/summarize_evidence_perturbation.py \
  --root "$OUTPUT_ROOT" \
  --output-prefix evidence_perturbation

echo "Done. See $OUTPUT_ROOT/evidence_perturbation_report.md"
