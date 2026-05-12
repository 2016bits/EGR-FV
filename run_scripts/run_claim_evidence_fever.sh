#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

TRAIN_CONFIG="${CLAIM_EVIDENCE_TRAIN_CONFIG:-${TRAIN_CONFIG:-configs/baseline_claim_evidence_fever.yaml}}"
FEVER_EVAL_CONFIG="${CLAIM_EVIDENCE_FEVER_EVAL_CONFIG:-${FEVER_EVAL_CONFIG:-configs/baseline_claim_evidence_fever.yaml}}"
SYMMETRIC_EVAL_CONFIG="${CLAIM_EVIDENCE_SYMMETRIC_EVAL_CONFIG:-${SYMMETRIC_EVAL_CONFIG:-configs/baseline_claim_evidence_fever_symmetric.yaml}}"
RUN_TRAIN="${RUN_TRAIN:-1}"
PREPROCESS="${PREPROCESS:-auto}"

need_preprocess=0
for data_file in \
  data/FEVER/converted_data/train.json \
  data/FEVER/converted_data/dev.json \
  data/FEVER/converted_data/test.json \
  data/FEVER/converted_data/symmetric.json; do
  if [ ! -f "$data_file" ]; then
    need_preprocess=1
  fi
done

if [ "$PREPROCESS" = "1" ]; then
  "$PYTHON_BIN" scripts/convert_fever.py --splits train dev test symmetric
elif [ "$PREPROCESS" = "auto" ] && [ "$need_preprocess" = "1" ]; then
  "$PYTHON_BIN" scripts/convert_fever.py --splits train dev test symmetric
fi

if [ "$RUN_TRAIN" = "1" ]; then
  "$PYTHON_BIN" -m src.main --config "$TRAIN_CONFIG" --mode train_claim_evidence
fi

"$PYTHON_BIN" -m src.main --config "$FEVER_EVAL_CONFIG" --mode eval
"$PYTHON_BIN" -m src.main --config "$SYMMETRIC_EVAL_CONFIG" --mode eval

echo "claim-evidence FEVER baseline finished"
echo "FEVER report: outputs/FEVER/baselines/claim_evidence/predictions/fever/eval_report.json"
echo "symmetric-FEVER report: outputs/FEVER/baselines/claim_evidence/predictions/symmetric_fever/eval_report.json"
