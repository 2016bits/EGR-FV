#!/bin/sh
# Train + eval claim-evidence baseline under retrieved evidence on FEVER.
#
# Independent of the EGR pipeline; trains its own grounded model from scratch
# under outputs/FEVER/retrieved/baselines/claim_evidence/.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

CONFIG="${CONFIG:-configs/baseline_claim_evidence_fever_retrieved.yaml}"
SYM_CONFIG="${SYM_CONFIG:-configs/baseline_claim_evidence_fever_retrieved_symmetric_eval.yaml}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [ "$RUN_TRAIN" = "1" ]; then
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode train_claim_evidence
fi

if [ "$RUN_EVAL" = "1" ]; then
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode eval
  "$PYTHON_BIN" -m src.main --config "$SYM_CONFIG" --mode eval
fi
