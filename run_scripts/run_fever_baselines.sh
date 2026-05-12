#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export PREPROCESS="${PREPROCESS:-auto}"

if [ "${RUN_CLAIM_ONLY:-1}" = "1" ]; then
  sh "$SCRIPT_DIR/run_claim_only_fever.sh"
fi

if [ "${RUN_CLAIM_EVIDENCE:-1}" = "1" ]; then
  sh "$SCRIPT_DIR/run_claim_evidence_fever.sh"
fi

echo "all FEVER baselines finished"
