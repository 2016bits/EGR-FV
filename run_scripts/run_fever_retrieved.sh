#!/bin/sh
# Train + eval EGR-FV under retrieved evidence on FEVER.
#
# Reuses every stage of run_fever.sh but with retrieved-evidence configs.
# Outputs land under outputs/FEVER/retrieved/.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

CONFIG="${CONFIG:-configs/fever_retrieved.yaml}"
SYM_CONFIG="${SYM_CONFIG:-configs/fever_retrieved_symmetric.yaml}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [ "$RUN_TRAIN" = "1" ]; then
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode warmup_shortcut
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode warmup_grounded
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode routing
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode remix
fi

if [ "$RUN_EVAL" = "1" ]; then
  "$PYTHON_BIN" -m src.main --config "$CONFIG" --mode eval
  "$PYTHON_BIN" -m src.main --config "$SYM_CONFIG" --mode eval
fi
