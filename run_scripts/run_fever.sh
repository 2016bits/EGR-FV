#!/bin/sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

FEVER_CONFIG="${FEVER_CONFIG:-configs/fever.yaml}"
SYMMETRIC_FEVER_CONFIG="${SYMMETRIC_FEVER_CONFIG:-configs/fever_symmetric.yaml}"

"$PYTHON_BIN" -m src.main --config "$FEVER_CONFIG" --mode warmup_shortcut
"$PYTHON_BIN" -m src.main --config "$FEVER_CONFIG" --mode warmup_grounded
"$PYTHON_BIN" -m src.main --config "$FEVER_CONFIG" --mode routing
"$PYTHON_BIN" -m src.main --config "$FEVER_CONFIG" --mode remix

"$PYTHON_BIN" -m src.main --config "$FEVER_CONFIG" --mode eval
"$PYTHON_BIN" -m src.main --config "$SYMMETRIC_FEVER_CONFIG" --mode eval
