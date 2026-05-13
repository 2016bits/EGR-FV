#!/bin/sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

POLITIHOP_CONFIG="${POLITIHOP_CONFIG:-configs/politihop.yaml}"
SYMMETRIC_POLITIHOP_CONFIG="${SYMMETRIC_POLITIHOP_CONFIG:-configs/politihop_symmetric.yaml}"

"$PYTHON_BIN" -m src.main --config "$POLITIHOP_CONFIG" --mode warmup_shortcut
"$PYTHON_BIN" -m src.main --config "$POLITIHOP_CONFIG" --mode warmup_grounded
"$PYTHON_BIN" -m src.main --config "$POLITIHOP_CONFIG" --mode routing
"$PYTHON_BIN" -m src.main --config "$POLITIHOP_CONFIG" --mode remix

"$PYTHON_BIN" -m src.main --config "$POLITIHOP_CONFIG" --mode eval
"$PYTHON_BIN" -m src.main --config "$SYMMETRIC_POLITIHOP_CONFIG" --mode eval

echo "finish time: $(date)"