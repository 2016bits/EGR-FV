#!/bin/sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
"$PYTHON_BIN" -m src.main \
  --config configs/ablation_remix_heuristic.yaml \
  --mode remix

echo "finish ablation remix heuristic"
