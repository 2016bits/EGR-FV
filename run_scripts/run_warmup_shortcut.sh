#!/bin/sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
CONFIG="${CONFIG:-configs/shortcut.yaml}"
"$PYTHON_BIN" -m src.main \
  --config "$CONFIG" \
  --mode warmup_shortcut

echo "finish warmup shortcut training: $CONFIG"
