#!/bin/sh
# Full EGR-FV + claim-evidence baseline on HOVER, DeBERTa-v3-base backbone.

set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

CONFIG="${CONFIG:-configs/backbones/deberta_v3_hover.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/backbones/baseline_claim_evidence_deberta_v3_hover.yaml}"

# claim-evidence baseline first (faster, gives an immediate signal)
"$PYTHON_BIN" -m src.main --config "$BASELINE_CONFIG" --mode train_claim_evidence
"$PYTHON_BIN" -m src.main --config "$BASELINE_CONFIG" --mode eval

# full EGR-FV pipeline
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode warmup_shortcut
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode warmup_grounded
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode routing
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode remix
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode eval

echo "[deberta_v3 hover] finish time: $(date)"
