#!/bin/sh
# Full EGR-FV + claim-evidence baseline on PolitiHop, DeBERTa-v3-base backbone.

set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

CONFIG="${CONFIG:-configs/backbones/deberta_v3_politihop.yaml}"
SYM_CONFIG="${SYM_CONFIG:-configs/backbones/deberta_v3_politihop_symmetric.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/backbones/baseline_claim_evidence_deberta_v3_politihop.yaml}"
BASELINE_SYM_CONFIG="${BASELINE_SYM_CONFIG:-configs/backbones/baseline_claim_evidence_deberta_v3_politihop_symmetric.yaml}"

# claim-evidence baseline
"$PYTHON_BIN" -m src.main --config "$BASELINE_CONFIG" --mode train_claim_evidence
"$PYTHON_BIN" -m src.main --config "$BASELINE_CONFIG" --mode eval
"$PYTHON_BIN" -m src.main --config "$BASELINE_SYM_CONFIG" --mode eval

# full EGR-FV
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode warmup_shortcut
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode warmup_grounded
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode routing
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode remix
"$PYTHON_BIN" -m src.main --config "$CONFIG" --mode eval
"$PYTHON_BIN" -m src.main --config "$SYM_CONFIG" --mode eval

echo "[deberta_v3 politihop] finish time: $(date)"
