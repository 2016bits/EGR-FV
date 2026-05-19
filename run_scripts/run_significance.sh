#!/bin/sh
# Multi-seed significance experiment driver.
#
# Convention (matches existing ablation scripts): for a given (dataset, seed),
# warmup_shortcut + warmup_grounded + routing are trained ONCE using the
# dataset's main config and then SHARED by every ablation method
# (two_branch / full_wo_remix / full_egr_fv). claim_evidence is independent
# and trains its own grounded baseline.
#
# Each run is isolated under a per-seed run_tag (e.g. `seed13`), so different
# seeds never overwrite each other and partial runs are safe to resume.
#
# Defaults:
#   DATASETS="fever politihop hover"
#   SEEDS="13 42 2024"
#   METHODS="claim_evidence two_branch full_wo_remix full_egr_fv"
#
# Usage:
#   sh run_scripts/run_significance.sh
#   SEEDS=13 DATASETS=politihop METHODS=claim_evidence sh run_scripts/run_significance.sh
#   SKIP_EXISTING=0 sh run_scripts/run_significance.sh   # force re-run
#   RUN_SUMMARIZE=0 sh run_scripts/run_significance.sh   # skip summarizer step

set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(resolve_repo_root)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

DATASETS="${DATASETS:-fever politihop hover}"
SEEDS="${SEEDS:-13 42 2024}"
METHODS="${METHODS:-claim_evidence two_branch full_wo_remix full_egr_fv}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_SUMMARIZE="${RUN_SUMMARIZE:-1}"

# Per-dataset main config (drives shared warmup_shortcut + warmup_grounded + routing).
shared_main_config() {
  case "$1" in
    fever)     echo "configs/fever.yaml" ;;
    politihop) echo "configs/politihop.yaml" ;;
    hover)     echo "configs/remix.yaml" ;;
    *)         echo "" ;;
  esac
}

# Resolve (config, sym_config, action) for a given (dataset, method).
# action ∈ { claim_evidence | remix_only | remix_with_warmup_skip }.
#   claim_evidence: just train_claim_evidence in its own dir
#   remix_only: remix stage only; assumes shared warmup+routing already done
resolve_combo() {
  combo="$1"
  case "$combo" in
    fever:claim_evidence)
      CONFIG=configs/baseline_claim_evidence_fever.yaml
      SYM_CONFIG=configs/baseline_claim_evidence_fever_symmetric.yaml
      ACTION=claim_evidence ;;
    fever:two_branch)
      CONFIG=configs/fever_ablation_two_branch.yaml
      SYM_CONFIG=configs/fever_ablation_two_branch_symmetric.yaml
      ACTION=remix_only ;;
    fever:full_wo_remix)
      CONFIG=configs/fever_ablation_full_wo_remix.yaml
      SYM_CONFIG=configs/fever_ablation_full_wo_remix_symmetric.yaml
      ACTION=remix_only ;;
    fever:full_egr_fv)
      CONFIG=configs/fever.yaml
      SYM_CONFIG=configs/fever_symmetric.yaml
      ACTION=remix_only ;;

    politihop:claim_evidence)
      CONFIG=configs/baseline_claim_evidence_politihop.yaml
      SYM_CONFIG=configs/baseline_claim_evidence_politihop_symmetric.yaml
      ACTION=claim_evidence ;;
    politihop:two_branch)
      CONFIG=configs/politihop_ablation_two_branch.yaml
      SYM_CONFIG=configs/politihop_ablation_two_branch_symmetric.yaml
      ACTION=remix_only ;;
    politihop:full_wo_remix)
      CONFIG=configs/politihop_ablation_full_wo_remix.yaml
      SYM_CONFIG=configs/politihop_ablation_full_wo_remix_symmetric.yaml
      ACTION=remix_only ;;
    politihop:full_egr_fv)
      CONFIG=configs/politihop.yaml
      SYM_CONFIG=configs/politihop_symmetric.yaml
      ACTION=remix_only ;;

    hover:claim_evidence)
      CONFIG=configs/baseline_claim_evidence_hover.yaml
      SYM_CONFIG=""
      ACTION=claim_evidence ;;
    hover:two_branch)
      CONFIG=configs/ablation_two_branch.yaml
      SYM_CONFIG=""
      ACTION=remix_only ;;
    hover:full_wo_remix)
      CONFIG=configs/ablation_full_wo_remix.yaml
      SYM_CONFIG=""
      ACTION=remix_only ;;
    hover:full_egr_fv)
      CONFIG=configs/remix.yaml
      SYM_CONFIG=""
      ACTION=remix_only ;;

    *)
      echo "[skip] unknown combo: $combo" >&2
      CONFIG=""
      SYM_CONFIG=""
      ACTION="" ;;
  esac
}

# Read a fully-resolved (deep-merged) config and emit the path with run_tag
# applied to a single key like `outputs.checkpoint_dir` or `data.routing_path`.
# Mirrors src/main.py:apply_run_tag exactly so the driver can predict the
# on-disk paths each stage will read/write.
resolve_path() {
  config="$1"; tag="$2"; key="$3"
  "$PYTHON_BIN" - "$config" "$tag" "$key" <<'PY'
import sys
from pathlib import Path

from src.utils.config import load_config

config_path, tag, key = sys.argv[1:4]
config = load_config(config_path)


def insert_dir(path, tag):
    return str(Path(path) / tag)


def insert_file(path, tag):
    p = Path(path)
    return str(p.parent / tag / p.name)


bucket, name = key.split(".", 1)
value = config.get(bucket, {}).get(name, "")
if not value:
    print("")
    sys.exit(0)
if bucket == "outputs":
    print(insert_dir(value, tag))
elif bucket in ("checkpoints", "data"):
    print(insert_file(value, tag))
else:
    print(value)
PY
}

run_stage() {
  config="$1"; mode="$2"; seed="$3"; tag="$4"; expected="$5"

  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$expected" ] && [ -f "$expected" ]; then
    echo "[skip] $config :: $mode (seed=$seed, tag=$tag) — $expected already exists"
    return 0
  fi

  echo "[run]  $config :: $mode (seed=$seed, tag=$tag)"
  "$PYTHON_BIN" -m src.main \
    --config "$config" \
    --mode "$mode" \
    --seed "$seed" \
    --run_tag "$tag"
}

# Train per-(dataset, seed) shared warmups + routing once. These are read by
# every ablation method (two_branch / full_wo_remix / full_egr_fv) via the
# shared `checkpoints.shortcut/grounded` + `data.routing_path` paths.
ensure_shared_warmup_and_routing() {
  dataset="$1"; seed="$2"; tag="$3"
  main_config=$(shared_main_config "$dataset")
  if [ -z "$main_config" ]; then
    return 0
  fi
  ckpt_dir=$(resolve_path "$main_config" "$tag" outputs.checkpoint_dir)
  routing_path=$(resolve_path "$main_config" "$tag" data.routing_path)

  run_stage "$main_config" warmup_shortcut "$seed" "$tag" "$ckpt_dir/shortcut_best.pt"
  run_stage "$main_config" warmup_grounded "$seed" "$tag" "$ckpt_dir/grounded_best.pt"
  run_stage "$main_config" routing         "$seed" "$tag" "$routing_path"
}

# Run a single (dataset, method, seed) combo end-to-end (training + eval).
run_combo() {
  dataset="$1"; method="$2"; seed="$3"; tag="$4"
  resolve_combo "$dataset:$method"
  if [ -z "$CONFIG" ]; then
    return 0
  fi
  echo
  echo "=== dataset=$dataset method=$method seed=$seed tag=$tag ==="

  ckpt_dir=$(resolve_path "$CONFIG" "$tag" outputs.checkpoint_dir)
  case "$ACTION" in
    claim_evidence)
      run_stage "$CONFIG" train_claim_evidence "$seed" "$tag" "$ckpt_dir/grounded_best.pt"
      ;;
    remix_only)
      run_stage "$CONFIG" remix "$seed" "$tag" "$ckpt_dir/remix_best.pt"
      ;;
    *)
      echo "[error] unknown action: $ACTION" >&2
      exit 1
      ;;
  esac

  pred_dir=$(resolve_path "$CONFIG" "$tag" outputs.prediction_dir)
  run_stage "$CONFIG" eval "$seed" "$tag" "$pred_dir/eval_report.json"

  if [ -n "$SYM_CONFIG" ]; then
    sym_pred_dir=$(resolve_path "$SYM_CONFIG" "$tag" outputs.prediction_dir)
    run_stage "$SYM_CONFIG" eval "$seed" "$tag" "$sym_pred_dir/eval_report.json"
  fi
}

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    tag="seed${seed}"

    # Per-seed shared pipeline (warmups + routing) — only needed if any
    # remix_only method is requested.
    need_shared=0
    for method in $METHODS; do
      case "$method" in
        two_branch|full_wo_remix|full_egr_fv) need_shared=1 ;;
      esac
    done
    if [ "$need_shared" = "1" ]; then
      echo
      echo "### shared warmup+routing: dataset=$dataset seed=$seed tag=$tag ###"
      ensure_shared_warmup_and_routing "$dataset" "$seed" "$tag"
    fi

    for method in $METHODS; do
      run_combo "$dataset" "$method" "$seed" "$tag"
    done
  done
done

if [ "$RUN_SUMMARIZE" = "1" ]; then
  echo
  echo "=== summarizing significance ==="
  "$PYTHON_BIN" scripts/summarize_significance.py \
    --datasets $DATASETS \
    --seeds $SEEDS \
    --methods $METHODS
fi

echo
echo "significance experiment finished"
