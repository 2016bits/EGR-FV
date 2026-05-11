#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

sh run_scripts/run_warmup_shortcut.sh
sh run_scripts/run_warmup_grounded.sh

sh run_scripts/run_routing.sh
CONFIG=configs/remix.yaml sh run_scripts/run_remix.sh

CONFIG=configs/remix.yaml sh run_scripts/run_eval.sh