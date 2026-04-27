#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

sh scripts/run_warmup_shortcut.sh
sh scripts/run_warmup_grounded.sh

bash scripts/run_routing.sh
bash scripts/run_remix.sh
bash scripts/run_ablation_two_branch.sh
bash scripts/run_ablation_routing_only.sh
bash scripts/run_ablation_real_remix_no_weight.sh
bash scripts/run_ablation_remix_random.sh
bash scripts/run_ablation_remix_heuristic.sh
bash scripts/run_ablation_no_fusion.sh
bash scripts/run_ablation_no_orth.sh
