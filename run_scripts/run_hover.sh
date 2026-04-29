export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

sh run_scripts/run_warmup_shortcut.sh
sh run_scripts/run_warmup_grounded.sh

sh run_scripts/run_routing.sh
sh run_scripts/run_remix.sh
sh run_scripts/run_ablation_two_branch.sh
sh run_scripts/run_ablation_routing_only.sh
sh run_scripts/run_ablation_real_remix_no_weight.sh
sh run_scripts/run_ablation_remix_random.sh
sh run_scripts/run_ablation_remix_heuristic.sh
sh run_scripts/run_ablation_no_fusion.sh
sh run_scripts/run_ablation_no_orth.sh
