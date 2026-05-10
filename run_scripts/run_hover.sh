export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

sh run_scripts/run_warmup_shortcut.sh
sh run_scripts/run_warmup_grounded.sh

sh run_scripts/run_routing.sh
CONFIG=configs/remix.yaml sh run_scripts/run_remix.sh
CONFIG=configs/ablation_two_branch.yaml sh run_scripts/run_remix.sh
CONFIG=configs/ablation_routing_only.yaml sh run_scripts/run_remix.sh
CONFIG=configs/ablation_random_remix_only.yaml sh run_scripts/run_remix.sh
CONFIG=configs/ablation_full_wo_remix.yaml sh run_scripts/run_remix.sh
CONFIG=configs/ablation_full_wo_evidence_contrast.yaml sh run_scripts/run_remix.sh
CONFIG=configs/ablation_full_wo_grounded_dominant.yaml sh run_scripts/run_remix.sh

CONFIG=configs/ablation_full_hard_routing.yaml sh run_scripts/run_routing.sh
CONFIG=configs/ablation_full_hard_routing.yaml sh run_scripts/run_remix.sh

CONFIG=configs/ablation_full_in_sample_routing.yaml sh run_scripts/run_routing.sh
CONFIG=configs/ablation_full_in_sample_routing.yaml sh run_scripts/run_remix.sh

sh run_scripts/run_eval.sh
