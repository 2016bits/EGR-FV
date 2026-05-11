python scripts/summarize_hop_macro_f1.py \
  --test_file data/HOVER/converted_data/test.json \
  --output outputs/HOVER/predictions/hop_macro_f1_by_experiment.csv \
  --markdown_output outputs/HOVER/predictions/hop_macro_f1_by_experiment.md \
  --experiments \
    grounded_only:outputs/HOVER/predictions/grounded_only/eval_predictions.jsonl \
    two_branches_joint:outputs/HOVER/predictions/two_branches_joint/eval_predictions.jsonl \
    routing_only:outputs/HOVER/predictions/routing_only/eval_predictions.jsonl \
    random_remix_only:outputs/HOVER/predictions/random_remix_only/eval_predictions.jsonl \
    full_wo_remix:outputs/HOVER/predictions/full_wo_remix/eval_predictions.jsonl \
    full_wo_evidence_contrast:outputs/HOVER/predictions/full_wo_evidence_contrast/eval_predictions.jsonl \
    full_wo_grounded_dominant:outputs/HOVER/predictions/full_wo_grounded_dominant/eval_predictions.jsonl \
    full_hard_routing:outputs/HOVER/predictions/full_hard_routing/eval_predictions.jsonl \
    full_in_sample_routing:outputs/HOVER/predictions/full_in_sample_routing/eval_predictions.jsonl \
    fusion_inference:outputs/HOVER/predictions/fusion_inference/eval_predictions.jsonl \
    full_egr_fv_v2:outputs/HOVER/predictions/full_egr_fv_v2/eval_predictions.jsonl