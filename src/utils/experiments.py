from __future__ import annotations

from typing import Any, Mapping


_MODE_ALIASES = {
    "full": "full",
    "full_egr_fv_v2": "full",
    "egr_fv": "full",
    "egr-fv": "full",
    "grounded_only": "grounded_only",
    "grounded-only": "grounded_only",
    "two_branch": "two_branch_only",
    "two_branches": "two_branch_only",
    "two_branch_only": "two_branch_only",
    "two-branch-only": "two_branch_only",
    "routing_only": "routing_only",
    "routing-only": "routing_only",
    "remix_only": "remix_only",
    "remix-only": "remix_only",
    "random_remix_only": "remix_only",
    "length_remix_only": "length_remix_only",
    "heuristic_remix_only": "remix_only",
    "full_wo_remix": "full_wo_remix",
    "full_without_remix": "full_wo_remix",
    "full_w/o_remix": "full_wo_remix",
    "full_wo_routing": "full_wo_routing",
    "full_without_routing": "full_wo_routing",
    "full_w/o_routing": "full_wo_routing",
    "full_wo_evidence_contrast": "full_wo_evidence_contrast",
    "full_without_evidence_contrast": "full_wo_evidence_contrast",
    "full_w/o_evidence_contrast": "full_wo_evidence_contrast",
    "full_wo_grounded_dominant": "full_wo_grounded_dominant",
    "full_without_grounded_dominant": "full_wo_grounded_dominant",
    "full_w/o_grounded_dominant": "full_wo_grounded_dominant",
    "full_hard_routing": "full_hard_routing",
    "full_w_hard_routing": "full_hard_routing",
    "full_in_sample_routing": "full_in_sample_routing",
    "full_w_in_sample_routing": "full_in_sample_routing",
    "fusion_inference": "fusion_inference",
    "shortcut_only": "shortcut_only",
}


def experiment_mode(config: Mapping[str, Any]) -> str:
    experiment_cfg = config.get("experiment", {})
    raw_mode = experiment_cfg.get("mode", "full") if isinstance(experiment_cfg, Mapping) else "full"
    normalized = str(raw_mode).strip().lower()
    if normalized not in _MODE_ALIASES:
        valid = ", ".join(sorted(set(_MODE_ALIASES.values())))
        raise ValueError(f"Unsupported experiment.mode={raw_mode!r}. Expected one of: {valid}.")
    return _MODE_ALIASES[normalized]


def uses_real_routing(config: Mapping[str, Any]) -> bool:
    return experiment_mode(config) in {
        "full",
        "routing_only",
        "full_wo_remix",
        "full_wo_evidence_contrast",
        "full_wo_grounded_dominant",
        "full_hard_routing",
        "full_in_sample_routing",
        "fusion_inference",
    }


def requires_routing_file(config: Mapping[str, Any]) -> bool:
    return uses_real_routing(config)


def uses_batch_remix(config: Mapping[str, Any]) -> bool:
    return experiment_mode(config) in {
        "full",
        "remix_only",
        "length_remix_only",
        "full_wo_routing",
        "full_wo_evidence_contrast",
        "full_wo_grounded_dominant",
        "full_hard_routing",
        "full_in_sample_routing",
        "fusion_inference",
    }


def uses_pseudo_groups(config: Mapping[str, Any]) -> bool:
    return experiment_mode(config) in {"remix_only", "length_remix_only", "full_wo_routing"}


def uses_uniform_sample_weights(config: Mapping[str, Any]) -> bool:
    experiment_cfg = config.get("experiment", {})
    if isinstance(experiment_cfg, Mapping) and "uniform_sample_weights" in experiment_cfg:
        return bool(experiment_cfg["uniform_sample_weights"])
    return experiment_mode(config) in {"two_branch_only", "remix_only", "length_remix_only", "full_wo_grounded_dominant"}


def uses_grounded_dominant_objective(config: Mapping[str, Any]) -> bool:
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, Mapping) and "use_grounded_dominant" in loss_cfg:
        return bool(loss_cfg["use_grounded_dominant"])
    return experiment_mode(config) in {
        "full",
        "routing_only",
        "full_wo_remix",
        "full_wo_evidence_contrast",
        "full_hard_routing",
        "full_in_sample_routing",
        "fusion_inference",
    }


def uses_evidence_contrast(config: Mapping[str, Any]) -> bool:
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, Mapping) and "use_evidence_contrast" in loss_cfg:
        return bool(loss_cfg["use_evidence_contrast"])
    return experiment_mode(config) in {
        "full",
        "full_wo_routing",
        "full_wo_remix",
        "full_wo_grounded_dominant",
        "full_hard_routing",
        "full_in_sample_routing",
        "fusion_inference",
    }
