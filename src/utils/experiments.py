from __future__ import annotations

from typing import Any, Mapping


_MODE_ALIASES = {
    "full": "full",
    "egr_fv": "full",
    "egr-fv": "full",
    "two_branch": "two_branch_only",
    "two_branches": "two_branch_only",
    "two_branch_only": "two_branch_only",
    "two-branch-only": "two_branch_only",
    "routing_only": "routing_only",
    "routing-only": "routing_only",
    "remix_only": "remix_only",
    "remix-only": "remix_only",
    "random_remix_only": "remix_only",
    "heuristic_remix_only": "remix_only",
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
    return experiment_mode(config) in {"full", "routing_only"}


def requires_routing_file(config: Mapping[str, Any]) -> bool:
    return uses_real_routing(config)


def uses_batch_remix(config: Mapping[str, Any]) -> bool:
    return experiment_mode(config) in {"full", "remix_only"}


def uses_pseudo_groups(config: Mapping[str, Any]) -> bool:
    return experiment_mode(config) == "remix_only"


def uses_uniform_sample_weights(config: Mapping[str, Any]) -> bool:
    return experiment_mode(config) == "two_branch_only"
