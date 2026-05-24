#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.routing import (
    add_loss_weights,
    apply_routing_strategy,
    compute_necessity_score,
    summarize_groups,
)


HARD_GROUP_DEFAULTS = {
    "bias_easy": {"grounded": 0.5, "shortcut": 1.0},
    "hard": {"grounded": 1.0, "shortcut": 0.5},
    "grounded_needed": {"grounded": 1.5, "shortcut": 0.0},
}


NECESSITY_COMPONENT_KEYS = ("grounded_correct", "shortcut_wrong", "disagreement", "evidence_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a routing-cache variant from an existing cache.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source routing JSONL path. May be provided multiple times.",
    )
    parser.add_argument(
        "--source-glob",
        action="append",
        default=[],
        help="Glob for source routing JSONL paths. May be provided multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output routing JSONL path.")
    parser.add_argument(
        "--variant",
        default="hard",
        choices=["hard", "necessity"],
        help="Routing variant to materialize.",
    )
    parser.add_argument("--stats-output", default=None, help="Optional stats JSON path.")
    parser.add_argument(
        "--sample-weight-source",
        default="necessity",
        choices=["necessity", "group"],
        help="How to set sample_weight in the materialized cache.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")

    # ----- necessity-variant arguments -----
    parser.add_argument(
        "--necessity-weights",
        default="",
        help=(
            "Comma-separated 'key=val' overrides for necessity-score component weights. "
            f"Keys: {','.join(NECESSITY_COMPONENT_KEYS)}. "
            "Missing keys keep the routing.py defaults (0.4/0.3/0.2/0.1)."
        ),
    )
    parser.add_argument("--grounded-threshold", type=float, default=0.7, help="grounded_needed_threshold")
    parser.add_argument("--bias-threshold", type=float, default=0.3, help="bias_easy_threshold")
    parser.add_argument("--tau-shortcut-high", type=float, default=0.8)
    parser.add_argument("--tau-disagreement-low", type=float, default=0.1)
    parser.add_argument(
        "--strategy",
        default="soft_evidence_necessity",
        help="Routing strategy fed to apply_routing_strategy (default mirrors configs/default.yaml).",
    )
    parser.add_argument(
        "--stratified-strategy",
        default="soft_evidence_necessity",
        help="Stratified strategy when --stratify-by-label is set.",
    )
    parser.add_argument(
        "--stratify-by-label",
        dest="stratify_by_label",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-stratify-by-label",
        dest="stratify_by_label",
        action="store_false",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha_grounded_weight")
    parser.add_argument("--beta", type=float, default=1.0, help="beta_shortcut_gate")
    parser.add_argument(
        "--weighting",
        default="soft",
        choices=["soft", "hard"],
        help="Weighting scheme for add_loss_weights.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def resolve_sources(paths: Iterable[str], patterns: Iterable[str]) -> List[Path]:
    sources = [Path(path) for path in paths]
    for pattern in patterns:
        sources.extend(Path(match) for match in sorted(glob.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for source in sources:
        normalized = str(source)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(source)
    if not unique:
        raise ValueError("At least one --source or --source-glob path is required.")
    return unique


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def apply_hard_weights(record: Dict[str, Any], sample_weight_source: str) -> Dict[str, Any]:
    updated = dict(record)
    group = str(updated.get("group", "hard"))
    weights = HARD_GROUP_DEFAULTS.get(group, HARD_GROUP_DEFAULTS["hard"])
    grounded_weight = float(weights["grounded"])
    shortcut_weight = float(weights["shortcut"])
    updated["grounded_loss_weight"] = grounded_weight
    updated["shortcut_loss_weight"] = shortcut_weight
    updated["sample_weight"] = float(updated.get("sample_weight", 1.0)) if sample_weight_source == "group" else grounded_weight
    updated["routing_weighting"] = "hard"
    return updated


def parse_necessity_weights(spec: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    if not spec:
        return parsed
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Malformed necessity-weights entry (expected key=value): {chunk!r}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        if key not in NECESSITY_COMPONENT_KEYS:
            raise ValueError(
                f"Unknown necessity-weight key {key!r}. "
                f"Allowed keys: {','.join(NECESSITY_COMPONENT_KEYS)}"
            )
        try:
            parsed[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Necessity weight for {key!r} is not numeric: {value!r}") from exc
    return parsed


def build_routing_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "strategy": args.strategy,
        "stratified_strategy": args.stratified_strategy,
        "stratify_by_label": bool(args.stratify_by_label),
        "grounded_needed_threshold": float(args.grounded_threshold),
        "bias_easy_threshold": float(args.bias_threshold),
        "tau_shortcut_high": float(args.tau_shortcut_high),
        "tau_disagreement_low": float(args.tau_disagreement_low),
        "alpha_grounded_weight": float(args.alpha),
        "beta_shortcut_gate": float(args.beta),
        "weighting": args.weighting,
        "sample_weight_source": args.sample_weight_source,
        "necessity_weights": parse_necessity_weights(args.necessity_weights),
        "estimation": "materialized_necessity_v1",
    }


def apply_necessity_variant(
    records: List[Dict[str, Any]],
    routing_config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    necessity_weights = routing_config.get("necessity_weights", {})
    rebuilt: List[Dict[str, Any]] = []
    missing_field_errors: Counter = Counter()
    for record in records:
        updated = dict(record)
        try:
            shortcut_conf = float(updated["shortcut_conf"])
            shortcut_correct = bool(updated["shortcut_correct"])
            grounded_conf = float(updated["grounded_conf"])
            grounded_correct = bool(updated["grounded_correct"])
            disagreement = float(updated["disagreement"])
        except KeyError as exc:
            missing_field_errors[str(exc.args[0])] += 1
            continue
        length_score = float(updated.get("evidence_length_score", 0.0))
        necessity_score = compute_necessity_score(
            shortcut_conf=shortcut_conf,
            shortcut_correct=shortcut_correct,
            grounded_conf=grounded_conf,
            grounded_correct=grounded_correct,
            disagreement=disagreement,
            evidence_length_score=length_score,
            weights=necessity_weights,
        )
        updated["necessity_score"] = necessity_score
        rebuilt.append(updated)

    if missing_field_errors:
        formatted = ", ".join(f"{name}: {count}" for name, count in missing_field_errors.items())
        raise ValueError(
            "Some source records were missing required necessity-component fields. "
            f"Counts by missing key: {formatted}"
        )

    routed = apply_routing_strategy(rebuilt, dict(routing_config))

    routing_strategy = str(routing_config.get("strategy", "soft_evidence_necessity")).lower()
    stratified_strategy = str(routing_config.get("stratified_strategy", routing_strategy)).lower()
    stratify_by_label = bool(routing_config.get("stratify_by_label", False))
    for record in routed:
        add_loss_weights(record, routing_config)
        record["routing_strategy"] = routing_strategy
        record["routing_stratified_strategy"] = stratified_strategy
        record["routing_stratify_by_label"] = stratify_by_label
        record["routing_estimation"] = str(routing_config.get("estimation", "materialized_necessity_v1"))
        record["routing_weighting"] = str(routing_config.get("weighting", "soft"))
    return routed


def summarize_hard(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(record.get("group", "hard")) for record in records)
    summary: Dict[str, Any] = {
        "num_total": len(records),
        "num_bias_easy": counts.get("bias_easy", 0),
        "num_grounded_needed": counts.get("grounded_needed", 0),
        "num_hard": counts.get("hard", 0),
        "routing_weighting": "hard",
    }
    if records and "routing_strategy" in records[0]:
        summary["routing_strategy"] = records[0]["routing_strategy"]
        summary["routing_stratified_strategy"] = records[0].get("routing_stratified_strategy")
        summary["routing_stratify_by_label"] = records[0].get("routing_stratify_by_label")
    if any("label_id" in record for record in records):
        label_counts: Dict[str, Dict[str, int]] = {}
        for record in records:
            group = str(record.get("group", "hard"))
            label_id = str(record.get("label_id", "unknown"))
            label_counts.setdefault(group, {})
            label_counts[group][label_id] = label_counts[group].get(label_id, 0) + 1
        summary["label_counts_by_group"] = label_counts
    return summary


def main() -> None:
    args = parse_args()
    source_paths = resolve_sources(args.source, args.source_glob)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    records: List[Dict[str, Any]] = []
    for source_path in source_paths:
        records.extend(load_jsonl(source_path))

    if args.variant == "hard":
        materialized = [apply_hard_weights(record, args.sample_weight_source) for record in records]
        stats = summarize_hard(materialized)
    elif args.variant == "necessity":
        routing_config = build_routing_config(args)
        materialized = apply_necessity_variant(records, routing_config)
        stats = summarize_groups(materialized)
        stats["routing_weighting"] = routing_config.get("weighting", "soft")
        stats["necessity_weights"] = routing_config.get("necessity_weights", {})
        stats["grounded_needed_threshold"] = routing_config["grounded_needed_threshold"]
        stats["bias_easy_threshold"] = routing_config["bias_easy_threshold"]
        stats["alpha_grounded_weight"] = routing_config["alpha_grounded_weight"]
        stats["beta_shortcut_gate"] = routing_config["beta_shortcut_gate"]
    else:
        raise ValueError(f"Unsupported variant: {args.variant}")

    write_jsonl(output_path, materialized)
    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")
    write_json(stats_path, stats)
    print(f"Wrote {output_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
