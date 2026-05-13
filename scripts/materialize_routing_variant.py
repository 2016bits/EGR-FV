#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


HARD_GROUP_DEFAULTS = {
    "bias_easy": {"grounded": 0.5, "shortcut": 1.0},
    "hard": {"grounded": 1.0, "shortcut": 0.5},
    "grounded_needed": {"grounded": 1.5, "shortcut": 0.0},
}


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
        choices=["hard"],
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


def summarize(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
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
    else:
        raise ValueError(f"Unsupported variant: {args.variant}")

    write_jsonl(output_path, materialized)
    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")
    write_json(stats_path, summarize(materialized))
    print(f"Wrote {output_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
