#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = ("id", "claim", "label", "evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split HOVER raw train data into train/dev and use the raw dev "
            "split as held-out test data."
        ),
    )
    parser.add_argument("--raw-dir", default="data/HOVER/raw", help="Directory containing raw train.json/dev.json")
    parser.add_argument(
        "--output-dir",
        default="data/HOVER/converted_data",
        help="Directory where converted train.json/dev.json/test.json are written",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Fraction of raw train used for dev")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    parser.add_argument(
        "--stratify-fields",
        nargs="*",
        default=["label", "num_hops"],
        help="Fields used for stratified train/dev splitting",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use -1 for compact JSON")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list, got {type(payload).__name__}")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError(f"{path} must contain only JSON objects")
    return payload


def validate_records(records: list[dict[str, Any]], path: Path) -> None:
    missing = [
        (index, field)
        for index, record in enumerate(records)
        for field in REQUIRED_FIELDS
        if field not in record
    ]
    if missing:
        preview = ", ".join(f"#{index}:{field}" for index, field in missing[:10])
        raise ValueError(f"{path} has records missing required fields: {preview}")

    ids = [str(record["id"]) for record in records]
    duplicated = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if duplicated:
        preview = ", ".join(duplicated[:10])
        raise ValueError(f"{path} has duplicate ids: {preview}")


def stable_rank(record: dict[str, Any], index: int, group_key: tuple[str, ...], seed: int) -> str:
    sep = "\x1f"
    payload = sep.join((str(seed), sep.join(group_key), str(record.get("id", "")), str(index)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_dev_indices(
    records: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
    stratify_fields: Iterable[str],
) -> set[int]:
    if not 0 < dev_ratio < 1:
        raise ValueError(f"--dev-ratio must be between 0 and 1, got {dev_ratio}")

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    fields = tuple(stratify_fields)
    for index, record in enumerate(records):
        key = tuple(str(record.get(field, "")) for field in fields)
        groups[key].append(index)

    dev_indices: set[int] = set()
    for key in sorted(groups):
        indices = sorted(
            groups[key],
            key=lambda index: stable_rank(records[index], index, key, seed),
        )
        dev_count = int(round(len(indices) * dev_ratio))
        if dev_count == 0 and len(indices) > 1:
            dev_count = 1
        if dev_count >= len(indices):
            dev_count = len(indices) - 1
        dev_indices.update(indices[:dev_count])
    return dev_indices


def assert_disjoint(split_to_records: dict[str, list[dict[str, Any]]]) -> None:
    split_to_ids = {
        split: {str(record["id"]) for record in records}
        for split, records in split_to_records.items()
    }
    splits = sorted(split_to_ids)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = split_to_ids[left] & split_to_ids[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:10])
                raise ValueError(f"{left} and {right} overlap by id: {preview}")


def write_records(path: Path, records: list[dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_indent = None if indent < 0 else indent
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=json_indent)
        handle.write("\n")


def label_hop_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(f"{record.get('label')}|{record.get('num_hops')}" for record in records)
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    raw_train_path = raw_dir / "train.json"
    raw_dev_path = raw_dir / "dev.json"
    raw_train = load_records(raw_train_path)
    raw_dev = load_records(raw_dev_path)
    validate_records(raw_train, raw_train_path)
    validate_records(raw_dev, raw_dev_path)

    dev_indices = choose_dev_indices(
        records=raw_train,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
        stratify_fields=args.stratify_fields,
    )
    train_records = [record for index, record in enumerate(raw_train) if index not in dev_indices]
    dev_records = [record for index, record in enumerate(raw_train) if index in dev_indices]
    test_records = list(raw_dev)

    splits = {
        "train": train_records,
        "dev": dev_records,
        "test": test_records,
    }
    assert_disjoint(splits)

    write_records(output_dir / "train.json", train_records, int(args.indent))
    write_records(output_dir / "dev.json", dev_records, int(args.indent))
    write_records(output_dir / "test.json", test_records, int(args.indent))

    for split, records in splits.items():
        print(f"{split}: {len(records)} records")
        print(f"{split} label|num_hops: {label_hop_summary(records)}")


if __name__ == "__main__":
    main()
