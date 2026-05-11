#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize Macro-F1 by hop for EGR-FV / HoVer experiments.

Example:
python scripts/summarize_hop_macro_f1.py \
  --test_file data/HOVER/processed/test.jsonl \
  --pred_root outputs/HOVER/predictions \
  --output outputs/HOVER/predictions/hop_macro_f1_by_experiment.csv

This script will automatically scan:
outputs/HOVER/predictions/*/eval_predictions.jsonl

It also supports the root-level:
outputs/HOVER/predictions/eval_predictions.jsonl
if it exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sklearn.metrics import f1_score


# -----------------------------
# Basic IO
# -----------------------------

def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        records.append(obj)
    return records


# -----------------------------
# Robust field extraction
# -----------------------------

ID_KEYS = [
    "id",
    "uid",
    "qid",
    "example_id",
    "claim_id",
    "sample_id",
]

HOP_KEYS = [
    "num_hops",
    "num_hop",
    "n_hops",
    "hop",
    "hops",
    "level",
]

GOLD_KEYS = [
    "gold",
    "gold_label",
    "label",
    "label_id",
    "target",
    "answer",
]

PRED_KEYS = [
    "prediction",
    "pred",
    "pred_label",
    "label_pred",
    "predicted_label",
    "normal_prediction",
    "grounded_prediction",
]


def first_existing_key(record: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in record:
            return key
    return None


def get_record_id(record: Dict[str, Any]) -> Optional[str]:
    key = first_existing_key(record, ID_KEYS)
    if key is None:
        return None
    return str(record[key])


def normalize_hop_value(value: Any) -> Optional[int]:
    """
    Accepts values like:
    2, "2", "2-hop", "hop2", "two_hop" is not supported by default.
    """
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    text = str(value).strip().lower()
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))

    return None


def infer_hop_from_record(record: Dict[str, Any]) -> Optional[int]:
    key = first_existing_key(record, HOP_KEYS)
    if key is not None:
        hop = normalize_hop_value(record[key])
        if hop is not None:
            return hop

    # Fallback 1: some processed data may store evidence groups / docs.
    # Be conservative: only use these if they look like hop-level units.
    for key in [
        "supporting_docs",
        "supporting_documents",
        "evidence_docs",
        "context_docs",
    ]:
        value = record.get(key)
        if isinstance(value, list) and value:
            return len(value)

    # Fallback 2: supporting_facts may be list of [title, sent_id].
    # Counting unique titles is closer to HoVer hop count than counting sentences.
    for key in [
        "supporting_facts",
        "supporting_fact",
        "evidence",
    ]:
        value = record.get(key)
        if isinstance(value, list) and value:
            titles = set()
            for item in value:
                if isinstance(item, (list, tuple)) and item:
                    titles.add(str(item[0]))
                elif isinstance(item, dict):
                    title = (
                        item.get("title")
                        or item.get("doc")
                        or item.get("page")
                        or item.get("wiki_title")
                    )
                    if title is not None:
                        titles.add(str(title))
            if titles:
                return len(titles)

    return None


def normalize_label(value: Any) -> Any:
    """
    Normalize common HoVer / FEVER-style labels into stable comparable values.
    Keeps unknown labels as lower-case strings.
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value

    text = str(value).strip().lower()

    mapping = {
        "1": 1,
        "true": 1,
        "supported": 1,
        "support": 1,
        "supports": 1,
        "entailment": 1,
        "yes": 1,

        "0": 0,
        "false": 0,
        "not_supported": 0,
        "not-supported": 0,
        "not supported": 0,
        "refuted": 0,
        "refute": 0,
        "contradiction": 0,
        "no": 0,
    }

    return mapping.get(text, text)


def get_gold_label(record: Dict[str, Any]) -> Optional[Any]:
    key = first_existing_key(record, GOLD_KEYS)
    if key is None:
        return None
    return normalize_label(record[key])


def get_pred_label(record: Dict[str, Any]) -> Optional[Any]:
    key = first_existing_key(record, PRED_KEYS)
    if key is None:
        return None
    return normalize_label(record[key])


# -----------------------------
# Experiment discovery
# -----------------------------

def discover_prediction_files(pred_root: Path) -> List[Tuple[str, Path]]:
    """
    Discover prediction files from:
    pred_root/<experiment>/eval_predictions.jsonl

    Also include:
    pred_root/eval_predictions.jsonl
    if present.
    """
    items: List[Tuple[str, Path]] = []

    root_pred = pred_root / "eval_predictions.jsonl"
    if root_pred.exists():
        items.append(("root_eval", root_pred))

    for path in sorted(pred_root.glob("*/eval_predictions.jsonl")):
        experiment = path.parent.name
        items.append((experiment, path))

    # Remove duplicates while preserving order.
    seen = set()
    deduped: List[Tuple[str, Path]] = []
    for name, path in items:
        real = str(path.resolve())
        if real in seen:
            continue
        seen.add(real)
        deduped.append((name, path))

    return deduped


# -----------------------------
# Main computation
# -----------------------------

def build_test_meta(
    test_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
    - id_to_meta for id-based matching
    - ordered_meta for fallback order-based matching
    """
    id_to_meta: Dict[str, Dict[str, Any]] = {}
    ordered_meta: List[Dict[str, Any]] = []

    missing_hop = 0
    missing_gold = 0

    for index, record in enumerate(test_records):
        rid = get_record_id(record)
        hop = infer_hop_from_record(record)
        gold = get_gold_label(record)

        if hop is None:
            missing_hop += 1
        if gold is None:
            missing_gold += 1

        meta = {
            "index": index,
            "id": rid,
            "hop": hop,
            "gold": gold,
        }

        ordered_meta.append(meta)

        if rid is not None:
            id_to_meta[rid] = meta

    if missing_hop:
        print(f"[WARN] {missing_hop} test records have no detectable hop field.")
    if missing_gold:
        print(f"[WARN] {missing_gold} test records have no detectable gold label field.")

    return id_to_meta, ordered_meta


def align_predictions_with_test(
    pred_records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Any]],
    ordered_meta: List[Dict[str, Any]],
    pred_path: Path,
) -> List[Dict[str, Any]]:
    """
    Prefer id-based alignment.
    If prediction records do not contain ids, fallback to order-based alignment.
    """
    aligned: List[Dict[str, Any]] = []

    pred_has_any_id = any(get_record_id(record) is not None for record in pred_records)

    if pred_has_any_id:
        skipped_no_id = 0
        skipped_missing_test_id = 0

        for pred_record in pred_records:
            rid = get_record_id(pred_record)
            if rid is None:
                skipped_no_id += 1
                continue

            meta = id_to_meta.get(rid)
            if meta is None:
                skipped_missing_test_id += 1
                continue

            pred = get_pred_label(pred_record)
            gold = get_gold_label(pred_record)
            if gold is None:
                gold = meta["gold"]

            aligned.append({
                "id": rid,
                "hop": meta["hop"],
                "gold": gold,
                "pred": pred,
            })

        if skipped_no_id:
            print(f"[WARN] {pred_path}: skipped {skipped_no_id} prediction records with no id.")
        if skipped_missing_test_id:
            print(f"[WARN] {pred_path}: skipped {skipped_missing_test_id} prediction records whose ids are not in test file.")

        return aligned

    # Fallback: order-based matching.
    if len(pred_records) != len(ordered_meta):
        raise ValueError(
            f"{pred_path} has no id field and length mismatch prevents order-based alignment: "
            f"pred={len(pred_records)}, test={len(ordered_meta)}"
        )

    print(f"[WARN] {pred_path}: prediction records have no id; using order-based alignment.")

    for index, pred_record in enumerate(pred_records):
        meta = ordered_meta[index]
        pred = get_pred_label(pred_record)
        gold = get_gold_label(pred_record)
        if gold is None:
            gold = meta["gold"]

        aligned.append({
            "id": meta["id"],
            "hop": meta["hop"],
            "gold": gold,
            "pred": pred,
        })

    return aligned


def macro_f1_for_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    valid = [
        row for row in rows
        if row.get("gold") is not None
        and row.get("pred") is not None
        and row.get("hop") is not None
    ]

    if not valid:
        return None

    y_true = [row["gold"] for row in valid]
    y_pred = [row["pred"] for row in valid]

    # labels=None lets sklearn infer labels appearing in y_true/y_pred.
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_by_hop(
    experiment: str,
    pred_path: Path,
    pred_records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Any]],
    ordered_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aligned = align_predictions_with_test(
        pred_records=pred_records,
        id_to_meta=id_to_meta,
        ordered_meta=ordered_meta,
        pred_path=pred_path,
    )

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    skipped = 0
    for row in aligned:
        hop = row.get("hop")
        if hop is None:
            skipped += 1
            continue
        grouped[int(hop)].append(row)

    if skipped:
        print(f"[WARN] {experiment}: skipped {skipped} aligned rows with no hop.")

    output_rows: List[Dict[str, Any]] = []

    # Per-hop macro-F1.
    for hop in sorted(grouped.keys()):
        rows = grouped[hop]
        mf1 = macro_f1_for_rows(rows)
        output_rows.append({
            "experiment": experiment,
            "hop": hop,
            "n": len(rows),
            "macro_f1": "" if mf1 is None else f"{mf1:.6f}",
            "macro_f1_percent": "" if mf1 is None else f"{mf1 * 100:.2f}",
            "prediction_file": str(pred_path),
        })

    # Optional overall macro-F1 for sanity check.
    overall = macro_f1_for_rows(aligned)
    output_rows.append({
        "experiment": experiment,
        "hop": "all",
        "n": len([r for r in aligned if r.get("gold") is not None and r.get("pred") is not None]),
        "macro_f1": "" if overall is None else f"{overall:.6f}",
        "macro_f1_percent": "" if overall is None else f"{overall * 100:.2f}",
        "prediction_file": str(pred_path),
    })

    return output_rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "hop",
        "n",
        "macro_f1",
        "macro_f1_percent",
        "prediction_file",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_pivot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Create a compact markdown table:
    experiment | 2-hop | 3-hop | 4-hop | all
    """
    by_exp: Dict[str, Dict[str, str]] = defaultdict(dict)
    hop_values = set()

    for row in rows:
        exp = str(row["experiment"])
        hop = str(row["hop"])
        val = str(row["macro_f1_percent"])
        by_exp[exp][hop] = val
        hop_values.add(hop)

    def hop_sort_key(x: str) -> Tuple[int, str]:
        if x == "all":
            return (999, x)
        try:
            return (int(x), x)
        except ValueError:
            return (998, x)

    sorted_hops = sorted(hop_values, key=hop_sort_key)

    lines = []
    header = ["Experiment"] + [f"{hop}-hop Macro-F1" if hop != "all" else "All Macro-F1" for hop in sorted_hops]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * len(sorted_hops)) + " |")

    for exp in sorted(by_exp.keys()):
        values = [by_exp[exp].get(hop, "") for hop in sorted_hops]
        lines.append("| " + " | ".join([exp] + values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-hop Macro-F1 for EGR-FV experiments."
    )

    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test JSON/JSONL file containing id, label, and hop fields.",
    )

    parser.add_argument(
        "--pred_root",
        type=str,
        default="outputs/HOVER/predictions",
        help="Root directory containing experiment prediction folders.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/HOVER/predictions/hop_macro_f1_by_experiment.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--markdown_output",
        type=str,
        default="outputs/HOVER/predictions/hop_macro_f1_by_experiment.md",
        help="Output Markdown table path.",
    )

    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help=(
            "Optional explicit experiments in format name:path/to/eval_predictions.jsonl. "
            "If omitted, the script scans pred_root/*/eval_predictions.jsonl."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_file = Path(args.test_file)
    pred_root = Path(args.pred_root)
    output_path = Path(args.output)
    markdown_output_path = Path(args.markdown_output)

    test_records = read_json_or_jsonl(test_file)
    id_to_meta, ordered_meta = build_test_meta(test_records)

    if args.experiments:
        prediction_files: List[Tuple[str, Path]] = []
        for item in args.experiments:
            if ":" not in item:
                raise ValueError(
                    f"Invalid --experiments item: {item}. "
                    f"Expected format: experiment_name:path/to/eval_predictions.jsonl"
                )
            name, path = item.split(":", 1)
            prediction_files.append((name, Path(path)))
    else:
        prediction_files = discover_prediction_files(pred_root)

    if not prediction_files:
        raise FileNotFoundError(
            f"No eval_predictions.jsonl found under {pred_root}. "
            f"Expected files like {pred_root}/full_egr_fv_v2/eval_predictions.jsonl"
        )

    all_rows: List[Dict[str, Any]] = []

    print(f"[INFO] Loaded test records: {len(test_records)}")
    print(f"[INFO] Found prediction files: {len(prediction_files)}")

    for experiment, pred_path in prediction_files:
        print(f"[INFO] Processing {experiment}: {pred_path}")
        pred_records = read_json_or_jsonl(pred_path)

        rows = compute_by_hop(
            experiment=experiment,
            pred_path=pred_path,
            pred_records=pred_records,
            id_to_meta=id_to_meta,
            ordered_meta=ordered_meta,
        )
        all_rows.extend(rows)

    write_csv(all_rows, output_path)
    write_markdown_pivot(all_rows, markdown_output_path)

    print(f"[DONE] Wrote CSV: {output_path}")
    print(f"[DONE] Wrote Markdown table: {markdown_output_path}")


if __name__ == "__main__":
    main()