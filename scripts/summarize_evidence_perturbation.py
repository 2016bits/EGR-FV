#!/usr/bin/env python3
"""Aggregate per-(dataset, method) evidence-perturbation reports into a single
table.

Reads JSON reports written by `scripts/eval_evidence_perturbation.py` and
produces:

- evidence_perturbation_summary.csv  (long format, one row per condition)
- evidence_perturbation_wide.csv     (wide format, one row per dataset+method)
- evidence_perturbation_report.md    (markdown table matching the analysis doc)

Layout expected under --root:

    <root>/<dataset>/<method-slug>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DATASET_ORDER = ("HOVER", "FEVER", "PolitiHop")
# (slug, display name) — slug matches the filenames produced by the runner.
METHOD_ORDER = (
    ("claim-evidence", "claim-evidence"),
    ("wo-contrast", "w/o contrast"),
    ("wo-remix", "w/o remix"),
    ("EGR-FV", "EGR-FV"),
)
CONDITION_ORDER = ("clean", "shuffled", "null", "wrong_evidence")
CONDITION_HEADERS = {
    "clean": "Clean",
    "shuffled": "Shuffled",
    "null": "Null",
    "wrong_evidence": "Wrong Evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evidence-perturbation reports.")
    parser.add_argument(
        "--root",
        default="outputs/analysis/evidence_perturbation",
        help="Directory containing <dataset>/<method-slug>.json reports.",
    )
    parser.add_argument(
        "--output-prefix",
        default="evidence_perturbation",
        help="Filename prefix written under --root.",
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy"],
        help="Which metric to render in the markdown table.",
    )
    return parser.parse_args()


def load_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def fmt_percent(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}"


def build_long_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for slug, display in METHOD_ORDER:
            path = root / dataset / f"{slug}.json"
            report = load_report(path)
            for condition in CONDITION_ORDER:
                metric_macro: Optional[float] = None
                metric_acc: Optional[float] = None
                num_samples: Optional[int] = None
                delta_vs_clean: Optional[float] = None
                if report is not None:
                    payload = report.get("conditions", {}).get(condition, {})
                    if payload:
                        metric_macro = payload.get("macro_f1")
                        metric_acc = payload.get("accuracy")
                        num_samples = payload.get("num_samples")
                        delta_vs_clean = payload.get("delta_vs_clean")
                rows.append(
                    {
                        "dataset": dataset,
                        "method_slug": slug,
                        "method": display,
                        "condition": condition,
                        "macro_f1": metric_macro,
                        "accuracy": metric_acc,
                        "delta_vs_clean": delta_vs_clean,
                        "num_samples": num_samples,
                        "report_path": str(path),
                        "available": report is not None,
                    }
                )
    return rows


def build_wide_rows(long_rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    index: Dict[tuple, Dict[str, Any]] = {}
    for row in long_rows:
        key = (row["dataset"], row["method_slug"])
        index.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "method_slug": row["method_slug"],
                "method": row["method"],
                "available": row["available"],
            },
        )
        bucket = index[key]
        bucket[row["condition"]] = row.get(metric)
        if row["condition"] != "clean":
            bucket[f"{row['condition']}_delta_vs_clean"] = row.get("delta_vs_clean")
    ordered: List[Dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for slug, _ in METHOD_ORDER:
            entry = index.get((dataset, slug))
            if entry is not None:
                ordered.append(entry)
    return ordered


def write_long_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "method_slug",
        "condition",
        "macro_f1",
        "accuracy",
        "delta_vs_clean",
        "num_samples",
        "available",
        "report_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_wide_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "method_slug",
        "clean",
        "shuffled",
        "shuffled_delta_vs_clean",
        "null",
        "null_delta_vs_clean",
        "wrong_evidence",
        "wrong_evidence_delta_vs_clean",
        "available",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def write_markdown(path: Path, wide_rows: List[Dict[str, Any]], metric: str) -> None:
    lines: List[str] = [
        f"# Evidence Perturbation Analysis ({metric})",
        "",
        "Each cell reports macro-F1 (%) on the test split of the corresponding "
        "dataset after replacing the gold evidence according to the column "
        "condition: ",
        "",
        "- **Clean**: original gold evidence.",
        "- **Shuffled**: evidence randomly re-assigned across samples (label-agnostic).",
        "- **Null**: evidence replaced with the `[NO_EVIDENCE]` placeholder.",
        "- **Wrong Evidence**: evidence drawn from a sample with a *different* gold label.",
        "",
    ]
    body_rows: List[List[str]] = []
    for row in wide_rows:
        body_rows.append(
            [
                row["dataset"],
                row["method"],
                fmt_percent(row.get("clean")),
                fmt_percent(row.get("shuffled")),
                fmt_percent(row.get("null")),
                fmt_percent(row.get("wrong_evidence")),
            ]
        )
    lines.extend(
        markdown_table(
            ["Dataset", "Method", "Clean", "Shuffled", "Null", "Wrong Evidence"],
            body_rows,
        )
    )

    # Drop-from-clean view to make sensitivity differences easy to read.
    lines.append("")
    lines.append("## Drop vs Clean (percentage points)")
    drop_rows: List[List[str]] = []
    for row in wide_rows:
        clean = row.get("clean")

        def _drop(value: Optional[float]) -> str:
            if clean is None or value is None:
                return "—"
            return f"{(clean - value) * 100:+.2f}"

        drop_rows.append(
            [
                row["dataset"],
                row["method"],
                _drop(row.get("shuffled")),
                _drop(row.get("null")),
                _drop(row.get("wrong_evidence")),
            ]
        )
    lines.extend(
        markdown_table(
            ["Dataset", "Method", "Clean → Shuffled", "Clean → Null", "Clean → Wrong"],
            drop_rows,
        )
    )

    missing = [row for row in wide_rows if not row.get("available")]
    if missing:
        lines.append("")
        lines.append("## Missing Reports")
        for row in missing:
            lines.append(f"- {row['dataset']} / {row['method']}")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    long_rows = build_long_rows(root)
    wide_rows = build_wide_rows(long_rows, metric=args.metric)

    summary_path = root / f"{args.output_prefix}_summary.csv"
    wide_path = root / f"{args.output_prefix}_wide.csv"
    md_path = root / f"{args.output_prefix}_report.md"

    write_long_csv(summary_path, long_rows)
    write_wide_csv(wide_path, wide_rows)
    write_markdown(md_path, wide_rows, metric=args.metric)

    print(f"Wrote {summary_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
