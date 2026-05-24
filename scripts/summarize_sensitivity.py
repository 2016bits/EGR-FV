#!/usr/bin/env python3
"""Aggregate HOVER sensitivity sweep eval_report.json files into CSV + Markdown.

Reads `outputs/HOVER/predictions/sensitivity/<variant>/eval_report.json` for
each variant produced by `run_scripts/run_sensitivity.sh`, plus the baseline
`outputs/HOVER/predictions/full_egr_fv_v2/eval_report.json` for delta columns.
Writes:
    outputs/analysis/sensitivity/sensitivity_table.csv
    outputs/analysis/sensitivity/sensitivity_report.md

Usage:
    python scripts/summarize_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

BASELINE_NAME = "Baseline (full EGR-FV)"
BASELINE_PATH = REPO_ROOT / "outputs/HOVER/predictions/full_egr_fv_v2/eval_report.json"

# Each entry: (variant_name, human_label)
# A variant_name == "__baseline__" inserts the baseline row.
SWEEPS: Dict[str, List[Tuple[str, str]]] = {
    "necessity": [
        ("__baseline__",                 f"{BASELINE_NAME} — (0.4, 0.3, 0.2, 0.1)"),
        ("necessity_only_grounded",      "Only grounded_correct (1, 0, 0, 0)"),
        ("necessity_only_shortcut",      "Only shortcut_wrong (0, 1, 0, 0)"),
        ("necessity_only_disagreement",  "Only disagreement (0, 0, 1, 0)"),
        ("necessity_only_evlen",         "Only evidence_length (0, 0, 0, 1)"),
        ("necessity_equal",              "Equal weights (0.25 × 4)"),
    ],
    "threshold": [
        ("__baseline__",       f"{BASELINE_NAME} — thresholds 0.7 / 0.3"),
        ("threshold_60_40",    "Thresholds 0.6 / 0.4 (more grounded_needed + bias_easy)"),
        ("threshold_80_20",    "Thresholds 0.8 / 0.2 (tighter buckets)"),
    ],
    "remix_ratio": [
        ("__baseline__",         f"{BASELINE_NAME} — 50 / 30 / 15 / 5"),
        ("remix_less_grounded",  "Less grounded — 60 / 20 / 15 / 5"),
        ("remix_more_grounded",  "More grounded — 40 / 40 / 15 / 5"),
        ("remix_equal",          "Equal — 25 / 25 / 25 / 25"),
    ],
    "loss_weight": [
        ("__baseline__",          f"{BASELINE_NAME} — α=0.5, β=1.0, λ_ctr=0.2"),
        ("alpha_025",             "α = 0.25"),
        ("alpha_200",             "α = 2.0"),
        ("beta_050",              "β = 0.5"),
        ("beta_200",              "β = 2.0"),
        ("lambda_contrast_010",   "λ_ctr = 0.1"),
        ("lambda_contrast_040",   "λ_ctr = 0.4"),
    ],
}


SWEEP_LABELS = {
    "necessity":   "Necessity-Score Component Ablation",
    "threshold":   "Routing-Threshold Sensitivity",
    "remix_ratio": "Remix-Ratio Sensitivity",
    "loss_weight": "Loss-Weight Sensitivity (α / β / λ_ctr)",
}


def load_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def variant_report_path(variant: str) -> Path:
    if variant == "__baseline__":
        return BASELINE_PATH
    return REPO_ROOT / f"outputs/HOVER/predictions/sensitivity/{variant}/eval_report.json"


def metric(report: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    base = report.get("base", {}) if report else {}
    metrics = base.get("metrics", {}) if isinstance(base, Mapping) else {}
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analysis_metric(report: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    if not report:
        return default
    value = report.get("analysis_metrics", {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value: float, precision: int = 4) -> str:
    if value != value:
        return ""
    return f"{value:.{precision}f}"


def format_delta(value: float) -> str:
    if value != value:
        return ""
    return f"{value:+.4f}"


def collect_rows(baseline_macro_f1: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sweep_group, variants in SWEEPS.items():
        for variant_name, label in variants:
            path = variant_report_path(variant_name)
            report = load_report(path)
            if report is None:
                rows.append({
                    "sweep_group": sweep_group,
                    "variant": variant_name,
                    "name": label,
                    "accuracy": "",
                    "macro_f1": "",
                    "grounded_needed_f1": "",
                    "delta_remove": "",
                    "delta_macro_f1_vs_baseline": "",
                    "report": str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path),
                    "status": "missing",
                })
                continue

            acc = metric(report, "accuracy")
            macro_f1 = metric(report, "macro_f1")
            grounded_needed_f1 = analysis_metric(report, "grounded_needed_macro_f1")
            delta_remove = analysis_metric(report, "delta_remove")

            if variant_name == "__baseline__" or baseline_macro_f1 != baseline_macro_f1:
                delta_vs_baseline_str = ""
            else:
                delta_vs_baseline_str = format_delta(macro_f1 - baseline_macro_f1)

            rows.append({
                "sweep_group": sweep_group,
                "variant": variant_name,
                "name": label,
                "accuracy": format_float(acc),
                "macro_f1": format_float(macro_f1),
                "grounded_needed_f1": format_float(grounded_needed_f1),
                "delta_remove": format_float(delta_remove),
                "delta_macro_f1_vs_baseline": delta_vs_baseline_str,
                "report": str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path),
                "status": "ok",
            })
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sweep_group", "variant", "name",
        "accuracy", "macro_f1", "grounded_needed_f1", "delta_remove",
        "delta_macro_f1_vs_baseline", "report", "status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Sensitivity / Component Ablation Report (HOVER)\n")
    lines.append(
        "All variants reuse the EGR-FV full pipeline on HOVER with `RoBERTa-base`, "
        "single seed=42 (matching the headline `full_egr_fv_v2` checkpoint). "
        "Baseline row is the unchanged EGR-FV full pipeline.\n"
    )
    lines.append(
        "Columns: `accuracy` and `macro_f1` are on the HOVER test set; "
        "`grounded_needed_f1` is macro-F1 over the routing-assigned grounded_needed group; "
        "`delta_remove` is the F1 drop when evidence is removed (a positive number indicates "
        "the model relies more on evidence on this split); "
        "`Δ macro_f1` is macro-F1 minus the baseline.\n"
    )

    by_group: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["sweep_group"], []).append(row)

    for sweep_group, group_rows in by_group.items():
        title = SWEEP_LABELS.get(sweep_group, sweep_group)
        lines.append(f"## {title}\n")
        lines.append("| Variant | accuracy | macro_f1 | grounded_needed F1 | Δ remove | Δ macro_f1 vs. baseline |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in group_rows:
            if row["status"] == "missing":
                lines.append(f"| {row['name']} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {row['name']} | {row['accuracy']} | {row['macro_f1']} | "
                f"{row['grounded_needed_f1']} | {row['delta_remove']} | "
                f"{row['delta_macro_f1_vs_baseline']} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HOVER sensitivity sweep.")
    parser.add_argument(
        "--output_dir",
        default="outputs/analysis/sensitivity",
        help="Where to write the CSV and Markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()

    baseline_report = load_report(BASELINE_PATH)
    baseline_macro_f1 = (
        metric(baseline_report, "macro_f1") if baseline_report else float("nan")
    )

    rows = collect_rows(baseline_macro_f1)
    csv_path = output_dir / "sensitivity_table.csv"
    md_path = output_dir / "sensitivity_report.md"
    write_csv(rows, csv_path)
    render_markdown(rows, md_path)
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
