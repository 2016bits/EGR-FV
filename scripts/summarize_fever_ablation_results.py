#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_OUTPUT_ROOT = Path("outputs/FEVER/predictions")
REPORT_NAME = "eval_report.json"

EXPERIMENTS: List[Tuple[str, str, Optional[str]]] = [
    ("routing_only", "Routing-only", "routing_only"),
    ("full_wo_evidence_contrast", "Full w/o Evidence Contrast", "full_wo_evidence_contrast"),
    ("full_wo_grounded_dominant", "Full w/o Grounded-dominant", "full_wo_grounded_dominant"),
    ("full_wo_remix", "Full w/o Remix", "full_wo_remix"),
    ("full_hard_routing", "Hard routing", "full_hard_routing"),
    ("full_in_sample_routing", "In-sample routing", "full_in_sample_routing"),
    ("full_egr_fv", "EGR (full pipeline)", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FEVER/symmetric-FEVER ablation reports.")
    parser.add_argument("--root", default=str(DEFAULT_OUTPUT_ROOT), help="FEVER prediction root.")
    parser.add_argument("--output-prefix", default="fever_ablation", help="Output filename prefix.")
    return parser.parse_args()


def report_path(root: Path, experiment_dir: Optional[str], split: str) -> Path:
    if experiment_dir is None:
        if split == "fever":
            return root / REPORT_NAME
        return root / "symmetric_fever" / REPORT_NAME
    return root / experiment_dir / split / REPORT_NAME


def load_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric(report: Dict[str, Any], key: str) -> Optional[float]:
    return as_float(report.get("base", {}).get("metrics", {}).get(key))


def calibration(report: Dict[str, Any], key: str) -> Optional[float]:
    return as_float(report.get("base", {}).get("calibration", {}).get(key))


def analysis_metric(report: Dict[str, Any], key: str) -> Optional[float]:
    analysis = report.get("analysis_metrics", {})
    if key in analysis:
        return as_float(analysis.get(key))
    return as_float(report.get("evidence_sensitivity", {}).get("summary", {}).get(key))


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def fmt_decimal(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def fmt_percent(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def optional_subtract(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def optional_divide(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right in (None, 0.0):
        return None
    return left / right


def build_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for experiment, display_name, experiment_dir in EXPERIMENTS:
        for split in ("fever", "symmetric_fever"):
            path = report_path(root, experiment_dir, split)
            report = load_report(path)
            if report is None:
                rows.append(
                    {
                        "experiment": experiment,
                        "display_name": display_name,
                        "split": split,
                        "mode": "missing",
                        "num_samples": None,
                        "accuracy": None,
                        "macro_precision": None,
                        "macro_recall": None,
                        "macro_f1": None,
                        "supports_f1": None,
                        "refutes_f1": None,
                        "ece": None,
                        "brier": None,
                        "clean_f1": None,
                        "no_evidence_f1": None,
                        "shuffled_evidence_f1": None,
                        "delta_remove": None,
                        "delta_shuffle": None,
                        "claim_only_gap": None,
                        "report_path": display_path(path),
                    }
                )
                continue

            base = report.get("base", {})
            rows.append(
                {
                    "experiment": experiment,
                    "display_name": display_name,
                    "split": split,
                    "mode": str(base.get("mode", "")),
                    "num_samples": as_float(base.get("num_samples")),
                    "accuracy": metric(report, "accuracy"),
                    "macro_precision": metric(report, "macro_precision"),
                    "macro_recall": metric(report, "macro_recall"),
                    "macro_f1": metric(report, "macro_f1"),
                    "supports_f1": metric(report, "supports_f1"),
                    "refutes_f1": metric(report, "refutes_f1"),
                    "ece": calibration(report, "ece"),
                    "brier": calibration(report, "brier"),
                    "clean_f1": analysis_metric(report, "clean_f1"),
                    "no_evidence_f1": analysis_metric(report, "no_evidence_f1"),
                    "shuffled_evidence_f1": analysis_metric(report, "shuffled_evidence_f1"),
                    "delta_remove": analysis_metric(report, "delta_remove"),
                    "delta_shuffle": analysis_metric(report, "delta_shuffle"),
                    "claim_only_gap": analysis_metric(report, "claim_only_gap"),
                    "report_path": display_path(path),
                }
            )
    return rows


def build_robustness_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_experiment: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row["experiment"]), {})[str(row["split"])] = row

    robustness_rows: List[Dict[str, Any]] = []
    for experiment, display_name, _ in EXPERIMENTS:
        splits = by_experiment.get(experiment, {})
        fever = splits.get("fever", {})
        symmetric = splits.get("symmetric_fever", {})
        fever_f1 = as_float(fever.get("macro_f1"))
        symmetric_f1 = as_float(symmetric.get("macro_f1"))
        fever_accuracy = as_float(fever.get("accuracy"))
        symmetric_accuracy = as_float(symmetric.get("accuracy"))
        robustness_rows.append(
            {
                "experiment": experiment,
                "display_name": display_name,
                "fever_macro_f1": fever_f1,
                "symmetric_macro_f1": symmetric_f1,
                "macro_f1_drop": optional_subtract(fever_f1, symmetric_f1),
                "retention": optional_divide(symmetric_f1, fever_f1),
                "fever_accuracy": fever_accuracy,
                "symmetric_accuracy": symmetric_accuracy,
                "accuracy_drop": optional_subtract(fever_accuracy, symmetric_accuracy),
            }
        )
    return robustness_rows


def build_delta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full_by_split = {
        row["split"]: row
        for row in rows
        if row["experiment"] == "full_egr_fv" and row["macro_f1"] is not None
    }
    by_experiment: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row["experiment"]), {})[str(row["split"])] = row

    delta_rows: List[Dict[str, Any]] = []
    for experiment, display_name, _ in EXPERIMENTS:
        if experiment == "full_egr_fv":
            continue
        splits = by_experiment.get(experiment, {})
        fever_f1 = as_float(splits.get("fever", {}).get("macro_f1"))
        sym_f1 = as_float(splits.get("symmetric_fever", {}).get("macro_f1"))
        full_fever = as_float(full_by_split.get("fever", {}).get("macro_f1"))
        full_sym = as_float(full_by_split.get("symmetric_fever", {}).get("macro_f1"))
        delta_rows.append(
            {
                "experiment": experiment,
                "display_name": display_name,
                "fever_macro_f1": fever_f1,
                "fever_delta_vs_full": optional_subtract(fever_f1, full_fever),
                "symmetric_macro_f1": sym_f1,
                "symmetric_delta_vs_full": optional_subtract(sym_f1, full_sym),
            }
        )
    return delta_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            output = {}
            for key, value in row.items():
                if key == "num_samples" and value is not None:
                    output[key] = str(int(float(value)))
                elif isinstance(value, float):
                    output[key] = fmt_decimal(value)
                else:
                    output[key] = value
            writer.writerow(output)


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    robustness_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = ["# FEVER EGR Ablation Results", ""]
    lines.append("## Main Metrics")
    lines.extend(
        markdown_table(
            ["Experiment", "Split", "N", "Accuracy (%)", "Macro-F1 (%)", "Delta Remove (pp)", "Delta Shuffle (pp)"],
            [
                [
                    str(row["display_name"]),
                    str(row["split"]),
                    "" if row["num_samples"] is None else str(int(float(row["num_samples"]))),
                    fmt_percent(as_float(row["accuracy"])),
                    fmt_percent(as_float(row["macro_f1"])),
                    fmt_percent(as_float(row["delta_remove"])),
                    fmt_percent(as_float(row["delta_shuffle"])),
                ]
                for row in rows
            ],
        )
    )
    lines.append("")
    lines.append("## Symmetric Robustness")
    lines.extend(
        markdown_table(
            ["Experiment", "FEVER F1 (%)", "Symmetric F1 (%)", "F1 Drop (pp)", "Retention (%)"],
            [
                [
                    str(row["display_name"]),
                    fmt_percent(as_float(row["fever_macro_f1"])),
                    fmt_percent(as_float(row["symmetric_macro_f1"])),
                    fmt_percent(as_float(row["macro_f1_drop"])),
                    fmt_percent(as_float(row["retention"])),
                ]
                for row in robustness_rows
            ],
        )
    )
    lines.append("")
    lines.append("## Delta Vs Full EGR")
    lines.extend(
        markdown_table(
            ["Experiment", "FEVER F1 (%)", "FEVER Delta (pp)", "Symmetric F1 (%)", "Symmetric Delta (pp)"],
            [
                [
                    str(row["display_name"]),
                    fmt_percent(as_float(row["fever_macro_f1"])),
                    fmt_percent(as_float(row["fever_delta_vs_full"])),
                    fmt_percent(as_float(row["symmetric_macro_f1"])),
                    fmt_percent(as_float(row["symmetric_delta_vs_full"])),
                ]
                for row in delta_rows
            ],
        )
    )
    missing = [row for row in rows if row["mode"] == "missing"]
    if missing:
        lines.append("")
        lines.append("## Missing Reports")
        for row in missing:
            lines.append(f"- {row['display_name']} / {row['split']}: `{row['report_path']}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = build_rows(root)
    robustness_rows = build_robustness_rows(rows)
    delta_rows = build_delta_rows(rows)

    summary_path = root / f"{args.output_prefix}_summary.csv"
    robustness_path = root / f"{args.output_prefix}_robustness.csv"
    delta_path = root / f"{args.output_prefix}_delta_vs_full.csv"
    report_path = root / f"{args.output_prefix}_report.md"

    write_csv(summary_path, rows, rows[0].keys())
    write_csv(robustness_path, robustness_rows, robustness_rows[0].keys())
    write_csv(delta_path, delta_rows, delta_rows[0].keys())
    write_markdown(report_path, rows, robustness_rows, delta_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {robustness_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
