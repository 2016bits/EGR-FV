#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = Path("outputs/FEVER/baselines")
REPORT_NAME = "eval_report.json"
PREDICTION_NAME = "eval_predictions.jsonl"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


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
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def fmt_percent(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "support": "SUPPORTS",
        "supports": "SUPPORTS",
        "supported": "SUPPORTS",
        "entailment": "SUPPORTS",
        "true": "SUPPORTS",
        "yes": "SUPPORTS",
        "refute": "REFUTES",
        "refutes": "REFUTES",
        "refuted": "REFUTES",
        "contradiction": "REFUTES",
        "false": "REFUTES",
        "no": "REFUTES",
        "nei": "NEI",
        "not_enough_info": "NEI",
        "notenoughinfo": "NEI",
        "unknown": "NEI",
    }
    return mapping.get(normalized, text.upper())


def first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def compute_prediction_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs: List[Tuple[str, str]] = []
    missing = 0

    for record in records:
        gold = normalize_label(first_present(record, ["label", "gold_label", "gold"]))
        pred = normalize_label(
            first_present(record, ["pred_label", "prediction", "pred", "predicted_label"])
        )
        if gold is None or pred is None:
            missing += 1
            continue
        pairs.append((gold, pred))

    labels = sorted({label for pair in pairs for label in pair})
    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in labels}
        for gold in labels
    }
    for gold, pred in pairs:
        confusion[gold][pred] += 1

    total = len(pairs)
    correct = sum(1 for gold, pred in pairs if gold == pred)
    per_label: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    return {
        "num_records": len(records),
        "num_scored": total,
        "num_missing_label_or_prediction": missing,
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "labels": labels,
        "per_label": per_label,
        "confusion": confusion,
    }


def discover_reports(root: Path) -> List[Tuple[str, str, Path]]:
    reports: List[Tuple[str, str, Path]] = []
    for report_path in sorted(root.glob(f"*/predictions/*/{REPORT_NAME}")):
        try:
            baseline = report_path.relative_to(root).parts[0]
            split = report_path.parent.name
        except IndexError:
            continue
        reports.append((baseline, split, report_path))
    return reports


def metric(report: Dict[str, Any], key: str) -> Optional[float]:
    return as_float(report.get("base", {}).get("metrics", {}).get(key))


def calibration(report: Dict[str, Any], key: str) -> Optional[float]:
    return as_float(report.get("base", {}).get("calibration", {}).get(key))


def analysis_metric(report: Dict[str, Any], key: str) -> Optional[float]:
    analysis = report.get("analysis_metrics", {})
    if key in analysis:
        return as_float(analysis.get(key))
    return as_float(report.get("evidence_sensitivity", {}).get("summary", {}).get(key))


def build_row(
    baseline: str,
    split: str,
    report_path: Path,
    root: Path,
    load_predictions: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = load_json(report_path)
    base = report.get("base", {})
    prediction_path = report_path.with_name(PREDICTION_NAME)
    prediction_metrics: Dict[str, Any] = {}

    if load_predictions and prediction_path.exists():
        prediction_metrics = compute_prediction_metrics(load_jsonl(prediction_path))
    elif load_predictions:
        prediction_metrics = {"warning": f"Missing prediction file: {prediction_path}"}

    row = {
        "baseline": baseline,
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
        "prediction_num_records": as_float(prediction_metrics.get("num_records")),
        "prediction_num_scored": as_float(prediction_metrics.get("num_scored")),
        "prediction_accuracy": as_float(prediction_metrics.get("accuracy")),
        "prediction_macro_f1": as_float(prediction_metrics.get("macro_f1")),
        "report_path": display_path(report_path),
    }
    return row, prediction_metrics


def build_claim_evidence_gaps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_split: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_split[str(row["split"])][str(row["baseline"])] = row

    gap_rows: List[Dict[str, Any]] = []
    for split, baselines in sorted(by_split.items()):
        claim_only = baselines.get("claim_only")
        claim_evidence = baselines.get("claim_evidence")
        if not claim_only or not claim_evidence:
            continue

        gap_rows.append(
            {
                "split": split,
                "claim_only_accuracy": claim_only["accuracy"],
                "claim_evidence_accuracy": claim_evidence["accuracy"],
                "accuracy_gap": optional_subtract(claim_evidence["accuracy"], claim_only["accuracy"]),
                "claim_only_macro_f1": claim_only["macro_f1"],
                "claim_evidence_macro_f1": claim_evidence["macro_f1"],
                "macro_f1_gap": optional_subtract(claim_evidence["macro_f1"], claim_only["macro_f1"]),
            }
        )
    return gap_rows


def build_robustness_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_baseline: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_baseline[str(row["baseline"])][str(row["split"])] = row

    robustness_rows: List[Dict[str, Any]] = []
    for baseline, splits in sorted(by_baseline.items()):
        fever = splits.get("fever")
        symmetric = splits.get("symmetric_fever")
        if not fever or not symmetric:
            continue

        fever_f1 = fever["macro_f1"]
        symmetric_f1 = symmetric["macro_f1"]
        drop = optional_subtract(fever_f1, symmetric_f1)
        retention = symmetric_f1 / fever_f1 if fever_f1 not in (None, 0.0) and symmetric_f1 is not None else None
        robustness_rows.append(
            {
                "baseline": baseline,
                "fever_macro_f1": fever_f1,
                "symmetric_macro_f1": symmetric_f1,
                "macro_f1_drop": drop,
                "retention": retention,
                "fever_accuracy": fever["accuracy"],
                "symmetric_accuracy": symmetric["accuracy"],
                "accuracy_drop": optional_subtract(fever["accuracy"], symmetric["accuracy"]),
            }
        )
    return robustness_rows


def optional_subtract(left: Any, right: Any) -> Optional[float]:
    left_value = as_float(left)
    right_value = as_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    count_fields = {"num_samples", "prediction_num_records", "prediction_num_scored"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            output_row = {}
            for key, value in row.items():
                if key in count_fields and value is not None:
                    output_row[key] = str(int(value))
                elif isinstance(value, float):
                    output_row[key] = fmt_float(value)
                else:
                    output_row[key] = value
            writer.writerow(output_row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_markdown(
    path: Path,
    root: Path,
    rows: List[Dict[str, Any]],
    gap_rows: List[Dict[str, Any]],
    robustness_rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# FEVER Baseline Evaluation")
    lines.append("")
    lines.append(f"Source root: `{root.as_posix()}`")
    lines.append("")

    lines.append("## Main Metrics")
    lines.extend(
        markdown_table(
            [
                "Baseline",
                "Split",
                "Mode",
                "N",
                "Accuracy (%)",
                "Macro-F1 (%)",
                "Supports F1 (%)",
                "Refutes F1 (%)",
                "ECE",
                "Brier",
            ],
            [
                [
                    str(row["baseline"]),
                    str(row["split"]),
                    str(row["mode"]),
                    str(int(row["num_samples"])) if row["num_samples"] is not None else "",
                    fmt_percent(row["accuracy"]),
                    fmt_percent(row["macro_f1"]),
                    fmt_percent(row["supports_f1"]),
                    fmt_percent(row["refutes_f1"]),
                    fmt_float(row["ece"], 4),
                    fmt_float(row["brier"], 4),
                ]
                for row in sorted(rows, key=lambda item: (str(item["split"]), str(item["baseline"])))
            ],
        )
    )
    lines.append("")

    lines.append("## Evidence Sensitivity")
    lines.extend(
        markdown_table(
            [
                "Baseline",
                "Split",
                "Clean F1 (%)",
                "No Evidence F1 (%)",
                "Shuffled F1 (%)",
                "Delta Remove (pp)",
                "Delta Shuffle (pp)",
            ],
            [
                [
                    str(row["baseline"]),
                    str(row["split"]),
                    fmt_percent(row["clean_f1"]),
                    fmt_percent(row["no_evidence_f1"]),
                    fmt_percent(row["shuffled_evidence_f1"]),
                    fmt_percent(row["delta_remove"]),
                    fmt_percent(row["delta_shuffle"]),
                ]
                for row in sorted(rows, key=lambda item: (str(item["split"]), str(item["baseline"])))
            ],
        )
    )
    lines.append("")

    if gap_rows:
        lines.append("## Claim-Evidence Gains")
        lines.extend(
            markdown_table(
                [
                    "Split",
                    "Claim-Only F1 (%)",
                    "Claim-Evidence F1 (%)",
                    "F1 Gain (pp)",
                    "Accuracy Gain (pp)",
                ],
                [
                    [
                        str(row["split"]),
                        fmt_percent(row["claim_only_macro_f1"]),
                        fmt_percent(row["claim_evidence_macro_f1"]),
                        fmt_percent(row["macro_f1_gap"]),
                        fmt_percent(row["accuracy_gap"]),
                    ]
                    for row in gap_rows
                ],
            )
        )
        lines.append("")

    if robustness_rows:
        lines.append("## Symmetric Robustness")
        lines.extend(
            markdown_table(
                ["Baseline", "FEVER F1 (%)", "Symmetric F1 (%)", "F1 Drop (pp)", "Retention (%)"],
                [
                    [
                        str(row["baseline"]),
                        fmt_percent(row["fever_macro_f1"]),
                        fmt_percent(row["symmetric_macro_f1"]),
                        fmt_percent(row["macro_f1_drop"]),
                        fmt_percent(row["retention"]),
                    ]
                    for row in robustness_rows
                ],
            )
        )
        lines.append("")

    lines.append("Detailed prediction checks and confusion matrices are stored in the JSON output.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(rows: List[Dict[str, Any]], gap_rows: List[Dict[str, Any]], robustness_rows: List[Dict[str, Any]]) -> None:
    print("baseline,split,mode,n,accuracy,macro_f1,delta_shuffle")
    for row in sorted(rows, key=lambda item: (str(item["split"]), str(item["baseline"]))):
        n_value = int(row["num_samples"]) if row["num_samples"] is not None else ""
        print(
            f"{row['baseline']},{row['split']},{row['mode']},{n_value},"
            f"{fmt_float(row['accuracy'])},{fmt_float(row['macro_f1'])},{fmt_float(row['delta_shuffle'])}"
        )

    if gap_rows:
        print("")
        print("claim_evidence_minus_claim_only")
        print("split,accuracy_gap,macro_f1_gap")
        for row in gap_rows:
            print(f"{row['split']},{fmt_float(row['accuracy_gap'])},{fmt_float(row['macro_f1_gap'])}")

    if robustness_rows:
        print("")
        print("symmetric_robustness")
        print("baseline,macro_f1_drop,retention")
        for row in robustness_rows:
            print(f"{row['baseline']},{fmt_float(row['macro_f1_drop'])},{fmt_float(row['retention'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and sanity-check FEVER baseline outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing baseline outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for summary files. Defaults to --root.",
    )
    parser.add_argument(
        "--skip_predictions",
        action="store_true",
        help="Only read eval_report.json files; skip prediction-file checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    output_dir = args.output_dir or root

    if not root.exists():
        raise FileNotFoundError(f"Baseline root not found: {root}")

    discovered = discover_reports(root)
    if not discovered:
        raise FileNotFoundError(
            f"No {REPORT_NAME} files found under {root}. "
            f"Expected paths like {root}/claim_only/predictions/fever/{REPORT_NAME}"
        )

    rows: List[Dict[str, Any]] = []
    prediction_checks: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for baseline, split, report_path in discovered:
        row, prediction_metrics = build_row(
            baseline=baseline,
            split=split,
            report_path=report_path,
            root=root,
            load_predictions=not args.skip_predictions,
        )
        rows.append(row)
        prediction_checks[baseline][split] = prediction_metrics

    gap_rows = build_claim_evidence_gaps(rows)
    robustness_rows = build_robustness_rows(rows)

    summary_csv = output_dir / "fever_baseline_eval_summary.csv"
    gaps_csv = output_dir / "fever_baseline_eval_gaps.csv"
    robustness_csv = output_dir / "fever_baseline_eval_robustness.csv"
    details_json = output_dir / "fever_baseline_eval_details.json"
    markdown_path = output_dir / "fever_baseline_eval_report.md"

    summary_fields = [
        "baseline",
        "split",
        "mode",
        "num_samples",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "supports_f1",
        "refutes_f1",
        "ece",
        "brier",
        "clean_f1",
        "no_evidence_f1",
        "shuffled_evidence_f1",
        "delta_remove",
        "delta_shuffle",
        "claim_only_gap",
        "prediction_num_records",
        "prediction_num_scored",
        "prediction_accuracy",
        "prediction_macro_f1",
        "report_path",
    ]
    write_csv(summary_csv, rows, summary_fields)

    if gap_rows:
        write_csv(
            gaps_csv,
            gap_rows,
            [
                "split",
                "claim_only_accuracy",
                "claim_evidence_accuracy",
                "accuracy_gap",
                "claim_only_macro_f1",
                "claim_evidence_macro_f1",
                "macro_f1_gap",
            ],
        )

    if robustness_rows:
        write_csv(
            robustness_csv,
            robustness_rows,
            [
                "baseline",
                "fever_macro_f1",
                "symmetric_macro_f1",
                "macro_f1_drop",
                "retention",
                "fever_accuracy",
                "symmetric_accuracy",
                "accuracy_drop",
            ],
        )

    write_json(
        details_json,
        {
            "root": str(root),
            "summary": rows,
            "claim_evidence_gaps": gap_rows,
            "symmetric_robustness": robustness_rows,
            "prediction_checks": prediction_checks,
        },
    )
    write_markdown(markdown_path, root, rows, gap_rows, robustness_rows)

    print_console_summary(rows, gap_rows, robustness_rows)
    print("")
    print(f"Wrote {summary_csv}")
    if gap_rows:
        print(f"Wrote {gaps_csv}")
    if robustness_rows:
        print(f"Wrote {robustness_csv}")
    print(f"Wrote {details_json}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
