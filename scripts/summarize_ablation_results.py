from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("full_egr_fv", Path("outputs/predictions/eval_report.json")),
    ("two_branch_baseline", Path("outputs/predictions/ablation_two_branch/eval_report.json")),
    ("no_batch_remix", Path("outputs/predictions/ablation_routing_only/eval_report.json")),
    ("no_sample_weights", Path("outputs/predictions/ablation_real_remix_no_weight/eval_report.json")),
    ("no_routing_random", Path("outputs/predictions/ablation_remix_random/eval_report.json")),
    ("no_routing_heuristic", Path("outputs/predictions/ablation_remix_heuristic/eval_report.json")),
    ("no_fusion", Path("outputs/predictions/ablation_no_fusion/eval_report.json")),
    ("no_orth", Path("outputs/predictions/ablation_no_orth/eval_report.json")),
]


def _metric(report: dict, key: str) -> float:
    return float(report["base"]["metrics"].get(key, 0.0))


def _sensitivity_drop(report: dict) -> float:
    sensitivity = report.get("evidence_sensitivity", {})
    base_key = report.get("base", {}).get("mode", "grounded")
    base = sensitivity.get(base_key, report["base"]["metrics"])
    remove = sensitivity.get("remove_evidence", {})
    return float(base.get("macro_f1", 0.0)) - float(remove.get("macro_f1", 0.0))


def main() -> None:
    rows = []
    for name, path in EXPERIMENTS:
        if not path.exists():
            rows.append(
                {
                    "experiment": name,
                    "mode": "missing",
                    "accuracy": "",
                    "macro_f1": "",
                    "evidence_drop_f1": "",
                    "report": str(path),
                }
            )
            continue

        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        rows.append(
            {
                "experiment": name,
                "mode": report["base"]["mode"],
                "accuracy": f"{_metric(report, 'accuracy'):.6f}",
                "macro_f1": f"{_metric(report, 'macro_f1'):.6f}",
                "evidence_drop_f1": f"{_sensitivity_drop(report):.6f}",
                "report": str(path),
            }
        )

    output_path = Path("outputs/predictions/ablation_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    print("experiment,mode,accuracy,macro_f1,evidence_drop_f1")
    for row in rows:
        print(
            f"{row['experiment']},{row['mode']},{row['accuracy']},"
            f"{row['macro_f1']},{row['evidence_drop_f1']}"
        )


if __name__ == "__main__":
    main()
