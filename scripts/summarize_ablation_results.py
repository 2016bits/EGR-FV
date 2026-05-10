from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("grounded_only", Path("outputs/HOVER/predictions/grounded_only/eval_report.json")),
    ("two_branches_joint", Path("outputs/HOVER/predictions/two_branches_joint/eval_report.json")),
    ("routing_only", Path("outputs/HOVER/predictions/routing_only/eval_report.json")),
    ("random_remix_only", Path("outputs/HOVER/predictions/random_remix_only/eval_report.json")),
    ("full_wo_remix", Path("outputs/HOVER/predictions/full_wo_remix/eval_report.json")),
    ("full_wo_evidence_contrast", Path("outputs/HOVER/predictions/full_wo_evidence_contrast/eval_report.json")),
    ("full_wo_grounded_dominant", Path("outputs/HOVER/predictions/full_wo_grounded_dominant/eval_report.json")),
    ("full_hard_routing", Path("outputs/HOVER/predictions/full_hard_routing/eval_report.json")),
    ("full_in_sample_routing", Path("outputs/HOVER/predictions/full_in_sample_routing/eval_report.json")),
    ("fusion_inference", Path("outputs/HOVER/predictions/fusion_inference/eval_report.json")),
    ("full_egr_fv_v2", Path("outputs/HOVER/predictions/full_egr_fv_v2/eval_report.json")),
]


def _metric(report: dict, key: str) -> float:
    return float(report["base"]["metrics"].get(key, 0.0))


def _sensitivity_drop(report: dict) -> float:
    sensitivity = report.get("evidence_sensitivity", {})
    base_key = report.get("base", {}).get("mode", "grounded")
    base = sensitivity.get(base_key, report["base"]["metrics"])
    remove = sensitivity.get("remove_evidence", {})
    return float(base.get("macro_f1", 0.0)) - float(remove.get("macro_f1", 0.0))


def _analysis(report: dict, key: str) -> float:
    return float(report.get("analysis_metrics", {}).get(key, 0.0) or 0.0)


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
                    "grounded_needed_f1": "",
                    "delta_remove": "",
                    "delta_shuffle": "",
                    "claim_only_gap": "",
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
                "grounded_needed_f1": f"{_analysis(report, 'grounded_needed_macro_f1'):.6f}",
                "delta_remove": f"{_analysis(report, 'delta_remove'):.6f}",
                "delta_shuffle": f"{_analysis(report, 'delta_shuffle'):.6f}",
                "claim_only_gap": f"{_analysis(report, 'claim_only_gap'):.6f}",
                "report": str(path),
            }
        )

    output_path = Path("outputs/HOVER/predictions/ablation_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    print("experiment,mode,accuracy,macro_f1,grounded_needed_f1,delta_remove,delta_shuffle,claim_only_gap")
    for row in rows:
        print(
            f"{row['experiment']},{row['mode']},{row['accuracy']},"
            f"{row['macro_f1']},{row['grounded_needed_f1']},"
            f"{row['delta_remove']},{row['delta_shuffle']},{row['claim_only_gap']}"
        )


if __name__ == "__main__":
    main()
