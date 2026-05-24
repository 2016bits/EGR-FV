#!/usr/bin/env python3
"""Consolidate DeBERTa-v3 backbone-generalization results and Llama-3-8B LLM
baselines into a single comparison table.

Produces:
  outputs/analysis/backbone_llm/backbone_llm_report.md
  outputs/analysis/backbone_llm/backbone_llm_report.csv

Each cell is filled from the corresponding `eval_report.json`. Missing files
become "—" (the script still works on partial results, so you can run it before
all trainings finish).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# (backbone, method, dataset, split): eval_report.json path
CELLS: Dict[Tuple[str, str, str, str], Path] = {
    # ---------- RoBERTa-base (original headline) ----------
    # HOVER
    ("roberta_base", "claim_evidence", "hover", "test"):
        Path("outputs/HOVER/baselines/claim_evidence/predictions/eval_report.json"),
    ("roberta_base", "egr_fv", "hover", "test"):
        Path("outputs/HOVER/predictions/eval_report.json"),
    # FEVER
    ("roberta_base", "claim_evidence", "fever", "test"):
        Path("outputs/FEVER/baselines/claim_evidence/predictions/fever/eval_report.json"),
    ("roberta_base", "claim_evidence", "fever", "symmetric"):
        Path("outputs/FEVER/baselines/claim_evidence/predictions/symmetric_fever/eval_report.json"),
    ("roberta_base", "egr_fv", "fever", "test"):
        Path("outputs/FEVER/predictions/eval_report.json"),
    ("roberta_base", "egr_fv", "fever", "symmetric"):
        Path("outputs/FEVER/predictions/symmetric_fever/eval_report.json"),
    # PolitiHop
    ("roberta_base", "claim_evidence", "politihop", "test"):
        Path("outputs/PolitiHop/baselines/claim_evidence/predictions/politihop/eval_report.json"),
    ("roberta_base", "claim_evidence", "politihop", "symmetric"):
        Path("outputs/PolitiHop/baselines/claim_evidence/predictions/symmetric_politihop/eval_report.json"),
    ("roberta_base", "egr_fv", "politihop", "test"):
        Path("outputs/PolitiHop/predictions/eval_report.json"),
    ("roberta_base", "egr_fv", "politihop", "symmetric"):
        Path("outputs/PolitiHop/predictions/symmetric_politihop/eval_report.json"),

    # ---------- DeBERTa-v3-base (generalization) ----------
    # HOVER
    ("deberta_v3", "claim_evidence", "hover", "test"):
        Path("outputs/HOVER/deberta_v3/baselines/claim_evidence/predictions/eval_report.json"),
    ("deberta_v3", "egr_fv", "hover", "test"):
        Path("outputs/HOVER/deberta_v3/predictions/eval_report.json"),
    # FEVER
    ("deberta_v3", "claim_evidence", "fever", "test"):
        Path("outputs/FEVER/deberta_v3/baselines/claim_evidence/predictions/fever/eval_report.json"),
    ("deberta_v3", "claim_evidence", "fever", "symmetric"):
        Path("outputs/FEVER/deberta_v3/baselines/claim_evidence/predictions/symmetric/eval_report.json"),
    ("deberta_v3", "egr_fv", "fever", "test"):
        Path("outputs/FEVER/deberta_v3/predictions/eval_report.json"),
    ("deberta_v3", "egr_fv", "fever", "symmetric"):
        Path("outputs/FEVER/deberta_v3/predictions/symmetric/eval_report.json"),
    # PolitiHop
    ("deberta_v3", "claim_evidence", "politihop", "test"):
        Path("outputs/PolitiHop/deberta_v3/baselines/claim_evidence/predictions/politihop/eval_report.json"),
    ("deberta_v3", "claim_evidence", "politihop", "symmetric"):
        Path("outputs/PolitiHop/deberta_v3/baselines/claim_evidence/predictions/symmetric/eval_report.json"),
    ("deberta_v3", "egr_fv", "politihop", "test"):
        Path("outputs/PolitiHop/deberta_v3/predictions/eval_report.json"),
    ("deberta_v3", "egr_fv", "politihop", "symmetric"):
        Path("outputs/PolitiHop/deberta_v3/predictions/symmetric/eval_report.json"),

    # ---------- Llama-3-8B-Instruct (LLM zero/few-shot) ----------
    ("llama3_8b", "llm_4shot", "fever", "test"):
        Path("outputs/llm_baseline/FEVER/test/eval_report.json"),
    ("llama3_8b", "llm_4shot", "fever", "symmetric"):
        Path("outputs/llm_baseline/FEVER/symmetric/eval_report.json"),
    ("llama3_8b", "llm_4shot", "politihop", "test"):
        Path("outputs/llm_baseline/PolitiHop/test/eval_report.json"),
    ("llama3_8b", "llm_4shot", "politihop", "symmetric"):
        Path("outputs/llm_baseline/PolitiHop/symmetric/eval_report.json"),
    ("llama3_8b", "llm_4shot", "hover", "test"):
        Path("outputs/llm_baseline/HOVER/test/eval_report.json"),
}


BACKBONE_LABEL = {
    "roberta_base": "RoBERTa-base (original)",
    "deberta_v3":   "DeBERTa-v3-base",
    "llama3_8b":    "Llama-3-8B-Instruct (4-shot)",
}
METHOD_LABEL = {
    "claim_evidence": "claim-evidence",
    "egr_fv":         "**EGR-FV**",
    "llm_4shot":      "Llama-3-8B-Instruct (4-shot)",
}
DATASET_LABEL = {
    "hover":     "HOVER",
    "fever":     "FEVER",
    "politihop": "PolitiHop",
}
SPLIT_LABEL = {
    "test":      "Test (Acc / F1)",
    "symmetric": "Symmetric (Acc / F1)",
}


def load_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    base = data.get("base", {})
    metrics = base.get("metrics", {})
    return {
        "num_samples": base.get("num_samples"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "supports_recall": metrics.get("supports_recall"),
        "refutes_recall": metrics.get("refutes_recall"),
    }


def fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/analysis/backbone_llm")
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[Tuple[str, str, str, str], Optional[Dict[str, Any]]] = {}
    for key, rel in CELLS.items():
        data[key] = load_report(REPO_ROOT / rel)

    # CSV long format
    with (out_dir / "backbone_llm_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backbone", "method", "dataset", "split",
                         "num_samples", "accuracy", "macro_f1",
                         "supports_recall", "refutes_recall"])
        for key, info in data.items():
            backbone, method, dataset, split = key
            if info is None:
                writer.writerow([backbone, method, dataset, split,
                                 "", "", "", "", ""])
                continue
            writer.writerow([backbone, method, dataset, split,
                             info["num_samples"],
                             fmt(info["accuracy"]),
                             fmt(info["macro_f1"]),
                             fmt(info["supports_recall"]),
                             fmt(info["refutes_recall"])])

    md: List[str] = []
    md.append("# Backbone Generalization + LLM Baseline\n")
    md.append(
        "Two extensions of the main results:\n\n"
        "1. **Backbone generalization** — re-train `claim-evidence` and full "
        "`EGR-FV` with DeBERTa-v3-base instead of RoBERTa-base, on all three "
        "datasets. Shows the method is not RoBERTa-specific.\n"
        "2. **LLM baseline** — Llama-3-8B-Instruct with a 4-shot prompt "
        "(`SUPPORTS / REFUTES` answers). Frames EGR-FV against a recent "
        "LLM-style verifier without finetuning.\n"
    )

    md.append("## Accuracy / Macro-F1 (%)\n")

    # Table per dataset
    for dataset in ("hover", "fever", "politihop"):
        md.append(f"### {DATASET_LABEL[dataset]}\n")
        header_splits = ["test"]
        if dataset != "hover":
            header_splits.append("symmetric")
        header = ["Backbone", "Method"] + [SPLIT_LABEL[s] for s in header_splits]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        for backbone in ("roberta_base", "deberta_v3", "llama3_8b"):
            for method in ("claim_evidence", "egr_fv", "llm_4shot"):
                key_any = next((k for k in data.keys() if k[0] == backbone and k[1] == method and k[2] == dataset), None)
                if key_any is None:
                    continue
                row = [BACKBONE_LABEL[backbone], METHOD_LABEL[method]]
                any_data = False
                for split in header_splits:
                    info = data.get((backbone, method, dataset, split))
                    if info is None:
                        row.append("—")
                    else:
                        row.append(f"{fmt(info['accuracy'])} / {fmt(info['macro_f1'])}")
                        any_data = True
                if any_data:
                    md.append("| " + " | ".join(row) + " |")
        md.append("")

    md.append("## Per-class recall on FEVER-sym (DeBERTa-v3 + LLM)\n")
    md.append(
        "Same collapse check as the multi-seed report: when a model predicts "
        "only one class on the balanced symmetric set, the absent class has "
        "0 % recall.\n"
    )
    md.append("| Backbone | Method | recall(supports) | recall(refutes) |")
    md.append("|---|---|---|---|")
    for backbone in ("roberta_base", "deberta_v3", "llama3_8b"):
        for method in ("claim_evidence", "egr_fv", "llm_4shot"):
            info = data.get((backbone, method, "fever", "symmetric"))
            if info is None:
                continue
            md.append(
                f"| {BACKBONE_LABEL[backbone]} | {METHOD_LABEL[method]} | "
                f"{fmt(info['supports_recall'])} | {fmt(info['refutes_recall'])} |"
            )

    (out_dir / "backbone_llm_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'backbone_llm_report.md'}")


if __name__ == "__main__":
    main()
