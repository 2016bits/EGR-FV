#!/usr/bin/env python3
"""Consolidate retrieved-evidence FEVER results into a paper-ready table.

Produces a 2×3 comparison: { claim-evidence, EGR-FV } × { gold-train→gold-test,
gold-train→retrieved-test, retrieved-train→retrieved-test }, on both the
original test split (6666 claims) and the symmetric stress split (717 claims).

Inputs (eval_report.json paths):
  Gold-train → gold-test
    outputs/FEVER/baselines/claim_evidence/predictions/fever/eval_report.json
    outputs/FEVER/predictions/eval_report.json
    + corresponding symmetric_fever/eval_report.json

  Gold-train → retrieved-test (MVP)
    outputs/FEVER/baselines/claim_evidence/predictions/retrieved_test/eval_report.json
    outputs/FEVER/predictions/retrieved_test/eval_report.json
    + retrieved_symmetric/eval_report.json

  Retrieved-train → retrieved-test (after retrain)
    outputs/FEVER/retrieved/baselines/claim_evidence/predictions/fever/eval_report.json
    outputs/FEVER/retrieved/predictions/eval_report.json
    + symmetric/eval_report.json

Output:
  outputs/FEVER/retrieved/retrieved_evidence_report.md
  outputs/FEVER/retrieved/retrieved_evidence_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


CELLS: Dict[Tuple[str, str, str], Path] = {
    # (method, train_evidence, test_split): path-to-eval_report.json
    ("claim_evidence", "gold",      "fever"):     Path("outputs/FEVER/baselines/claim_evidence/predictions/fever/eval_report.json"),
    ("claim_evidence", "gold",      "symmetric"): Path("outputs/FEVER/baselines/claim_evidence/predictions/symmetric_fever/eval_report.json"),
    ("egr_fv",         "gold",      "fever"):     Path("outputs/FEVER/predictions/eval_report.json"),
    ("egr_fv",         "gold",      "symmetric"): Path("outputs/FEVER/predictions/symmetric_fever/eval_report.json"),

    ("claim_evidence", "gold_retrieved_test", "fever"):     Path("outputs/FEVER/baselines/claim_evidence/predictions/retrieved_test/eval_report.json"),
    ("claim_evidence", "gold_retrieved_test", "symmetric"): Path("outputs/FEVER/baselines/claim_evidence/predictions/retrieved_symmetric/eval_report.json"),
    ("egr_fv",         "gold_retrieved_test", "fever"):     Path("outputs/FEVER/predictions/retrieved_test/eval_report.json"),
    ("egr_fv",         "gold_retrieved_test", "symmetric"): Path("outputs/FEVER/predictions/retrieved_symmetric/eval_report.json"),

    ("claim_evidence", "retrieved", "fever"):     Path("outputs/FEVER/retrieved/baselines/claim_evidence/predictions/fever/eval_report.json"),
    ("claim_evidence", "retrieved", "symmetric"): Path("outputs/FEVER/retrieved/baselines/claim_evidence/predictions/symmetric/eval_report.json"),
    ("egr_fv",         "retrieved", "fever"):     Path("outputs/FEVER/retrieved/predictions/eval_report.json"),
    ("egr_fv",         "retrieved", "symmetric"): Path("outputs/FEVER/retrieved/predictions/symmetric/eval_report.json"),
}


METHOD_LABEL = {
    "claim_evidence": "claim-evidence",
    "egr_fv": "**Full EGR-FV**",
}
TRAIN_LABEL = {
    "gold":                "Gold → Gold-test (original headline)",
    "gold_retrieved_test": "Gold → Retrieved-test (zero-shot retrieval)",
    "retrieved":           "Retrieved → Retrieved-test (apples-to-apples)",
}
SPLIT_LABEL = {
    "fever":     "FEVER orig (Acc)",
    "symmetric": "FEVER-sym (Acc)",
}


def load_metric(path: Path) -> Optional[Dict[str, Any]]:
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


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/FEVER/retrieved")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect values
    cells: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]] = {}
    for key, rel in CELLS.items():
        cells[key] = load_metric(REPO_ROOT / rel)

    # CSV: long-format
    csv_path = out_dir / "retrieved_evidence_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "train_evidence", "test_split",
                         "num_samples", "accuracy", "macro_f1",
                         "supports_recall", "refutes_recall"])
        for (method, train, split), info in cells.items():
            if info is None:
                writer.writerow([method, train, split, "", "", "", "", ""])
                continue
            writer.writerow([
                method, train, split,
                info["num_samples"],
                fmt(info["accuracy"]),
                fmt(info["macro_f1"]),
                fmt(info["supports_recall"]),
                fmt(info["refutes_recall"]),
            ])

    # Markdown: 3-row × 2-col x 2-method format
    md_lines: List[str] = []
    md_lines.append("# FEVER Retrieved-Evidence Results\n")
    md_lines.append(
        "Compares EGR-FV against the claim-evidence baseline under three "
        "training-evidence regimes:\n\n"
        "1. **Gold → Gold-test** — the headline numbers in the main results "
        "table (`outputs/FEVER/baselines/claim_evidence/...` and "
        "`outputs/FEVER/predictions/...`).\n"
        "2. **Gold → Retrieved-test** — checkpoint trained on gold evidence, "
        "evaluated on BM25 sentences retrieved from FEVER Wikipedia. Measures "
        "robustness of a gold-only model to retrieval noise at inference time.\n"
        "3. **Retrieved → Retrieved-test** — the apples-to-apples comparison "
        "under a realistic FEVER pipeline. Both methods trained on the same "
        "BM25-retrieved evidence (~100k claims after dropping the 9k with "
        "empty candidate-doc retrieval).\n"
    )
    md_lines.append(
        "**Retrieval pipeline**: documents come from FEVER_baseline's "
        "title-fuzzy-match cache (`expanded_doc_ids.pkl`); sentences are "
        "BM25-reranked against the claim; top-5 sentences are concatenated. "
        "Symmetric claims (not in FEVER_baseline cache) use a TF-IDF index "
        "over all 5.4M Wikipedia titles followed by the same BM25 sentence "
        "ranking. See `scripts/retrieve_fever_evidence.py`.\n"
    )

    md_lines.append("## Accuracy (%)\n")
    header = ["Method", "Training evidence", *[SPLIT_LABEL[s] for s in ("fever", "symmetric")]]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "---|" * len(header))
    for method in ("claim_evidence", "egr_fv"):
        for train in ("gold", "gold_retrieved_test", "retrieved"):
            row = [METHOD_LABEL[method], TRAIN_LABEL[train]]
            for split in ("fever", "symmetric"):
                info = cells.get((method, train, split))
                row.append(fmt(info["accuracy"]) if info else "—")
            md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("\n## Macro-F1 (%)\n")
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "---|" * len(header))
    for method in ("claim_evidence", "egr_fv"):
        for train in ("gold", "gold_retrieved_test", "retrieved"):
            row = [METHOD_LABEL[method], TRAIN_LABEL[train]]
            for split in ("fever", "symmetric"):
                info = cells.get((method, train, split))
                row.append(fmt(info["macro_f1"]) if info else "—")
            md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("\n## Per-class recall (%) on FEVER-sym\n")
    md_lines.append(
        "Surfaces label-balance behaviour on the balanced symmetric split. "
        "A model that collapses to one class shows ~100% recall on that class "
        "and ~0% on the other.\n"
    )
    md_lines.append("| Method | Training evidence | recall(supports) | recall(refutes) |")
    md_lines.append("|---|---|---|---|")
    for method in ("claim_evidence", "egr_fv"):
        for train in ("gold", "gold_retrieved_test", "retrieved"):
            info = cells.get((method, train, "symmetric"))
            row = [METHOD_LABEL[method], TRAIN_LABEL[train]]
            row.append(fmt(info["supports_recall"]) if info else "—")
            row.append(fmt(info["refutes_recall"]) if info else "—")
            md_lines.append("| " + " | ".join(row) + " |")

    (out_dir / "retrieved_evidence_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {out_dir / 'retrieved_evidence_report.md'}")
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
