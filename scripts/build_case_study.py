#!/usr/bin/env python3
"""Build a paper-ready case study from routing-analysis records.

For each dataset, joins `outputs/analysis/routing_analysis/<DS>.jsonl` with
`data/<DS>/converted_data/test.json` and selects representative cases in four
categories that EMNLP reviewers care about:

  A. Evidence-rescue  (grounded_needed group; shortcut wrong; EGR-FV correct
                       under clean evidence, wrong under null evidence)
  B. Evidence-sensitive  (EGR-FV correct only with intact evidence: clean✓,
                          null✗, shuffled✗; highest necessity_score first)
  C. Bias-easy trade-off  (bias_easy group; shortcut correct; EGR-FV wrong) —
                          the cases where remix "sacrifices" easy claim-only wins
  D. Hard win  (hard group; both warmups wrong; EGR-FV correct)

The script writes:
  outputs/analysis/case_study/cases_<DS>.jsonl   # all selected cases
  outputs/analysis/case_study/cases_<DS>.csv     # tabular view (for appendix)
  outputs/analysis/case_study/case_study.md      # paper-ready section

Usage:
  python scripts/build_case_study.py
  python scripts/build_case_study.py --datasets HOVER FEVER --per_category 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


LABEL_MAP_INT2STR = {0: "supports", 1: "refutes"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def pred_label(correct: bool, gold: int) -> int:
    """Infer predicted label id (0/1) from correctness + gold for binary tasks."""
    return gold if correct else (1 - gold)


def select_category_a(records: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Evidence-rescue: grounded_needed AND shortcut wrong AND EGR-FV clean correct
    AND EGR-FV null wrong. Rank by necessity_score desc, then EGR-FV nll_null - nll_clean desc."""
    cands = [
        r for r in records
        if r.get("group") == "grounded_needed"
        and r.get("shortcut_correct") is False
        and r.get("egrfv_correct_clean") is True
        and r.get("egrfv_correct_null") is False
    ]
    cands.sort(
        key=lambda r: (
            -float(r.get("necessity_score") or 0.0),
            -(float(r.get("egrfv_nll_null") or 0.0)
              - float(r.get("egrfv_nll_clean") or 0.0)),
        )
    )
    return cands[:k]


def select_category_b(
    records: Sequence[Dict[str, Any]],
    k: int,
    exclude_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Evidence-sensitive cases that do NOT come from grounded_needed.

    Category A already covers grounded_needed evidence-rescues; B widens the
    picture by surfacing hard/bias_easy samples where EGR-FV still relies on
    intact evidence (clean✓, null✗, shuffled✗). Ranked by NLL gap
    (null − clean) to favour cases where the evidence makes the most
    dramatic difference.
    """
    exclude_ids = exclude_ids or set()
    cands = [
        r for r in records
        if r.get("group") != "grounded_needed"
        and str(r.get("id")) not in exclude_ids
        and r.get("egrfv_correct_clean") is True
        and r.get("egrfv_correct_null") is False
        and r.get("egrfv_correct_shuffled") is False
    ]
    cands.sort(
        key=lambda r: -(float(r.get("egrfv_nll_null") or 0.0)
                        - float(r.get("egrfv_nll_clean") or 0.0))
    )
    return cands[:k]


def select_category_c(records: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Bias-easy trade-off: bias_easy AND shortcut correct AND EGR-FV wrong on clean."""
    cands = [
        r for r in records
        if r.get("group") == "bias_easy"
        and r.get("shortcut_correct") is True
        and r.get("egrfv_correct_clean") is False
    ]
    # most "easy" first: lowest necessity_score
    cands.sort(key=lambda r: float(r.get("necessity_score") or 0.0))
    return cands[:k]


def select_category_d(
    records: Sequence[Dict[str, Any]],
    k: int,
    exclude_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Hard win: hard group AND both warmups wrong AND EGR-FV correct."""
    exclude_ids = exclude_ids or set()
    cands = [
        r for r in records
        if r.get("group") == "hard"
        and str(r.get("id")) not in exclude_ids
        and r.get("shortcut_correct") is False
        and r.get("grounded_correct_clean") is False
        and r.get("egrfv_correct_clean") is True
    ]
    cands.sort(
        key=lambda r: -(float(r.get("egrfv_nll_null") or 0.0)
                        - float(r.get("egrfv_nll_clean") or 0.0))
    )
    return cands[:k]


CATEGORIES = [
    ("A_evidence_rescue", "Evidence-rescue (grounded_needed; claim-only fails, EGR-FV recovers using evidence)",
     select_category_a),
    ("B_evidence_sensitive", "Evidence-sensitive (clean✓, null✗, shuffled✗ — EGR-FV truly relies on the gold evidence)",
     select_category_b),
    ("C_bias_easy_tradeoff", "Bias-easy trade-off (bias_easy; claim-only correct, EGR-FV wrong — cost of de-biasing)",
     select_category_c),
    ("D_hard_win", "Hard-bucket win (both warmups wrong, EGR-FV correct)",
     select_category_d),
]


def enrich(records: Sequence[Dict[str, Any]], texts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join routing records with claim/evidence text by id (string-keyed)."""
    out: List[Dict[str, Any]] = []
    for r in records:
        sid = str(r.get("id"))
        text = texts.get(sid, {})
        merged = dict(r)
        merged["claim"] = text.get("claim", "")
        merged["evidence"] = text.get("evidence", "")
        out.append(merged)
    return out


def to_row(rec: Dict[str, Any], category: str, claim_chars: int, evidence_chars: int) -> Dict[str, Any]:
    gold_int = int(rec.get("label"))
    gold = LABEL_MAP_INT2STR.get(gold_int, str(gold_int))
    sc = LABEL_MAP_INT2STR.get(int(rec.get("shortcut_pred") or 0), "?")
    return {
        "category": category,
        "id": rec.get("id"),
        "num_hops": rec.get("num_hops"),
        "group": rec.get("group"),
        "necessity_score": round(float(rec.get("necessity_score") or 0.0), 3),
        "claim": truncate(str(rec.get("claim") or ""), claim_chars),
        "evidence": truncate(str(rec.get("evidence") or ""), evidence_chars),
        "gold": gold,
        "claim_only_pred": sc,
        "claim_only_correct": rec.get("shortcut_correct"),
        "claim_only_conf": round(float(rec.get("shortcut_conf") or 0.0), 3),
        "claim_evidence_correct_clean": rec.get("grounded_correct_clean"),
        "claim_evidence_conf_clean": round(float(rec.get("grounded_conf_clean") or 0.0), 3),
        "egrfv_correct_clean": rec.get("egrfv_correct_clean"),
        "egrfv_correct_null": rec.get("egrfv_correct_null"),
        "egrfv_correct_shuffled": rec.get("egrfv_correct_shuffled"),
        "egrfv_nll_clean": round(float(rec.get("egrfv_nll_clean") or 0.0), 3),
        "egrfv_nll_null": round(float(rec.get("egrfv_nll_null") or 0.0), 3),
        "egrfv_nll_shuffled": round(float(rec.get("egrfv_nll_shuffled") or 0.0), 3),
    }


def render_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Compact markdown view: one row per case, key fields only."""
    headers = [
        "ID", "group", "r_i", "claim", "evidence", "gold",
        "c-only", "c+e", "EGR-FV (clean / null / shuf)",
        "NLL (clean / null / shuf)",
    ]
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for r in rows:
        sym = {True: "✓", False: "✗", None: "?"}
        egr = " / ".join([
            sym[r["egrfv_correct_clean"]],
            sym[r["egrfv_correct_null"]],
            sym[r["egrfv_correct_shuffled"]],
        ])
        nll = " / ".join([
            f"{r['egrfv_nll_clean']:.2f}",
            f"{r['egrfv_nll_null']:.2f}",
            f"{r['egrfv_nll_shuffled']:.2f}",
        ])
        lines.append("| " + " | ".join([
            f"`{str(r['id'])[:10]}`",
            str(r["group"]),
            f"{r['necessity_score']:.2f}",
            r["claim"],
            r["evidence"],
            r["gold"],
            f"{r['claim_only_pred']}{sym[r['claim_only_correct']]}",
            f"{sym[r['claim_evidence_correct_clean']]}",
            egr,
            nll,
        ]) + " |")
    return "\n".join(lines)


def build_for_dataset(
    dataset: str,
    analysis_dir: Path,
    data_dir: Path,
    out_dir: Path,
    per_category: int,
    claim_chars: int,
    evidence_chars: int,
) -> Optional[List[Tuple[str, str, List[Dict[str, Any]]]]]:
    analysis_path = analysis_dir / f"{dataset}.jsonl"
    data_path = data_dir / dataset / "converted_data" / "test.json"
    if not analysis_path.exists():
        print(f"[warn] routing analysis missing: {analysis_path}")
        return None
    if not data_path.exists():
        # try lowercased
        alt = data_dir / dataset.lower() / "converted_data" / "test.json"
        if alt.exists():
            data_path = alt
        else:
            print(f"[warn] data missing: {data_path}")
            return None

    records = load_jsonl(analysis_path)
    raw = load_json(data_path)
    texts = {str(r["id"]): r for r in raw}
    enriched = enrich(records, texts)

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_per_category: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    all_rows: List[Dict[str, Any]] = []
    all_cases_jsonl: List[Dict[str, Any]] = []

    seen_ids: set = set()
    for cat_id, cat_title, selector in CATEGORIES:
        if cat_id in ("B_evidence_sensitive", "D_hard_win"):
            picks = selector(enriched, per_category, exclude_ids=seen_ids)
        else:
            picks = selector(enriched, per_category)
        rows = [to_row(r, cat_id, claim_chars, evidence_chars) for r in picks]
        selected_per_category.append((cat_id, cat_title, rows))
        all_rows.extend(rows)
        for r in picks:
            seen_ids.add(str(r.get("id")))
            r_copy = dict(r)
            r_copy["case_category"] = cat_id
            all_cases_jsonl.append(r_copy)

    # JSONL
    (out_dir / f"cases_{dataset}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_cases_jsonl) + "\n",
        encoding="utf-8",
    )

    # CSV (tabular for appendix)
    if all_rows:
        with (out_dir / f"cases_{dataset}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

    return selected_per_category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["HOVER", "FEVER"])
    parser.add_argument("--analysis_dir", default="outputs/analysis/routing_analysis")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/analysis/case_study")
    parser.add_argument("--per_category", type=int, default=4,
                        help="Number of cases per category per dataset.")
    parser.add_argument("--claim_chars", type=int, default=140,
                        help="Max characters of claim text shown in the markdown table.")
    parser.add_argument("--evidence_chars", type=int, default=260,
                        help="Max characters of evidence text shown in the markdown table.")
    args = parser.parse_args()

    analysis_dir = REPO_ROOT / args.analysis_dir
    data_dir = REPO_ROOT / args.data_dir
    out_dir = REPO_ROOT / args.output_dir

    md_lines: List[str] = []
    md_lines.append("# EGR-FV Case Study\n")
    md_lines.append(
        "Representative samples drawn from the test split of each dataset, "
        "joined with the per-sample routing-analysis record (`outputs/analysis/"
        "routing_analysis/<DS>.jsonl`). Each block surfaces a different angle "
        "of how EGR-FV behaves relative to the claim-only warm-up (`c-only`) "
        "and the claim-evidence warm-up (`c+e`).\n"
    )
    md_lines.append(
        "Columns under **EGR-FV (clean / null / shuf)** show correctness of "
        "the full model when fed (a) the gold evidence, (b) an empty "
        "placeholder, and (c) evidence randomly swapped from another sample. "
        "Columns under **NLL (clean / null / shuf)** show the corresponding "
        "loss on the gold label — a healthy evidence-grounded model has low "
        "loss under clean evidence and substantially higher loss when "
        "evidence is corrupted.\n"
    )

    for dataset in args.datasets:
        result = build_for_dataset(
            dataset=dataset,
            analysis_dir=analysis_dir,
            data_dir=data_dir,
            out_dir=out_dir,
            per_category=args.per_category,
            claim_chars=args.claim_chars,
            evidence_chars=args.evidence_chars,
        )
        if result is None:
            continue
        md_lines.append(f"\n## {dataset}\n")
        for cat_id, cat_title, rows in result:
            md_lines.append(f"### {cat_id} — {cat_title}\n")
            if not rows:
                md_lines.append("_No matching samples in this dataset._\n")
                continue
            md_lines.append(render_markdown_table(rows))
            md_lines.append("")

    (out_dir / "case_study.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'case_study.md'}")


if __name__ == "__main__":
    main()
