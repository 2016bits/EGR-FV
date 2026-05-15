#!/usr/bin/env python3
"""Aggregate routing-analysis per-sample records into:

  1. Per-group accuracy table (claim-only / claim-evidence / EGR-FV / gain).
  2. Pearson + Spearman correlations between necessity score r_i and
     evidence sensitivity Δ_i, reported in both indicator and NLL forms,
     for both the EGR-FV checkpoint and the grounded warmup checkpoint.
  3. A bucketed routing visualization (clean-correct / shuffled-wrong /
     null-wrong rate per necessity-score bucket), one figure per dataset
     with two panels (EGR-FV vs. grounded warmup).

Reads per-sample JSONL files produced by `scripts/eval_routing_analysis.py`
from <root>/<dataset>.jsonl and writes the report files to <root>/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    _HAVE_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    _HAVE_MPL = True
except Exception:  # pragma: no cover - matplotlib optional
    _HAVE_MPL = False


DATASET_ORDER = ("HOVER", "FEVER", "PolitiHop")
GROUP_ORDER = ("bias_easy", "hard", "grounded_needed")
BUCKET_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
BUCKET_LABELS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize routing analysis.")
    parser.add_argument(
        "--root",
        default="outputs/analysis/routing_analysis",
        help="Directory containing per-dataset JSONL files (e.g. HOVER.jsonl).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_ORDER),
        help="Datasets to include (order is preserved).",
    )
    parser.add_argument(
        "--output-prefix",
        default="routing_analysis",
        help="Filename prefix for the summary files.",
    )
    return parser.parse_args()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _acc(values: Sequence[bool]) -> float:
    if not values:
        return float("nan")
    return float(np.mean([1.0 if value else 0.0 for value in values]))


def _correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    if len(x) < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan")}
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan")}
    if _HAVE_SCIPY:
        pr = pearsonr(x_arr, y_arr)
        sr = spearmanr(x_arr, y_arr)
        return {
            "pearson_r": float(pr.statistic if hasattr(pr, "statistic") else pr[0]),
            "pearson_p": float(pr.pvalue if hasattr(pr, "pvalue") else pr[1]),
            "spearman_r": float(sr.statistic if hasattr(sr, "statistic") else sr[0]),
            "spearman_p": float(sr.pvalue if hasattr(sr, "pvalue") else sr[1]),
        }
    # Fallback: NumPy-only Pearson, and a Spearman that ranks then Pearson-correlates.
    pearson = float(np.corrcoef(x_arr, y_arr)[0, 1])
    ranks_x = np.argsort(np.argsort(x_arr)).astype(np.float64)
    ranks_y = np.argsort(np.argsort(y_arr)).astype(np.float64)
    spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    return {
        "pearson_r": pearson,
        "pearson_p": float("nan"),
        "spearman_r": spearman,
        "spearman_p": float("nan"),
    }


def _bucketize(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    for index, (lo, hi) in enumerate(zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])):
        bucket_records = [
            record for record in records
            if lo <= float(record["necessity_score"]) < hi
        ]
        if not bucket_records:
            buckets.append(
                {
                    "label": BUCKET_LABELS[index],
                    "lo": lo,
                    "hi": min(hi, 1.0),
                    "count": 0,
                }
            )
            continue

        def _rate(field: str, expected: bool) -> float:
            return float(np.mean([1.0 if bool(rec[field]) is expected else 0.0 for rec in bucket_records]))

        buckets.append(
            {
                "label": BUCKET_LABELS[index],
                "lo": lo,
                "hi": min(hi, 1.0),
                "count": len(bucket_records),
                "grounded_clean_correct": _rate("grounded_correct_clean", True),
                "grounded_shuffled_wrong": _rate("grounded_correct_shuffled", False),
                "grounded_null_wrong": _rate("grounded_correct_null", False),
                "egrfv_clean_correct": _rate("egrfv_correct_clean", True),
                "egrfv_shuffled_wrong": _rate("egrfv_correct_shuffled", False),
                "egrfv_null_wrong": _rate("egrfv_correct_null", False),
            }
        )
    return buckets


def _per_dataset_summary(dataset: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"dataset": dataset, "num_samples": len(records)}
    if not records:
        return summary

    summary["overall"] = {
        "claim_only_acc": _acc([bool(rec["shortcut_correct"]) for rec in records]),
        "claim_evidence_acc": _acc([bool(rec["grounded_correct_clean"]) for rec in records]),
        "egrfv_acc": _acc([bool(rec["egrfv_correct_clean"]) for rec in records]),
    }

    groups: Dict[str, Any] = {}
    for group_name in GROUP_ORDER:
        bucket = [rec for rec in records if rec["group"] == group_name]
        co_acc = _acc([bool(rec["shortcut_correct"]) for rec in bucket])
        ce_acc = _acc([bool(rec["grounded_correct_clean"]) for rec in bucket])
        eg_acc = _acc([bool(rec["egrfv_correct_clean"]) for rec in bucket])
        groups[group_name] = {
            "count": len(bucket),
            "share": (len(bucket) / len(records)) if records else 0.0,
            "claim_only_acc": co_acc,
            "claim_evidence_acc": ce_acc,
            "egrfv_acc": eg_acc,
            "gain_egrfv_vs_claim_only": eg_acc - co_acc if not (np.isnan(eg_acc) or np.isnan(co_acc)) else float("nan"),
        }
    summary["groups"] = groups

    r_vals = [float(rec["necessity_score"]) for rec in records]

    def _delta_indicator(model: str) -> List[float]:
        with_e = f"{model}_correct_clean"
        no_e = f"{model}_correct_null"
        return [
            (1.0 if bool(rec[with_e]) else 0.0) - (1.0 if bool(rec[no_e]) else 0.0)
            for rec in records
        ]

    def _delta_nll(model: str) -> List[float]:
        with_e = f"{model}_nll_clean"
        no_e = f"{model}_nll_null"
        # Δ_NLL_i = ℓ(c,∅) − ℓ(c,e): high when adding evidence reduces loss.
        return [float(rec[no_e]) - float(rec[with_e]) for rec in records]

    summary["correlations"] = {
        "grounded_warmup": {
            "indicator": _correlation(r_vals, _delta_indicator("grounded")),
            "nll": _correlation(r_vals, _delta_nll("grounded")),
        },
        "egrfv": {
            "indicator": _correlation(r_vals, _delta_indicator("egrfv")),
            "nll": _correlation(r_vals, _delta_nll("egrfv")),
        },
    }

    summary["buckets"] = _bucketize(records)
    return summary


# ---------------------------------- output ------------------------------------

def _format_pct(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{100.0 * value:.{digits}f}"


def _format_signed_pp(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{100.0 * value:+.{digits}f}"


def _format_rho(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:+.{digits}f}"


def _write_group_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset", "group", "count", "share",
        "claim_only_acc", "claim_evidence_acc", "egrfv_acc", "gain_egrfv_vs_claim_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for group_name in GROUP_ORDER:
                group = summary.get("groups", {}).get(group_name, {})
                writer.writerow(
                    {
                        "dataset": summary["dataset"],
                        "group": group_name,
                        "count": group.get("count", 0),
                        "share": group.get("share", float("nan")),
                        "claim_only_acc": group.get("claim_only_acc", float("nan")),
                        "claim_evidence_acc": group.get("claim_evidence_acc", float("nan")),
                        "egrfv_acc": group.get("egrfv_acc", float("nan")),
                        "gain_egrfv_vs_claim_only": group.get("gain_egrfv_vs_claim_only", float("nan")),
                    }
                )


def _write_correlation_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    fieldnames = ["dataset", "model", "delta_kind", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            correlations = summary.get("correlations", {})
            for model_name in ("grounded_warmup", "egrfv"):
                for delta_kind in ("indicator", "nll"):
                    row = correlations.get(model_name, {}).get(delta_kind, {})
                    writer.writerow(
                        {
                            "dataset": summary["dataset"],
                            "model": model_name,
                            "delta_kind": delta_kind,
                            "pearson_r": row.get("pearson_r", float("nan")),
                            "pearson_p": row.get("pearson_p", float("nan")),
                            "spearman_r": row.get("spearman_r", float("nan")),
                            "spearman_p": row.get("spearman_p", float("nan")),
                        }
                    )


def _write_bucket_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset", "bucket", "count",
        "grounded_clean_correct", "grounded_shuffled_wrong", "grounded_null_wrong",
        "egrfv_clean_correct", "egrfv_shuffled_wrong", "egrfv_null_wrong",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for bucket in summary.get("buckets", []):
                writer.writerow(
                    {
                        "dataset": summary["dataset"],
                        "bucket": bucket["label"],
                        "count": bucket["count"],
                        "grounded_clean_correct": bucket.get("grounded_clean_correct", float("nan")),
                        "grounded_shuffled_wrong": bucket.get("grounded_shuffled_wrong", float("nan")),
                        "grounded_null_wrong": bucket.get("grounded_null_wrong", float("nan")),
                        "egrfv_clean_correct": bucket.get("egrfv_clean_correct", float("nan")),
                        "egrfv_shuffled_wrong": bucket.get("egrfv_shuffled_wrong", float("nan")),
                        "egrfv_null_wrong": bucket.get("egrfv_null_wrong", float("nan")),
                    }
                )


def _write_markdown(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines: List[str] = ["# Routing Analysis", ""]

    lines.append("## 1. Per-Group Performance")
    lines.append("")
    lines.append(
        "For each test sample we compute the necessity score `r_i` from the "
        "warm-up shortcut / grounded models and assign it to a routing group "
        "(`bias_easy` / `hard` / `grounded_needed`) using the same thresholds "
        "as training. We then evaluate three models on each group: the "
        "claim-only warm-up, the claim-evidence warm-up, and the full EGR-FV."
    )
    lines.append("")
    for summary in summaries:
        lines.append(f"### {summary['dataset']} (N={summary['num_samples']})")
        lines.append("")
        lines.append("| Group | % Samples | claim-only Acc | claim-evidence Acc | EGR-FV Acc | Gain (EGR-FV − claim-only) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        groups = summary.get("groups", {})
        for group_name in GROUP_ORDER:
            group = groups.get(group_name, {})
            share = group.get("share", float("nan"))
            lines.append(
                "| {name} | {share} | {co} | {ce} | {eg} | {gain} |".format(
                    name=group_name,
                    share=_format_pct(share, digits=1),
                    co=_format_pct(group.get("claim_only_acc", float("nan"))),
                    ce=_format_pct(group.get("claim_evidence_acc", float("nan"))),
                    eg=_format_pct(group.get("egrfv_acc", float("nan"))),
                    gain=_format_signed_pp(group.get("gain_egrfv_vs_claim_only", float("nan"))),
                )
            )
        lines.append("")

    lines.append("## 2. Correlation of necessity score with evidence sensitivity")
    lines.append("")
    lines.append(
        "We report Pearson and Spearman correlations between the necessity "
        "score `r_i` and the per-sample evidence-sensitivity Δ_i. Two forms of "
        "Δ_i are reported:"
    )
    lines.append("")
    lines.append("- **indicator**: Δ_i = 1[f(c,e)=y] − 1[f(c,∅)=y]")
    lines.append("- **NLL**: Δ_i = ℓ(f(c,∅), y) − ℓ(f(c,e), y)")
    lines.append("")
    lines.append("Positive correlation ⇒ samples that the routing score flags as evidence-needed are exactly the ones whose predictions improve when evidence is added.")
    lines.append("")
    lines.append("| Dataset | Model | Δ form | Pearson r | Spearman ρ |")
    lines.append("|---|---|---|---:|---:|")
    for summary in summaries:
        correlations = summary.get("correlations", {})
        for model_name, model_display in (("grounded_warmup", "claim-evidence (warmup)"), ("egrfv", "EGR-FV")):
            for delta_kind in ("indicator", "nll"):
                row = correlations.get(model_name, {}).get(delta_kind, {})
                lines.append(
                    "| {ds} | {model} | {kind} | {pr} | {sr} |".format(
                        ds=summary["dataset"],
                        model=model_display,
                        kind=delta_kind,
                        pr=_format_rho(row.get("pearson_r", float("nan"))),
                        sr=_format_rho(row.get("spearman_r", float("nan"))),
                    )
                )
    lines.append("")

    lines.append("## 3. Routing Visualization (bucketed)")
    lines.append("")
    lines.append(
        "Test samples are bucketed by necessity score r_i into 0.0-0.2, "
        "0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0. For each bucket we report the "
        "clean-correct, shuffled-wrong, and null-wrong rates of the EGR-FV "
        "checkpoint (the grounded warm-up version is in the CSV)."
    )
    lines.append("")
    for summary in summaries:
        lines.append(f"### {summary['dataset']}")
        lines.append("")
        lines.append("| r_i bucket | count | clean-correct | shuffled-wrong | null-wrong |")
        lines.append("|---|---:|---:|---:|---:|")
        for bucket in summary.get("buckets", []):
            if bucket["count"] == 0:
                lines.append(f"| {bucket['label']} | 0 | - | - | - |")
                continue
            lines.append(
                "| {label} | {count} | {clean} | {shuffled} | {null} |".format(
                    label=bucket["label"],
                    count=bucket["count"],
                    clean=_format_pct(bucket["egrfv_clean_correct"]),
                    shuffled=_format_pct(bucket["egrfv_shuffled_wrong"]),
                    null=_format_pct(bucket["egrfv_null_wrong"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_buckets(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not _HAVE_MPL:
        return
    valid = [summary for summary in summaries if summary.get("buckets")]
    if not valid:
        return
    num_rows = len(valid)
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 3.0 * num_rows), squeeze=False)
    for row_index, summary in enumerate(valid):
        buckets = summary["buckets"]
        labels = [bucket["label"] for bucket in buckets]
        counts = np.asarray([bucket["count"] for bucket in buckets])
        x = np.arange(len(labels))
        width = 0.27
        for col_index, model_prefix, model_title in (
            (0, "grounded", "claim-evidence (warmup)"),
            (1, "egrfv", "EGR-FV"),
        ):
            ax = axes[row_index][col_index]
            clean = np.asarray(
                [bucket.get(f"{model_prefix}_clean_correct", float("nan")) for bucket in buckets]
            )
            shuffled = np.asarray(
                [bucket.get(f"{model_prefix}_shuffled_wrong", float("nan")) for bucket in buckets]
            )
            null = np.asarray(
                [bucket.get(f"{model_prefix}_null_wrong", float("nan")) for bucket in buckets]
            )
            bar1 = ax.bar(x - width, clean, width, label="clean-correct", color="#4C72B0")
            bar2 = ax.bar(x, shuffled, width, label="shuffled-wrong", color="#DD8452")
            bar3 = ax.bar(x + width, null, width, label="null-wrong", color="#55A868")
            for index, count in enumerate(counts):
                ax.text(index, 1.02, f"n={count}", ha="center", va="bottom", fontsize=7, color="#555")
            ax.set_ylim(0.0, 1.15)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_xlabel("necessity score r_i")
            ax.set_ylabel("rate")
            ax.set_title(f"{summary['dataset']} — {model_title}")
            ax.grid(axis="y", alpha=0.3)
            if row_index == 0 and col_index == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        records = _load_records(root / f"{dataset}.jsonl")
        if not records:
            print(f"[skip] {dataset}: per-sample file is missing or empty.")
            continue
        summaries.append(_per_dataset_summary(dataset, records))

    if not summaries:
        print("No datasets to summarize.")
        return

    root.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    _write_group_csv(root / f"{prefix}_groups.csv", summaries)
    _write_correlation_csv(root / f"{prefix}_correlations.csv", summaries)
    _write_bucket_csv(root / f"{prefix}_buckets.csv", summaries)
    _write_markdown(root / f"{prefix}_report.md", summaries)
    _plot_buckets(root / f"{prefix}_buckets.png", summaries)

    with (root / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    print(f"Wrote summary files under {root}/")
    for summary in summaries:
        overall = summary.get("overall", {})
        print(
            f"  {summary['dataset']}: N={summary['num_samples']}  "
            f"claim-only={overall.get('claim_only_acc', float('nan')):.4f}  "
            f"claim-evidence={overall.get('claim_evidence_acc', float('nan')):.4f}  "
            f"EGR-FV={overall.get('egrfv_acc', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
