#!/usr/bin/env python3
"""Aggregate multi-seed evaluation runs into mean +- std with paired significance tests.

For each (dataset, method) combo, this script:
  1. Loads per-seed `eval_predictions.jsonl` produced by `run_significance.sh`
  2. Aligns predictions across methods by sample id within the same (dataset, seed)
  3. Reports mean +- std of the headline metric (accuracy for FEVER-sym / PolitiHop-sym,
     macro-F1 for HOVER)
  4. Runs two paired significance tests vs. the `claim_evidence` baseline:
       - paired bootstrap (default 10,000 resamples)
       - approximate randomization (default 10,000 permutations)
     Per-seed p-values are combined into a single number using Fisher's method.

Outputs:
  outputs/significance/significance_table.csv
  outputs/significance/significance_pvalues.csv
  outputs/significance/significance_report.md
  outputs/significance/significance_summary.json

Usage:
  python scripts/summarize_significance.py
  python scripts/summarize_significance.py --datasets fever politihop --seeds 13 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score


# (config_orig, sym_config_or_None, prediction_subdir_orig, prediction_subdir_sym, eval_metric)
# We point at the *prediction_dir* base; the per-seed run_tag is appended automatically.
# eval_metric is the headline metric reported in the table.

# Predictions land at <prediction_dir>/<run_tag>/eval_predictions.jsonl.
# We track BOTH the original-test prediction_dir and (optionally) the symmetric one
# because FEVER-sym and PolitiHop-sym are the metric of interest for those datasets.

COMBO_TABLE: Dict[Tuple[str, str], Dict[str, Any]] = {
    # --- FEVER -------------------------------------------------------------
    ("fever", "claim_evidence"): {
        "pred_dir_orig": "outputs/FEVER/baselines/claim_evidence/predictions/fever",
        "pred_dir_sym":  "outputs/FEVER/baselines/claim_evidence/predictions/symmetric_fever",
    },
    ("fever", "two_branch"): {
        "pred_dir_orig": "outputs/FEVER/predictions/two_branches_joint/fever",
        "pred_dir_sym":  "outputs/FEVER/predictions/two_branches_joint/symmetric_fever",
    },
    ("fever", "full_wo_remix"): {
        "pred_dir_orig": "outputs/FEVER/predictions/full_wo_remix/fever",
        "pred_dir_sym":  "outputs/FEVER/predictions/full_wo_remix/symmetric_fever",
    },
    ("fever", "full_egr_fv"): {
        "pred_dir_orig": "outputs/FEVER/predictions",
        "pred_dir_sym":  "outputs/FEVER/predictions/symmetric_fever",
    },
    # --- PolitiHop ---------------------------------------------------------
    ("politihop", "claim_evidence"): {
        "pred_dir_orig": "outputs/PolitiHop/baselines/claim_evidence/predictions/politihop",
        "pred_dir_sym":  "outputs/PolitiHop/baselines/claim_evidence/predictions/symmetric_politihop",
    },
    ("politihop", "two_branch"): {
        "pred_dir_orig": "outputs/PolitiHop/predictions/two_branches_joint/politihop",
        "pred_dir_sym":  "outputs/PolitiHop/predictions/two_branches_joint/symmetric_politihop",
    },
    ("politihop", "full_wo_remix"): {
        "pred_dir_orig": "outputs/PolitiHop/predictions/full_wo_remix",
        "pred_dir_sym":  "outputs/PolitiHop/predictions/full_wo_remix/symmetric_politihop",
    },
    ("politihop", "full_egr_fv"): {
        "pred_dir_orig": "outputs/PolitiHop/predictions",
        "pred_dir_sym":  "outputs/PolitiHop/predictions/symmetric_politihop",
    },
    # --- HOVER (no symmetric split; metric is macro-F1 on test) -----------
    ("hover", "claim_evidence"): {
        "pred_dir_orig": "outputs/HOVER/baselines/claim_evidence/predictions",
        "pred_dir_sym":  None,
    },
    ("hover", "two_branch"): {
        "pred_dir_orig": "outputs/HOVER/predictions/two_branches_joint",
        "pred_dir_sym":  None,
    },
    ("hover", "full_wo_remix"): {
        "pred_dir_orig": "outputs/HOVER/predictions/full_wo_remix",
        "pred_dir_sym":  None,
    },
    ("hover", "full_egr_fv"): {
        "pred_dir_orig": "outputs/HOVER/predictions/full_egr_fv_v2",
        "pred_dir_sym":  None,
    },

    # --- Tier-Should extended methods --------------------------------------
    # FEVER (orig at <base>/fever, sym at <base>/symmetric_fever, mirrors two_branch)
    ("fever", "hard_routing"): {
        "pred_dir_orig": "outputs/FEVER/predictions/full_hard_routing/fever",
        "pred_dir_sym":  "outputs/FEVER/predictions/full_hard_routing/symmetric_fever",
    },
    ("fever", "random_remix_only"): {
        "pred_dir_orig": "outputs/FEVER/predictions/random_remix_only/fever",
        "pred_dir_sym":  "outputs/FEVER/predictions/random_remix_only/symmetric_fever",
    },
    ("fever", "wo_evidence_contrast"): {
        "pred_dir_orig": "outputs/FEVER/predictions/full_wo_evidence_contrast/fever",
        "pred_dir_sym":  "outputs/FEVER/predictions/full_wo_evidence_contrast/symmetric_fever",
    },

    # PolitiHop (orig at <base>, sym at <base>/symmetric_politihop, mirrors full_wo_remix)
    ("politihop", "hard_routing"): {
        "pred_dir_orig": "outputs/PolitiHop/predictions/full_hard_routing",
        "pred_dir_sym":  "outputs/PolitiHop/predictions/full_hard_routing/symmetric_politihop",
    },
    ("politihop", "random_remix_only"): {
        "pred_dir_orig": "outputs/PolitiHop/predictions/random_remix_only",
        "pred_dir_sym":  "outputs/PolitiHop/predictions/random_remix_only/symmetric_politihop",
    },
    ("politihop", "wo_evidence_contrast"): {
        "pred_dir_orig": "outputs/PolitiHop/predictions/full_wo_evidence_contrast",
        "pred_dir_sym":  "outputs/PolitiHop/predictions/full_wo_evidence_contrast/symmetric_politihop",
    },

    # HOVER (orig only; no symmetric split exists for HOVER)
    ("hover", "hard_routing"): {
        "pred_dir_orig": "outputs/HOVER/predictions/full_hard_routing",
        "pred_dir_sym":  None,
    },
    ("hover", "random_remix_only"): {
        "pred_dir_orig": "outputs/HOVER/predictions/random_remix_only",
        "pred_dir_sym":  None,
    },
    ("hover", "wo_evidence_contrast"): {
        "pred_dir_orig": "outputs/HOVER/predictions/full_wo_evidence_contrast",
        "pred_dir_sym":  None,
    },
}


# Which prediction file is the "headline" for each dataset, and which metric to report.
DATASET_HEADLINE: Dict[str, Dict[str, str]] = {
    "fever":     {"split": "sym",  "metric": "accuracy", "label": "FEVER-sym (Acc)"},
    "politihop": {"split": "sym",  "metric": "accuracy", "label": "PolitiHop-sym (Acc)"},
    "hover":     {"split": "orig", "metric": "macro_f1", "label": "HOVER (Macro-F1)"},
}


METHOD_LABELS = {
    "claim_evidence":       "claim-evidence",
    "two_branch":           "Two-branch joint",
    "full_wo_remix":        "Full w/o remix",
    "hard_routing":         "Hard routing",
    "random_remix_only":    "Random remix only",
    "wo_evidence_contrast": "Full w/o evidence-contrast",
    "full_egr_fv":          "**Full EGR-FV**",
}


# ----- IO --------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_pred_dir(dataset: str, method: str, split: str) -> Optional[str]:
    info = COMBO_TABLE.get((dataset, method))
    if info is None:
        return None
    key = "pred_dir_orig" if split == "orig" else "pred_dir_sym"
    return info.get(key)


def load_predictions_for(
    dataset: str,
    method: str,
    seed: int,
    split: str,
) -> Optional[List[Dict[str, Any]]]:
    pred_dir = resolve_pred_dir(dataset, method, split)
    if not pred_dir:
        return None
    pred_path = Path(pred_dir) / f"seed{seed}" / "eval_predictions.jsonl"
    if not pred_path.exists():
        return None
    return read_jsonl(pred_path)


# ----- Metrics ---------------------------------------------------------------


def _normalize(text: Any) -> str:
    return str(text).strip().lower() if text is not None else ""


def per_sample_correct(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Return [(sample_id, 1/0 correct)] for accuracy-style metrics."""
    out: List[Tuple[str, int]] = []
    for rec in records:
        sid = rec.get("id")
        gold = _normalize(rec.get("label"))
        pred = _normalize(rec.get("pred_label"))
        if sid is None or not gold or not pred or gold == "none":
            continue
        out.append((str(sid), int(gold == pred)))
    return out


def per_sample_pairs(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Return [(sample_id, gold, pred)] for macro-F1 computation."""
    out: List[Tuple[str, str, str]] = []
    for rec in records:
        sid = rec.get("id")
        gold = _normalize(rec.get("label"))
        pred = _normalize(rec.get("pred_label"))
        if sid is None or not gold or not pred or gold == "none":
            continue
        out.append((str(sid), gold, pred))
    return out


def macro_f1(golds: Sequence[str], preds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    return float(f1_score(list(golds), list(preds), average="macro"))


def accuracy(golds: Sequence[int]) -> float:
    if not golds:
        return 0.0
    return float(np.mean(golds))


def per_class_recall(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return {class_label_lower: recall} for every class that has at least one gold sample.

    Used to expose minority-class collapse on imbalanced datasets (e.g. PolitiHop
    where claim-evidence predicts all-REFUTES and SUPPORTS recall is 0).
    """
    pairs = per_sample_pairs(records)
    counts: Dict[str, Tuple[int, int]] = {}
    for _, g, p in pairs:
        tp, n = counts.get(g, (0, 0))
        counts[g] = (tp + int(g == p), n + 1)
    return {cls: (tp / n if n > 0 else 0.0) for cls, (tp, n) in counts.items()}


def collapse_indicator(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    """If predictions all share one label, return that label; else None.

    Helps flag baselines that collapsed to a single class on a balanced
    stress-test split.
    """
    preds = {_normalize(r.get("pred_label")) for r in records if r.get("pred_label") is not None}
    if len(preds) == 1:
        return preds.pop()
    return None


# ----- Paired tests ----------------------------------------------------------


def align_by_id(
    method_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_index = {str(r.get("id")): r for r in baseline_records if r.get("id") is not None}
    aligned_m: List[Dict[str, Any]] = []
    aligned_b: List[Dict[str, Any]] = []
    for rec in method_records:
        sid = str(rec.get("id")) if rec.get("id") is not None else None
        if sid is None or sid not in base_index:
            continue
        aligned_m.append(rec)
        aligned_b.append(base_index[sid])
    return aligned_m, aligned_b


def _metric_from_records(records: Sequence[Dict[str, Any]], metric: str) -> float:
    if metric == "accuracy":
        return accuracy([c for _, c in per_sample_correct(records)])
    if metric == "macro_f1":
        pairs = per_sample_pairs(records)
        return macro_f1([g for _, g, _ in pairs], [p for _, _, p in pairs])
    raise ValueError(f"Unknown metric: {metric}")


def _encode_pairs(
    records: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (gold_ids, pred_ids, label_universe) as int arrays for vectorized metrics."""
    pairs = per_sample_pairs(records)
    if not pairs:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), []
    universe = sorted({g for _, g, _ in pairs} | {p for _, _, p in pairs})
    lut = {label: i for i, label in enumerate(universe)}
    gold = np.fromiter((lut[g] for _, g, _ in pairs), dtype=np.int32, count=len(pairs))
    pred = np.fromiter((lut[p] for _, _, p in pairs), dtype=np.int32, count=len(pairs))
    return gold, pred, universe


def _vec_accuracy(correct: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean correctness across resamples (B, N)-shaped index → (B,)."""
    return correct[idx].mean(axis=1)


def _vec_macro_f1(gold: np.ndarray, pred: np.ndarray, idx: np.ndarray, num_labels: int) -> np.ndarray:
    """Macro-F1 per resample using vectorized confusion counts.

    Args:
        gold: (N,) int gold labels.
        pred: (N,) int predicted labels.
        idx:  (B, N) int sample indices for B resamples (with replacement).
    Returns:
        (B,) macro-F1 scores.
    """
    g = gold[idx]   # (B, N)
    p = pred[idx]   # (B, N)
    B = idx.shape[0]
    L = num_labels
    f1s = np.zeros((B, L), dtype=np.float64)
    for c in range(L):
        tp = ((g == c) & (p == c)).sum(axis=1)
        fp = ((g != c) & (p == c)).sum(axis=1)
        fn = ((g == c) & (p != c)).sum(axis=1)
        denom = (2 * tp + fp + fn)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s[:, c] = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return f1s.mean(axis=1)


def _prepare_paired(
    method_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
    metric: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Align by id and encode into int arrays. Returns (mg, mp, bg, bp, L) or None if empty.

    For accuracy: mg/bg hold correctness (0/1), mp/bp are unused — collapsed to that.
    Caller distinguishes via `metric`.
    """
    aligned_m, aligned_b = align_by_id(method_records, baseline_records)
    if not aligned_m:
        return None
    if metric == "accuracy":
        # We re-extract correctness arrays aligned in the same order.
        mc = np.array([1 if _normalize(r.get("pred_label")) == _normalize(r.get("label")) else 0
                       for r in aligned_m], dtype=np.int32)
        bc = np.array([1 if _normalize(r.get("pred_label")) == _normalize(r.get("label")) else 0
                       for r in aligned_b], dtype=np.int32)
        return mc, mc, bc, bc, 0
    if metric == "macro_f1":
        mg, mp, m_uni = _encode_pairs(aligned_m)
        bg, bp, b_uni = _encode_pairs(aligned_b)
        # Re-encode under a common universe so class ids match.
        universe = sorted(set(m_uni) | set(b_uni))
        if universe != m_uni or universe != b_uni:
            m_remap = {m_uni[i]: i for i in range(len(m_uni))}
            b_remap = {b_uni[i]: i for i in range(len(b_uni))}
            new = {label: i for i, label in enumerate(universe)}
            mg = np.array([new[m_uni[v]] for v in mg], dtype=np.int32)
            mp = np.array([new[m_uni[v]] for v in mp], dtype=np.int32)
            bg = np.array([new[b_uni[v]] for v in bg], dtype=np.int32)
            bp = np.array([new[b_uni[v]] for v in bp], dtype=np.int32)
        return mg, mp, bg, bp, len(universe)
    raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_p(
    method_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
    metric: str,
    n_resamples: int = 10000,
    seed: int = 0,
    chunk: int = 256,
) -> Tuple[float, float]:
    """Two-sided p-value via paired bootstrap. Returns (observed_delta, p_value).

    Vectorized over resamples: builds a (chunk, N) index matrix per chunk and
    computes the metric for all resamples at once.
    """
    prep = _prepare_paired(method_records, baseline_records, metric)
    if prep is None:
        return 0.0, 1.0
    mg, mp, bg, bp, L = prep
    n = mg.shape[0]
    rng = np.random.default_rng(seed)

    if metric == "accuracy":
        observed = float(mg.mean() - bg.mean())
    else:
        f1_m = _vec_macro_f1(mg, mp, np.arange(n)[None, :], L)[0]
        f1_b = _vec_macro_f1(bg, bp, np.arange(n)[None, :], L)[0]
        observed = float(f1_m - f1_b)

    extreme = 0
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(b, n))
        if metric == "accuracy":
            delta = mg[idx].mean(axis=1) - bg[idx].mean(axis=1)
        else:
            delta = _vec_macro_f1(mg, mp, idx, L) - _vec_macro_f1(bg, bp, idx, L)
        extreme += int(np.sum(np.abs(delta - observed) >= abs(observed)))
        done += b
    p_value = (extreme + 1) / (n_resamples + 1)
    return float(observed), float(p_value)


def approximate_randomization_p(
    method_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
    metric: str,
    n_iters: int = 10000,
    seed: int = 0,
    chunk: int = 256,
) -> Tuple[float, float]:
    """Two-sided p-value via paired approximate randomization, vectorized."""
    prep = _prepare_paired(method_records, baseline_records, metric)
    if prep is None:
        return 0.0, 1.0
    mg, mp, bg, bp, L = prep
    n = mg.shape[0]
    rng = np.random.default_rng(seed)

    if metric == "accuracy":
        observed = float(mg.mean() - bg.mean())
    else:
        f1_m = _vec_macro_f1(mg, mp, np.arange(n)[None, :], L)[0]
        f1_b = _vec_macro_f1(bg, bp, np.arange(n)[None, :], L)[0]
        observed = float(f1_m - f1_b)

    extreme = 0
    done = 0
    while done < n_iters:
        b = min(chunk, n_iters - done)
        swap = rng.integers(0, 2, size=(b, n)).astype(bool)
        if metric == "accuracy":
            # broadcast: swap chooses where to pick baseline vs method
            m_correct = np.where(swap, bg[None, :], mg[None, :])
            b_correct = np.where(swap, mg[None, :], bg[None, :])
            delta = m_correct.mean(axis=1) - b_correct.mean(axis=1)
        else:
            m_g = np.where(swap, bg[None, :], mg[None, :])
            m_p = np.where(swap, bp[None, :], mp[None, :])
            b_g = np.where(swap, mg[None, :], bg[None, :])
            b_p = np.where(swap, mp[None, :], bp[None, :])
            # _vec_macro_f1 expects an index matrix; here our (b,N) arrays already
            # hold per-resample labels, so compute confusion counts inline.
            f1_m = _macro_f1_from_arrays(m_g, m_p, L)
            f1_b = _macro_f1_from_arrays(b_g, b_p, L)
            delta = f1_m - f1_b
        extreme += int(np.sum(np.abs(delta) >= abs(observed)))
        done += b
    p_value = (extreme + 1) / (n_iters + 1)
    return float(observed), float(p_value)


def _macro_f1_from_arrays(g: np.ndarray, p: np.ndarray, num_labels: int) -> np.ndarray:
    """Macro-F1 per row of (B, N) gold / pred arrays."""
    B = g.shape[0]
    L = num_labels
    f1s = np.zeros((B, L), dtype=np.float64)
    for c in range(L):
        tp = ((g == c) & (p == c)).sum(axis=1)
        fp = ((g != c) & (p == c)).sum(axis=1)
        fn = ((g == c) & (p != c)).sum(axis=1)
        denom = (2 * tp + fp + fn)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s[:, c] = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return f1s.mean(axis=1)


def fisher_combined_p(p_values: Sequence[float]) -> Optional[float]:
    """Fisher's method for combining independent p-values. Falls back to None if empty."""
    pv = [max(min(p, 1.0 - 1e-12), 1e-12) for p in p_values if p is not None]
    if not pv:
        return None
    chi2 = -2.0 * sum(math.log(p) for p in pv)
    df = 2 * len(pv)
    # survival function of chi-squared with df degrees of freedom
    try:
        from scipy.stats import chi2 as _chi2_dist  # type: ignore
        return float(_chi2_dist.sf(chi2, df))
    except ImportError:
        # Numerical approximation via regularized upper incomplete gamma.
        return float(_chi2_sf(chi2, df))


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared without scipy (sufficient for df even)."""
    if x <= 0.0:
        return 1.0
    # For even df, the chi-squared sf has a closed form:
    # P(X > x) = exp(-x/2) * sum_{k=0}^{df/2-1} (x/2)^k / k!
    if df % 2 == 0:
        half = x / 2.0
        term = 1.0
        total = 1.0
        for k in range(1, df // 2):
            term *= half / k
            total += term
        return math.exp(-half) * total
    # Generic fallback via Wilson–Hilferty normal approximation.
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ----- Top-level summary -----------------------------------------------------


def summarize(
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    output_dir: Path,
    bootstrap_iters: int,
    randomization_iters: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "config": {
            "datasets": list(datasets),
            "seeds": list(seeds),
            "methods": list(methods),
            "bootstrap_iters": bootstrap_iters,
            "randomization_iters": randomization_iters,
        },
        "per_run": defaultdict(dict),
        "aggregated": defaultdict(dict),
        "pvalues": defaultdict(dict),
        "per_class": defaultdict(dict),
        "collapse": defaultdict(dict),
    }

    headline_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_class_seeds: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for dataset in datasets:
        headline = DATASET_HEADLINE[dataset]
        split = headline["split"]
        metric = headline["metric"]

        for method in methods:
            for seed in seeds:
                records = load_predictions_for(dataset, method, seed, split)
                if records is None:
                    continue
                value = _metric_from_records(records, metric)
                summary["per_run"][f"{dataset}|{method}"][f"seed{seed}"] = value
                headline_values[(dataset, method)].append(value)
                recalls = per_class_recall(records)
                summary["per_class"][f"{dataset}|{method}"][f"seed{seed}"] = recalls
                for cls, val in recalls.items():
                    per_class_seeds[(dataset, method)][cls].append(val)
                col = collapse_indicator(records)
                if col is not None:
                    summary["collapse"][f"{dataset}|{method}"][f"seed{seed}"] = col

    for (dataset, method), values in headline_values.items():
        arr = np.asarray(values, dtype=float)
        summary["aggregated"][f"{dataset}|{method}"] = {
            "n_seeds": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
            "values": arr.tolist(),
        }

    # paired tests vs. claim_evidence
    for dataset in datasets:
        headline = DATASET_HEADLINE[dataset]
        split = headline["split"]
        metric = headline["metric"]

        for method in methods:
            if method == "claim_evidence":
                continue
            per_seed_boot: List[float] = []
            per_seed_rand: List[float] = []
            per_seed_delta: List[float] = []
            seed_records: List[Dict[str, Any]] = []
            for seed in seeds:
                method_records = load_predictions_for(dataset, method, seed, split)
                baseline_records = load_predictions_for(dataset, "claim_evidence", seed, split)
                if method_records is None or baseline_records is None:
                    continue
                delta_b, p_b = paired_bootstrap_p(
                    method_records, baseline_records, metric,
                    n_resamples=bootstrap_iters, seed=seed,
                )
                delta_r, p_r = approximate_randomization_p(
                    method_records, baseline_records, metric,
                    n_iters=randomization_iters, seed=seed,
                )
                per_seed_boot.append(p_b)
                per_seed_rand.append(p_r)
                per_seed_delta.append(delta_b)
                seed_records.append({
                    "seed": seed,
                    "delta": delta_b,
                    "bootstrap_p": p_b,
                    "randomization_p": p_r,
                })

            summary["pvalues"][f"{dataset}|{method}"] = {
                "mean_delta": float(np.mean(per_seed_delta)) if per_seed_delta else 0.0,
                "bootstrap_combined_p": fisher_combined_p(per_seed_boot),
                "randomization_combined_p": fisher_combined_p(per_seed_rand),
                "per_seed": seed_records,
            }

    # ----- write outputs -----
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per-class recall: mean ± std per (dataset, method, class).
    per_class_agg: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (dataset, method), cls_map in per_class_seeds.items():
        for cls, vals in cls_map.items():
            arr = np.asarray(vals, dtype=float)
            per_class_agg[f"{dataset}|{method}"][cls] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
                "n_seeds": int(arr.size),
            }

    # JSON dump
    summary_json = {
        "config": summary["config"],
        "per_run": dict(summary["per_run"]),
        "aggregated": dict(summary["aggregated"]),
        "pvalues": dict(summary["pvalues"]),
        "per_class": dict(summary["per_class"]),
        "per_class_aggregated": dict(per_class_agg),
        "collapse": dict(summary["collapse"]),
    }
    (output_dir / "significance_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8",
    )

    # CSVs
    with (output_dir / "significance_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "metric", "n_seeds", "mean", "std", *[f"seed{s}" for s in seeds]])
        for dataset in datasets:
            metric = DATASET_HEADLINE[dataset]["metric"]
            for method in methods:
                agg = summary["aggregated"].get(f"{dataset}|{method}")
                per_run = summary["per_run"].get(f"{dataset}|{method}", {})
                if not agg:
                    writer.writerow([dataset, method, metric, 0, "", "", *["" for _ in seeds]])
                    continue
                row = [
                    dataset, method, metric, agg["n_seeds"],
                    f"{agg['mean'] * 100:.2f}", f"{agg['std'] * 100:.2f}",
                ]
                for seed in seeds:
                    val = per_run.get(f"seed{seed}")
                    row.append(f"{val * 100:.2f}" if val is not None else "")
                writer.writerow(row)

    with (output_dir / "significance_pvalues.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "method", "vs", "mean_delta_pp",
            "bootstrap_combined_p", "randomization_combined_p", "n_seeds",
        ])
        for dataset in datasets:
            for method in methods:
                if method == "claim_evidence":
                    continue
                info = summary["pvalues"].get(f"{dataset}|{method}")
                if info is None:
                    continue
                writer.writerow([
                    dataset, method, "claim_evidence",
                    f"{info['mean_delta'] * 100:.2f}",
                    "" if info["bootstrap_combined_p"] is None else f"{info['bootstrap_combined_p']:.4f}",
                    "" if info["randomization_combined_p"] is None else f"{info['randomization_combined_p']:.4f}",
                    len(info["per_seed"]),
                ])

    # Markdown
    md_lines: List[str] = []
    md_lines.append("# Multi-seed Significance Report\n")
    md_lines.append(
        f"Seeds: {', '.join(map(str, seeds))}. "
        f"Bootstrap resamples: {bootstrap_iters}. "
        f"Randomization iters: {randomization_iters}. "
        "p-values are Fisher-combined across seeds.\n"
    )
    md_lines.append("## Mean ± Std (percentage points)\n")
    header = ["Method"] + [DATASET_HEADLINE[d]["label"] for d in datasets]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "---|" * len(header))
    for method in methods:
        cells = [METHOD_LABELS.get(method, method)]
        for dataset in datasets:
            agg = summary["aggregated"].get(f"{dataset}|{method}")
            if not agg:
                cells.append("—")
            elif agg["n_seeds"] < 2:
                cells.append(f"{agg['mean'] * 100:.2f} (n=1)")
            else:
                cells.append(f"{agg['mean'] * 100:.2f} ± {agg['std'] * 100:.2f}")
        md_lines.append("| " + " | ".join(cells) + " |")

    md_lines.append("\n## Paired significance vs. claim-evidence (Fisher-combined p)\n")
    md_lines.append("| Method | Dataset | Δ (pp) | bootstrap p | randomization p |")
    md_lines.append("|---|---|---:|---:|---:|")
    for dataset in datasets:
        for method in methods:
            if method == "claim_evidence":
                continue
            info = summary["pvalues"].get(f"{dataset}|{method}")
            label = METHOD_LABELS.get(method, method)
            dataset_label = DATASET_HEADLINE[dataset]["label"]
            if info is None:
                md_lines.append(f"| {label} | {dataset_label} | — | — | — |")
                continue
            boot_p = info["bootstrap_combined_p"]
            rand_p = info["randomization_combined_p"]
            boot_cell = "—" if boot_p is None else f"{boot_p:.4f}"
            rand_cell = "—" if rand_p is None else f"{rand_p:.4f}"
            md_lines.append(
                f"| {label} | {dataset_label} | "
                f"{info['mean_delta'] * 100:+.2f} | "
                f"{boot_cell} | {rand_cell} |"
            )

    # Per-class recall: surfaces minority-class collapse on imbalanced stress splits.
    if per_class_agg:
        md_lines.append("\n## Per-class recall (mean ± std across seeds)\n")
        md_lines.append(
            "Surfaces minority-class collapse: when a baseline predicts only one "
            "class on a balanced stress split, accuracy ≈ 50 % and the missing-class "
            "recall is 0.\n"
        )
        # Collect class universe per dataset (sorted).
        for dataset in datasets:
            classes_seen: List[str] = []
            for method in methods:
                cls_info = per_class_agg.get(f"{dataset}|{method}", {})
                for cls in cls_info.keys():
                    if cls not in classes_seen:
                        classes_seen.append(cls)
            if not classes_seen:
                continue
            classes_seen.sort()
            md_lines.append(f"### {DATASET_HEADLINE[dataset]['label']}\n")
            header = ["Method"] + [f"recall({c})" for c in classes_seen] + ["collapsed seeds"]
            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("|" + "---|" * len(header))
            for method in methods:
                cls_info = per_class_agg.get(f"{dataset}|{method}", {})
                if not cls_info:
                    continue
                cells = [METHOD_LABELS.get(method, method)]
                for cls in classes_seen:
                    info = cls_info.get(cls)
                    if info is None:
                        cells.append("—")
                    elif info["n_seeds"] < 2:
                        cells.append(f"{info['mean'] * 100:.1f} (n=1)")
                    else:
                        cells.append(f"{info['mean'] * 100:.1f} ± {info['std'] * 100:.1f}")
                col_info = summary["collapse"].get(f"{dataset}|{method}", {})
                if col_info:
                    cells.append(
                        ", ".join(f"{k}→{v}" for k, v in sorted(col_info.items()))
                    )
                else:
                    cells.append("—")
                md_lines.append("| " + " | ".join(cells) + " |")
            md_lines.append("")

    md_lines.append("\n## Per-seed raw values\n")
    md_lines.append("| Dataset | Method | " + " | ".join(f"seed{s}" for s in seeds) + " |")
    md_lines.append("|---|---|" + "---:|" * len(seeds))
    for dataset in datasets:
        metric_label = DATASET_HEADLINE[dataset]["label"]
        for method in methods:
            per_run = summary["per_run"].get(f"{dataset}|{method}", {})
            row = [metric_label, METHOD_LABELS.get(method, method)]
            for seed in seeds:
                val = per_run.get(f"seed{seed}")
                row.append("—" if val is None else f"{val * 100:.2f}")
            md_lines.append("| " + " | ".join(row) + " |")

    (output_dir / "significance_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed significance experiment.")
    parser.add_argument("--datasets", nargs="+", default=["fever", "politihop", "hover"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2024])
    parser.add_argument(
        "--methods", nargs="+",
        default=["claim_evidence", "two_branch", "full_wo_remix", "full_egr_fv"],
    )
    parser.add_argument("--output_dir", default="outputs/significance")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--randomization_iters", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(
        datasets=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        output_dir=Path(args.output_dir),
        bootstrap_iters=args.bootstrap_iters,
        randomization_iters=args.randomization_iters,
    )
    print(f"[done] wrote {Path(args.output_dir) / 'significance_report.md'}")


if __name__ == "__main__":
    main()
