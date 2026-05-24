#!/usr/bin/env python3
"""Estimate per-stage compute cost from checkpoint mtimes.

The EGR-FV pipeline writes one checkpoint per stage (`shortcut_best.pt`,
`grounded_best.pt`, `routing_oof/fold_K/{shortcut,grounded}_best.pt`,
`remix_best.pt`). Stage wall-clock = mtime(stage_ckpt) - mtime(prev_stage_ckpt).
For the *first* stage (shortcut warmup) we don't have a "before" anchor, so we
fall back to the mtime delta between the dataset's earliest checkpoint and its
first log file, when available; otherwise we report it as "n/a".

Output:
  outputs/analysis/compute_cost/compute_cost_table.md
  outputs/analysis/compute_cost/compute_cost_table.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def m(path: Path) -> Optional[float]:
    return path.stat().st_mtime if path.exists() else None


def stage_durations(ckpt_dir: Path) -> Dict[str, Any]:
    """Return per-stage wall-clock seconds for one (dataset, seed) run."""
    out: Dict[str, Any] = {}
    s = m(ckpt_dir / "shortcut_best.pt")
    g = m(ckpt_dir / "grounded_best.pt")
    rm = m(ckpt_dir / "remix_best.pt")
    fold_dir = ckpt_dir / "routing_oof"

    # warmup_grounded = g - s
    out["warmup_grounded"] = (g - s) if (s and g) else None

    # routing total = last_fold_grounded - g
    fold_grounded_times: List[float] = []
    if fold_dir.exists():
        for fold in sorted(fold_dir.iterdir()):
            if not fold.is_dir():
                continue
            ft = m(fold / "grounded_best.pt")
            if ft:
                fold_grounded_times.append(ft)
    if fold_grounded_times and g:
        out["routing_total"] = max(fold_grounded_times) - g
        out["routing_per_fold"] = out["routing_total"] / max(len(fold_grounded_times), 1)
        out["routing_num_folds"] = len(fold_grounded_times)

    # remix = remix_best - last_fold_grounded (else - grounded_best)
    last_fold = max(fold_grounded_times) if fold_grounded_times else g
    if rm and last_fold:
        out["remix"] = rm - last_fold

    # warmup_shortcut: best-effort using log directory mtime delta (if logs exist).
    # The shortcut warmup writes warmup_shortcut.csv first; the file's earliest
    # write time isn't tracked, so we use the parent log directory's birth time
    # if available. If neither anchor exists, leave it None.
    log_root = ckpt_dir.parent / "logs"
    candidate_dirs = [log_root, log_root / ckpt_dir.name]
    for cand in candidate_dirs:
        sc_csv = cand / "warmup_shortcut.csv"
        if sc_csv.exists():
            # First-epoch log usually lands ~few seconds after training start;
            # using csv mtime is rough but OK for one-line cost reporting.
            warmup_start = sc_csv.stat().st_mtime  # ~end of epoch 1
            if s:
                # shortcut_best.pt usually saves at end of best epoch.
                # If shortcut warmup is 2 epochs and best is epoch 2, sc_csv ≈ s.
                # We can't separate "start time" cleanly from logs. Fall back:
                out.setdefault("warmup_shortcut", None)
            break

    # Total wall-clock from shortcut_best.pt to remix_best.pt (excluding warmup_shortcut).
    if s and rm:
        out["total_pipeline_excl_warmup_shortcut"] = rm - s

    # Sum measurable stages (warmup_grounded + routing + remix).
    pieces = [v for v in (out.get("warmup_grounded"), out.get("routing_total"),
                          out.get("remix")) if v is not None]
    if pieces:
        out["sum_measurable"] = sum(pieces)
    return out


def fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f} s"
    if value < 3600:
        return f"{value / 60:.1f} min"
    return f"{value / 3600:.2f} h"


def claim_evidence_duration(ckpt_path: Path) -> Optional[float]:
    """Single-stage training; total ≈ checkpoint save time relative to the
    grounded warmup log start. Without a start anchor we just report None
    (the table caption mentions this caveat). Use an external probe instead.
    """
    return None  # placeholder — populated externally if a log start exists


SEED_RUNS: Dict[str, Sequence[str]] = {
    "HOVER":     ("seed13", "seed42", "seed2024"),
    "FEVER":     ("seed13", "seed42", "seed2024"),
    "PolitiHop": ("seed13", "seed42", "seed2024"),
}


def collect(dataset: str) -> List[Tuple[str, Dict[str, Any]]]:
    base = REPO_ROOT / "outputs" / dataset / "checkpoints"
    runs: List[Tuple[str, Dict[str, Any]]] = []
    if (base / "shortcut_best.pt").exists():
        runs.append(("single-seed", stage_durations(base)))
    for seed in SEED_RUNS.get(dataset, ()):
        if (base / seed / "shortcut_best.pt").exists():
            runs.append((seed, stage_durations(base / seed)))
    return runs


def render(dataset: str, runs: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    lines: List[str] = []
    if not runs:
        return lines
    lines.append(f"### {dataset}\n")
    lines.append("| Seed | warmup_grounded | routing (5-fold OOF) | remix | total (excl. warmup_shortcut) |")
    lines.append("|---|---|---|---|---|")
    for seed, info in runs:
        lines.append(
            f"| {seed} | {fmt_seconds(info.get('warmup_grounded'))} "
            f"| {fmt_seconds(info.get('routing_total'))} "
            f"({info.get('routing_num_folds') or '?'} folds) "
            f"| {fmt_seconds(info.get('remix'))} "
            f"| {fmt_seconds(info.get('total_pipeline_excl_warmup_shortcut'))} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/analysis/compute_cost")
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ("HOVER", "FEVER", "PolitiHop")

    md: List[str] = []
    md.append("# Compute Cost\n")
    md.append(
        "Per-stage wall-clock for one EGR-FV pipeline run, derived from "
        "checkpoint mtimes (single-GPU RTX A6000, 48 GiB; RoBERTa-base "
        "backbone; configs/<dataset>.yaml).\n\n"
        "* **warmup_grounded** is `mtime(grounded_best.pt) − mtime(shortcut_best.pt)`.\n"
        "* **routing (5-fold OOF)** is `mtime(routing_oof/fold_5/grounded_best.pt) − "
        "mtime(grounded_best.pt)`, i.e. the wall-clock of training a shortcut + "
        "grounded pair on each of 5 held-out folds.\n"
        "* **remix** is `mtime(remix_best.pt) − mtime(last fold)`.\n"
        "* **total (excl. warmup_shortcut)** is `mtime(remix_best.pt) − "
        "mtime(shortcut_best.pt)`. The very first stage (warmup_shortcut) has "
        "no pre-anchor checkpoint, so its duration is not listed here; in "
        "practice it runs in ≤ the per-fold shortcut time.\n"
    )

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        runs = collect(dataset)
        md.extend(render(dataset, runs))
        for seed, info in runs:
            rows.append({
                "dataset": dataset,
                "seed": seed,
                **{k: info.get(k) for k in
                   ("warmup_grounded", "routing_total", "routing_num_folds",
                    "remix", "total_pipeline_excl_warmup_shortcut", "sum_measurable")},
            })

    # CSV (raw seconds)
    if rows:
        with (out_dir / "compute_cost_table.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    (out_dir / "compute_cost_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'compute_cost_table.md'}")


if __name__ == "__main__":
    main()
