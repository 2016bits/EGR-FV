#!/usr/bin/env python3
"""Evaluate a trained checkpoint under four evidence conditions.

Conditions:
- clean:           original gold evidence.
- shuffled:        evidence randomly re-assigned across samples (label-agnostic).
- null:            evidence replaced with the data evidence_placeholder token.
- wrong_evidence:  evidence swapped with evidence drawn from a sample whose
                   gold label differs from the current sample's label (a
                   deterministic, label-aware perturbation that is strictly
                   "wrong" rather than possibly-accidentally-correct).

The script loads the same model wiring as `src.main` so it works for any
existing config + checkpoint pair (claim-evidence warm-up, the EGR ablations,
or the full pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data.dataset import FactVerificationDataset
from src.main import (
    build_collators,
    build_dataset,
    build_grounded_model,
    build_label_maps,
    build_shortcut_model,
    build_tokenizers,
)
from src.trainers.evaluator import Evaluator
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_model_state, write_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-perturbation evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to the trained checkpoint to load.")
    parser.add_argument(
        "--ckpt-kind",
        choices=["remix", "grounded"],
        default="remix",
        help="remix: full EGR / ablation checkpoint (loads grounded + maybe shortcut). "
        "grounded: claim-evidence warm-up checkpoint (loads grounded only).",
    )
    parser.add_argument("--dataset-name", required=True, help="Label used in the report (HOVER/FEVER/PolitiHop).")
    parser.add_argument("--method-name", required=True, help="Label used in the report (e.g. EGR-FV, w/o remix).")
    parser.add_argument("--output", required=True, help="Path to write the JSON report.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _make_shuffled_records(records: Sequence[Mapping[str, Any]], seed: int) -> List[Dict[str, Any]]:
    evidences = [str(record.get("evidence_text", "")) for record in records]
    if not evidences:
        return [deepcopy(record) for record in records]
    rng = np.random.default_rng(seed)
    permuted = list(rng.permutation(evidences))
    out: List[Dict[str, Any]] = []
    for record, evidence_text in zip(records, permuted):
        updated = deepcopy(record)
        updated["evidence_list"] = [str(evidence_text)] if evidence_text else []
        updated["evidence_text"] = str(evidence_text)
        out.append(updated)
    return out


def _make_null_records(
    records: Sequence[Mapping[str, Any]],
    placeholder: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        updated = deepcopy(record)
        updated["evidence_list"] = [placeholder]
        updated["evidence_text"] = placeholder
        out.append(updated)
    return out


def _make_wrong_evidence_records(
    records: Sequence[Mapping[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    """For each sample, swap evidence with one drawn from a sample whose gold
    label differs. Falls back to any other sample if no different-label
    candidate exists (degenerate; unlikely for the datasets we use here)."""

    by_label: Dict[int, List[int]] = {}
    for index, record in enumerate(records):
        label_id = int(record.get("label", -1))
        by_label.setdefault(label_id, []).append(index)

    all_label_ids = [lid for lid in by_label.keys() if lid >= 0]
    rng = random.Random(seed)

    out: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        current_label = int(record.get("label", -1))
        candidate_labels = [lid for lid in all_label_ids if lid != current_label]
        donor_index: int | None = None
        if candidate_labels:
            chosen_label = rng.choice(candidate_labels)
            pool = by_label[chosen_label]
            if pool:
                donor_index = rng.choice(pool)
        if donor_index is None:
            # Fallback: pick any other sample.
            for _ in range(8):
                cand = rng.randrange(len(records))
                if cand != index:
                    donor_index = cand
                    break
            if donor_index is None:
                donor_index = (index + 1) % len(records)

        donor = records[donor_index]
        donor_text = str(donor.get("evidence_text", ""))
        donor_list = list(donor.get("evidence_list", [])) or ([donor_text] if donor_text else [])
        updated = deepcopy(record)
        updated["evidence_list"] = [str(item) for item in donor_list]
        updated["evidence_text"] = donor_text
        out.append(updated)
    return out


def _condition_dataset(
    base: FactVerificationDataset,
    condition: str,
    placeholder: str,
    seed: int,
) -> FactVerificationDataset:
    if condition == "clean":
        return base
    if condition == "shuffled":
        records = _make_shuffled_records(base.records, seed=seed)
    elif condition == "null":
        records = _make_null_records(base.records, placeholder=placeholder)
    elif condition == "wrong_evidence":
        records = _make_wrong_evidence_records(base.records, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return base.clone_with_records(records)


def _load_remix_checkpoint(
    ckpt_path: str,
    grounded_model: torch.nn.Module,
    shortcut_model: torch.nn.Module,
    device: torch.device,
) -> bool:
    """Load a remix-style checkpoint that may contain grounded and shortcut state.

    Returns whether the shortcut branch was loaded.
    """
    checkpoint = load_model_state(
        grounded_model,
        ckpt_path,
        ["grounded_model_state", "model_state"],
        map_location=device,
    )
    if "shortcut_model_state" in checkpoint:
        shortcut_model.load_state_dict(checkpoint["shortcut_model_state"])
        return True
    return False


def _load_grounded_checkpoint(
    ckpt_path: str,
    grounded_model: torch.nn.Module,
    device: torch.device,
) -> None:
    load_model_state(
        grounded_model,
        ckpt_path,
        ["model_state", "grounded_model_state"],
        map_location=device,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", args.seed)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_id, id_to_label = build_label_maps(config)
    shortcut_tokenizer, grounded_tokenizer = build_tokenizers(config)
    shortcut_collator, grounded_collator, joint_collator = build_collators(
        config, shortcut_tokenizer, grounded_tokenizer
    )

    test_dataset = build_dataset(config, "test", label_to_id, id_to_label)
    placeholder = str(config["data"].get("evidence_placeholder", "[NO_EVIDENCE]"))

    grounded_model = build_grounded_model(config)
    shortcut_model = build_shortcut_model(config)

    shortcut_loaded = False
    if args.ckpt_kind == "remix":
        shortcut_loaded = _load_remix_checkpoint(args.ckpt, grounded_model, shortcut_model, device)
        if not shortcut_loaded:
            # If the remix checkpoint did not include the shortcut, fall back to
            # the warmup shortcut from config (matches main.py behaviour).
            shortcut_ckpt = config.get("checkpoints", {}).get("shortcut")
            if shortcut_ckpt and Path(shortcut_ckpt).exists():
                load_model_state(
                    shortcut_model,
                    shortcut_ckpt,
                    ["model_state", "shortcut_model_state"],
                    map_location=device,
                )
                shortcut_loaded = True
    else:
        _load_grounded_checkpoint(args.ckpt, grounded_model, device)

    grounded_model.to(device)
    if shortcut_loaded:
        shortcut_model.to(device)
    else:
        shortcut_model = None

    evaluator = Evaluator(
        shortcut_model=shortcut_model,
        grounded_model=grounded_model,
        fusion_head=None,
        joint_collator=joint_collator,
        shortcut_collator=shortcut_collator,
        grounded_collator=grounded_collator,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        device=device,
        id_to_label=id_to_label,
        evidence_placeholder=placeholder,
        fusion_alpha=float(config.get("inference", {}).get("fusion_alpha", 0.5)),
        routing_config=config.get("routing", {}),
        # Use a fixed grouping that is stable across conditions so reports are comparable.
        eval_grouping="record",
    )

    conditions = ("clean", "shuffled", "null", "wrong_evidence")
    results: Dict[str, Any] = {
        "dataset": args.dataset_name,
        "method": args.method_name,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "ckpt_kind": args.ckpt_kind,
        "num_samples": len(test_dataset),
        "shortcut_branch_loaded": shortcut_loaded,
        "conditions": {},
    }

    for condition in conditions:
        seed_for_condition = int(config.get("seed", args.seed)) + {
            "clean": 0,
            "shuffled": 1,
            "null": 2,
            "wrong_evidence": 3,
        }[condition]
        conditioned = _condition_dataset(
            test_dataset,
            condition=condition,
            placeholder=placeholder,
            seed=seed_for_condition,
        )
        report = evaluator.evaluate_dataset(conditioned, mode="grounded")
        metrics = report["metrics"]
        results["conditions"][condition] = {
            "num_samples": int(report["num_samples"]),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "macro_precision": float(metrics.get("macro_precision", 0.0)),
            "macro_recall": float(metrics.get("macro_recall", 0.0)),
            "supports_f1": float(metrics.get("supports_f1", 0.0)) if "supports_f1" in metrics else None,
            "refutes_f1": float(metrics.get("refutes_f1", 0.0)) if "refutes_f1" in metrics else None,
            "hop_macro_f1": {
                hop: float(values.get("macro_f1", 0.0))
                for hop, values in report.get("hop_metrics", {}).items()
            },
        }
        clean_f1 = results["conditions"].get("clean", {}).get("macro_f1")
        if condition != "clean" and clean_f1 is not None:
            results["conditions"][condition]["delta_vs_clean"] = (
                results["conditions"][condition]["macro_f1"] - clean_f1
            )

    ensure_dir(str(Path(args.output).parent))
    write_json(args.output, results)
    print(json.dumps({k: v for k, v in results.items() if k != "conditions"}, indent=2))
    for condition, payload in results["conditions"].items():
        macro = payload["macro_f1"] * 100
        delta = payload.get("delta_vs_clean")
        delta_str = "" if delta is None else f"   (Δ={delta * 100:+.2f}pp)"
        print(f"  {condition:>14s}: macro-F1 = {macro:6.2f}{delta_str}")


if __name__ == "__main__":
    main()
