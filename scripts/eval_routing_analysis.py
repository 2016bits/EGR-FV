#!/usr/bin/env python3
"""Per-sample routing analysis on a test split.

For each test sample this script collects everything we need to evaluate the
"routing-is-meaningful" story:

  - the warmup shortcut prediction p_s(y | c)            (claim-only)
  - the warmup grounded prediction p_g(y | c, e)         (claim-evidence)
  - the EGR-FV prediction p_egr(y | c, e)
  - the warmup grounded / EGR-FV predictions when evidence is replaced with
    the placeholder ("null") or randomly permuted across samples ("shuffled")
  - the necessity score r_i and the routing group (bias_easy/hard/grounded_needed)
    that the score implies under the same thresholds used during training

The script writes a JSONL of per-sample records that is then consumed by
`scripts/summarize_routing_analysis.py` to produce:

  (1) per-group accuracy table (claim-only / claim-evidence / EGR-FV / gain),
  (2) Pearson/Spearman correlation between r_i and evidence sensitivity Δ_i,
  (3) a bucketed routing-vs-correctness visualization.

The model wiring follows `src.main` so the same configs and checkpoints used
for training/eval are valid here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import FactVerificationDataset
from src.data.routing import (
    compute_kl_disagreement,
    compute_necessity_score,
    evidence_length_score,
)
from src.main import (
    build_collators,
    build_dataset,
    build_grounded_model,
    build_label_maps,
    build_shortcut_model,
    build_tokenizers,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_model_state, write_json, write_jsonl
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Routing-analysis evaluator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shortcut-ckpt", required=True, help="Path to shortcut_best.pt.")
    parser.add_argument("--grounded-ckpt", required=True, help="Path to grounded_best.pt (claim-evidence warmup).")
    parser.add_argument("--egrfv-ckpt", required=True, help="Path to the full EGR-FV remix_best.pt.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True, help="Path to write the per-sample JSONL records.")
    parser.add_argument("--summary-output", default=None, help="Optional path for a JSON summary.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ----------------------------- evidence conditions -----------------------------

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


def _make_null_records(records: Sequence[Mapping[str, Any]], placeholder: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        updated = deepcopy(record)
        updated["evidence_list"] = [placeholder]
        updated["evidence_text"] = placeholder
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
        return base.clone_with_records(_make_shuffled_records(base.records, seed=seed))
    if condition == "null":
        return base.clone_with_records(_make_null_records(base.records, placeholder=placeholder))
    raise ValueError(f"Unknown condition: {condition}")


# ----------------------------- model wrappers ---------------------------------

def _load_remix_grounded(
    ckpt_path: str,
    grounded_model: torch.nn.Module,
    device: torch.device,
) -> None:
    load_model_state(
        grounded_model,
        ckpt_path,
        ["grounded_model_state", "model_state"],
        map_location=device,
    )


def _load_shortcut(
    ckpt_path: str,
    shortcut_model: torch.nn.Module,
    device: torch.device,
) -> None:
    load_model_state(
        shortcut_model,
        ckpt_path,
        ["shortcut_model_state", "model_state"],
        map_location=device,
    )


def _load_grounded(
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


def _move(inputs: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


# ----------------------------- inference passes -------------------------------

def _shortcut_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[str], List[np.ndarray]]:
    ids: List[str] = []
    probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="shortcut", leave=False):
            outputs = model(**_move(batch["shortcut_inputs"], device))
            batch_probs = outputs["probs"].detach().cpu().numpy()
            for index, sample_id in enumerate(batch["ids"]):
                ids.append(str(sample_id))
                probs.append(batch_probs[index])
    return ids, probs


def _grounded_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[str], List[np.ndarray]]:
    ids: List[str] = []
    probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="grounded", leave=False):
            outputs = model(**_move(batch["grounded_inputs"], device))
            batch_probs = outputs["probs"].detach().cpu().numpy()
            for index, sample_id in enumerate(batch["ids"]):
                ids.append(str(sample_id))
                probs.append(batch_probs[index])
    return ids, probs


def _build_loader(
    dataset: FactVerificationDataset,
    collator,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


# ----------------------------- routing helpers --------------------------------

def _route_group(
    necessity_score: float,
    shortcut_conf: float,
    shortcut_correct: bool,
    disagreement: float,
    routing_cfg: Mapping[str, Any],
) -> str:
    tau_shortcut_high = float(routing_cfg.get("tau_shortcut_high", 0.8))
    tau_disagreement_low = float(routing_cfg.get("tau_disagreement_low", 0.1))
    grounded_threshold = float(routing_cfg.get("grounded_needed_threshold", 0.7))
    bias_threshold = float(routing_cfg.get("bias_easy_threshold", 0.3))

    if (
        shortcut_conf >= tau_shortcut_high
        and shortcut_correct
        and disagreement <= tau_disagreement_low
    ):
        return "bias_easy"
    if necessity_score >= grounded_threshold:
        return "grounded_needed"
    if necessity_score <= bias_threshold:
        return "bias_easy"
    return "hard"


# --------------------------------- main ---------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", args.seed)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_id, id_to_label = build_label_maps(config)
    shortcut_tokenizer, grounded_tokenizer = build_tokenizers(config)
    shortcut_collator, grounded_collator, _ = build_collators(config, shortcut_tokenizer, grounded_tokenizer)

    test_dataset = build_dataset(config, "test", label_to_id, id_to_label)
    placeholder = str(config["data"].get("evidence_placeholder", "[NO_EVIDENCE]"))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    routing_cfg = dict(config.get("routing", {}))
    necessity_weights = routing_cfg.get("necessity_weights", {})

    # ---------- claim ordering + ground truth ----------
    records = list(test_dataset.records)
    sample_ids = [str(record["id"]) for record in records]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    claim_texts = [str(record.get("claim_text", "")) for record in records]
    evidence_texts = [str(record.get("evidence_text", "")) for record in records]
    num_hops = [record.get("num_hops") for record in records]

    # ---------- shortcut (claim-only) ----------
    shortcut_model = build_shortcut_model(config)
    _load_shortcut(args.shortcut_ckpt, shortcut_model, device)
    shortcut_model.to(device)
    shortcut_loader = _build_loader(test_dataset, shortcut_collator, batch_size, num_workers)
    shortcut_ids, shortcut_probs = _shortcut_probs(shortcut_model, shortcut_loader, device)
    shortcut_model.cpu()
    del shortcut_model
    torch.cuda.empty_cache()

    # Reorder according to id_to_index to be safe (DataLoader is shuffle=False so this is identity, but be defensive).
    shortcut_probs_array = np.zeros((len(records), shortcut_probs[0].shape[0]), dtype=np.float32)
    for sample_id, probs in zip(shortcut_ids, shortcut_probs):
        shortcut_probs_array[id_to_index[sample_id]] = probs

    # ---------- grounded warmup (claim-evidence) under three conditions ----------
    grounded_model = build_grounded_model(config)
    _load_grounded(args.grounded_ckpt, grounded_model, device)
    grounded_model.to(device)

    base_seed = int(config.get("seed", args.seed))
    grounded_probs_by_condition: Dict[str, np.ndarray] = {}
    for condition, seed_offset in (("clean", 0), ("shuffled", 1), ("null", 2)):
        conditioned = _condition_dataset(
            test_dataset,
            condition=condition,
            placeholder=placeholder,
            seed=base_seed + seed_offset,
        )
        loader = _build_loader(conditioned, grounded_collator, batch_size, num_workers)
        gids, gprobs = _grounded_probs(grounded_model, loader, device)
        out = np.zeros((len(records), gprobs[0].shape[0]), dtype=np.float32)
        for sample_id, probs in zip(gids, gprobs):
            out[id_to_index[sample_id]] = probs
        grounded_probs_by_condition[condition] = out

    grounded_model.cpu()
    del grounded_model
    torch.cuda.empty_cache()

    # ---------- EGR-FV (full pipeline) under three conditions ----------
    egrfv_model = build_grounded_model(config)
    _load_remix_grounded(args.egrfv_ckpt, egrfv_model, device)
    egrfv_model.to(device)

    egrfv_probs_by_condition: Dict[str, np.ndarray] = {}
    for condition, seed_offset in (("clean", 0), ("shuffled", 1), ("null", 2)):
        conditioned = _condition_dataset(
            test_dataset,
            condition=condition,
            placeholder=placeholder,
            seed=base_seed + seed_offset,
        )
        loader = _build_loader(conditioned, grounded_collator, batch_size, num_workers)
        eids, eprobs = _grounded_probs(egrfv_model, loader, device)
        out = np.zeros((len(records), eprobs[0].shape[0]), dtype=np.float32)
        for sample_id, probs in zip(eids, eprobs):
            out[id_to_index[sample_id]] = probs
        egrfv_probs_by_condition[condition] = out

    egrfv_model.cpu()
    del egrfv_model
    torch.cuda.empty_cache()

    # ---------- per-sample record assembly ----------
    out_records: List[Dict[str, Any]] = []
    for index in range(len(records)):
        label_id = int(labels[index])
        s_probs = shortcut_probs_array[index]
        g_probs_clean = grounded_probs_by_condition["clean"][index]
        g_probs_null = grounded_probs_by_condition["null"][index]
        g_probs_shuffled = grounded_probs_by_condition["shuffled"][index]
        e_probs_clean = egrfv_probs_by_condition["clean"][index]
        e_probs_null = egrfv_probs_by_condition["null"][index]
        e_probs_shuffled = egrfv_probs_by_condition["shuffled"][index]

        shortcut_pred = int(np.argmax(s_probs))
        shortcut_conf = float(np.max(s_probs))
        shortcut_correct = bool(shortcut_pred == label_id)

        g_pred_clean = int(np.argmax(g_probs_clean))
        g_conf_clean = float(np.max(g_probs_clean))
        g_correct_clean = bool(g_pred_clean == label_id)
        g_correct_null = bool(int(np.argmax(g_probs_null)) == label_id)
        g_correct_shuffled = bool(int(np.argmax(g_probs_shuffled)) == label_id)

        e_pred_clean = int(np.argmax(e_probs_clean))
        e_correct_clean = bool(e_pred_clean == label_id)
        e_correct_null = bool(int(np.argmax(e_probs_null)) == label_id)
        e_correct_shuffled = bool(int(np.argmax(e_probs_shuffled)) == label_id)

        # KL(grounded || shortcut) using the warmup grounded probs (clean evidence).
        disagreement = float(
            torch.sum(
                torch.from_numpy(g_probs_clean).clamp_min(1e-8)
                * (
                    torch.from_numpy(g_probs_clean).clamp_min(1e-8).log()
                    - torch.from_numpy(s_probs).clamp_min(1e-8).log()
                )
            ).item()
        )
        length_score = evidence_length_score(
            evidence_text=evidence_texts[index],
            claim_text=claim_texts[index],
        )
        necessity_score = compute_necessity_score(
            shortcut_conf=shortcut_conf,
            shortcut_correct=shortcut_correct,
            grounded_conf=g_conf_clean,
            grounded_correct=g_correct_clean,
            disagreement=disagreement,
            evidence_length_score=length_score,
            weights=necessity_weights,
        )
        group = _route_group(
            necessity_score=necessity_score,
            shortcut_conf=shortcut_conf,
            shortcut_correct=shortcut_correct,
            disagreement=disagreement,
            routing_cfg=routing_cfg,
        )

        # NLL with a tiny clamp for numerical stability.
        def _nll(probs: np.ndarray) -> float:
            return float(-math.log(max(float(probs[label_id]), 1e-12)))

        out_records.append(
            {
                "id": sample_ids[index],
                "label": label_id,
                "num_hops": num_hops[index],
                "necessity_score": necessity_score,
                "group": group,
                "shortcut_conf": shortcut_conf,
                "shortcut_correct": shortcut_correct,
                "shortcut_pred": shortcut_pred,
                "grounded_conf_clean": g_conf_clean,
                "grounded_correct_clean": g_correct_clean,
                "grounded_correct_null": g_correct_null,
                "grounded_correct_shuffled": g_correct_shuffled,
                "grounded_nll_clean": _nll(g_probs_clean),
                "grounded_nll_null": _nll(g_probs_null),
                "grounded_nll_shuffled": _nll(g_probs_shuffled),
                "egrfv_correct_clean": e_correct_clean,
                "egrfv_correct_null": e_correct_null,
                "egrfv_correct_shuffled": e_correct_shuffled,
                "egrfv_nll_clean": _nll(e_probs_clean),
                "egrfv_nll_null": _nll(e_probs_null),
                "egrfv_nll_shuffled": _nll(e_probs_shuffled),
                "disagreement": disagreement,
                "evidence_length_score": length_score,
            }
        )

    ensure_dir(str(Path(args.output).parent))
    write_jsonl(args.output, out_records)

    summary = _summarize(out_records, dataset_name=args.dataset_name)
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary["overall"], indent=2))


def _summarize(records: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    if not records:
        return {"dataset": dataset_name, "num_samples": 0, "overall": {}, "groups": {}}

    def _acc(values: Sequence[bool]) -> float:
        return float(np.mean(values)) if values else 0.0

    overall = {
        "num_samples": len(records),
        "claim_only_acc": _acc([record["shortcut_correct"] for record in records]),
        "claim_evidence_acc": _acc([record["grounded_correct_clean"] for record in records]),
        "egrfv_acc": _acc([record["egrfv_correct_clean"] for record in records]),
    }

    groups: Dict[str, Dict[str, Any]] = {}
    for group_name in ("bias_easy", "hard", "grounded_needed"):
        bucket = [record for record in records if record["group"] == group_name]
        groups[group_name] = {
            "count": len(bucket),
            "share": len(bucket) / len(records),
            "claim_only_acc": _acc([record["shortcut_correct"] for record in bucket]),
            "claim_evidence_acc": _acc([record["grounded_correct_clean"] for record in bucket]),
            "egrfv_acc": _acc([record["egrfv_correct_clean"] for record in bucket]),
        }
        groups[group_name]["gain_egrfv_vs_claim_only"] = (
            groups[group_name]["egrfv_acc"] - groups[group_name]["claim_only_acc"]
        )

    return {"dataset": dataset_name, "num_samples": len(records), "overall": overall, "groups": groups}


if __name__ == "__main__":
    main()
