#!/usr/bin/env python3
"""Llama-3-8B-Instruct fact-verification baseline.

For each test claim+evidence pair, prompt the model in a few-shot setting to
emit SUPPORTS / REFUTES, then compute accuracy / macro-F1. Writes outputs that
mirror EGR-FV's eval_predictions.jsonl + eval_report.json so the existing
significance / summarization scripts can ingest them.

Datasets covered (each with claim+evidence text already in the JSON):
  FEVER orig (data/FEVER/converted_data/test.json)
  FEVER sym  (data/FEVER/converted_data/symmetric.json)
  PolitiHop  (data/PolitiHop/converted_data/test.json)
  PolitiHop sym (data/PolitiHop/converted_data/symmetric.json)
  HOVER      (data/HOVER/converted_data/test.json)

Usage:
  python scripts/run_llm_baseline.py --dataset fever
  python scripts/run_llm_baseline.py --dataset all
  python scripts/run_llm_baseline.py --dataset fever_sym --num_shots 4 --max_new_tokens 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


DATASETS: Dict[str, Dict[str, Any]] = {
    "fever": {
        "input": "data/FEVER/converted_data/test.json",
        "output_dir": "outputs/llm_baseline/FEVER/test",
        "labels": ("supports", "refutes"),
    },
    "fever_sym": {
        "input": "data/FEVER/converted_data/symmetric.json",
        "output_dir": "outputs/llm_baseline/FEVER/symmetric",
        "labels": ("supports", "refutes"),
    },
    "politihop": {
        "input": "data/PolitiHop/converted_data/test.json",
        "output_dir": "outputs/llm_baseline/PolitiHop/test",
        "labels": ("supports", "refutes"),
    },
    "politihop_sym": {
        "input": "data/PolitiHop/converted_data/symmetric.json",
        "output_dir": "outputs/llm_baseline/PolitiHop/symmetric",
        "labels": ("supports", "refutes"),
    },
    "hover": {
        "input": "data/HOVER/converted_data/test.json",
        "output_dir": "outputs/llm_baseline/HOVER/test",
        "labels": ("supports", "refutes"),
    },
}


FEW_SHOT_EXAMPLES = [
    {
        "claim": "The Eiffel Tower is located in Paris.",
        "evidence": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "label": "supports",
    },
    {
        "claim": "Mount Everest is the lowest mountain on Earth.",
        "evidence": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
        "label": "refutes",
    },
    {
        "claim": "Albert Einstein won the Nobel Prize in Physics in 1921.",
        "evidence": "He received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
        "label": "supports",
    },
    {
        "claim": "Shakespeare wrote the play 1984.",
        "evidence": "Nineteen Eighty-Four (also published as 1984) is a dystopian social science fiction novel by the English novelist George Orwell.",
        "label": "refutes",
    },
]


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


SYSTEM_PROMPT = (
    "You are a careful fact verifier. Given a claim and a short Wikipedia "
    "evidence passage, decide whether the evidence SUPPORTS or REFUTES the "
    "claim. Reply with exactly one word: SUPPORTS or REFUTES."
)


def build_messages(claim: str, evidence: str, num_shots: int) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES[:num_shots]:
        messages.append({
            "role": "user",
            "content": f"Claim: {ex['claim']}\nEvidence: {ex['evidence']}\nAnswer with SUPPORTS or REFUTES.",
        })
        messages.append({"role": "assistant", "content": ex["label"].upper()})
    messages.append({
        "role": "user",
        "content": f"Claim: {claim}\nEvidence: {evidence}\nAnswer with SUPPORTS or REFUTES.",
    })
    return messages


# Llama-3 chat-template assembly without relying on tokenizer.apply_chat_template
# (older transformers builds don't have it). Special-token IDs are looked up by
# name at runtime so the implementation remains portable.
def render_llama3_prompt(messages: Sequence[Dict[str, str]]) -> str:
    pieces: List[str] = ["<|begin_of_text|>"]
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        pieces.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    pieces.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(pieces)


LABEL_RE = re.compile(r"\b(SUPPORTS?|REFUTES?|TRUE|FALSE|YES|NO)\b", re.IGNORECASE)


def parse_label(text: str) -> str:
    """Extract first SUPPORTS/REFUTES (or natural synonym) from generated text."""
    m = LABEL_RE.search(text or "")
    if not m:
        return "refutes"  # neutral fallback (matches the dominant FEVER/PolitiHop class)
    word = m.group(1).lower()
    if word in {"support", "supports", "true", "yes"}:
        return "supports"
    return "refutes"


@torch.no_grad()
def run_one_dataset(
    name: str,
    info: Dict[str, Any],
    tokenizer,
    model,
    num_shots: int,
    max_new_tokens: int,
    limit: Optional[int],
    log_every: int,
    max_evidence_chars: int = 600,
    flush_every: int = 50,
) -> None:
    in_path = REPO_ROOT / info["input"]
    out_dir = REPO_ROOT / info["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(in_path)
    if limit:
        records = records[:limit]
    print(f"[llm] {name}: {len(records)} records → {out_dir}", flush=True)

    pred_path = out_dir / "eval_predictions.jsonl"
    # Resume support: skip any IDs already present in the existing predictions file.
    seen_ids: set = set()
    if pred_path.exists():
        try:
            with pred_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seen_ids.add(str(json.loads(line).get("id")))
            print(f"[llm:{name}] resuming, skipping {len(seen_ids)} already-done ids",
                  flush=True)
        except Exception as e:
            print(f"[llm:{name}] failed to read existing predictions: {e}", flush=True)
            seen_ids = set()

    device = next(model.parameters()).device
    predictions: List[Dict[str, Any]] = []
    n_correct = 0
    n_seen = 0
    t0 = time.time()
    pred_handle = pred_path.open("a", encoding="utf-8")
    for rec in records:
        sid = str(rec.get("id"))
        if sid in seen_ids:
            continue
        # Truncate evidence to avoid OOM on long passages.
        rec = dict(rec)
        ev = (rec.get("evidence") or "").strip()
        if max_evidence_chars and len(ev) > max_evidence_chars:
            ev = ev[:max_evidence_chars].rstrip() + "…"
        rec["evidence"] = ev
        claim = rec["claim"]
        evidence = (rec.get("evidence") or "").strip()
        gold = str(rec.get("label", "")).lower()
        msgs = build_messages(claim, evidence, num_shots)
        prompt_text = render_llama3_prompt(msgs)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, eot_id] if eot_id != tokenizer.unk_token_id else tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_label(gen)
        n_seen += 1
        if pred == gold:
            n_correct += 1
        new_pred = {
            "id": rec.get("id"),
            "group": "hard",
            "record_group": "hard",
            "num_hops": rec.get("num_hops", -1),
            "label": gold.upper(),
            "pred_label": pred.upper(),
            "confidence": 1.0,
            "llm_text": gen.strip(),
        }
        predictions.append(new_pred)
        pred_handle.write(json.dumps(new_pred, ensure_ascii=False) + "\n")
        if n_seen % flush_every == 0:
            pred_handle.flush()
        if n_seen % log_every == 0:
            rate = n_seen / max(time.time() - t0, 1e-6)
            print(f"  [llm:{name}] {n_seen}/{len(records)}  acc={n_correct / n_seen:.4f}  "
                  f"({rate:.2f}/s)", flush=True)

    pred_handle.flush()
    pred_handle.close()

    # Re-load the full prediction file so resumed runs include earlier rows
    # when computing the final report.
    with pred_path.open("r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    golds = [p["label"].lower() for p in predictions]
    preds = [p["pred_label"].lower() for p in predictions]
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / max(len(golds), 1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        golds, preds, average="macro", zero_division=0, labels=list(info["labels"]),
    )
    per_cls = {}
    for cls in info["labels"]:
        p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
            golds, preds, average="binary", pos_label=cls, zero_division=0,
        )
        per_cls[cls] = {"precision": p_cls, "recall": r_cls, "f1": f_cls,
                        "support": sum(1 for g in golds if g == cls)}

    report = {
        "base": {
            "mode": "llm",
            "num_samples": len(predictions),
            "metrics": {
                "accuracy": acc,
                "macro_precision": float(p_macro),
                "macro_recall": float(r_macro),
                "macro_f1": float(f_macro),
                **{
                    f"{cls}_precision": per_cls[cls]["precision"]
                    for cls in info["labels"]
                },
                **{
                    f"{cls}_recall": per_cls[cls]["recall"] for cls in info["labels"]
                },
                **{f"{cls}_f1": per_cls[cls]["f1"] for cls in info["labels"]},
                **{
                    f"{cls}_support": per_cls[cls]["support"]
                    for cls in info["labels"]
                },
            },
        },
        "llm_config": {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "num_shots": num_shots,
            "max_new_tokens": max_new_tokens,
            "decoding": "greedy",
        },
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")
    print(f"[llm:{name}] done: acc={acc * 100:.2f}, macro_f1={f_macro * 100:.2f}, "
          f"wall={time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fever_sym",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_evidence_chars", type=int, default=600,
                        help="Truncate evidence longer than this to keep prompts bounded.")
    parser.add_argument("--flush_every", type=int, default=50)
    args = parser.parse_args()

    print(f"[llm] loading model={args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        run_one_dataset(
            name=name,
            info=DATASETS[name],
            tokenizer=tokenizer,
            model=model,
            num_shots=args.num_shots,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            log_every=args.log_every,
            max_evidence_chars=args.max_evidence_chars,
            flush_every=args.flush_every,
        )


if __name__ == "__main__":
    main()
