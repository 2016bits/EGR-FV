#!/usr/bin/env python3
"""Filter retrieved-evidence FEVER train.jsonl: drop rows with empty / near-empty evidence.

Rows with no retrieved evidence (~7% of FEVER train where FEVER_baseline doc
retrieval came back empty) would otherwise become claim-only training signal,
which confounds the apples-to-apples comparison.

Usage:
  python scripts/filter_retrieved_train.py
  python scripts/filter_retrieved_train.py --input data/FEVER/retrieved/train.jsonl \
      --output data/FEVER/retrieved/train_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/FEVER/retrieved/train.jsonl")
    parser.add_argument("--output", default="data/FEVER/retrieved/train.jsonl")
    parser.add_argument("--min_chars", type=int, default=20,
                        help="Drop rows whose evidence has fewer characters than this.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows: list = []
    n_in = 0
    n_dropped = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            ev = (rec.get("evidence") or "").strip()
            if len(ev) < args.min_chars:
                n_dropped += 1
                continue
            rows.append(rec)

    if out_path == in_path:
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[filter] kept {len(rows)}/{n_in} rows (dropped {n_dropped}; min_chars={args.min_chars})")


if __name__ == "__main__":
    main()
