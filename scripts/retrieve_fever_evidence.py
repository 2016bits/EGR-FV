#!/usr/bin/env python3
"""Retrieve top-K wiki sentences as evidence for each FEVER claim.

Two modes, both write a single JSONL whose schema matches the gold FEVER files
(`data/FEVER/converted_data/<split>.json`) so the existing dataset loaders work
without modification.

Mode A — `paper`:
  For splits `train` / `dev` / `test`, reuse the FEVER_baseline pre-computed
  candidate document IDs (`processed/<split>/expanded_doc_ids.pkl`). For each
  claim, pull every sentence from the top-`candidate_docs_top` candidate docs
  out of `processed/data.db`, rank them with BM25 against the claim, take the
  top-`top_k`. Joined back to EGR-FV's converted-data records by claim id.

Mode B — `symmetric`:
  Symmetric-FEVER claims aren't in FEVER_baseline. We build a TF-IDF index over
  Wikipedia page titles on first run (cached to disk), pick the top-N candidate
  pages per claim, pull their sentences and BM25-rerank.

Output JSONL records:
  {"id": ..., "claim": ..., "evidence": "<concat of top-K sentences>",
   "label": ..., "num_hops": -1, "retrieved_meta": {...}}

Usage:
  python scripts/retrieve_fever_evidence.py --split test
  python scripts/retrieve_fever_evidence.py --split symmetric
  python scripts/retrieve_fever_evidence.py --split train --candidate_docs_top 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
FEVER_BASELINE_DIR = Path("/mnt/data/yangjun/fact/FEVER_baseline/data/FEVER")

# Allow `import utils` for unpickling FEVER_baseline DataSample objects.
if str(FEVER_BASELINE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(FEVER_BASELINE_DIR / "scripts"))

import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402


REPLACEMENTS = {
    "-LRB-": "(", "-LSB-": "[", "-LCB-": "{",
    "-RRB-": ")", "-RSB-": "]", "-RCB-": "}",
    "-COLON-": ":",
}


def untokenize(text: str) -> str:
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


# ---------- DB helpers --------------------------------------------------------


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = 1")
    conn.execute("PRAGMA cache_size = -200000")  # ~200MB page cache
    return conn


def fetch_sentences_for_docs(
    conn: sqlite3.Connection,
    doc_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    """Return a list of {page_id, page_name, line_num, line} for every line in
    the given page set. Empty lines are dropped."""
    if not doc_ids:
        return []
    placeholders = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"SELECT lines.page_id, texts.page_name, lines.line_num, lines.line "
        f"FROM lines JOIN texts ON lines.page_id = texts.page_id "
        f"WHERE lines.page_id IN ({placeholders})",
        tuple(doc_ids),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for page_id, page_name, line_num, line in rows:
        line = (line or "").strip()
        if not line:
            continue
        out.append({
            "page_id": page_id,
            "page_name": page_name,
            "line_num": line_num,
            "line": untokenize(line),
        })
    return out


# ---------- Mode A: paper splits with pre-retrieved doc IDs -------------------


def load_baseline_doc_index(split: str) -> Dict[int, List[int]]:
    """Build {claim_id: [doc_id, ...]} from FEVER_baseline's pickles."""
    claims_path = FEVER_BASELINE_DIR / "processed" / split / "claims.pkl"
    docs_path = FEVER_BASELINE_DIR / "processed" / split / "expanded_doc_ids.pkl"
    with claims_path.open("rb") as f:
        claims = pickle.load(f)
    with docs_path.open("rb") as f:
        docs = pickle.load(f)
    if len(claims) != len(docs):
        raise RuntimeError(
            f"claims/docs length mismatch in split={split}: "
            f"{len(claims)} vs {len(docs)}"
        )
    return {int(c.id): list(d) for c, d in zip(claims, docs)}


# ---------- Mode B: title-index for symmetric --------------------------------


def build_title_index(
    conn: sqlite3.Connection,
    cache_path: Path,
    rebuild: bool = False,
) -> Tuple[List[Tuple[int, str]], Any]:
    """Build (or load) a TF-IDF index over Wikipedia page titles."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    pairs_cache = cache_path / "title_pairs.pkl"
    vec_cache = cache_path / "title_vectorizer.pkl"
    mat_cache = cache_path / "title_matrix.pkl"
    cache_path.mkdir(parents=True, exist_ok=True)
    if not rebuild and pairs_cache.exists() and vec_cache.exists() and mat_cache.exists():
        with pairs_cache.open("rb") as f:
            pairs = pickle.load(f)
        with vec_cache.open("rb") as f:
            vec = pickle.load(f)
        with mat_cache.open("rb") as f:
            mat = pickle.load(f)
        return pairs, (vec, mat)

    print("[title-index] streaming page titles from data.db ...", flush=True)
    pairs: List[Tuple[int, str]] = []
    titles_for_fit: List[str] = []
    t0 = time.time()
    cursor = conn.execute("SELECT page_id, page_name FROM texts")
    for page_id, page_name in cursor:
        # Replace underscores in `og_page_name`-style titles; the `page_name`
        # column is already detokenized.
        if page_name is None:
            continue
        text = page_name.replace("_", " ")
        pairs.append((int(page_id), text))
        titles_for_fit.append(text)
    print(f"[title-index] {len(pairs):,} titles in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    mat = vec.fit_transform(titles_for_fit)
    print(f"[title-index] tfidf fitted in {time.time() - t0:.1f}s; "
          f"shape={mat.shape}", flush=True)
    with pairs_cache.open("wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with vec_cache.open("wb") as f:
        pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    with mat_cache.open("wb") as f:
        pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pairs, (vec, mat)


def title_topk_docs(
    claims: Sequence[str],
    title_pairs: List[Tuple[int, str]],
    vec_mat: Tuple[Any, Any],
    top_n: int,
) -> List[List[int]]:
    """Return, for each claim, its top-`top_n` candidate page_ids by title TF-IDF."""
    vec, mat = vec_mat
    q = vec.transform(claims)
    sims = q @ mat.T  # (n_claims, n_titles)
    out: List[List[int]] = []
    n_titles = mat.shape[0]
    if hasattr(sims, "toarray"):
        for i in range(sims.shape[0]):
            row = sims.getrow(i).toarray().ravel()
            if row.max() == 0:
                out.append([])
                continue
            top = np.argpartition(-row, min(top_n, n_titles - 1))[:top_n]
            top = top[np.argsort(-row[top])]
            out.append([int(title_pairs[j][0]) for j in top])
    else:
        for i in range(sims.shape[0]):
            row = np.asarray(sims[i]).ravel()
            top = np.argpartition(-row, min(top_n, n_titles - 1))[:top_n]
            top = top[np.argsort(-row[top])]
            out.append([int(title_pairs[j][0]) for j in top])
    return out


# ---------- BM25 sentence reranker -------------------------------------------


def bm25_topk_sentences(
    claim: str,
    sentences: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not sentences:
        return []
    corpus = [tokenize(s["line"]) for s in sentences]
    bm25 = BM25Okapi(corpus)
    q = tokenize(claim)
    scores = bm25.get_scores(q)
    if not len(scores):
        return []
    order = np.argsort(-scores)[:top_k]
    out: List[Dict[str, Any]] = []
    for idx in order:
        rec = dict(sentences[int(idx)])
        rec["bm25"] = float(scores[int(idx)])
        out.append(rec)
    return out


# ---------- Top-level driver --------------------------------------------------


def load_egr_records(input_path: Path) -> List[Dict[str, Any]]:
    """Load EGR-FV converted-data records (JSON list or JSONL)."""
    text = input_path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def join_evidence(sents: Sequence[Dict[str, Any]]) -> str:
    return " ".join(s["line"] for s in sents).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_jsonable(r), ensure_ascii=False) + "\n")


def run_paper_split(
    split: str,
    input_path: Path,
    db_path: Path,
    output_path: Path,
    top_k: int,
    candidate_docs_top: int,
    limit: Optional[int],
) -> None:
    records = load_egr_records(input_path)
    if limit:
        records = records[:limit]
    doc_index = load_baseline_doc_index(split)
    conn = open_db(db_path)
    out: List[Dict[str, Any]] = []
    n_missing = 0
    n_processed = 0
    t0 = time.time()
    for i, rec in enumerate(records):
        cid = int(rec["id"])
        cand_docs = doc_index.get(cid, [])[:candidate_docs_top]
        if not cand_docs:
            n_missing += 1
            evidence = ""
            meta = {"candidate_docs": [], "note": "no_candidate_docs"}
        else:
            sents = fetch_sentences_for_docs(conn, cand_docs)
            top = bm25_topk_sentences(rec["claim"], sents, top_k)
            evidence = join_evidence(top)
            meta = {
                "candidate_docs": cand_docs,
                "top_sentences": [
                    {
                        "page_id": s["page_id"],
                        "page_name": s["page_name"],
                        "line_num": s["line_num"],
                        "bm25": s["bm25"],
                    }
                    for s in top
                ],
            }
        new_rec = dict(rec)
        new_rec["evidence"] = evidence
        new_rec["retrieved_meta"] = meta
        new_rec.setdefault("num_hops", -1)
        out.append(new_rec)
        n_processed += 1
        if n_processed % 200 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-3)
            print(f"  [{split}] {n_processed}/{len(records)} "
                  f"({rate:.1f} claims/s, missing={n_missing})", flush=True)
    write_jsonl(out, output_path)
    print(f"[done] {split}: wrote {len(out)} → {output_path}; missing={n_missing}; "
          f"elapsed={time.time() - t0:.1f}s")


def run_symmetric_split(
    input_path: Path,
    db_path: Path,
    output_path: Path,
    top_k: int,
    candidate_docs_top: int,
    title_index_dir: Path,
    rebuild_index: bool,
    limit: Optional[int],
) -> None:
    records = load_egr_records(input_path)
    if limit:
        records = records[:limit]
    conn = open_db(db_path)
    title_pairs, vec_mat = build_title_index(conn, title_index_dir, rebuild=rebuild_index)
    claims = [r["claim"] for r in records]
    print(f"[symmetric] ranking titles for {len(claims)} claims ...", flush=True)
    cand_lists = title_topk_docs(claims, title_pairs, vec_mat, top_n=candidate_docs_top)
    out: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (rec, cand_docs) in enumerate(zip(records, cand_lists)):
        sents = fetch_sentences_for_docs(conn, cand_docs)
        top = bm25_topk_sentences(rec["claim"], sents, top_k)
        evidence = join_evidence(top)
        new_rec = dict(rec)
        new_rec["evidence"] = evidence
        new_rec["retrieved_meta"] = {
            "candidate_docs": cand_docs,
            "top_sentences": [
                {
                    "page_id": s["page_id"],
                    "page_name": s["page_name"],
                    "line_num": s["line_num"],
                    "bm25": s["bm25"],
                }
                for s in top
            ],
        }
        new_rec.setdefault("num_hops", -1)
        out.append(new_rec)
        if (i + 1) % 100 == 0:
            print(f"  [symmetric] {i + 1}/{len(records)} "
                  f"({(i + 1) / max(time.time() - t0, 1e-3):.1f} claims/s)",
                  flush=True)
    write_jsonl(out, output_path)
    print(f"[done] symmetric: wrote {len(out)} → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True,
                        choices=["test", "train", "dev", "symmetric"])
    parser.add_argument("--input_path", default=None,
                        help="Override the EGR-FV converted-data file.")
    parser.add_argument("--output_path", default=None,
                        help="Override the output JSONL path.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of sentences kept per claim.")
    parser.add_argument("--candidate_docs_top", type=int, default=10,
                        help="(paper) cap on candidate doc IDs to inspect per claim; "
                             "(symmetric) number of title-matched pages.")
    parser.add_argument("--db_path", default=str(FEVER_BASELINE_DIR / "processed" / "data.db"))
    parser.add_argument("--title_index_dir", default="data/FEVER/retrieved/_title_index")
    parser.add_argument("--rebuild_title_index", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only retrieve the first N claims (debug).")
    args = parser.parse_args()

    egr_data_dir = REPO_ROOT / "data" / "FEVER"
    default_input = {
        "test": egr_data_dir / "converted_data" / "test.json",
        "train": egr_data_dir / "converted_data" / "train.json",
        "dev": egr_data_dir / "converted_data" / "dev.json",
        "symmetric": egr_data_dir / "converted_data" / "symmetric.json",
    }[args.split]
    default_output = REPO_ROOT / "data" / "FEVER" / "retrieved" / f"{args.split}.jsonl"
    input_path = Path(args.input_path) if args.input_path else default_input
    output_path = Path(args.output_path) if args.output_path else default_output

    print(f"[start] split={args.split}  in={input_path}  out={output_path}  "
          f"top_k={args.top_k}  candidate_docs_top={args.candidate_docs_top}", flush=True)

    if args.split == "symmetric":
        run_symmetric_split(
            input_path=input_path,
            db_path=Path(args.db_path),
            output_path=output_path,
            top_k=args.top_k,
            candidate_docs_top=args.candidate_docs_top,
            title_index_dir=REPO_ROOT / args.title_index_dir,
            rebuild_index=args.rebuild_title_index,
            limit=args.limit,
        )
    else:
        run_paper_split(
            split={"test": "test", "train": "train", "dev": "dev"}[args.split],
            input_path=input_path,
            db_path=Path(args.db_path),
            output_path=output_path,
            top_k=args.top_k,
            candidate_docs_top=args.candidate_docs_top,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
