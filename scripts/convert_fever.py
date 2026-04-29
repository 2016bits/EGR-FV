from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SPLITS = ("train", "dev", "test")
RAW_FILENAME_TEMPLATE = "{split}_2.json"
IPA_CHARS_PATTERN = re.compile(r"[^\x00-\x7F]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FEVER raw JSON files to EGR-FV format.")
    parser.add_argument("--raw-dir", default="data/FEVER/raw", help="Directory containing train_2/dev_2/test_2 JSON files")
    parser.add_argument(
        "--output-dir",
        default="data/FEVER/converted_data",
        help="Directory where converted train/dev/test JSON files are written",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to convert. Defaults to train dev test.",
    )
    parser.add_argument("--indent", type=int, default=4, help="JSON indentation for converted files")
    return parser.parse_args()


def flatten_evidence(evidence: Any) -> str:
    if evidence is None:
        return ""
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, Iterable):
        parts = []
        for item in evidence:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(flatten_evidence(item))
        return " ".join(part for part in parts if part)
    return str(evidence)


def clean_evidence(raw_evidence: Any) -> str:
    text = flatten_evidence(raw_evidence)
    text = re.sub(r"\[[^\[\]]*\]", " ", text)
    text = IPA_CHARS_PATTERN.sub(" ", text)
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "claim": str(record.get("claim", "")).strip(),
        "label": str(record["label"]).lower(),
        "num_hops": int(record.get("num_hops", -1) or -1),
        "evidence": clean_evidence(record.get("gold_evidence", record.get("evidence", ""))),
    }


def convert_split(raw_dir: Path, output_dir: Path, split: str, indent: int) -> Path:
    raw_path = raw_dir / RAW_FILENAME_TEMPLATE.format(split=split)
    output_path = output_dir / f"{split}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw FEVER split not found: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as handle:
        raw_records = json.load(handle)
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a list of FEVER records in {raw_path}")

    converted = [convert_record(record) for record in raw_records]
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=indent)
    return output_path


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    for split in args.splits:
        output_path = convert_split(raw_dir, output_dir, split, args.indent)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
