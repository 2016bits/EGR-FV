from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {file_path}: {type(payload)!r}")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_torch_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    file_path = ensure_parent_dir(path)
    torch.save(payload, file_path)


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")
    checkpoint = torch.load(file_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {file_path} is not a dict payload.")
    return checkpoint


def load_model_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    candidate_keys: Iterable[str],
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=map_location)
    for key in candidate_keys:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            return checkpoint
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        model.load_state_dict(checkpoint)
        return checkpoint
    raise KeyError(
        f"None of the candidate keys {list(candidate_keys)} were found in checkpoint {checkpoint_path}."
    )
