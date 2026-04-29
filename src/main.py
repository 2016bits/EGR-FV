from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


REQUIRED_MODULES = ("torch", "transformers", "yaml", "sklearn", "tqdm")
PYTHON_BOOTSTRAP_FLAG = "EGRFV_BOOTSTRAPPED_PYTHON"


def _missing_required_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _python_has_required_modules(python_bin: str) -> bool:
    code = (
        "import importlib.util, sys; "
        f"required = {REQUIRED_MODULES!r}; "
        "missing = [name for name in required if importlib.util.find_spec(name) is None]; "
        "sys.exit(0 if not missing else 1)"
    )
    try:
        result = subprocess.run(
            [python_bin, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _iter_python_candidates():
    requested_python = os.environ.get("PYTHON_BIN", "python")
    current_user = os.environ.get("USER", "")
    home_dir = os.environ.get("HOME", "")

    preferred = [
        requested_python,
        os.path.join(home_dir, ".conda", "envs", "tor230", "bin", "python"),
        os.path.join("/data", current_user, ".conda", "envs", "tor230", "bin", "python"),
        "/data/yangjun/.conda/envs/tor230/bin/python",
        os.path.join(home_dir, ".conda", "envs", "egrfv", "bin", "python"),
        os.path.join("/data", current_user, ".conda", "envs", "egrfv", "bin", "python"),
        "/data/yangjun/.conda/envs/egrfv/bin/python",
    ]

    for root in (
        os.path.join(home_dir, ".conda", "envs"),
        os.path.join("/data", current_user, ".conda", "envs"),
        "/data/yangjun/.conda/envs",
    ):
        preferred.extend(glob.glob(os.path.join(root, "*", "bin", "python")))

    seen = set()
    for candidate in preferred:
        if not candidate:
            continue
        has_path_sep = os.path.sep in candidate or (os.path.altsep is not None and os.path.altsep in candidate)
        if has_path_sep:
            resolved = candidate
        else:
            resolved = shutil.which(candidate)
            if resolved is None:
                continue
        resolved = os.path.abspath(resolved)
        if resolved in seen or not os.path.exists(resolved):
            continue
        seen.add(resolved)
        yield resolved


def _bootstrap_python_with_required_modules() -> None:
    missing = _missing_required_modules()
    if not missing:
        return

    if os.environ.get(PYTHON_BOOTSTRAP_FLAG) == "1":
        missing_text = ", ".join(missing)
        raise SystemExit(
            f"Missing required Python modules: {missing_text}. "
            "Install dependencies with `pip install -r requirements.txt`."
        )

    current_python = os.path.abspath(sys.executable)
    for candidate in _iter_python_candidates():
        if os.path.abspath(candidate) == current_python:
            continue
        if _python_has_required_modules(candidate):
            env = os.environ.copy()
            env[PYTHON_BOOTSTRAP_FLAG] = "1"
            os.execve(candidate, [candidate, "-m", "src.main", *sys.argv[1:]], env)

    missing_text = ", ".join(missing)
    raise SystemExit(
        f"Missing required Python modules in {sys.executable}: {missing_text}. "
        "Set PYTHON_BIN=/path/to/python or install dependencies with `pip install -r requirements.txt`."
    )


_bootstrap_python_with_required_modules()

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.ablation_groups import assign_pseudo_groups, summarize_record_groups
from src.data.collator import GroundedCollator, JointCollator, ShortcutCollator
from src.data.dataset import DEFAULT_ID_TO_LABEL, FactVerificationDataset
from src.data.routing import run_routing
from src.models.fusion_head import GatedFusionHead
from src.models.grounded_model import GroundedModel
from src.models.shortcut_model import ShortcutModel
from src.trainers.evaluator import Evaluator
from src.trainers.remix_trainer import RemixTrainer
from src.trainers.warmup_trainer import WarmupTrainer
from src.utils.config import load_config
from src.utils.experiments import experiment_mode, requires_routing_file, uses_pseudo_groups, uses_real_routing
from src.utils.io import ensure_dir, load_json_or_jsonl, load_model_state
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGR-FV main entry")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["warmup_shortcut", "warmup_grounded", "routing", "remix", "eval"],
    )
    parser.add_argument("--shortcut_ckpt", default=None, help="Optional shortcut checkpoint override")
    parser.add_argument("--grounded_ckpt", default=None, help="Optional grounded checkpoint override")
    parser.add_argument("--ckpt", default=None, help="Optional evaluation/remix checkpoint override")
    return parser.parse_args()


def build_label_maps(config: Mapping[str, Any]) -> tuple[Dict[str, int], Dict[int, str]]:
    label_map = config.get("label_map") or config.get("data", {}).get("label_map")
    if not label_map:
        inverse = dict(DEFAULT_ID_TO_LABEL)
        forward = {label.lower(): idx for idx, label in inverse.items()}
        return forward, inverse

    inverse = {int(value): str(key).upper() for key, value in label_map.items()}
    forward = {str(key).lower(): int(value) for key, value in label_map.items()}
    return forward, inverse


def patch_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token


def build_tokenizers(config: Mapping[str, Any]):
    shortcut_tokenizer = AutoTokenizer.from_pretrained(config["model"]["shortcut_encoder"], use_fast=True)
    grounded_tokenizer = AutoTokenizer.from_pretrained(config["model"]["grounded_encoder"], use_fast=True)
    patch_tokenizer(shortcut_tokenizer)
    patch_tokenizer(grounded_tokenizer)
    return shortcut_tokenizer, grounded_tokenizer


def build_collators(config: Mapping[str, Any], shortcut_tokenizer, grounded_tokenizer):
    data_cfg = config["data"]
    kwargs = {
        "shortcut_tokenizer": shortcut_tokenizer,
        "grounded_tokenizer": grounded_tokenizer,
        "max_claim_len": int(data_cfg.get("max_claim_len", 128)),
        "max_evidence_len": int(data_cfg.get("max_evidence_len", 384)),
        "max_seq_len": int(data_cfg.get("max_seq_len", 512)),
    }
    return (
        ShortcutCollator(**kwargs),
        GroundedCollator(**kwargs),
        JointCollator(**kwargs),
    )


def build_dataset(
    config: Mapping[str, Any],
    split: str,
    label_to_id: Mapping[str, int],
    id_to_label: Mapping[int, str],
    use_routing: bool = False,
) -> FactVerificationDataset:
    data_cfg = config["data"]
    loss_cfg = config["loss"]
    group_weights = {
        "grounded_needed": float(loss_cfg.get("grounded_needed_weight", 1.5)),
        "hard": float(loss_cfg.get("hard_weight", 1.0)),
        "bias_easy": float(loss_cfg.get("bias_easy_weight", 0.5)),
    }
    routing_path = data_cfg.get("routing_path") if use_routing else None
    return FactVerificationDataset(
        data_path=data_cfg[f"{split}_path"],
        routing_path=routing_path,
        group_weights=group_weights,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )


def build_dataloader(config: Mapping[str, Any], dataset, collator, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
    )


def build_shortcut_model(config: Mapping[str, Any]) -> ShortcutModel:
    model_cfg = config["model"]
    return ShortcutModel(
        encoder_name=model_cfg["shortcut_encoder"],
        num_labels=int(config["num_labels"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def build_grounded_model(config: Mapping[str, Any]) -> GroundedModel:
    model_cfg = config["model"]
    return GroundedModel(
        encoder_name=model_cfg["grounded_encoder"],
        num_labels=int(config["num_labels"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def build_models(config: Mapping[str, Any]):
    shortcut_model = build_shortcut_model(config)
    grounded_model = build_grounded_model(config)
    fusion_head = None
    if bool(config["model"].get("use_fusion", False)):
        fusion_head = GatedFusionHead(
            hidden_size=grounded_model.config.hidden_size,
            num_labels=int(config["num_labels"]),
        )
    return shortcut_model, grounded_model, fusion_head


def resolve_ckpt(config: Mapping[str, Any], key: str, override: str | None = None) -> str | None:
    if override:
        return override
    return config.get("checkpoints", {}).get(key)


def require_existing_file(path: str | None, description: str) -> str:
    if not path:
        raise FileNotFoundError(f"{description} path is not configured.")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return str(file_path)


def ensure_output_dirs(config: Mapping[str, Any]) -> None:
    outputs_cfg = config["outputs"]
    ensure_dir(outputs_cfg["checkpoint_dir"])
    ensure_dir(outputs_cfg["log_dir"])
    ensure_dir(outputs_cfg["prediction_dir"])
    routing_path = Path(config["data"]["routing_path"])
    ensure_dir(routing_path.parent)


def validate_routing_cache(config: Mapping[str, Any], routing_path: Path) -> None:
    records = load_json_or_jsonl(routing_path)
    if not records:
        raise ValueError(
            f"Routing cache {routing_path} is empty. "
            "Please rerun routing mode before remix training."
        )
    required_keys = {"id", "group", "sample_weight"}
    missing_keys = required_keys - set(records[0])
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Routing cache {routing_path} is missing required key(s): {missing}. "
            "Please rerun routing mode before remix training."
        )
    if not bool(config.get("routing", {}).get("stratify_by_label", False)):
        return
    if records and "label_id" not in records[0]:
        raise ValueError(
            f"Routing cache {routing_path} was generated before label-stratified routing was enabled. "
            "Please rerun routing mode before remix training."
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print("Loaded config:")
    print(f"Experiment mode: {experiment_mode(config)}")
    ensure_output_dirs(config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_id, id_to_label = build_label_maps(config)
    shortcut_tokenizer, grounded_tokenizer = build_tokenizers(config)
    shortcut_collator, grounded_collator, joint_collator = build_collators(
        config,
        shortcut_tokenizer,
        grounded_tokenizer,
    )
    print(f"Using device: {device}")

    if args.mode == "warmup_shortcut":
        train_dataset = build_dataset(config, "train", label_to_id, id_to_label)
        val_dataset = build_dataset(config, "val", label_to_id, id_to_label)
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        train_loader = build_dataloader(config, train_dataset, shortcut_collator, shuffle=True)
        val_loader = build_dataloader(config, val_dataset, shortcut_collator, shuffle=False)
        print("DataLoaders built.")

        shortcut_model = build_shortcut_model(config)
        print(f"Shortcut model built.")
        trainer = WarmupTrainer(
            config=config,
            model=shortcut_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            id_to_label=id_to_label,
            checkpoint_dir=config["outputs"]["checkpoint_dir"],
            log_dir=config["outputs"]["log_dir"],
        )
        trainer.train_shortcut()
        return

    if args.mode == "warmup_grounded":
        train_dataset = build_dataset(config, "train", label_to_id, id_to_label)
        val_dataset = build_dataset(config, "val", label_to_id, id_to_label)
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        train_loader = build_dataloader(config, train_dataset, grounded_collator, shuffle=True)
        val_loader = build_dataloader(config, val_dataset, grounded_collator, shuffle=False)
        print("DataLoaders built.")

        grounded_model = build_grounded_model(config)
        print(f"Grounded model built.")
        trainer = WarmupTrainer(
            config=config,
            model=grounded_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            id_to_label=id_to_label,
            checkpoint_dir=config["outputs"]["checkpoint_dir"],
            log_dir=config["outputs"]["log_dir"],
        )
        trainer.train_grounded()
        return

    if args.mode == "routing":
        train_dataset = build_dataset(config, "train", label_to_id, id_to_label)
        dataloader = build_dataloader(config, train_dataset, joint_collator, shuffle=False)
        print("DataLoader built.")

        shortcut_model, grounded_model, _ = build_models(config)
        print("Models built.")
        shortcut_ckpt = resolve_ckpt(config, "shortcut", args.shortcut_ckpt)
        grounded_ckpt = resolve_ckpt(config, "grounded", args.grounded_ckpt)
        shortcut_ckpt = require_existing_file(shortcut_ckpt, "Shortcut warmup checkpoint")
        grounded_ckpt = require_existing_file(grounded_ckpt, "Grounded warmup checkpoint")
        load_model_state(shortcut_model, shortcut_ckpt, ["model_state", "shortcut_model_state"], map_location=device)
        load_model_state(grounded_model, grounded_ckpt, ["model_state", "grounded_model_state"], map_location=device)
        shortcut_model.to(device)
        grounded_model.to(device)
        routing_path = config["data"]["routing_path"]
        stats_path = str(Path(routing_path).with_suffix(".stats.json"))
        run_routing(
            shortcut_model=shortcut_model,
            grounded_model=grounded_model,
            dataloader=dataloader,
            device=device,
            routing_config=config["routing"],
            output_path=routing_path,
            stats_path=stats_path,
        )
        return

    if args.mode == "remix":
        routing_path = Path(config["data"]["routing_path"])
        if requires_routing_file(config) and not routing_path.exists():
            raise FileNotFoundError(
                f"Routing file not found: {routing_path}. "
                "This experiment mode uses real sample routing; please run routing mode first."
            )
        if uses_real_routing(config):
            validate_routing_cache(config, routing_path)
        train_dataset = build_dataset(
            config,
            "train",
            label_to_id,
            id_to_label,
            use_routing=uses_real_routing(config),
        )
        if uses_pseudo_groups(config):
            train_dataset = assign_pseudo_groups(train_dataset, config)
            print(f"Pseudo group stats: {summarize_record_groups(train_dataset.records)}")
        elif uses_real_routing(config):
            print(f"Routing group stats: {summarize_record_groups(train_dataset.records)}")
        val_dataset = build_dataset(config, "val", label_to_id, id_to_label)
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        shortcut_model, grounded_model, fusion_head = build_models(config)
        print(f"Models built.")
        remix_ckpt = args.ckpt
        loaded_remix = False
        if remix_ckpt:
            remix_ckpt = require_existing_file(remix_ckpt, "Remix checkpoint")
            checkpoint = load_model_state(
                grounded_model,
                remix_ckpt,
                ["grounded_model_state", "model_state"],
                map_location=device,
            )
            if "shortcut_model_state" in checkpoint:
                shortcut_model.load_state_dict(checkpoint["shortcut_model_state"])
            if fusion_head is not None and "fusion_head_state" in checkpoint:
                fusion_head.load_state_dict(checkpoint["fusion_head_state"])
            loaded_remix = True
        shortcut_ckpt = resolve_ckpt(config, "shortcut", args.shortcut_ckpt)
        grounded_ckpt = resolve_ckpt(config, "grounded", args.grounded_ckpt)
        if not loaded_remix:
            shortcut_ckpt = require_existing_file(shortcut_ckpt, "Shortcut warmup checkpoint")
            grounded_ckpt = require_existing_file(grounded_ckpt, "Grounded warmup checkpoint")
            load_model_state(shortcut_model, shortcut_ckpt, ["model_state", "shortcut_model_state"], map_location=device)
            load_model_state(grounded_model, grounded_ckpt, ["model_state", "grounded_model_state"], map_location=device)
        trainer = RemixTrainer(
            config=config,
            shortcut_model=shortcut_model,
            grounded_model=grounded_model,
            fusion_head=fusion_head,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            joint_collator=joint_collator,
            shortcut_collator=shortcut_collator,
            grounded_collator=grounded_collator,
            device=device,
            id_to_label=id_to_label,
            checkpoint_dir=config["outputs"]["checkpoint_dir"],
            log_dir=config["outputs"]["log_dir"],
        )
        trainer.train()
        return

    test_dataset = build_dataset(config, "test", label_to_id, id_to_label)
    shortcut_model, grounded_model, fusion_head = build_models(config)
    shortcut_loaded = False
    fusion_loaded = False

    eval_ckpt = resolve_ckpt(config, "remix", args.ckpt)
    if args.ckpt and not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {args.ckpt}")
    if eval_ckpt and Path(eval_ckpt).exists():
        checkpoint = load_model_state(
            grounded_model,
            eval_ckpt,
            ["grounded_model_state", "model_state"],
            map_location=device,
        )
        if "shortcut_model_state" in checkpoint:
            shortcut_model.load_state_dict(checkpoint["shortcut_model_state"])
            shortcut_loaded = True
        else:
            shortcut_ckpt = resolve_ckpt(config, "shortcut", args.shortcut_ckpt)
            if shortcut_ckpt and Path(shortcut_ckpt).exists():
                load_model_state(
                    shortcut_model,
                    shortcut_ckpt,
                    ["model_state", "shortcut_model_state"],
                    map_location=device,
                )
                shortcut_loaded = True
        if fusion_head is not None and "fusion_head_state" in checkpoint:
            fusion_head.load_state_dict(checkpoint["fusion_head_state"])
            fusion_loaded = True
    else:
        grounded_ckpt = resolve_ckpt(config, "grounded", args.grounded_ckpt)
        shortcut_ckpt = resolve_ckpt(config, "shortcut", args.shortcut_ckpt)
        if grounded_ckpt and Path(grounded_ckpt).exists():
            load_model_state(
                grounded_model,
                grounded_ckpt,
                ["model_state", "grounded_model_state"],
                map_location=device,
            )
        if shortcut_ckpt and Path(shortcut_ckpt).exists():
            load_model_state(
                shortcut_model,
                shortcut_ckpt,
                ["model_state", "shortcut_model_state"],
                map_location=device,
            )
            shortcut_loaded = True

    fusion_requested = bool(config["model"].get("use_fusion", False)) and fusion_head is not None
    if fusion_requested and not fusion_loaded:
        raise ValueError(
            "Evaluation config requests model.use_fusion=true, but no trained fusion_head_state was loaded. "
            f"Checkpoint checked: {eval_ckpt}. Please rerun remix training with configs/remix.yaml "
            "or evaluate with configs/ablation_no_fusion.yaml."
        )

    if not shortcut_loaded:
        shortcut_model = None

    if shortcut_model is not None:
        shortcut_model.to(device)
    grounded_model.to(device)
    if fusion_head is not None:
        fusion_head.to(device)

    evaluator = Evaluator(
        shortcut_model=shortcut_model,
        grounded_model=grounded_model,
        fusion_head=fusion_head,
        joint_collator=joint_collator,
        shortcut_collator=shortcut_collator,
        grounded_collator=grounded_collator,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        device=device,
        id_to_label=id_to_label,
        evidence_placeholder=str(config["data"].get("evidence_placeholder", "[NO_EVIDENCE]")),
    )
    eval_mode = (
        "fusion"
        if bool(config["model"].get("use_fusion", False)) and shortcut_model is not None and fusion_head is not None
        else "grounded"
    )
    evaluator.run_full_evaluation(
        dataset=test_dataset,
        report_path=str(Path(config["outputs"]["prediction_dir"]) / "eval_report.json"),
        predictions_path=str(Path(config["outputs"]["prediction_dir"]) / "eval_predictions.jsonl"),
        mode=eval_mode,
    )


if __name__ == "__main__":
    main()
