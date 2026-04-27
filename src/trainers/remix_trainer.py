from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data.remix_sampler import RemixBatchScheduler
from src.models.losses import cross_entropy_loss, orthogonality_loss, weighted_cross_entropy
from src.trainers.evaluator import Evaluator
from src.utils.experiments import experiment_mode, uses_batch_remix, uses_uniform_sample_weights
from src.utils.io import save_torch_checkpoint
from src.utils.logger import CSVLogger


class RemixTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        shortcut_model,
        grounded_model,
        fusion_head,
        train_dataset,
        val_dataset,
        joint_collator,
        shortcut_collator,
        grounded_collator,
        device: torch.device,
        id_to_label: Mapping[int, str],
        checkpoint_dir: str,
        log_dir: str,
    ) -> None:
        self.config = config
        self.shortcut_model = shortcut_model.to(device)
        self.grounded_model = grounded_model.to(device)
        self.fusion_head = fusion_head.to(device) if fusion_head is not None else None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.joint_collator = joint_collator
        self.shortcut_collator = shortcut_collator
        self.grounded_collator = grounded_collator
        self.device = device
        self.id_to_label = dict(id_to_label)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

    def _build_loader(self, dataset, shuffle: bool = False, sampler=None) -> DataLoader:
        num_workers = int(self.config["data"].get("num_workers", 0))
        batch_size = int(self.config["training"]["batch_size"])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self.joint_collator,
        )

    def _build_balanced_sampler(self, dataset, group_ratios: Mapping[str, float] | None = None):
        if not len(dataset):
            return None

        label_balance = bool(self.config["remix"].get("label_balance", False))
        if group_ratios is None and not label_balance:
            return None

        counts_by_label = Counter(record["label"] for record in dataset.records if int(record["label"]) >= 0)
        counts_by_group = Counter(record["group"] for record in dataset.records)
        weights = []
        for record in dataset.records:
            weight = 1.0
            if group_ratios is not None:
                group = record["group"]
                target_ratio = float(group_ratios.get(group, 0.0))
                count = max(1, counts_by_group.get(group, 0))
                weight *= target_ratio / count if target_ratio > 0 else 0.0
            if label_balance and int(record["label"]) >= 0:
                weight *= 1.0 / max(1, counts_by_label.get(record["label"], 0))
            weights.append(weight)

        if not any(weight > 0 for weight in weights):
            return None
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def _build_mixed_sampler(self):
        ratios = self.config["remix"].get(
            "mixed_ratio",
            {"bias_easy": 0.3, "grounded_needed": 0.4, "hard": 0.3},
        )
        return self._build_balanced_sampler(self.train_dataset, group_ratios=ratios)

    def _batch_loss_coefficients(
        self,
        batch_type: str,
        mode: str,
        lambda_shortcut: float,
        lambda_orth: float,
    ) -> tuple[float, float, float]:
        configured = self.config["remix"].get("loss_weights", {})
        if batch_type in configured:
            weights = configured[batch_type]
            return (
                float(weights.get("grounded", 1.0)),
                float(weights.get("shortcut", lambda_shortcut)),
                float(weights.get("orth", 1.0)) * lambda_orth,
            )

        if mode == "two_branch_only":
            return 1.0, lambda_shortcut, lambda_orth
        if batch_type == "bias":
            return 0.1, 1.0, 0.0
        if batch_type == "grounded":
            return 1.5, 0.2, lambda_orth
        return 1.0, lambda_shortcut, lambda_orth

    def _build_evaluator(self) -> Evaluator:
        return Evaluator(
            shortcut_model=self.shortcut_model,
            grounded_model=self.grounded_model,
            fusion_head=self.fusion_head,
            joint_collator=self.joint_collator,
            shortcut_collator=self.shortcut_collator,
            grounded_collator=self.grounded_collator,
            batch_size=int(self.config["training"]["batch_size"]),
            num_workers=int(self.config["data"].get("num_workers", 0)),
            device=self.device,
            id_to_label=self.id_to_label,
            evidence_placeholder=str(self.config["data"].get("evidence_placeholder", "[NO_EVIDENCE]")),
        )

    def train(self) -> str:
        training_cfg = self.config["training"]
        loss_cfg = self.config["loss"]
        remix_cfg = self.config["remix"]
        mode = experiment_mode(self.config)
        batch_remix_enabled = uses_batch_remix(self.config)
        uniform_sample_weights = uses_uniform_sample_weights(self.config)
        num_epochs = int(training_cfg["epochs"])
        lr = float(training_cfg["lr"])
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        lambda_shortcut = float(loss_cfg.get("lambda_shortcut", 0.3))
        lambda_orth = float(loss_cfg.get("lambda_orth", 0.0))
        fusion_weight = float(loss_cfg.get("lambda_fusion", 0.5))
        use_fusion = bool(self.config["model"].get("use_fusion", False)) and self.fusion_head is not None

        plain_loader = None
        scheduler = None
        if batch_remix_enabled:
            bias_dataset = self.train_dataset.subset({"bias_easy"})
            grounded_dataset = self.train_dataset.subset({"grounded_needed"})
            mixed_sampler = self._build_mixed_sampler()
            bias_sampler = self._build_balanced_sampler(bias_dataset)
            grounded_sampler = self._build_balanced_sampler(grounded_dataset)

            bias_loader = (
                self._build_loader(bias_dataset, shuffle=bias_sampler is None, sampler=bias_sampler)
                if len(bias_dataset)
                else None
            )
            grounded_loader = (
                self._build_loader(grounded_dataset, shuffle=grounded_sampler is None, sampler=grounded_sampler)
                if len(grounded_dataset)
                else None
            )
            mixed_loader = self._build_loader(self.train_dataset, sampler=mixed_sampler) if len(self.train_dataset) else None

            scheduler = RemixBatchScheduler(
                bias_loader=bias_loader,
                grounded_loader=grounded_loader,
                mixed_loader=mixed_loader,
                schedule_type=str(remix_cfg.get("schedule", "alternating")),
                fixed_ratio=remix_cfg.get("batch_ratio"),
                max_epochs=num_epochs,
                seed=int(self.config.get("seed", 42)),
            )
            default_steps_per_epoch = max(
                len(loader) for loader in [bias_loader, grounded_loader, mixed_loader] if loader is not None
            )
        else:
            plain_sampler = self._build_balanced_sampler(self.train_dataset)
            plain_loader = self._build_loader(
                self.train_dataset,
                shuffle=plain_sampler is None,
                sampler=plain_sampler,
            )
            default_steps_per_epoch = len(plain_loader)

        steps_per_epoch = int(remix_cfg.get("steps_per_epoch", default_steps_per_epoch))

        parameters = list(self.shortcut_model.parameters()) + list(self.grounded_model.parameters())
        if self.fusion_head is not None:
            parameters += list(self.fusion_head.parameters())
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
        total_steps = max(1, num_epochs * steps_per_epoch)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )
        csv_logger = CSVLogger(str(self.log_dir / "remix_train.csv"))
        evaluator = self._build_evaluator()
        checkpoint_path = str(self.checkpoint_dir / "remix_best.pt")
        best_f1 = -1.0

        for epoch in range(num_epochs):
            self.shortcut_model.train()
            self.grounded_model.train()
            if self.fusion_head is not None:
                self.fusion_head.train()

            running = {
                "loss": 0.0,
                "loss_shortcut": 0.0,
                "loss_grounded": 0.0,
                "loss_orth": 0.0,
                "loss_fusion": 0.0,
            }
            plain_iterator = iter(plain_loader) if plain_loader is not None else None
            progress = tqdm(range(steps_per_epoch), desc=f"{mode}:epoch{epoch + 1}", leave=False)
            for global_step in progress:
                if scheduler is not None:
                    batch_type, batch = scheduler.next_batch(global_step=global_step, epoch=epoch)
                else:
                    batch_type = "random"
                    try:
                        batch = next(plain_iterator)
                    except StopIteration:
                        plain_iterator = iter(plain_loader)
                        batch = next(plain_iterator)

                shortcut_inputs = {key: value.to(self.device) for key, value in batch["shortcut_inputs"].items()}
                grounded_inputs = {key: value.to(self.device) for key, value in batch["grounded_inputs"].items()}
                labels = batch["labels"].to(self.device)
                sample_weights = batch["weights"].to(self.device)
                if uniform_sample_weights:
                    sample_weights = torch.ones_like(sample_weights)

                shortcut_outputs = self.shortcut_model(**shortcut_inputs)
                grounded_outputs = self.grounded_model(**grounded_inputs)

                loss_s = cross_entropy_loss(shortcut_outputs["logits"], labels)
                loss_g = weighted_cross_entropy(grounded_outputs["logits"], labels, sample_weights)
                loss_o = orthogonality_loss(shortcut_outputs["hidden"], grounded_outputs["hidden"])
                loss_f = torch.tensor(0.0, device=self.device)

                if use_fusion:
                    fused_outputs = self.fusion_head(
                        shortcut_hidden=shortcut_outputs["hidden"],
                        grounded_hidden=grounded_outputs["hidden"],
                        shortcut_probs=shortcut_outputs["probs"],
                        grounded_probs=grounded_outputs["probs"],
                    )
                    loss_f = weighted_cross_entropy(fused_outputs["logits"], labels, sample_weights)

                grounded_coef, shortcut_coef, orth_coef = self._batch_loss_coefficients(
                    batch_type=batch_type,
                    mode=mode,
                    lambda_shortcut=lambda_shortcut,
                    lambda_orth=lambda_orth,
                )
                total_loss = grounded_coef * loss_g + shortcut_coef * loss_s + orth_coef * loss_o

                if use_fusion:
                    total_loss = total_loss + fusion_weight * loss_f

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                running["loss"] += float(total_loss.item())
                running["loss_shortcut"] += float(loss_s.item())
                running["loss_grounded"] += float(loss_g.item())
                running["loss_orth"] += float(loss_o.item())
                running["loss_fusion"] += float(loss_f.item())
                progress.set_postfix(batch=batch_type, loss=f"{total_loss.item():.4f}")

            epoch_log = {key: value / max(1, steps_per_epoch) for key, value in running.items()}
            eval_mode = "fusion" if use_fusion else "grounded"
            val_report = evaluator.evaluate_dataset(self.val_dataset, mode=eval_mode)
            csv_logger.log(
                {
                    "epoch": epoch + 1,
                    **epoch_log,
                    **{f"val_{key}": value for key, value in val_report["metrics"].items()},
                    **{f"val_calibration_{key}": value for key, value in val_report["calibration"].items()},
                }
            )

            if val_report["metrics"]["macro_f1"] > best_f1:
                best_f1 = val_report["metrics"]["macro_f1"]
                payload = {
                    "shortcut_model_state": self.shortcut_model.state_dict(),
                    "grounded_model_state": self.grounded_model.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": val_report["metrics"],
                    "config": dict(self.config),
                }
                if self.fusion_head is not None:
                    payload["fusion_head_state"] = self.fusion_head.state_dict()
                save_torch_checkpoint(checkpoint_path, payload)

        return checkpoint_path
