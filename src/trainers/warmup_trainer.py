from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models.losses import cross_entropy_loss
from src.utils.io import save_torch_checkpoint
from src.utils.logger import CSVLogger
from src.utils.metrics import classification_metrics


class WarmupTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        id_to_label: Mapping[int, str],
        checkpoint_dir: str,
        log_dir: str,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.id_to_label = dict(id_to_label)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

    def _move_inputs(self, batch_inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch_inputs.items()}

    def _evaluate(self, input_key: str) -> Dict[str, float]:
        self.model.eval()
        labels = []
        predictions = []

        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(**self._move_inputs(batch[input_key]))
                preds = outputs["probs"].argmax(dim=-1).detach().cpu().tolist()
                batch_labels = batch["labels"].detach().cpu().tolist()
                for label, pred in zip(batch_labels, preds):
                    if label >= 0:
                        labels.append(int(label))
                        predictions.append(int(pred))

        return classification_metrics(labels, predictions, label_names=self.id_to_label)

    def _train(self, input_key: str, checkpoint_name: str, log_name: str) -> str:
        training_cfg = self.config["training"]
        epochs = int(training_cfg["epochs"])
        lr = float(training_cfg["lr"])
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = max(1, epochs * len(self.train_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )
        csv_logger = CSVLogger(str(self.log_dir / log_name))
        best_f1 = -1.0
        checkpoint_path = str(self.checkpoint_dir / checkpoint_name)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            progress = tqdm(self.train_loader, desc=f"warmup:{input_key}:epoch{epoch + 1}", leave=False)
            for batch in progress:
                outputs = self.model(**self._move_inputs(batch[input_key]))
                labels = batch["labels"].to(self.device)
                loss = cross_entropy_loss(outputs["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / max(1, len(self.train_loader))
            val_metrics = self._evaluate(input_key)
            csv_logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
            )

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                save_torch_checkpoint(
                    checkpoint_path,
                    {
                        "model_state": self.model.state_dict(),
                        "metrics": val_metrics,
                        "epoch": epoch + 1,
                        "config": dict(self.config),
                    },
                )

        return checkpoint_path

    def train_shortcut(self) -> str:
        return self._train("shortcut_inputs", "shortcut_best.pt", "warmup_shortcut.csv")

    def train_grounded(self) -> str:
        return self._train("grounded_inputs", "grounded_best.pt", "warmup_grounded.csv")
