from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import FactVerificationDataset
from src.data.routing import compute_kl_disagreement
from src.utils.io import write_json, write_jsonl
from src.utils.metrics import (
    classification_metrics,
    expected_calibration_error,
    group_metrics,
    multiclass_brier_score,
)


class Evaluator:
    def __init__(
        self,
        shortcut_model,
        grounded_model,
        fusion_head,
        joint_collator,
        shortcut_collator,
        grounded_collator,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        id_to_label: Mapping[int, str],
        evidence_placeholder: str = "[NO_EVIDENCE]",
    ) -> None:
        self.shortcut_model = shortcut_model
        self.grounded_model = grounded_model
        self.fusion_head = fusion_head
        self.joint_collator = joint_collator
        self.shortcut_collator = shortcut_collator
        self.grounded_collator = grounded_collator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.id_to_label = dict(id_to_label)
        self.evidence_placeholder = evidence_placeholder

    def _build_loader(self, dataset: FactVerificationDataset, mode: str) -> DataLoader:
        if mode == "shortcut":
            collator = self.shortcut_collator
        elif mode == "grounded_only":
            collator = self.grounded_collator
        else:
            collator = self.joint_collator
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
        )

    def _move_inputs(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in inputs.items()}

    def evaluate_dataset(
        self,
        dataset: FactVerificationDataset,
        mode: str = "grounded",
        predictions_path: str | None = None,
    ) -> Dict[str, Any]:
        if mode not in {"shortcut", "grounded", "fusion"}:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
        if mode == "shortcut" and self.shortcut_model is None:
            raise ValueError("Shortcut evaluation requested, but shortcut_model is not loaded.")

        loader_mode = "shortcut" if mode == "shortcut" else "joint"
        dataloader = self._build_loader(dataset, mode=loader_mode)

        if self.shortcut_model is not None:
            self.shortcut_model.eval()
        if self.grounded_model is not None:
            self.grounded_model.eval()
        if self.fusion_head is not None:
            self.fusion_head.eval()

        labels: List[int] = []
        predictions: List[int] = []
        probabilities: List[np.ndarray] = []
        groups: List[str] = []
        hop_groups: List[str] = []
        prediction_records: List[Dict[str, Any]] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"eval:{mode}", leave=False):
                shortcut_outputs = None
                grounded_outputs = None
                fused_outputs = None

                if mode == "shortcut":
                    shortcut_outputs = self.shortcut_model(**self._move_inputs(batch["shortcut_inputs"]))
                    selected_probs = shortcut_outputs["probs"]
                else:
                    grounded_outputs = self.grounded_model(**self._move_inputs(batch["grounded_inputs"]))
                    selected_probs = grounded_outputs["probs"]
                    if self.shortcut_model is not None:
                        shortcut_outputs = self.shortcut_model(**self._move_inputs(batch["shortcut_inputs"]))
                    if mode == "fusion":
                        if self.fusion_head is None or shortcut_outputs is None:
                            raise ValueError("Fusion mode requires both shortcut_model and fusion_head.")
                        fused_outputs = self.fusion_head(
                            shortcut_hidden=shortcut_outputs["hidden"],
                            grounded_hidden=grounded_outputs["hidden"],
                            shortcut_probs=shortcut_outputs["probs"],
                            grounded_probs=grounded_outputs["probs"],
                        )
                        selected_probs = fused_outputs["probs"]

                selected_probs_cpu = selected_probs.detach().cpu()
                predicted_ids = selected_probs_cpu.argmax(dim=-1)
                batch_labels = batch["labels"].detach().cpu().tolist()

                if shortcut_outputs is not None:
                    shortcut_probs_cpu = shortcut_outputs["probs"].detach().cpu()
                else:
                    shortcut_probs_cpu = None

                if grounded_outputs is not None:
                    grounded_probs_cpu = grounded_outputs["probs"].detach().cpu()
                else:
                    grounded_probs_cpu = None

                disagreements_cpu = None
                if shortcut_probs_cpu is not None and grounded_probs_cpu is not None:
                    disagreements_cpu = compute_kl_disagreement(grounded_probs_cpu, shortcut_probs_cpu)

                for index, sample_id in enumerate(batch["ids"]):
                    label_id = int(batch_labels[index])
                    pred_id = int(predicted_ids[index].item())
                    if label_id >= 0:
                        labels.append(label_id)
                        predictions.append(pred_id)
                        probabilities.append(selected_probs_cpu[index].numpy())
                        groups.append(batch["groups"][index])
                        hop_value = batch["num_hops"][index]
                        hop_groups.append(f"hop_{hop_value}" if hop_value is not None else "hop_unknown")

                    shortcut_conf = (
                        float(shortcut_probs_cpu[index].max().item()) if shortcut_probs_cpu is not None else None
                    )
                    grounded_conf = (
                        float(grounded_probs_cpu[index].max().item()) if grounded_probs_cpu is not None else None
                    )
                    record = {
                        "id": sample_id,
                        "group": batch["groups"][index],
                        "num_hops": batch["num_hops"][index],
                        "label": self.id_to_label.get(label_id) if label_id >= 0 else None,
                        "pred_label": self.id_to_label.get(pred_id, str(pred_id)),
                        "confidence": float(selected_probs_cpu[index].max().item()),
                        "shortcut_conf": shortcut_conf,
                        "grounded_conf": grounded_conf,
                    }
                    if disagreements_cpu is not None:
                        record["disagreement"] = float(disagreements_cpu[index].item())
                    prediction_records.append(record)

        base_metrics = classification_metrics(labels, predictions, label_names=self.id_to_label)
        probability_array = np.stack(probabilities) if probabilities else np.zeros((0, len(self.id_to_label)))
        result = {
            "mode": mode,
            "num_samples": len(labels),
            "metrics": base_metrics,
            "group_metrics": group_metrics(labels, predictions, groups, label_names=self.id_to_label),
            "hop_metrics": group_metrics(labels, predictions, hop_groups, label_names=self.id_to_label),
            "calibration": {
                "ece": expected_calibration_error(probability_array, labels) if len(labels) else 0.0,
                "brier": multiclass_brier_score(probability_array, labels) if len(labels) else 0.0,
            },
            "predictions": prediction_records,
        }

        if predictions_path:
            write_jsonl(predictions_path, prediction_records)
        return result

    def evaluate_sensitivity(self, dataset: FactVerificationDataset) -> Dict[str, Any]:
        base_report = self.evaluate_dataset(dataset, mode="grounded")

        no_evidence_records = []
        for record in dataset.records:
            updated = deepcopy(record)
            updated["evidence_list"] = [self.evidence_placeholder]
            updated["evidence_text"] = self.evidence_placeholder
            no_evidence_records.append(updated)
        no_evidence_dataset = dataset.clone_with_records(no_evidence_records)
        remove_report = self.evaluate_dataset(no_evidence_dataset, mode="grounded")

        shuffled_evidence = [record["evidence_text"] for record in dataset.records]
        if shuffled_evidence:
            rng = np.random.default_rng(42)
            shuffled_evidence = list(rng.permutation(shuffled_evidence))

        shuffled_records = []
        for record, evidence_text in zip(dataset.records, shuffled_evidence):
            updated = deepcopy(record)
            updated["evidence_list"] = [str(evidence_text)]
            updated["evidence_text"] = str(evidence_text)
            shuffled_records.append(updated)
        shuffled_dataset = dataset.clone_with_records(shuffled_records)
        shuffle_report = self.evaluate_dataset(shuffled_dataset, mode="grounded")

        sensitivity = {
            "grounded": base_report["metrics"],
            "remove_evidence": remove_report["metrics"],
            "shuffle_evidence": shuffle_report["metrics"],
        }
        if self.shortcut_model is not None:
            shortcut_report = self.evaluate_dataset(dataset, mode="shortcut")
            sensitivity["claim_only"] = shortcut_report["metrics"]
        return sensitivity

    def run_full_evaluation(
        self,
        dataset: FactVerificationDataset,
        report_path: str,
        predictions_path: str,
        mode: str = "grounded",
    ) -> Dict[str, Any]:
        base_report = self.evaluate_dataset(dataset, mode=mode, predictions_path=predictions_path)
        sensitivity_report = self.evaluate_sensitivity(dataset)
        full_report = {
            "base": {
                "mode": base_report["mode"],
                "num_samples": base_report["num_samples"],
                "metrics": base_report["metrics"],
                "group_metrics": base_report["group_metrics"],
                "hop_metrics": base_report["hop_metrics"],
                "calibration": base_report["calibration"],
            },
            "evidence_sensitivity": sensitivity_report,
        }
        write_json(report_path, full_report)
        return full_report
