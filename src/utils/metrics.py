from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    label_names: Dict[int, str] | None = None,
) -> Dict[str, float]:
    if not labels:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=sorted(set(labels) | set(predictions)),
        zero_division=0,
    )

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

    for label_id, label_precision, label_recall, label_f1, label_support in zip(
        sorted(set(labels) | set(predictions)),
        per_label_precision,
        per_label_recall,
        per_label_f1,
        per_label_support,
    ):
        label_name = label_names[label_id] if label_names else str(label_id)
        prefix = label_name.lower()
        metrics[f"{prefix}_precision"] = float(label_precision)
        metrics[f"{prefix}_recall"] = float(label_recall)
        metrics[f"{prefix}_f1"] = float(label_f1)
        metrics[f"{prefix}_support"] = float(label_support)

    return metrics


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    if len(labels) == 0:
        return 0.0

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    labels_array = np.asarray(labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
        if not np.any(mask):
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = (predictions[mask] == labels_array[mask]).mean()
        ece += abs(bin_confidence - bin_accuracy) * mask.mean()

    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, labels: Sequence[int]) -> float:
    if len(labels) == 0:
        return 0.0
    labels_array = np.asarray(labels)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels_array)), labels_array] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def group_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    groups: Sequence[str],
    label_names: Dict[int, str] | None = None,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[int]]] = {}
    for label, pred, group in zip(labels, predictions, groups):
        bucket = grouped.setdefault(group, {"labels": [], "predictions": []})
        bucket["labels"].append(label)
        bucket["predictions"].append(pred)

    return {
        group: classification_metrics(bucket["labels"], bucket["predictions"], label_names=label_names)
        for group, bucket in grouped.items()
    }
