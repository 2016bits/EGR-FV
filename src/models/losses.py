from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(losses.device).float()
    normalizer = weights.sum().clamp_min(1e-8)
    return (losses * weights).sum() / normalizer


def orthogonality_loss(
    shortcut_hidden: torch.Tensor,
    grounded_hidden: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    if mode == "dot":
        dot_products = torch.sum(shortcut_hidden * grounded_hidden, dim=-1)
        return torch.mean(dot_products.pow(2))

    shortcut_norm = F.normalize(shortcut_hidden, dim=-1)
    grounded_norm = F.normalize(grounded_hidden, dim=-1)
    cosine = torch.sum(shortcut_norm * grounded_norm, dim=-1)
    return torch.mean(cosine.pow(2))


def group_to_sample_weights(
    groups: Sequence[str],
    mapping: Dict[str, float],
    default_weight: float = 1.0,
) -> torch.Tensor:
    return torch.tensor([mapping.get(group, default_weight) for group in groups], dtype=torch.float)
