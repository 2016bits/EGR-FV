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


def weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.to(values.device).float()
    normalizer = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / normalizer


def evidence_contrastive_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    null_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    null_kl_gamma: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pos_ce = F.cross_entropy(pos_logits, labels, reduction="none")
    neg_ce = F.cross_entropy(neg_logits, labels, reduction="none")
    hinge = torch.relu(float(margin) + pos_ce - neg_ce)
    hinge_loss = weighted_mean(hinge, sample_weights)

    pos_probs = F.softmax(pos_logits.detach(), dim=-1).clamp_min(1e-8)
    null_log_probs = F.log_softmax(null_logits, dim=-1)
    null_kl = F.kl_div(null_log_probs, pos_probs, reduction="none").sum(dim=-1)
    null_loss = weighted_mean(null_kl, sample_weights)

    total = hinge_loss + float(null_kl_gamma) * null_loss
    return total, {
        "hinge": hinge_loss.detach(),
        "null_kl": null_loss.detach(),
    }


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
