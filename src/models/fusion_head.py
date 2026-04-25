from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GatedFusionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, 1)
        self.num_labels = num_labels

    def forward(
        self,
        shortcut_hidden: torch.Tensor,
        grounded_hidden: torch.Tensor,
        shortcut_probs: torch.Tensor,
        grounded_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat([shortcut_hidden, grounded_hidden], dim=-1)))
        fused_probs = gate * grounded_probs + (1.0 - gate) * shortcut_probs
        fused_probs = fused_probs.clamp_min(1e-8)
        fused_logits = fused_probs.log()
        return {
            "logits": fused_logits,
            "probs": fused_probs,
            "gate": gate.squeeze(-1),
        }
