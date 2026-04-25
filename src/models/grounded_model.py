from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def _pool_hidden(outputs: Any) -> torch.Tensor:
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


class GroundedModel(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        hidden = self.dropout(_pool_hidden(outputs))
        logits = self.classifier(hidden)
        probs = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "hidden": hidden,
        }
