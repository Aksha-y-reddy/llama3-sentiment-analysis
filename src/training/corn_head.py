"""CORN ordinal classification head for 3-class sentiment.

CORN converts K ordinal classes into K-1 conditional binary decisions. For
negative < neutral < positive:

    y=0 (negative): [0, 0]
    y=1 (neutral): [1, 0]
    y=2 (positive): [1, 1]

This module is intentionally small and does not require coral-pytorch at runtime.
It follows the CORN target formulation and can be swapped for coral-pytorch later
if desired.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CORNHead(nn.Module):
    """Rank-consistent ordinal head over pooled hidden states."""

    def __init__(self, hidden_size: int, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if num_classes < 3:
            raise ValueError("CORNHead expects at least 3 ordinal classes")
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled_hidden))


def labels_to_corn_targets(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Convert class labels to cumulative CORN binary targets."""

    labels = labels.long()
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


def corn_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits into class probabilities.

    For 3 classes:
        P(y=0) = 1 - P(y>0)
        P(y=1) = P(y>0) * (1 - P(y>1))
        P(y=2) = P(y>0) * P(y>1)
    """

    conditional = torch.sigmoid(logits)
    probs = []
    running = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
    for idx in range(logits.shape[1]):
        p_greater = conditional[:, idx]
        probs.append(running * (1.0 - p_greater))
        running = running * p_greater
    probs.append(running)
    return torch.stack(probs, dim=-1)


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    targets = labels_to_corn_targets(labels, num_classes=num_classes)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-padding token from decoder hidden states."""

    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, lengths]
