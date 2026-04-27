"""Differential LAFT losses for clean, hard-clean, and noisy samples."""

from __future__ import annotations

import torch
from torch.nn import functional as F

try:
    from laft_partition import partition_laft_batch
except ImportError:  # pragma: no cover
    from src.training.laft_partition import partition_laft_batch


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1)


def laft_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    judge_probs: torch.Tensor,
    confidence_threshold: float = 0.70,
    hard_clean_weight: float = 0.50,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Compute LAFT differential loss over class logits."""

    masks = partition_laft_batch(labels, judge_probs, confidence_threshold)
    losses = []
    sample_count = 0

    if masks.easy_clean.any():
        easy_loss = F.cross_entropy(
            logits[masks.easy_clean],
            labels[masks.easy_clean],
            reduction="sum",
            label_smoothing=label_smoothing,
        )
        losses.append(easy_loss)
        sample_count += int(masks.easy_clean.sum().item())

    if masks.hard_clean.any():
        hard_loss = F.cross_entropy(
            logits[masks.hard_clean],
            labels[masks.hard_clean],
            reduction="sum",
            label_smoothing=label_smoothing,
        )
        losses.append(hard_clean_weight * hard_loss)
        sample_count += int(masks.hard_clean.sum().item())

    if masks.true_noisy.any():
        noisy_loss = soft_cross_entropy(
            logits[masks.true_noisy],
            judge_probs[masks.true_noisy],
        ).sum()
        losses.append(noisy_loss)
        sample_count += int(masks.true_noisy.sum().item())

    if not losses:
        # Keep graph connected if an empty batch slips through.
        loss = logits.sum() * 0.0
    else:
        loss = sum(losses) / max(sample_count, 1)

    stats = {
        **masks.counts(),
        "sample_count": sample_count,
        "confidence_threshold": float(confidence_threshold),
        "hard_clean_weight": float(hard_clean_weight),
    }
    return loss, stats


def laft_compute_loss_from_outputs(
    outputs,
    labels: torch.Tensor,
    judge_probs: torch.Tensor,
    confidence_threshold: float = 0.70,
    hard_clean_weight: float = 0.50,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Adapter for trainers whose model output exposes ``logits``."""

    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    return laft_classification_loss(
        logits=logits,
        labels=labels,
        judge_probs=judge_probs,
        confidence_threshold=confidence_threshold,
        hard_clean_weight=hard_clean_weight,
        label_smoothing=label_smoothing,
    )
