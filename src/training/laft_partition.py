"""LAFT-style partitioning utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LAFTMasks:
    easy_clean: torch.Tensor
    hard_clean: torch.Tensor
    true_noisy: torch.Tensor

    def counts(self) -> dict[str, int]:
        return {
            "easy_clean": int(self.easy_clean.sum().item()),
            "hard_clean": int(self.hard_clean.sum().item()),
            "true_noisy": int(self.true_noisy.sum().item()),
        }


def partition_laft_batch(
    labels: torch.Tensor,
    judge_probs: torch.Tensor,
    confidence_threshold: float = 0.70,
) -> LAFTMasks:
    """Partition a batch into Easy Clean, Hard Clean, and True Noisy.

    Args:
        labels: Hard class labels, shape ``(batch,)``.
        judge_probs: Judge probability distribution, shape ``(batch, 3)``.
        confidence_threshold: Threshold for confident judge disagreement.
    """

    labels = labels.long()
    judge_probs = judge_probs.float()
    judge_conf, judge_label = judge_probs.max(dim=-1)
    agrees = judge_label == labels

    easy_clean = agrees
    true_noisy = (~agrees) & (judge_conf >= confidence_threshold)
    hard_clean = (~agrees) & (judge_conf < confidence_threshold)
    return LAFTMasks(
        easy_clean=easy_clean,
        hard_clean=hard_clean,
        true_noisy=true_noisy,
    )
