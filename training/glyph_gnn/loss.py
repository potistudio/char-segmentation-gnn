"""Focal loss for the imbalanced edge classification problem.

Most edges connect points of the same character (label 1); the rare but
critical negatives are inter-character edges. Focal loss down-weights the
easy majority and focuses gradient on hard boundary edges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss on raw logits.

    ``alpha`` weights the positive class; set it below 0.5 when positives
    dominate (our case), above 0.5 when negatives dominate.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
