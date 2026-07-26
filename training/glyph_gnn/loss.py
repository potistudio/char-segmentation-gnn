"""Losses for the edge classification problem.

The model emits one logit per point edge, but inference does not consume those
directly: ``group_characters`` collapses every point edge between two contours
to their maximum and unions the pair when it clears the threshold. Training on
point edges alone leaves two gaps:

* **Scale.** 75-79% of edges sit inside a single contour, are always positive,
  and are perfectly predicted by the ``same_contour`` input feature. They
  dominate the loss without teaching anything.
* **Pooling.** A contour pair spans 24 point edges at the Latin median, and the
  max ignores 23 of them. One edge that spikes above the threshold merges two
  characters, and the point-edge loss barely notices -- it sees one mistake out
  of 24, not one destroyed character. Union-find then spreads the damage down
  the line.

So the loss is computed at both levels: focal loss on point edges (with the
intra-contour majority down-weighted) plus focal loss on pooled contour pairs,
where the pooling mirrors the max that ships.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import contour_pair_index, scatter_amax


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary focal loss on raw logits.

    ``alpha`` weights the positive class; set it below 0.5 when positives
    dominate, above 0.5 when negatives dominate. Note that the right value
    differs per level: point edges run ~95% positive while contour pairs run
    56% (Japanese) to 20% (Latin), so they need separate alphas.

    ``weights`` scales the per-element loss, which is how intra-contour edges
    are held back without being dropped from the graph.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if weights is not None:
        loss = loss * weights
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def soft_max_pool(
    logits: torch.Tensor, index: torch.Tensor, size: int, temperature: float
) -> torch.Tensor:
    """Differentiable stand-in for the max that inference takes over a pair.

    Pools with softmax weights rather than ``logsumexp``: ``logsumexp``
    overshoots the max by ``T * log(n)``, and ``n`` swings from 6 to 590 edges
    per pair, so that bias would vary per pair and the model would have to
    learn around it. This weighted mean has no such term -- it tends to the
    exact max as ``temperature`` tends to 0 and to the plain mean as it grows.

    At ``temperature=1`` the weight concentrates on the top edges exactly when
    they stand out, which is the case that matters: one spiking edge among two
    dozen quiet ones is what merges two characters.
    """
    scaled = logits / temperature
    # Subtract the per-pair max before exponentiating, or a confident pair
    # overflows.
    shifted = scaled - scatter_amax(scaled.detach(), index, size)[index]
    weight = shifted.exp()
    total = weight.new_zeros(size).scatter_add_(0, index, weight)
    weight = weight / (total[index] + 1e-9)
    return logits.new_zeros(size).scatter_add_(0, index, weight * logits)


def group_loss(group_logits: torch.Tensor, group_labels: torch.Tensor) -> torch.Tensor:
    """BCE over candidate merges, rebalanced to the sampler's actual ratio.

    Only multi-contour characters can be cut in two, so positives are scarcer
    than negatives -- about 3 in 10 for Japanese, fewer for Latin. Left alone
    that bias lands squarely on "do not merge", and a decoder that will not
    merge leaves every multi-stroke character in pieces. The weight is measured
    per batch rather than assumed, since the ratio follows the script.
    """
    positives = group_labels.sum()
    negatives = group_labels.numel() - positives
    weight = (negatives / positives.clamp(min=1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(group_logits, group_labels,
                                              pos_weight=weight)


def glyph_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    contour_id: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pair_weight: float = 0.5,
    pair_alpha: float = 0.5,
    intra_weight: float = 0.05,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Point-edge focal loss blended with the pooled contour-pair loss.

    ``pair_weight`` is the share given to the pair term; 0 reproduces the old
    point-only objective. ``intra_weight`` scales the intra-contour edges,
    which are trivially positive: at 0.05 they fall from 79% of the point loss
    to about 16%, while their outputs stay calibrated for the ``glyph-infer``
    eval report and the GUI overlay. Set it to 0 to drop them entirely.

    Returns the loss and its components for logging.
    """
    inter, inverse, keys, _ = contour_pair_index(contour_id, edge_index)

    weights = torch.where(inter, torch.ones_like(logits),
                          torch.full_like(logits, intra_weight))
    point = focal_loss(logits, targets, alpha=alpha, gamma=gamma, weights=weights)

    if pair_weight <= 0.0 or keys.numel() == 0:
        return point, {"point": float(point.detach()), "pair": 0.0}

    pooled = soft_max_pool(logits[inter], inverse, keys.numel(), temperature)
    # Every point edge between one contour pair carries the same label, so the
    # max is simply that shared label.
    pair_target = scatter_amax(targets[inter], inverse, keys.numel())
    pair = focal_loss(pooled, pair_target, alpha=pair_alpha, gamma=gamma)

    total = (1.0 - pair_weight) * point + pair_weight * pair
    return total, {"point": float(point.detach()), "pair": float(pair.detach())}
