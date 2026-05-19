"""
losses.py — Loss functions for ordinal DR classification.

Fix 2 applied here: rebalanced ordinal threshold_weights.

DIAGNOSIS FROM CONFUSION MATRIX:
  The most expensive single error type is class 4 → class 2 (15 samples,
  ordinal distance 2, QWK penalty 4×). This happens because the current
  threshold weights [1.0, 2.0, 2.0, 1.5] make the model focus on the
  mild/moderate and moderate/severe boundaries but underpenalise the
  severe/proliferative boundary (threshold index 3).

  Meanwhile, mean squared ordinal distance of all errors = 2.1, which is
  high — many errors are skipping grades, not just swapping neighbours.
  Increasing the weight on threshold 3 (P(Y>3), severe vs proliferative)
  forces the model to separate these two classes more aggressively.

  New weights: [1.0, 2.0, 2.5, 2.5]
    Index 0: P(Y>0), no-DR vs any-DR       — 1.0 (easy boundary, keep low)
    Index 1: P(Y>1), mild vs moderate      — 2.0 (unchanged, class 1 confusion)
    Index 2: P(Y>2), moderate vs severe    — 2.5 (increased; class 3 confusion)
    Index 3: P(Y>3), severe vs prolif      — 2.5 (increased; class 4→2 is worst error)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2.5 stays — strong focus on hard/rare examples.
    smoothing=0.0 stays — smoothing hurts minority recall.
    """

    def __init__(
        self,
        gamma: float = 2.5,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None

        ce_loss    = F.cross_entropy(logits, targets, reduction='none',
                                     label_smoothing=self.smoothing)
        pt         = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class OrdinalCrossEntropyLoss(nn.Module):
    """
    K-class ordinal problem as K-1 binary threshold problems.

    For true grade k:
        P(Y > j) = 1  for j < k
        P(Y > j) = 0  for j >= k

    FIX 2: threshold_weights updated to [1.0, 2.0, 2.5, 2.5]
      The previous [1.0, 2.0, 2.0, 1.5] underweighted the severe/proliferative
      boundary (index 3). Analysis shows 15 class-4 samples are predicted as
      class 2 (distance-2 error, 4× QWK penalty). Raising index 3 from 1.5 → 2.5
      makes the model pay more attention to separating grade 3 from grade 4.
      Index 2 also raised to 2.5 (was 2.0) because 26 class-2 samples are
      predicted as class 3, and 6 class-3 samples go to class 2.
    """

    def __init__(
        self,
        num_classes: int = 5,
        threshold_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        if threshold_weights is None:
            # [P(Y>0), P(Y>1), P(Y>2), P(Y>3)]
            threshold_weights = torch.tensor([1.0, 2.0, 2.5, 2.5])
        self.register_buffer('threshold_weights', threshold_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        K          = self.num_classes

        probs        = F.softmax(logits, dim=1)
        right_cumsum = torch.cumsum(probs.flip(dims=[1]), dim=1).flip(dims=[1])
        cum_probs    = right_cumsum[:, 1:]   # (batch, K-1), drops P(Y>=0)=1

        ordinal_targets = torch.zeros(batch_size, K - 1, device=logits.device)
        for j in range(K - 1):
            ordinal_targets[:, j] = (targets > j).float()

        cum_probs = cum_probs.clamp(1e-7, 1.0 - 1e-7)
        bce = -(
            ordinal_targets       * torch.log(cum_probs) +
            (1.0 - ordinal_targets) * torch.log(1.0 - cum_probs)
        )

        tw  = self.threshold_weights.to(logits.device)
        bce = bce * tw.unsqueeze(0)

        return bce.mean()


class CombinedLoss(nn.Module):
    """
    CombinedLoss = focal_weight × FocalLoss + ordinal_weight × OrdinalLoss

    focal_weight=0.5, ordinal_weight=0.5 (FIX 3: was 0.6/0.4).
    Giving the ordinal loss equal weight forces the model to respect
    grade ordering more strongly. Particularly helps class 4→2 errors
    because the ordinal signal directly penalises distance-2 mistakes.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        ordinal_weight: float = 0.5,
        gamma: float = 2.5,
        alpha: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        num_classes: int = 5,
        threshold_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.focal_weight   = focal_weight
        self.ordinal_weight = ordinal_weight
        self.focal   = FocalLoss(gamma=gamma, alpha=alpha, smoothing=smoothing)
        self.ordinal = OrdinalCrossEntropyLoss(
            num_classes=num_classes,
            threshold_weights=threshold_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight   * self.focal(logits, targets) +
            self.ordinal_weight * self.ordinal(logits, targets)
        )