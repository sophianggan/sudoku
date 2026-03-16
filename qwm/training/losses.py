"""Loss functions for QWM training.

Four individual losses + a weighted combination:
  1. dynamics_loss     — MSE between predicted and target quotient embeddings
  2. quotient_consistency_loss — InfoNCE contrastive loss for equivalence
  3. obstruction_loss  — cross-entropy for failure-type classification
  4. merge_value_loss  — MSE for predicted utility of merging
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qwm.config import Config


def dynamics_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and target next-state quotient embeddings."""
    return F.mse_loss(z_pred, z_target)


def quotient_consistency_loss(
    z_anchors: torch.Tensor,
    z_positives: torch.Tensor,
    z_negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE contrastive loss for quotient consistency.

    Parameters
    ----------
    z_anchors  : (B, D) anchor quotient embeddings
    z_positives: (B, D) positive (equivalent) quotient embeddings
    z_negatives: (B, N, D) negative (non-equivalent) quotient embeddings
    temperature: scalar temperature for scaling similarities
    """
    B, D = z_anchors.shape

    # Positive similarity: (B,)
    pos_sim = (z_anchors * z_positives).sum(dim=-1) / temperature

    # Negative similarities: (B, N)
    neg_sim = torch.bmm(
        z_negatives, z_anchors.unsqueeze(-1)
    ).squeeze(-1) / temperature  # (B, N)

    # Log-sum-exp denominator: log( exp(pos) + sum_j exp(neg_j) )
    # Concatenate pos and neg for stable logsumexp
    all_logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1+N)
    log_denom = torch.logsumexp(all_logits, dim=-1)  # (B,)

    # InfoNCE = -log( exp(pos) / sum(exp(all)) ) = -(pos - log_denom)
    loss = -(pos_sim - log_denom)
    return loss.mean()


def obstruction_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for obstruction classification."""
    return F.cross_entropy(logits, targets)


def merge_value_loss(values_pred: torch.Tensor, values_target: torch.Tensor) -> torch.Tensor:
    """MSE loss for merge-value predictions."""
    return F.mse_loss(values_pred, values_target)


def total_loss(
    L_dyn: torch.Tensor,
    L_quot: torch.Tensor,
    L_obs: torch.Tensor,
    L_value: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Weighted combination of all four losses."""
    return (
        config.lambda_dyn * L_dyn
        + config.lambda_quot * L_quot
        + config.lambda_obs * L_obs
        + config.lambda_value * L_value
    )
