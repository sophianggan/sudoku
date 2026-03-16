"""Obstruction predictor — classifies failure category from quotient embedding.

Predicts which of the 6 obstruction types (row_conflict, col_conflict,
box_conflict, naked_single_violation, hidden_single_violation, empty_domain)
is likely to occur if the search continues from this state.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwm.config import Config


class ObstructionPredictor(nn.Module):
    """Classifier: z_t → logits over obstruction classes."""

    def __init__(self, config: Config) -> None:
        """Three-layer MLP with GELU activation and dropout."""
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.quotient_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, config.num_obstruction_classes),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Return raw logits: (B, num_obstruction_classes)."""
        return self.net(z_t)

    def predict_with_confidence(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (class_ids, confidence_scores) using softmax."""
        logits = self.forward(z_t)
        probs = F.softmax(logits, dim=-1)
        confidence, class_ids = probs.max(dim=-1)
        return class_ids, confidence

    def __repr__(self) -> str:
        """Readable string."""
        return f"ObstructionPredictor(z_dim={self.config.quotient_dim}, n_cls={self.config.num_obstruction_classes})"
