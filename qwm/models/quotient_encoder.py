"""Quotient encoder — projects state embeddings into obligation space.

Maps h_s (state_embed_dim) → z_s (quotient_dim), L2-normalised so that
cosine similarity directly measures equivalence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwm.config import Config


class QuotientEncoder(nn.Module):
    """MLP projection: h_s → z_s (unit-normalised quotient embedding)."""

    def __init__(self, config: Config) -> None:
        """Two-layer MLP with L2 normalisation on output."""
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.state_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.quotient_dim),
        )

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise: (B, state_embed_dim) → (B, quotient_dim)."""
        z = self.net(h_s)
        z = F.normalize(z, p=2, dim=-1)
        return z

    def __repr__(self) -> str:
        """Readable string."""
        return f"QuotientEncoder(in={self.config.state_embed_dim}, out={self.config.quotient_dim})"
