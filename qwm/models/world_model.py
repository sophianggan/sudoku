"""JEPA-style latent world model for QWM.

Predicts the *next* quotient embedding z_{t+1} from (z_t, a_t) entirely
in latent space, without generating raw board states.

Also includes an auxiliary solvability head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwm.config import Config


class LatentWorldModel(nn.Module):
    """Latent dynamics: (z_t, a_t) → z_hat_{t+1} (unit-normalised)."""

    def __init__(self, config: Config) -> None:
        """Build action embeddings, predictor MLP, and solvability head."""
        super().__init__()
        self.config = config

        # ── Action embedding ────────────────────────────────────────
        self.cell_embed = nn.Embedding(81, 16)
        self.digit_embed = nn.Embedding(10, 16)   # 0 = no action / padding
        self.action_proj = nn.Linear(32, config.action_embed_dim)

        # ── Predictor ───────────────────────────────────────────────
        in_dim = config.quotient_dim + config.action_embed_dim
        h = config.world_model_hidden
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, config.quotient_dim),
        )

        # ── Solvability head ────────────────────────────────────────
        self.solvability_head = nn.Sequential(
            nn.Linear(config.quotient_dim, 1),
            nn.Sigmoid(),
        )

    def embed_action(self, action: torch.Tensor) -> torch.Tensor:
        """Embed a (B, 2) action tensor [cell_idx, digit] → (B, action_embed_dim)."""
        cell_idx = action[:, 0].clamp(0, 80).long()
        digit = action[:, 1].clamp(0, 9).long()
        ce = self.cell_embed(cell_idx)   # (B, 16)
        de = self.digit_embed(digit)     # (B, 16)
        a = torch.cat([ce, de], dim=-1)  # (B, 32)
        return self.action_proj(a)       # (B, action_embed_dim)

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict z_{t+1} from z_t and action. Output is L2-normalised."""
        a_emb = self.embed_action(action)
        inp = torch.cat([z_t, a_emb], dim=-1)     # (B, quotient_dim + action_embed_dim)
        z_hat = self.predictor(inp)                 # (B, quotient_dim)
        z_hat = F.normalize(z_hat, p=2, dim=-1)
        return z_hat

    def predict_solvability(self, z: torch.Tensor) -> torch.Tensor:
        """Predict P(branch eventually solvable) ∈ [0, 1]."""
        return self.solvability_head(z).squeeze(-1)  # (B,)

    def __repr__(self) -> str:
        """Readable string."""
        return (f"LatentWorldModel(z_dim={self.config.quotient_dim}, "
                f"a_dim={self.config.action_embed_dim})")
