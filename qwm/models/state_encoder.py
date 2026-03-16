"""GNN-based state encoder for QWM.

Converts a homogeneous Sudoku PyG graph into a dense state embedding
using a 3-layer GAT architecture with global pooling.

Also provides a ``TargetEncoder`` (EMA copy) for JEPA-style training.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from qwm.config import Config


class StateEncoder(nn.Module):
    """GAT-based graph encoder: PyG Data → (batch_size, state_embed_dim)."""

    def __init__(self, config: Config) -> None:
        """Initialise three GAT layers, layer norms, and a final projection."""
        super().__init__()
        self.config = config
        in_dim = config.node_feat_dim
        h = config.hidden_dim
        heads = 4

        # Layer 1: in_dim → h*heads (512)
        self.gat1 = GATConv(in_dim, h, heads=heads, concat=True)
        self.ln1 = nn.LayerNorm(h * heads)

        # Layer 2: h*heads → h*heads (512)
        self.gat2 = GATConv(h * heads, h, heads=heads, concat=True)
        self.ln2 = nn.LayerNorm(h * heads)

        # Layer 3: h*heads → state_embed_dim (256)
        self.gat3 = GATConv(h * heads, config.state_embed_dim, heads=1, concat=False)
        self.ln3 = nn.LayerNorm(config.state_embed_dim)

        # Global pool concat (mean + max) → 2 * state_embed_dim → state_embed_dim
        self.proj = nn.Linear(2 * config.state_embed_dim, config.state_embed_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a (batched) graph into a graph-level embedding."""
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.gat1(x, edge_index)
        x = self.ln1(x)
        x = torch.relu(x)

        x = self.gat2(x, edge_index)
        x = self.ln2(x)
        x = torch.relu(x)

        x = self.gat3(x, edge_index)
        x = self.ln3(x)
        x = torch.relu(x)

        # Global pooling: mean ‖ max
        h_mean = global_mean_pool(x, batch)  # (B, state_embed_dim)
        h_max = global_max_pool(x, batch)    # (B, state_embed_dim)
        h = torch.cat([h_mean, h_max], dim=-1)  # (B, 2*state_embed_dim)
        h = self.proj(h)                         # (B, state_embed_dim)
        return h

    def __repr__(self) -> str:
        """Readable string."""
        return f"StateEncoder(in={self.config.node_feat_dim}, out={self.config.state_embed_dim})"


# ────────────────────────────────────────────────────────────────────
# Target (EMA) encoder
# ────────────────────────────────────────────────────────────────────

class TargetEncoder(nn.Module):
    """EMA copy of StateEncoder — weights updated only via exponential moving average."""

    def __init__(self, config: Config) -> None:
        """Create a TargetEncoder whose architecture mirrors StateEncoder."""
        super().__init__()
        self.encoder = StateEncoder(config)
        # Disable gradient computation for all parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass (always in no-grad mode)."""
        return self.encoder(data)

    @torch.no_grad()
    def ema_update(self, online_encoder: StateEncoder, momentum: float) -> None:
        """Update parameters: θ_target ← m * θ_target + (1 - m) * θ_online."""
        for p_target, p_online in zip(self.encoder.parameters(), online_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)

    def copy_from(self, online_encoder: StateEncoder) -> None:
        """Hard-copy weights from online encoder (used at initialization)."""
        self.encoder.load_state_dict(online_encoder.state_dict())

    def __repr__(self) -> str:
        """Readable string."""
        return f"TargetEncoder(ema, inner={self.encoder})"
