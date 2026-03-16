"""Global configuration for Quotient World Models (QWM)."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """All hyperparameters for the QWM framework."""

    # ── Feature dimensions ──────────────────────────────────────────────
    node_feat_dim: int = 18           # per-node feature size (padded)
    hidden_dim: int = 128
    state_embed_dim: int = 256        # output of state encoder
    quotient_dim: int = 64            # output of quotient encoder
    action_embed_dim: int = 32
    world_model_hidden: int = 256

    # ── Task-specific ───────────────────────────────────────────────────
    num_obstruction_classes: int = 6

    # ── Search thresholds ───────────────────────────────────────────────
    merge_threshold: float = 0.93     # cosine similarity for DAG merge
    temperature: float = 0.07         # contrastive loss temperature

    # ── Loss weights ────────────────────────────────────────────────────
    lambda_dyn: float = 1.0
    lambda_quot: float = 0.5
    lambda_obs: float = 0.3
    lambda_value: float = 0.1

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 64
    lr: float = 3e-4
    ema_momentum: float = 0.996

    # ── Search controller ───────────────────────────────────────────────
    max_search_nodes: int = 500
    verify_every_n_steps: int = 20

    # ── Reproducibility ─────────────────────────────────────────────────
    seed: int = 42

    def __repr__(self) -> str:
        """Return a readable string of all config fields."""
        fields = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({fields})"
