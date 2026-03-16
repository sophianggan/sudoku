"""Joint training loop for all QWM model components.

Orchestrates forward passes through the state encoder, quotient encoder,
world model, and obstruction predictor.  Computes the 4-part loss,
back-props, and applies EMA updates to the target encoder.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from qwm.config import Config
from qwm.models.state_encoder import StateEncoder, TargetEncoder
from qwm.models.quotient_encoder import QuotientEncoder
from qwm.models.world_model import LatentWorldModel
from qwm.models.obstruction_predictor import ObstructionPredictor
from qwm.training.losses import (
    dynamics_loss,
    merge_value_loss,
    obstruction_loss,
    quotient_consistency_loss,
    total_loss,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore[override]
        return it


class QWMTrainer:
    """End-to-end trainer for Quotient World Models."""

    def __init__(self, config: Config) -> None:
        """Initialise all model components, optimizer, and scheduler."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Models ──────────────────────────────────────────────────
        self.state_encoder = StateEncoder(config).to(self.device)
        self.target_encoder = TargetEncoder(config).to(self.device)
        self.target_encoder.copy_from(self.state_encoder)

        self.quotient_encoder = QuotientEncoder(config).to(self.device)
        self.world_model = LatentWorldModel(config).to(self.device)
        self.obstruction_predictor = ObstructionPredictor(config).to(self.device)

        # Merge-value head (simple linear from quotient space)
        self.value_head = nn.Linear(config.quotient_dim, 1).to(self.device)

        # ── Optimizer ───────────────────────────────────────────────
        params = (
            list(self.state_encoder.parameters())
            + list(self.quotient_encoder.parameters())
            + list(self.world_model.parameters())
            + list(self.obstruction_predictor.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = AdamW(params, lr=config.lr, weight_decay=1e-4)
        self.scheduler: Optional[CosineAnnealingLR] = None  # set in train()

    # ── Collation helper ────────────────────────────────────────────

    @staticmethod
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate that batches PyG Data objects via Batch.from_data_list."""
        graphs_t = Batch.from_data_list([item["graph_t"] for item in batch])
        graphs_t1 = Batch.from_data_list([item["graph_t1"] for item in batch])
        actions = torch.stack([item["action"] for item in batch])
        obs_labels = torch.tensor([item["obstruction_label"] for item in batch], dtype=torch.long)
        merge_vals = torch.tensor([item["merge_value"] for item in batch], dtype=torch.float32)
        is_failed = torch.tensor([item["is_failed"] for item in batch], dtype=torch.bool)

        out: Dict[str, Any] = {
            "graph_t": graphs_t,
            "graph_t1": graphs_t1,
            "action": actions,
            "obstruction_label": obs_labels,
            "merge_value": merge_vals,
            "is_failed": is_failed,
        }

        # Optional pair data
        if "graph_pos" in batch[0]:
            out["graph_pos"] = Batch.from_data_list([item["graph_pos"] for item in batch])
            # Negatives: list of lists → batch each negative position
            n_neg = len(batch[0]["graphs_neg"])
            neg_batches = []
            for k in range(n_neg):
                neg_batches.append(Batch.from_data_list([item["graphs_neg"][k] for item in batch]))
            out["graphs_neg"] = neg_batches

        return out

    # ── Single training step ────────────────────────────────────────

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one training step; return dict of named loss values."""
        self.state_encoder.train()
        self.quotient_encoder.train()
        self.world_model.train()
        self.obstruction_predictor.train()

        cfg = self.config
        dev = self.device

        graph_t = batch["graph_t"].to(dev)
        graph_t1 = batch["graph_t1"].to(dev)
        actions = batch["action"].to(dev)
        obs_labels = batch["obstruction_label"].to(dev)
        merge_vals = batch["merge_value"].to(dev)
        is_failed = batch["is_failed"].to(dev)

        # 1) Encode current and next states (online)
        h_t = self.state_encoder(graph_t)
        h_t1 = self.state_encoder(graph_t1)

        # 2) Quotient projection
        z_t = self.quotient_encoder(h_t)
        z_t1 = self.quotient_encoder(h_t1)

        # 3) Target encoder (EMA, no grad)
        with torch.no_grad():
            z_t1_target = self.quotient_encoder(self.target_encoder(graph_t1))

        # 4) World model prediction
        z_hat_t1 = self.world_model(z_t, actions)

        # 5) Obstruction prediction
        obs_logits = self.obstruction_predictor(z_t)

        # 6) Value prediction
        value_pred = self.value_head(z_t).squeeze(-1)

        # ── Compute losses ──────────────────────────────────────────
        # Dynamics loss
        L_dyn = dynamics_loss(z_hat_t1, z_t1_target)

        # Quotient consistency loss (only if pair data available)
        if "graph_pos" in batch:
            graph_pos = batch["graph_pos"].to(dev)
            h_pos = self.state_encoder(graph_pos)
            z_pos = self.quotient_encoder(h_pos)

            neg_batches = batch["graphs_neg"]
            z_negs_list = []
            for neg_batch in neg_batches:
                neg_batch = neg_batch.to(dev)
                h_neg = self.state_encoder(neg_batch)
                z_neg = self.quotient_encoder(h_neg)
                z_negs_list.append(z_neg)
            z_negatives = torch.stack(z_negs_list, dim=1)  # (B, n_neg, D)

            L_quot = quotient_consistency_loss(z_t, z_pos, z_negatives, cfg.temperature)
        else:
            L_quot = torch.tensor(0.0, device=dev, requires_grad=True)

        # Obstruction loss (only on failed transitions with valid labels)
        failed_mask = (obs_labels >= 0)
        if failed_mask.any():
            L_obs = obstruction_loss(obs_logits[failed_mask], obs_labels[failed_mask])
        else:
            L_obs = torch.tensor(0.0, device=dev, requires_grad=True)

        # Merge value loss
        L_value = merge_value_loss(value_pred, merge_vals)

        # Combined
        L_total = total_loss(L_dyn, L_quot, L_obs, L_value, cfg)

        # 7) Backprop
        self.optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 1.0)
        self.optimizer.step()

        # 8) EMA update
        self.target_encoder.ema_update(self.state_encoder, cfg.ema_momentum)

        return {
            "loss_total": L_total.item(),
            "loss_dyn": L_dyn.item(),
            "loss_quot": L_quot.item(),
            "loss_obs": L_obs.item(),
            "loss_value": L_value.item(),
        }

    # ── Full training loop ──────────────────────────────────────────

    def train(self, n_epochs: int, dataloader: DataLoader) -> List[Dict[str, float]]:
        """Train for *n_epochs*, returning per-epoch average losses."""
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        history: List[Dict[str, float]] = []

        for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs"):
            epoch_losses: Dict[str, float] = {}
            n_batches = 0

            for batch in dataloader:
                losses = self.train_step(batch)
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_batches += 1

            # Average
            avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            history.append(avg_losses)

            self.scheduler.step()

            # Checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt_dir = pathlib.Path("checkpoints")
                ckpt_dir.mkdir(exist_ok=True)
                self.save_checkpoint(ckpt_dir / f"qwm_epoch_{epoch}.pt")

        return history

    # ── Checkpoint I/O ──────────────────────────────────────────────

    def save_checkpoint(self, path: pathlib.Path) -> None:
        """Save all model weights and optimizer state."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "target_encoder": self.target_encoder.encoder.state_dict(),
            "quotient_encoder": self.quotient_encoder.state_dict(),
            "world_model": self.world_model.state_dict(),
            "obstruction_predictor": self.obstruction_predictor.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, str(path))

    def load_checkpoint(self, path: pathlib.Path) -> None:
        """Load all model weights and optimizer state."""
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.target_encoder.encoder.load_state_dict(ckpt["target_encoder"])
        self.quotient_encoder.load_state_dict(ckpt["quotient_encoder"])
        self.world_model.load_state_dict(ckpt["world_model"])
        self.obstruction_predictor.load_state_dict(ckpt["obstruction_predictor"])
        self.value_head.load_state_dict(ckpt["value_head"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

    def get_models_dict(self) -> Dict[str, nn.Module]:
        """Return a dict of all inference-time models (for the search controller)."""
        return {
            "state_encoder": self.state_encoder,
            "quotient_encoder": self.quotient_encoder,
            "world_model": self.world_model,
            "obstruction_predictor": self.obstruction_predictor,
            "value_head": self.value_head,
        }

    def __repr__(self) -> str:
        """Readable string."""
        return f"QWMTrainer(device={self.device}, config={self.config})"
