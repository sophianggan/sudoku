"""Tests for model components (Phase 2).

Verifies output shapes, normalisation, and basic forward-pass correctness.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import pytest
from torch_geometric.data import Batch

from qwm.config import Config
from qwm.data.sudoku_generator import generate_puzzle, solve
from qwm.data.sudoku_graph import board_to_homogeneous_graph
from qwm.models.state_encoder import StateEncoder, TargetEncoder
from qwm.models.quotient_encoder import QuotientEncoder
from qwm.models.world_model import LatentWorldModel
from qwm.models.obstruction_predictor import ObstructionPredictor


@pytest.fixture
def config() -> Config:
    """Default config for tests."""
    return Config()


@pytest.fixture
def batch_graph() -> Batch:
    """Batch of 4 random Sudoku boards as a single PyG Batch."""
    rng = random.Random(42)
    graphs = []
    for _ in range(4):
        puzzle, _ = generate_puzzle(rng, n_clues=30)
        graphs.append(board_to_homogeneous_graph(puzzle))
    return Batch.from_data_list(graphs)


class TestStateEncoder:
    """Tests for the GAT-based state encoder."""

    def test_output_shape(self, config: Config, batch_graph: Batch) -> None:
        """Output should be (batch_size, state_embed_dim=256)."""
        enc = StateEncoder(config)
        h = enc(batch_graph)
        assert h.shape == (4, config.state_embed_dim)

    def test_output_differentiable(self, config: Config, batch_graph: Batch) -> None:
        """Output should have a grad_fn (connected to the computation graph)."""
        enc = StateEncoder(config)
        h = enc(batch_graph)
        assert h.grad_fn is not None


class TestTargetEncoder:
    """Tests for the EMA target encoder."""

    def test_forward_matches_after_copy(self, config: Config, batch_graph: Batch) -> None:
        """After copy_from, target and online should produce identical outputs."""
        online = StateEncoder(config)
        target = TargetEncoder(config)
        target.copy_from(online)
        h_online = online(batch_graph)
        h_target = target(batch_graph)
        assert torch.allclose(h_online, h_target, atol=1e-5)

    def test_ema_update_changes_weights(self, config: Config, batch_graph: Batch) -> None:
        """EMA update should change target weights (unless momentum=1)."""
        online = StateEncoder(config)
        target = TargetEncoder(config)
        target.copy_from(online)
        # Perturb online
        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        h_before = target(batch_graph).clone()
        target.ema_update(online, momentum=0.5)
        h_after = target(batch_graph)
        assert not torch.allclose(h_before, h_after, atol=1e-6)


class TestQuotientEncoder:
    """Tests for the quotient projection head."""

    def test_output_shape(self, config: Config) -> None:
        """Output should be (B, quotient_dim)."""
        qe = QuotientEncoder(config)
        h = torch.randn(8, config.state_embed_dim)
        z = qe(h)
        assert z.shape == (8, config.quotient_dim)

    def test_unit_normalised(self, config: Config) -> None:
        """Output vectors should have unit L2 norm."""
        qe = QuotientEncoder(config)
        h = torch.randn(8, config.state_embed_dim)
        z = qe(h)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestWorldModel:
    """Tests for the JEPA latent dynamics model."""

    def test_output_shape(self, config: Config) -> None:
        """Predicted z should have shape (B, quotient_dim)."""
        wm = LatentWorldModel(config)
        z_t = torch.randn(8, config.quotient_dim)
        action = torch.randint(0, 81, (8, 2))
        z_hat = wm(z_t, action)
        assert z_hat.shape == (8, config.quotient_dim)

    def test_unit_normalised(self, config: Config) -> None:
        """Predicted z should be L2-normalised."""
        wm = LatentWorldModel(config)
        z_t = torch.randn(8, config.quotient_dim)
        action = torch.randint(0, 81, (8, 2))
        z_hat = wm(z_t, action)
        norms = z_hat.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_solvability_head(self, config: Config) -> None:
        """Solvability prediction should be in [0, 1]."""
        wm = LatentWorldModel(config)
        z = torch.randn(8, config.quotient_dim)
        s = wm.predict_solvability(z)
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()


class TestObstructionPredictor:
    """Tests for the obstruction classifier."""

    def test_output_shape(self, config: Config) -> None:
        """Logits should be (B, num_obstruction_classes)."""
        op = ObstructionPredictor(config)
        z = torch.randn(8, config.quotient_dim)
        logits = op(z)
        assert logits.shape == (8, config.num_obstruction_classes)

    def test_predict_with_confidence(self, config: Config) -> None:
        """predict_with_confidence should return class_ids and confidence in [0,1]."""
        op = ObstructionPredictor(config)
        z = torch.randn(8, config.quotient_dim)
        cls_ids, conf = op.predict_with_confidence(z)
        assert cls_ids.shape == (8,)
        assert conf.shape == (8,)
        assert (conf >= 0).all() and (conf <= 1).all()
        assert (cls_ids >= 0).all() and (cls_ids < config.num_obstruction_classes).all()
