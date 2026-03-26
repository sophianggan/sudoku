"""Tests for QWM loss functions (Phase 2).

Verifies that all losses are positive scalars and differentiable.
"""

import torch
import pytest

from qwm.config import Config
from qwm.training.losses import (
    dynamics_loss,
    quotient_consistency_loss,
    obstruction_loss,
    merge_value_loss,
    total_loss,
)


class TestLosses:
    """Test all four loss functions and their gradients."""

    def test_dynamics_loss(self):
        z_pred = torch.randn(8, 64, requires_grad=True)
        z_target = torch.randn(8, 64)
        loss = dynamics_loss(z_pred, z_target)
        assert loss.item() >= 0
        loss.backward()
        assert z_pred.grad is not None

    def test_quotient_consistency_loss(self):
        z_anchors = torch.randn(4, 64, requires_grad=True)
        z_positives = torch.randn(4, 64)
        z_negatives = torch.randn(4, 3, 64)
        loss = quotient_consistency_loss(z_anchors, z_positives, z_negatives, temperature=0.07)
        assert loss.item() >= 0
        loss.backward()
        assert z_anchors.grad is not None

    def test_obstruction_loss(self):
        logits = torch.randn(8, 6, requires_grad=True)
        targets = torch.randint(0, 6, (8,))
        loss = obstruction_loss(logits, targets)
        assert loss.item() >= 0
        loss.backward()
        assert logits.grad is not None

    def test_merge_value_loss(self):
        values_pred = torch.randn(8, requires_grad=True)
        values_target = torch.rand(8)
        loss = merge_value_loss(values_pred, values_target)
        assert loss.item() >= 0
        loss.backward()
        assert values_pred.grad is not None

    def test_total_loss(self):
        cfg = Config()
        L_dyn = torch.tensor(1.0, requires_grad=True)
        L_quot = torch.tensor(2.0, requires_grad=True)
        L_obs = torch.tensor(3.0, requires_grad=True)
        L_value = torch.tensor(4.0, requires_grad=True)
        loss = total_loss(L_dyn, L_quot, L_obs, L_value, cfg)
        assert loss.item() > 0
        loss.backward()
        assert L_dyn.grad is not None
        assert L_quot.grad is not None
        assert L_obs.grad is not None
        assert L_value.grad is not None
