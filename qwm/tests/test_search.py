"""Tests for QWM search system (Phase 3).

Covers: SudokuVerifier, QuotientDAG, QWMController, Evaluator.
"""
from __future__ import annotations
import numpy as np
import pytest
from qwm.config import Config
from qwm.search.verifier import SudokuVerifier
from qwm.search.dag import QuotientDAG, DAGNode
from qwm.search.controller import QWMController
from qwm.evaluation.metrics import Evaluator
from qwm.models.state_encoder import StateEncoder
from qwm.models.quotient_encoder import QuotientEncoder
from qwm.models.world_model import LatentWorldModel
from qwm.models.obstruction_predictor import ObstructionPredictor
import torch

class TestVerifier:
    def test_valid_action(self):
        board = np.zeros((9,9), dtype=np.int32)
        verifier = SudokuVerifier()
        assert verifier.is_valid_action(board, 0, 0, 5)
        board[0,0] = 5
        assert not verifier.is_valid_action(board, 0, 0, 5)
        board[0,1] = 5
        assert not verifier.is_valid_action(board, 0, 2, 5)

    def test_is_complete(self):
        verifier = SudokuVerifier()
        board = np.array([
            [1,2,3,4,5,6,7,8,9],
            [4,5,6,7,8,9,1,2,3],
            [7,8,9,1,2,3,4,5,6],
            [2,3,4,5,6,7,8,9,1],
            [5,6,7,8,9,1,2,3,4],
            [8,9,1,2,3,4,5,6,7],
            [3,4,5,6,7,8,9,1,2],
            [6,7,8,9,1,2,3,4,5],
            [9,1,2,3,4,5,6,7,8],
        ], dtype=np.int32)
        assert verifier.is_complete(board)
        board[0,0] = 0
        assert not verifier.is_complete(board)

    def test_get_error_type(self):
        verifier = SudokuVerifier()
        board = np.zeros((9,9), dtype=np.int32)
        board[0,0] = 5
        assert verifier.get_error_type(board, 0, 0, 5) == "cell_filled"
        board[0,1] = 5
        assert verifier.get_error_type(board, 0, 2, 5) == "row_conflict"
        board[1,0] = 5
        assert verifier.get_error_type(board, 2, 0, 5) == "col_conflict"
        board[1,1] = 5
        assert verifier.get_error_type(board, 2, 2, 5) == "box_conflict"

    def test_apply_action(self):
        verifier = SudokuVerifier()
        board = np.zeros((9,9), dtype=np.int32)
        new_board = verifier.apply_action(board, 0, 0, 5)
        assert new_board[0,0] == 5
        with pytest.raises(ValueError):
            verifier.apply_action(new_board, 0, 0, 5)

class TestQuotientDAG:
    def test_add_and_merge(self):
        dag = QuotientDAG(merge_threshold=0.8)
        z1 = torch.randn(64)
        z2 = z1 + 0.01 * torch.randn(64)
        z2 = z2 / z2.norm()
        board = np.zeros((9,9), dtype=np.int32)
        id1 = dag.add_root(z1, board)
        id2, merged = dag.add_or_merge(z2, board, id1, 1.0, 0.0)
        assert merged
        assert id2 == id1
        z3 = torch.randn(64)
        id3, merged2 = dag.add_or_merge(z3, board, id1, 1.0, 0.0)
        assert not merged2
        assert id3 != id1

    def test_mark_pruned_and_solved(self):
        dag = QuotientDAG(merge_threshold=0.8)
        z = torch.randn(64)
        board = np.zeros((9,9), dtype=np.int32)
        id1 = dag.add_root(z, board)
        id2, _ = dag.add_or_merge(z, board, id1, 1.0, 0.0)
        dag.mark_pruned(id1)
        assert dag.nodes[id1].is_pruned
        dag.mark_solved(id2)
        assert dag.nodes[id2].is_verified_solved

class TestControllerAndEvaluator:
    def test_controller_runs_search(self):
        config = Config()
        verifier = SudokuVerifier()
        # Use minimal models (random weights)
        models = {
            "state_encoder": StateEncoder(config),
            "quotient_encoder": QuotientEncoder(config),
            "world_model": LatentWorldModel(config),
            "obstruction_predictor": ObstructionPredictor(config),
            "value_head": torch.nn.Linear(config.quotient_dim, 1),
        }
        controller = QWMController(models, config, verifier)
        board = np.zeros((9,9), dtype=np.int32)
        result = controller.search(board, max_nodes=10)
        assert isinstance(result, dict)
        assert "solved" in result

    def test_evaluator_metrics_format(self):
        config = Config()
        verifier = SudokuVerifier()
        models = {
            "state_encoder": StateEncoder(config),
            "quotient_encoder": QuotientEncoder(config),
            "world_model": LatentWorldModel(config),
            "obstruction_predictor": ObstructionPredictor(config),
            "value_head": torch.nn.Linear(config.quotient_dim, 1),
        }
        controller = QWMController(models, config, verifier)
        evaluator = Evaluator()
        boards = [np.zeros((9,9), dtype=np.int32) for _ in range(3)]
        result = evaluator.evaluate_batch(boards, controller)
        assert hasattr(result, "solve_rate")
        assert hasattr(result, "avg_nodes_expanded")
        assert hasattr(result, "avg_merges")
        assert hasattr(result, "avg_verifier_calls")
        assert hasattr(result, "avg_merge_rate")
        assert hasattr(result, "avg_obstruction_reuse")
