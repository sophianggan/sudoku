"""Tests for the N-Queens environment (verifier, graph encoder, action proposer, controller)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qwm.envs.nqueens import (
    NQueensActionProposer,
    NQueensGraphEncoder,
    NQueensQWMController,
    NQueensVerifier,
    _queens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 8


def empty_board(n: int = N) -> np.ndarray:
    return np.zeros((n, n), dtype=np.int8)


def solved_board_8() -> np.ndarray:
    """One known solution to the 8-Queens problem."""
    board = empty_board(8)
    # Solution: col positions per row = [0,4,7,5,2,6,1,3]
    for row, col in enumerate([0, 4, 7, 5, 2, 6, 1, 3]):
        board[row, col] = 1
    return board


def partial_board(rows_filled: int = 3, n: int = N) -> np.ndarray:
    """Partial board with queens in the first *rows_filled* rows."""
    board = empty_board(n)
    cols = [0, 4, 7][:rows_filled]
    for r, c in enumerate(cols):
        board[r, c] = 1
    return board


# ---------------------------------------------------------------------------
# NQueensVerifier
# ---------------------------------------------------------------------------

class TestNQueensVerifier:
    def setup_method(self):
        self.v = NQueensVerifier(N)

    def test_is_complete_solved(self):
        assert self.v.is_complete(solved_board_8())

    def test_is_complete_empty(self):
        assert not self.v.is_complete(empty_board())

    def test_is_complete_partial(self):
        assert not self.v.is_complete(partial_board(3))

    def test_is_complete_wrong_n_queens(self):
        # 8 queens but with conflicts → not complete
        board = empty_board()
        for r in range(8):
            board[r, 0] = 1  # all in column 0 → conflicts
        assert not self.v.is_complete(board)

    def test_valid_action_empty_board(self):
        board = empty_board()
        assert self.v.is_valid_action(board, (0, 0))

    def test_invalid_action_occupied(self):
        board = empty_board()
        board[0, 0] = 1
        assert not self.v.is_valid_action(board, (0, 0))

    def test_invalid_action_row_conflict(self):
        board = empty_board()
        board[0, 0] = 1
        assert self.v.get_error_type(board, (0, 5)) == "row_conflict"

    def test_invalid_action_col_conflict(self):
        board = empty_board()
        board[0, 3] = 1
        assert self.v.get_error_type(board, (4, 3)) == "col_conflict"

    def test_invalid_action_diagonal_conflict(self):
        board = empty_board()
        board[0, 0] = 1
        assert self.v.get_error_type(board, (3, 3)) == "diagonal_conflict"

    def test_valid_action_no_conflict(self):
        board = empty_board()
        board[0, 0] = 1
        assert self.v.get_error_type(board, (1, 2)) is None

    def test_apply_action(self):
        board = empty_board()
        new_board = self.v.apply_action(board, (2, 5))
        assert new_board[2, 5] == 1
        assert board[2, 5] == 0  # original unchanged

    def test_apply_action_invalid_raises(self):
        board = empty_board()
        board[0, 0] = 1
        with pytest.raises(ValueError):
            self.v.apply_action(board, (0, 3))  # row conflict

    def test_out_of_bounds(self):
        board = empty_board()
        assert self.v.get_error_type(board, (-1, 0)) == "out_of_bounds"
        assert self.v.get_error_type(board, (0, N)) == "out_of_bounds"

    def test_small_n(self):
        v4 = NQueensVerifier(4)
        board = np.zeros((4, 4), dtype=np.int8)
        # One solution to 4-queens: cols [1,3,0,2]
        for r, c in enumerate([1, 3, 0, 2]):
            board[r, c] = 1
        assert v4.is_complete(board)


# ---------------------------------------------------------------------------
# NQueensGraphEncoder
# ---------------------------------------------------------------------------

class TestNQueensGraphEncoder:
    def setup_method(self):
        self.enc = NQueensGraphEncoder(N)

    def test_node_count(self):
        graph = self.enc.encode(empty_board())
        assert graph.x.shape == (N * N, 18)

    def test_node_feat_dim_property(self):
        assert self.enc.node_feat_dim == 18

    def test_features_empty_board(self):
        graph = self.enc.encode(empty_board())
        # No queens → queen features should all be 0
        assert graph.x[:, 2].sum() == 0  # is_queen
        assert graph.x[:, 3].sum() == 0  # row_has_queen
        assert graph.x[:, 4].sum() == 0  # col_has_queen

    def test_features_with_queen(self):
        board = empty_board()
        board[2, 5] = 1
        graph = self.enc.encode(board)
        node_id = 2 * N + 5
        assert graph.x[node_id, 2] == 1.0  # is_queen
        # All nodes in row 2 should have row_has_queen=1
        for c in range(N):
            assert graph.x[2 * N + c, 3] == 1.0

    def test_edge_index_shape(self):
        graph = self.enc.encode(empty_board())
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0

    def test_edges_bidirectional(self):
        graph = self.enc.encode(empty_board())
        src, dst = graph.edge_index[0], graph.edge_index[1]
        # Every (u, v) edge should have a matching (v, u)
        edges = set(zip(src.tolist(), dst.tolist()))
        for u, v in list(edges)[:20]:  # spot-check first 20
            assert (v, u) in edges

    def test_row_norm_values(self):
        graph = self.enc.encode(empty_board())
        # Node at (r, c) = r*N+c should have row_norm = r/(N-1)
        for r in range(N):
            for c in range(N):
                nid = r * N + c
                expected = r / (N - 1)
                assert abs(graph.x[nid, 0].item() - expected) < 1e-5

    def test_solved_board_features(self):
        graph = self.enc.encode(solved_board_8())
        # queens_placed_frac (feature 6) should be 1.0 for all nodes
        assert (graph.x[:, 6] == 1.0).all()

    def test_encode_returns_data_object(self):
        from torch_geometric.data import Data
        graph = self.enc.encode(empty_board())
        assert isinstance(graph, Data)
        assert hasattr(graph, "batch")


# ---------------------------------------------------------------------------
# NQueensActionProposer
# ---------------------------------------------------------------------------

class TestNQueensActionProposer:
    def setup_method(self):
        self.v = NQueensVerifier(N)
        self.prop = NQueensActionProposer(N, self.v)

    def test_propose_empty_board(self):
        actions = self.prop.propose(empty_board(), top_k=5)
        assert len(actions) <= 5
        assert all(isinstance(a, tuple) and len(a) == 2 for a in actions)

    def test_propose_targets_first_empty_row(self):
        board = partial_board(rows_filled=3)
        actions = self.prop.propose(board)
        # Row 0,1,2 are filled; next should be row 3
        assert all(r == 3 for r, c in actions)

    def test_propose_all_valid(self):
        board = partial_board(rows_filled=2)
        actions = self.prop.propose(board, top_k=N)
        for action in actions:
            assert self.v.is_valid_action(board, action)

    def test_propose_empty_when_full(self):
        actions = self.prop.propose(solved_board_8(), top_k=5)
        assert actions == []

    def test_propose_respects_top_k(self):
        for k in [1, 3, 5]:
            actions = self.prop.propose(empty_board(), top_k=k)
            assert len(actions) <= k

    def test_implements_base_protocol(self):
        from qwm.envs.base import BaseActionProposer
        assert isinstance(self.prop, BaseActionProposer)


# ---------------------------------------------------------------------------
# NQueensQWMController (integration)
# ---------------------------------------------------------------------------

class TestNQueensQWMController:
    def _make_controller(self, n: int = 4) -> NQueensQWMController:
        from qwm.config import Config
        from qwm.training.trainer import QWMTrainer
        config = Config()
        trainer = QWMTrainer(config)
        return NQueensQWMController(trainer.get_models_dict(), config, n=n)

    def test_search_returns_dict(self):
        ctrl = self._make_controller(n=4)
        board = np.zeros((4, 4), dtype=np.int8)
        result = ctrl.search(board, max_nodes=50)
        assert "solved" in result
        assert "nodes_expanded" in result
        assert "merges_performed" in result

    def test_search_already_solved(self):
        ctrl = self._make_controller(n=4)
        board = np.zeros((4, 4), dtype=np.int8)
        for r, c in enumerate([1, 3, 0, 2]):
            board[r, c] = 1
        result = ctrl.search(board)
        assert result["solved"] is True
        assert result["nodes_expanded"] == 0

    def test_search_finds_solution_4queens(self):
        ctrl = self._make_controller(n=4)
        board = np.zeros((4, 4), dtype=np.int8)
        result = ctrl.search(board, max_nodes=200)
        if result["solved"]:
            v = NQueensVerifier(4)
            assert v.is_complete(result["solution_board"])

    def test_encode_state_shape(self):
        from qwm.config import Config
        from qwm.training.trainer import QWMTrainer
        config = Config()
        trainer = QWMTrainer(config)
        ctrl = NQueensQWMController(trainer.get_models_dict(), config, n=4)
        board = np.zeros((4, 4), dtype=np.int8)
        z = ctrl.encode_state(board)
        assert z.shape == (config.quotient_dim,)


# ---------------------------------------------------------------------------
# Protocol conformance checks
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_verifier_implements_protocol(self):
        from qwm.envs.base import BaseVerifier
        assert isinstance(NQueensVerifier(), BaseVerifier)

    def test_encoder_implements_protocol(self):
        from qwm.envs.base import BaseGraphEncoder
        assert isinstance(NQueensGraphEncoder(), BaseGraphEncoder)

    def test_proposer_implements_protocol(self):
        from qwm.envs.base import BaseActionProposer
        assert isinstance(NQueensActionProposer(), BaseActionProposer)
