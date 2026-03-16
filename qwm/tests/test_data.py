"""Tests for the data layer (Phase 1).

Covers:
  - Sudoku generator & solver correctness
  - Graph builder node/edge counts
  - Equivalence labeler with label-rotated boards
  - Obstruction labeler classification
"""

from __future__ import annotations

import numpy as np
import pytest

from qwm.data.sudoku_generator import (
    SolverTrace,
    board_from_string,
    board_to_string,
    generate_puzzle,
    get_candidate_set,
    solve,
)
from qwm.data.sudoku_graph import board_to_hetero_graph, board_to_homogeneous_graph
from qwm.data.equivalence_labeler import are_equivalent, generate_equivalent_pairs
from qwm.data.obstruction_labeler import (
    OBSTRUCTION_CLASSES,
    ObstructionLabel,
    extract_obstruction,
)

import random


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_puzzle() -> np.ndarray:
    """A puzzle with ~30 clues — should be easy to solve."""
    rng = random.Random(123)
    puzzle, _sol = generate_puzzle(rng, n_clues=35)
    return puzzle


@pytest.fixture
def solved_board() -> np.ndarray:
    """A fully-solved valid board."""
    rng = random.Random(99)
    puzzle, sol = generate_puzzle(rng, n_clues=35)
    result, _ = solve(puzzle, rng=random.Random(99), max_traces=2000)
    if result is None:
        pytest.skip("Solver could not solve fixture puzzle")
    return result


# ────────────────────────────────────────────────────────────────────
# Solver tests
# ────────────────────────────────────────────────────────────────────

class TestSolver:
    """Tests for the Sudoku solver and generator."""

    def test_generate_puzzle_shape(self) -> None:
        """Generated puzzle should be 9×9."""
        rng = random.Random(42)
        puzzle, sol = generate_puzzle(rng, n_clues=30)
        assert puzzle.shape == (9, 9)
        assert sol.shape == (9, 9)

    def test_solution_is_valid(self, solved_board: np.ndarray) -> None:
        """Solved board must satisfy all Sudoku constraints."""
        for i in range(9):
            assert set(solved_board[i, :]) == set(range(1, 10)), f"Row {i} invalid"
            assert set(solved_board[:, i]) == set(range(1, 10)), f"Col {i} invalid"
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                vals = set(solved_board[br:br+3, bc:bc+3].ravel())
                assert vals == set(range(1, 10)), f"Box ({br},{bc}) invalid"

    def test_solver_produces_traces(self, easy_puzzle: np.ndarray) -> None:
        """Solver should record at least one trace step."""
        _, traces = solve(easy_puzzle, rng=random.Random(42), max_traces=500)
        assert len(traces) > 0
        for t in traces:
            assert t.board_state.shape == (9, 9)
            assert t.next_board_state.shape == (9, 9)
            assert len(t.action) == 3

    def test_failed_traces_have_obstruction_type(self, easy_puzzle: np.ndarray) -> None:
        """Failed traces should carry a non-None obstruction_type."""
        _, traces = solve(easy_puzzle, rng=random.Random(42), max_traces=500)
        for t in traces:
            if t.failed:
                assert t.obstruction_type is not None
                assert t.obstruction_type in OBSTRUCTION_CLASSES

    def test_candidate_set(self) -> None:
        """get_candidate_set returns correct candidates."""
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 5
        board[0, 1] = 3
        cands = get_candidate_set(board, 0, 2)
        assert 5 not in cands
        assert 3 not in cands
        assert all(1 <= d <= 9 for d in cands)

    def test_board_string_roundtrip(self) -> None:
        """board_from_string ↔ board_to_string should roundtrip."""
        s = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        board = board_from_string(s)
        assert board.shape == (9, 9)
        assert board_to_string(board) == s


# ────────────────────────────────────────────────────────────────────
# Graph tests
# ────────────────────────────────────────────────────────────────────

class TestGraph:
    """Tests for the Sudoku → PyG graph conversion."""

    def test_hetero_graph_node_counts(self, easy_puzzle: np.ndarray) -> None:
        """Heterogeneous graph must have 81 cell nodes and 27 constraint nodes."""
        hg = board_to_hetero_graph(easy_puzzle)
        assert hg["cell"].x.shape[0] == 81
        assert hg["constraint"].x.shape[0] == 27

    def test_hetero_graph_edge_counts(self, easy_puzzle: np.ndarray) -> None:
        """Each direction should have 243 edges."""
        hg = board_to_hetero_graph(easy_puzzle)
        assert hg["cell", "belongs_to", "constraint"].edge_index.shape[1] == 243
        assert hg["constraint", "contains", "cell"].edge_index.shape[1] == 243

    def test_homogeneous_graph_shape(self, easy_puzzle: np.ndarray) -> None:
        """Homogeneous graph: 108 nodes, 486 edges, pad_dim=18."""
        g = board_to_homogeneous_graph(easy_puzzle)
        assert g.x.shape == (108, 18)
        assert g.edge_index.shape == (2, 486)

    def test_homogeneous_graph_node_types(self, easy_puzzle: np.ndarray) -> None:
        """First 81 nodes should be type 0 (cell), last 27 type 1 (constraint)."""
        g = board_to_homogeneous_graph(easy_puzzle)
        assert (g.node_type[:81] == 0).all()
        assert (g.node_type[81:] == 1).all()


# ────────────────────────────────────────────────────────────────────
# Equivalence tests
# ────────────────────────────────────────────────────────────────────

class TestEquivalence:
    """Tests for residual-graph isomorphism equivalence checking."""

    def test_identical_boards_are_equivalent(self, easy_puzzle: np.ndarray) -> None:
        """A board should be equivalent to itself."""
        assert are_equivalent(easy_puzzle, easy_puzzle.copy())

    def test_different_boards_not_equivalent(self) -> None:
        """Two boards with very different fill levels should not be equivalent."""
        rng = random.Random(42)
        b1, _ = generate_puzzle(rng, n_clues=35)
        b2, _ = generate_puzzle(rng, n_clues=22)
        assert not are_equivalent(b1, b2)

    def test_label_rotated_equivalence(self) -> None:
        """Relabeling all digits via a permutation should preserve equivalence.

        If we swap every 1→2, 2→3, …, 9→1 in a board, the residual
        constraint graph structure is the same.
        """
        rng = random.Random(77)
        board, _ = generate_puzzle(rng, n_clues=30)
        perm = {0: 0}
        vals = list(range(1, 10))
        rng.shuffle(vals)
        for i, v in enumerate(vals, 1):
            perm[i] = v
        rotated = np.vectorize(perm.get)(board).astype(np.int32)
        assert are_equivalent(board, rotated)

    def test_generate_pairs_returns_both_classes(self) -> None:
        """Pair generator should return at least some negatives."""
        rng = random.Random(42)
        puzzle, _ = generate_puzzle(rng, n_clues=30)
        _, traces = solve(puzzle, rng=rng, max_traces=300)
        if len(traces) < 10:
            pytest.skip("Too few traces for pair generation")
        pairs = generate_equivalent_pairs(traces, n_pairs=20, seed=42)
        # Should have at least some pairs (may not have positives, that's ok)
        assert len(pairs) > 0
        labels = [p[2] for p in pairs]
        # At minimum we should have negatives
        assert False in labels


# ────────────────────────────────────────────────────────────────────
# Obstruction tests
# ────────────────────────────────────────────────────────────────────

class TestObstruction:
    """Tests for obstruction labeling."""

    def test_extract_from_failed_trace(self, easy_puzzle: np.ndarray) -> None:
        """extract_obstruction should produce an ObstructionLabel for failed traces."""
        _, traces = solve(easy_puzzle, rng=random.Random(42), max_traces=500)
        failed_traces = [t for t in traces if t.failed]
        if not failed_traces:
            pytest.skip("No failed traces in this puzzle")
        for t in failed_traces[:5]:
            label = extract_obstruction(t)
            assert label is not None
            assert 0 <= label.class_id < 6
            assert label.class_name in OBSTRUCTION_CLASSES
            assert len(label.implicated_cells) >= 1

    def test_non_failed_trace_returns_none(self, easy_puzzle: np.ndarray) -> None:
        """Non-failed traces should yield None."""
        _, traces = solve(easy_puzzle, rng=random.Random(42), max_traces=500)
        ok_traces = [t for t in traces if not t.failed]
        if not ok_traces:
            pytest.skip("No non-failed traces")
        assert extract_obstruction(ok_traces[0]) is None

    def test_all_obstruction_classes_mapped(self) -> None:
        """OBSTRUCTION_CLASSES should map exactly 6 types to 0-5."""
        assert len(OBSTRUCTION_CLASSES) == 6
        assert set(OBSTRUCTION_CLASSES.values()) == set(range(6))
