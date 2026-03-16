"""Equivalence labeling for Sudoku board states.

Two board states are *equivalent* if their residual constraint graphs are
isomorphic (up to value relabeling).  This module builds those graphs and
checks isomorphism via ``networkx.algorithms.isomorphism.GraphMatcher``.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from qwm.data.sudoku_generator import SolverTrace, get_candidate_set


# ────────────────────────────────────────────────────────────────────
# Residual constraint graph
# ────────────────────────────────────────────────────────────────────

def _residual_graph(board: np.ndarray) -> nx.Graph:
    """Build the residual constraint graph of a Sudoku board.

    Nodes
    -----
    - One node per empty cell, labelled ``("cell", r, c)`` with attribute
      ``cands`` = frozenset of remaining candidates.
    - One node per constraint unit that still has unsatisfied positions,
      labelled ``("unit", kind, idx)`` where *kind* ∈ {0,1,2} (row/col/box).

    Edges
    -----
    An edge ``(cell, unit)`` exists when the empty cell participates in
    that unsatisfied constraint unit.
    """
    G = nx.Graph()

    # Add cell nodes
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                cands = frozenset(get_candidate_set(board, r, c))
                G.add_node(("cell", r, c), ntype="cell", cand_size=len(cands), cands=cands)

    # Add constraint nodes and edges
    units = _get_units_with_labels()
    for kind, idx, cells in units:
        empty_cells = [(r, c) for r, c in cells if board[r, c] == 0]
        if not empty_cells:
            continue  # fully satisfied
        placed = frozenset(int(board[r, c]) for r, c in cells if board[r, c] != 0)
        remaining = frozenset(range(1, 10)) - placed
        G.add_node(("unit", kind, idx), ntype="unit",
                    remaining_count=len(remaining), remaining=remaining)
        for r, c in empty_cells:
            G.add_edge(("cell", r, c), ("unit", kind, idx))

    return G


def _get_units_with_labels() -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    """Return (kind, idx, cells) for all 27 Sudoku units."""
    units: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for r in range(9):
        units.append((0, r, [(r, c) for c in range(9)]))
    for c in range(9):
        units.append((1, c, [(r, c) for r in range(9)]))
    for b in range(9):
        br, bc = 3 * (b // 3), 3 * (b % 3)
        units.append((2, b, [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]))
    return units


# ────────────────────────────────────────────────────────────────────
# Isomorphism check
# ────────────────────────────────────────────────────────────────────

def _node_match(n1: dict, n2: dict) -> bool:
    """Node compatibility for isomorphism: same type and same candidate count."""
    if n1["ntype"] != n2["ntype"]:
        return False
    if n1["ntype"] == "cell":
        return n1["cand_size"] == n2["cand_size"]
    else:  # unit
        return n1["remaining_count"] == n2["remaining_count"]


def are_equivalent(board1: np.ndarray, board2: np.ndarray) -> bool:
    """Check if two board states have isomorphic residual constraint graphs."""
    g1 = _residual_graph(board1)
    g2 = _residual_graph(board2)

    # Quick rejection: different number of nodes/edges
    if g1.number_of_nodes() != g2.number_of_nodes():
        return False
    if g1.number_of_edges() != g2.number_of_edges():
        return False

    gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2, node_match=_node_match)
    return gm.is_isomorphic()


# ────────────────────────────────────────────────────────────────────
# Pair generation
# ────────────────────────────────────────────────────────────────────

def generate_equivalent_pairs(
    traces: List[SolverTrace],
    n_pairs: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """Generate positive (equivalent) and negative (non-equivalent) board pairs.

    Returns *n_pairs* tuples of ``(board1, board2, is_equivalent)``.
    Approximately half positive, half negative.
    """
    rng = random.Random(seed)
    boards = [t.board_state for t in traces if np.sum(t.board_state == 0) > 10]

    if len(boards) < 4:
        return []

    pairs: List[Tuple[np.ndarray, np.ndarray, bool]] = []
    attempts = 0
    max_attempts = n_pairs * 20

    # ── Positive pairs ──────────────────────────────────────────────
    target_pos = n_pairs // 2
    pos_found = 0
    while pos_found < target_pos and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(len(boards)), 2)
        b1, b2 = boards[i], boards[j]
        # Quick filter: same number of empties
        if np.sum(b1 == 0) != np.sum(b2 == 0):
            continue
        if are_equivalent(b1, b2):
            pairs.append((b1.copy(), b2.copy(), True))
            pos_found += 1

    # ── Negative pairs ──────────────────────────────────────────────
    target_neg = n_pairs - pos_found
    neg_found = 0
    attempts = 0
    while neg_found < target_neg and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(len(boards)), 2)
        b1, b2 = boards[i], boards[j]
        # Boards with different empty counts are clearly non-equivalent
        if np.sum(b1 == 0) != np.sum(b2 == 0):
            pairs.append((b1.copy(), b2.copy(), False))
            neg_found += 1
        elif not are_equivalent(b1, b2):
            pairs.append((b1.copy(), b2.copy(), False))
            neg_found += 1

    rng.shuffle(pairs)
    return pairs[:n_pairs]
