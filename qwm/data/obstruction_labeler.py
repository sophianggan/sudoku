"""Obstruction labeling — classify *why* a solver branch failed.

Maps the 6 obstruction types used by the solver into structured
``ObstructionLabel`` objects suitable for training the obstruction predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qwm.data.sudoku_generator import SolverTrace, get_candidate_set


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

OBSTRUCTION_CLASSES: Dict[str, int] = {
    "row_conflict": 0,
    "col_conflict": 1,
    "box_conflict": 2,
    "naked_single_violation": 3,
    "hidden_single_violation": 4,
    "empty_domain": 5,
}

CLASS_NAMES: Dict[int, str] = {v: k for k, v in OBSTRUCTION_CLASSES.items()}


# ────────────────────────────────────────────────────────────────────
# Data structure
# ────────────────────────────────────────────────────────────────────

@dataclass
class ObstructionLabel:
    """Structured label for one obstruction event."""

    class_id: int
    class_name: str
    implicated_cells: List[Tuple[int, int]]
    implicated_constraint: Optional[int]  # unit index 0-26

    def __repr__(self) -> str:
        """Concise representation."""
        return (f"ObstructionLabel(id={self.class_id}, "
                f"name={self.class_name!r}, "
                f"cells={self.implicated_cells})")


# ────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────

def _find_implicated_cells(board: np.ndarray, row: int, col: int,
                           digit: int, obs_type: str) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    """Identify which cells and constraint unit are involved in the obstruction."""
    cells: List[Tuple[int, int]] = [(row, col)]
    constraint_id: Optional[int] = None

    if obs_type == "row_conflict":
        # Find the other cell in the same row that already has *digit*
        for c2 in range(9):
            if c2 != col and board[row, c2] == digit:
                cells.append((row, c2))
        constraint_id = row  # row unit index

    elif obs_type == "col_conflict":
        for r2 in range(9):
            if r2 != row and board[r2, col] == digit:
                cells.append((r2, col))
        constraint_id = 9 + col  # col unit index

    elif obs_type == "box_conflict":
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for dr in range(3):
            for dc in range(3):
                r2, c2 = br + dr, bc + dc
                if (r2, c2) != (row, col) and board[r2, c2] == digit:
                    cells.append((r2, c2))
        box_id = (row // 3) * 3 + (col // 3)
        constraint_id = 18 + box_id  # box unit index

    elif obs_type == "naked_single_violation":
        # The action cell plus any peer whose domain was wiped
        test_board = board.copy()
        test_board[row, col] = digit
        for r2 in range(9):
            for c2 in range(9):
                if test_board[r2, c2] == 0:
                    if len(get_candidate_set(test_board, r2, c2)) == 0:
                        cells.append((r2, c2))

    elif obs_type == "hidden_single_violation":
        # Similar logic: find cells that lost all placements for some digit
        test_board = board.copy()
        test_board[row, col] = digit
        for r2 in range(9):
            for c2 in range(9):
                if test_board[r2, c2] == 0:
                    if len(get_candidate_set(test_board, r2, c2)) <= 1:
                        cells.append((r2, c2))

    elif obs_type == "empty_domain":
        test_board = board.copy()
        test_board[row, col] = digit
        for r2 in range(9):
            for c2 in range(9):
                if test_board[r2, c2] == 0:
                    if len(get_candidate_set(test_board, r2, c2)) == 0:
                        cells.append((r2, c2))

    # Deduplicate
    cells = list(dict.fromkeys(cells))
    return cells, constraint_id


def extract_obstruction(trace: SolverTrace) -> Optional[ObstructionLabel]:
    """Extract a structured obstruction label from a failed SolverTrace."""
    if not trace.failed or trace.obstruction_type is None:
        return None

    obs_type = trace.obstruction_type
    class_id = OBSTRUCTION_CLASSES.get(obs_type)
    if class_id is None:
        return None

    row, col, digit = trace.action
    cells, constraint = _find_implicated_cells(
        trace.board_state, row, col, digit, obs_type)

    return ObstructionLabel(
        class_id=class_id,
        class_name=obs_type,
        implicated_cells=cells,
        implicated_constraint=constraint,
    )
