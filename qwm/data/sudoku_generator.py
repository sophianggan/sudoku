"""Sudoku puzzle generator and backtracking solver with constraint propagation.

Generates puzzles of varying difficulty, solves them with AC-3 + backtracking,
and records SolverTrace objects at every decision point for training QWM.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class SolverTrace:
    """One step of the backtracking solve process."""

    board_state: np.ndarray          # 9×9, 0 = empty
    action: Tuple[int, int, int]     # (row, col, digit)
    next_board_state: np.ndarray     # board after action
    failed: bool                     # does this branch eventually fail?
    obstruction_type: Optional[str]  # why it failed (None if successful)

    def __repr__(self) -> str:
        """Compact representation of a solver trace."""
        r, c, d = self.action
        tag = self.obstruction_type or "ok"
        return f"SolverTrace(({r},{c})={d}, failed={self.failed}, {tag})"


# ────────────────────────────────────────────────────────────────────
# Candidate helpers
# ────────────────────────────────────────────────────────────────────

def get_candidate_set(board: np.ndarray, row: int, col: int) -> Set[int]:
    """Return the set of valid digits {1..9} for *board[row, col]*."""
    if board[row, col] != 0:
        return set()
    used: Set[int] = set()
    used.update(board[row, :].tolist())
    used.update(board[:, col].tolist())
    br, bc = 3 * (row // 3), 3 * (col // 3)
    used.update(board[br:br + 3, bc:bc + 3].ravel().tolist())
    used.discard(0)
    return set(range(1, 10)) - used


def _all_candidates(board: np.ndarray) -> Dict[Tuple[int, int], Set[int]]:
    """Build candidate sets for every empty cell."""
    cands: Dict[Tuple[int, int], Set[int]] = {}
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                cands[(r, c)] = get_candidate_set(board, r, c)
    return cands


# ────────────────────────────────────────────────────────────────────
# AC-3 constraint propagation
# ────────────────────────────────────────────────────────────────────

def _peers(row: int, col: int) -> List[Tuple[int, int]]:
    """Return all peer cells of (row, col) — same row, col, or box."""
    peers: List[Tuple[int, int]] = []
    for c in range(9):
        if c != col:
            peers.append((row, c))
    for r in range(9):
        if r != row:
            peers.append((r, col))
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for dr in range(3):
        for dc in range(3):
            r2, c2 = br + dr, bc + dc
            if (r2, c2) != (row, col) and (r2, c2) not in peers:
                peers.append((r2, c2))
    return peers


def _ac3(cands: Dict[Tuple[int, int], Set[int]]) -> Optional[str]:
    """Run AC-3 on *cands* in-place. Return obstruction_type or None if ok."""
    queue: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    # Initialize arcs: for every empty cell, for every peer
    for cell in list(cands):
        for peer in _peers(*cell):
            if peer in cands:
                queue.append((cell, peer))

    while queue:
        xi, xj = queue.pop(0)
        if xi not in cands or xj not in cands:
            continue
        # If xj has exactly one value, remove it from xi's domain
        if len(cands[xj]) == 1:
            val = next(iter(cands[xj]))
            if val in cands.get(xi, set()):
                cands[xi].discard(val)
                if len(cands[xi]) == 0:
                    return "empty_domain"
                # If xi is now a singleton, propagate
                if len(cands[xi]) == 1:
                    for peer in _peers(*xi):
                        if peer in cands and peer != xj:
                            queue.append((peer, xi))
    return None


def _detect_obstruction(board: np.ndarray, row: int, col: int, digit: int,
                        cands: Dict[Tuple[int, int], Set[int]]) -> Optional[str]:
    """Classify why placing *digit* at (row, col) leads to a contradiction."""
    # Direct conflict checks (on the board before placement)
    if digit in board[row, :]:
        return "row_conflict"
    if digit in board[:, col]:
        return "col_conflict"
    br, bc = 3 * (row // 3), 3 * (col // 3)
    if digit in board[br:br + 3, bc:bc + 3]:
        return "box_conflict"

    # Check constraint-propagation-level violations
    # After tentative placement, see what happens to candidates
    test_board = board.copy()
    test_board[row, col] = digit
    test_cands = _all_candidates(test_board)
    obs = _ac3(test_cands)
    if obs is not None:
        return obs

    # Check for naked single violations in peers
    for peer in _peers(row, col):
        if peer in test_cands and len(test_cands[peer]) == 0:
            return "naked_single_violation"

    # Check hidden single violations: a digit has no valid cell in a unit
    for unit_cells in _get_units():
        empty_in_unit = [c for c in unit_cells if test_board[c[0], c[1]] == 0]
        placed_in_unit = set(test_board[c[0], c[1]] for c in unit_cells if test_board[c[0], c[1]] != 0)
        for d in range(1, 10):
            if d in placed_in_unit:
                continue
            positions = [c for c in empty_in_unit if c in test_cands and d in test_cands[c]]
            if len(positions) == 0:
                return "hidden_single_violation"

    return None


def _get_units() -> List[List[Tuple[int, int]]]:
    """Return all 27 units (9 rows + 9 cols + 9 boxes)."""
    units: List[List[Tuple[int, int]]] = []
    for r in range(9):
        units.append([(r, c) for c in range(9)])
    for c in range(9):
        units.append([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append([(br + dr, bc + dc) for dr in range(3) for dc in range(3)])
    return units


# ────────────────────────────────────────────────────────────────────
# Backtracking solver with trace recording
# ────────────────────────────────────────────────────────────────────

def solve(board: np.ndarray, rng: Optional[random.Random] = None,
          max_traces: int = 5000) -> Tuple[Optional[np.ndarray], List[SolverTrace]]:
    """Solve *board* via backtracking+AC-3, recording traces.

    Returns (solution_or_None, traces).
    """
    if rng is None:
        rng = random.Random(42)

    traces: List[SolverTrace] = []
    result: List[Optional[np.ndarray]] = [None]  # mutable container

    def _backtrack(b: np.ndarray) -> bool:
        if len(traces) >= max_traces:
            return False

        cands = _all_candidates(b)
        if not cands:
            # All cells filled — check if valid
            if _is_valid_complete(b):
                result[0] = b.copy()
                return True
            return False

        # MRV heuristic: pick cell with fewest candidates
        cell = min(cands, key=lambda c: len(cands[c]))
        r, c = cell
        if len(cands[cell]) == 0:
            return False

        vals = list(cands[cell])
        rng.shuffle(vals)
        for digit in vals:
            board_before = b.copy()
            next_board = b.copy()
            next_board[r, c] = digit

            # Try constraint propagation on copy
            test_cands = _all_candidates(next_board)
            obs = _ac3(test_cands)

            if obs is not None:
                # Immediate propagation failure
                traces.append(SolverTrace(
                    board_state=board_before,
                    action=(r, c, digit),
                    next_board_state=next_board,
                    failed=True,
                    obstruction_type=obs,
                ))
                continue

            # Apply naked/hidden singles from propagation
            progress = True
            prop_board = next_board.copy()
            while progress:
                progress = False
                for (cr, cc), vs in list(test_cands.items()):
                    if len(vs) == 1 and prop_board[cr, cc] == 0:
                        val = next(iter(vs))
                        prop_board[cr, cc] = val
                        del test_cands[(cr, cc)]
                        # Remove from peers
                        for peer in _peers(cr, cc):
                            if peer in test_cands:
                                test_cands[peer].discard(val)
                        progress = True

            # Check for empty domain after propagation
            domain_dead = False
            for (cr, cc), vs in test_cands.items():
                if len(vs) == 0:
                    domain_dead = True
                    break

            if domain_dead:
                traces.append(SolverTrace(
                    board_state=board_before,
                    action=(r, c, digit),
                    next_board_state=next_board,
                    failed=True,
                    obstruction_type="empty_domain",
                ))
                continue

            # Record trace optimistically; we'll update if it fails
            trace_idx = len(traces)
            traces.append(SolverTrace(
                board_state=board_before,
                action=(r, c, digit),
                next_board_state=prop_board.copy(),
                failed=False,
                obstruction_type=None,
            ))

            if _backtrack(prop_board):
                return True

            # Branch failed — retroactively mark trace as failed
            traces[trace_idx] = SolverTrace(
                board_state=board_before,
                action=(r, c, digit),
                next_board_state=prop_board.copy(),
                failed=True,
                obstruction_type=_classify_failure(board_before, r, c, digit, prop_board),
            )
        return False

    _backtrack(board.copy())
    return result[0], traces


def _classify_failure(board_before: np.ndarray, row: int, col: int,
                      digit: int, board_after: np.ndarray) -> str:
    """Classify a backtracking failure that happened deeper in the tree."""
    obs = _detect_obstruction(board_before, row, col, digit, _all_candidates(board_after))
    return obs if obs is not None else "empty_domain"


def _is_valid_complete(board: np.ndarray) -> bool:
    """Check whether a fully-filled board is valid."""
    for i in range(9):
        if set(board[i, :]) != set(range(1, 10)):
            return False
        if set(board[:, i]) != set(range(1, 10)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(board[br:br + 3, bc:bc + 3].ravel()) != set(range(1, 10)):
                return False
    return True


# ────────────────────────────────────────────────────────────────────
# Puzzle generation
# ────────────────────────────────────────────────────────────────────

def _generate_full_board(rng: random.Random) -> np.ndarray:
    """Generate a fully solved 9×9 Sudoku board."""
    board = np.zeros((9, 9), dtype=np.int32)
    # Fill the three diagonal boxes first (independent of each other)
    for box in range(3):
        digits = list(range(1, 10))
        rng.shuffle(digits)
        br, bc = box * 3, box * 3
        for i in range(3):
            for j in range(3):
                board[br + i, bc + j] = digits[i * 3 + j]
    # Solve the rest
    sol, _ = solve(board, rng=rng, max_traces=2000)
    if sol is not None:
        return sol
    # Fallback: brute-force fill
    return _fill_board(board, rng)


def _fill_board(board: np.ndarray, rng: random.Random) -> np.ndarray:
    """Recursively fill a partial board."""
    b = board.copy()
    for r in range(9):
        for c in range(9):
            if b[r, c] == 0:
                cands = list(get_candidate_set(b, r, c))
                rng.shuffle(cands)
                for d in cands:
                    b[r, c] = d
                    result = _fill_board(b, rng)
                    if result is not None:
                        return result
                    b[r, c] = 0
                return None  # type: ignore[return-value]
    return b


def generate_puzzle(rng: random.Random, n_clues: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a (puzzle, solution) pair with approximately *n_clues* given digits."""
    solution = _generate_full_board(rng)
    puzzle = solution.copy()
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    removed = 0
    target_remove = 81 - n_clues
    for r, c in cells:
        if removed >= target_remove:
            break
        old = puzzle[r, c]
        puzzle[r, c] = 0
        removed += 1
    return puzzle, solution


def generate_dataset(n_puzzles: int = 10000, seed: int = 42,
                     max_traces_per_puzzle: int = 200) -> List[Tuple[np.ndarray, List[SolverTrace]]]:
    """Generate *n_puzzles* (puzzle, traces) pairs of varying difficulty."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    dataset: List[Tuple[np.ndarray, List[SolverTrace]]] = []

    for i in range(n_puzzles):
        # Vary difficulty: fewer clues = harder
        n_clues = rng.randint(22, 36)
        puzzle, _solution = generate_puzzle(rng, n_clues=n_clues)
        _, traces = solve(puzzle, rng=rng, max_traces=max_traces_per_puzzle)
        dataset.append((puzzle.copy(), traces))

    return dataset


def board_from_string(s: str) -> np.ndarray:
    """Convert an 81-character string (0=empty) to a 9×9 numpy board."""
    assert len(s) == 81, f"Expected 81 chars, got {len(s)}"
    return np.array([int(ch) for ch in s], dtype=np.int32).reshape(9, 9)


def board_to_string(board: np.ndarray) -> str:
    """Convert a 9×9 board to an 81-character string."""
    return "".join(str(int(v)) for v in board.ravel())
