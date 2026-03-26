"""SudokuVerifier: exact rule checker for Sudoku boards and actions.

Provides methods to check action validity, board completeness, error types,
and to apply actions safely.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

class SudokuVerifier:
    """Exact Sudoku rule checker for board states and actions."""

    def __init__(self) -> None:
        """No state needed; all methods are static-like."""
        pass

    def is_valid_action(self, board: np.ndarray, row: int, col: int, digit: int) -> bool:
        """Return True if placing digit at (row, col) is valid under Sudoku rules."""
        if board[row, col] != 0:
            return False
        if digit in board[row, :]:
            return False
        if digit in board[:, col]:
            return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        if digit in board[br:br+3, bc:bc+3]:
            return False
        return True

    def is_complete(self, board: np.ndarray) -> bool:
        """Return True if the board is fully and correctly filled."""
        for i in range(9):
            if set(board[i, :]) != set(range(1, 10)):
                return False
            if set(board[:, i]) != set(range(1, 10)):
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                if set(board[br:br+3, bc:bc+3].ravel()) != set(range(1, 10)):
                    return False
        return True

    def get_error_type(self, board: np.ndarray, row: int, col: int, digit: int) -> Optional[str]:
        """Return the obstruction type string if the action is invalid, else None."""
        if board[row, col] != 0:
            return "cell_filled"
        if digit in board[row, :]:
            return "row_conflict"
        if digit in board[:, col]:
            return "col_conflict"
        br, bc = 3 * (row // 3), 3 * (col // 3)
        if digit in board[br:br+3, bc:bc+3]:
            return "box_conflict"
        return None

    def apply_action(self, board: np.ndarray, row: int, col: int, digit: int) -> np.ndarray:
        """Return a new board with digit placed at (row, col), or raise ValueError if invalid."""
        if not self.is_valid_action(board, row, col, digit):
            raise ValueError(f"Invalid action: ({row},{col})={digit}")
        new_board = board.copy()
        new_board[row, col] = digit
        return new_board

    def __repr__(self) -> str:
        return "SudokuVerifier()"
