"""QWM multi-domain demo: Sudoku · Lean theorem proving · N-Queens.

Loads the trained checkpoints and runs the QWM search controller on one
example from each domain, printing results and metrics.

Usage
-----
    python demo_all.py

Requires trained checkpoints in checkpoints/.  If a checkpoint is missing,
that domain runs with random (untrained) weights and results will be poor.
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import torch

from qwm.config import Config
from qwm.training.trainer import QWMTrainer


RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _header(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def _load_trainer(ckpt_name: str) -> QWMTrainer:
    config = Config()
    trainer = QWMTrainer(config)
    path = pathlib.Path(f"checkpoints/{ckpt_name}")
    if path.exists():
        trainer.load_checkpoint(path)
        print(f"  {DIM}checkpoint: {path}{RESET}")
    else:
        print(f"  {YELLOW}⚠ {path} not found — using random weights{RESET}")
    return trainer


# ---------------------------------------------------------------------------
# Domain 1: Sudoku
# ---------------------------------------------------------------------------

def demo_sudoku() -> None:
    _header("Domain 1 · Sudoku")

    # A solvable 22-clue puzzle
    puzzle_str = (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    )

    from qwm.data.sudoku_generator import board_from_string
    from qwm.search.controller import QWMController
    from qwm.search.verifier import SudokuVerifier

    board = board_from_string(puzzle_str)
    print(f"\n  Input puzzle ({(board != 0).sum()} clues):")
    _print_sudoku(board)

    trainer = _load_trainer("qwm_sudoku.pt")
    verifier = SudokuVerifier()
    controller = QWMController(trainer.get_models_dict(), Config(), verifier)

    print(f"\n  Running QWM search...")
    result = controller.search(board, max_nodes=500)

    status = f"{GREEN}SOLVED{RESET}" if result["solved"] else f"{YELLOW}unsolved{RESET}"
    print(f"  Status: {status}")
    print(f"  Nodes expanded:   {result['nodes_expanded']}")
    print(f"  Merges performed: {result['merges_performed']}")
    print(f"  Verifier calls:   {result['verifier_calls']}")
    if result["solved"]:
        print(f"\n  Solution:")
        _print_sudoku(result["solution_board"])


def _print_sudoku(board: np.ndarray) -> None:
    for r in range(9):
        if r in (3, 6):
            print(f"  {'─' * 21}")
        row = ""
        for c in range(9):
            if c in (3, 6):
                row += "│"
            v = board[r, c]
            row += f"{DIM}·{RESET}" if v == 0 else str(v)
            if c not in (2, 5, 8):
                row += " "
        print(f"  {row}")


# ---------------------------------------------------------------------------
# Domain 2: Lean theorem proving
# ---------------------------------------------------------------------------

def demo_lean() -> None:
    _header("Domain 2 · Lean Theorem Proving")

    from qwm.lean.controller import LeanQWMController
    from qwm.lean.lean_verifier import LeanVerifier
    from qwm.lean.proof_state import ProofState

    goals = [
        ("⊢ 7 + 5 = 12",              "arithmetic equality"),
        ("⊢ ∀ n : ℕ, n + 0 = n",      "universal statement"),
        ("h : 3 > 0\n⊢ 3 > 0",        "exact hypothesis match"),
    ]

    trainer = _load_trainer("qwm_lean.pt")
    verifier = LeanVerifier(mock=True)
    controller = LeanQWMController(trainer.get_models_dict(), Config(), verifier)

    for goal_str, desc in goals:
        state = ProofState.from_string(goal_str)
        result = controller.search(state, max_nodes=100)
        status = f"{GREEN}proved{RESET}" if result["proved"] else f"{YELLOW}open{RESET}"
        tactics = " → ".join(result["proof_tactics"]) if result["proof_tactics"] else "(trivial)"
        print(f"\n  Goal ({desc}):")
        print(f"    {DIM}{goal_str}{RESET}")
        print(f"  Status: {status}   tactics: {tactics}")
        print(f"  Nodes: {result['nodes_expanded']}  Merges: {result['merges_performed']}")


# ---------------------------------------------------------------------------
# Domain 3: N-Queens
# ---------------------------------------------------------------------------

def demo_nqueens() -> None:
    _header("Domain 3 · N-Queens")

    from qwm.envs.nqueens import NQueensQWMController, NQueensVerifier

    for n, prefill in [(4, 0), (8, 3)]:
        print(f"\n  {n}-Queens puzzle:")
        verifier = NQueensVerifier(n)

        # Build a partial board for N=8 (rows 0-2 filled with a known valid prefix)
        board = np.zeros((n, n), dtype=np.int8)
        if prefill > 0:
            prefix_cols = {4: [1, 3, 0], 8: [0, 4, 7]}[n][:prefill]
            for r, c in enumerate(prefix_cols):
                board[r, c] = 1

        _print_nqueens(board)
        print(f"  Queens placed: {(board == 1).sum()} / {n}")

        ckpt = f"qwm_nqueens_{n}.pt"
        trainer = _load_trainer(ckpt)
        controller = NQueensQWMController(trainer.get_models_dict(), Config(), n=n)

        print(f"  Running QWM search...")
        result = controller.search(board, max_nodes=300)

        status = f"{GREEN}SOLVED{RESET}" if result["solved"] else f"{YELLOW}unsolved{RESET}"
        print(f"  Status: {status}")
        print(f"  Nodes expanded:   {result['nodes_expanded']}")
        print(f"  Merges performed: {result['merges_performed']}")
        if result["solved"]:
            print(f"  Solution:")
            _print_nqueens(result["solution_board"])


def _print_nqueens(board: np.ndarray) -> None:
    n = board.shape[0]
    border = "  +" + ("─" * (2 * n - 1)) + "+"
    print(border)
    for r in range(n):
        row = "|" + " ".join(
            f"{GREEN}Q{RESET}" if board[r, c] else f"{DIM}·{RESET}"
            for c in range(n)
        ) + "|"
        print(f"  {row}")
    print(border)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{BOLD}Quotient World Models (QWM) — Multi-Domain Demo{RESET}")
    print(textwrap.dedent(f"""\
      {DIM}Framework: verifier-grounded best-first search with learned
      state merging (quotient DAG) and failure prediction (obstruction predictor).
      Three domains: Sudoku · Lean theorem proving · N-Queens{RESET}
    """))

    demo_sudoku()
    demo_lean()
    demo_nqueens()

    print(f"\n{BOLD}{GREEN}Demo complete.{RESET}\n")
