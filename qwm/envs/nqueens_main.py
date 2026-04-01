"""Entry point for QWM N-Queens training, evaluation, and demo.

Commands
--------
    python qwm/envs/nqueens_main.py train
        Generate N-Queens search traces, train all QWM components, save checkpoint.

    python qwm/envs/nqueens_main.py evaluate
        Load checkpoint, run QWM search on test puzzles, print metrics.

    python qwm/envs/nqueens_main.py demo [N]
        Interactive: enter a partial N-Queens board, run search, print result.
        Default N=8.

Run from the repo root:
    python qwm/envs/nqueens_main.py train
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from qwm.config import Config
from qwm.data.obstruction_labeler import OBSTRUCTION_CLASSES
from qwm.envs.nqueens import (
    NQueensActionProposer,
    NQueensBoard,
    NQueensGraphEncoder,
    NQueensQWMController,
    NQueensVerifier,
    _queens,
)
from qwm.training.trainer import QWMTrainer


# ---------------------------------------------------------------------------
# Obstruction mapping: N-Queens error types → 6 shared obstruction classes
# ---------------------------------------------------------------------------

_OBS_MAP: Dict[str, str] = {
    "row_conflict":      "row_conflict",
    "col_conflict":      "col_conflict",
    "diagonal_conflict": "box_conflict",          # closest structural analog
    "cell_occupied":     "naked_single_violation",
    "out_of_bounds":     "empty_domain",
}
_DEFAULT_OBS = "empty_domain"


# ---------------------------------------------------------------------------
# Trace data structure
# ---------------------------------------------------------------------------

@dataclass
class NQueensTrace:
    """One step in an N-Queens search trace."""

    board_before: NQueensBoard
    board_after: NQueensBoard          # copy of board_before if action failed
    action: Tuple[int, int]            # (row, col)
    branch_failed: bool
    obstruction_type: Optional[str]    # mapped to shared obstruction class
    solvability_label: float           # 1.0 if branch eventually solved, else 0.0


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _backtrack_solve(board: NQueensBoard, row: int, n: int,
                     rng: random.Random) -> bool:
    """In-place backtracking solver. Randomises column order for diversity."""
    if row == n:
        return True
    cols = list(range(n))
    rng.shuffle(cols)
    v = NQueensVerifier(n)
    for c in cols:
        if v.is_valid_action(board, (row, c)):
            board[row, c] = 1
            if _backtrack_solve(board, row + 1, n, rng):
                return True
            board[row, c] = 0
    return False


def _generate_partial_board(n: int, filled_rows: int,
                            rng: random.Random) -> Optional[NQueensBoard]:
    """Return a board with queens in the first *filled_rows* rows, or None."""
    board = np.zeros((n, n), dtype=np.int8)
    v = NQueensVerifier(n)
    for row in range(filled_rows):
        cols = list(range(n))
        rng.shuffle(cols)
        placed = False
        for c in cols:
            if v.is_valid_action(board, (row, c)):
                board[row, c] = 1
                placed = True
                break
        if not placed:
            return None  # no valid placement → skip this partial board
    return board


def _dfs_trace(
    board: NQueensBoard,
    row: int,
    n: int,
    verifier: NQueensVerifier,
    proposer: NQueensActionProposer,
    traces: List[NQueensTrace],
    encoder: NQueensGraphEncoder,
    rng: random.Random,
    max_depth: int = 12,
) -> bool:
    """DFS from *row*, collecting traces. Returns True if a solution was found."""
    if row == n:
        return True  # all queens placed
    if row - len(_queens(board)) >= max_depth:
        return False

    actions = list(range(n))
    rng.shuffle(actions)

    for col in actions:
        action = (row, col)
        obs_type_raw = verifier.get_error_type(board, action)
        succeeded = obs_type_raw is None

        if succeeded:
            try:
                next_board = verifier.apply_action(board, action)
            except ValueError:
                succeeded = False
                next_board = board.copy()
                obs_type_raw = "cell_occupied"
        else:
            next_board = board.copy()

        branch_ok = False
        if succeeded:
            branch_ok = _dfs_trace(
                next_board, row + 1, n, verifier, proposer,
                traces, encoder, rng, max_depth
            )

        obs_mapped = _OBS_MAP.get(obs_type_raw or "", _DEFAULT_OBS) if not succeeded else None

        traces.append(NQueensTrace(
            board_before=board.copy(),
            board_after=next_board.copy(),
            action=action,
            branch_failed=not branch_ok,
            obstruction_type=obs_mapped if not branch_ok else None,
            solvability_label=1.0 if branch_ok else 0.0,
        ))

        if branch_ok:
            return True

    return False


def generate_nqueens_dataset(
    n: int = 8,
    n_puzzles: int = 2000,
    max_prefill_rows: int = 4,
    seed: int = 42,
    max_traces_per_puzzle: int = 100,
) -> List[NQueensTrace]:
    """Generate N-Queens search traces for QWM training.

    Parameters
    ----------
    n:
        Board size (N-Queens).
    n_puzzles:
        Number of partial boards to attempt.
    max_prefill_rows:
        Queens are pre-placed in 0 to *max_prefill_rows* rows to vary difficulty.
    seed:
        Random seed.
    max_traces_per_puzzle:
        Cap on traces per puzzle (avoids blowup on hard instances).

    Returns
    -------
    List[NQueensTrace]
        Flat list of all recorded placement transitions.
    """
    rng = random.Random(seed)
    verifier = NQueensVerifier(n)
    proposer = NQueensActionProposer(n, verifier)
    encoder = NQueensGraphEncoder(n)
    all_traces: List[NQueensTrace] = []

    for _ in range(n_puzzles):
        filled = rng.randint(0, max_prefill_rows)
        board = _generate_partial_board(n, filled, rng)
        if board is None:
            continue
        start_row = filled
        puzzle_traces: List[NQueensTrace] = []
        _dfs_trace(board, start_row, n, verifier, proposer,
                   puzzle_traces, encoder, rng)
        all_traces.extend(puzzle_traces[:max_traces_per_puzzle])

    return all_traces


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class NQueensQWMDataset(Dataset):
    """Dataset for NQueensTrace objects — feeds QWMTrainer._collate."""

    def __init__(self, traces: List[NQueensTrace], n: int = 8) -> None:
        self.traces = traces
        self.encoder = NQueensGraphEncoder(n)
        self._obs_classes = OBSTRUCTION_CLASSES

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict:
        t = self.traces[idx]
        obs_label = -1
        if t.branch_failed and t.obstruction_type is not None:
            obs_label = self._obs_classes.get(t.obstruction_type, -1)
        row, col = t.action
        return {
            "graph_t":          self.encoder.encode(t.board_before),
            "graph_t1":         self.encoder.encode(t.board_after),
            "action":           torch.tensor([row, col], dtype=torch.long),
            "obstruction_label": obs_label,
            "merge_value":      t.solvability_label,
            "is_failed":        t.branch_failed,
        }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(n: int = 8) -> None:
    config = Config()
    print(f"[QWM-NQueens] Generating {n}-Queens search traces...")
    traces = generate_nqueens_dataset(
        n=n,
        n_puzzles=3000,
        max_prefill_rows=n // 2,
        seed=config.seed,
        max_traces_per_puzzle=80,
    )
    print(f"[QWM-NQueens] Total traces: {len(traces)}")

    dataset = NQueensQWMDataset(traces, n=n)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=QWMTrainer._collate,
    )

    print("[QWM-NQueens] Training QWM...")
    trainer = QWMTrainer(config)
    trainer.train(n_epochs=30, dataloader=loader)

    ckpt_path = pathlib.Path(f"checkpoints/qwm_nqueens_{n}.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[QWM-NQueens] Checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(n: int = 8) -> None:
    config = Config()
    ckpt_path = pathlib.Path(f"checkpoints/qwm_nqueens_{n}.pt")
    print(f"[QWM-NQueens] Loading checkpoint from {ckpt_path}...")
    trainer = QWMTrainer(config)
    trainer.load_checkpoint(ckpt_path)

    controller = NQueensQWMController(trainer.get_models_dict(), config, n=n)
    verifier = NQueensVerifier(n)

    rng = random.Random(config.seed + 99)
    puzzles: List[NQueensBoard] = []
    for _ in range(50):
        board = _generate_partial_board(n, filled_rows=rng.randint(0, n // 2), rng=rng)
        if board is not None:
            puzzles.append(board)

    print(f"[QWM-NQueens] Evaluating on {len(puzzles)} puzzles...")
    n_solved = 0
    total_nodes = 0
    total_merges = 0
    total_verifier_calls = 0

    for board in puzzles:
        result = controller.search(board, max_nodes=300)
        if result["solved"]:
            n_solved += 1
            assert verifier.is_complete(result["solution_board"])
        total_nodes        += result["nodes_expanded"]
        total_merges       += result["merges_performed"]
        total_verifier_calls += result["verifier_calls"]

    k = len(puzzles)
    metrics = {
        "n":                   n,
        "solve_rate":          n_solved / k,
        "avg_nodes_expanded":  total_nodes / k,
        "avg_merges":          total_merges / k,
        "avg_verifier_calls":  total_verifier_calls / k,
    }
    print("\n=== QWM N-Queens Evaluation Results ===")
    for key, val in metrics.items():
        print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")

    pathlib.Path("results").mkdir(exist_ok=True)
    out = pathlib.Path(f"results/nqueens_{n}_eval.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[QWM-NQueens] Results saved to {out}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo(n: int = 8) -> None:
    config = Config()
    ckpt_path = pathlib.Path(f"checkpoints/qwm_nqueens_{n}.pt")
    trainer = QWMTrainer(config)
    if ckpt_path.exists():
        trainer.load_checkpoint(ckpt_path)
        print(f"[QWM-NQueens] Loaded checkpoint from {ckpt_path}")
    else:
        print("[QWM-NQueens] No checkpoint found — using random weights.")

    controller = NQueensQWMController(trainer.get_models_dict(), config, n=n)
    verifier = NQueensVerifier(n)

    print(f"\nEnter a {n}x{n} board as {n} space-separated rows.")
    print(f"Each row is {n} values: 0=empty, 1=queen.  Press Enter twice when done.\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)

    board = np.zeros((n, n), dtype=np.int8)
    try:
        for r, line in enumerate(lines[:n]):
            vals = [int(v) for v in line.split()]
            for c, v in enumerate(vals[:n]):
                board[r, c] = v
    except ValueError:
        print("[QWM-NQueens] Could not parse board — starting from empty.")
        board = np.zeros((n, n), dtype=np.int8)

    print(f"\n[QWM-NQueens] Initial board ({len(_queens(board))} queens placed):")
    _print_board(board)

    if verifier.is_complete(board):
        print("[QWM-NQueens] Board is already a valid solution!")
        return

    print("\n[QWM-NQueens] Running search...")
    result = controller.search(board, max_nodes=500)

    if result["solved"]:
        print("[QWM-NQueens] Solution found!")
        _print_board(result["solution_board"])
        print(f"  Nodes expanded:   {result['nodes_expanded']}")
        print(f"  Merges performed: {result['merges_performed']}")
        print(f"  Verifier calls:   {result['verifier_calls']}")
    else:
        print("[QWM-NQueens] No solution found within node budget.")
        print(f"  Nodes expanded:   {result['nodes_expanded']}")


def _print_board(board: NQueensBoard) -> None:
    n = board.shape[0]
    sep = "+" + ("-" * (2 * n - 1)) + "+"
    print(sep)
    for r in range(n):
        row_str = "|" + " ".join("Q" if board[r, c] else "." for c in range(n)) + "|"
        print(row_str)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qwm/envs/nqueens_main.py [train|evaluate|demo] [N]")
        sys.exit(1)

    cmd = sys.argv[1]
    board_n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    if cmd == "train":
        train(n=board_n)
    elif cmd == "evaluate":
        evaluate(n=board_n)
    elif cmd == "demo":
        demo(n=board_n)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
