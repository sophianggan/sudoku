"""PyTorch Dataset classes for QWM training.

``QWMDataset`` — per-transition data (graph_t, graph_t1, action, labels).
``PairDataset`` — additionally provides positive/negative equivalence pairs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from qwm.data.sudoku_generator import SolverTrace
from qwm.data.sudoku_graph import board_to_homogeneous_graph
from qwm.data.obstruction_labeler import OBSTRUCTION_CLASSES


class QWMDataset(Dataset):
    """One item per solver-trace transition.

    Each item is a dict containing PyG graphs for t and t+1,
    the action taken, obstruction labels, and merge-value scores.
    """

    def __init__(
        self,
        traces: List[SolverTrace],
        merge_values: Optional[List[float]] = None,
    ) -> None:
        """Wrap a list of SolverTrace objects into a Dataset."""
        self.traces = traces
        # Pre-compute merge values if not supplied (default: 1.0 for success, 0.0 for failure)
        if merge_values is not None:
            self.merge_values = merge_values
        else:
            self.merge_values = [0.0 if t.failed else 1.0 for t in traces]

    def __len__(self) -> int:
        """Number of transitions."""
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one training sample as a dictionary."""
        trace = self.traces[idx]

        graph_t = board_to_homogeneous_graph(trace.board_state)
        graph_t1 = board_to_homogeneous_graph(trace.next_board_state)

        row, col, digit = trace.action
        cell_idx = row * 9 + col
        action = torch.tensor([cell_idx, digit], dtype=torch.long)

        obs_label = -1
        if trace.failed and trace.obstruction_type is not None:
            obs_label = OBSTRUCTION_CLASSES.get(trace.obstruction_type, -1)

        return {
            "graph_t": graph_t,
            "graph_t1": graph_t1,
            "action": action,
            "obstruction_label": obs_label,
            "merge_value": self.merge_values[idx],
            "is_failed": trace.failed,
        }

    def __repr__(self) -> str:
        """Readable string."""
        return f"QWMDataset(n={len(self)})"


class PairDataset(Dataset):
    """Extends QWMDataset with positive/negative equivalence pairs.

    Each item additionally contains ``graph_pos`` (an equivalent board)
    and ``graphs_neg`` (a list of non-equivalent boards).
    """

    def __init__(
        self,
        traces: List[SolverTrace],
        pairs: List[Tuple[np.ndarray, np.ndarray, bool]],
        n_neg: int = 4,
        merge_values: Optional[List[float]] = None,
        seed: int = 42,
    ) -> None:
        """Build pair-augmented dataset from traces and equivalence pairs."""
        self.base = QWMDataset(traces, merge_values=merge_values)
        self.n_neg = n_neg
        self.rng = random.Random(seed)

        # Separate pairs into positives and negatives
        self.pos_boards: List[Tuple[np.ndarray, np.ndarray]] = []
        self.neg_boards: List[Tuple[np.ndarray, np.ndarray]] = []
        for b1, b2, is_eq in pairs:
            if is_eq:
                self.pos_boards.append((b1, b2))
            else:
                self.neg_boards.append((b1, b2))

        # Fallback: if no positive pairs, use self-pairs
        if not self.pos_boards:
            for t in traces[:min(50, len(traces))]:
                self.pos_boards.append((t.board_state, t.board_state.copy()))

        # Collect a pool of negative boards for sampling
        self.neg_pool: List[np.ndarray] = []
        for b1, b2 in self.neg_boards:
            self.neg_pool.append(b1)
            self.neg_pool.append(b2)
        if not self.neg_pool:
            self.neg_pool = [t.board_state for t in traces[:100]]

    def __len__(self) -> int:
        """Number of transitions (same as base)."""
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return base sample augmented with equivalence pair data."""
        item = self.base[idx]

        # Pick a random positive pair and use the second board as positive
        pos_pair = self.rng.choice(self.pos_boards)
        item["graph_pos"] = board_to_homogeneous_graph(pos_pair[1])

        # Sample n_neg negative boards
        negs = self.rng.choices(self.neg_pool, k=self.n_neg)
        item["graphs_neg"] = [board_to_homogeneous_graph(b) for b in negs]

        return item

    def __repr__(self) -> str:
        """Readable string."""
        return f"PairDataset(n={len(self)}, pos={len(self.pos_boards)}, neg_pool={len(self.neg_pool)})"
