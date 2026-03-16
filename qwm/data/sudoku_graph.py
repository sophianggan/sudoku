"""Convert a Sudoku board state into a PyG graph representation.

Supports both heterogeneous (cell + constraint nodes) and homogeneous
(flattened, padded) graph formats.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

from qwm.data.sudoku_generator import get_candidate_set


# ────────────────────────────────────────────────────────────────────
# Heterogeneous graph builder
# ────────────────────────────────────────────────────────────────────

def board_to_hetero_graph(board: np.ndarray) -> HeteroData:
    """Build a heterogeneous PyG graph from a 9×9 Sudoku board.

    Node types
    ----------
    - ``cell`` (81 nodes, 15 features each)
    - ``constraint`` (27 nodes, 6 features each)

    Edge types
    ----------
    - ``(cell, belongs_to, constraint)`` — 243 edges
    - ``(constraint, contains, cell)``   — 243 edges (reverse)
    """
    data = HeteroData()

    # ── Cell node features ──────────────────────────────────────────
    cell_feats: List[List[float]] = []
    for r in range(9):
        for c in range(9):
            box_id = (r // 3) * 3 + (c // 3)
            val = int(board[r, c])
            is_empty = 1.0 if val == 0 else 0.0
            cands = get_candidate_set(board, r, c)
            cand_count = len(cands)

            # One-hot of current value (0 → all zeros)
            one_hot = [0.0] * 9
            if val > 0:
                one_hot[val - 1] = 1.0

            feat = [
                r / 8.0,
                c / 8.0,
                box_id / 8.0,
                val / 9.0,
                is_empty,
                cand_count / 9.0,
            ] + one_hot  # 6 + 9 = 15 features
            cell_feats.append(feat)

    data["cell"].x = torch.tensor(cell_feats, dtype=torch.float32)  # (81, 15)

    # ── Constraint node features ────────────────────────────────────
    # 0-8 = rows, 9-17 = columns, 18-26 = boxes
    constraint_feats: List[List[float]] = []
    for kind in range(3):  # 0=row, 1=col, 2=box
        for idx in range(9):
            one_hot_kind = [0.0, 0.0, 0.0]
            one_hot_kind[kind] = 1.0

            # Count satisfied (placed) and remaining (empty) cells
            if kind == 0:  # row
                cells_in_unit = [(idx, c) for c in range(9)]
            elif kind == 1:  # col
                cells_in_unit = [(r, idx) for r in range(9)]
            else:  # box
                br, bc = 3 * (idx // 3), 3 * (idx % 3)
                cells_in_unit = [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]

            satisfied = sum(1 for r2, c2 in cells_in_unit if board[r2, c2] != 0)
            remaining = 9 - satisfied

            feat = one_hot_kind + [idx / 8.0, satisfied / 9.0, remaining / 9.0]
            constraint_feats.append(feat)

    data["constraint"].x = torch.tensor(constraint_feats, dtype=torch.float32)  # (27, 6)

    # ── Edges: cell → constraint ────────────────────────────────────
    src_cell: List[int] = []
    dst_constraint: List[int] = []
    for r in range(9):
        for c in range(9):
            cell_id = r * 9 + c
            box_id = (r // 3) * 3 + (c // 3)
            src_cell.extend([cell_id, cell_id, cell_id])
            dst_constraint.extend([r, 9 + c, 18 + box_id])  # row, col, box

    edge_c2con = torch.tensor([src_cell, dst_constraint], dtype=torch.long)
    edge_con2c = torch.tensor([dst_constraint, src_cell], dtype=torch.long)

    data["cell", "belongs_to", "constraint"].edge_index = edge_c2con
    data["constraint", "contains", "cell"].edge_index = edge_con2c

    return data


# ────────────────────────────────────────────────────────────────────
# Homogeneous (flattened) graph builder
# ────────────────────────────────────────────────────────────────────

_PAD_DIM = 18  # node_feat_dim from Config — we pad all nodes to this width


def board_to_homogeneous_graph(board: np.ndarray, pad_dim: int = _PAD_DIM) -> Data:
    """Convert a Sudoku board into a flat (homogeneous) PyG ``Data`` graph.

    All 81 cell nodes and 27 constraint nodes are concatenated into a single
    node list.  Features are zero-padded to *pad_dim*.

    Returns a ``Data`` object with:
    - ``x``:          (108, pad_dim) node features
    - ``edge_index``: (2, 486) edges (243 cell→constraint + 243 reverse)
    - ``node_type``:  (108,) int tensor — 0=cell, 1=constraint
    """
    hetero = board_to_hetero_graph(board)

    cell_x = hetero["cell"].x           # (81, 15)
    cons_x = hetero["constraint"].x     # (27, 6)

    # Pad to common width
    def _pad(t: torch.Tensor, target: int) -> torch.Tensor:
        if t.size(1) >= target:
            return t[:, :target]
        return torch.cat([t, torch.zeros(t.size(0), target - t.size(1))], dim=1)

    cell_x_pad = _pad(cell_x, pad_dim)   # (81, pad_dim)
    cons_x_pad = _pad(cons_x, pad_dim)   # (27, pad_dim)

    x = torch.cat([cell_x_pad, cons_x_pad], dim=0)  # (108, pad_dim)
    node_type = torch.cat([
        torch.zeros(81, dtype=torch.long),
        torch.ones(27, dtype=torch.long),
    ])

    # Remap edges: constraint nodes shift by +81
    c2con = hetero["cell", "belongs_to", "constraint"].edge_index.clone()
    c2con[1] += 81  # constraint ids → 81..107

    con2c = hetero["constraint", "contains", "cell"].edge_index.clone()
    con2c[0] += 81

    edge_index = torch.cat([c2con, con2c], dim=1)  # (2, 486)

    data = Data(x=x, edge_index=edge_index)
    data.node_type = node_type
    # Store batch indicator (needed later for pooling)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data
