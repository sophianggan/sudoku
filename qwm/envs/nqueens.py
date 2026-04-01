"""N-Queens environment for QWM: verifier, graph encoder, action proposer, and controller.

Implements the three protocols defined in qwm.envs.base so the QWM search
framework can operate on N-Queens as a third reasoning domain alongside Sudoku
and Lean theorem proving.

State representation
--------------------
An (N, N) numpy int8 array.  0 = empty cell, 1 = queen placed.
The goal is to place exactly N queens such that no two attack each other.

Action representation
---------------------
A (row, col) tuple — place a queen at row *row*, column *col*.

Typical usage
-------------
    from qwm.envs.nqueens import NQueensVerifier, NQueensGraphEncoder, NQueensActionProposer

    n = 8
    state = np.zeros((n, n), dtype=np.int8)
    verifier = NQueensVerifier(n)
    encoder  = NQueensGraphEncoder(n)
    proposer = NQueensActionProposer(n, verifier)

    graph = encoder.encode(state)          # PyG Data, x: (n*n, 18)
    actions = proposer.propose(state, 5)   # list of (row, col) tuples
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

NQueensBoard = np.ndarray  # shape (n, n), dtype int8

_PAD_DIM = 18  # must match Config.node_feat_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _queens(state: NQueensBoard) -> List[Tuple[int, int]]:
    """Return list of (row, col) positions of all queens on the board."""
    rows, cols = np.where(state == 1)
    return list(zip(rows.tolist(), cols.tolist()))


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class NQueensVerifier:
    """Exact checker for N-Queens board states and queen-placement actions.

    Implements the BaseVerifier protocol from qwm.envs.base.
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def is_complete(self, state: NQueensBoard) -> bool:
        """True when exactly n non-attacking queens are placed."""
        queens = _queens(state)
        if len(queens) != self.n:
            return False
        for i, (r1, c1) in enumerate(queens):
            for r2, c2 in queens[i + 1:]:
                if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    def is_valid_action(self, state: NQueensBoard, action: Tuple[int, int]) -> bool:
        """True if placing a queen at (row, col) violates no constraint."""
        return self.get_error_type(state, action) is None

    def apply_action(self, state: NQueensBoard, action: Tuple[int, int]) -> NQueensBoard:
        """Return a new board with a queen placed at *action*, or raise ValueError."""
        if not self.is_valid_action(state, action):
            raise ValueError(f"Invalid queen placement at {action}")
        new_state = state.copy()
        new_state[action[0], action[1]] = 1
        return new_state

    def get_error_type(self, state: NQueensBoard, action: Tuple[int, int]) -> Optional[str]:
        """Return an obstruction-class string if *action* is illegal, else None."""
        row, col = action
        n = self.n
        if not (0 <= row < n and 0 <= col < n):
            return "out_of_bounds"
        if state[row, col] != 0:
            return "cell_occupied"
        for qr, qc in _queens(state):
            if qr == row:
                return "row_conflict"
            if qc == col:
                return "col_conflict"
            if abs(qr - row) == abs(qc - col):
                return "diagonal_conflict"
        return None

    def __repr__(self) -> str:
        return f"NQueensVerifier(n={self.n})"


# ---------------------------------------------------------------------------
# Graph encoder
# ---------------------------------------------------------------------------

class NQueensGraphEncoder:
    """Converts an N-Queens board state into a homogeneous PyG graph.

    Implements the BaseGraphEncoder protocol from qwm.envs.base.

    Graph structure
    ---------------
    - N*N nodes, one per cell.
    - Edges connect cells that attack each other (same row, col, or diagonal).
    - Node features (8 values, zero-padded to ``_PAD_DIM = 18``):
        0  row / (n-1)           normalised row position
        1  col / (n-1)           normalised col position
        2  is_queen              1.0 if a queen occupies this cell
        3  row_has_queen         row already contains a queen
        4  col_has_queen         column already contains a queen
        5  diag_attacked         cell lies on a diagonal of some queen
        6  queens_placed / n     fraction of queens placed so far
        7  distinct_cols / n     fraction of distinct columns occupied
    """

    _RAW_DIM = 8  # features before zero-padding

    def __init__(self, n: int = 8) -> None:
        self.n = n

    @property
    def node_feat_dim(self) -> int:
        return _PAD_DIM

    def encode(self, state: NQueensBoard) -> Data:
        """Return a PyG ``Data`` with x: (n*n, 18) and edge_index."""
        n = self.n
        queens = _queens(state)
        queen_rows = {qr for qr, _ in queens}
        queen_cols = {qc for _, qc in queens}
        diag1 = {qr - qc for qr, qc in queens}  # '\'  diagonal key
        diag2 = {qr + qc for qr, qc in queens}  # '/'  diagonal key

        # ── Node features ──────────────────────────────────────────────
        feats: List[List[float]] = []
        for r in range(n):
            row_q = 1.0 if r in queen_rows else 0.0
            for c in range(n):
                col_q = 1.0 if c in queen_cols else 0.0
                diag_atk = 1.0 if ((r - c) in diag1 or (r + c) in diag2) else 0.0
                feat = [
                    r / max(n - 1, 1),        # 0: row_norm
                    c / max(n - 1, 1),        # 1: col_norm
                    float(state[r, c] == 1),  # 2: is_queen
                    row_q,                    # 3: row_has_queen
                    col_q,                    # 4: col_has_queen
                    diag_atk,                 # 5: diag_attacked
                    len(queens) / n,          # 6: queens_placed_frac
                    len(queen_cols) / n,      # 7: distinct_cols_frac
                ]
                feat += [0.0] * (_PAD_DIM - self._RAW_DIM)
                feats.append(feat)

        x = torch.tensor(feats, dtype=torch.float32)  # (n*n, _PAD_DIM)

        # ── Edges: bidirectional between every attacking-cell pair ──────
        src, dst = [], []
        for r1 in range(n):
            for c1 in range(n):
                nid1 = r1 * n + c1
                for r2 in range(n):
                    for c2 in range(n):
                        if r2 == r1 and c2 == c1:
                            continue
                        nid2 = r2 * n + c2
                        if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                            src.append(nid1)
                            dst.append(nid2)

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else \
                     torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(n * n, dtype=torch.long)
        return data

    def __repr__(self) -> str:
        return f"NQueensGraphEncoder(n={self.n})"


# ---------------------------------------------------------------------------
# Action proposer
# ---------------------------------------------------------------------------

class NQueensActionProposer:
    """Proposes candidate queen-placement actions for a given board state.

    Implements the BaseActionProposer protocol from qwm.envs.base.

    Strategy: fill rows in order 0 → n-1.  Within the next unfilled row,
    prefer columns that are not yet under attack (fewer conflicts first).
    """

    def __init__(self, n: int = 8, verifier: Optional[NQueensVerifier] = None) -> None:
        self.n = n
        self.verifier = verifier or NQueensVerifier(n)

    def propose(self, state: NQueensBoard, top_k: int = 5) -> List[Tuple[int, int]]:
        """Return up to *top_k* valid (row, col) actions for *state*."""
        n = self.n
        queen_rows = {qr for qr, _ in _queens(state)}

        # Find the first unfilled row
        target_row: Optional[int] = None
        for r in range(n):
            if r not in queen_rows:
                target_row = r
                break

        if target_row is None:
            return []

        # Score columns: 0 = unattacked (preferred), 1 = attacked
        queen_cols = {qc for _, qc in _queens(state)}
        queens = _queens(state)
        diag1 = {qr - qc for qr, qc in queens}
        diag2 = {qr + qc for qr, qc in queens}

        candidates: List[Tuple[int, int, int]] = []  # (score, row, col)
        for c in range(n):
            if not self.verifier.is_valid_action(state, (target_row, c)):
                continue
            atk = (
                (c in queen_cols)
                or ((target_row - c) in diag1)
                or ((target_row + c) in diag2)
            )
            candidates.append((int(atk), target_row, c))

        candidates.sort()
        return [(r, c) for _, r, c in candidates[:top_k]]

    def __repr__(self) -> str:
        return f"NQueensActionProposer(n={self.n})"


# ---------------------------------------------------------------------------
# QWM search controller
# ---------------------------------------------------------------------------

class NQueensQWMController:
    """Best-first quotient-DAG search controller for N-Queens.

    Follows the same pattern as QWMController (Sudoku) and LeanQWMController,
    but operates on NQueensBoard states using NQueensGraphEncoder and
    NQueensActionProposer.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        config: Any,
        n: int = 8,
    ) -> None:
        self.n = n
        self.models = models
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_encoder = models["state_encoder"]
        self.quotient_encoder = models["quotient_encoder"]
        self.world_model = models["world_model"]
        self.obstruction_predictor = models["obstruction_predictor"]
        self.value_head = models["value_head"]

        self._verifier = NQueensVerifier(n)
        self._encoder = NQueensGraphEncoder(n)
        self._proposer = NQueensActionProposer(n, self._verifier)

    def encode_state(self, state: NQueensBoard) -> torch.Tensor:
        from torch_geometric.data import Batch
        graph = self._encoder.encode(state)
        batch = Batch.from_data_list([graph]).to(self.device)
        h = self.state_encoder(batch)
        z = self.quotient_encoder(h)
        return z.squeeze(0)

    def search(
        self,
        initial_state: NQueensBoard,
        max_nodes: int = 500,
    ) -> Dict[str, Any]:
        """Run best-first QWM proof search.

        Returns
        -------
        dict with keys:
            solved, solution_board, nodes_expanded, merges_performed,
            obstructions_reused, verifier_calls
        """
        import heapq
        from qwm.search.dag import QuotientDAG

        if self._verifier.is_complete(initial_state):
            return dict(solved=True, solution_board=initial_state,
                        nodes_expanded=0, merges_performed=0,
                        obstructions_reused=0, verifier_calls=0)

        dag = QuotientDAG(self.config.merge_threshold)
        z0 = self.encode_state(initial_state)
        root_id = dag.add_root(z0, initial_state)

        heap: List[Tuple[float, int]] = []
        heapq.heappush(heap, (0.0, root_id))

        nodes_expanded = 0
        merges_performed = 0
        obstructions_reused = 0
        verifier_calls = 0
        step = 0

        while heap and nodes_expanded < max_nodes:
            _, nid = heapq.heappop(heap)
            node = dag.nodes[nid]
            if node.is_pruned or node.is_verified_solved:
                continue

            board: NQueensBoard = node.board_state
            actions = self._proposer.propose(board, top_k=self.n)

            for action in actions:
                if self._verifier.get_error_type(board, action) is not None:
                    continue
                try:
                    next_board = self._verifier.apply_action(board, action)
                except ValueError:
                    continue

                z_next = self.encode_state(next_board)
                value_score = float(self.value_head(z_next.unsqueeze(0)).item())
                obs_logits = self.obstruction_predictor(z_next.unsqueeze(0))
                _, confidence_t = torch.softmax(obs_logits, dim=-1).max(dim=-1)
                confidence = float(confidence_t.item())

                new_id, merged = dag.add_or_merge(
                    z_next, next_board, nid, value_score, confidence
                )
                merges_performed += int(merged)
                priority = -(value_score - 0.3 * confidence
                             + 0.2 * dag.nodes[new_id].merge_count)
                heapq.heappush(heap, (priority, new_id))

            nodes_expanded += 1
            step += 1

            if step % self.config.verify_every_n_steps == 0:
                frontier = dag.get_frontier()
                top_nodes = sorted(frontier, key=lambda nd: -nd.value_score)[:3]
                for nd in top_nodes:
                    verifier_calls += 1
                    if self._verifier.is_complete(nd.board_state):
                        dag.mark_solved(nd.node_id)
                        return dict(solved=True, solution_board=nd.board_state,
                                    nodes_expanded=nodes_expanded,
                                    merges_performed=merges_performed,
                                    obstructions_reused=obstructions_reused,
                                    verifier_calls=verifier_calls)

        return dict(solved=False, solution_board=None,
                    nodes_expanded=nodes_expanded,
                    merges_performed=merges_performed,
                    obstructions_reused=obstructions_reused,
                    verifier_calls=verifier_calls)

    def __repr__(self) -> str:
        return f"NQueensQWMController(n={self.n}, device={self.device})"
