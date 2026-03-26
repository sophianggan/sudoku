"""QWMController: best-first quotient-DAG search for Sudoku using learned models.

Implements encode_state, propose_actions, and the full search loop.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import heapq
import numpy as np
import torch
from qwm.config import Config
from qwm.search.dag import QuotientDAG
from qwm.search.verifier import SudokuVerifier
from qwm.data.sudoku_generator import get_candidate_set

class QWMController:
    """Best-first search controller for QWM quotient-DAG search."""
    def __init__(self, models: Dict[str, Any], config: Config, verifier: SudokuVerifier):
        self.models = models
        self.config = config
        self.verifier = verifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_encoder = models["state_encoder"]
        self.quotient_encoder = models["quotient_encoder"]
        self.world_model = models["world_model"]
        self.obstruction_predictor = models["obstruction_predictor"]
        self.value_head = models["value_head"]

    def encode_state(self, board: np.ndarray) -> torch.Tensor:
        """Run full encoding pipeline: board → graph → h_s → z_s."""
        from qwm.data.sudoku_graph import board_to_homogeneous_graph
        graph = board_to_homogeneous_graph(board)
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph]).to(self.device)
        h = self.state_encoder(batch)
        z = self.quotient_encoder(h)
        return z.squeeze(0)

    def propose_actions(self, board: np.ndarray, top_k: int = 5) -> List[Tuple[int, int, int]]:
        """Return top_k (row, col, digit) actions ranked by constraint and world model value."""
        cands = []
        for r in range(9):
            for c in range(9):
                if board[r, c] != 0:
                    continue
                cell_cands = get_candidate_set(board, r, c)
                for d in cell_cands:
                    cands.append((r, c, d, len(cell_cands)))
        # Sort by fewest candidates (MRV)
        cands.sort(key=lambda x: x[3])
        # Score with world model solvability head
        scored = []
        z = self.encode_state(board).unsqueeze(0)
        for (r, c, d, mrv) in cands[:30]:
            action = torch.tensor([[r * 9 + c, d]], dtype=torch.long, device=self.device)
            value = float(self.world_model.predict_solvability(z).item())
            scored.append(((r, c, d), mrv, value))
        # Sort by value, then MRV
        scored.sort(key=lambda x: (-x[2], x[1]))
        return [x[0] for x in scored[:top_k]]

    def search(self, initial_board: np.ndarray, max_nodes: int = 500) -> Any:
        """Run best-first quotient-DAG search. Returns SearchResult dict."""
        dag = QuotientDAG(self.config.merge_threshold)
        verifier = self.verifier
        z0 = self.encode_state(initial_board)
        root_id = dag.add_root(z0, initial_board)
        obstruction_cache: Set[int] = set()
        step = 0
        nodes_expanded = 0
        merges_performed = 0
        obstructions_reused = 0
        verifier_calls = 0
        last_actions: Dict[int, Tuple[int, int, int]] = {}
        # Priority queue: (priority, node_id)
        heap: List[Tuple[float, int]] = []
        heapq.heappush(heap, (0.0, root_id))
        while heap and nodes_expanded < max_nodes:
            _, nid = heapq.heappop(heap)
            node = dag.nodes[nid]
            if node.is_pruned or node.is_verified_solved:
                continue
            board = node.board_state
            actions = self.propose_actions(board, top_k=5)
            for (r, c, d) in actions:
                try:
                    next_board = verifier.apply_action(board, r, c, d)
                except ValueError:
                    continue
                z_next = self.encode_state(next_board)
                value_score = float(self.value_head(z_next.unsqueeze(0)).item())
                obs_logits = self.obstruction_predictor(z_next.unsqueeze(0))
                class_id, confidence = torch.softmax(obs_logits, dim=-1).max(dim=-1)
                class_id = int(class_id.item())
                confidence = float(confidence.item())
                new_id, merged = dag.add_or_merge(z_next, next_board, nid, value_score, float(confidence))
                merges_performed += int(merged)
                last_actions[new_id] = (r, c, d)
                if confidence > 0.85 and class_id in obstruction_cache:
                    dag.mark_pruned(new_id)
                    obstructions_reused += 1
                else:
                    heapq.heappush(heap, (-(value_score - 0.3 * confidence + 0.2 * dag.nodes[new_id].merge_count), new_id))
            nodes_expanded += 1
            step += 1
            # OUTER LOOP: verifier checks
            if step % self.config.verify_every_n_steps == 0:
                frontier = dag.get_frontier()
                top_nodes = sorted(frontier, key=lambda n: -n.value_score)[:3]
                for n in top_nodes:
                    verifier_calls += 1
                    if verifier.is_complete(n.board_state):
                        dag.mark_solved(n.node_id)
                        return dict(solved=True, solution_board=n.board_state, nodes_expanded=nodes_expanded, merges_performed=merges_performed, obstructions_reused=obstructions_reused, verifier_calls=verifier_calls)
                    # Check last action
                    if n.node_id in last_actions:
                        r, c, d = last_actions[n.node_id]
                        if not verifier.is_valid_action(n.board_state, r, c, d):
                            err_type = verifier.get_error_type(n.board_state, r, c, d)
                            if err_type is not None:
                                obstruction_cache.add(class_id)
                                dag.mark_pruned(n.node_id)
        return dict(solved=False, solution_board=None, nodes_expanded=nodes_expanded, merges_performed=merges_performed, obstructions_reused=obstructions_reused, verifier_calls=verifier_calls)

    def __repr__(self) -> str:
        return f"QWMController(device={self.device})"
