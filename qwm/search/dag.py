"""QuotientDAG: dynamic DAG for merging equivalent Sudoku states during search.

Implements node addition, merging, pruning, and statistics for QWM search.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


def _copy_state(state: Any) -> Any:
    """Return a copy of *state* if it supports .copy(), otherwise return as-is.

    numpy arrays need .copy() to avoid aliasing.
    Immutable objects (ProofState, frozensets, etc.) are safe to share.
    """
    if hasattr(state, "copy"):
        return state.copy()
    return state

@dataclass
class DAGNode:
    """Node in the quotient-DAG representing a unique (possibly merged) state."""
    node_id: int
    z: torch.Tensor
    board_state: np.ndarray
    parent_ids: List[int] = field(default_factory=list)
    child_ids: List[int] = field(default_factory=list)
    value_score: float = 0.0
    obstruction_risk: float = 0.0
    is_pruned: bool = False
    is_verified_solved: bool = False
    is_verified_failed: bool = False
    merge_count: int = 0

    def __repr__(self) -> str:
        return (f"DAGNode(id={self.node_id}, pruned={self.is_pruned}, "
                f"solved={self.is_verified_solved}, merges={self.merge_count})")

class QuotientDAG:
    """Dynamic DAG for QWM search, merging nodes by cosine similarity."""
    def __init__(self, merge_threshold: float) -> None:
        self.merge_threshold = merge_threshold
        self.nodes: Dict[int, DAGNode] = {}
        self.next_id: int = 0
        self.frontier: List[int] = []
        self.total_merges: int = 0
        self.total_pruned: int = 0
        self.total_solved: int = 0

    def add_root(self, z: torch.Tensor, board_state: np.ndarray) -> int:
        node = DAGNode(
            node_id=self.next_id,
            z=z.detach().cpu(),
            board_state=_copy_state(board_state),
        )
        self.nodes[self.next_id] = node
        self.frontier.append(self.next_id)
        self.next_id += 1
        return node.node_id

    def add_or_merge(self, z: torch.Tensor, board_state: np.ndarray, parent_id: int, value_score: float, obstruction_risk: float) -> Tuple[int, bool]:
        z = z.detach().cpu()
        best_sim = -1.0
        best_id = None
        for nid, node in self.nodes.items():
            if node.is_pruned:
                continue
            sim = F.cosine_similarity(z.unsqueeze(0), node.z.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        if best_sim > self.merge_threshold:
            # Merge into existing node
            node = self.nodes[best_id]
            node.parent_ids.append(parent_id)
            node.merge_count += 1
            self.nodes[parent_id].child_ids.append(best_id)
            self.total_merges += 1
            return best_id, True
        # Add new node
        node = DAGNode(
            node_id=self.next_id,
            z=z,
            board_state=_copy_state(board_state),
            parent_ids=[parent_id],
            value_score=value_score,
            obstruction_risk=obstruction_risk,
        )
        self.nodes[self.next_id] = node
        self.nodes[parent_id].child_ids.append(self.next_id)
        self.frontier.append(self.next_id)
        self.next_id += 1
        return node.node_id, False

    def mark_pruned(self, node_id: int) -> None:
        """Mark node and all descendants as pruned."""
        stack = [node_id]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            if node.is_pruned:
                continue
            node.is_pruned = True
            self.total_pruned += 1
            stack.extend(node.child_ids)
        if node_id in self.frontier:
            self.frontier.remove(node_id)

    def mark_solved(self, node_id: int) -> None:
        """Mark node as solved and backpropagate success to parents."""
        node = self.nodes[node_id]
        node.is_verified_solved = True
        self.total_solved += 1
        # Optionally, propagate solved status up
        for pid in node.parent_ids:
            if not self.nodes[pid].is_verified_solved:
                self.mark_solved(pid)
        if node_id in self.frontier:
            self.frontier.remove(node_id)

    def get_frontier(self) -> List[DAGNode]:
        """Return all non-pruned, non-terminal leaf nodes."""
        return [self.nodes[nid] for nid in self.frontier if not self.nodes[nid].is_pruned and not self.nodes[nid].is_verified_solved and not self.nodes[nid].is_verified_failed]

    def get_stats(self) -> Dict[str, float]:
        total_nodes = len(self.nodes)
        pruned_nodes = sum(1 for n in self.nodes.values() if n.is_pruned)
        merged_nodes = self.total_merges
        solved_nodes = sum(1 for n in self.nodes.values() if n.is_verified_solved)
        avg_merge_count = float(np.mean([n.merge_count for n in self.nodes.values()])) if self.nodes else 0.0
        return dict(
            total_nodes=total_nodes,
            pruned_nodes=pruned_nodes,
            merged_nodes=merged_nodes,
            solved_nodes=solved_nodes,
            avg_merge_count=avg_merge_count,
        )

    def __repr__(self) -> str:
        return f"QuotientDAG(nodes={len(self.nodes)}, merges={self.total_merges}, pruned={self.total_pruned})"
