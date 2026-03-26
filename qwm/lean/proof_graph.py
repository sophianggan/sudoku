"""Graph encoding for Lean proof states.

Converts a ProofState into a homogeneous PyG graph with 18-dimensional node
features — matching the ``node_feat_dim=18`` used by the Sudoku StateEncoder,
so the same GNN weights can be reused or fine-tuned for Lean.

Node encoding (18 dims)
-----------------------
Each token in the proof state becomes a node.

    dims 0-3   token-type one-hot  [keyword, operator, numeral, identifier]
    dims 4-6   role one-hot        [hyp-type token, hyp-name token, goal-target token]
    dim  7     position within expression (0→1, normalised by expr length)
    dim  8     bracket depth at this token (0→1, normalised by max observed)
    dim  9     token index within entire proof state (0→1)
    dim  10    goal index (0→1, normalised by n_goals)
    dim  11    is-first-in-expression  (0/1)
    dim  12    is-last-in-expression   (0/1)
    dims 13-17 5-bit vocabulary hash bucket (stable across runs)

Edge encoding
-------------
Two edge types are flattened into a single undirected edge list:
  - Sequential  : consecutive tokens within the same expression
  - Cross       : every (hyp-token, goal-target-token) pair within a goal
                  (represents "this hypothesis is available for this target")
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from qwm.lean.proof_state import (
    Goal, Hypothesis, ProofState,
    _LEAN_KEYWORDS, _LEAN_OPERATORS, _tokenize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Node feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _token_type_onehot(tok: str) -> List[float]:
    """Return 4-dim one-hot: [keyword, operator, numeral, identifier]."""
    if tok in _LEAN_KEYWORDS:
        return [1.0, 0.0, 0.0, 0.0]
    if tok in _LEAN_OPERATORS:
        return [0.0, 1.0, 0.0, 0.0]
    if re.fullmatch(r"-?\d+(\.\d+)?", tok):
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _vocab_hash(tok: str) -> List[float]:
    """Return a stable 5-bit bucket embedding via MD5."""
    digest = int(hashlib.md5(tok.encode()).hexdigest(), 16)
    return [(digest >> i) & 1 for i in range(5)]


import re as _re


def _bracket_depths(tokens: List[str]) -> List[int]:
    """Return bracket depth at each token position."""
    open_chars = {"(", "[", "{", "⟨"}
    close_chars = {")", "]", "}", "⟩"}
    depths: List[int] = []
    depth = 0
    for t in tokens:
        if t in open_chars:
            depth += 1
        depths.append(depth)
        if t in close_chars:
            depth = max(0, depth - 1)
    return depths


# ─────────────────────────────────────────────────────────────────────────────
# Main encoder
# ─────────────────────────────────────────────────────────────────────────────

class LeanProofGraphEncoder:
    """Encodes a ProofState as an 18-dim homogeneous PyG graph.

    Implements the BaseGraphEncoder protocol.
    """

    NODE_FEAT_DIM = 18

    @property
    def node_feat_dim(self) -> int:
        return self.NODE_FEAT_DIM

    def encode(self, state: ProofState) -> Data:
        """Return a PyG Data object for *state*."""
        if state.is_proved():
            # Empty graph for a completed proof — one dummy node.
            x = torch.zeros(1, self.NODE_FEAT_DIM)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            return Data(x=x, edge_index=edge_index)

        node_features: List[List[float]] = []
        edges_src: List[int] = []
        edges_dst: List[int] = []

        n_goals = len(state.goals)
        global_token_idx = 0          # token position across the full state
        # Count total tokens first for global normalisation
        total_tokens = sum(
            sum(len(_tokenize(h.type_str)) for h in g.hypotheses) +
            len(_tokenize(g.target))
            for g in state.goals
        )
        total_tokens = max(total_tokens, 1)

        for goal_idx, goal in enumerate(state.goals):
            goal_norm = goal_idx / max(n_goals - 1, 1) if n_goals > 1 else 0.0

            # ── Hypothesis expressions ───────────────────────────────────
            hyp_token_global_indices: List[int] = []  # for cross-edges
            for hyp in goal.hypotheses:
                # hyp name token (role = hyp-name)
                name_feat = _make_feat(
                    tok=hyp.name,
                    role=[0.0, 1.0, 0.0],
                    pos=0.0,
                    depth=0.0,
                    global_pos=global_token_idx / total_tokens,
                    goal_norm=goal_norm,
                    is_first=True,
                    is_last=False,
                )
                node_features.append(name_feat)
                hyp_token_global_indices.append(len(node_features) - 1)
                prev_idx = len(node_features) - 1
                global_token_idx += 1

                # hyp type tokens (role = hyp-type)
                type_toks = _tokenize(hyp.type_str)
                depths = _bracket_depths(type_toks)
                max_depth = max(depths) if depths else 1
                for tok_i, (tok, dep) in enumerate(zip(type_toks, depths)):
                    feat = _make_feat(
                        tok=tok,
                        role=[1.0, 0.0, 0.0],
                        pos=tok_i / max(len(type_toks) - 1, 1),
                        depth=dep / max(max_depth, 1),
                        global_pos=global_token_idx / total_tokens,
                        goal_norm=goal_norm,
                        is_first=(tok_i == 0),
                        is_last=(tok_i == len(type_toks) - 1),
                    )
                    node_features.append(feat)
                    cur_idx = len(node_features) - 1
                    hyp_token_global_indices.append(cur_idx)
                    # Sequential edge within hyp expression
                    edges_src.append(prev_idx); edges_dst.append(cur_idx)
                    edges_src.append(cur_idx);  edges_dst.append(prev_idx)
                    prev_idx = cur_idx
                    global_token_idx += 1

            # ── Goal target expression ───────────────────────────────────
            tgt_toks = _tokenize(goal.target)
            depths = _bracket_depths(tgt_toks)
            max_depth = max(depths) if depths else 1
            tgt_start = len(node_features)  # index of first target node
            prev_idx = tgt_start - 1 if tgt_start > 0 else None  # connect from last hyp

            for tok_i, (tok, dep) in enumerate(zip(tgt_toks, depths)):
                feat = _make_feat(
                    tok=tok,
                    role=[0.0, 0.0, 1.0],
                    pos=tok_i / max(len(tgt_toks) - 1, 1),
                    depth=dep / max(max_depth, 1),
                    global_pos=global_token_idx / total_tokens,
                    goal_norm=goal_norm,
                    is_first=(tok_i == 0),
                    is_last=(tok_i == len(tgt_toks) - 1),
                )
                node_features.append(feat)
                cur_idx = len(node_features) - 1
                # Sequential edge
                if prev_idx is not None:
                    edges_src.append(prev_idx); edges_dst.append(cur_idx)
                    edges_src.append(cur_idx);  edges_dst.append(prev_idx)
                prev_idx = cur_idx
                global_token_idx += 1

            # ── Cross edges: hyp tokens → target tokens ──────────────────
            tgt_indices = list(range(tgt_start, len(node_features)))
            for h_idx in hyp_token_global_indices:
                for t_idx in tgt_indices:
                    edges_src.append(h_idx); edges_dst.append(t_idx)
                    edges_src.append(t_idx); edges_dst.append(h_idx)

        # Fallback: if state had no tokens at all, add a dummy node
        if not node_features:
            node_features.append([0.0] * self.NODE_FEAT_DIM)

        x = torch.tensor(node_features, dtype=torch.float)
        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# Feature constructor
# ─────────────────────────────────────────────────────────────────────────────

def _make_feat(
    tok: str,
    role: List[float],        # 3-dim one-hot [hyp-type, hyp-name, goal-target]
    pos: float,               # position within expression (0→1)
    depth: float,             # bracket depth (0→1)
    global_pos: float,        # position in full proof state (0→1)
    goal_norm: float,         # goal index (0→1)
    is_first: bool,
    is_last: bool,
) -> List[float]:
    """Assemble one 18-dim node feature vector."""
    tok_type = _token_type_onehot(tok)           # 4 dims  (0-3)
    vocab_hash = _vocab_hash(tok)                # 5 dims  (13-17)
    return (
        tok_type                                 # 0-3
        + role                                   # 4-6
        + [pos, depth, global_pos, goal_norm,    # 7-10
           float(is_first), float(is_last)]      # 11-12
        + vocab_hash                             # 13-17
    )


def _token_type_onehot(tok: str) -> List[float]:
    """4-dim one-hot: [keyword, operator, numeral, identifier]."""
    if tok in _LEAN_KEYWORDS:
        return [1.0, 0.0, 0.0, 0.0]
    if tok in _LEAN_OPERATORS:
        return [0.0, 1.0, 0.0, 0.0]
    if _re.fullmatch(r"-?\d+(\.\d+)?", tok):
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]
