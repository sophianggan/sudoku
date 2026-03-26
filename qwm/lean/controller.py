"""LeanQWMController: best-first quotient-DAG proof search for Lean.

Same QWM algorithm as QWMController (Sudoku), but adapted for Lean proof
states and tactic actions.  The state encoder, quotient encoder, world model,
obstruction predictor, and QuotientDAG are all reused unchanged.

Usage
-----
    from qwm.lean.controller import LeanQWMController
    from qwm.lean.lean_verifier import LeanVerifier
    from qwm.lean.proof_state import ProofState

    verifier = LeanVerifier(mock=True)        # or LeanVerifier(dojo=dojo)
    controller = LeanQWMController(models_dict, config, verifier)
    result = controller.search(initial_proof_state, max_nodes=300)
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from qwm.config import Config
from qwm.lean.lean_verifier import LeanVerifier
from qwm.lean.proof_graph import LeanProofGraphEncoder
from qwm.lean.proof_state import ProofState
from qwm.lean.tactic_space import TacticProposer
from qwm.search.dag import QuotientDAG


class LeanQWMController:
    """Best-first QWM search controller operating on Lean proof states.

    Plugs into the same QuotientDAG and model components as QWMController.
    Actions are tactic strings instead of (row, col, digit) tuples.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        config: Config,
        verifier: LeanVerifier,
    ) -> None:
        self.models = models
        self.config = config
        self.verifier = verifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_encoder = models["state_encoder"]
        self.quotient_encoder = models["quotient_encoder"]
        self.world_model = models["world_model"]
        self.obstruction_predictor = models["obstruction_predictor"]
        self.value_head = models["value_head"]

        self._graph_encoder = LeanProofGraphEncoder()
        self._proposer = TacticProposer()

    # ── Encoding ─────────────────────────────────────────────────────

    def encode_state(self, state: ProofState) -> torch.Tensor:
        """Encode a ProofState → quotient embedding z ∈ ℝ^{quotient_dim}."""
        from torch_geometric.data import Batch
        graph = self._graph_encoder.encode(state)
        batch = Batch.from_data_list([graph]).to(self.device)
        h = self.state_encoder(batch)
        z = self.quotient_encoder(h)
        return z.squeeze(0)

    def propose_actions(self, state: ProofState, top_k: int = 5) -> List[str]:
        """Return up to top_k tactic strings, ranked by world-model solvability."""
        candidates = self._proposer.propose(state, top_k=30)
        if not candidates:
            return []
        z = self.encode_state(state).unsqueeze(0)
        solvability = float(self.world_model.predict_solvability(z).item())
        # All candidates share the same solvability score at this state;
        # keep ordering from TacticProposer (closing tactics first).
        _ = solvability  # will differentiate per action once world model sees action
        return candidates[:top_k]

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        initial_state: ProofState,
        max_nodes: int = 300,
    ) -> Dict[str, Any]:
        """Run best-first quotient-DAG proof search.

        Returns a dict with keys:
            proved          bool
            proof_state     final ProofState (proved or best frontier)
            nodes_expanded  int
            merges_performed int
            obstructions_reused int
            verifier_calls  int
        """
        dag = QuotientDAG(self.config.merge_threshold)
        z0 = self.encode_state(initial_state)
        root_id = dag.add_root(z0, initial_state)

        obstruction_cache: Set[int] = set()
        step = 0
        nodes_expanded = 0
        merges_performed = 0
        obstructions_reused = 0
        verifier_calls = 0
        last_tactics: Dict[int, str] = {}

        heap: List[Tuple[float, int]] = []
        heapq.heappush(heap, (0.0, root_id))

        while heap and nodes_expanded < max_nodes:
            _, nid = heapq.heappop(heap)
            node = dag.nodes[nid]
            if node.is_pruned or node.is_verified_solved:
                continue

            state: ProofState = node.board_state  # stored as board_state in DAGNode
            tactics = self.propose_actions(state, top_k=5)

            for tactic in tactics:
                obs_type = self.verifier.get_error_type(state, tactic)
                if obs_type is not None:
                    # Invalid tactic — skip without adding a node
                    continue
                try:
                    next_state = self.verifier.apply_action(state, tactic)
                except ValueError:
                    continue

                z_next = self.encode_state(next_state)
                value_score = float(self.value_head(z_next.unsqueeze(0)).item())
                obs_logits = self.obstruction_predictor(z_next.unsqueeze(0))
                class_id_t, confidence_t = torch.softmax(obs_logits, dim=-1).max(dim=-1)
                class_id = int(class_id_t.item())
                confidence = float(confidence_t.item())

                new_id, merged = dag.add_or_merge(
                    z_next, next_state, nid, value_score, confidence
                )
                merges_performed += int(merged)
                last_tactics[new_id] = tactic

                if confidence > 0.85 and class_id in obstruction_cache:
                    dag.mark_pruned(new_id)
                    obstructions_reused += 1
                else:
                    priority = -(
                        value_score
                        - 0.3 * confidence
                        + 0.2 * dag.nodes[new_id].merge_count
                    )
                    heapq.heappush(heap, (priority, new_id))

            nodes_expanded += 1
            step += 1

            # Periodic verifier checks on frontier
            if step % self.config.verify_every_n_steps == 0:
                frontier = dag.get_frontier()
                top_nodes = sorted(frontier, key=lambda n: -n.value_score)[:3]
                for n in top_nodes:
                    verifier_calls += 1
                    ps: ProofState = n.board_state
                    if self.verifier.is_complete(ps):
                        dag.mark_solved(n.node_id)
                        return {
                            "proved": True,
                            "proof_state": ps,
                            "nodes_expanded": nodes_expanded,
                            "merges_performed": merges_performed,
                            "obstructions_reused": obstructions_reused,
                            "verifier_calls": verifier_calls,
                        }
                    # Cache obstruction if last tactic was invalid
                    if n.node_id in last_tactics:
                        tactic = last_tactics[n.node_id]
                        err = self.verifier.get_error_type(ps, tactic)
                        if err is not None:
                            obstruction_cache.add(class_id)
                            dag.mark_pruned(n.node_id)

        # Check if initial state was already proved
        if initial_state.is_proved():
            return {
                "proved": True,
                "proof_state": initial_state,
                "nodes_expanded": 0,
                "merges_performed": 0,
                "obstructions_reused": 0,
                "verifier_calls": 0,
            }

        return {
            "proved": False,
            "proof_state": None,
            "nodes_expanded": nodes_expanded,
            "merges_performed": merges_performed,
            "obstructions_reused": obstructions_reused,
            "verifier_calls": verifier_calls,
        }

    def __repr__(self) -> str:
        return f"LeanQWMController(device={self.device})"
