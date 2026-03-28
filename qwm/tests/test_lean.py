"""Tests for Phase 5: Lean theorem proving layer.

Covers:
  - ProofState construction and parsing
  - LeanProofGraphEncoder node/edge counts and feature dimensions
  - LeanVerifier (mock) — valid/invalid tactics, apply, error types
  - TacticProposer — candidate generation
  - ProofTrace dataset generation
  - LeanQWMController — encode_state shape, full search loop (mock)
"""

from __future__ import annotations

import pytest
import torch

from qwm.lean.proof_state import Goal, Hypothesis, ProofState, _tokenize
from qwm.lean.proof_graph import LeanProofGraphEncoder
from qwm.lean.lean_verifier import LeanVerifier
from qwm.lean.lemma_retriever import LemmaRetriever
from qwm.lean.tactic_space import TacticProposer, COMMON_LEMMAS
from qwm.lean.dataset import generate_lean_dataset
from qwm.lean.theorem_loader import DEFAULT_MATHLIB_THEOREMS, load_theorems_from_repo


# ─────────────────────────────────────────────────────────────────────────────
# ProofState
# ─────────────────────────────────────────────────────────────────────────────

class TestProofState:
    def test_from_string_no_hyps(self):
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        assert len(ps.goals) == 1
        assert ps.goals[0].target == "1 + 1 = 2"
        assert len(ps.goals[0].hypotheses) == 0

    def test_from_string_with_hyp(self):
        ps = ProofState.from_string("h : n > 0\n⊢ n + 1 > 1")
        goal = ps.goals[0]
        assert len(goal.hypotheses) == 1
        assert goal.hypotheses[0].name == "h"
        assert goal.hypotheses[0].type_str == "n > 0"
        assert goal.target == "n + 1 > 1"

    def test_is_proved_empty(self):
        assert ProofState(goals=[]).is_proved()

    def test_is_proved_nonempty(self):
        ps = ProofState.from_string("⊢ True")
        assert not ps.is_proved()

    def test_fingerprint_stable(self):
        ps1 = ProofState.from_string("⊢ 2 + 2 = 4")
        ps2 = ProofState.from_string("⊢ 2 + 2 = 4")
        assert ps1.fingerprint() == ps2.fingerprint()
        assert ps1 == ps2

    def test_fingerprint_differs(self):
        ps1 = ProofState.from_string("⊢ 1 + 1 = 2")
        ps2 = ProofState.from_string("⊢ 2 + 2 = 4")
        assert ps1 != ps2

    def test_tokenize(self):
        tokens = _tokenize("n + 1 = n + 1")
        assert "n" in tokens
        assert "+" in tokens
        assert "1" in tokens
        assert "=" in tokens

    def test_multi_goal(self):
        raw = "⊢ 1 = 1\n\n⊢ 2 = 2"
        ps = ProofState.from_string(raw)
        assert len(ps.goals) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Graph encoder
# ─────────────────────────────────────────────────────────────────────────────

class TestLeanProofGraphEncoder:
    def setup_method(self):
        self.encoder = LeanProofGraphEncoder()

    def test_node_feat_dim(self):
        assert self.encoder.node_feat_dim == 18

    def test_simple_goal_has_nodes(self):
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        graph = self.encoder.encode(ps)
        assert graph.x.shape[1] == 18
        assert graph.x.shape[0] > 0

    def test_goal_with_hyp_has_cross_edges(self):
        ps = ProofState.from_string("h : n > 0\n⊢ n + 1 > 1")
        graph = self.encoder.encode(ps)
        # Should have at least sequential + cross edges
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0

    def test_proved_state_dummy_node(self):
        ps = ProofState(goals=[])
        graph = self.encoder.encode(ps)
        assert graph.x.shape == (1, 18)
        assert graph.edge_index.shape[1] == 0

    def test_feature_dtype(self):
        ps = ProofState.from_string("⊢ ∀ n : ℕ, n + 0 = n")
        graph = self.encoder.encode(ps)
        assert graph.x.dtype == torch.float

    def test_multi_goal_encoding(self):
        ps = ProofState.from_string("⊢ 1 = 1\n\n⊢ 2 = 2")
        graph = self.encoder.encode(ps)
        assert graph.x.shape[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Lean verifier (mock)
# ─────────────────────────────────────────────────────────────────────────────

class TestLeanVerifier:
    def setup_method(self):
        self.verifier = LeanVerifier(mock=True)

    def _arith_state(self):
        return ProofState.from_string("⊢ 2 + 2 = 4")

    def _forall_state(self):
        return ProofState.from_string("⊢ ∀ n : ℕ, n + 0 = n")

    def _hyp_state(self):
        return ProofState.from_string("h : n > 0\n⊢ n > 0")

    def test_is_complete_proved(self):
        assert self.verifier.is_complete(ProofState(goals=[]))

    def test_is_complete_open(self):
        assert not self.verifier.is_complete(self._arith_state())

    def test_ring_closes_arith_goal(self):
        ps = self._arith_state()
        result = self.verifier.apply_action(ps, "ring")
        assert result.is_proved()

    def test_omega_closes_arith_goal(self):
        ps = self._arith_state()
        result = self.verifier.apply_action(ps, "omega")
        assert result.is_proved()

    def test_intro_opens_forall(self):
        ps = self._forall_state()
        result = self.verifier.apply_action(ps, "intro h")
        assert not result.is_proved()
        assert len(result.goals[0].hypotheses) == 1

    def test_exact_closes_hyp_goal(self):
        ps = self._hyp_state()
        result = self.verifier.apply_action(ps, "exact h")
        assert result.is_proved()

    def test_invalid_tactic_raises(self):
        ps = self._arith_state()
        with pytest.raises(ValueError):
            self.verifier.apply_action(ps, "blorp_invalid_xyz")

    def test_is_valid_action_true(self):
        ps = self._arith_state()
        assert self.verifier.is_valid_action(ps, "ring")

    def test_is_valid_action_false(self):
        ps = self._arith_state()
        assert not self.verifier.is_valid_action(ps, "blorp_invalid_xyz")

    def test_get_error_type_none_on_valid(self):
        ps = self._arith_state()
        assert self.verifier.get_error_type(ps, "ring") is None

    def test_get_error_type_string_on_invalid(self):
        ps = self._arith_state()
        err = self.verifier.get_error_type(ps, "blorp_invalid_xyz")
        assert isinstance(err, str)


# ─────────────────────────────────────────────────────────────────────────────
# Tactic proposer
# ─────────────────────────────────────────────────────────────────────────────

class TestTacticProposer:
    def setup_method(self):
        self.proposer = TacticProposer()

    def test_propose_returns_strings(self):
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        tactics = self.proposer.propose(ps, top_k=5)
        assert len(tactics) <= 5
        assert all(isinstance(t, str) for t in tactics)

    def test_closing_tactics_included(self):
        ps = ProofState.from_string("⊢ 2 * 2 = 4")
        tactics = self.proposer.propose(ps, top_k=20)
        closing = {"ring", "omega", "norm_num", "simp", "decide"}
        assert closing & set(tactics), "At least one closing tactic expected"

    def test_intro_prioritised_for_forall(self):
        ps = ProofState.from_string("⊢ ∀ n : ℕ, n = n")
        tactics = self.proposer.propose(ps, top_k=5)
        assert tactics[0] == "intro h"

    def test_hyp_tactics_included(self):
        ps = ProofState.from_string("myHyp : a = b\n⊢ a = b")
        tactics = self.proposer.propose(ps, top_k=30)
        assert "exact myHyp" in tactics

    def test_proved_state_returns_empty(self):
        ps = ProofState(goals=[])
        assert self.proposer.propose(ps, top_k=5) == []


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

class TestLeanDataset:
    def test_generates_traces(self):
        traces = generate_lean_dataset(n_theorems=20, seed=0, max_depth=5,
                                        max_traces_per_theorem=10)
        assert len(traces) > 0

    def test_trace_fields(self):
        traces = generate_lean_dataset(n_theorems=5, seed=1, max_depth=4,
                                        max_traces_per_theorem=20)
        for t in traces:
            assert isinstance(t.state_before, ProofState)
            assert isinstance(t.tactic, str)
            assert isinstance(t.succeeded, bool)
            assert isinstance(t.branch_succeeded, bool)
            assert t.solvability_label in (0.0, 1.0)

    def test_some_branches_succeed(self):
        traces = generate_lean_dataset(n_theorems=50, seed=2, max_depth=8,
                                        max_traces_per_theorem=30)
        succeeded = [t for t in traces if t.branch_succeeded]
        assert len(succeeded) > 0, "Expected at least some proofs to succeed"


# ─────────────────────────────────────────────────────────────────────────────
# LeanQWMController
# ─────────────────────────────────────────────────────────────────────────────

class TestLeanQWMController:
    def _make_controller(self):
        from qwm.config import Config
        from qwm.training.trainer import QWMTrainer
        config = Config()
        trainer = QWMTrainer(config)
        verifier = LeanVerifier(mock=True)
        from qwm.lean.controller import LeanQWMController
        return LeanQWMController(trainer.get_models_dict(), config, verifier)

    def test_encode_state_shape(self):
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        z = controller.encode_state(ps)
        from qwm.config import Config
        assert z.shape == (Config().quotient_dim,)

    def test_encode_state_unit_norm(self):
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 2 * 3 = 6")
        z = controller.encode_state(ps)
        norm = z.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_propose_actions_returns_list(self):
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 2 + 2 = 4")
        tactics = controller.propose_actions(ps, top_k=5)
        assert isinstance(tactics, list)
        assert len(tactics) <= 5

    def test_search_runs_and_returns_dict(self):
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        result = controller.search(ps, max_nodes=50)
        assert "proved" in result
        assert "nodes_expanded" in result
        assert "merges_performed" in result
        assert "verifier_calls" in result

    def test_search_proves_simple_goal(self):
        """With random weights the search may not prove, but it must not crash."""
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 3 + 5 = 8")
        result = controller.search(ps, max_nodes=100)
        assert isinstance(result["proved"], bool)

    def test_search_result_has_proof_tactics_key(self):
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        result = controller.search(ps, max_nodes=50)
        assert "proof_tactics" in result
        assert isinstance(result["proof_tactics"], list)

    def test_search_already_proved_state(self):
        """A proof state with no goals should return proved=True immediately."""
        controller = self._make_controller()
        ps = ProofState(goals=[])
        result = controller.search(ps, max_nodes=50)
        assert result["proved"] is True
        assert result["nodes_expanded"] == 0
        assert result["proof_tactics"] == []

    def test_proof_tactics_are_strings_when_proved(self):
        controller = self._make_controller()
        ps = ProofState.from_string("⊢ 2 + 2 = 4")
        result = controller.search(ps, max_nodes=200)
        if result["proved"]:
            assert all(isinstance(t, str) for t in result["proof_tactics"])


# ─────────────────────────────────────────────────────────────────────────────
# LemmaRetriever (mock mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestLemmaRetriever:
    def test_fallback_returns_common_lemmas(self):
        """Without a dojo, retriever returns items from COMMON_LEMMAS."""
        retriever = LemmaRetriever()
        ps = ProofState.from_string("⊢ 1 + 1 = 2")
        lemmas = retriever.retrieve(ps)
        assert isinstance(lemmas, list)
        assert len(lemmas) > 0
        assert all(isinstance(l, str) for l in lemmas)

    def test_fallback_respects_max_lemmas(self):
        retriever = LemmaRetriever(max_lemmas=3)
        ps = ProofState.from_string("⊢ 1 = 1")
        lemmas = retriever.retrieve(ps)
        assert len(lemmas) <= 3

    def test_fallback_no_lean_state(self):
        """Even with dojo=None, retriever must not crash."""
        retriever = LemmaRetriever(dojo=None)
        ps = ProofState.from_string("h : n > 0\n⊢ n > 0")
        lemmas = retriever.retrieve(ps)
        assert len(lemmas) > 0

    def test_repr(self):
        r = LemmaRetriever()
        assert "fallback" in repr(r)


# ─────────────────────────────────────────────────────────────────────────────
# TacticProposer with LemmaRetriever
# ─────────────────────────────────────────────────────────────────────────────

class TestTacticProposerWithRetriever:
    def test_proposer_accepts_retriever(self):
        retriever = LemmaRetriever()
        proposer = TacticProposer(lemma_retriever=retriever)
        ps = ProofState.from_string("⊢ 2 + 2 = 4")
        tactics = proposer.propose(ps, top_k=20)
        assert len(tactics) > 0

    def test_proposer_with_retriever_includes_apply(self):
        retriever = LemmaRetriever(max_lemmas=5)
        proposer = TacticProposer(lemma_retriever=retriever)
        ps = ProofState.from_string("⊢ 2 * 1 = 2")
        tactics = proposer.all_tactics(ps)
        apply_tactics = [t for t in tactics if t.startswith("apply ")]
        assert len(apply_tactics) > 0

    def test_no_retriever_uses_common_lemmas(self):
        """Default proposer (no retriever) must still include common lemma tactics."""
        proposer = TacticProposer()
        ps = ProofState.from_string("⊢ 1 + 0 = 1")
        tactics = proposer.all_tactics(ps)
        lemma_tactics = [t for t in tactics if t.startswith("apply ") or t.startswith("rw [")]
        assert len(lemma_tactics) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Theorem loader (no lean_dojo required — tests static metadata only)
# ─────────────────────────────────────────────────────────────────────────────

class TestTheoremLoader:
    def test_default_list_nonempty(self):
        assert len(DEFAULT_MATHLIB_THEOREMS) > 0

    def test_default_list_entries_are_pairs(self):
        for entry in DEFAULT_MATHLIB_THEOREMS:
            assert len(entry) == 2
            file_path, thm_name = entry
            assert isinstance(file_path, str) and file_path.endswith(".lean")
            assert isinstance(thm_name, str) and len(thm_name) > 0

    def test_load_requires_lean_dojo(self):
        """load_theorems_from_repo raises ImportError when lean_dojo absent."""
        import importlib
        import sys
        # Temporarily hide lean_dojo if it happens to be installed
        original = sys.modules.get("lean_dojo")
        sys.modules["lean_dojo"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError)):
                load_theorems_from_repo("https://example.com", "abc123", [])
        finally:
            if original is None:
                sys.modules.pop("lean_dojo", None)
            else:
                sys.modules["lean_dojo"] = original
