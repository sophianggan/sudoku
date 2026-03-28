"""Tactic space definition for Lean theorem proving.

Defines the action vocabulary available to the QWM search controller when
operating on Lean proof states.  Tactics are split into two categories:

Closed tactics (no parameters)
    Terminal tactics that attempt to close the current goal entirely.
    Examples: ``ring``, ``omega``, ``simp``, ``decide``.

Parameterised tactics
    Tactics that take a term reference (hypothesis name or theorem name).
    Generated dynamically from the current proof state.
    Examples: ``intro h``, ``apply mul_comm``, ``exact h``, ``rw [h]``.

The ``TacticProposer`` class implements the ``BaseActionProposer`` protocol,
returning a ranked list of tactic strings for a given ProofState.
"""

from __future__ import annotations

import re
from typing import List, Optional, TYPE_CHECKING

from qwm.lean.proof_state import ProofState

if TYPE_CHECKING:
    from qwm.lean.lemma_retriever import LemmaRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Tactic vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Closing tactics: attempt to close the first goal with a single tactic.
CLOSING_TACTICS: List[str] = [
    "ring",
    "omega",
    "linarith",
    "norm_num",
    "decide",
    "simp",
    "simp_all",
    "tauto",
    "trivial",
    "rfl",
    "ring_nf",
    "norm_cast",
    "push_cast",
    "positivity",
    "field_simp",
    "aesop",
]

# Tactics that introduce or split the goal (structural).
STRUCTURAL_TACTICS: List[str] = [
    "intro h",
    "intros",
    "constructor",
    "left",
    "right",
    "cases h",
    "split",
    "use",
]

# Common Mathlib lemma names used as apply / rw arguments.
# This is a small fixed vocabulary — a real system would retrieve these
# from Lean's environment via lean_dojo or a lemma retriever.
COMMON_LEMMAS: List[str] = [
    "mul_comm",
    "add_comm",
    "mul_assoc",
    "add_assoc",
    "mul_zero",
    "zero_mul",
    "one_mul",
    "mul_one",
    "add_zero",
    "zero_add",
    "Nat.succ_pos",
    "Nat.lt_of_sub_eq_succ",
    "Nat.add_succ",
    "List.length_cons",
    "List.nil_append",
    "le_refl",
    "le_trans",
    "lt_irrefl",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tactic proposer
# ─────────────────────────────────────────────────────────────────────────────

class TacticProposer:
    """Generates candidate tactic strings for a ProofState.

    Implements the BaseActionProposer protocol.

    Parameters
    ----------
    lemma_retriever:
        Optional :class:`~qwm.lean.lemma_retriever.LemmaRetriever` instance.
        When provided, its :meth:`retrieve` method is called to obtain
        environment-specific lemma names that supplement or replace the
        hardcoded ``COMMON_LEMMAS`` list.  Pass ``None`` (default) to use
        the static list — suitable for mock / no-Lean environments.

    Priority order
    --------------
    1. Closing tactics (most likely to terminate search quickly)
    2. Parameterised tactics using current hypothesis names
    3. Intro if goal is a ∀
    4. Parameterised tactics using retrieved / common lemma names
    """

    def __init__(
        self,
        lemma_retriever: Optional["LemmaRetriever"] = None,
    ) -> None:
        self._lemma_retriever = lemma_retriever

    def propose(self, state: ProofState, top_k: int = 5) -> List[str]:
        """Return up to *top_k* candidate tactic strings for *state*."""
        if state.is_proved():
            return []

        tactics: List[str] = []
        first_goal = state.goals[0]
        hyp_names = [h.name for h in first_goal.hypotheses]
        target = first_goal.target

        # 1. Closing tactics — add all, filter later by world model
        tactics += CLOSING_TACTICS

        # 2. hypothesis-parameterised tactics
        for name in hyp_names:
            tactics.append(f"exact {name}")
            tactics.append(f"apply {name}")
            tactics.append(f"rw [{name}]")
            tactics.append(f"rw [← {name}]")
            tactics.append(f"cases {name}")

        # 3. Intro if goal starts with ∀
        if target.strip().startswith("∀"):
            tactics.insert(0, "intro h")  # high priority

        # 4. Lemma applications — dynamic retrieval when available, else static
        lemma_names = (
            self._lemma_retriever.retrieve(state)
            if self._lemma_retriever is not None
            else COMMON_LEMMAS
        )
        for lemma in lemma_names:
            tactics.append(f"apply {lemma}")
            tactics.append(f"rw [{lemma}]")

        # Deduplicate preserving order
        seen: set = set()
        unique: List[str] = []
        for t in tactics:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return unique[:top_k]

    def all_tactics(self, state: ProofState) -> List[str]:
        """Return the full candidate list without truncation."""
        return self.propose(state, top_k=10_000)
