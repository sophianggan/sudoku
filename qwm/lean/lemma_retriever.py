"""Dynamic lemma retrieval for the Lean TacticProposer.

When lean_dojo is available and a live Dojo context is provided, LemmaRetriever
queries the Lean environment for premises (lemmas, definitions) that are in scope
for the current proof state.  This supplements or replaces the hardcoded
COMMON_LEMMAS list in tactic_space.py.

When lean_dojo is not installed, or when no Dojo context is provided (e.g.
in mock mode or tests), the retriever falls back to the static COMMON_LEMMAS.

Usage — with real lean_dojo
----------------------------
    from lean_dojo import Dojo, Theorem
    from qwm.lean.lemma_retriever import LemmaRetriever

    with Dojo(theorem) as (dojo, init_state):
        retriever = LemmaRetriever(dojo=dojo)
        ps = ProofState.from_lean_dojo(init_state)
        lemmas = retriever.retrieve(ps)   # list of lemma name strings
        # e.g. ["mul_comm", "Nat.add_zero", "List.length_cons", ...]

Usage — mock / no Lean
-----------------------
    from qwm.lean.lemma_retriever import LemmaRetriever
    retriever = LemmaRetriever()
    lemmas = retriever.retrieve(any_proof_state)  # returns COMMON_LEMMAS
"""

from __future__ import annotations

from typing import Any, List, Optional

from qwm.lean.tactic_space import COMMON_LEMMAS


class LemmaRetriever:
    """Retrieves relevant lemma names for a given proof state.

    Parameters
    ----------
    dojo:
        An active ``lean_dojo.Dojo`` context (already entered via ``__enter__``).
        Pass ``None`` (default) to operate in fallback-only mode.
    max_lemmas:
        Maximum number of lemma names to return per call.  The lean_dojo
        premise list can be very long — capping avoids bloating the tactic
        proposal list.
    """

    def __init__(
        self,
        dojo: Optional[Any] = None,
        max_lemmas: int = 20,
    ) -> None:
        self._dojo = dojo
        self._max_lemmas = max_lemmas

    # ── Public API ───────────────────────────────────────────────────

    def retrieve(self, state: "ProofState") -> List[str]:  # noqa: F821
        """Return a list of lemma *names* relevant to *state*.

        Tries lean_dojo premise retrieval first; falls back to
        ``COMMON_LEMMAS`` on any failure (missing lean_state, API mismatch,
        timeout, etc.).
        """
        if self._dojo is not None and state._lean_state is not None:
            try:
                return self._retrieve_from_dojo(state)
            except Exception:
                pass
        return list(COMMON_LEMMAS[: self._max_lemmas])

    # ── Internal helpers ─────────────────────────────────────────────

    def _retrieve_from_dojo(self, state: "ProofState") -> List[str]:  # noqa: F821
        """Query lean_dojo for premises visible in the current tactic state.

        lean_dojo exposes premises via two possible APIs depending on version:
          * ``tactic_state.get_premises()``   (lean_dojo >= 1.7)
          * ``dojo.get_premises(tactic_state)``   (older versions)

        Both return a list of Premise objects with a ``full_name`` attribute.
        """
        lean_state = state._lean_state

        # Attempt 1: method on the tactic state itself (lean_dojo >= 1.7)
        if hasattr(lean_state, "get_premises"):
            premises = lean_state.get_premises()
            return self._names_from_premises(premises)

        # Attempt 2: method on the dojo object (older lean_dojo)
        if hasattr(self._dojo, "get_premises"):
            premises = self._dojo.get_premises(lean_state)
            return self._names_from_premises(premises)

        # No premise API found — fall back to static list
        return list(COMMON_LEMMAS[: self._max_lemmas])

    def _names_from_premises(self, premises: List[Any]) -> List[str]:
        """Extract string names from a list of lean_dojo Premise objects."""
        names: List[str] = []
        for p in premises:
            name = getattr(p, "full_name", None) or getattr(p, "name", None)
            if name and isinstance(name, str):
                names.append(name)
            if len(names) >= self._max_lemmas:
                break
        return names if names else list(COMMON_LEMMAS[: self._max_lemmas])

    def __repr__(self) -> str:
        mode = "dojo" if self._dojo is not None else "fallback"
        return f"LemmaRetriever(mode={mode}, max_lemmas={self._max_lemmas})"
