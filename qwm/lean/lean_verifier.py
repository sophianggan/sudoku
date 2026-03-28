"""LeanVerifier: QWM verifier backed by lean_dojo (or a mock for testing).

lean_dojo is an optional dependency.  When it is not installed, or when
``mock=True`` is passed, a lightweight mock verifier is used instead.

The mock verifier is useful for:
  - unit tests (no Lean install required)
  - prototyping the search loop before connecting to a real Lean server

Real usage (requires lean_dojo + Lean 4 + Mathlib)
---------------------------------------------------
    from lean_dojo import Dojo, Theorem, LeanGitRepo
    from qwm.lean.lean_verifier import LeanVerifier
    from qwm.lean.proof_state import ProofState

    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "your-commit-hash",
    )
    theorem = Theorem(repo, "Mathlib.Algebra.Group.Basic", "mul_comm")

    with Dojo(theorem) as (dojo, init_state):
        verifier = LeanVerifier(dojo)
        ps = ProofState.from_lean_dojo(init_state)
        result = verifier.apply_action(ps, "ring")

Mock usage (no Lean install)
----------------------------
    from qwm.lean.lean_verifier import LeanVerifier
    verifier = LeanVerifier(mock=True)
"""

from __future__ import annotations

from typing import Any, Optional

from qwm.lean.proof_state import Goal, Hypothesis, ProofState


# ─────────────────────────────────────────────────────────────────────────────
# Lean verifier
# ─────────────────────────────────────────────────────────────────────────────

class LeanVerifier:
    """Wraps lean_dojo to act as a QWM BaseVerifier for Lean proof states.

    Parameters
    ----------
    dojo:
        A ``lean_dojo.Dojo`` context (already entered with ``__enter__``).
        Pass ``None`` to run in mock mode.
    mock:
        If True, use a lightweight mock that never actually calls Lean.
        Useful for tests and debugging the search loop.
    """

    def __init__(self, dojo: Optional[Any] = None, *, mock: bool = False) -> None:
        if mock or dojo is None:
            self._impl: _BaseImpl = _MockImpl()
        else:
            self._impl = _LeanDojoImpl(dojo)

    # ── BaseVerifier protocol ─────────────────────────────────────────

    def is_complete(self, state: ProofState) -> bool:
        """True when there are no remaining goals."""
        return state.is_proved()

    def is_valid_action(self, state: ProofState, tactic: str) -> bool:
        """Return True if *tactic* does not raise an error in *state*."""
        if state.is_proved():
            return False
        try:
            self._impl.run_tactic(state, tactic)
            return True
        except TacticError:
            return False

    def apply_action(self, state: ProofState, tactic: str) -> ProofState:
        """Apply *tactic* to *state* and return the resulting ProofState.

        Raises ValueError if the tactic fails.
        """
        if state.is_proved():
            raise ValueError("Cannot apply tactic to a completed proof.")
        try:
            return self._impl.run_tactic(state, tactic)
        except TacticError as exc:
            raise ValueError(str(exc)) from exc

    def get_error_type(self, state: ProofState, tactic: str) -> Optional[str]:
        """Return an obstruction-class string if the tactic fails, else None."""
        if state.is_proved():
            return "proof_already_complete"
        try:
            self._impl.run_tactic(state, tactic)
            return None
        except TacticError as exc:
            return exc.error_type

    def __repr__(self) -> str:
        return f"LeanVerifier(impl={self._impl!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Internal error type
# ─────────────────────────────────────────────────────────────────────────────

class TacticError(Exception):
    """Raised when a Lean tactic fails."""

    def __init__(self, message: str, error_type: str = "tactic_error") -> None:
        super().__init__(message)
        self.error_type = error_type


# ─────────────────────────────────────────────────────────────────────────────
# Implementation back-ends
# ─────────────────────────────────────────────────────────────────────────────

class _BaseImpl:
    def run_tactic(self, state: ProofState, tactic: str) -> ProofState:
        raise NotImplementedError


class _LeanDojoImpl(_BaseImpl):
    """Calls the real lean_dojo Dojo to run a tactic."""

    def __init__(self, dojo: Any) -> None:
        self._dojo = dojo

    def run_tactic(self, state: ProofState, tactic: str) -> ProofState:
        try:
            from lean_dojo import ProofFinished, TacticError as LDTacticError
        except ImportError as exc:
            raise ImportError(
                "lean_dojo is not installed.  Run: pip install lean-dojo"
            ) from exc

        lean_state = state._lean_state
        if lean_state is None:
            raise TacticError(
                "ProofState has no attached lean_dojo TacticState.  "
                "Use ProofState.from_lean_dojo() to construct it.",
                error_type="missing_lean_state",
            )

        result = self._dojo.run_tac(lean_state, tactic)

        if isinstance(result, LDTacticError):
            raise TacticError(str(result), error_type="tactic_error")
        if isinstance(result, ProofFinished):
            return ProofState(goals=[])
        # result is a new TacticState
        return ProofState.from_lean_dojo(result)

    def __repr__(self) -> str:
        return "LeanDojoImpl()"


class _MockImpl(_BaseImpl):
    """Deterministic mock that simulates Lean tactic application.

    Behaviour:
    - ``ring`` / ``omega`` / ``simp`` / ``decide`` / ``norm_num`` / ``linarith``
      are treated as *closing tactics*: they succeed with probability 1 on any
      goal whose target is a simple arithmetic equality or inequality
      (heuristically detected by the presence of ``=``, ``<``, ``≤``, etc.).
    - ``intro <name>`` always succeeds, adding a new hypothesis.
    - ``apply <name>`` always succeeds if <name> matches a hypothesis name,
      closing the goal (simplified mock behaviour).
    - ``exact <name>`` succeeds if <name> matches a hypothesis whose type
      matches the target (very simplified check).
    - All other tactics fail with a generic error.

    This is intentionally simple — it exists to let the full search loop run
    end-to-end in tests without a real Lean installation.
    """

    _CLOSING_TACTICS = frozenset({
        "ring", "omega", "simp", "decide", "norm_num", "linarith",
        "ring_nf", "simp_all", "tauto", "trivial", "rfl",
    })
    _ARITH_PATTERNS = frozenset({"=", "≠", "<", "≤", ">", "≥", "+", "-", "*"})

    def run_tactic(self, state: ProofState, tactic: str) -> ProofState:
        tactic = tactic.strip()
        if not state.goals:
            raise TacticError("No goals.", error_type="no_goals")

        first_goal = state.goals[0]
        remaining = state.goals[1:]

        # ── Closing tactics ───────────────────────────────────────────
        tactic_name = tactic.split()[0].lower()
        if tactic_name in self._CLOSING_TACTICS:
            target = first_goal.target
            if any(op in target for op in self._ARITH_PATTERNS):
                return ProofState(goals=remaining)
            raise TacticError(
                f"Mock: '{tactic}' cannot close goal '{target}'",
                error_type="tactic_failed",
            )

        # ── intro <name> ──────────────────────────────────────────────
        if tactic_name == "intro":
            parts = tactic.split()
            name = parts[1] if len(parts) > 1 else "h"
            # Move leading ∀ binder into a hypothesis
            target = first_goal.target.strip()
            if target.startswith("∀"):
                # Heuristic: strip the ∀ and extract the binder type
                inner = target[1:].strip()
                colon_idx = inner.find(",")
                binder = inner[:colon_idx].strip() if colon_idx >= 0 else inner
                new_target = inner[colon_idx + 1:].strip() if colon_idx >= 0 else "True"
                # Binder may be "x : T" or just "x"
                if ":" in binder:
                    _, btype = binder.split(":", 1)
                    new_hyp = Hypothesis(name=name, type_str=btype.strip())
                else:
                    new_hyp = Hypothesis(name=name, type_str=binder.strip())
                new_goal = Goal(
                    hypotheses=first_goal.hypotheses + (new_hyp,),
                    target=new_target,
                )
                return ProofState(goals=[new_goal] + remaining)
            raise TacticError(
                f"Mock: 'intro' expects a ∀ goal, got '{target}'",
                error_type="tactic_failed",
            )

        # ── exact <name> ──────────────────────────────────────────────
        if tactic_name == "exact":
            parts = tactic.split()
            ref = parts[1] if len(parts) > 1 else ""
            hyp_names = {h.name for h in first_goal.hypotheses}
            if ref in hyp_names:
                return ProofState(goals=remaining)
            raise TacticError(
                f"Mock: unknown reference '{ref}'",
                error_type="unknown_identifier",
            )

        # ── apply <name> ─────────────────────────────────────────────
        if tactic_name == "apply":
            parts = tactic.split()
            ref = parts[1] if len(parts) > 1 else ""
            hyp_names = {h.name for h in first_goal.hypotheses}
            if ref in hyp_names:
                return ProofState(goals=remaining)
            raise TacticError(
                f"Mock: cannot apply unknown '{ref}'",
                error_type="unknown_identifier",
            )

        raise TacticError(
            f"Mock: unsupported tactic '{tactic}'",
            error_type="unsupported_tactic",
        )

    def __repr__(self) -> str:
        return "MockImpl()"
