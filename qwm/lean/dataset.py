"""Dataset generation for Lean proof search traces.

Analogous to ``qwm.data.sudoku_generator`` but for Lean theorem proving.
Each ``ProofTrace`` records the sequence of (state, tactic, next_state)
transitions produced by a solver (mock or real lean_dojo), together with
success/failure labels and obstruction types.

The generated traces feed into the same QWMDataset / PairDataset training
pipeline used for Sudoku.

Generating traces with mock verifier (no Lean install)
------------------------------------------------------
    from qwm.lean.dataset import generate_lean_dataset
    traces = generate_lean_dataset(n_theorems=500, seed=42)

Generating traces with real lean_dojo
--------------------------------------
    from lean_dojo import Dojo, Theorem, LeanGitRepo
    from qwm.lean.dataset import generate_lean_dataset_from_dojo
    traces = generate_lean_dataset_from_dojo(theorems_list, max_traces=1000)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qwm.lean.proof_state import Goal, Hypothesis, ProofState
from qwm.lean.lean_verifier import LeanVerifier, TacticError
from qwm.lean.tactic_space import TacticProposer


# ─────────────────────────────────────────────────────────────────────────────
# Trace data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProofTrace:
    """One step in a proof search trace.

    Mirrors SolverTrace from sudoku_generator.py.

    Attributes
    ----------
    state_before:
        The proof state before the tactic is applied.
    tactic:
        The tactic string that was tried.
    state_after:
        The resulting proof state (None if the tactic failed).
    succeeded:
        True when the tactic did not raise an error.
    branch_succeeded:
        True when this branch eventually reached a complete proof.
    obstruction_type:
        Obstruction class string (from LeanVerifier.get_error_type) or None.
    theorem_name:
        Identifier of the theorem being proved.
    depth:
        Depth of this step in the proof tree.
    """

    state_before: ProofState
    tactic: str
    state_after: Optional[ProofState]
    succeeded: bool
    branch_succeeded: bool
    obstruction_type: Optional[str]
    theorem_name: str
    depth: int = 0

    # Solvability label: 1.0 if branch succeeded, 0.0 otherwise.
    # Used as regression target for the world model solvability head.
    @property
    def solvability_label(self) -> float:
        return 1.0 if self.branch_succeeded else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Mock theorem bank
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_theorems(n: int, rng: random.Random) -> List[Tuple[str, ProofState]]:
    """Return *n* (name, initial_proof_state) pairs using mock goals.

    The goals are simple arithmetic facts that the mock LeanVerifier can
    handle with ``ring``, ``omega``, or ``intro`` + ``exact``.
    """
    templates = [
        # (theorem_name, goal_string)
        ("add_comm_{a}_{b}",       "⊢ {a} + {b} = {b} + {a}"),
        ("mul_pos_{a}_{b}",        "⊢ {a} * {b} > 0"),
        ("forall_add_zero_{n}",    "⊢ ∀ n : ℕ, n + 0 = n"),
        ("hyp_exact_{a}",          "h : {a} > 0\n⊢ {a} > 0"),
        ("add_zero_{a}",           "⊢ {a} + 0 = {a}"),
        ("le_refl_{a}",            "⊢ {a} ≤ {a}"),
        ("mul_one_{a}",            "⊢ {a} * 1 = {a}"),
        ("forall_le_refl",         "⊢ ∀ n : ℕ, n ≤ n"),
    ]
    results: List[Tuple[str, ProofState]] = []
    for i in range(n):
        tmpl_name, tmpl_goal = rng.choice(templates)
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        name = tmpl_name.format(a=a, b=b, n=a)
        goal_str = tmpl_goal.format(a=a, b=b, n=a)
        ps = ProofState.from_string(goal_str)
        ps.metadata["theorem_name"] = name
        results.append((name, ps))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Trace generator
# ─────────────────────────────────────────────────────────────────────────────

def _search_and_trace(
    name: str,
    initial_state: ProofState,
    verifier: LeanVerifier,
    proposer: TacticProposer,
    max_depth: int = 10,
    rng: Optional[random.Random] = None,
) -> List[ProofTrace]:
    """DFS proof search, recording every tactic attempt as a ProofTrace."""
    if rng is None:
        rng = random.Random(0)

    traces: List[ProofTrace] = []

    def dfs(state: ProofState, depth: int) -> bool:
        if state.is_proved():
            return True
        if depth >= max_depth:
            return False
        tactics = proposer.propose(state, top_k=8)
        rng.shuffle(tactics)
        for tactic in tactics:
            obs_type = verifier.get_error_type(state, tactic)
            succeeded = obs_type is None
            if succeeded:
                try:
                    next_state = verifier.apply_action(state, tactic)
                except ValueError:
                    succeeded = False
                    next_state = None
                    obs_type = "apply_error"
            else:
                next_state = None

            # Recurse to find out if this branch eventually succeeds
            branch_ok = False
            if succeeded and next_state is not None:
                branch_ok = dfs(next_state, depth + 1)

            traces.append(ProofTrace(
                state_before=state,
                tactic=tactic,
                state_after=next_state,
                succeeded=succeeded,
                branch_succeeded=branch_ok,
                obstruction_type=obs_type,
                theorem_name=name,
                depth=depth,
            ))

            if branch_ok:
                return True  # found a proof — propagate success upward

        return False

    dfs(initial_state, depth=0)
    # Second pass: mark branch_succeeded correctly
    # (DFS sets it only for branches that led to a proof; already done above)
    return traces


def generate_lean_dataset(
    n_theorems: int = 1000,
    seed: int = 42,
    max_depth: int = 10,
    max_traces_per_theorem: int = 100,
) -> List[ProofTrace]:
    """Generate proof search traces using mock theorems and the mock verifier.

    Parameters
    ----------
    n_theorems:
        Number of mock theorems to attempt proving.
    seed:
        Random seed for reproducibility.
    max_depth:
        Maximum proof search depth (DFS cutoff).
    max_traces_per_theorem:
        Cap on traces recorded per theorem (avoids huge datasets for hard goals).

    Returns
    -------
    List[ProofTrace]
        Flat list of all recorded tactic transitions.
    """
    rng = random.Random(seed)
    verifier = LeanVerifier(mock=True)
    proposer = TacticProposer()
    theorems = _make_mock_theorems(n_theorems, rng)

    all_traces: List[ProofTrace] = []
    for name, initial_state in theorems:
        traces = _search_and_trace(
            name, initial_state, verifier, proposer,
            max_depth=max_depth, rng=rng,
        )
        all_traces.extend(traces[:max_traces_per_theorem])

    return all_traces


def generate_lean_dataset_from_dojo(
    theorems: List[Any],
    max_traces_per_theorem: int = 200,
    max_depth: int = 15,
    seed: int = 42,
) -> List[ProofTrace]:
    """Generate proof traces using real lean_dojo Theorem objects.

    Parameters
    ----------
    theorems:
        List of ``lean_dojo.Theorem`` objects to prove.
    max_traces_per_theorem, max_depth, seed:
        Same semantics as generate_lean_dataset.

    Returns
    -------
    List[ProofTrace]
    """
    try:
        from lean_dojo import Dojo
    except ImportError as exc:
        raise ImportError(
            "lean_dojo is required for generate_lean_dataset_from_dojo.  "
            "Install it with: pip install lean-dojo"
        ) from exc

    rng = random.Random(seed)
    proposer = TacticProposer()
    all_traces: List[ProofTrace] = []

    for theorem in theorems:
        try:
            with Dojo(theorem) as (dojo, init_tactic_state):
                verifier = LeanVerifier(dojo=dojo)
                initial_state = ProofState.from_lean_dojo(init_tactic_state)
                traces = _search_and_trace(
                    theorem.full_name, initial_state, verifier, proposer,
                    max_depth=max_depth, rng=rng,
                )
                all_traces.extend(traces[:max_traces_per_theorem])
        except Exception:
            # Skip theorems that lean_dojo cannot load
            continue

    return all_traces
