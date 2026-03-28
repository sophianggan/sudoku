"""Theorem loader for lean_dojo integration.

Provides utilities to load Lean 4 theorems from a git repository via lean_dojo.
Also ships a curated list of simple Mathlib 4 theorems suitable for bootstrapping
the QWM training pipeline without writing custom theorem lists.

Usage
-----
    # Load specific theorems from a custom repo:
    from qwm.lean.theorem_loader import load_theorems_from_repo
    theorems = load_theorems_from_repo(
        repo_url="https://github.com/leanprover-community/mathlib4",
        commit="v4.3.0",
        theorem_specs=[
            ("Mathlib/Algebra/Group/Basic.lean", "mul_comm"),
            ("Mathlib/Data/Nat/Basic.lean", "Nat.zero_add"),
        ],
    )

    # Load the default curated Mathlib set:
    from qwm.lean.theorem_loader import load_default_mathlib_theorems
    theorems = load_default_mathlib_theorems(max_theorems=10)
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Curated Mathlib 4 theorem list
# ─────────────────────────────────────────────────────────────────────────────

# Pinned to a stable lean_dojo-compatible Mathlib4 tag.
DEFAULT_MATHLIB_REPO_URL = "https://github.com/leanprover-community/mathlib4"
DEFAULT_MATHLIB_COMMIT = "v4.3.0"

# (file_path_in_repo, theorem_name) pairs.
# Chosen to be simple enough for a proof-of-concept training run:
# most are provable by ring / omega / simp / rfl in ≤ 3 tactics.
DEFAULT_MATHLIB_THEOREMS: List[Tuple[str, str]] = [
    # Algebra — group / ring basics
    ("Mathlib/Algebra/Group/Basic.lean",        "mul_comm"),
    ("Mathlib/Algebra/Group/Basic.lean",        "mul_assoc"),
    ("Mathlib/Algebra/Group/Basic.lean",        "add_comm"),
    ("Mathlib/Algebra/Group/Basic.lean",        "add_assoc"),
    ("Mathlib/Algebra/Group/Basic.lean",        "mul_one"),
    ("Mathlib/Algebra/Group/Basic.lean",        "one_mul"),
    ("Mathlib/Algebra/Group/Basic.lean",        "mul_zero"),
    ("Mathlib/Algebra/Group/Basic.lean",        "zero_mul"),
    ("Mathlib/Algebra/Group/Basic.lean",        "add_zero"),
    ("Mathlib/Algebra/Group/Basic.lean",        "zero_add"),
    # Ring / semiring
    ("Mathlib/Algebra/Ring/Basic.lean",         "add_mul"),
    ("Mathlib/Algebra/Ring/Basic.lean",         "mul_add"),
    # Natural numbers
    ("Mathlib/Data/Nat/Basic.lean",             "Nat.zero_add"),
    ("Mathlib/Data/Nat/Basic.lean",             "Nat.add_zero"),
    ("Mathlib/Data/Nat/Basic.lean",             "Nat.succ_pos"),
    ("Mathlib/Data/Nat/Basic.lean",             "Nat.add_succ"),
    ("Mathlib/Data/Nat/Basic.lean",             "Nat.succ_add"),
    ("Mathlib/Data/Nat/Defs.lean",              "Nat.pos_of_ne_zero"),
    # Lists
    ("Mathlib/Data/List/Basic.lean",            "List.length_nil"),
    ("Mathlib/Data/List/Basic.lean",            "List.length_cons"),
    ("Mathlib/Data/List/Basic.lean",            "List.nil_append"),
    ("Mathlib/Data/List/Basic.lean",            "List.append_nil"),
    # Order
    ("Mathlib/Order/Basic.lean",                "le_refl"),
    ("Mathlib/Order/Basic.lean",                "le_antisymm"),
    ("Mathlib/Order/Basic.lean",                "lt_irrefl"),
    # Tactic-friendly identities
    ("Mathlib/Tactic/Ring.lean",                "sq_nonneg"),
    ("Mathlib/Algebra/GroupPower/Basic.lean",   "pow_zero"),
    ("Mathlib/Algebra/GroupPower/Basic.lean",   "pow_one"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_theorems_from_repo(
    repo_url: str,
    commit: str,
    theorem_specs: List[Tuple[str, str]],
) -> List[Any]:
    """Load lean_dojo Theorem objects from a git repository.

    Parameters
    ----------
    repo_url:
        URL of the Lean git repository (e.g. Mathlib4 GitHub URL).
    commit:
        Git commit hash or tag identifying the exact revision to check out.
    theorem_specs:
        List of ``(file_path, theorem_name)`` pairs.  ``file_path`` should be
        relative to the repository root.

    Returns
    -------
    List of ``lean_dojo.Theorem`` objects for every spec that could be loaded.
    Specs that fail silently (e.g. wrong path or theorem name) are skipped.

    Raises
    ------
    ImportError
        When lean_dojo is not installed.
    """
    try:
        from lean_dojo import LeanGitRepo, Theorem
    except ImportError as exc:
        raise ImportError(
            "lean_dojo is required for load_theorems_from_repo. "
            "Install with: pip install lean-dojo"
        ) from exc

    repo = LeanGitRepo(repo_url, commit)
    theorems: List[Any] = []
    for file_path, theorem_name in theorem_specs:
        try:
            theorems.append(Theorem(repo, file_path, theorem_name))
        except Exception:
            # Skip theorems whose file or name cannot be resolved
            continue
    return theorems


def load_default_mathlib_theorems(
    repo_url: str = DEFAULT_MATHLIB_REPO_URL,
    commit: str = DEFAULT_MATHLIB_COMMIT,
    max_theorems: Optional[int] = None,
) -> List[Any]:
    """Load the curated set of simple Mathlib 4 theorems via lean_dojo.

    This is the recommended starting point for training QWM on real Lean
    theorems without manually specifying theorem names.

    Parameters
    ----------
    repo_url, commit:
        The Mathlib4 repository and commit to check out.  Override to pin
        to a different Lean / Mathlib version.
    max_theorems:
        Optional cap on the number of theorems to load.  Useful for quick
        smoke-test runs.

    Returns
    -------
    List of ``lean_dojo.Theorem`` objects (those that loaded successfully).
    """
    specs = DEFAULT_MATHLIB_THEOREMS
    if max_theorems is not None:
        specs = specs[:max_theorems]
    return load_theorems_from_repo(repo_url, commit, specs)
