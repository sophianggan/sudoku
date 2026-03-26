"""Lean proof state representation.

A ProofState mirrors what lean_dojo exposes as a TacticState: a list of
open goals, each goal having zero or more typed hypotheses and a target
expression.  The class is self-contained (no lean_dojo import required) so
it can be constructed from serialised data or from lean_dojo live objects.

Usage with lean_dojo
--------------------
    from lean_dojo import Dojo, Theorem
    from qwm.lean.proof_state import ProofState

    with Dojo(theorem) as (dojo, init_state):
        ps = ProofState.from_lean_dojo(init_state)

Usage with mock data (no Lean install needed)
---------------------------------------------
    from qwm.lean.proof_state import Goal, Hypothesis, ProofState

    ps = ProofState(goals=[
        Goal(
            hypotheses=[Hypothesis("h", "n > 0")],
            target="n + 1 > 1",
        )
    ])
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hypothesis:
    """A single named hypothesis in a Lean proof goal."""

    name: str          # e.g. "h", "h1", "ih"
    type_str: str      # e.g. "n > 0", "List α", "∀ x, f x = x"

    def __str__(self) -> str:
        return f"{self.name} : {self.type_str}"


@dataclass(frozen=True)
class Goal:
    """One open goal: a list of hypotheses plus a target to prove."""

    hypotheses: tuple[Hypothesis, ...]  # ordered, possibly empty
    target: str                         # the ⊢ expression

    def __str__(self) -> str:
        lines = [str(h) for h in self.hypotheses] + [f"⊢ {self.target}"]
        return "\n".join(lines)

    @property
    def all_tokens(self) -> List[str]:
        """Flat list of all tokens across hypotheses and target."""
        tokens: List[str] = []
        for h in self.hypotheses:
            tokens += _tokenize(h.type_str)
        tokens += _tokenize(self.target)
        return tokens


@dataclass
class ProofState:
    """The full tactic state of a Lean proof: zero or more open goals.

    ``goals == []`` means the proof is complete.
    """

    goals: List[Goal]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional back-reference to the live lean_dojo TacticState object.
    # Not serialisable; set to None when constructing from data.
    _lean_state: Optional[Any] = field(default=None, compare=False, repr=False)

    # ── Derived properties ────────────────────────────────────────────

    def is_proved(self) -> bool:
        """True when there are no remaining goals."""
        return len(self.goals) == 0

    @property
    def n_goals(self) -> int:
        return len(self.goals)

    def fingerprint(self) -> str:
        """Stable string fingerprint used for equivalence checking."""
        return "\n---\n".join(str(g) for g in self.goals)

    def __hash__(self) -> int:
        return int(hashlib.md5(self.fingerprint().encode()).hexdigest(), 16)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProofState):
            return NotImplemented
        return self.fingerprint() == other.fingerprint()

    def __str__(self) -> str:
        if self.is_proved():
            return "<proof complete>"
        parts = [f"Goal {i+1}:\n{g}" for i, g in enumerate(self.goals)]
        return "\n\n".join(parts)

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_lean_dojo(cls, tactic_state: Any) -> "ProofState":
        """Build a ProofState from a lean_dojo TacticState object.

        lean_dojo TacticState exposes a ``goals`` attribute that is a list of
        strings, each formatted as::

            hyp1 : type1
            hyp2 : type2
            ⊢ target

        or just ``⊢ target`` when there are no hypotheses.
        """
        parsed_goals: List[Goal] = []
        for goal_str in tactic_state.goals:
            parsed_goals.append(_parse_goal_string(goal_str))
        return cls(goals=parsed_goals, _lean_state=tactic_state)

    @classmethod
    def from_string(cls, raw: str) -> "ProofState":
        """Parse a multi-goal tactic state from a plain string.

        Goals are separated by blank lines.  Each goal follows the same
        ``hyp : type … ⊢ target`` format that lean_dojo uses.
        """
        raw = raw.strip()
        if not raw:
            return cls(goals=[])
        blocks = re.split(r"\n\s*\n", raw)
        goals = [_parse_goal_string(b) for b in blocks if b.strip()]
        return cls(goals=goals)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Lean keywords that appear in tactic states
_LEAN_KEYWORDS = frozenset({
    "fun", "let", "have", "show", "calc", "by", "intro", "intros",
    "apply", "exact", "rw", "rewrite", "simp", "ring", "omega",
    "linarith", "norm_num", "decide", "match", "cases", "induction",
    "at", "with", "in", "if", "then", "else",
    "∀", "∃", "λ", "Prop", "Type", "Sort",
    "def", "theorem", "lemma", "example", "where",
    "variable", "namespace", "section", "end",
})

# Lean operators / punctuation
_LEAN_OPERATORS = frozenset({
    "+", "-", "*", "/", "=", "≠", "<", "≤", ">", "≥",
    "→", "↔", "∧", "∨", "¬", "⊢", ":", ":=", "=>", "<|>",
    "(", ")", "[", "]", "{", "}", "⟨", "⟩", ",", ".", "..",
    "@", "#", "!", "?", "^", "&", "|", "~",
})

# Regex that splits a lean expression into tokens
_TOKEN_RE = re.compile(
    r"(?:"
    r"[∀∃λ→↔∧∨¬⊢≠≤≥⟨⟩]"      # unicode math operators (single char each)
    r"|:=|\.\.|\<\|>"             # multi-char operators
    r"|[+\-*/=<>:,()\[\]{}.@#!?^&|~]"  # ASCII operators
    r"|[^\s+\-*/=<>:,()\[\]{}.@#!?^&|~∀∃λ→↔∧∨¬⊢≠≤≥⟨⟩]+"  # identifiers / numerals
    r")"
)


def _tokenize(expr: str) -> List[str]:
    """Split a lean expression string into a flat token list."""
    return _TOKEN_RE.findall(expr)


def _parse_goal_string(goal_str: str) -> Goal:
    """Parse a single goal block (lean_dojo format) into a Goal dataclass."""
    lines = [l for l in goal_str.strip().splitlines() if l.strip()]
    hypotheses: List[Hypothesis] = []
    target: str = ""

    for line in lines:
        line = line.strip()
        if line.startswith("⊢"):
            target = line[1:].strip()
        else:
            # Hypothesis line: "name : type" (colon is the separator)
            # Be careful: the type may itself contain colons
            match = re.match(r"^(\S+)\s*:\s*(.+)$", line)
            if match:
                hypotheses.append(Hypothesis(name=match.group(1),
                                             type_str=match.group(2).strip()))
            else:
                # Fallback: treat whole line as part of the target
                target = target + " " + line if target else line

    return Goal(hypotheses=tuple(hypotheses), target=target)
