"""Abstract base protocols for QWM environments.

Any reasoning environment (Sudoku, Lean theorem proving, etc.) plugs into the
QWM search framework by implementing these three protocols:

    BaseVerifier        — exact oracle: validity, completeness, error type
    BaseGraphEncoder    — converts a state to a PyG graph for the StateEncoder
    BaseActionProposer  — generates candidate actions for a given state

The QWM controller and DAG are fully generic over these abstractions.
"""

from __future__ import annotations
from typing import Any, List, Optional, Protocol, runtime_checkable

from torch_geometric.data import Data


@runtime_checkable
class BaseVerifier(Protocol):
    """Exact checker for environment states and actions.

    Analogous to SudokuVerifier, but domain-agnostic.
    States and actions can be any Python object.
    """

    def is_complete(self, state: Any) -> bool:
        """Return True if *state* is a fully solved / proved goal."""
        ...

    def is_valid_action(self, state: Any, action: Any) -> bool:
        """Return True if *action* is legal in *state*."""
        ...

    def apply_action(self, state: Any, action: Any) -> Any:
        """Return the successor state after *action*, or raise ValueError."""
        ...

    def get_error_type(self, state: Any, action: Any) -> Optional[str]:
        """Return an obstruction-class string if the action is invalid, else None."""
        ...


@runtime_checkable
class BaseGraphEncoder(Protocol):
    """Converts a domain state into a PyG graph consumable by StateEncoder."""

    @property
    def node_feat_dim(self) -> int:
        """Feature dimension of each node (must match StateEncoder input dim)."""
        ...

    def encode(self, state: Any) -> Data:
        """Return a ``torch_geometric.data.Data`` with *x* and *edge_index*."""
        ...


@runtime_checkable
class BaseActionProposer(Protocol):
    """Proposes candidate actions for best-first search expansion."""

    def propose(self, state: Any, top_k: int = 5) -> List[Any]:
        """Return up to *top_k* candidate actions for *state*, ranked best-first."""
        ...
