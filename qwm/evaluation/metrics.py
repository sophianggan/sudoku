"""Evaluator for QWM Sudoku search — computes solve rate and efficiency metrics."""

from __future__ import annotations
from typing import Any, Dict, List, NamedTuple
import numpy as np

class EvalResult(NamedTuple):
    solve_rate: float
    avg_nodes_expanded: float
    avg_merges: float
    avg_verifier_calls: float
    avg_merge_rate: float
    avg_obstruction_reuse: float
    raw: List[Dict[str, Any]]

class Evaluator:
    """Batch evaluator for QWM Sudoku search."""
    def __init__(self) -> None:
        pass

    def evaluate_batch(self, puzzles: List[np.ndarray], controller) -> EvalResult:
        results = []
        for board in puzzles:
            res = controller.search(board)
            results.append(res)
        n = len(results)
        solve_rate = sum(1 for r in results if r["solved"]) / n
        avg_nodes_expanded = np.mean([r["nodes_expanded"] for r in results])
        avg_merges = np.mean([r["merges_performed"] for r in results])
        avg_verifier_calls = np.mean([r["verifier_calls"] for r in results])
        avg_merge_rate = avg_merges / max(avg_nodes_expanded, 1)
        avg_obstruction_reuse = np.mean([r["obstructions_reused"] for r in results])
        return EvalResult(
            solve_rate=solve_rate,
            avg_nodes_expanded=avg_nodes_expanded,
            avg_merges=avg_merges,
            avg_verifier_calls=avg_verifier_calls,
            avg_merge_rate=avg_merge_rate,
            avg_obstruction_reuse=avg_obstruction_reuse,
            raw=results,
        )

    def __repr__(self) -> str:
        return "Evaluator()"
