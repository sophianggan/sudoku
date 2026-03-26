# Quotient World Models (QWM) — Sudoku Prototype

Quotient World Models (QWM) is a novel AI reasoning framework for verifier-grounded recursive reasoning. QWM teaches an AI to solve hard reasoning problems (like Sudoku, theorem proving, and symbolic planning) more efficiently than standard search by learning to merge semantically equivalent states and recognize known failure patterns early.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r qwm/requirements.txt
```

## How to Run

- **Train:**
  ```bash
  python qwm/main.py train
  ```
- **Evaluate:**
  ```bash
  python qwm/main.py evaluate
  ```
- **Demo:**
  ```bash
  python qwm/main.py demo
  # Then enter a puzzle as an 81-char string (0=empty)
  ```

## Architecture Diagram

```
+-------------------+      +-------------------+      +-------------------+
|   State Encoder   |----->|  Quotient Encoder |----->|   World Model     |
|   (GNN, E_theta)  |      |   (MLP, Q_psi)    |      | (JEPA, M_phi)     |
+-------------------+      +-------------------+      +-------------------+
         |                          |                        |
         v                          v                        v
+-------------------+      +-------------------+      +-------------------+
|  Obstruction      |      |  Quotient-DAG     |      |   Verifier        |
|  Predictor        |      |  Controller       |      |   (Rule Checker)  |
|  (O_omega)        |      |  (Best-First)     |      |                   |
+-------------------+      +-------------------+      +-------------------+
```

## Metrics
- **solve_rate:** Fraction of test puzzles solved
- **avg_nodes_expanded:** Mean number of search nodes expanded
- **avg_merges:** Mean number of node merges performed
- **avg_verifier_calls:** Mean number of calls to the exact verifier
- **avg_merge_rate:** Fraction of nodes merged (merges / total nodes)
- **avg_obstruction_reuse:** How often cached obstructions pruned search

## Extending to Lean Theorem Proving (Stage 3 Roadmap)
- Swap out the Sudoku generator/verifier for Lean proof state generation and checking
- Implement Lean-specific graph encoders and action spaces
- Integrate with `lean_dojo` for formal proof search

---

For more details, see the code and comments in each module.
