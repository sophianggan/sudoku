"""Entry point for QWM Lean theorem proving (Phase 5).

Commands
--------
    python qwm/lean_main.py train
        Generate mock proof traces, train all QWM components, save checkpoint.

    python qwm/lean_main.py evaluate
        Load checkpoint, run QWM search on a batch of mock theorems, print metrics.

    python qwm/lean_main.py demo
        Interactive: enter a goal string, run proof search, print result.

    python qwm/lean_main.py demo_dojo <repo_url> <commit> <file> <theorem>
        Run proof search against a real Lean theorem via lean_dojo.
        Requires a Lean 4 installation and lean_dojo.

Run from the repo root:
    python qwm/lean_main.py train
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import List

import torch

from qwm.config import Config
from qwm.lean.dataset import generate_lean_dataset
from qwm.lean.lean_verifier import LeanVerifier
from qwm.lean.proof_state import ProofState
from qwm.lean.controller import LeanQWMController
from qwm.lean.tactic_space import TacticProposer
from qwm.training.trainer import QWMTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Adapter: convert ProofTrace → SolverTrace-compatible dict for QWMTrainer
# ─────────────────────────────────────────────────────────────────────────────

def _proof_traces_to_solver_traces(proof_traces):
    """Wrap ProofTrace objects so they satisfy the SolverTrace duck-type.

    QWMTrainer / QWMDataset expects objects with attributes:
        board_before, board_after, action, branch_failed, obstruction_type,
        solvability_label
    We map:
        board_before  → state_before (ProofState)
        board_after   → state_after  (ProofState or None)
        action        → (0, 0, 0)    dummy — replaced by tactic embedding
        branch_failed → not branch_succeeded
        obstruction_type → obstruction_type
        solvability_label → solvability_label
    """
    from qwm.lean.proof_graph import LeanProofGraphEncoder
    from qwm.data.obstruction_labeler import OBSTRUCTION_CLASSES

    encoder = LeanProofGraphEncoder()
    obs_index = {c: i for i, c in enumerate(OBSTRUCTION_CLASSES)}

    class _WrappedTrace:
        def __init__(self, pt):
            self.board_before = encoder.encode(pt.state_before)
            self.board_after = (
                encoder.encode(pt.state_after)
                if pt.state_after is not None
                else encoder.encode(pt.state_before)  # fallback: same graph
            )
            self.action = (0, 0, 0)  # placeholder
            self.branch_failed = not pt.branch_succeeded
            # Map Lean obstruction types to the 6 Sudoku classes as best-effort
            lean_obs = pt.obstruction_type or ""
            if "tactic_failed" in lean_obs or "unsupported" in lean_obs:
                self.obstruction_type = "empty_domain"
            elif "unknown_identifier" in lean_obs:
                self.obstruction_type = "naked_single_violation"
            else:
                self.obstruction_type = "row_conflict"  # default fallback
            self.solvability_label = pt.solvability_label

    return [_WrappedTrace(pt) for pt in proof_traces]


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train():
    config = Config()
    print("[QWM-Lean] Generating proof traces (mock)...")
    proof_traces = generate_lean_dataset(n_theorems=2000, seed=config.seed,
                                         max_depth=10, max_traces_per_theorem=50)
    print(f"[QWM-Lean] Total traces: {len(proof_traces)}")

    wrapped = _proof_traces_to_solver_traces(proof_traces)

    # Build a minimal PairDataset using only the wrapped traces
    # (no equivalence pairs for now — Phase 6 will add Lean-specific pairs)
    from qwm.training.dataset import QWMDataset
    from torch.utils.data import DataLoader

    # QWMDataset expects SolverTrace objects; use the wrapped ones
    dataset = QWMDataset(wrapped)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=QWMTrainer._collate,
    )

    print("[QWM-Lean] Training QWM on Lean traces...")
    trainer = QWMTrainer(config)
    trainer.train(n_epochs=30, dataloader=loader)

    ckpt_path = pathlib.Path("checkpoints/qwm_lean.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[QWM-Lean] Checkpoint saved to {ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():
    config = Config()
    ckpt_path = pathlib.Path("checkpoints/qwm_lean.pt")
    print(f"[QWM-Lean] Loading checkpoint from {ckpt_path}...")
    trainer = QWMTrainer(config)
    trainer.load_checkpoint(ckpt_path)

    verifier = LeanVerifier(mock=True)
    controller = LeanQWMController(trainer.get_models_dict(), config, verifier)

    import random
    from qwm.lean.dataset import _make_mock_theorems
    rng = random.Random(config.seed + 1)
    theorems = _make_mock_theorems(50, rng)

    n_proved = 0
    total_nodes = 0
    total_merges = 0
    total_verifier_calls = 0

    print("[QWM-Lean] Evaluating on 50 mock theorems...")
    for name, initial_state in theorems:
        result = controller.search(initial_state, max_nodes=200)
        if result["proved"]:
            n_proved += 1
        total_nodes += result["nodes_expanded"]
        total_merges += result["merges_performed"]
        total_verifier_calls += result["verifier_calls"]

    n = len(theorems)
    metrics = {
        "prove_rate": n_proved / n,
        "avg_nodes_expanded": total_nodes / n,
        "avg_merges": total_merges / n,
        "avg_verifier_calls": total_verifier_calls / n,
    }
    print("\n=== QWM Lean Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/lean_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[QWM-Lean] Results saved to results/lean_eval.json")


# ─────────────────────────────────────────────────────────────────────────────
# Demo (mock)
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    config = Config()
    ckpt_path = pathlib.Path("checkpoints/qwm_lean.pt")
    trainer = QWMTrainer(config)
    if ckpt_path.exists():
        trainer.load_checkpoint(ckpt_path)
        print(f"[QWM-Lean] Loaded checkpoint from {ckpt_path}")
    else:
        print("[QWM-Lean] No checkpoint found — using random weights.")

    verifier = LeanVerifier(mock=True)
    controller = LeanQWMController(trainer.get_models_dict(), config, verifier)

    print("Enter a Lean goal (e.g.  ⊢ 2 + 2 = 4   or multi-line with blank lines).")
    print("End input with Ctrl-D (Unix) or Ctrl-Z Enter (Windows).\n")
    try:
        lines = sys.stdin.read()
    except EOFError:
        lines = ""

    state = ProofState.from_string(lines.strip())
    if not state.goals:
        print("[QWM-Lean] No goals parsed.  Using default: ⊢ 1 + 1 = 2")
        state = ProofState.from_string("⊢ 1 + 1 = 2")

    print(f"\n[QWM-Lean] Initial state:\n{state}\n")
    print("[QWM-Lean] Running proof search...")
    result = controller.search(state, max_nodes=200)

    if result["proved"]:
        print("[QWM-Lean] Proof found!")
        print(f"  Nodes expanded:    {result['nodes_expanded']}")
        print(f"  Merges performed:  {result['merges_performed']}")
        print(f"  Verifier calls:    {result['verifier_calls']}")
    else:
        print("[QWM-Lean] No proof found within node budget.")
        print(f"  Nodes expanded:    {result['nodes_expanded']}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo with real lean_dojo
# ─────────────────────────────────────────────────────────────────────────────

def demo_dojo(repo_url: str, commit: str, file_path: str, theorem_name: str):
    """Prove a real Lean 4 theorem using lean_dojo + QWM search."""
    try:
        from lean_dojo import Dojo, Theorem, LeanGitRepo
    except ImportError:
        print("lean_dojo is not installed.  Run: pip install lean-dojo")
        sys.exit(1)

    config = Config()
    ckpt_path = pathlib.Path("checkpoints/qwm_lean.pt")
    trainer = QWMTrainer(config)
    if ckpt_path.exists():
        trainer.load_checkpoint(ckpt_path)

    repo = LeanGitRepo(repo_url, commit)
    theorem = Theorem(repo, file_path, theorem_name)

    print(f"[QWM-Lean] Proving {theorem_name} from {file_path}...")
    with Dojo(theorem) as (dojo, init_state):
        verifier = LeanVerifier(dojo=dojo)
        initial_ps = ProofState.from_lean_dojo(init_state)
        controller = LeanQWMController(trainer.get_models_dict(), config, verifier)
        result = controller.search(initial_ps, max_nodes=500)

    if result["proved"]:
        print(f"[QWM-Lean] Proved {theorem_name}!")
    else:
        print(f"[QWM-Lean] Could not prove {theorem_name} within budget.")
    print(f"  Nodes: {result['nodes_expanded']}  "
          f"Merges: {result['merges_performed']}  "
          f"Verifier calls: {result['verifier_calls']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qwm/lean_main.py [train|evaluate|demo|demo_dojo ...]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "demo":
        demo()
    elif cmd == "demo_dojo":
        if len(sys.argv) < 6:
            print("Usage: python qwm/lean_main.py demo_dojo <repo_url> <commit> <file> <theorem>")
            sys.exit(1)
        demo_dojo(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
