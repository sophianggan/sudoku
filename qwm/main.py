"""Main entry point for QWM Sudoku prototype: train, evaluate, demo."""

from __future__ import annotations
import sys
import pathlib
import json
import numpy as np
from qwm.config import Config
from qwm.data.sudoku_generator import generate_dataset, board_from_string
from qwm.data.equivalence_labeler import generate_equivalent_pairs
from qwm.training.dataset import QWMDataset, PairDataset
from qwm.training.trainer import QWMTrainer
from qwm.search.verifier import SudokuVerifier
from qwm.search.controller import QWMController
from qwm.evaluation.metrics import Evaluator

# ────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────
def train():
    config = Config()
    print("[QWM] Generating Sudoku training data...")
    dataset = generate_dataset(n_puzzles=10000, seed=config.seed, max_traces_per_puzzle=200)
    all_traces = [trace for _, traces in dataset for trace in traces]
    print(f"[QWM] Total traces: {len(all_traces)}")
    print("[QWM] Generating equivalence pairs...")
    pairs = generate_equivalent_pairs(all_traces, n_pairs=5000, seed=config.seed)
    print(f"[QWM] Pairs: {len(pairs)}")
    print("[QWM] Building datasets...")
    qwmds = QWMDataset(all_traces)
    pairds = PairDataset(all_traces, pairs, n_neg=4)
    print("[QWM] Initializing trainer...")
    trainer = QWMTrainer(config)
    from torch.utils.data import DataLoader
    loader = DataLoader(pairds, batch_size=config.batch_size, shuffle=True, collate_fn=QWMTrainer._collate)
    print("[QWM] Training...")
    trainer.train(n_epochs=50, dataloader=loader)
    ckpt_path = pathlib.Path("checkpoints/qwm_sudoku.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[QWM] Checkpoint saved to {ckpt_path}")

# ────────────────────────────────────────────────────────────────────
# Evaluate
# ────────────────────────────────────────────────────────────────────
def evaluate():
    config = Config()
    print("[QWM] Loading checkpoint...")
    trainer = QWMTrainer(config)
    ckpt_path = pathlib.Path("checkpoints/qwm_sudoku.pt")
    trainer.load_checkpoint(ckpt_path)
    verifier = SudokuVerifier()
    print("[QWM] Generating test puzzles...")
    from qwm.data.sudoku_generator import generate_puzzle
    rng = np.random.RandomState(config.seed + 123)
    puzzles = [generate_puzzle(rng, n_clues=22)[0] for _ in range(100)]
    evaluator = Evaluator()
    # Full QWM
    print("[QWM] Evaluating full QWM...")
    controller = QWMController(trainer.get_models_dict(), config, verifier)
    res_full = evaluator.evaluate_batch(puzzles, controller)
    # Ablation 1: No quotient merging
    print("[QWM] Evaluating ablation: no quotient merging...")
    class NoMergeDAG(QWMController):
        def encode_state(self, board):
            return super().encode_state(board)
        def search(self, initial_board, max_nodes=500):
            # Patch QuotientDAG to never merge
            from qwm.search.dag import QuotientDAG
            dag = QuotientDAG(merge_threshold=2.0)  # never merge
            verifier = self.verifier
            z0 = self.encode_state(initial_board)
            root_id = dag.add_root(z0, initial_board)
            step = 0
            nodes_expanded = 0
            merges_performed = 0
            obstructions_reused = 0
            verifier_calls = 0
            last_actions = {}
            import heapq
            heap = []
            heapq.heappush(heap, (0.0, root_id))
            while heap and nodes_expanded < max_nodes:
                _, nid = heapq.heappop(heap)
                node = dag.nodes[nid]
                if node.is_pruned or node.is_verified_solved:
                    continue
                board = node.board_state
                actions = self.propose_actions(board, top_k=5)
                for (r, c, d) in actions:
                    try:
                        next_board = verifier.apply_action(board, r, c, d)
                    except ValueError:
                        continue
                    z_next = self.encode_state(next_board)
                    value_score = float(self.value_head(z_next.unsqueeze(0)).item())
                    obs_logits = self.obstruction_predictor(z_next.unsqueeze(0))
                    class_id, confidence = torch.softmax(obs_logits, dim=-1).max(dim=-1)
                    class_id = int(class_id.item())
                    confidence = float(confidence.item())
                    new_id, merged = dag.add_or_merge(z_next, next_board, nid, value_score, float(confidence))
                    merges_performed += int(merged)
                    last_actions[new_id] = (r, c, d)
                    heapq.heappush(heap, (-(value_score - 0.3 * confidence + 0.2 * dag.nodes[new_id].merge_count), new_id))
                nodes_expanded += 1
                step += 1
                if step % self.config.verify_every_n_steps == 0:
                    frontier = dag.get_frontier()
                    top_nodes = sorted(frontier, key=lambda n: -n.value_score)[:3]
                    for n in top_nodes:
                        verifier_calls += 1
                        if verifier.is_complete(n.board_state):
                            dag.mark_solved(n.node_id)
                            return dict(solved=True, solution_board=n.board_state, nodes_expanded=nodes_expanded, merges_performed=merges_performed, obstructions_reused=obstructions_reused, verifier_calls=verifier_calls)
            return dict(solved=False, solution_board=None, nodes_expanded=nodes_expanded, merges_performed=merges_performed, obstructions_reused=obstructions_reused, verifier_calls=verifier_calls)
    res_nomerge = evaluator.evaluate_batch(puzzles, NoMergeDAG(trainer.get_models_dict(), config, verifier))
    # Ablation 2: No obstruction prediction
    print("[QWM] Evaluating ablation: no obstruction prediction...")
    class NoObstructionController(QWMController):
        def search(self, initial_board, max_nodes=500):
            # Patch to never prune by obstruction
            dag = QuotientDAG(self.config.merge_threshold)
            verifier = self.verifier
            z0 = self.encode_state(initial_board)
            root_id = dag.add_root(z0, initial_board)
            step = 0
            nodes_expanded = 0
            merges_performed = 0
            obstructions_reused = 0
            verifier_calls = 0
            last_actions = {}
            import heapq
            heap = []
            heapq.heappush(heap, (0.0, root_id))
            while heap and nodes_expanded < max_nodes:
                _, nid = heapq.heappop(heap)
                node = dag.nodes[nid]
                if node.is_pruned or node.is_verified_solved:
                    continue
                board = node.board_state
                actions = self.propose_actions(board, top_k=5)
                for (r, c, d) in actions:
                    try:
                        next_board = verifier.apply_action(board, r, c, d)
                    except ValueError:
                        continue
                    z_next = self.encode_state(next_board)
                    value_score = float(self.value_head(z_next.unsqueeze(0)).item())
                    obs_logits = self.obstruction_predictor(z_next.unsqueeze(0))
                    class_id, confidence = torch.softmax(obs_logits, dim=-1).max(dim=-1)
                    class_id = int(class_id.item())
                    confidence = float(confidence.item())
                    new_id, merged = dag.add_or_merge(z_next, next_board, nid, value_score, float(confidence))
                    merges_performed += int(merged)
                    last_actions[new_id] = (r, c, d)
                    heapq.heappush(heap, (-(value_score - 0.3 * confidence + 0.2 * dag.nodes[new_id].merge_count), new_id))
                nodes_expanded += 1
                step += 1
                if step % self.config.verify_every_n_steps == 0:
                    frontier = dag.get_frontier()
                    top_nodes = sorted(frontier, key=lambda n: -n.value_score)[:3]
                    for n in top_nodes:
                        verifier_calls += 1
                        if verifier.is_complete(n.board_state):
                            dag.mark_solved(n.node_id)
                            return dict(solved=True, solution_board=n.board_state, nodes_expanded=nodes_expanded, merges_performed=merges_performed, obstructions_reused=obstructions_reused, verifier_calls=verifier_calls)
            return dict(solved=False, solution_board=None, nodes_expanded=nodes_expanded, merges_performed=merges_performed, obstructions_reused=obstructions_reused, verifier_calls=verifier_calls)
    res_noobs = evaluator.evaluate_batch(puzzles, NoObstructionController(trainer.get_models_dict(), config, verifier))
    # Ablation 3: No world model
    print("[QWM] Evaluating ablation: no world model...")
    class RandomActionController(QWMController):
        def propose_actions(self, board, top_k=5):
            cands = []
            for r in range(9):
                for c in range(9):
                    if board[r, c] != 0:
                        continue
                    cell_cands = get_candidate_set(board, r, c)
                    for d in cell_cands:
                        cands.append((r, c, d))
            import random
            random.shuffle(cands)
            return cands[:top_k]
    res_nowm = evaluator.evaluate_batch(puzzles, RandomActionController(trainer.get_models_dict(), config, verifier))
    # Print results
    print("\n=== QWM Sudoku Evaluation Results ===")
    print(f"Full QWM:         Solve rate: {res_full.solve_rate:.2f}  Avg nodes: {res_full.avg_nodes_expanded:.1f}  Avg merges: {res_full.avg_merges:.1f}  Avg verifier calls: {res_full.avg_verifier_calls:.1f}")
    print(f"No quotient merge: Solve rate: {res_nomerge.solve_rate:.2f}  Avg nodes: {res_nomerge.avg_nodes_expanded:.1f}  Avg merges: {res_nomerge.avg_merges:.1f}  Avg verifier calls: {res_nomerge.avg_verifier_calls:.1f}")
    print(f"No obstruction:    Solve rate: {res_noobs.solve_rate:.2f}  Avg nodes: {res_noobs.avg_nodes_expanded:.1f}  Avg merges: {res_noobs.avg_merges:.1f}  Avg verifier calls: {res_noobs.avg_verifier_calls:.1f}")
    print(f"No world model:    Solve rate: {res_nowm.solve_rate:.2f}  Avg nodes: {res_nowm.avg_nodes_expanded:.1f}  Avg merges: {res_nowm.avg_merges:.1f}  Avg verifier calls: {res_nowm.avg_verifier_calls:.1f}")
    # Save results
    results = {
        "full": res_full._asdict(),
        "no_merge": res_nomerge._asdict(),
        "no_obstruction": res_noobs._asdict(),
        "no_world_model": res_nowm._asdict(),
    }
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/sudoku_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[QWM] Results saved to results/sudoku_eval.json")

# ────────────────────────────────────────────────────────────────────
# Demo
# ────────────────────────────────────────────────────────────────────
def demo():
    import torch
    config = Config()
    trainer = QWMTrainer(config)
    ckpt_path = pathlib.Path("checkpoints/qwm_sudoku.pt")
    trainer.load_checkpoint(ckpt_path)
    verifier = SudokuVerifier()
    controller = QWMController(trainer.get_models_dict(), config, verifier)
    s = input("Enter 81-char Sudoku string (0=empty): ").strip()
    board = board_from_string(s)
    print("[QWM] Initial board:")
    print(board)
    print("[QWM] Running search...")
    result = controller.search(board, max_nodes=500)
    print("[QWM] Search finished.")
    if result["solved"]:
        print("[QWM] Solution:")
        print(result["solution_board"])
    else:
        print("[QWM] No solution found.")

# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|evaluate|demo]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "demo":
        demo()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
