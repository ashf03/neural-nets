"""Eval sudoku net: per-cell accuracy + greedy full-board solve rate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from board import EMPTY, copy_board, generate_solved, is_complete, is_consistent
from dataset import blank_cells
from nn import predict_digit
from task import iter_cell_targets
from train import load_weights

out_dir = Path(__file__).resolve().parent


def per_cell_accuracy(params, n_grids=20, holes=45, seed=0):
    """Fraction of empty cells where masked argmax == solution digit."""
    correct = 0
    total = 0
    for g in range(n_grids):
        solution = generate_solved(seed=seed + g)
        rng = np.random.default_rng(seed + 1000 + g)
        puzzle = blank_cells(solution, holes=holes, rng=rng)
        for r, c, digit in iter_cell_targets(puzzle, solution):
            pred, _ = predict_digit(puzzle, r, c, params)
            correct += int(pred == digit)
            total += 1
    return correct / max(total, 1), correct, total


def greedy_solve(puzzle, params, max_steps=81):
    """
    Repeatedly fill the empty cell where the net is most confident
    (among legal digits). Returns (board, solved_ok).
    """
    board = copy_board(puzzle)
    for _ in range(max_steps):
        empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == EMPTY]
        if not empties:
            break
        best = None  # (confidence, r, c, digit)
        for r, c in empties:
            digit, probs = predict_digit(board, r, c, params)
            conf = float(probs[digit - 1])
            if best is None or conf > best[0]:
                best = (conf, r, c, digit)
        _, r, c, digit = best
        board[r][c] = digit
    ok = is_complete(board) and is_consistent(board)
    return board, ok


def greedy_solve_rate(params, n_grids=20, holes=35, seed=0):
    """How often greedy fill yields a complete consistent grid matching solution."""
    wins = 0
    exact = 0
    for g in range(n_grids):
        solution = generate_solved(seed=seed + 5000 + g)
        rng = np.random.default_rng(seed + 6000 + g)
        puzzle = blank_cells(solution, holes=holes, rng=rng)
        filled, ok = greedy_solve(puzzle, params)
        if ok:
            wins += 1
        if filled == solution:
            exact += 1
    return {
        "n": n_grids,
        "consistent_complete": wins / n_grids,
        "exact_solution": exact / n_grids,
    }


if __name__ == "__main__":
    weights = out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit("missing weights.npz — run train.py first")

    params = load_weights(weights)
    acc, correct, total = per_cell_accuracy(params, n_grids=15, holes=45, seed=0)
    print(f"per-cell accuracy: {acc:.4f}  ({correct}/{total})")

    stats = greedy_solve_rate(params, n_grids=15, holes=30, seed=0)
    print(
        f"greedy solve ({stats['n']} puzzles, 30 holes): "
        f"complete+consistent={stats['consistent_complete']:.2f}  "
        f"exact={stats['exact_solution']:.2f}"
    )
    print("eval ok")
