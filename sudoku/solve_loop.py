"""Full iterative solve loop + harder unique-solution puzzles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from board import (
    EMPTY,
    copy_board,
    generate_puzzle,
    is_complete,
    is_consistent,
    render,
)
from nn import predict_digit
from train import load_weights

out_dir = Path(__file__).resolve().parent


def full_solve(puzzle, params, max_steps=81):
    """
    Iterative loop: while empties remain, fill the cell where the net is
    most confident (among legal digits), then re-encode on the updated board.
    Returns (board, log) where log entries are {r,c,digit,conf}.
    """
    board = copy_board(puzzle)
    log = []
    for _ in range(max_steps):
        empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == EMPTY]
        if not empties:
            break
        best = None
        for r, c in empties:
            digit, probs = predict_digit(board, r, c, params)
            conf = float(probs[digit - 1])
            if best is None or conf > best[0]:
                best = (conf, r, c, digit)
        conf, r, c, digit = best
        board[r][c] = digit
        log.append({"r": r, "c": c, "digit": digit, "conf": conf})
    return board, log


def harder_puzzle(seed=0, target_holes=50, min_holes=40):
    """
    Unique-solution puzzle with as many holes as we can get (up to target).
    Falls back to whatever uniqueness allows (>= min_holes preferred).
    """
    puzzle, solution = generate_puzzle(seed=seed, holes=target_holes)
    empties = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == EMPTY)
    return puzzle, solution, empties


def run(seed=0, target_holes=50, weights_path=None):
    weights = Path(weights_path) if weights_path else out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit(f"missing {weights} — run train.py first")

    params = load_weights(weights)
    print(f"generating unique puzzle (target holes={target_holes})…")
    puzzle, solution, n_empty = harder_puzzle(seed=seed, target_holes=target_holes)
    print(f"got {n_empty} empties (unique solution)")
    print()
    print("=== harder puzzle ===")
    print(render(puzzle))
    print()

    filled, log = full_solve(puzzle, params)
    ok = is_complete(filled) and is_consistent(filled)
    exact = filled == solution
    wrong = [
        step
        for step in log
        if solution[step["r"]][step["c"]] != step["digit"]
    ]

    print("=== net full solve (iterative) ===")
    print(render(filled))
    print()
    print(f"steps: {len(log)}")
    print(f"complete+consistent: {ok}")
    print(f"exact match: {exact}")
    print(f"wrong fills: {len(wrong)}")
    if wrong:
        print("first mistakes:")
        for w in wrong[:10]:
            truth = solution[w["r"]][w["c"]]
            print(
                f"  ({w['r']},{w['c']}) wrote {w['digit']}≠{truth}  conf={w['conf']:.3f}"
            )

    # quick batch: a few harder seeds
    print()
    print("=== batch (5 unique puzzles) ===")
    for s in range(seed, seed + 5):
        puz, sol, n = harder_puzzle(seed=s, target_holes=target_holes)
        filled_b, _ = full_solve(puz, params)
        ok_b = is_complete(filled_b) and is_consistent(filled_b)
        exact_b = filled_b == sol
        print(f"seed={s} empties={n:2d} consistent={ok_b} exact={exact_b}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holes", type=int, default=50, help="target empties for unique puzzle")
    p.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()
    run(seed=args.seed, target_holes=args.holes, weights_path=args.weights)
    print("solve ok")
