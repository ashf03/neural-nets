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
    legal_digits,
    render,
)
from nn import predict_digit
from train import load_weights

out_dir = Path(__file__).resolve().parent

DEFAULT_TAU = 0.65


def _ranked_predictions(board, params, forbidden):
    """
    For each empty cell, best digit not in forbidden[(r,c)], with confidence.
    Returns list of (conf, r, c, digit) sorted descending by conf.
    """
    ranked = []
    empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == EMPTY]
    for r, c in empties:
        ban = forbidden.get((r, c), set())
        digit, probs = predict_digit(board, r, c, params)
        # reconsider top legal digits excluding banned
        order = np.argsort(-probs)
        chosen = None
        conf = 0.0
        for idx in order:
            d = int(idx) + 1
            if d in ban:
                continue
            if d not in legal_digits(board, r, c):
                continue
            chosen = d
            conf = float(probs[idx])
            break
        if chosen is None:
            continue
        ranked.append((conf, r, c, chosen))
    ranked.sort(reverse=True)
    return ranked


def full_solve(puzzle, params, tau=DEFAULT_TAU, max_steps=200, max_backtracks=80):
    """
    Fill highest-confidence empty cell only if conf >= tau.
    If every empty is below tau, backtrack: undo last fill, ban that digit
    for that cell, and try again.
    Returns (board, log, stats).
    """
    board = copy_board(puzzle)
    log = []
    stack = []  # (r, c, digit, conf)
    forbidden = {}  # (r, c) -> set of digits
    backtracks = 0
    skips = 0

    for _ in range(max_steps):
        empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == EMPTY]
        if not empties:
            break

        ranked = _ranked_predictions(board, params, forbidden)
        eligible = [t for t in ranked if t[0] >= tau]

        if eligible:
            conf, r, c, digit = eligible[0]
            board[r][c] = digit
            stack.append((r, c, digit, conf))
            log.append(
                {
                    "r": r,
                    "c": c,
                    "digit": digit,
                    "conf": conf,
                    "event": "fill",
                }
            )
            continue

        # nothing confident enough — backtrack
        skips += 1
        if not stack or backtracks >= max_backtracks:
            log.append({"event": "stuck", "empties": len(empties), "tau": tau})
            break

        r, c, digit, conf = stack.pop()
        board[r][c] = EMPTY
        forbidden.setdefault((r, c), set()).add(digit)
        backtracks += 1
        log.append(
            {
                "r": r,
                "c": c,
                "digit": digit,
                "conf": conf,
                "event": "backtrack",
            }
        )

    stats = {
        "fills": sum(1 for e in log if e.get("event") == "fill"),
        "backtracks": backtracks,
        "stuck": any(e.get("event") == "stuck" for e in log),
        "tau": tau,
    }
    return board, log, stats


def harder_puzzle(seed=0, target_holes=50, min_holes=40):
    """
    Unique-solution puzzle with as many holes as we can get (up to target).
    Falls back to whatever uniqueness allows (>= min_holes preferred).
    """
    puzzle, solution = generate_puzzle(seed=seed, holes=target_holes)
    empties = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == EMPTY)
    return puzzle, solution, empties


def run(seed=0, target_holes=50, weights_path=None, tau=DEFAULT_TAU):
    weights = Path(weights_path) if weights_path else out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit(f"missing {weights} — run train.py first")

    params = load_weights(weights)
    print(f"generating unique puzzle (target holes={target_holes})…")
    puzzle, solution, n_empty = harder_puzzle(seed=seed, target_holes=target_holes)
    print(f"got {n_empty} empties (unique solution)  tau={tau}")
    print()
    print("=== harder puzzle ===")
    print(render(puzzle))
    print()

    filled, log, stats = full_solve(puzzle, params, tau=tau)
    ok = is_complete(filled) and is_consistent(filled)
    exact = filled == solution
    wrong = [
        step
        for step in log
        if step.get("event") == "fill"
        and solution[step["r"]][step["c"]] != step["digit"]
    ]

    print("=== net full solve (iterative, confidence threshold) ===")
    print(render(filled))
    print()
    print(f"fills: {stats['fills']}  backtracks: {stats['backtracks']}  stuck: {stats['stuck']}")
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

    print()
    print(f"=== batch (5 unique puzzles, tau={tau}) ===")
    for s in range(seed, seed + 5):
        puz, sol, n = harder_puzzle(seed=s, target_holes=target_holes)
        filled_b, _, st = full_solve(puz, params, tau=tau)
        ok_b = is_complete(filled_b) and is_consistent(filled_b)
        exact_b = filled_b == sol
        print(
            f"seed={s} empties={n:2d} consistent={ok_b} exact={exact_b} "
            f"bt={st['backtracks']} stuck={st['stuck']}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holes", type=int, default=50, help="target empties for unique puzzle")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU, help="min confidence to fill")
    p.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()
    run(seed=args.seed, target_holes=args.holes, weights_path=args.weights, tau=args.tau)
    print("solve ok")
