"""Demo: show a puzzle, let the net fill empties, highlight mistakes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from board import EMPTY, copy_board, generate_solved, render
from dataset import blank_cells
from nn import predict_digit
from train import load_weights

out_dir = Path(__file__).resolve().parent


def render_compare(puzzle, filled, solution):
    """Side-by-side-ish: filled grid with markers for wrong / right / given."""
    lines = []
    lines.append("legend:  given=.digit  ok=digit  BAD=*digit*")
    lines.append("")
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row = []
        for c in range(9):
            if c in (3, 6):
                row.append("|")
            if puzzle[r][c] != EMPTY:
                row.append(str(puzzle[r][c]))
            else:
                pred = filled[r][c]
                if pred == solution[r][c]:
                    row.append(str(pred))
                else:
                    row.append(f"*{pred}*")
        lines.append(" ".join(row))
    return "\n".join(lines)


def fill_all_independent(puzzle, params):
    """Predict every empty cell from the *original* puzzle (no chaining)."""
    filled = copy_board(puzzle)
    mistakes = []
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != EMPTY:
                continue
            pred, probs = predict_digit(puzzle, r, c, params)
            filled[r][c] = pred
            mistakes.append(
                {
                    "r": r,
                    "c": c,
                    "pred": pred,
                    "conf": float(probs[pred - 1]),
                }
            )
    return filled, mistakes


def demo(seed=0, holes=35, weights_path=None):
    weights = Path(weights_path) if weights_path else out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit(f"missing {weights} — run train.py first")

    params = load_weights(weights)
    solution = generate_solved(seed=seed)
    puzzle = blank_cells(solution, holes=holes, rng=np.random.default_rng(seed + 1))

    print("=== puzzle ===")
    print(render(puzzle))
    print()

    filled, cells = fill_all_independent(puzzle, params)
    wrong = []
    for info in cells:
        r, c = info["r"], info["c"]
        truth = solution[r][c]
        info["truth"] = truth
        if info["pred"] != truth:
            wrong.append(info)

    print("=== net fill (independent predictions) ===")
    print(render_compare(puzzle, filled, solution))
    print()
    print(f"empties: {len(cells)}  correct: {len(cells) - len(wrong)}  mistakes: {len(wrong)}")
    if wrong:
        print("mistakes (row,col pred≠truth conf):")
        for w in wrong[:20]:
            print(
                f"  ({w['r']},{w['c']})  {w['pred']}≠{w['truth']}  p={w['conf']:.3f}"
            )
        if len(wrong) > 20:
            print(f"  … +{len(wrong) - 20} more")
    else:
        print("no mistakes")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holes", type=int, default=35)
    p.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()
    demo(seed=args.seed, holes=args.holes, weights_path=args.weights)
    print("demo ok")
