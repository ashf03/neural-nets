"""Learning task: one empty cell → digit 1–9 (from the unique solution)."""

from __future__ import annotations

import random

from board import EMPTY, DIGITS, generate_puzzle, legal_digits


def empty_cells(puzzle):
    return [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == EMPTY]


def target_digit(solution, r, c):
    d = solution[r][c]
    if d not in DIGITS:
        raise ValueError(f"no solution digit at {(r, c)}")
    return d


def sample_cell(puzzle, solution, rng=None):
    """
    Pick one empty cell uniformly.
    Returns (r, c, digit) where digit is the solution value 1–9.
    """
    rng = rng or random
    cells = empty_cells(puzzle)
    if not cells:
        raise ValueError("puzzle has no empty cells")
    r, c = cells[rng.randrange(len(cells))]
    return r, c, target_digit(solution, r, c)


def iter_cell_targets(puzzle, solution):
    """Yield (r, c, digit) for every empty cell."""
    for r, c in empty_cells(puzzle):
        yield r, c, target_digit(solution, r, c)


def example_bundle(puzzle, solution, r, c):
    """
    Everything the model needs for one supervised example (features come later).
      - puzzle snapshot
      - focus cell (r, c)
      - label digit 1–9
      - legal_digits at that cell (for masking later)
    """
    digit = target_digit(solution, r, c)
    return {
        "puzzle": puzzle,
        "r": r,
        "c": c,
        "digit": digit,
        "legal": legal_digits(puzzle, r, c),
    }


if __name__ == "__main__":
    puzzle, solution = generate_puzzle(seed=2, holes=30)
    cells = empty_cells(puzzle)
    assert len(cells) > 0

    r, c, d = sample_cell(puzzle, solution, rng=random.Random(0))
    assert puzzle[r][c] == EMPTY
    assert solution[r][c] == d
    assert d in DIGITS
    assert d in legal_digits(puzzle, r, c)

    bundles = [example_bundle(puzzle, solution, r, c) for r, c, _ in iter_cell_targets(puzzle, solution)]
    assert len(bundles) == len(cells)
    assert all(b["digit"] in b["legal"] for b in bundles)

    print(f"empties: {len(cells)}")
    print(f"sample focus=({r},{c}) digit={d} legal={sorted(legal_digits(puzzle, r, c))}")
    print("task ok")
