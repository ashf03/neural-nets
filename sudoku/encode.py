"""Encode a sudoku example for the MLP.

Chosen scheme (stick to this):
  1) Board: 81 cells × 10 one-hot (empty + digits 1–9)     → 810
  2) Focus: which empty cell we're filling (81 one-hot)     →  81
  3) Local presence: digits already in focus row/col/box    →  27
  4) Candidates: legal digits at focus (9 binary)           →   9
                                                     total → 927

Candidates stay last so the net can mask illegal logits from X[..., -9:].
"""

from __future__ import annotations

import numpy as np

from board import DIGITS, EMPTY, legal_digits

CELL_OH = 10  # index 0 = empty, 1..9 = digit
BOARD_SIZE = 81
FOCUS_SIZE = 81
PRESENCE_UNIT = 9  # digits 1–9 present in one unit
PRESENCE_SIZE = 3 * PRESENCE_UNIT  # row + col + box
CAND_SIZE = 9
FEATURE_SIZE = (
    BOARD_SIZE * CELL_OH + FOCUS_SIZE + PRESENCE_SIZE + CAND_SIZE
)  # 927
N_CLASSES = 9  # digits 1–9 → softmax index 0..8


def _cell_one_hot(value: int) -> np.ndarray:
    v = np.zeros(CELL_OH, dtype=np.float64)
    if value == EMPTY:
        v[0] = 1.0
    else:
        v[value] = 1.0
    return v


def encode_board(puzzle) -> np.ndarray:
    parts = [_cell_one_hot(puzzle[r][c]) for r in range(9) for c in range(9)]
    return np.concatenate(parts, axis=0)


def encode_focus(r: int, c: int) -> np.ndarray:
    v = np.zeros(FOCUS_SIZE, dtype=np.float64)
    v[r * 9 + c] = 1.0
    return v


def encode_presence(puzzle, r: int, c: int) -> np.ndarray:
    """Digits already filled in focus row, then col, then 3×3 box (27 bits)."""
    row = np.zeros(PRESENCE_UNIT, dtype=np.float64)
    col = np.zeros(PRESENCE_UNIT, dtype=np.float64)
    box = np.zeros(PRESENCE_UNIT, dtype=np.float64)

    for j in range(9):
        d = puzzle[r][j]
        if d != EMPTY:
            row[d - 1] = 1.0
    for i in range(9):
        d = puzzle[i][c]
        if d != EMPTY:
            col[d - 1] = 1.0
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            d = puzzle[i][j]
            if d != EMPTY:
                box[d - 1] = 1.0

    return np.concatenate([row, col, box], axis=0)


def encode_candidates(puzzle, r: int, c: int) -> np.ndarray:
    v = np.zeros(CAND_SIZE, dtype=np.float64)
    for d in legal_digits(puzzle, r, c):
        v[d - 1] = 1.0
    return v


def encode(puzzle, r: int, c: int) -> np.ndarray:
    """Feature vector shape (FEATURE_SIZE,)."""
    return np.concatenate(
        [
            encode_board(puzzle),
            encode_focus(r, c),
            encode_presence(puzzle, r, c),
            encode_candidates(puzzle, r, c),
        ],
        axis=0,
    )


def encode_label(digit: int) -> np.ndarray:
    """Digit 1–9 → one-hot length 9."""
    if digit not in DIGITS:
        raise ValueError(digit)
    y = np.zeros(N_CLASSES, dtype=np.float64)
    y[digit - 1] = 1.0
    return y


def decode_label(vec) -> int:
    return int(np.argmax(vec)) + 1


if __name__ == "__main__":
    from board import generate_puzzle
    from task import sample_cell

    puzzle, solution = generate_puzzle(seed=3, holes=25)
    r, c, digit = sample_cell(puzzle, solution, rng=__import__("random").Random(0))

    x = encode(puzzle, r, c)
    y = encode_label(digit)
    assert x.shape == (FEATURE_SIZE,)
    assert y.shape == (N_CLASSES,)
    assert y.sum() == 1
    assert decode_label(y) == digit

    board_end = BOARD_SIZE * CELL_OH
    focus = x[board_end : board_end + FOCUS_SIZE]
    assert focus[r * 9 + c] == 1.0
    assert focus.sum() == 1.0

    presence = x[board_end + FOCUS_SIZE : board_end + FOCUS_SIZE + PRESENCE_SIZE]
    assert presence.shape == (PRESENCE_SIZE,)
    # true digit must not already be present in row/col/box
    assert presence[digit - 1] == 0.0
    assert presence[PRESENCE_UNIT + digit - 1] == 0.0
    assert presence[2 * PRESENCE_UNIT + digit - 1] == 0.0

    cands = x[-CAND_SIZE:]
    assert cands[digit - 1] == 1.0
    # candidate ≈ not (row ∪ col ∪ box)
    row_p, col_p, box_p = (
        presence[:PRESENCE_UNIT],
        presence[PRESENCE_UNIT : 2 * PRESENCE_UNIT],
        presence[2 * PRESENCE_UNIT :],
    )
    blocked = np.maximum(np.maximum(row_p, col_p), box_p)
    assert np.all(cands == (1.0 - blocked))

    print(f"FEATURE_SIZE={FEATURE_SIZE} focus=({r},{c}) digit={digit}")
    print(f"candidates: {np.where(cands == 1)[0] + 1}")
    print("encode ok")
