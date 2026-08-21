"""Sudoku board: validate row/col/box; generate puzzles with a unique solution."""

from __future__ import annotations

import random
from copy import deepcopy

EMPTY = 0
DIGITS = tuple(range(1, 10))


def new_board():
    return [[EMPTY] * 9 for _ in range(9)]


def copy_board(board):
    return [row[:] for row in board]


def box_id(r, c):
    return (r // 3) * 3 + (c // 3)


def peers(r, c):
    """All cells that share row, col, or box with (r, c), excluding itself."""
    seen = set()
    for i in range(9):
        seen.add((r, i))
        seen.add((i, c))
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            seen.add((i, j))
    seen.discard((r, c))
    return seen


def is_valid_placement(board, r, c, digit):
    """True if putting `digit` at (r, c) doesn't clash (ignores board[r][c])."""
    if digit not in DIGITS:
        return False
    for i in range(9):
        if i != c and board[r][i] == digit:
            return False
        if i != r and board[i][c] == digit:
            return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if (i, j) != (r, c) and board[i][j] == digit:
                return False
    return True


def legal_digits(board, r, c):
    if board[r][c] != EMPTY:
        return []
    return [d for d in DIGITS if is_valid_placement(board, r, c, d)]


def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == EMPTY:
                return r, c
    return None


def is_complete(board):
    return find_empty(board) is None


def is_consistent(board):
    """No clashes among filled cells."""
    for r in range(9):
        for c in range(9):
            d = board[r][c]
            if d == EMPTY:
                continue
            board[r][c] = EMPTY
            ok = is_valid_placement(board, r, c, d)
            board[r][c] = d
            if not ok:
                return False
    return True


def solve(board, count_solutions=False, limit=2):
    """
    Backtracking solve. Mutates a copy.
    If count_solutions: return number of solutions found (up to `limit`).
    Else: return a solved board or None.
    """
    b = copy_board(board)

    def search():
        pos = find_empty(b)
        if pos is None:
            return True
        r, c = pos
        opts = legal_digits(b, r, c)
        random.shuffle(opts)
        for d in opts:
            b[r][c] = d
            if search():
                return True
            b[r][c] = EMPTY
        return False

    def count(acc):
        if acc[0] >= limit:
            return
        pos = find_empty(b)
        if pos is None:
            acc[0] += 1
            return
        r, c = pos
        for d in legal_digits(b, r, c):
            b[r][c] = d
            count(acc)
            b[r][c] = EMPTY
            if acc[0] >= limit:
                return

    if count_solutions:
        acc = [0]
        count(acc)
        return acc[0]

    return b if search() else None


def count_solutions(board, limit=2):
    return solve(board, count_solutions=True, limit=limit)


def has_unique_solution(board):
    return count_solutions(board, limit=2) == 1


def generate_solved(seed=None):
    """Full valid grid via randomized backtracking from empty."""
    rng_state = random.getstate()
    if seed is not None:
        random.seed(seed)
    board = new_board()
    # fill diagonal boxes first (independent) for speed/variety
    for k in range(3):
        digits = list(DIGITS)
        random.shuffle(digits)
        for i in range(3):
            for j in range(3):
                board[3 * k + i][3 * k + j] = digits[i * 3 + j]
    solved = solve(board)
    random.setstate(rng_state)
    if solved is None:
        raise RuntimeError("failed to generate solved grid")
    return solved


def generate_puzzle(seed=None, holes=40):
    """
    Unique-solution puzzle: start from a full grid, remove cells while uniqueness holds.
    Returns (puzzle, solution).
    """
    rng = random.Random(seed)
    solution = generate_solved(seed=seed)
    puzzle = copy_board(solution)
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    removed = 0
    for r, c in cells:
        if removed >= holes:
            break
        backup = puzzle[r][c]
        puzzle[r][c] = EMPTY
        if has_unique_solution(puzzle):
            removed += 1
        else:
            puzzle[r][c] = backup
    return puzzle, solution


def render(board):
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row = []
        for c in range(9):
            if c in (3, 6):
                row.append("|")
            v = board[r][c]
            row.append(str(v) if v != EMPTY else ".")
        lines.append(" ".join(row))
    return "\n".join(lines)


if __name__ == "__main__":
    # clash detection
    b = new_board()
    b[0][0] = 5
    assert not is_valid_placement(b, 0, 1, 5)
    assert is_valid_placement(b, 0, 1, 6)
    assert 5 not in legal_digits(b, 1, 0)

    solved = generate_solved(seed=0)
    assert is_complete(solved) and is_consistent(solved)
    assert has_unique_solution(solved)

    puzzle, solution = generate_puzzle(seed=1, holes=35)
    assert has_unique_solution(puzzle)
    assert is_consistent(puzzle)
    empties = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == EMPTY)
    assert empties > 0
    filled = solve(copy_board(puzzle))
    assert filled == solution

    print(render(puzzle))
    print(f"empties: {empties}")
    print("board ok")
