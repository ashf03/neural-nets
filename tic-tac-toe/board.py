"""Tic-tac-toe board: 9 cells, win / draw / legal moves."""

import numpy as np

EMPTY, X, O = 0, 1, -1

WINS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def new_board():
    return [EMPTY] * 9


def legal_moves(board):
    return [i for i, c in enumerate(board) if c == EMPTY]


def winner(board):
    """Return X, O, or None if nobody has three in a row yet."""
    for a, b, c in WINS:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board):
    return winner(board) is None and not legal_moves(board)


def is_terminal(board):
    return winner(board) is not None or is_draw(board)


def place(board, index, player):
    """Return a new board with `player` at `index`. Does not mutate."""
    if index not in legal_moves(board):
        raise ValueError(f"illegal move {index} on {board}")
    if player not in (X, O):
        raise ValueError(f"player must be X or O, got {player}")
    out = board[:]
    out[index] = player
    return out


def encode(board):
    """Board → float vector of length 9: X=1, O=-1, empty=0."""
    return np.asarray(board, dtype=np.float64)


def decode(vec):
    """Length-9 vector → board list (values cast to int)."""
    arr = np.asarray(vec).reshape(9)
    return [int(v) for v in arr]


def render(board):
    marks = {EMPTY: ".", X: "X", O: "O"}
    rows = []
    for r in range(3):
        cells = [marks[board[r * 3 + c]] for c in range(3)]
        rows.append(" ".join(cells))
    return "\n".join(rows)


if __name__ == "__main__":
    b = new_board()
    assert legal_moves(b) == list(range(9))
    assert winner(b) is None and not is_draw(b)

    b = place(b, 0, X)
    b = place(b, 1, X)
    b = place(b, 2, X)
    assert winner(b) == X
    assert legal_moves(b) == [3, 4, 5, 6, 7, 8]

    v = encode(b)
    assert v.shape == (9,)
    assert np.array_equal(v, [1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert decode(v) == b

    print(render(b))
    print("encode:", v)
    print("board ok")
