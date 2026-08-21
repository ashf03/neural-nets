"""Generate (board, best-move) pairs labeled by minimax."""

from pathlib import Path

import numpy as np

from board import EMPTY, O, X, encode, is_terminal, legal_moves, new_board, place
from minimax import best_move_only

out_dir = Path(__file__).resolve().parent


def one_hot_move(index, n=9):
    y = np.zeros(n, dtype=np.float64)
    y[index] = 1.0
    return y


def random_play_position(rng):
    """Play random legal moves until a non-terminal mid-game board (or empty)."""
    board = new_board()
    player = X
    # 0–8 plies; stop early if game ends so caller can skip terminals
    plies = int(rng.integers(0, 9))
    for _ in range(plies):
        moves = legal_moves(board)
        if not moves or is_terminal(board):
            break
        board = place(board, int(rng.choice(moves)), player)
        player = -player
    return board, player


def enumerate_reachable(max_positions=None):
    """
    BFS every reachable non-terminal position from the empty board.
    Yields (board, player_to_move).
    """
    seen = set()
    queue = [(new_board(), X)]
    while queue:
        board, player = queue.pop(0)
        key = (tuple(board), player)
        if key in seen:
            continue
        seen.add(key)
        if is_terminal(board):
            continue
        yield board, player
        if max_positions is not None and len(seen) >= max_positions:
            return
        for m in legal_moves(board):
            queue.append((place(board, m, player), -player))


def build_dataset(seed=42, extra_random=0):
    """
    All reachable non-terminal positions (unique board+side), plus optional
    random duplicates for variety. Labels = one-hot of minimax best move.
    """
    rng = np.random.default_rng(seed)
    rows_x = []
    rows_y = []
    seen = set()

    def add(board, player):
        key = (tuple(board), player)
        if key in seen or is_terminal(board):
            return
        seen.add(key)
        move = best_move_only(board, player)
        rows_x.append(encode(board))
        rows_y.append(one_hot_move(move))

    for board, player in enumerate_reachable():
        add(board, player)

    for _ in range(extra_random):
        board, player = random_play_position(rng)
        if not is_terminal(board) and legal_moves(board):
            # may already be in seen — skip duplicates
            add(board, player)

    X_data = np.stack(rows_x, axis=0)
    y_data = np.stack(rows_y, axis=0)
    return X_data, y_data


def save_dataset(path=None, seed=42):
    path = Path(path) if path else out_dir / "dataset.npz"
    X_data, y_data = build_dataset(seed=seed)
    np.savez_compressed(path, X=X_data, y=y_data)
    return path, X_data, y_data


def load_dataset(path=None):
    path = Path(path) if path else out_dir / "dataset.npz"
    data = np.load(path)
    return data["X"], data["y"]


if __name__ == "__main__":
    path, X_data, y_data = save_dataset()
    assert X_data.shape[1] == 9 and y_data.shape[1] == 9
    assert np.all(y_data.sum(axis=1) == 1)

    # Empty board, X to move: label should be a single 1.
    empty = encode(new_board())
    # find rows matching empty board
    matches = np.where(np.all(X_data == empty, axis=1))[0]
    assert len(matches) >= 1
    label = y_data[matches[0]]
    assert label.sum() == 1
    move = int(np.argmax(label))
    assert move == best_move_only(new_board(), X)

    print(f"wrote {path}")
    print(f"samples: {len(X_data)}")
    print(f"X shape: {X_data.shape}  y shape: {y_data.shape}")
    print(f"empty-board best move index: {move}")
    print("dataset ok")
