"""Perfect-play minimax for tic-tac-toe."""

from functools import lru_cache

from board import EMPTY, O, X, is_draw, legal_moves, place, winner


def _key(board):
    return tuple(board)


def _score(board, perspective):
    """+1 if `perspective` won, -1 if opponent won, 0 otherwise (draw / ongoing)."""
    w = winner(board)
    if w == perspective:
        return 1
    if w is not None:
        return -1
    return 0


@lru_cache(maxsize=None)
def _minimax(board_t, player, perspective):
    """
    Return (best_score, best_move) for `player` to move.
    Scores are from `perspective`'s point of view.
    """
    board = list(board_t)
    if winner(board) is not None or is_draw(board):
        return _score(board, perspective), None

    moves = legal_moves(board)
    if player == perspective:
        best = (-2, moves[0])  # maximize
        for m in moves:
            child = place(board, m, player)
            s, _ = _minimax(_key(child), -player, perspective)
            if s > best[0]:
                best = (s, m)
        return best

    best = (2, moves[0])  # minimize
    for m in moves:
        child = place(board, m, player)
        s, _ = _minimax(_key(child), -player, perspective)
        if s < best[0]:
            best = (s, m)
    return best


def best_move(board, player):
    """Optimal cell index (0–8) for `player` (X or O) on this board."""
    if player not in (X, O):
        raise ValueError(f"player must be X or O, got {player}")
    if not legal_moves(board):
        raise ValueError("no legal moves")
    if winner(board) is not None:
        raise ValueError("game already over")

    score, move = _minimax(_key(board), player, player)
    return move, score


def best_move_only(board, player):
    move, _ = best_move(board, player)
    return move


if __name__ == "__main__":
    from board import new_board, render

    empty = new_board()
    move, score = best_move(empty, X)
    # Perfect X on empty board: center (4) or a corner; center is classic.
    assert move in (0, 2, 4, 6, 8)
    assert score == 0  # perfect play → draw from empty

    # X threatens two-in-a-row; O must block.
    # X X .
    # . O .
    # . . .
    b = [X, X, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY]
    move, score = best_move(b, O)
    assert move == 2, move
    print(render(b))
    print(f"O blocks at {move} (score {score})")

    # X can win now: two in a row with empty third.
    # X X .
    # O O .
    # . . .
    b = [X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY]
    move, score = best_move(b, X)
    assert move == 2, move
    assert score == 1
    print(f"X wins at {move} (score {score})")
    print("minimax ok")
