"""Human vs trained net — CLI."""

from pathlib import Path

import board as B
from nn import predict_move
from train import load_weights

out_dir = Path(__file__).resolve().parent
CELL_HINT = """
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
"""


def print_board(state):
    print()
    print(B.render(state))
    print()


def human_move(state):
    while True:
        raw = input("Your move (0-8, or q to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise SystemExit("bye")
        try:
            idx = int(raw)
        except ValueError:
            print("enter a number 0-8")
            continue
        if idx not in B.legal_moves(state):
            print("illegal — pick an empty cell")
            print(CELL_HINT)
            continue
        return idx


def net_move(state, params):
    move, probs = predict_move(state, params)
    print(f"Net plays {move}  (p={probs[move]:.3f})")
    return move


def play(params, human_is_x=True):
    state = B.new_board()
    human = B.X if human_is_x else B.O
    turn = B.X
    print("You are", "X" if human_is_x else "O")
    print("Cell indices:")
    print(CELL_HINT)
    print_board(state)

    while not B.is_terminal(state):
        if turn == human:
            idx = human_move(state)
        else:
            idx = net_move(state, params)
        state = B.place(state, idx, turn)
        print_board(state)
        turn = -turn

    w = B.winner(state)
    if w is None:
        print("Draw.")
    elif w == human:
        print("You win.")
    else:
        print("Net wins.")


if __name__ == "__main__":
    weights = out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit("missing weights.npz — run train.py first")

    params = load_weights(weights)
    side = input("Play as X (first) or O (second)? [X/O]: ").strip().upper() or "X"
    play(params, human_is_x=(side != "O"))
