"""Compare net moves to minimax on held-out boards."""

from pathlib import Path

import numpy as np

import board as B
from dataset import load_dataset
from minimax import best_move, best_move_only
from nn import predict_move
from train import load_weights

out_dir = Path(__file__).resolve().parent


def score_after_move(state, move, player):
    """Outcome for `player` after playing `move`, assuming perfect replies."""
    placed = B.place(state, move, player)
    w = B.winner(placed)
    if w == player:
        return 1
    if w is not None:
        return -1
    if B.is_draw(placed):
        return 0
    _, opp_score = best_move(placed, -player)
    return -opp_score


def all_optimal_moves(state, player):
    """Every legal move that achieves the minimax-optimal score."""
    _, best_score = best_move(state, player)
    return [
        m
        for m in B.legal_moves(state)
        if score_after_move(state, m, player) == best_score
    ]


def player_to_move(state):
    """X starts; equal piece counts → X, else O."""
    xs = sum(1 for c in state if c == B.X)
    os = sum(1 for c in state if c == B.O)
    return B.X if xs == os else B.O


def evaluate(features, labels, params, seed=0, holdout_frac=0.2):
    rng = np.random.default_rng(seed)
    n = len(features)
    hold = rng.permutation(n)[: max(1, int(n * holdout_frac))]

    strict_ok = 0
    optimal_ok = 0
    illegal = 0

    for i in hold:
        state = B.decode(features[i])
        if B.is_terminal(state):
            continue
        player = player_to_move(state)
        net_move, _ = predict_move(state, params)

        if state[net_move] != B.EMPTY:
            illegal += 1
            continue

        if net_move == best_move_only(state, player):
            strict_ok += 1
        if net_move in all_optimal_moves(state, player):
            optimal_ok += 1

    n_eval = len(hold)
    return {
        "n": n_eval,
        "strict_acc": strict_ok / n_eval,
        "optimal_acc": optimal_ok / n_eval,
        "illegal": illegal,
    }


if __name__ == "__main__":
    weights = out_dir / "weights.npz"
    data = out_dir / "dataset.npz"
    if not weights.exists():
        raise SystemExit("missing weights.npz — run train.py first")
    if not data.exists():
        raise SystemExit("missing dataset.npz — run dataset.py first")

    params = load_weights(weights)
    features, labels = load_dataset(data)
    stats = evaluate(features, labels, params)
    print(f"held-out boards: {stats['n']}")
    print(f"strict match (net == one minimax pick): {stats['strict_acc']:.4f}")
    print(f"optimal match (net is any perfect move): {stats['optimal_acc']:.4f}")
    print(f"illegal moves: {stats['illegal']}")
    print("eval ok")
