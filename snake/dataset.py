"""Collect (state encoding, action) pairs from heuristic rollouts."""

from pathlib import Path

import numpy as np

from encode import FEATURE_SIZE, N_ACTIONS, encode
from game import new_game
from heuristic import choose_action, choose_direction

out_dir = Path(__file__).resolve().parent


def one_hot_action(action: int, n=N_ACTIONS) -> np.ndarray:
    y = np.zeros(n, dtype=np.float64)
    y[action] = 1.0
    return y


def rollout(seed: int, width=10, height=10, max_steps=500):
    """Yield (features, one_hot_action) each tick until death or max_steps."""
    game = new_game(width=width, height=height, seed=seed)
    steps = 0
    while game.alive and steps < max_steps:
        feats = encode(game)
        action = choose_action(game)
        yield feats, one_hot_action(action)
        game.step(choose_direction(game))
        steps += 1


def build_dataset(
    n_games=80,
    seed=42,
    width=10,
    height=10,
    max_steps=500,
):
    rows_x = []
    rows_y = []
    for g in range(n_games):
        for feats, label in rollout(seed + g, width=width, height=height, max_steps=max_steps):
            rows_x.append(feats)
            rows_y.append(label)
    X = np.stack(rows_x, axis=0)
    y = np.stack(rows_y, axis=0)
    return X, y


def save_dataset(path=None, **kwargs):
    path = Path(path) if path else out_dir / "dataset.npz"
    X, y = build_dataset(**kwargs)
    np.savez_compressed(path, X=X, y=y)
    return path, X, y


def load_dataset(path=None):
    path = Path(path) if path else out_dir / "dataset.npz"
    data = np.load(path)
    return data["X"], data["y"]


if __name__ == "__main__":
    path, X, y = save_dataset(n_games=80, seed=42)
    assert X.shape[1] == FEATURE_SIZE
    assert y.shape[1] == N_ACTIONS
    assert np.all(y.sum(axis=1) == 1)
    print(f"wrote {path}")
    print(f"samples: {len(X)}")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    print(f"action counts L/S/R: {y.sum(axis=0).astype(int)}")
    print("dataset ok")
