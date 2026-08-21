"""NumPy MLP: snake state (11) → U/D/L/R (4). He init, ReLU, softmax, CE, SGD."""

from __future__ import annotations

import numpy as np

from encode import FEATURE_SIZE, INDEX_TO_DIR, N_ACTIONS, encode, is_danger
from game import OPPOSITE, SnakeGame

# 11 → 64 → 32 → 4
ARCHITECTURE = [FEATURE_SIZE, 64, 32, N_ACTIONS]


def initialize_weights(architecture=None, seed=42):
    architecture = architecture or ARCHITECTURE
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(1, len(architecture)):
        n_in, n_out = architecture[i - 1], architecture[i]
        params[f"W{i}"] = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out))
        params[f"b{i}"] = np.zeros((1, n_out))
    return params


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, params):
    cache = {"A0": X}
    cache["Z1"] = X @ params["W1"] + params["b1"]
    cache["A1"] = relu(cache["Z1"])
    cache["Z2"] = cache["A1"] @ params["W2"] + params["b2"]
    cache["A2"] = relu(cache["Z2"])
    cache["Z3"] = cache["A2"] @ params["W3"] + params["b3"]
    cache["A3"] = softmax(cache["Z3"])
    return cache["A3"], cache


def cross_entropy_loss(y_pred, y_true):
    N = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.sum(y_true * np.log(y_pred)) / N


def backward(y_pred, y_true, cache, params):
    N = y_true.shape[0]
    grads = {}

    dZ3 = (y_pred - y_true) / N
    grads["dW3"] = cache["A2"].T @ dZ3
    grads["db3"] = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    grads["dW2"] = cache["A1"].T @ dZ2
    grads["db2"] = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    grads["dW1"] = cache["A0"].T @ dZ1
    grads["db1"] = np.sum(dZ1, axis=0, keepdims=True)

    return grads


def sgd_step(params, grads, lr):
    for key in params:
        params[key] = params[key] - lr * grads[f"d{key}"]
    return params


def mask_actions(probs, game: SnakeGame, mask_danger=True):
    """
    Zero out 180° reverse; optionally zero instant-death moves.
    Renormalize over remaining. Shape (4,) or (1, 4).
    """
    p = np.asarray(probs, dtype=np.float64).reshape(N_ACTIONS).copy()
    banned = OPPOSITE[game.direction]
    p[INDEX_TO_DIR.index(banned)] = 0.0

    if mask_danger:
        for i, d in enumerate(INDEX_TO_DIR):
            if d != banned and is_danger(game, d):
                p[i] = 0.0

    total = p.sum()
    if total <= 0:
        # all masked — fall back to anything except 180°
        p = np.ones(N_ACTIONS, dtype=np.float64)
        p[INDEX_TO_DIR.index(banned)] = 0.0
        if mask_danger:
            for i, d in enumerate(INDEX_TO_DIR):
                if d != banned and is_danger(game, d):
                    p[i] = 0.0
        if p.sum() <= 0:
            p = np.ones(N_ACTIONS, dtype=np.float64)
            p[INDEX_TO_DIR.index(banned)] = 0.0
        p /= p.sum()
    else:
        p /= total
    return p


def predict_direction(game: SnakeGame, params, mask_danger=True):
    """Forward + mask → absolute direction tuple."""
    probs, _ = forward(encode(game).reshape(1, FEATURE_SIZE), params)
    masked = mask_actions(probs[0], game, mask_danger=mask_danger)
    idx = int(np.argmax(masked))
    return INDEX_TO_DIR[idx], masked


if __name__ == "__main__":
    from game import RIGHT, new_game

    params = initialize_weights()
    assert params["W1"].shape == (11, 64)
    assert params["W3"].shape == (32, 4)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, FEATURE_SIZE))
    y = np.eye(N_ACTIONS)[rng.integers(0, N_ACTIONS, size=32)]
    pred, cache = forward(X, params)
    assert pred.shape == (32, 4)
    loss = cross_entropy_loss(pred, y)
    for _ in range(20):
        pred, cache = forward(X, params)
        grads = backward(pred, y, cache, params)
        params = sgd_step(params, grads, lr=0.5)
    loss2 = cross_entropy_loss(forward(X, params)[0], y)
    assert loss2 < loss, (loss, loss2)

    g = new_game(width=5, height=5, seed=0)
    g.snake = [(2, 2), (2, 1), (2, 0)]
    g.direction = RIGHT
    g.food = (2, 4)
    direction, masked = predict_direction(g, params)
    assert masked[INDEX_TO_DIR.index(OPPOSITE[RIGHT])] == 0.0  # LEFT banned
    assert direction != OPPOSITE[RIGHT]

    # facing right into wall — straight danger masked
    g2 = new_game(width=3, height=3, seed=1)
    g2.snake = [(1, 2), (1, 1), (1, 0)]
    g2.direction = RIGHT
    g2.food = (0, 0)
    _, masked2 = predict_direction(g2, params, mask_danger=True)
    assert masked2[INDEX_TO_DIR.index(RIGHT)] == 0.0

    print(f"arch: {ARCHITECTURE}")
    print(f"loss {loss:.4f} → {loss2:.4f} after 20 SGD steps")
    print(f"predict dir: {direction}")
    print("nn ok")
