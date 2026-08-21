"""NumPy MLP: sudoku features → digit 1–9. Legal-only softmax CE."""

from __future__ import annotations

import numpy as np

from board import legal_digits
from encode import CAND_SIZE, FEATURE_SIZE, N_CLASSES, encode

# FEATURE_SIZE → 256 → 128 → 9  (927 with row/col/box presence)
ARCHITECTURE = [FEATURE_SIZE, 256, 128, N_CLASSES]
NEG_INF = -1e9


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


def candidates_from_features(X):
    """Last 9 dims of encode() are legal-digit bits."""
    return np.asarray(X[..., -CAND_SIZE:], dtype=np.float64)


def apply_legal_mask(logits, legal_mask):
    """Set illegal class logits to -inf before softmax. legal_mask: (N, 9)."""
    mask = np.asarray(legal_mask, dtype=np.float64)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    out = np.asarray(logits, dtype=np.float64).copy()
    # rows with no legal bits → leave unmasked (avoid NaN)
    has_legal = mask.sum(axis=1, keepdims=True) > 0
    blocked = (mask <= 0) & has_legal
    out[blocked] = NEG_INF
    return out


def forward(X, params, legal_mask=None):
    """
    If legal_mask is None, use candidate bits from X (train/infer default).
    Pass legal_mask=False to disable masking.
    """
    cache = {"A0": X}
    cache["Z1"] = X @ params["W1"] + params["b1"]
    cache["A1"] = relu(cache["Z1"])
    cache["Z2"] = cache["A1"] @ params["W2"] + params["b2"]
    cache["A2"] = relu(cache["Z2"])
    cache["Z3"] = cache["A2"] @ params["W3"] + params["b3"]

    if legal_mask is False:
        mask = None
    elif legal_mask is None:
        mask = candidates_from_features(X)
    else:
        mask = legal_mask

    if mask is not None:
        cache["Z3"] = apply_legal_mask(cache["Z3"], mask)
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


def predict_digit(puzzle, r, c, params):
    """Forward with legal-only softmax → digit 1–9 and probs over candidates."""
    x = encode(puzzle, r, c).reshape(1, FEATURE_SIZE)
    probs, _ = forward(x, params)  # masks via candidate bits in x
    digit = int(np.argmax(probs[0])) + 1
    return digit, probs[0]


if __name__ == "__main__":
    from board import EMPTY, generate_puzzle

    params = initialize_weights()
    assert params["W1"].shape == (FEATURE_SIZE, 256)
    assert params["W3"].shape == (128, 9)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, FEATURE_SIZE))
    # fake candidate masks: at least one legal class matching label
    y = np.eye(N_CLASSES)[rng.integers(0, N_CLASSES, size=16)]
    X[:, -CAND_SIZE:] = 0.0
    X[np.arange(16), -CAND_SIZE + np.argmax(y, axis=1)] = 1.0
    # also allow a couple extras
    X[:, -CAND_SIZE:] += rng.random((16, CAND_SIZE)) > 0.7
    X[:, -CAND_SIZE:] = (X[:, -CAND_SIZE:] > 0).astype(np.float64)

    pred, cache = forward(X, params)
    # illegal classes ≈ 0
    illegal = candidates_from_features(X) <= 0
    assert np.all(pred[illegal] < 1e-6)

    loss = cross_entropy_loss(pred, y)
    for _ in range(25):
        pred, cache = forward(X, params)
        grads = backward(pred, y, cache, params)
        params = sgd_step(params, grads, lr=0.3)
    loss2 = cross_entropy_loss(forward(X, params)[0], y)
    assert loss2 < loss, (loss, loss2)

    puzzle, solution = generate_puzzle(seed=4, holes=20)
    r = c = None
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == EMPTY:
                r, c = i, j
                break
        if r is not None:
            break
    digit, probs = predict_digit(puzzle, r, c, params)
    legal = set(legal_digits(puzzle, r, c))
    for d in range(1, 10):
        if d not in legal:
            assert probs[d - 1] < 1e-6
    assert digit in legal

    print(f"arch: {ARCHITECTURE}")
    print(f"loss {loss:.4f} → {loss2:.4f} after 25 SGD steps (legal-only CE)")
    print(f"predict focus=({r},{c}) digit={digit} legal={sorted(legal)}")
    print("nn ok")
