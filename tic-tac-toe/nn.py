"""NumPy MLP: board (9) → move probs (9). He init, ReLU, softmax, CE, SGD."""

import numpy as np

# 9 → 64 → 64 → 9
ARCHITECTURE = [9, 64, 64, 9]


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


if __name__ == "__main__":
    params = initialize_weights()
    assert params["W1"].shape == (9, 64)
    assert params["W2"].shape == (64, 64)
    assert params["W3"].shape == (64, 9)

    X = np.zeros((4, 9))
    y = np.eye(9)[[0, 1, 2, 4]]
    pred, cache = forward(X, params)
    assert pred.shape == (4, 9)
    assert np.allclose(pred.sum(axis=1), 1.0)

    loss = cross_entropy_loss(pred, y)
    grads = backward(pred, y, cache, params)
    params = sgd_step(params, grads, lr=0.1)
    pred2, _ = forward(X, params)
    loss2 = cross_entropy_loss(pred2, y)
    assert loss2 < loss

    print(f"arch: {ARCHITECTURE}")
    print(f"loss {loss:.4f} → {loss2:.4f} after 1 SGD step")
    print("nn ok")
