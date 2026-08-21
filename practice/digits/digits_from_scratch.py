"""sklearn digits (8x8) with a 64 → 128 → 64 → 10 NumPy MLP.

He init, ReLU hidden, softmax + cross-entropy, mini-batch SGD.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
out_dir = Path(__file__).resolve().parent


def initialize_weights(architecture):
    params = {}
    for i in range(1, len(architecture)):
        n_in, n_out = architecture[i - 1], architecture[i]
        params[f"W{i}"] = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
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
    y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.sum(y_true * np.log(y_pred_clipped)) / N


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


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


def train(X_train, y_train, X_val, y_val, architecture, epochs=200, lr=0.1, batch_size=64):
    params = initialize_weights(architecture)
    y_train_oh = one_hot_encode(y_train, 10)
    N = X_train.shape[0]
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]

        epoch_loss = 0
        num_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            y_pred, cache = forward(X_batch, params)
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_loss += loss
            num_batches += 1
            grads = backward(y_pred, y_batch, cache, params)
            for key in params:
                params[key] -= lr * grads[f"d{key}"]

        avg_loss = epoch_loss / num_batches
        val_pred, _ = forward(X_val, params)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    return params, history


def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}, Classes: 10")

    architecture = [64, 128, 64, 10]
    params = initialize_weights(architecture)
    print(
        f"W1 shape: {params['W1'].shape}, std: {params['W1'].std():.4f}\n"
        f"W2 shape: {params['W2'].shape}, std: {params['W2'].std():.4f}\n"
        f"W3 shape: {params['W3'].shape}, std: {params['W3'].std():.4f}"
    )
    # re-init inside train() with the same seed path; reset seed so train matches tutorial
    np.random.seed(42)

    params, history = train(X_train, y_train, X_val, y_val, architecture)

    val_pred, _ = forward(X_val, params)
    val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[1].plot(history["val_acc"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val accuracy")
    fig.tight_layout()
    path = out_dir / "training.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
