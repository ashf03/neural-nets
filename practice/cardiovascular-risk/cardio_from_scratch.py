"""Binary classifier: weight + height → cardiovascular risk.

Follows Aadil Mallick / Stackademic:

Architecture: 2 → 3 → 3 → 1 (all sigmoid). Binary cross-entropy, vectorized backprop.

Raw pounds/inches are standard-scaled first (the article's challenge 2). Without that,
sigmoid saturates on values like 150 and 70 and training barely moves.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
out_dir = Path(__file__).resolve().parent

L = 3
n = [2, 3, 3, 1]

W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)


def prepare_data():
    X = np.array(
        [
            [150, 70],
            [254, 73],
            [312, 68],
            [120, 60],
            [154, 61],
            [212, 65],
            [216, 67],
            [145, 67],
            [184, 64],
            [130, 69],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    m = 10
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    A0 = X.T
    Y = y.reshape(n[L], m)
    return A0, Y, m


def cost(y_hat, y):
    y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
    losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))
    samples = y_hat.reshape(-1).shape[0]
    summed_losses = (1 / samples) * np.sum(losses, axis=1)
    return np.sum(summed_losses)


def g(z):
    return 1 / (1 + np.exp(-1 * z))


def feed_forward(A0):
    Z1 = W1 @ A0 + b1
    A1 = g(Z1)
    Z2 = W2 @ A1 + b2
    A2 = g(Z2)
    Z3 = W3 @ A2 + b3
    A3 = g(Z3)
    cache = {"A0": A0, "A1": A1, "A2": A2}
    return A3, cache


def backprop_layer_3(y_hat, Y, m, A2, W3):
    A3 = y_hat
    dC_dZ3 = (1 / m) * (A3 - Y)
    assert dC_dZ3.shape == (n[3], m)
    dZ3_dW3 = A2
    assert dZ3_dW3.shape == (n[2], m)
    dC_dW3 = dC_dZ3 @ dZ3_dW3.T
    assert dC_dW3.shape == (n[3], n[2])
    dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)
    assert dC_db3.shape == (n[3], 1)
    dC_dA2 = W3.T @ dC_dZ3
    assert dC_dA2.shape == (n[2], m)
    return dC_dW3, dC_db3, dC_dA2


def backprop_layer_2(propagator_dC_dA2, A1, A2, W2):
    dA2_dZ2 = A2 * (1 - A2)
    dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
    assert dC_dZ2.shape == (n[2], m)
    dZ2_dW2 = A1
    assert dZ2_dW2.shape == (n[1], m)
    dC_dW2 = dC_dZ2 @ dZ2_dW2.T
    assert dC_dW2.shape == (n[2], n[1])
    dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)
    assert dC_db2.shape == (n[2], 1)
    dC_dA1 = W2.T @ dC_dZ2
    assert dC_dA1.shape == (n[1], m)
    return dC_dW2, dC_db2, dC_dA1


def backprop_layer_1(propagator_dC_dA1, A1, A0, W1):
    dA1_dZ1 = A1 * (1 - A1)
    dC_dZ1 = propagator_dC_dA1 * dA1_dZ1
    assert dC_dZ1.shape == (n[1], m)
    dZ1_dW1 = A0
    assert dZ1_dW1.shape == (n[0], m)
    dC_dW1 = dC_dZ1 @ dZ1_dW1.T
    assert dC_dW1.shape == (n[1], n[0])
    dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)
    assert dC_db1.shape == (n[1], 1)
    return dC_dW1, dC_db1


def train():
    global W3, W2, W1, b3, b2, b1
    epochs = 1000
    alpha = 0.1
    costs = []
    for e in range(epochs):
        y_hat, cache = feed_forward(A0)
        error = cost(y_hat, Y)
        costs.append(error)
        dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
            y_hat, Y, m, A2=cache["A2"], W3=W3
        )
        dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(
            propagator_dC_dA2=dC_dA2,
            A1=cache["A1"],
            A2=cache["A2"],
            W2=W2,
        )
        dC_dW1, dC_db1 = backprop_layer_1(
            propagator_dC_dA1=dC_dA1,
            A1=cache["A1"],
            A0=cache["A0"],
            W1=W1,
        )
        W3 = W3 - (alpha * dC_dW3)
        W2 = W2 - (alpha * dC_dW2)
        W1 = W1 - (alpha * dC_dW1)
        b3 = b3 - (alpha * dC_db3)
        b2 = b2 - (alpha * dC_db2)
        b1 = b1 - (alpha * dC_db1)
        if e % 20 == 0:
            print(f"epoch {e}: cost = {error:.4f}")
    return costs


def main():
    global A0, Y, m
    A0, Y, m = prepare_data()
    costs = train()
    y_hat, _ = feed_forward(A0)
    preds = (y_hat >= 0.5).astype(int)
    print("--- predictions ---")
    print("true:     ", Y.astype(int).ravel())
    print("predicted:", preds.ravel())
    print("probs:    ", np.round(y_hat.ravel(), 3))
    print(f"final cost: {costs[-1]:.4f}")

    plt.plot(costs)
    plt.xlabel("EPOCHS")
    plt.ylabel("Cost")
    plt.title("Binary cross-entropy")
    plt.tight_layout()
    path = out_dir / "cost.png"
    plt.savefig(path)
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
