"""XOR logic gate with a tiny neural net in NumPy. """

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# XOR truth table: columns are the four examples (shape (2, 4))
a = np.array([0, 0, 1, 1])
b = np.array([0, 1, 0, 1])
y_xor = np.array([[0, 1, 1, 0]])
total_input = np.array([a, b])

input_neurons, hidden_neurons, output_neurons = 2, 2, 1
samples = total_input.shape[1]
lr = 0.1
np.random.seed(42)

w1 = np.random.rand(hidden_neurons, input_neurons)
w2 = np.random.rand(output_neurons, hidden_neurons)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def back_prop(m, w1, w2, z1, a1, z2, a2, y, x):
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / m
    dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)
    dw1 = np.dot(dz1, x.T) / m
    dw1 = np.reshape(dw1, w1.shape)
    dw2 = np.reshape(dw2, w2.shape)
    return dz2, dw2, dz1, dw1


def predict(w1, w2, x):
    _, _, _, a2 = forward_prop(w1, w2, x)
    a2 = np.squeeze(a2)
    bits = [int(row[0]) for row in x]
    out = 1 if a2 >= 0.5 else 0
    print(f"For input {bits} output is {out} (p={float(a2):.4f})")
    return out


def main():
    global w1, w2
    losses = []
    iterations = 10_000

    for _ in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, w2, total_input)
        loss = -(1 / samples) * np.sum(
            y_xor * np.log(a2) + (1 - y_xor) * np.log(1 - a2)
        )
        losses.append(loss)
        _, dw2, _, dw1 = back_prop(
            samples, w1, w2, z1, a1, z2, a2, y_xor, total_input
        )
        w2 = w2 - lr * dw2
        w1 = w1 - lr * dw1

    print(f"final loss: {losses[-1]:.6f}")
    for pair in ([[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]):
        predict(w1, w2, np.array(pair))

    out_dir = Path(__file__).resolve().parent
    plt.plot(losses)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")
    plt.title("XOR training loss")
    plt.tight_layout()
    plot_path = out_dir / "xor_loss.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
