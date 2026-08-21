"""Classify 5x6 pixel letters A, B, C with a NumPy MLP."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
out_dir = Path(__file__).resolve().parent

# 5x6 binary grids (row-major, 30 pixels)
a = [
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1,
]
b = [
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
]
c = [
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0,
]

y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
x = [
    np.array(a).reshape(1, 30),
    np.array(b).reshape(1, 30),
    np.array(c).reshape(1, 30),
]
letters = ("A", "B", "C")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f_forward(x, w1, w2):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    return a2


def generate_wt(n_in, n_out):
    return np.random.randn(n_in, n_out)


def loss(out, Y):
    s = np.square(out - Y)
    return np.sum(s) / len(y)


def back_prop(x, y, w1, w2, alpha):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    d2 = a2 - y
    d1 = np.multiply(
        (w2.dot(d2.transpose())).transpose(),
        np.multiply(a1, 1 - a1),
    )
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)
    w1 = w1 - (alpha * w1_adj)
    w2 = w2 - (alpha * w2_adj)
    return w1, w2


def train(x, Y, w1, w2, alpha=0.01, epoch=10):
    acc = []
    losss = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append(loss(out, Y[i]))
            w1, w2 = back_prop(x[i], Y[i], w1, w2, alpha)
        epoch_acc = (1 - (sum(l) / len(x))) * 100
        print(f"epochs: {j + 1} ======== acc: {epoch_acc:.4f}")
        acc.append(epoch_acc)
        losss.append(sum(l) / len(x))
    return acc, losss, w1, w2


def predict(sample, w1, w2, name):
    out = f_forward(sample, w1, w2)
    k = int(np.argmax(out[0]))
    probs = ", ".join(f"{letters[i]}={out[0][i]:.3f}" for i in range(3))
    print(f"true {name} → predicted {letters[k]}  ({probs})")
    return letters[k]


def save_letters():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, pixels, name in zip(axes, (a, b, c), letters):
        ax.imshow(np.array(pixels).reshape(5, 6), cmap="gray_r")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    path = out_dir / "letters.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def save_curves(acc, losss):
    plt.plot(acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.tight_layout()
    acc_path = out_dir / "accuracy.png"
    plt.savefig(acc_path)
    plt.close()

    plt.plot(losss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.tight_layout()
    loss_path = out_dir / "loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"wrote {acc_path}")
    print(f"wrote {loss_path}")


def main():
    save_letters()
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 3)
    acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)
    save_curves(acc, losss)
    print("--- predictions ---")
    for sample, name in zip(x, letters):
        predict(sample, w1, w2, name)


if __name__ == "__main__":
    main()
