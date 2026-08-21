"""Moons binary classifier — same idea as micrograd/demo.ipynb.

Scalar autograd is slow; 100 points × MLP(2,[16,16,1]) × 100 steps is the official demo.
"""

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

sys.path.insert(0, str(Path(__file__).resolve().parent))

from micrograd.engine import Value
from micrograd.nn import MLP

out_dir = Path(__file__).resolve().parent
random.seed(42)
np.random.seed(42)

X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # -1 / +1 for max-margin loss
model = MLP(2, [16, 16, 1])


def loss_and_acc():
    scores = [model([Value(x[0]), Value(x[1])]) for x in X]
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total = data_loss + reg_loss
    acc = sum((yi > 0) == (s.data > 0) for yi, s in zip(y, scores)) / len(y)
    return total, acc


print(model)
print(f"parameters: {len(model.parameters())}")

for k in range(100):
    total, acc = loss_and_acc()
    model.zero_grad()
    total.backward()
    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if k % 10 == 0 or k == 99:
        print(f"step {k:3d} | loss {total.data:.4f} | acc {acc:.2f}")

preds = []
for x in X:
    s = model([Value(x[0]), Value(x[1])])
    preds.append(1 if s.data > 0 else -1)

preds = np.array(preds)
print(f"final acc: {np.mean(preds == y):.2f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], c=preds, cmap="coolwarm", edgecolors="k")
ax.set_title("micrograd MLP on moons")
fig.tight_layout()
path = out_dir / "moons.png"
fig.savefig(path)
plt.close(fig)
print(f"wrote {path}")
