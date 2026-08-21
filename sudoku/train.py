"""Train sudoku MLP and save weights."""

from pathlib import Path

import numpy as np

from dataset import load_dataset, save_dataset
from encode import FEATURE_SIZE, N_CLASSES
from nn import (
    ARCHITECTURE,
    backward,
    cross_entropy_loss,
    forward,
    initialize_weights,
    sgd_step,
)

out_dir = Path(__file__).resolve().parent
WEIGHTS_PATH = out_dir / "weights.npz"


def train(
    X,
    y,
    epochs=25,
    lr=0.2,
    batch_size=64,
    seed=42,
    val_frac=0.1,
):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    params = initialize_weights(seed=seed)
    history = {"train_loss": [], "val_acc": []}
    N = len(X_train)

    for epoch in range(epochs):
        order = rng.permutation(N)
        X_shuffled, y_shuffled = X_train[order], y_train[order]
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb, yb = X_shuffled[start:end], y_shuffled[start:end]
            pred, cache = forward(xb, params)
            epoch_loss += cross_entropy_loss(pred, yb)
            n_batches += 1
            grads = backward(pred, yb, cache, params)
            params = sgd_step(params, grads, lr)

        avg_loss = epoch_loss / n_batches
        val_pred, _ = forward(X_val, params)
        val_acc = float(np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1)))
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}")

    return params, history


def save_weights(params, path=None):
    path = Path(path) if path else WEIGHTS_PATH
    np.savez_compressed(path, architecture=np.array(ARCHITECTURE), **params)
    return path


def load_weights(path=None):
    path = Path(path) if path else WEIGHTS_PATH
    data = np.load(path)
    return {k: data[k] for k in data.files if k.startswith(("W", "b"))}


if __name__ == "__main__":
    data_path = out_dir / "dataset.npz"
    if not data_path.exists():
        print("no dataset.npz — building…")
        save_dataset(data_path)

    X, y = load_dataset(data_path)
    assert X.shape[1] == FEATURE_SIZE and y.shape[1] == N_CLASSES
    print(f"training on {len(X)} samples")
    params, history = train(X, y)
    path = save_weights(params)
    assert set(load_weights(path)) == set(params)
    print(f"wrote {path}")
    print(f"final val_acc: {history['val_acc'][-1]:.4f}")
    print("train ok")
