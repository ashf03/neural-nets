"""Build (features, digit one-hot) from solved grids with cells blanked."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from board import EMPTY, copy_board, generate_solved
from encode import FEATURE_SIZE, N_CLASSES, encode, encode_label
from task import iter_cell_targets

out_dir = Path(__file__).resolve().parent


def blank_cells(solution, holes: int, rng: np.random.Generator):
    """Copy a full grid and clear `holes` random cells (no uniqueness check)."""
    puzzle = copy_board(solution)
    cells = [(r, c) for r in range(9) for c in range(9)]
    order = rng.permutation(len(cells))
    for i in order[:holes]:
        r, c = cells[int(i)]
        puzzle[r][c] = EMPTY
    return puzzle


def build_dataset(
    n_grids=40,
    holes=45,
    seed=42,
    blanks_per_grid=3,
):
    """
    For each of `n_grids` solved boards, create `blanks_per_grid` random
    blankings (each with `holes` empties). Every empty cell → one training row.
    Labels = digit from the generating solution.
    """
    rng = np.random.default_rng(seed)
    rows_x = []
    rows_y = []

    for g in range(n_grids):
        solution = generate_solved(seed=int(seed) + g)
        for b in range(blanks_per_grid):
            # independent blank pattern
            sub = np.random.default_rng(seed + 10_000 * g + b)
            puzzle = blank_cells(solution, holes=holes, rng=sub)
            for r, c, digit in iter_cell_targets(puzzle, solution):
                rows_x.append(encode(puzzle, r, c))
                rows_y.append(encode_label(digit))

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
    path, X, y = save_dataset(n_grids=30, holes=45, blanks_per_grid=2, seed=42)
    assert X.shape[1] == FEATURE_SIZE
    assert y.shape[1] == N_CLASSES
    assert np.all(y.sum(axis=1) == 1)
    print(f"wrote {path}")
    print(f"samples: {len(X)}")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    print(f"digit counts 1–9: {y.sum(axis=0).astype(int)}")
    print("dataset ok")
