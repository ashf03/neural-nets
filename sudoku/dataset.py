"""Build (features, digit) from blankings + mid-solve teacher fills (+ unique mix)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from board import EMPTY, copy_board, generate_puzzle, generate_solved
from encode import FEATURE_SIZE, N_CLASSES, encode, encode_label

out_dir = Path(__file__).resolve().parent


def blank_cells(solution, holes: int, rng: np.random.Generator):
    """Copy a full grid and clear `holes` random cells (no uniqueness check)."""
    puzzle = copy_board(solution)
    cells = [(r, c) for r in range(9) for c in range(9)]
    order = rng.permutation(len(cells))
    holes = int(min(max(holes, 0), 81))
    for i in order[:holes]:
        r, c = cells[int(i)]
        puzzle[r][c] = EMPTY
    return puzzle


def mid_solve_from_board(
    board,
    solution,
    rng: np.random.Generator,
    samples_per_step: int = 2,
):
    """
    Teacher-fill empties on `board` in random order; sample remaining empties
    before each fill → (features, label).
    """
    board = copy_board(board)
    empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == EMPTY]
    if not empties:
        return []

    fill_order = [empties[int(i)] for i in rng.permutation(len(empties))]
    rows = []

    for step in range(len(fill_order)):
        remaining = fill_order[step:]
        n_pick = min(samples_per_step, len(remaining))
        pick_idx = rng.choice(len(remaining), size=n_pick, replace=False)
        for pi in np.atleast_1d(pick_idx):
            r, c = remaining[int(pi)]
            rows.append((encode(board, r, c), encode_label(solution[r][c])))

        tr, tc = fill_order[step]
        board[tr][tc] = solution[tr][tc]

    return rows


def mid_solve_examples(
    solution,
    holes: int,
    rng: np.random.Generator,
    samples_per_step: int = 2,
    hole_jitter: int = 0,
):
    """Blank then mid-solve. Optional ±hole_jitter for more blank patterns."""
    h = holes
    if hole_jitter > 0:
        h = int(holes + rng.integers(-hole_jitter, hole_jitter + 1))
        h = int(np.clip(h, 15, 60))
    board = blank_cells(solution, holes=h, rng=rng)
    return mid_solve_from_board(board, solution, rng, samples_per_step=samples_per_step)


def build_dataset(
    n_grids=40,
    holes=45,
    seed=42,
    trajectories_per_grid=3,
    samples_per_step=2,
    hole_jitter=3,
    n_unique=0,
    unique_holes=None,
):
    """
    Mid-solve dataset from many solved grids + blank patterns.
    If n_unique > 0, also mix trajectories starting from unique-solution puzzles
    (slower — generate_puzzle uniqueness checks).
    """
    rows_x = []
    rows_y = []
    unique_holes = holes if unique_holes is None else unique_holes

    for g in range(n_grids):
        solution = generate_solved(seed=int(seed) + g)
        for t in range(trajectories_per_grid):
            sub = np.random.default_rng(seed + 10_000 * g + t)
            for x, y in mid_solve_examples(
                solution,
                holes=holes,
                rng=sub,
                samples_per_step=samples_per_step,
                hole_jitter=hole_jitter,
            ):
                rows_x.append(x)
                rows_y.append(y)

    for u in range(n_unique):
        print(f"  unique puzzle {u + 1}/{n_unique} (holes≈{unique_holes})…")
        puzzle, solution = generate_puzzle(
            seed=int(seed) + 50_000 + u, holes=unique_holes
        )
        sub = np.random.default_rng(seed + 60_000 + u)
        # one trajectory from the unique starting puzzle; optional extra blank jitter
        for x, y in mid_solve_from_board(
            puzzle, solution, sub, samples_per_step=samples_per_step
        ):
            rows_x.append(x)
            rows_y.append(y)
        # second blank pattern on same solution (non-unique fast path)
        for x, y in mid_solve_examples(
            solution,
            holes=unique_holes,
            rng=np.random.default_rng(seed + 70_000 + u),
            samples_per_step=samples_per_step,
            hole_jitter=hole_jitter,
        ):
            rows_x.append(x)
            rows_y.append(y)

    X = np.stack(rows_x, axis=0)
    y = np.stack(rows_y, axis=0)
    return X, y


CURRICULUM_HOLES = (20, 35, 45, 55)


def build_curriculum_dataset(
    n_grids=24,
    holes_schedule=None,
    seed=42,
    trajectories_per_grid=3,
    samples_per_step=2,
    hole_jitter=3,
    n_unique_per_stage=3,
):
    """Concatenate richer mid-solve data for each curriculum hole count."""
    holes_schedule = holes_schedule or CURRICULUM_HOLES
    parts_x, parts_y, parts_h = [], [], []
    for stage, holes in enumerate(holes_schedule):
        print(f"building holes={holes}…")
        X, y = build_dataset(
            n_grids=n_grids,
            holes=holes,
            seed=seed + 1000 * stage,
            trajectories_per_grid=trajectories_per_grid,
            samples_per_step=samples_per_step,
            hole_jitter=hole_jitter,
            n_unique=n_unique_per_stage,
            unique_holes=holes,
        )
        parts_x.append(X)
        parts_y.append(y)
        parts_h.append(np.full(len(X), holes, dtype=np.int32))
        print(f"  curriculum holes={holes}: {len(X)} samples")
    X = np.concatenate(parts_x, axis=0)
    y = np.concatenate(parts_y, axis=0)
    holes_of = np.concatenate(parts_h, axis=0)
    return X, y, holes_of


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
    sol = generate_solved(seed=0)
    rng = np.random.default_rng(1)
    rows = mid_solve_examples(sol, holes=20, rng=rng, samples_per_step=1)
    assert len(rows) == 20
    assert rows[0][0].shape == (FEATURE_SIZE,)

    print("building richer dataset (grids + jitter + a few unique)…")
    path, X, y = save_dataset(
        n_grids=30,
        holes=40,
        trajectories_per_grid=3,
        samples_per_step=2,
        hole_jitter=4,
        n_unique=2,
        unique_holes=40,
        seed=42,
    )
    assert X.shape[1] == FEATURE_SIZE
    assert y.shape[1] == N_CLASSES
    assert np.all(y.sum(axis=1) == 1)
    print(f"wrote {path}")
    print(f"samples: {len(X)}")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    print(f"digit counts 1–9: {y.sum(axis=0).astype(int)}")
    print("dataset ok (more data + unique mix)")
