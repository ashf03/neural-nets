"""Curriculum: train 20 → 35 → 45 → 55 holes, then eval at each level."""

from __future__ import annotations

from pathlib import Path

from board import generate_puzzle, is_complete, is_consistent
from dataset import CURRICULUM_HOLES, build_dataset
from eval import per_cell_accuracy
from solve_loop import full_solve
from train import save_weights, train

out_dir = Path(__file__).resolve().parent
WEIGHTS_PATH = out_dir / "weights_curriculum.npz"


def train_curriculum(
    holes_schedule=None,
    n_grids=24,
    trajectories_per_grid=3,
    samples_per_step=2,
    hole_jitter=3,
    n_unique_per_stage=3,
    epochs_per_stage=12,
    lr=0.2,
    seed=42,
):
    holes_schedule = holes_schedule or CURRICULUM_HOLES
    params = None
    stage_logs = []

    for stage, holes in enumerate(holes_schedule):
        print(f"\n=== stage {stage + 1}/{len(holes_schedule)}: holes={holes} ===")
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
        print(f"samples: {len(X)}")
        params, history = train(
            X,
            y,
            epochs=epochs_per_stage,
            lr=lr,
            seed=seed + stage,
            params=params,
            log_prefix=f"[h={holes}] ",
        )
        stage_logs.append(
            {
                "holes": holes,
                "samples": len(X),
                "final_val_acc": history["val_acc"][-1],
            }
        )

    path = save_weights(params, WEIGHTS_PATH)
    print(f"\nwrote {path}")
    return params, stage_logs


def eval_curriculum(params, holes_schedule=None, n_grids=8, n_unique=4, seed=0, tau=0.65):
    """Per-cell acc on blankings + exact solve rate on unique puzzles, per hole count."""
    holes_schedule = holes_schedule or CURRICULUM_HOLES
    print("\n=== curriculum eval ===")
    for holes in holes_schedule:
        acc, correct, total = per_cell_accuracy(
            params, n_grids=n_grids, holes=holes, seed=seed + holes
        )
        exact = 0
        consistent = 0
        for i in range(n_unique):
            puzzle, solution = generate_puzzle(
                seed=seed + 10_000 + holes * 10 + i, holes=holes
            )
            filled, _, _st = full_solve(puzzle, params, tau=tau)
            if is_complete(filled) and is_consistent(filled):
                consistent += 1
            if filled == solution:
                exact += 1
        print(
            f"holes={holes:2d} | per-cell {acc:.3f} ({correct}/{total}) | "
            f"unique exact={exact}/{n_unique} consistent={consistent}/{n_unique}"
        )


if __name__ == "__main__":
    params, logs = train_curriculum()
    for row in logs:
        print(
            f"stage holes={row['holes']}: samples={row['samples']} "
            f"val_acc={row['final_val_acc']:.4f}"
        )
    eval_curriculum(params)
    print("curriculum ok")
