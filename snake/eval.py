"""Compare net vs random (and heuristic): mean steps survived and score."""

from pathlib import Path

import numpy as np

from game import DIRS, OPPOSITE, new_game
from heuristic import choose_direction
from nn import predict_direction
from train import load_weights

out_dir = Path(__file__).resolve().parent


def random_direction(game, rng):
    banned = OPPOSITE[game.direction]
    choices = [d for d in DIRS if d != banned]
    return choices[int(rng.integers(0, len(choices)))]


def run_episode(policy, seed, width=10, height=10, max_steps=500):
    game = new_game(width=width, height=height, seed=seed)
    steps = 0
    while game.alive and steps < max_steps:
        game.step(policy(game))
        steps += 1
    return {"steps": steps, "score": game.score}


def summarize(rows):
    steps = np.array([r["steps"] for r in rows], dtype=float)
    scores = np.array([r["score"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "steps_mean": float(steps.mean()),
        "steps_std": float(steps.std()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_max": int(scores.max()),
    }


def evaluate(params, n_games=40, seed=0, max_steps=500):
    def net_policy(game):
        return predict_direction(game, params, mask_danger=True)[0]

    out = {}
    for name in ("random", "net", "heuristic"):
        rows = []
        for i in range(n_games):
            if name == "random":
                rng = np.random.default_rng(seed + 10_000 + i)

                def policy(game, _rng=rng):
                    return random_direction(game, _rng)

            elif name == "net":
                policy = net_policy
            else:
                policy = choose_direction
            rows.append(run_episode(policy, seed=seed + i, max_steps=max_steps))
        out[name] = summarize(rows)
    return out


if __name__ == "__main__":
    weights = out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit("missing weights.npz — run train.py first")

    params = load_weights(weights)
    stats = evaluate(params, n_games=40, seed=0)
    for name in ("random", "net", "heuristic"):
        s = stats[name]
        print(
            f"{name:10s} | steps {s['steps_mean']:.1f}±{s['steps_std']:.1f} | "
            f"score {s['score_mean']:.2f}±{s['score_std']:.2f} | max {s['score_max']}"
        )
    if stats["net"]["score_mean"] <= stats["random"]["score_mean"]:
        print("warning: net did not beat random on mean score")
    else:
        print("net beat random on mean score")
    print("eval ok")
