"""Watch net / heuristic / random play Snake in the terminal."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from game import DIRS, DIR_NAMES, OPPOSITE, new_game
from heuristic import choose_direction
from nn import predict_direction
from train import load_weights

out_dir = Path(__file__).resolve().parent


def random_direction(game, rng):
    banned = OPPOSITE[game.direction]
    choices = [d for d in DIRS if d != banned]
    return choices[int(rng.integers(0, len(choices)))]


def clear():
    print("\033[2J\033[H", end="")


def play(
    policy_name: str = "net",
    seed: int = 0,
    delay: float = 0.12,
    max_steps: int = 500,
    width: int = 10,
    height: int = 10,
):
    if policy_name == "net":
        weights = out_dir / "weights.npz"
        if not weights.exists():
            raise SystemExit("missing weights.npz — run train.py first")
        params = load_weights(weights)

        def policy(game):
            return predict_direction(game, params, mask_danger=True)[0]

    elif policy_name == "heuristic":
        policy = choose_direction
    elif policy_name == "random":
        rng = np.random.default_rng(seed + 99)

        def policy(game, _rng=rng):
            return random_direction(game, _rng)

    else:
        raise SystemExit(f"unknown policy {policy_name}")

    game = new_game(width=width, height=height, seed=seed)
    steps = 0
    try:
        while game.alive and steps < max_steps:
            clear()
            print(f"snake | policy={policy_name} | step={steps} | score={game.score}")
            print(f"heading={DIR_NAMES[game.direction]}")
            print()
            print(game.render())
            print()
            print("H=head  o=body  *=food  .=empty   Ctrl+C to quit")
            direction = policy(game)
            game.step(direction)
            steps += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nstopped.")
        return

    clear()
    print(game.render())
    print()
    if game.alive:
        print(f"hit max_steps={max_steps} | score={game.score}")
    else:
        print(f"dead after {steps} steps | score={game.score}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render Snake in the terminal")
    p.add_argument(
        "--policy",
        choices=("net", "heuristic", "random"),
        default="net",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.12, help="seconds between frames")
    p.add_argument("--max-steps", type=int, default=500)
    args = p.parse_args()
    play(
        policy_name=args.policy,
        seed=args.seed,
        delay=args.delay,
        max_steps=args.max_steps,
    )
