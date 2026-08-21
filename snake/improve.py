"""REINFORCE fine-tune: improve cloned policy with reward signals.

Rewards per step:
  +1.0  ate food
  -1.0  died
  +0.01 survived a step

Updates: advantage-weighted cross-entropy (REINFORCE with mean baseline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from encode import FEATURE_SIZE, N_ACTIONS, encode
from game import new_game
from nn import (
    add_grads,
    backward,
    forward,
    sample_direction,
    scale_grads,
    sgd_step,
)
from train import load_weights, save_weights

out_dir = Path(__file__).resolve().parent
IMPROVED_PATH = out_dir / "weights_improved.npz"

R_FOOD = 1.0
R_DIE = -1.0
R_STEP = 0.01
GAMMA = 0.95


def step_reward(info: dict) -> float:
    if not info["alive"]:
        return R_DIE
    r = R_STEP
    if info["ate"]:
        r += R_FOOD
    return r


def discounted_returns(rewards, gamma=GAMMA):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return np.array(out, dtype=np.float64)


def collect_episode(params, seed, rng, max_steps=400):
    game = new_game(width=10, height=10, seed=seed)
    states = []
    actions = []
    rewards = []
    steps = 0
    while game.alive and steps < max_steps:
        feats = encode(game)
        direction, _, idx = sample_direction(game, params, rng, mask_danger=True)
        states.append(feats)
        actions.append(idx)
        info = game.step(direction)
        rewards.append(step_reward(info))
        steps += 1
    returns = discounted_returns(rewards)
    return {
        "states": np.stack(states, axis=0) if states else np.zeros((0, FEATURE_SIZE)),
        "actions": np.array(actions, dtype=int),
        "returns": returns,
        "score": game.score,
        "steps": steps,
    }


def reinforce_update(params, batch_episodes, lr=0.01):
    """One policy-gradient step over a list of episode dicts."""
    all_G = np.concatenate([ep["returns"] for ep in batch_episodes if len(ep["returns"])])
    if len(all_G) == 0:
        return params, 0.0
    baseline = float(all_G.mean())

    acc = None
    n = 0
    for ep in batch_episodes:
        if len(ep["returns"]) == 0:
            continue
        for t in range(len(ep["returns"])):
            x = ep["states"][t : t + 1]
            a = ep["actions"][t]
            adv = ep["returns"][t] - baseline
            if adv == 0:
                continue
            y = np.zeros((1, N_ACTIONS))
            y[0, a] = 1.0
            pred, cache = forward(x, params)
            grads = backward(pred, y, cache, params)
            # CE grads are d(-log π)/dW. Ascent on A·log π ⇒ params -= lr · A · g_ce
            acc = add_grads(acc, scale_grads(grads, adv))
            n += 1

    if acc is None or n == 0:
        return params, baseline
    for k in acc:
        acc[k] = acc[k] / n
    params = sgd_step(params, acc, lr)
    return params, baseline


def improve(
    params,
    iterations=40,
    episodes_per_iter=16,
    lr=0.02,
    seed=0,
):
    rng = np.random.default_rng(seed)
    history = []
    for it in range(iterations):
        batch = [
            collect_episode(params, seed=seed + it * 1000 + i, rng=rng)
            for i in range(episodes_per_iter)
        ]
        mean_score = float(np.mean([ep["score"] for ep in batch]))
        mean_steps = float(np.mean([ep["steps"] for ep in batch]))
        params, baseline = reinforce_update(params, batch, lr=lr)
        history.append({"score": mean_score, "steps": mean_steps, "baseline": baseline})
        if (it + 1) % 5 == 0 or it == 0:
            print(
                f"iter {it + 1:3d} | score {mean_score:.2f} | "
                f"steps {mean_steps:.1f} | baseline {baseline:.3f}"
            )
    return params, history


if __name__ == "__main__":
    weights = out_dir / "weights.npz"
    if not weights.exists():
        raise SystemExit("missing weights.npz — run train.py first")

    params = load_weights(weights)
    print("REINFORCE fine-tune from cloned weights…")
    params, history = improve(params, iterations=40, episodes_per_iter=16, lr=0.02)
    path = save_weights(params, IMPROVED_PATH)
    print(f"wrote {path}")
    print(f"start score {history[0]['score']:.2f} → end {history[-1]['score']:.2f}")
    print("improve ok")
    print("play with: python snake/play.py --weights snake/weights_improved.npz")
