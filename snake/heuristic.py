"""Greedy snake bot: never suicide if a safe move exists; else chase food."""

from __future__ import annotations

from game import DIRS, SnakeGame
from encode import (
    _rotate,
    absolute_dir_to_relative,
    action_to_direction,
    is_danger,
)


def safe_directions(game: SnakeGame) -> list[tuple[int, int]]:
    """Absolute directions that don't die in one step (excludes 180° reverse)."""
    out = []
    for which in ("left", "straight", "right"):
        d = _rotate(game.direction, which)
        if not is_danger(game, d):
            out.append(d)
    return out


def toward_food(game: SnakeGame, direction: tuple[int, int]) -> int:
    """How much this step reduces Manhattan distance to food (higher = better)."""
    if game.food is None:
        return 0
    hr, hc = game.snake[0]
    fr, fc = game.food
    before = abs(hr - fr) + abs(hc - fc)
    dr, dc = direction
    nr, nc = hr + dr, hc + dc
    after = abs(nr - fr) + abs(nc - fc)
    return before - after


def choose_direction(game: SnakeGame) -> tuple[int, int]:
    """
    Pick an absolute direction:
      1. among safe relative moves, maximize progress toward food
      2. if none safe, still pick relative move that maximizes progress
         (forced suicide / trapped — label whatever looks least bad)
    """
    candidates = safe_directions(game)
    if not candidates:
        candidates = [
            _rotate(game.direction, w) for w in ("left", "straight", "right")
        ]

    best = candidates[0]
    best_score = toward_food(game, best)
    for d in candidates[1:]:
        s = toward_food(game, d)
        if s > best_score:
            best, best_score = d, s
    return best


def choose_action(game: SnakeGame) -> int:
    """Relative action 0=left, 1=straight, 2=right."""
    return absolute_dir_to_relative(game, choose_direction(game))


if __name__ == "__main__":
    from game import RIGHT, UP, new_game

    # Open board: food to the right → go straight
    g = new_game(width=8, height=8, seed=0)
    g.snake = [(4, 2), (4, 1), (4, 0)]
    g.direction = RIGHT
    g.food = (4, 6)
    d = choose_direction(g)
    assert d == RIGHT
    assert choose_action(g) == 1

    # Straight is wall; food is up → turn left (facing right → left is UP)
    g2 = new_game(width=5, height=5, seed=1)
    g2.snake = [(2, 4), (2, 3), (2, 2)]
    g2.direction = RIGHT
    g2.food = (0, 4)
    assert is_danger(g2, RIGHT)
    d2 = choose_direction(g2)
    assert d2 == UP
    assert not is_danger(g2, d2)

    # Rollout: should survive a while and often eat
    g3 = new_game(width=10, height=10, seed=2)
    steps = 0
    max_steps = 200
    while g3.alive and steps < max_steps:
        g3.step(choose_direction(g3))
        steps += 1
    print(f"rollout steps={steps} score={g3.score} alive={g3.alive}")
    assert steps > 10
    print("heuristic ok")
