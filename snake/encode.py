"""Encode SnakeGame → feature vector for the net.

Layout (11 floats):
  [0:3]  danger relative to heading: left, straight, right (1 = would die)
  [3:7]  food direction: up, down, left, right (1 if food is that way; both axes can be 1)
  [7:11] current heading one-hot: U, D, L, R
"""

from __future__ import annotations

import numpy as np

from game import DOWN, DIRS, LEFT, OPPOSITE, RIGHT, UP, SnakeGame

# turn left / right in grid coords (row, col)
_LEFT_OF = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
_RIGHT_OF = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

HEADING_ORDER = (UP, DOWN, LEFT, RIGHT)  # matches one-hot slots 7..10


def _rotate(direction: tuple[int, int], which: str) -> tuple[int, int]:
    if which == "left":
        return _LEFT_OF[direction]
    if which == "right":
        return _RIGHT_OF[direction]
    return direction  # straight


def is_danger(game: SnakeGame, direction: tuple[int, int]) -> bool:
    """True if one step in `direction` hits wall or body (tail cell frees if not growing)."""
    hr, hc = game.snake[0]
    dr, dc = direction
    nxt = (hr + dr, hc + dc)
    if not game.in_bounds(nxt):
        return True
    will_eat = game.food is not None and nxt == game.food
    body = game.snake if will_eat else game.snake[:-1]
    return nxt in body


def food_dir_flags(game: SnakeGame) -> tuple[float, float, float, float]:
    """up, down, left, right — 1 if food is strictly that way from head."""
    if game.food is None:
        return (0.0, 0.0, 0.0, 0.0)
    hr, hc = game.snake[0]
    fr, fc = game.food
    up = 1.0 if fr < hr else 0.0
    down = 1.0 if fr > hr else 0.0
    left = 1.0 if fc < hc else 0.0
    right = 1.0 if fc > hc else 0.0
    return (up, down, left, right)


def heading_one_hot(direction: tuple[int, int]) -> tuple[float, float, float, float]:
    return tuple(1.0 if direction == d else 0.0 for d in HEADING_ORDER)


def encode(game: SnakeGame) -> np.ndarray:
    """Return shape (11,) float64 feature vector."""
    heading = game.direction
    danger_l = 1.0 if is_danger(game, _rotate(heading, "left")) else 0.0
    danger_s = 1.0 if is_danger(game, _rotate(heading, "straight")) else 0.0
    danger_r = 1.0 if is_danger(game, _rotate(heading, "right")) else 0.0
    fu, fd, fl, fr = food_dir_flags(game)
    hu, hd, hl, hr = heading_one_hot(heading)
    return np.array(
        [danger_l, danger_s, danger_r, fu, fd, fl, fr, hu, hd, hl, hr],
        dtype=np.float64,
    )


def action_to_direction(game: SnakeGame, action: int) -> tuple[int, int]:
    """
    action: 0=turn left, 1=straight, 2=turn right (relative to current heading).
    """
    if action == 0:
        return _rotate(game.direction, "left")
    if action == 2:
        return _rotate(game.direction, "right")
    return game.direction


def absolute_dir_to_relative(game: SnakeGame, direction: tuple[int, int]) -> int:
    """Map absolute U/D/L/R to relative action 0/1/2. Opposite → straight (no-op)."""
    if direction == OPPOSITE[game.direction]:
        return 1
    if direction == _rotate(game.direction, "left"):
        return 0
    if direction == _rotate(game.direction, "right"):
        return 2
    return 1


FEATURE_SIZE = 11
N_ACTIONS = 4  # absolute U, D, L, R (same order as HEADING_ORDER)
INDEX_TO_DIR = HEADING_ORDER
DIR_TO_INDEX = {d: i for i, d in enumerate(INDEX_TO_DIR)}


def one_hot_dir(direction: tuple[int, int]) -> np.ndarray:
    y = np.zeros(N_ACTIONS, dtype=np.float64)
    y[DIR_TO_INDEX[direction]] = 1.0
    return y


if __name__ == "__main__":
    from game import new_game

    g = new_game(width=5, height=5, seed=0)
    g.snake = [(2, 2), (2, 1), (2, 0)]
    g.direction = RIGHT
    g.food = (0, 4)  # up and right of head

    v = encode(g)
    assert v.shape == (FEATURE_SIZE,)
    # facing right at (2,2): left=up toward (1,2) clear; straight=(2,3) clear; right=down clear
    assert v[0] == 0 and v[1] == 0 and v[2] == 0
    assert v[3] == 1 and v[4] == 0 and v[5] == 0 and v[6] == 1  # food up+right
    assert list(v[7:11]) == [0, 0, 0, 1]  # heading RIGHT

    # wall straight ahead
    g2 = new_game(width=3, height=3, seed=1)
    g2.snake = [(1, 2), (1, 1), (1, 0)]
    g2.direction = RIGHT
    g2.food = (0, 0)
    v2 = encode(g2)
    assert v2[1] == 1.0  # danger straight

    assert action_to_direction(g, 0) == UP
    assert action_to_direction(g, 1) == RIGHT
    assert action_to_direction(g, 2) == DOWN

    print("encode:", v)
    print("encode ok")
