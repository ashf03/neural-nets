"""Snake grid game: move, eat/grow, die on wall or self."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# row, col deltas
UP, DOWN, LEFT, RIGHT = (-1, 0), (1, 0), (0, -1), (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
DIR_NAMES = {UP: "U", DOWN: "D", LEFT: "L", RIGHT: "R"}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


@dataclass
class SnakeGame:
    width: int = 10
    height: int = 10
    snake: list[tuple[int, int]] = field(default_factory=list)
    food: tuple[int, int] | None = None
    direction: tuple[int, int] = RIGHT
    alive: bool = True
    score: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        if not self.snake:
            mid_r, mid_c = self.height // 2, self.width // 2
            # head first, facing right
            self.snake = [(mid_r, mid_c), (mid_r, mid_c - 1), (mid_r, mid_c - 2)]
            self.direction = RIGHT
        if self.food is None:
            self.food = self._spawn_food()

    def cells(self) -> set[tuple[int, int]]:
        return set(self.snake)

    def _spawn_food(self) -> tuple[int, int] | None:
        free = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.cells()
        ]
        if not free:
            return None
        return self.rng.choice(free)

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def step(self, direction: tuple[int, int] | None = None) -> dict:
        """
        Advance one tick. Optional new direction (ignored if 180° reverse).
        Returns {"alive", "ate", "score", "won"}.
        """
        if not self.alive:
            return {"alive": False, "ate": False, "score": self.score, "won": False}

        if direction is not None and direction in DIRS:
            if direction != OPPOSITE[self.direction]:
                self.direction = direction

        hr, hc = self.snake[0]
        dr, dc = self.direction
        new_head = (hr + dr, hc + dc)

        # wall
        if not self.in_bounds(new_head):
            self.alive = False
            return {"alive": False, "ate": False, "score": self.score, "won": False}

        # self — allow moving into current tail only if we are not growing
        will_eat = self.food is not None and new_head == self.food
        body = self.snake if will_eat else self.snake[:-1]
        if new_head in body:
            self.alive = False
            return {"alive": False, "ate": False, "score": self.score, "won": False}

        self.snake.insert(0, new_head)
        ate = False
        if will_eat:
            ate = True
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                # filled the grid
                return {"alive": True, "ate": True, "score": self.score, "won": True}
        else:
            self.snake.pop()

        return {"alive": True, "ate": ate, "score": self.score, "won": False}

    def render(self) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        if self.food is not None:
            fr, fc = self.food
            grid[fr][fc] = "*"
        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = "H" if i == 0 else "o"
        return "\n".join(" ".join(row) for row in grid)


def new_game(width=10, height=10, seed=None) -> SnakeGame:
    rng = random.Random(seed)
    return SnakeGame(width=width, height=height, rng=rng)


if __name__ == "__main__":
    g = new_game(seed=0)
    assert g.alive and len(g.snake) == 3
    assert g.food is not None

    # crawl right into food or empty — keep moving until we hit something or 50 steps
    start_len = len(g.snake)
    for _ in range(50):
        info = g.step(RIGHT)
        if not info["alive"]:
            break
        if info["ate"]:
            assert len(g.snake) == start_len + 1
            break
    else:
        # no food eaten in 50 steps — still fine; just check wall death separately
        pass

    # wall death
    g2 = new_game(width=5, height=5, seed=1)
    g2.snake = [(0, 2), (0, 1), (0, 0)]
    g2.direction = UP
    g2.food = (4, 4)
    info = g2.step(UP)
    assert info["alive"] is False

    # self collision
    g3 = new_game(width=5, height=5, seed=2)
    g3.snake = [(2, 2), (2, 1), (2, 0), (1, 0), (1, 1), (1, 2), (1, 3)]
    g3.direction = UP
    g3.food = (4, 4)
    # head at (2,2) going UP → (1,2) which is in body
    info = g3.step(UP)
    assert info["alive"] is False

    g4 = new_game(seed=3)
    print(g4.render())
    print("score:", g4.score, "dir:", DIR_NAMES[g4.direction])
    print("game ok")
