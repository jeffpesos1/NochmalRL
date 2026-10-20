"""Random *Noch mal!*-style board generator (used by scripts/make_board.py and ``--random-boards``)."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .board import Board, board_from_spec

COLORS = {"g": ("grün", "#8cc63f"), "y": ("gelb", "#f7d23e"), "b": ("blau", "#5bc0eb"), "r": ("rot", "#e8378b"), "o": ("orange", "#f28c28")}


def neighbors(i: int, w: int, h: int) -> List[int]:
    r, c = divmod(i, w)
    out = []
    if r > 0:
        out.append(i - w)
    if r < h - 1:
        out.append(i + w)
    if c > 0:
        out.append(i - 1)
    if c < w - 1:
        out.append(i + 1)
    return out


def generate(width: int, height: int, max_region: int, n_stars: int, seed: int, colors=COLORS):
    """Grow colour blocks of 1..max_region cells until the grid is full; balanced colour counts; stars."""
    rnd = random.Random(seed)
    n = width * height
    keys = list(colors)
    for _ in range(5000):
        grid = [None] * n
        counts = {k: 0 for k in keys}
        order = list(range(n))
        rnd.shuffle(order)
        ok = True
        for start in order:
            if grid[start] is not None:
                continue
            nbcols = {grid[nb] for nb in neighbors(start, width, height) if grid[nb] is not None}
            cands = [k for k in keys if k not in nbcols]
            if not cands:
                ok = False
                break
            cands.sort(key=lambda k: counts[k])
            color = rnd.choice(cands[: max(1, len(cands) // 2)])
            size = rnd.randint(1, max_region)
            block = [start]
            grid[start] = color
            frontier = [nb for nb in neighbors(start, width, height) if grid[nb] is None]
            while len(block) < size and frontier:
                c = rnd.choice(frontier)
                frontier.remove(c)
                if grid[c] is not None:
                    continue
                if any(grid[nb] == color and nb not in block for nb in neighbors(c, width, height)):
                    continue
                grid[c] = color
                block.append(c)
                frontier += [nb for nb in neighbors(c, width, height) if grid[nb] is None and nb not in frontier]
            counts[color] += len(block)
        if ok and max(counts.values()) - min(counts.values()) <= 4:
            stars = set()
            while len(stars) < n_stars:
                stars.add(rnd.randrange(n))
            return grid, stars
    raise RuntimeError("could not generate a balanced board; try another seed")


def column_values(width: int) -> Tuple[List[int], List[int]]:
    """Symmetric values: outer columns 5/3, then 3/2, inner 2/1, start column 1/0."""
    first, second = [], []
    mid = width // 2
    for i in range(width):
        d = abs(i - mid)
        if d == 0:
            f, s = 1, 0
        elif d <= mid * 0.45:
            f, s = 2, 1
        elif d < mid:
            f, s = 3, 2
        else:
            f, s = 5, 3
        first.append(f)
        second.append(s)
    return first, second


def random_spec(seed: int, width: int = 15, height: int = 7, max_region: int = 6, n_stars: int = 15,
                name: str | None = None) -> Dict:
    grid, stars = generate(width, height, max_region, n_stars, seed)
    rows = ["".join(grid[r * width:(r + 1) * width]) for r in range(height)]
    star_rows = ["".join("*" if r * width + c in stars else "." for c in range(width)) for r in range(height)]
    first, second = column_values(width)
    return {
        "name": name or f"Zufallsblatt #{seed} ({width}x{height})",
        "description": "Zufällig generiertes Spielblatt (nochmal.boardgen) – kein offizielles Layout.",
        "width": width, "height": height,
        "columns": [chr(ord("A") + i) for i in range(width)],
        "start_column": width // 2,
        "colors": {k: {"name": v[0], "hex": v[1]} for k, v in COLORS.items()},
        "rows": rows, "stars": star_rows,
        "column_points_first": first, "column_points_second": second,
        "color_points_first": 5, "color_points_second": 3,
        "jokers": 8, "joker_bonus": 1, "star_penalty": 2,
        "color_die": list(COLORS) + ["joker"], "number_die": [1, 2, 3, 4, 5, "joker"],
        "end_colors": 2,
    }


def random_board(seed: int, **kw) -> Board:
    return board_from_spec(random_spec(seed, **kw), source=f"random-seed{seed}")


def random_boards(n: int, seed: int = 0, **kw) -> List[Board]:
    return [random_board(seed * 100003 + i, **kw) for i in range(n)]
