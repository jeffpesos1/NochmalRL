#!/usr/bin/env python3
"""Generate a random *Noch mal!*-style board JSON (for testing board-agnostic training).

Usage: python scripts/make_board.py --out boards/random1.json --seed 1 [--width 15 --height 7 --max-region 6 --stars 15]

The generator grows colour blocks of 1..max-region cells until the grid is full, keeps the
colour counts balanced, places stars and uses the official column values pattern (scaled to width).
"""
import argparse
import json
import random

COLORS = {"g": ("grün", "#8cc63f"), "y": ("gelb", "#f7d23e"), "b": ("blau", "#5bc0eb"), "r": ("rot", "#e8378b"), "o": ("orange", "#f28c28")}


def generate(width, height, max_region, n_stars, seed, colors):
    rnd = random.Random(seed)
    n = width * height
    keys = list(colors)
    for attempt in range(2000):
        grid = [None] * n
        counts = {k: 0 for k in keys}
        order = list(range(n))
        rnd.shuffle(order)
        ok = True
        for start in order:
            if grid[start] is not None:
                continue
            # candidate colours: not equal to any neighbouring block colour (else blocks would merge)
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
                # adding c must not touch another block of the same colour
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


def neighbors(i, w, h):
    r, c = divmod(i, w)
    out = []
    if r > 0: out.append(i - w)
    if r < h - 1: out.append(i + w)
    if c > 0: out.append(i - 1)
    if c < w - 1: out.append(i + 1)
    return out


def column_values(width):
    """Symmetric values: outer columns 5/3, then 3/2, inner 2/1, start column 1/0."""
    first, second = [], []
    mid = width // 2
    for i in range(width):
        d = abs(i - mid)
        if d == 0: f, s = 1, 0
        elif d <= mid * 0.45: f, s = 2, 1
        elif d < mid: f, s = 3, 2
        else: f, s = 5, 3
        first.append(f); second.append(s)
    return first, second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--max-region", type=int, default=6)
    ap.add_argument("--stars", type=int, default=15)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    grid, stars = generate(a.width, a.height, a.max_region, a.stars, a.seed, COLORS)
    rows = ["".join(grid[r * a.width:(r + 1) * a.width]) for r in range(a.height)]
    star_rows = ["".join("*" if r * a.width + c in stars else "." for c in range(a.width)) for r in range(a.height)]
    first, second = column_values(a.width)
    spec = {
        "name": a.name or f"Zufallsblatt #{a.seed} ({a.width}x{a.height})",
        "description": "Zufällig generiertes Spielblatt (scripts/make_board.py) – kein offizielles Layout.",
        "width": a.width, "height": a.height,
        "columns": [chr(ord("A") + i) for i in range(a.width)],
        "start_column": a.width // 2,
        "colors": {k: {"name": v[0], "hex": v[1]} for k, v in COLORS.items()},
        "rows": rows, "stars": star_rows,
        "column_points_first": first, "column_points_second": second,
        "color_points_first": 5, "color_points_second": 3,
        "jokers": 8, "joker_bonus": 1, "star_penalty": 2,
        "color_die": list(COLORS) + ["joker"], "number_die": [1, 2, 3, 4, 5, "joker"],
        "end_colors": 2,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print("\n".join(rows))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
