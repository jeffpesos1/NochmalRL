"""Board definition and static move tables for *Noch mal!*.

A board is described by a JSON file (see ``boards/*.json``).  Everything that
depends on the board layout – colour regions, the set of all connected cell
groups a player could ever mark in one turn, column/colour bookkeeping
matrices – is pre-computed once here and handed to the vectorised environment
and the neural network as tensors.  Nothing else in the project hard-codes the
layout, so any rectangular board with any number of colours works.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

JOKER = "joker"


@dataclass
class Board:
    name: str
    width: int
    height: int
    columns: List[str]
    start_column: int
    color_keys: List[str]                 # index -> key, e.g. ["g","y","b","r","o"]
    color_names: Dict[str, str]
    color_hex: Dict[str, str]
    cell_color: np.ndarray                # (N,) int, colour index of each cell
    cell_star: np.ndarray                 # (N,) bool
    column_points_first: np.ndarray       # (W,)
    column_points_second: np.ndarray      # (W,)
    color_points_first: int
    color_points_second: int
    jokers: int
    joker_bonus: int
    star_penalty: int
    color_die: List[str]                  # faces, colour keys or "joker"
    number_die: List                      # faces, ints or "joker"
    end_colors: int
    description: str = ""
    source: str = ""
    # derived -------------------------------------------------------------
    cell_region: np.ndarray = field(default=None, repr=False)
    region_size: np.ndarray = field(default=None, repr=False)
    region_color: np.ndarray = field(default=None, repr=False)
    moves: np.ndarray = field(default=None, repr=False)         # (K,N) bool
    move_color: np.ndarray = field(default=None, repr=False)    # (K,)
    move_size: np.ndarray = field(default=None, repr=False)     # (K,)
    move_region: np.ndarray = field(default=None, repr=False)   # (K,)
    move_stars: np.ndarray = field(default=None, repr=False)    # (K,)
    move_cells: List[Tuple[int, ...]] = field(default=None, repr=False)
    adjacency: np.ndarray = field(default=None, repr=False)     # (N,N) bool

    # ------------------------------------------------------------------ basics
    @property
    def n_cells(self) -> int:
        return self.width * self.height

    @property
    def n_colors(self) -> int:
        return len(self.color_keys)

    @property
    def n_moves(self) -> int:
        return int(self.moves.shape[0])

    @property
    def max_mark(self) -> int:
        return max(f for f in self.number_die if f != JOKER)

    @property
    def number_faces(self) -> List[int]:
        """Distinct numeric faces of the number die, sorted (e.g. [1,2,3,4,5])."""
        return sorted({f for f in self.number_die if f != JOKER})

    @property
    def n_color_faces(self) -> int:
        return self.n_colors + 1          # + joker

    @property
    def n_number_faces(self) -> int:
        return len(self.number_faces) + 1  # + joker

    def cell_index(self, row: int, col: int) -> int:
        return row * self.width + col

    def cell_rc(self, idx: int) -> Tuple[int, int]:
        return divmod(int(idx), self.width)

    # ------------------------------------------------------------------ dice
    def color_face_probs(self) -> np.ndarray:
        """Probability of each colour-face index (0..n_colors-1 colours, n_colors = joker)."""
        p = np.zeros(self.n_color_faces)
        for f in self.color_die:
            p[self.n_colors if f == JOKER else self.color_keys.index(f)] += 1
        return p / p.sum()

    def number_face_probs(self) -> np.ndarray:
        """Probability of each number-face index (0..len-1 numbers, last = joker)."""
        faces = self.number_faces
        p = np.zeros(self.n_number_faces)
        for f in self.number_die:
            p[len(faces) if f == JOKER else faces.index(f)] += 1
        return p / p.sum()

    def size_to_number_face(self) -> np.ndarray:
        """Map mark-count (1..max_mark) -> number-face index, -1 when no such face."""
        faces = self.number_faces
        m = -np.ones(self.max_mark + 1, dtype=np.int64)
        for i, f in enumerate(faces):
            m[f] = i
        return m

    # ------------------------------------------------------------------ setup
    def finalize(self) -> "Board":
        self._compute_regions()
        self._compute_adjacency()
        self._enumerate_moves()
        return self

    def _neighbors(self, idx: int) -> List[int]:
        r, c = self.cell_rc(idx)
        out = []
        if r > 0:
            out.append(idx - self.width)
        if r < self.height - 1:
            out.append(idx + self.width)
        if c > 0:
            out.append(idx - 1)
        if c < self.width - 1:
            out.append(idx + 1)
        return out

    def _compute_adjacency(self) -> None:
        n = self.n_cells
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in self._neighbors(i):
                adj[i, j] = True
        self.adjacency = adj

    def _compute_regions(self) -> None:
        n = self.n_cells
        region = -np.ones(n, dtype=np.int64)
        sizes, colors = [], []
        for start in range(n):
            if region[start] >= 0:
                continue
            rid = len(sizes)
            stack = [start]
            region[start] = rid
            cnt = 0
            while stack:
                cur = stack.pop()
                cnt += 1
                for nb in self._neighbors(cur):
                    if region[nb] < 0 and self.cell_color[nb] == self.cell_color[start]:
                        region[nb] = rid
                        stack.append(nb)
            sizes.append(cnt)
            colors.append(int(self.cell_color[start]))
        self.cell_region = region
        self.region_size = np.array(sizes, dtype=np.int64)
        self.region_color = np.array(colors, dtype=np.int64)

    def _enumerate_moves(self) -> None:
        """All orthogonally-connected cell groups of 1..max_mark cells inside one colour block."""
        max_size = self.max_mark
        allowed_sizes = set(self.number_faces)
        n = self.n_cells
        groups: List[Tuple[int, ...]] = []
        for rid in range(len(self.region_size)):
            cells = [i for i in range(n) if self.cell_region[i] == rid]
            cellset = set(cells)
            frontier = {frozenset([c]) for c in cells}
            seen = set(frontier)
            level = 1
            while frontier and level <= max_size:
                if level in allowed_sizes:
                    groups.extend(tuple(sorted(g)) for g in frontier)
                nxt = set()
                for g in frontier:
                    for c in g:
                        for nb in self._neighbors(c):
                            if nb in cellset and nb not in g:
                                h = g | {nb}
                                if h not in seen:
                                    seen.add(h)
                                    nxt.add(h)
                frontier = nxt
                level += 1
        groups.sort(key=lambda g: (len(g), g))
        k = len(groups)
        moves = np.zeros((k, n), dtype=bool)
        for i, g in enumerate(groups):
            moves[i, list(g)] = True
        self.moves = moves
        self.move_cells = groups
        self.move_size = moves.sum(1).astype(np.int64)
        first = np.array([g[0] for g in groups], dtype=np.int64)
        self.move_region = self.cell_region[first]
        self.move_color = self.cell_color[first]
        self.move_stars = (moves & self.cell_star[None, :]).sum(1).astype(np.int64)

    # ------------------------------------------------------------------ helpers
    def find_move(self, cells: Sequence[int]) -> int:
        """Index of the move marking exactly ``cells`` or -1."""
        key = tuple(sorted(int(c) for c in cells))
        try:
            return self._move_lookup[key]
        except AttributeError:
            self._move_lookup = {g: i for i, g in enumerate(self.move_cells)}
            return self._move_lookup.get(key, -1)
        except KeyError:
            return -1

    def to_public_dict(self) -> dict:
        """JSON-serialisable description for the web UI."""
        return {
            "name": self.name,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "columns": self.columns,
            "start_column": self.start_column,
            "color_keys": self.color_keys,
            "color_names": self.color_names,
            "color_hex": self.color_hex,
            "cell_color": self.cell_color.tolist(),
            "cell_star": self.cell_star.astype(int).tolist(),
            "cell_region": self.cell_region.tolist(),
            "column_points_first": self.column_points_first.tolist(),
            "column_points_second": self.column_points_second.tolist(),
            "color_points_first": self.color_points_first,
            "color_points_second": self.color_points_second,
            "jokers": self.jokers,
            "joker_bonus": self.joker_bonus,
            "star_penalty": self.star_penalty,
            "color_die": self.color_die,
            "number_die": self.number_die,
            "number_faces": self.number_faces,
            "end_colors": self.end_colors,
            "n_moves": self.n_moves,
        }


def load_board(path_or_name: str, boards_dir: str | None = None) -> Board:
    """Load a board from a JSON file path or a name inside ``boards_dir``."""
    if boards_dir is None:
        boards_dir = os.environ.get(
            "NOCHMAL_BOARDS", os.path.join(os.path.dirname(os.path.dirname(__file__)), "boards")
        )
    path = path_or_name
    if not os.path.isfile(path):
        cand = os.path.join(boards_dir, path_or_name)
        if os.path.isfile(cand):
            path = cand
        elif os.path.isfile(cand + ".json"):
            path = cand + ".json"
        else:
            raise FileNotFoundError(f"board '{path_or_name}' not found (looked in {boards_dir})")
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return board_from_spec(spec, source=os.path.splitext(os.path.basename(path))[0])


def board_from_spec(spec: dict, source: str = "") -> Board:
    rows = spec["rows"]
    height = len(rows)
    width = len(rows[0])
    assert all(len(r) == width for r in rows), "all rows must have the same length"
    assert spec.get("height", height) == height and spec.get("width", width) == width
    color_keys = list(spec["colors"].keys())
    cell_color = np.array([color_keys.index(ch) for r in rows for ch in r], dtype=np.int64)
    stars = spec.get("stars", ["." * width] * height)
    if isinstance(stars[0], str):
        cell_star = np.array([ch == "*" for r in stars for ch in r], dtype=bool)
    else:  # list of [row, col]
        cell_star = np.zeros(width * height, dtype=bool)
        for r, c in stars:
            cell_star[r * width + c] = True
    columns = spec.get("columns") or [chr(ord("A") + i) for i in range(width)]
    start_column = spec.get("start_column", width // 2)
    number_die = [f if f == JOKER else int(f) for f in spec.get("number_die", [1, 2, 3, 4, 5, JOKER])]
    b = Board(
        name=spec.get("name", source or "board"),
        description=spec.get("description", ""),
        source=source,
        width=width,
        height=height,
        columns=columns,
        start_column=start_column,
        color_keys=color_keys,
        color_names={k: v.get("name", k) for k, v in spec["colors"].items()},
        color_hex={k: v.get("hex", "#888888") for k, v in spec["colors"].items()},
        cell_color=cell_color,
        cell_star=cell_star,
        column_points_first=np.array(spec["column_points_first"], dtype=np.int64),
        column_points_second=np.array(spec["column_points_second"], dtype=np.int64),
        color_points_first=int(spec.get("color_points_first", 5)),
        color_points_second=int(spec.get("color_points_second", 3)),
        jokers=int(spec.get("jokers", 8)),
        joker_bonus=int(spec.get("joker_bonus", 1)),
        star_penalty=int(spec.get("star_penalty", 2)),
        color_die=list(spec.get("color_die", color_keys + [JOKER])),
        number_die=number_die,
        end_colors=int(spec.get("end_colors", 2)),
    )
    assert len(b.column_points_first) == width and len(b.column_points_second) == width
    for f in b.color_die:
        assert f == JOKER or f in color_keys, f"unknown colour face {f}"
    return b.finalize()


def list_boards(boards_dir: str | None = None) -> List[dict]:
    if boards_dir is None:
        boards_dir = os.environ.get(
            "NOCHMAL_BOARDS", os.path.join(os.path.dirname(os.path.dirname(__file__)), "boards")
        )
    out = []
    for fn in sorted(os.listdir(boards_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(boards_dir, fn), "r", encoding="utf-8") as f:
                spec = json.load(f)
            out.append({"id": fn[:-5], "name": spec.get("name", fn[:-5]), "description": spec.get("description", ""),
                        "official": bool(spec.get("official", False)), "sheet_color": spec.get("sheet_color"),
                        "order": spec.get("order", 999)})
    out.sort(key=lambda b: (not b["official"], b["order"], b["id"]))
    return out


def official_board_ids(boards_dir: str | None = None) -> List[str]:
    return [b["id"] for b in list_boards(boards_dir) if b["official"]]
