"""Board tensors on a device, legality masks, per-move features and network inputs.

Everything here is pure tensor arithmetic over a *batch* of decisions, so it
runs identically for 1 decision (web UI) and for 10 000 decisions (training),
on CPU or GPU.  The observation format (``Obs``) is deliberately compact so
that a PPO rollout buffer can store it cheaply.

``BoardTensors`` holds a *pool* of boards (B >= 1) with identical width/height/
colour count.  Every decision carries a board id (``obs["board"]``); all
board-dependent lookups are gathers over that id, so each parallel game can run
on its own board.  Moves are stored as padded cell lists (K_max x 5) instead of
dense masks, which keeps multi-board batches cheap.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np
import torch

from .board import Board

MAX_COLORS = 8          # colour channels / colour-die faces supported by the network
MAX_NUMBER = 8          # number-die faces supported by the network
N_SCALARS = 12          # global scalar features (see make_scalars)
N_DYN_CHANNELS = 9      # dynamic grid channels (see BoardTensors.grid_inputs)
N_MOVE_FEATS = 12       # per-move features (see move_features)
OPP_SCALARS = (4, 6, 10)  # indices in make_scalars that carry opponent information
OPPONENT_VIEWS = ("full", "none")


Obs = Dict[str, torch.Tensor]
# Keys of a compact observation (all tensors have leading batch dim M):
#   marked        (M,N)  bool   own crosses
#   opp_marked    (M,N)  bool   any opponent's crosses
#   opp_best      (M,N)  bool   crosses of the opponent with the highest score
#   pool_c        (M,Cf) uint8  available colour-die faces (counts), joker last
#   pool_n        (M,Nf) uint8  available number-die faces (counts), joker last
#   col_claimed   (M,W)  bool   column's upper value already taken by somebody
#   color_claimed (M,C)  bool   colour's 5-point bonus already taken by somebody
#   jokers_used   (M,)   int64
#   scalars       (M,S)  float32  (see make_scalars)
#   board         (M,)   int64  index into the board pool


class BoardTensors:
    """Static data of one board or a pool of boards as tensors on a device."""

    def __init__(self, boards: Union[Board, Sequence[Board]], device: torch.device | str = "cpu",
                 start_column_always: bool = False):
        """``start_column_always``: True = the start column may begin a new group at any time
        (literal rule-book wording); False = only the very first move uses the start column,
        afterwards crosses must touch an own cross (strict reading, default)."""
        boards = [boards] if isinstance(boards, Board) else list(boards)
        assert boards, "need at least one board"
        b0 = boards[0]
        for b in boards[1:]:
            assert (b.width, b.height, b.n_colors, b.jokers, b.number_faces, b.n_color_faces) == \
                   (b0.width, b0.height, b0.n_colors, b0.jokers, b0.number_faces, b0.n_color_faces), \
                   "all boards in a pool must share size, colour count, dice and joker count"
        self.board = b0
        self.boards = boards
        self.device = torch.device(device)
        self.start_column_always = start_column_always
        d = self.device
        f = torch.float32
        B = len(boards)
        N, W, H, C = b0.n_cells, b0.width, b0.height, b0.n_colors
        self.B, self.N, self.W, self.H, self.C = B, N, W, H, C
        self.Cf = b0.n_color_faces
        self.Nf = b0.n_number_faces
        self.max_mark = b0.max_mark
        self.jokers_max = b0.jokers
        self.end_colors = b0.end_colors
        self.star_penalty = float(b0.star_penalty)
        self.joker_bonus = float(b0.joker_bonus)
        self.color_first = float(b0.color_points_first)
        self.color_second = float(b0.color_points_second)
        self.K = max(b.n_moves for b in boards)
        self.R = max(len(b.region_size) for b in boards)
        self.MAXC = self.max_mark                       # cells per move (padded)
        size_face = torch.as_tensor(b0.size_to_number_face(), device=d)
        self.size_face = size_face

        # grid geometry (shared by all boards of the pool)
        self.cell_col = torch.arange(N, device=d) % W
        self.col_mat = torch.nn.functional.one_hot(self.cell_col, W).to(f)          # (N,W)
        self.col_height = self.col_mat.sum(0)                                        # (W,)
        self.adj_f = torch.as_tensor(b0.adjacency, device=d).to(f)                   # (N,N)
        self.color_face_probs = torch.as_tensor(b0.color_face_probs(), device=d).to(f)
        self.number_face_probs = torch.as_tensor(b0.number_face_probs(), device=d).to(f)

        # per-board cell data (B,N)
        self.cell_color = torch.stack([torch.as_tensor(b.cell_color, device=d) for b in boards])
        self.cell_star = torch.stack([torch.as_tensor(b.cell_star, device=d) for b in boards])
        self.cell_star_f = self.cell_star.to(f)
        self.cell_region = torch.stack([torch.as_tensor(b.cell_region, device=d) for b in boards])
        self.start_col = torch.stack([(self.cell_col == b.start_column) for b in boards])
        self.n_stars = self.cell_star_f.sum(1)                                        # (B,)
        self.region_size = torch.ones(B, self.R, device=d)
        self.color_total = torch.zeros(B, C, device=d)
        self.col_first = torch.stack([torch.as_tensor(b.column_points_first, device=d).to(f) for b in boards])
        self.col_second = torch.stack([torch.as_tensor(b.column_points_second, device=d).to(f) for b in boards])
        for i, b in enumerate(boards):
            self.region_size[i, : len(b.region_size)] = torch.as_tensor(b.region_size, device=d).to(f)
            self.color_total[i] = torch.bincount(torch.as_tensor(b.cell_color, device=d), minlength=C).to(f)

        # per-board move tables, padded to K (invalid padding moves are never legal)
        K, MAXC = self.K, self.MAXC
        move_cells = np.full((B, K, MAXC), N, dtype=np.int64)       # N = dummy cell (always unmarked, zero logit)
        move_valid = np.zeros((B, K), dtype=bool)
        move_color = np.zeros((B, K), dtype=np.int64)
        move_size = np.ones((B, K), dtype=np.int64)
        move_region = np.zeros((B, K), dtype=np.int64)
        move_stars = np.zeros((B, K), dtype=np.float32)
        move_col_count = np.zeros((B, K, W), dtype=np.float32)
        col_onehot = np.eye(W, dtype=np.float32)[np.arange(N) % W]
        nbr_lists = []
        maxdeg = 1
        for i, b in enumerate(boards):
            k = b.n_moves
            move_valid[i, :k] = True
            move_color[i, :k] = b.move_color
            move_size[i, :k] = b.move_size
            move_region[i, :k] = b.move_region
            move_stars[i, :k] = b.move_stars
            move_col_count[i, :k] = b.moves.astype(np.float32) @ col_onehot
            lists = []
            for j, cells in enumerate(b.move_cells):
                move_cells[i, j, : len(cells)] = cells
                cs = set(cells)
                nb = sorted({x for c in cells for x in b._neighbors(c) if x not in cs})
                lists.append(nb)
                maxdeg = max(maxdeg, len(nb))
            nbr_lists.append(lists)
        self.DEG = maxdeg
        move_nbr = np.full((B, K, maxdeg), N, dtype=np.int64)
        move_nbr_count = np.zeros((B, K), dtype=np.float32)
        for i, lists in enumerate(nbr_lists):
            for j, nb in enumerate(lists):
                move_nbr[i, j, : len(nb)] = nb
                move_nbr_count[i, j] = len(nb)
        t = lambda a: torch.as_tensor(a, device=d)
        self.move_cells, self.move_valid = t(move_cells), t(move_valid)
        self.move_color, self.move_size, self.move_region = t(move_color), t(move_size), t(move_region)
        self.move_stars, self.move_col_count = t(move_stars), t(move_col_count)
        self.move_nbr, self.move_nbr_count = t(move_nbr), t(move_nbr_count)
        self.move_number_face = size_face[self.move_size]                            # (B,K)

        # static grid channels: colour one-hot (MAX_COLORS), star, start column  -> (B, MAX_COLORS+2, H, W)
        static = torch.zeros(B, MAX_COLORS + 2, H, W, device=d)
        for i in range(B):
            static[i, : C] = torch.nn.functional.one_hot(self.cell_color[i], C).to(f).t().reshape(C, H, W)
            static[i, MAX_COLORS] = self.cell_star_f[i].reshape(H, W)
            static[i, MAX_COLORS + 1] = self.start_col[i].to(f).reshape(H, W)
        self.static_grid = static

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _bid(obs: Obs) -> torch.Tensor:
        b = obs.get("board")
        if b is None:
            return torch.zeros(obs["marked"].shape[0], dtype=torch.int64, device=obs["marked"].device)
        return b

    @staticmethod
    def _ext(x: torch.Tensor) -> torch.Tensor:
        """Append a zero dummy cell (index N) so padded cell indices gather zeros."""
        return torch.cat([x, torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)], 1)

    def cells_sum(self, x_ext: torch.Tensor, bid: torch.Tensor) -> torch.Tensor:
        """Sum of x over the cells of every move: x_ext (M,N+1) -> (M,K)."""
        cells = self.move_cells[bid]                                                 # (M,K,MAXC)
        M = x_ext.shape[0]
        return x_ext.gather(1, cells.reshape(M, -1)).view(M, self.K, -1).sum(-1)

    @staticmethod
    def count_by(index: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
        """values (M,N) summed per index (M,N) in [0,n) -> (M,n)."""
        out = torch.zeros(values.shape[0], n, device=values.device, dtype=values.dtype)
        return out.scatter_add_(1, index, values)

    def frontier(self, marked_f: torch.Tensor, bid: torch.Tensor) -> torch.Tensor:
        """Cells a new group may touch: neighbours of own crosses, plus the start column
        (always, or only while the player has no cross yet)."""
        adj = (marked_f @ self.adj_f) > 0.5
        start = self.start_col[bid]
        if self.start_column_always:
            return adj | start
        no_marks = (marked_f.sum(1, keepdim=True) < 0.5)
        return adj | (start & no_marks)

    # ---------------------------------------------------------------- legality
    def legal_moves(self, obs: Obs):
        """Boolean (M,K+1) mask of legal actions (last = pass) and jokers needed (M,K)."""
        bid = self._bid(obs)
        marked_f = obs["marked"].to(torch.float32)
        M = marked_f.shape[0]
        overlap = self.cells_sum(self._ext(marked_f), bid) > 0.5
        touch = self.cells_sum(self._ext(self.frontier(marked_f, bid).to(torch.float32)), bid) > 0.5
        pool_c, pool_n = obs["pool_c"], obs["pool_n"]
        mcolor, mface = self.move_color[bid], self.move_number_face[bid]
        color_reg = pool_c.gather(1, mcolor) > 0
        size_reg = pool_n.gather(1, mface) > 0
        cj = (pool_c[:, self.C] > 0).unsqueeze(1)
        nj = (pool_n[:, self.Nf - 1] > 0).unsqueeze(1)
        need = (~color_reg).to(torch.int64) + (~size_reg).to(torch.int64)
        jok_ok = (obs["jokers_used"].unsqueeze(1) + need) <= self.jokers_max
        legal = (~overlap) & touch & (color_reg | cj) & (size_reg | nj) & jok_ok & self.move_valid[bid]
        legal = torch.cat([legal, torch.ones(M, 1, dtype=torch.bool, device=legal.device)], 1)
        return legal, need

    # ---------------------------------------------------------------- features
    def move_features(self, obs: Obs, need: torch.Tensor) -> torch.Tensor:
        """Dense per-move features (M,K,F)."""
        f = torch.float32
        bid = self._bid(obs)
        marked_f = obs["marked"].to(f)
        M = marked_f.shape[0]
        size = self.move_size[bid].to(f)
        mregion, mcolor = self.move_region[bid], self.move_color[bid]
        region_cnt = self.count_by(self.cell_region[bid], marked_f, self.R).gather(1, mregion)
        rsize = self.region_size[bid].gather(1, mregion)
        completes_region = (region_cnt + size >= rsize - 0.5).to(f)
        region_fill = (region_cnt + size) / rsize
        color_cnt_all = self.count_by(self.cell_color[bid], marked_f, self.C)             # (M,C)
        color_total = self.color_total[bid]                                               # (M,C)
        color_cnt = color_cnt_all.gather(1, mcolor)
        ctotal = color_total.gather(1, mcolor)
        completes_color = (color_cnt + size >= ctotal - 0.5).to(f)
        col_after = (marked_f @ self.col_mat).unsqueeze(1) + self.move_col_count[bid]       # (M,K,W)
        col_complete = (col_after >= self.col_height.view(1, 1, -1) - 0.5).to(f)
        col_val = torch.where(obs["col_claimed"], self.col_second[bid], self.col_first[bid]).unsqueeze(1)
        n_cols = col_complete.sum(-1)
        col_pts = (col_complete * col_val).sum(-1)
        color_claimed_k = obs["color_claimed"].gather(1, mcolor)
        color_pts = completes_color * (self.color_first - color_claimed_k.to(f) * (self.color_first - self.color_second))
        own_colors_done = (color_cnt_all >= color_total - 0.5).to(f).sum(1, keepdim=True)
        ends_game = ((own_colors_done + completes_color) >= self.end_colors).to(f)
        nbr = self.move_nbr[bid]
        nbr_marked = self._ext(marked_f).gather(1, nbr.reshape(M, -1)).view(M, self.K, -1).sum(-1)
        new_frontier = self.move_nbr_count[bid] - nbr_marked
        opp_overlap = self.cells_sum(self._ext(obs["opp_marked"].to(f)), bid) / size
        feats = torch.stack(
            [
                size / 5.0,
                self.move_stars[bid],
                need.to(f),
                completes_region,
                region_fill,
                completes_color,
                n_cols,
                col_pts / 5.0,
                color_pts / 5.0,
                ends_game,
                new_frontier / 8.0,
                opp_overlap,
            ],
            dim=-1,
        )
        return feats

    def legal_cells(self, legal: torch.Tensor, bid: torch.Tensor) -> torch.Tensor:
        """(M,N) float: cell belongs to at least one legal move."""
        M = legal.shape[0]
        cells = self.move_cells[bid].reshape(M, -1)
        w = legal[:, : self.K].to(torch.float32).repeat_interleave(self.MAXC, 1)
        out = torch.zeros(M, self.N + 1, device=legal.device).scatter_add_(1, cells, w)
        return (out[:, : self.N] > 0.5).to(torch.float32)

    def grid_inputs(self, obs: Obs, legal: torch.Tensor) -> torch.Tensor:
        """Image-like network input (M, C_in, H, W)."""
        f = torch.float32
        bid = self._bid(obs)
        marked_f = obs["marked"].to(f)
        M = marked_f.shape[0]
        H, W = self.H, self.W
        legal_cells = self.legal_cells(legal, bid)
        col_fill = (marked_f @ self.col_mat) / self.col_height.unsqueeze(0)                # (M,W)
        cell_color = self.cell_color[bid]
        color_fill = self.count_by(cell_color, marked_f, self.C) / self.color_total[bid]    # (M,C)
        per_cell_color_fill = color_fill.gather(1, cell_color)
        per_cell_color_claimed = obs["color_claimed"].to(f).gather(1, cell_color)
        per_cell_col_claimed = obs["col_claimed"].to(f).gather(1, self.cell_col.unsqueeze(0).expand(M, -1))
        per_cell_col_fill = col_fill.gather(1, self.cell_col.unsqueeze(0).expand(M, -1))
        frontier = self.frontier(marked_f, bid).to(f) * (1 - marked_f)
        dyn = torch.stack(
            [
                marked_f,
                obs["opp_marked"].to(f),
                obs["opp_best"].to(f),
                legal_cells,
                frontier,
                per_cell_col_claimed,
                per_cell_col_fill,
                per_cell_color_claimed,
                per_cell_color_fill,
            ],
            dim=1,
        ).reshape(M, -1, H, W)
        glob = self.global_vector(obs)                                               # (M,G)
        glob_b = glob.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        static = self.static_grid[bid]
        return torch.cat([static, dyn, glob_b], 1)

    def move_logits(self, cell_logits: torch.Tensor, obs: Obs) -> torch.Tensor:
        """Sum of per-cell logits over each move's cells -> (M,K)."""
        return self.cells_sum(self._ext(cell_logits), self._bid(obs))

    def global_vector(self, obs: Obs) -> torch.Tensor:
        M = obs["marked"].shape[0]
        d = obs["marked"].device
        pc = torch.zeros(M, MAX_COLORS + 1, device=d)
        pc[:, : self.C] = obs["pool_c"][:, : self.C].to(torch.float32)
        pc[:, MAX_COLORS] = obs["pool_c"][:, self.C].to(torch.float32)
        pn = torch.zeros(M, MAX_NUMBER + 1, device=d)
        pn[:, : self.Nf - 1] = obs["pool_n"][:, : self.Nf - 1].to(torch.float32)
        pn[:, MAX_NUMBER] = obs["pool_n"][:, self.Nf - 1].to(torch.float32)
        return torch.cat([pc / 3.0, pn / 3.0, obs["scalars"]], 1)

    @property
    def n_grid_channels(self) -> int:
        return (MAX_COLORS + 2) + N_DYN_CHANNELS + self.n_global

    @property
    def n_global(self) -> int:
        return (MAX_COLORS + 1) + (MAX_NUMBER + 1) + N_SCALARS

    # ---------------------------------------------------------------- scoring
    def score_components(self, marked: torch.Tensor, jokers_used: torch.Tensor, bid: torch.Tensor | None = None):
        """Star penalty and joker bonus for marked (...,N) / jokers_used (...); ``bid`` indexes the board pool
        and must broadcast against marked[..., 0] (None = board 0)."""
        if bid is None:
            star, n_stars = self.cell_star[0], self.n_stars[0]
        else:
            star, n_stars = self.cell_star[bid], self.n_stars[bid]
        stars_marked = (marked & star).sum(-1).to(torch.float32)
        star_pen = -self.star_penalty * (n_stars - stars_marked)
        joker_pts = self.joker_bonus * (self.jokers_max - jokers_used).to(torch.float32)
        return star_pen, joker_pts


def make_scalars(
    round_idx: torch.Tensor,
    is_active: torch.Tensor,
    n_players: int,
    own_colors_done: torch.Tensor,
    opp_colors_done: torch.Tensor,
    own_score: torch.Tensor,
    opp_score: torch.Tensor,
    rounds_remaining: torch.Tensor,
    free_round: torch.Tensor,
    jokers_left: torch.Tensor,
    jokers_max: int,
    end_colors: int,
) -> torch.Tensor:
    f = torch.float32
    return torch.stack(
        [
            round_idx.to(f) / 30.0,
            is_active.to(f),
            torch.full_like(round_idx, n_players).to(f) / 6.0,
            own_colors_done.to(f) / end_colors,
            opp_colors_done.to(f) / end_colors,
            own_score.to(f) / 40.0,
            opp_score.to(f) / 40.0,
            rounds_remaining.to(f) / 30.0,
            free_round.to(f),
            jokers_left.to(f) / max(1, jokers_max),
            (own_score - opp_score).to(f) / 20.0,
            torch.full_like(round_idx, 1.0 if n_players > 1 else 0.0).to(f),
        ],
        dim=-1,
    )


def mask_opponent_info(obs: Obs) -> Obs:
    """Variant 'blind': remove everything about the opponents' sheets (public claims stay)."""
    out = dict(obs)
    out["opp_marked"] = torch.zeros_like(obs["opp_marked"])
    out["opp_best"] = torch.zeros_like(obs["opp_best"])
    sc = obs["scalars"].clone()
    sc[:, list(OPP_SCALARS)] = 0.0
    out["scalars"] = sc
    return out


def apply_opponent_view(obs: Obs, view: str) -> Obs:
    if view == "none":
        return mask_opponent_info(obs)
    return obs


def obs_to(obs: Obs, device) -> Obs:
    return {k: v.to(device) for k, v in obs.items()}


def obs_index(obs: Obs, idx) -> Obs:
    return {k: v[idx] for k, v in obs.items()}


def obs_cat(list_of_obs) -> Obs:
    keys = list_of_obs[0].keys()
    return {k: torch.cat([o[k] for o in list_of_obs], 0) for k in keys}
