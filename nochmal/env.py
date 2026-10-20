"""Vectorised *Noch mal!* environment: G games x P players as tensors on one device.

Rules implemented (official German rule book, Schmidt Spiele 49327):

* Each round the active player rolls all colour and number dice.  He picks one
  colour die + one number die; these two dice are removed for the other
  players, who may each pick any combination of the remaining dice (several
  players may use the same combination).  The rule book additionally lets
  everybody use all dice in the first 3 rounds (``free_rounds=3``); the default
  here is the strict table rule ``free_rounds=0``.  If the active player passes,
  all dice stay available.
* A move marks exactly ``number`` cells of ``colour``; they must be
  orthogonally connected, inside one colour block, and at least one of them
  must touch (orthogonally) an own cross.  The very first move must lie in the
  start column H.  With ``start_column_always=True`` (literal rule-book wording)
  column H may start a new group at any time.
* Joker faces (black cross / question mark) cost one of the 8 exclamation marks
  each; one may use two jokers in one turn.
* Completed column -> upper value if nobody completed it in an earlier round,
  else lower value (several players completing it in the same round all get the
  upper value).  Completed colour -> 5 / 3 points with the same rule.
* Game ends after the round in which any player completes the 2nd colour (the
  other players still finish that round).  Solo: 2 colour + 2 number dice,
  30 rounds.
* Score = columns + colours - 2 per unmarked star + 1 per unused joker.

Rewards handed out are *score deltas* (plus an optional terminal margin term),
so the sum of rewards over an episode is the final score minus the constant
initial potential.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .board import Board
from .features import BoardTensors, Obs, make_scalars


@dataclass
class RoundResult:
    obs: Obs                      # (G,P,...) observations of every stream this round
    action: torch.Tensor          # (G,P)
    logp: torch.Tensor            # (G,P)
    value: torch.Tensor           # (G,P)
    reward: torch.Tensor          # (G,P)
    done: torch.Tensor            # (G,) episode finished after this round
    final_scores: torch.Tensor    # (G,P) valid where done
    final_jokers_left: torch.Tensor  # (G,P) valid where done (tie-break)


class NochmalEnv:
    def __init__(
        self,
        board: Union[Board, Sequence[Board]],
        n_games: int,
        n_players: int,
        device: torch.device | str = "cpu",
        n_color_dice: Optional[int] = None,
        n_number_dice: Optional[int] = None,
        max_rounds: Optional[int] = None,
        free_rounds: int = 0,
        start_column_always: bool = False,
        margin_weight: float = 0.0,
        seed: int = 0,
        resample_boards: bool = True,
    ):
        """``free_rounds``: number of initial rounds in which everybody may use all dice
        (rule book: 3; strict table rule: 0).  ``start_column_always``: see BoardTensors.
        ``board`` may be a list (board pool): every game then plays on its own board, re-drawn at reset
        when ``resample_boards`` is set."""
        boards = [board] if isinstance(board, Board) else list(board)
        self.board = boards[0]
        self.boards = boards
        self.resample_boards = resample_boards and len(boards) > 1
        self.bt = BoardTensors(boards, device, start_column_always=start_column_always)
        self.start_column_always = start_column_always
        self.device = torch.device(device)
        self.G, self.P = n_games, n_players
        solo = n_players == 1
        self.n_color_dice = n_color_dice if n_color_dice is not None else (2 if solo else 3)
        self.n_number_dice = n_number_dice if n_number_dice is not None else (2 if solo else 3)
        self.max_rounds = max_rounds if max_rounds is not None else (30 if solo else 60)
        self.free_rounds = free_rounds
        self.margin_weight = margin_weight
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        bt, G, P, d = self.bt, self.G, self.P, self.device
        self.marked = torch.zeros(G, P, bt.N, dtype=torch.bool, device=d)
        self.jokers_used = torch.zeros(G, P, dtype=torch.int64, device=d)
        self.col_done = torch.zeros(G, P, bt.W, dtype=torch.bool, device=d)
        self.color_done = torch.zeros(G, P, bt.C, dtype=torch.bool, device=d)
        self.col_points = torch.zeros(G, P, device=d)
        self.color_points = torch.zeros(G, P, device=d)
        self.col_claimed = torch.zeros(G, bt.W, dtype=torch.bool, device=d)
        self.color_claimed = torch.zeros(G, bt.C, dtype=torch.bool, device=d)
        self.round = torch.zeros(G, dtype=torch.int64, device=d)
        self.color_faces = torch.zeros(G, self.n_color_dice, dtype=torch.int64, device=d)
        self.number_faces = torch.zeros(G, self.n_number_dice, dtype=torch.int64, device=d)
        self.pool_c = torch.zeros(G, bt.Cf, dtype=torch.uint8, device=d)
        self.pool_n = torch.zeros(G, bt.Nf, dtype=torch.uint8, device=d)
        self.board_id = torch.arange(G, device=d) % bt.B
        self.episodes = 0

    # ------------------------------------------------------------------ state
    def state_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            k: getattr(self, k).clone()
            for k in [
                "marked", "jokers_used", "col_done", "color_done", "col_points", "color_points",
                "col_claimed", "color_claimed", "round", "color_faces", "number_faces", "pool_c", "pool_n", "board_id",
            ]
        }

    def load_state_tensors(self, st: Dict[str, torch.Tensor]) -> None:
        for k, v in st.items():
            getattr(self, k).copy_(v)

    def reset_games(self, mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return
        self.marked[mask] = False
        self.jokers_used[mask] = 0
        self.col_done[mask] = False
        self.color_done[mask] = False
        self.col_points[mask] = 0.0
        self.color_points[mask] = 0.0
        self.col_claimed[mask] = False
        self.color_claimed[mask] = False
        self.round[mask] = 0
        if self.resample_boards:
            n = int(mask.sum())
            self.board_id[mask] = torch.randint(self.bt.B, (n,), device=self.device, generator=self.gen)

    def scores(self) -> torch.Tensor:
        star_pen, joker_pts = self.bt.score_components(self.marked, self.jokers_used, self.board_id.unsqueeze(1))
        return self.col_points + self.color_points + star_pen + joker_pts

    def active_player(self) -> torch.Tensor:
        return self.round % self.P

    # ------------------------------------------------------------------ dice
    def roll(self) -> None:
        bt, G = self.bt, self.G
        self.color_faces = torch.multinomial(
            bt.color_face_probs.expand(G, -1), self.n_color_dice, replacement=True, generator=self.gen
        )
        self.number_faces = torch.multinomial(
            bt.number_face_probs.expand(G, -1), self.n_number_dice, replacement=True, generator=self.gen
        )
        self._faces_to_pool()

    def set_dice(self, color_faces: torch.Tensor, number_faces: torch.Tensor) -> None:
        self.color_faces.copy_(color_faces)
        self.number_faces.copy_(number_faces)
        self._faces_to_pool()

    def _faces_to_pool(self) -> None:
        bt, G = self.bt, self.G
        pc = torch.zeros(G, bt.Cf, dtype=torch.int64, device=self.device)
        pc.scatter_add_(1, self.color_faces, torch.ones_like(self.color_faces))
        pn = torch.zeros(G, bt.Nf, dtype=torch.int64, device=self.device)
        pn.scatter_add_(1, self.number_faces, torch.ones_like(self.number_faces))
        self.pool_c = pc.to(torch.uint8)
        self.pool_n = pn.to(torch.uint8)

    def remove_dice(self, pool_c, pool_n, cface, nface):
        """Return pools with one die of face ``cface``/``nface`` removed (where >= 0)."""
        pool_c, pool_n = pool_c.clone(), pool_n.clone()
        m = cface >= 0
        if bool(m.any()):
            idx = torch.arange(pool_c.shape[0], device=self.device)[m]
            pool_c[idx, cface[m]] -= 1
            pool_n[idx, nface[m]] -= 1
        return pool_c, pool_n

    # ------------------------------------------------------------------ observe
    def make_obs(self, g: torch.Tensor, p: torch.Tensor, pool_c: torch.Tensor, pool_n: torch.Tensor,
                 is_active: torch.Tensor) -> Obs:
        bt, P = self.bt, self.P
        marked_all = self.marked[g]                                   # (M,P,N)
        own = marked_all[torch.arange(len(g), device=self.device), p]
        opp = (marked_all.sum(1) - own.to(torch.int64)) > 0 if P > 1 else torch.zeros_like(own)
        scores = self.scores()[g]                                     # (M,P)
        colors_done = self.color_done[g].sum(-1)                      # (M,P)
        ar = torch.arange(len(g), device=self.device)
        own_score = scores[ar, p]
        own_cd = colors_done[ar, p]
        if P > 1:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[ar, p] = False
            masked_scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
            opp_score, opp_idx = masked_scores.max(1)
            opp_cd = torch.where(mask, colors_done, torch.zeros_like(colors_done)).max(1).values
            opp_best = marked_all[ar, opp_idx]
        else:
            opp_score = torch.zeros_like(own_score)
            opp_cd = torch.zeros_like(own_cd)
            opp_best = torch.zeros_like(own)
        rnd = self.round[g]
        scalars = make_scalars(
            round_idx=rnd,
            is_active=is_active,
            n_players=P,
            own_colors_done=own_cd,
            opp_colors_done=opp_cd,
            own_score=own_score,
            opp_score=opp_score,
            rounds_remaining=(self.max_rounds - rnd).clamp(min=0),
            free_round=(rnd < self.free_rounds),
            jokers_left=bt.jokers_max - self.jokers_used[g, p],
            jokers_max=bt.jokers_max,
            end_colors=bt.end_colors,
        )
        return {
            "marked": own,
            "opp_marked": opp,
            "opp_best": opp_best,
            "pool_c": pool_c,
            "pool_n": pool_n,
            "col_claimed": self.col_claimed[g],
            "color_claimed": self.color_claimed[g],
            "jokers_used": self.jokers_used[g, p],
            "scalars": scalars,
            "board": self.board_id[g],
        }

    # ------------------------------------------------------------------ act
    def apply(self, g, p, action, pool_c, pool_n):
        """Apply actions for streams (g,p).  Returns reward (M,), used colour face, used number face."""
        bt = self.bt
        K = bt.K
        M = len(g)
        bid = self.board_id[g]
        is_move = action < K
        a = action.clamp(max=K - 1)
        cells = bt.move_cells[bid, a]                                                # (M,MAXC), padded with N
        mv = torch.zeros(M, bt.N + 1, dtype=torch.bool, device=self.device)
        mv.scatter_(1, cells, True)
        mv = mv[:, : bt.N] & is_move.unsqueeze(1)
        ar = torch.arange(M, device=self.device)
        mcolor, mface = bt.move_color[bid, a], bt.move_number_face[bid, a]
        color_reg = pool_c[ar, mcolor] > 0
        size_reg = pool_n[ar, mface] > 0
        need = ((~color_reg).to(torch.int64) + (~size_reg).to(torch.int64)) * is_move.to(torch.int64)
        stars = (mv & bt.cell_star[bid]).sum(1).to(torch.float32)
        self.marked[g, p] = self.marked[g, p] | mv
        self.jokers_used[g, p] = self.jokers_used[g, p] + need
        reward = bt.star_penalty * stars - bt.joker_bonus * need.to(torch.float32)
        cface = torch.where(is_move, torch.where(color_reg, mcolor, torch.full_like(a, bt.C)), torch.full_like(a, -1))
        nface = torch.where(is_move, torch.where(size_reg, mface, torch.full_like(a, bt.Nf - 1)), torch.full_like(a, -1))
        return reward, cface, nface

    def end_round(self, auto_reset: bool = True):
        """Score columns/colours completed this round, detect game end, reset finished games.

        Returns reward (G,P), done (G,), final_scores (G,P), final_jokers_left (G,P).
        """
        bt, G, P = self.bt, self.G, self.P
        f = torch.float32
        marked_f = self.marked.to(f)
        bid = self.board_id
        col_full = (marked_f @ bt.col_mat) >= bt.col_height.view(1, 1, -1) - 0.5
        new_col = col_full & ~self.col_done
        col_val = torch.where(self.col_claimed, bt.col_second[bid], bt.col_first[bid]).unsqueeze(1)
        col_reward = (new_col.to(f) * col_val).sum(-1)
        self.col_points += col_reward
        self.col_done = col_full
        self.col_claimed |= new_col.any(1)

        cell_color = bt.cell_color[bid].unsqueeze(1).expand(-1, P, -1)
        color_cnt = torch.zeros(G, P, bt.C, device=self.device).scatter_add_(2, cell_color, marked_f)
        color_full = color_cnt >= bt.color_total[bid].unsqueeze(1) - 0.5
        new_color = color_full & ~self.color_done
        color_val = (bt.color_first - self.color_claimed.to(f) * (bt.color_first - bt.color_second)).unsqueeze(1)
        color_reward = (new_color.to(f) * color_val).sum(-1)
        self.color_points += color_reward
        self.color_done = color_full
        self.color_claimed |= new_color.any(1)

        self.round += 1
        done = (self.color_done.sum(-1) >= bt.end_colors).any(1) | (self.round >= self.max_rounds)
        reward = col_reward + color_reward
        final = self.scores()
        jokers_left = (bt.jokers_max - self.jokers_used).to(f)
        if self.margin_weight != 0.0 and P > 1:
            opp_mean = (final.sum(1, keepdim=True) - final) / (P - 1)
            reward = reward + done.unsqueeze(1).to(f) * self.margin_weight * (final - opp_mean)
        self.episodes += int(done.sum())
        if auto_reset:
            self.reset_games(done)
        return reward, done, final, jokers_left

    # ------------------------------------------------------------------ driver
    def play_round(self, policy: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                   record: bool = True) -> RoundResult:
        """Roll dice, let the active player and then all passive players act, close the round.

        ``policy(obs, seat) -> (action, logp, value)`` is called twice: once for all active
        players (batch G) and once for all passive players (batch G*(P-1)).
        """
        G, P, d = self.G, self.P, self.device
        self.roll()
        g_all = torch.arange(G, device=d)
        active = self.active_player()
        ones = torch.ones(G, dtype=torch.bool, device=d)
        obs_a = self.make_obs(g_all, active, self.pool_c, self.pool_n, ones)
        act_a, logp_a, val_a = policy(obs_a, active)
        rew_a, cface, nface = self.apply(g_all, active, act_a, self.pool_c, self.pool_n)
        # dice removal only from the 4th round on
        keep = self.round < self.free_rounds
        cface = torch.where(keep, torch.full_like(cface, -1), cface)
        nface = torch.where(keep, torch.full_like(nface, -1), nface)
        pool_c2, pool_n2 = self.remove_dice(self.pool_c, self.pool_n, cface, nface)

        rec_obs = None
        action = torch.zeros(G, P, dtype=torch.int64, device=d)
        logp = torch.zeros(G, P, device=d)
        value = torch.zeros(G, P, device=d)
        reward = torch.zeros(G, P, device=d)
        action[g_all, active] = act_a
        logp[g_all, active] = logp_a
        value[g_all, active] = val_a
        reward[g_all, active] = rew_a
        if record:
            rec_obs = {k: torch.zeros((G, P) + tuple(v.shape[1:]), dtype=v.dtype, device=d) for k, v in obs_a.items()}
            for k, v in obs_a.items():
                rec_obs[k][g_all, active] = v

        if P > 1:
            g_p = g_all.repeat_interleave(P - 1)
            p_p = ((active.unsqueeze(1) + torch.arange(1, P, device=d).unsqueeze(0)) % P).reshape(-1)
            zeros = torch.zeros(len(g_p), dtype=torch.bool, device=d)
            obs_p = self.make_obs(g_p, p_p, pool_c2[g_p], pool_n2[g_p], zeros)
            act_p, logp_p, val_p = policy(obs_p, p_p)
            rew_p, _, _ = self.apply(g_p, p_p, act_p, pool_c2[g_p], pool_n2[g_p])
            action[g_p, p_p] = act_p
            logp[g_p, p_p] = logp_p
            value[g_p, p_p] = val_p
            reward[g_p, p_p] = rew_p
            if record:
                for k, v in obs_p.items():
                    rec_obs[k][g_p, p_p] = v

        r_end, done, final, jokers_left = self.end_round()
        reward = reward + r_end
        return RoundResult(rec_obs, action, logp, value, reward, done, final, jokers_left)
