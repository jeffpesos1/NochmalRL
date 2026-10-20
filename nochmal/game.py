"""Interactive single game (G = 1) on top of the vectorised environment.

Used by the web server: humans and agents share seats, dice can be rolled or
typed in from a real physical game, every step can be undone, and the agent
can be asked for its move distribution at any time.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional

import torch

from .agent import BasePolicy
from .board import JOKER, Board
from .env import NochmalEnv
from .features import apply_opponent_view


class GameSession:
    def __init__(self, board: Board, players: List[Dict], policies: Dict[str, BasePolicy], device="cpu",
                 seed: Optional[int] = None, max_rounds: Optional[int] = None, free_rounds: int = 0,
                 start_column_always: bool = False, opponent_view: str = "full"):
        """``opponent_view``: 'full' = agents see the other sheets, 'none' = blind.  Seats of type
        'extern' are real players whose sheet is *not* tracked: only their public actions (dice taken,
        announced columns/colours, second colour completed) are entered; this forces the blind view."""
        self.board = board
        self.players = players                       # [{name, type: 'human'|'agent'|'extern', agent: spec}]
        self.has_extern = any(p["type"] == "extern" for p in players)
        self.opponent_view = "none" if self.has_extern else opponent_view
        self.extern_pending: List[tuple] = []        # claims announced by extern players this round
        self.extern_finished = False
        self.policies = policies                     # spec -> policy (shared, same BoardTensors/device)
        P = len(players)
        self.env = NochmalEnv(board, 1, P, device, max_rounds=max_rounds, free_rounds=free_rounds,
                              start_column_always=start_column_always,
                              seed=seed if seed is not None else int(time.time() * 1000) % (2**31))
        self.device = self.env.device
        self.phase = "dice"                          # 'dice' | 'move' | 'finished'
        self.order: List[int] = []
        self.turn_pos = 0
        self.pool_c = self.env.pool_c.clone()
        self.pool_n = self.env.pool_n.clone()
        self.history: List[Dict] = []
        self.round_log: List[Dict] = []
        self.final: Optional[Dict] = None
        self.undo_stack: List[Dict] = []
        self.created = time.time()

    # ---------------------------------------------------------------- helpers
    @property
    def P(self) -> int:
        return len(self.players)

    def current_player(self) -> Optional[int]:
        if self.phase != "move":
            return None
        return self.order[self.turn_pos]

    def is_active_turn(self) -> bool:
        return self.turn_pos == 0

    def _snapshot(self) -> Dict:
        return {
            "env": self.env.state_tensors(),
            "phase": self.phase, "order": list(self.order), "turn_pos": self.turn_pos,
            "pool_c": self.pool_c.clone(), "pool_n": self.pool_n.clone(),
            "history": copy.deepcopy(self.history), "round_log": copy.deepcopy(self.round_log),
            "final": copy.deepcopy(self.final), "episodes": self.env.episodes,
            "extern_pending": list(self.extern_pending), "extern_finished": self.extern_finished,
        }

    def _restore(self, s: Dict) -> None:
        self.env.load_state_tensors(s["env"])
        self.phase, self.order, self.turn_pos = s["phase"], list(s["order"]), s["turn_pos"]
        self.pool_c, self.pool_n = s["pool_c"].clone(), s["pool_n"].clone()
        self.history, self.round_log, self.final = s["history"], s["round_log"], s["final"]
        self.env.episodes = s["episodes"]
        self.extern_pending, self.extern_finished = list(s["extern_pending"]), s["extern_finished"]

    def push_undo(self) -> None:
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self._restore(self.undo_stack.pop())
        return True

    # ---------------------------------------------------------------- dice
    def _face_names(self):
        b = self.board
        cf = [b.color_keys[int(i)] if int(i) < b.n_colors else JOKER for i in self.env.color_faces[0].tolist()]
        nf = [b.number_faces[int(i)] if int(i) < len(b.number_faces) else JOKER for i in self.env.number_faces[0].tolist()]
        return cf, nf

    def roll(self) -> None:
        assert self.phase == "dice", "dice are not expected now"
        self.push_undo()
        self.env.roll()
        self._start_round()

    def set_dice(self, colors: List[str], numbers: List) -> None:
        assert self.phase == "dice", "dice are not expected now"
        b = self.board
        if len(colors) != self.env.n_color_dice or len(numbers) != self.env.n_number_dice:
            raise ValueError(f"expected {self.env.n_color_dice} colour and {self.env.n_number_dice} number dice")
        cidx = [b.n_colors if c == JOKER else b.color_keys.index(c) for c in colors]
        nidx = [len(b.number_faces) if n == JOKER else b.number_faces.index(int(n)) for n in numbers]
        self.push_undo()
        self.env.set_dice(torch.tensor([cidx], device=self.device), torch.tensor([nidx], device=self.device))
        self._start_round()

    def _start_round(self) -> None:
        active = int(self.env.active_player()[0])
        self.order = [(active + i) % self.P for i in range(self.P)]
        self.turn_pos = 0
        self.pool_c = self.env.pool_c.clone()
        self.pool_n = self.env.pool_n.clone()
        self.round_log = []
        self.phase = "move"

    # ---------------------------------------------------------------- moves
    def _obs(self, p: int):
        g = torch.zeros(1, dtype=torch.int64, device=self.device)
        pt = torch.tensor([p], device=self.device)
        act = torch.tensor([self.is_active_turn()], device=self.device)
        return apply_opponent_view(self.env.make_obs(g, pt, self.pool_c, self.pool_n, act), self.opponent_view)

    def _move_info(self, k: int, need: int = 0) -> Dict:
        b = self.board
        return {
            "index": int(k),
            "cells": list(self.board.move_cells[k]),
            "color": b.color_keys[int(b.move_color[k])],
            "size": int(b.move_size[k]),
            "jokers": int(need),
            "stars": int(b.move_stars[k]),
        }

    def legal_moves(self) -> List[Dict]:
        p = self.current_player()
        if p is None:
            return []
        obs = self._obs(p)
        legal, need = self.env.bt.legal_moves(obs)
        ks = torch.nonzero(legal[0, : self.env.bt.K]).flatten().tolist()
        return [self._move_info(k, int(need[0, k])) for k in ks]

    def agent_distribution(self, p: Optional[int] = None, spec: Optional[str] = None, top_n: int = 8) -> Dict:
        """Top-n moves with probabilities from a policy (default: the seat's own agent, else first)."""
        p = self.current_player() if p is None else p
        if p is None:
            return {"suggestions": [], "pass_prob": 0.0}
        spec = spec or self.players[p].get("agent") or next(iter(self.policies))
        policy = self.policies[spec]
        obs = self._obs(p)
        with torch.no_grad():
            probs = policy.distribution(obs)[0]
            value = float(policy.values(obs)[0])
        K = self.env.bt.K
        _, need = self.env.bt.legal_moves(obs)
        order = torch.argsort(probs[:K], descending=True)[:top_n].tolist()
        sug = []
        for k in order:
            pr = float(probs[k])
            if pr <= 0:
                break
            info = self._move_info(k, int(need[0, k]))
            info["prob"] = pr
            sug.append(info)
        return {"suggestions": sug, "pass_prob": float(probs[K]), "value": value * 10.0, "agent": spec}

    def play_move(self, cells: Optional[List[int]], by: str = "human") -> Dict:
        """``cells`` = None means pass."""
        p = self.current_player()
        if p is None:
            raise ValueError("no move expected now")
        bt = self.env.bt
        obs = self._obs(p)
        legal, need = bt.legal_moves(obs)
        if cells is None:
            k = bt.K
        else:
            k = self.board.find_move(cells)
            if k < 0:
                raise ValueError("these cells are not a valid group (same colour block, connected, 1-5 cells)")
            if not bool(legal[0, k]):
                raise ValueError("move is not legal now (dice, adjacency, jokers)")
        self.push_undo()
        g = torch.zeros(1, dtype=torch.int64, device=self.device)
        pt = torch.tensor([p], device=self.device)
        a = torch.tensor([k], device=self.device)
        reward, cface, nface = self.env.apply(g, pt, a, self.pool_c, self.pool_n)
        entry = {"round": int(self.env.round[0]) + 1, "player": p, "name": self.players[p]["name"], "by": by,
                 "active": self.is_active_turn()}
        if k < bt.K:
            entry.update(self._move_info(k, int(need[0, k])))
        else:
            entry["pass"] = True
        self.round_log.append(entry)
        self.history.append(entry)
        # the active player removes the chosen dice (from the 4th round on)
        if self.is_active_turn() and int(self.env.round[0]) >= self.env.free_rounds:
            self.pool_c, self.pool_n = self.env.remove_dice(self.pool_c, self.pool_n, cface, nface)
        self.turn_pos += 1
        if self.turn_pos >= self.P:
            self._finish_round()
        return entry

    def play_extern(self, dice: Optional[Dict] = None, columns: Optional[List[int]] = None,
                    colors: Optional[List[int]] = None, finished: bool = False) -> Dict:
        """Public information about an untracked (extern) player's turn.

        ``dice`` = {"color": key|'joker', "number": int|'joker'} the active extern player took (None = passed),
        ``columns``/``colors`` = indices announced as completed this round, ``finished`` = second colour done."""
        p = self.current_player()
        if p is None or self.players[p]["type"] != "extern":
            raise ValueError("no extern player to move now")
        b = self.board
        columns = [int(c) for c in (columns or [])]
        colors = [int(c) for c in (colors or [])]
        if any(c < 0 or c >= b.width for c in columns) or any(c < 0 or c >= b.n_colors for c in colors):
            raise ValueError("invalid column/colour index")
        cface = nface = -1
        if dice and self.is_active_turn():
            ck, nk = dice.get("color"), dice.get("number")
            cface = b.n_colors if ck == JOKER else b.color_keys.index(ck)
            nface = len(b.number_faces) if nk == JOKER else b.number_faces.index(int(nk))
            if int(self.pool_c[0, cface]) < 1 or int(self.pool_n[0, nface]) < 1:
                raise ValueError("these dice are not in the pool")
        self.push_undo()
        entry = {"round": int(self.env.round[0]) + 1, "player": p, "name": self.players[p]["name"], "by": "extern",
                 "active": self.is_active_turn(), "extern": True, "dice": dice if cface >= 0 else None,
                 "columns": columns, "colors": colors, "finished": bool(finished)}
        if cface >= 0:
            self.pool_c, self.pool_n = self.env.remove_dice(
                self.pool_c, self.pool_n, torch.tensor([cface], device=self.device), torch.tensor([nface], device=self.device))
        self.extern_pending.append((columns, colors))
        if finished:
            self.extern_finished = True
        self.round_log.append(entry)
        self.history.append(entry)
        self.turn_pos += 1
        if self.turn_pos >= self.P:
            self._finish_round()
        return entry

    def _finish_round(self) -> None:
        env = self.env
        col_before = env.col_points.clone()
        color_before = env.color_points.clone()
        reward, done, final, jl = env.end_round(auto_reset=False)
        # announcements of extern players count for the same round (everybody completing now gets the upper value)
        for columns, colors in self.extern_pending:
            for c in columns:
                env.col_claimed[0, c] = True
            for c in colors:
                env.color_claimed[0, c] = True
        self.extern_pending = []
        gained = (env.col_points - col_before + env.color_points - color_before)[0].tolist()
        for e in self.round_log:
            if not e.get("extern"):
                e["round_points"] = gained[e["player"]]
        if bool(done[0]) or self.extern_finished:
            self.phase = "finished"
            scores = final[0].tolist()
            jokers_left = jl[0].tolist()
            tracked = [i for i in range(self.P) if self.players[i]["type"] != "extern"]
            rank = sorted(tracked, key=lambda i: (-scores[i], -jokers_left[i]))
            self.final = {"scores": [scores[i] if self.players[i]["type"] != "extern" else None for i in range(self.P)],
                          "jokers_left": jokers_left, "ranking": rank, "rounds": int(env.round[0])}
        else:
            self.phase = "dice"

    # ---------------------------------------------------------------- export
    def to_dict(self) -> Dict:
        env, b = self.env, self.board
        cf, nf = self._face_names()
        scores = env.scores()[0].tolist()
        star_pen, joker_pts = env.bt.score_components(env.marked, env.jokers_used)
        players = []
        for p, pl in enumerate(self.players):
            players.append({
                "name": pl["name"], "type": pl["type"], "agent": pl.get("agent"),
                "marked": env.marked[0, p].to(torch.int64).tolist(),
                "jokers_used": int(env.jokers_used[0, p]),
                "col_done": env.col_done[0, p].to(torch.int64).tolist(),
                "color_done": env.color_done[0, p].to(torch.int64).tolist(),
                "col_points": float(env.col_points[0, p]),
                "color_points": float(env.color_points[0, p]),
                "star_penalty": float(star_pen[0, p]),
                "joker_points": float(joker_pts[0, p]),
                "score": scores[p] if pl["type"] != "extern" else None,
            })
        pool_c = self.pool_c[0].tolist()
        pool_n = self.pool_n[0].tolist()
        cur = self.current_player()
        return {
            "board": b.source,
            "phase": self.phase,
            "round": int(env.round[0]) + 1,
            "max_rounds": env.max_rounds,
            "free_rounds": env.free_rounds,
            "start_column_always": env.start_column_always,
            "opponent_view": self.opponent_view,
            "n_color_dice": env.n_color_dice,
            "n_number_dice": env.n_number_dice,
            "active_player": int(env.active_player()[0]) if self.phase != "finished" else None,
            "current_player": cur,
            "current_is_active": self.is_active_turn() if cur is not None else None,
            "order": self.order,
            "turn_pos": self.turn_pos,
            "dice": {"colors": cf, "numbers": nf} if self.phase != "dice" else None,
            "pool": {"colors": {(b.color_keys[i] if i < b.n_colors else JOKER): pool_c[i] for i in range(len(pool_c))},
                     "numbers": {(str(b.number_faces[i]) if i < len(b.number_faces) else JOKER): pool_n[i] for i in range(len(pool_n))}},
            "col_claimed": env.col_claimed[0].to(torch.int64).tolist(),
            "color_claimed": env.color_claimed[0].to(torch.int64).tolist(),
            "players": players,
            "history": self.history[-60:],
            "round_log": self.round_log,
            "final": self.final,
            "can_undo": len(self.undo_stack) > 0,
        }
