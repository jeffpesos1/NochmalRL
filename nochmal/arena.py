"""Arena: let agents (checkpoints, heuristic, random) compete at different player counts.

Example::

    python -m nochmal.arena --agents heuristic random duo/best.pt duo-blind/best.pt --players 1 2 3 4 \
        --games 240 --out checkpoints/arena/result.json

For every player count P >= 2 all combinations of P distinct agents play ``games`` games (seats rotated so
every agent sits in every position equally often).  P = 1 is the solo game (30 rolls, 2+2 dice) and
compares mean scores.  Results: win rate (ties shared), mean score, mean rank per agent, and for P = 2 a
head-to-head matrix.  Neural agents play greedily and with the opponent view they were trained with.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Dict, List

import torch

from .agent import SeatPolicy, make_policy
from .board import load_board
from .env import NochmalEnv
from .features import BoardTensors


def play_matches(env: NochmalEnv, policy, n_games: int):
    """Play until ``n_games`` games finished; return final scores (n,P) and jokers left (n,P)."""
    env.reset_games(torch.ones(env.G, dtype=torch.bool, device=env.device))
    scores, jokers = [], []
    got = 0
    for _ in range(400):
        res = env.play_round(policy, record=False)
        if bool(res.done.any()):
            scores.append(res.final_scores[res.done])
            jokers.append(res.final_jokers_left[res.done])
            got += int(res.done.sum())
            if got >= n_games:
                break
    return torch.cat(scores)[:n_games], torch.cat(jokers)[:n_games]


def run_arena(agent_specs: List[str], player_counts: List[int], games: int, board_name: str = "classic",
              device: str = "auto", ck_dir: str = "checkpoints", free_rounds: int = 0,
              start_column_always: bool = False, progress=print) -> Dict:
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    torch.backends.cudnn.benchmark = True
    board = load_board(board_name)
    bt = BoardTensors(board, dev, start_column_always=start_column_always)
    policies = {spec: make_policy(spec, bt, ck_dir, greedy=True) for spec in agent_specs}
    names = list(agent_specs)
    out: Dict = {"board": board_name, "games": games, "free_rounds": free_rounds,
                 "start_column_always": start_column_always, "agents": names, "players": {}, "started": time.time()}
    t0 = time.time()
    for P in player_counts:
        stats = {n: {"games": 0, "wins": 0.0, "score_sum": 0.0, "rank_sum": 0.0} for n in names}
        matrix = {a: {b: [0.0, 0] for b in names} for a in names} if P == 2 else None
        if P == 1:
            G = max(64, games)
            env = NochmalEnv(board, G, 1, dev, free_rounds=free_rounds, start_column_always=start_column_always,
                             seed=1, max_rounds=30)
            env.bt = bt
            for n in names:
                sc, _ = play_matches(env, policies[n], games)
                stats[n]["games"] += games
                stats[n]["score_sum"] += float(sc[:, 0].sum())
                progress(f"P=1 {n}: mean score {float(sc[:, 0].mean()):.2f}")
        else:
            combos = list(itertools.combinations(names, P))
            per_rot = max(8, games // P)
            G = per_rot
            env = NochmalEnv(board, G, P, dev, free_rounds=free_rounds, start_column_always=start_column_always, seed=2)
            env.bt = bt
            for ci, combo in enumerate(combos):
                for rot in range(P):
                    seats = combo[rot:] + combo[:rot]
                    pol = SeatPolicy(bt, [policies[s] for s in seats])
                    sc, jl = play_matches(env, pol, per_rot)
                    key = sc * 100 + jl                       # tie-break: more jokers left
                    best = key.max(1, keepdim=True).values
                    is_best = (key >= best).to(torch.float32)
                    share = is_best / is_best.sum(1, keepdim=True)  # ties share the win
                    rank = 1 + (key.unsqueeze(2) < key.unsqueeze(1)).sum(2).to(torch.float32)
                    for si, name in enumerate(seats):
                        stats[name]["games"] += per_rot
                        stats[name]["wins"] += float(share[:, si].sum())
                        stats[name]["score_sum"] += float(sc[:, si].sum())
                        stats[name]["rank_sum"] += float(rank[:, si].sum())
                    if matrix is not None:
                        a, b = seats
                        matrix[a][b][0] += float(share[:, 0].sum()); matrix[a][b][1] += per_rot
                        matrix[b][a][0] += float(share[:, 1].sum()); matrix[b][a][1] += per_rot
                progress(f"P={P} {ci + 1}/{len(combos)} {' vs '.join(combo)} done ({time.time() - t0:.0f}s)")
        table = []
        for n in names:
            st = stats[n]
            g = max(1, st["games"])
            table.append({"agent": n, "games": st["games"], "win_rate": st["wins"] / g if P > 1 else None,
                          "mean_score": st["score_sum"] / g, "mean_rank": st["rank_sum"] / g if P > 1 else None})
        table.sort(key=lambda r: (-(r["win_rate"] or 0), -r["mean_score"]))
        entry = {"table": table}
        if matrix is not None:
            entry["matrix"] = {a: {b: (matrix[a][b][0] / matrix[a][b][1] if matrix[a][b][1] else None) for b in names} for a in names}
        out["players"][str(P)] = entry
    out["seconds"] = time.time() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Noch mal! agent arena")
    ap.add_argument("--agents", nargs="+", required=True, help="heuristic, random or checkpoint ids (relative to checkpoints/)")
    ap.add_argument("--players", nargs="+", type=int, default=[2])
    ap.add_argument("--games", type=int, default=200, help="games per matchup (per combination of agents)")
    ap.add_argument("--board", default="classic")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ck-dir", default=os.environ.get("NOCHMAL_CHECKPOINTS", "checkpoints"))
    ap.add_argument("--free-rounds", type=int, default=0)
    ap.add_argument("--start-column-always", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run_arena(args.agents, args.players, args.games, args.board, args.device, args.ck_dir,
                    args.free_rounds, args.start_column_always)
    for P, entry in res["players"].items():
        print(f"\n=== {P} Spieler ===")
        for r in entry["table"]:
            wr = f"{r['win_rate'] * 100:5.1f} %" if r["win_rate"] is not None else "   –   "
            rk = f"{r['mean_rank']:.2f}" if r["mean_rank"] is not None else "–"
            print(f"{r['agent']:40s} Spiele {r['games']:5d}  Sieg {wr}  Ø Score {r['mean_score']:6.2f}  Ø Rang {rk}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
