"""Independent evaluation of agents on chosen "datasets" (boards / player counts / rules).

Example::

    python -m nochmal.evalrun --agents duo/best.pt heuristic --boards classic --random-boards 8 --board-seed 3 \
        --players 2 --games 512 --out checkpoints/evals/x.json

For every agent and every board set the agent plays ``games`` games with all seats controlled by itself
(greedy), and – for P >= 2 – additionally seat 0 against the heuristic in the other seats.  Reported:
mean/std score, game length, win rate vs heuristic.  Board sets: 'classic' (or any board id) = that board;
'random:N:seed' = N generated boards, each game on its own board (the same seed always gives the same set,
so results stay comparable).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from .agent import HeuristicPolicy, SeatPolicy, make_policy
from .board import load_board
from .boardgen import random_boards
from .env import NochmalEnv
from .train import evaluate


def board_set(spec: str):
    """'classic' | '<board id>' | 'official' (all official sheets, each game on its own) | 'random:N:seed'  ->  (label, [boards])"""
    if spec == "official":
        from .board import official_board_ids
        ids = official_board_ids()
        return f"alle offiziellen Blätter ({len(ids)})", [load_board(i) for i in ids]
    if spec.startswith("random:"):
        parts = spec.split(":")
        n = int(parts[1]); seed = int(parts[2]) if len(parts) > 2 else 0
        return f"random×{n} (seed {seed})", random_boards(n, seed=seed)
    return spec, [load_board(spec)]


def run_eval(agents: List[str], board_specs: List[str], players: int, games: int, device: str = "auto",
             ck_dir: str = "checkpoints", free_rounds: int = 0, start_column_always: bool = False,
             seed: int = 123, progress=print, opponent: str = "heuristic") -> Dict:
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    torch.backends.cudnn.benchmark = True
    out: Dict = {"agents": agents, "board_sets": board_specs, "players": players, "games": games, "seed": seed,
                 "free_rounds": free_rounds, "start_column_always": start_column_always, "opponent": opponent,
                 "results": [], "started": time.time()}
    t0 = time.time()
    for spec in board_specs:
        label, boards = board_set(spec)
        env = NochmalEnv(boards, max(64, games), players, dev, free_rounds=free_rounds,
                         start_column_always=start_column_always, seed=seed)
        bt = env.bt
        heur = HeuristicPolicy(bt) if opponent in ("heuristic", "greedy") else make_policy(opponent, bt, ck_dir, greedy=True)
        for a in agents:
            pol = make_policy(a, bt, ck_dir, greedy=True)
            env.gen.manual_seed(seed)                      # identical dice for every agent
            ev = evaluate(env, pol, games)
            row = {"agent": a, "board_set": label, "spec": spec, "eval_score": ev["eval_score"],
                   "eval_score_std": ev.get("eval_score_std"), "eval_len": ev["eval_len"], "games": ev["eval_episodes"]}
            if players > 1:
                env.gen.manual_seed(seed + 1)
                vs = evaluate(env, SeatPolicy(bt, [pol] + [heur] * (players - 1)), games)
                row["vs_heuristic_win"] = vs.get("eval_win")      # 'vs reference opponent' (name kept for the dashboard)
                row["vs_heuristic_score"] = vs["eval_score"]
                row["opponent"] = opponent
            out["results"].append(row)
            progress(f"{label} | {a}: score {row['eval_score']:.2f}" + (f", win vs {opponent} {row['vs_heuristic_win']*100:.0f} %" if players > 1 else "") + f" ({time.time()-t0:.0f}s)")
    out["seconds"] = time.time() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent agent evaluation")
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--boards", nargs="+", default=["classic"], help="board ids and/or random:N:seed")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ck-dir", default=os.environ.get("NOCHMAL_CHECKPOINTS", "checkpoints"))
    ap.add_argument("--free-rounds", type=int, default=0)
    ap.add_argument("--start-column-always", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--opponent", default="heuristic", help="reference opponent: heuristic, random or checkpoint id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run_eval(args.agents, args.boards, args.players, args.games, args.device, args.ck_dir,
                   args.free_rounds, args.start_column_always, args.seed, opponent=args.opponent)
    print()
    for r in res["results"]:
        extra = f"  Sieg vs {r.get('opponent', 'heuristic')} {r['vs_heuristic_win']*100:5.1f} %" if r.get("vs_heuristic_win") is not None else ""
        print(f"{r['board_set']:24s} {r['agent']:40s} Ø {r['eval_score']:6.2f} ± {r['eval_score_std'] or 0:5.2f}  Länge {r['eval_len']:4.1f}{extra}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
