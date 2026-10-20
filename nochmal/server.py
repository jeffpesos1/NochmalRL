"""FastAPI web server: game API + training dashboard API + static web UI.

Run with ``python -m nochmal.server`` (or uvicorn nochmal.server:app).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import BasePolicy, list_checkpoints, make_policy
from .board import Board, list_boards, load_board
from .features import BoardTensors
from .game import GameSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
CK_DIR = os.environ.get("NOCHMAL_CHECKPOINTS", os.path.join(ROOT, "checkpoints"))
BOARDS_DIR = os.environ.get("NOCHMAL_BOARDS", os.path.join(ROOT, "boards"))
_dev = os.environ.get("NOCHMAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if _dev.startswith("cuda") and not torch.cuda.is_available():
    print("WARNING: NOCHMAL_DEVICE=cuda but no CUDA device is visible (NVIDIA Container Toolkit missing?) – using CPU")
    _dev = "cpu"
DEVICE = torch.device(_dev)

app = FastAPI(title="Noch mal! RL")

_boards: Dict[str, Board] = {}
_bts: Dict[str, BoardTensors] = {}
_policies: Dict[str, BasePolicy] = {}
_games: Dict[str, GameSession] = {}


def get_board(bid: str) -> Board:
    if bid not in _boards:
        _boards[bid] = load_board(bid, BOARDS_DIR)
        _bts[bid] = BoardTensors(_boards[bid], DEVICE)
    return _boards[bid]


def get_policy(bid: str, spec: str, start_column_always: bool = False) -> BasePolicy:
    key = f"{bid}:{int(start_column_always)}::{spec}"
    if key not in _policies:
        board = get_board(bid)
        bt = _bts[bid] if not start_column_always else BoardTensors(board, DEVICE, start_column_always=True)
        _policies[key] = make_policy(spec, bt, CK_DIR, greedy=False, temperature=0.5)
    return _policies[key]


# ---------------------------------------------------------------- models
class PlayerSpec(BaseModel):
    name: str
    type: str = "human"              # 'human' | 'agent' | 'extern' (real player, sheet not tracked)
    agent: Optional[str] = None      # 'heuristic' | 'random' | checkpoint id


class NewGame(BaseModel):
    board: str = "classic"
    players: List[PlayerSpec]
    seed: Optional[int] = None
    max_rounds: Optional[int] = None
    free_rounds: int = 0                 # rule book: 3
    start_column_always: bool = False    # rule-book wording: True
    opponent_view: str = "full"          # 'full' | 'none' (agents play blind)


class DiceSpec(BaseModel):
    colors: List[str]
    numbers: List


class MoveSpec(BaseModel):
    cells: Optional[List[int]] = None   # None = pass


class ExternSpec(BaseModel):
    dice: Optional[Dict] = None          # {"color": key|"joker", "number": int|"joker"} taken by the active extern player
    columns: List[int] = []
    colors: List[int] = []
    finished: bool = False


class AgentSpec(BaseModel):
    agent: Optional[str] = None
    top_n: int = 8


class AgentPlay(BaseModel):
    until_human: bool = True
    auto_roll: bool = True
    greedy: bool = True


class ArenaSpec(BaseModel):
    agents: List[str]
    players: List[int] = [2]
    games: int = 200
    board: str = "classic"
    free_rounds: int = 0
    start_column_always: bool = False


class TrainSpec(BaseModel):
    run: str = "run"
    board: str = "classic"
    players: int = 2
    games: int = 2048
    rounds: int = 32
    iters: int = 1000
    lr: float = 3e-4
    ent: float = 0.01
    margin_weight: float = 0.0
    eval_every: int = 20
    channels: int = 64
    blocks: int = 4
    resume: Optional[str] = None
    extra: Optional[str] = None
    free_rounds: int = 0
    start_column_always: bool = False
    opponent_view: str = "full"      # 'full' | 'none' (blind)
    algo: str = "ppo"                # 'ppo' | 'a2c' | 'vpg' | 'dqn'
    minibatch: Optional[int] = None  # PPO minibatch (reduce for very large networks)
    finetune_from: Optional[str] = None
    kl_ref: Optional[float] = None
    l2_ref: Optional[float] = None
    warmup_iters: Optional[int] = None
    freeze_trunk_iters: Optional[int] = None
    opponent_pool: Optional[List[str]] = None
    boards: Optional[str] = None             # extra training boards (comma separated ids, or 'official')
    random_boards: int = 0                   # generated boards added to the training pool
    board_seed: int = 0
    eval_opponent: str = "heuristic"         # reference opponent for the 'vs' metric


# ---------------------------------------------------------------- static
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/info")
def info():
    return {
        "device": str(DEVICE),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }


@app.get("/api/boards")
def boards():
    return list_boards(BOARDS_DIR)


@app.get("/api/boards/{bid}")
def board_detail(bid: str):
    try:
        return get_board(bid).to_public_dict()
    except FileNotFoundError:
        raise HTTPException(404, "board not found")


ENSEMBLES_FILE = os.path.join(CK_DIR, "ensembles.json")


def _ensembles() -> List[Dict]:
    if os.path.isfile(ENSEMBLES_FILE):
        with open(ENSEMBLES_FILE) as f:
            return json.load(f)
    return []


class EnsembleSpec(BaseModel):
    members: List[str]
    mode: str = "mean"
    label: Optional[str] = None


@app.post("/api/ensembles")
def ensembles_add(spec: EnsembleSpec):
    if spec.mode not in ("mean", "vote") or len(spec.members) < 2:
        raise HTTPException(400, "mode must be mean|vote and at least 2 members")
    sid = f"ensemble:{spec.mode}:{'+'.join(spec.members)}"
    ens = _ensembles()
    if not any(e["id"] == sid for e in ens):
        ens.append({"id": sid, "label": spec.label or f"Ensemble {spec.mode} ({len(spec.members)}): " + ", ".join(m.split('/')[0] for m in spec.members)})
        os.makedirs(CK_DIR, exist_ok=True)
        with open(ENSEMBLES_FILE, "w") as f:
            json.dump(ens, f, indent=1)
    return {"id": sid, "ensembles": ens}


@app.delete("/api/ensembles")
def ensembles_delete(id: str):
    ens = [e for e in _ensembles() if e["id"] != id]
    with open(ENSEMBLES_FILE, "w") as f:
        json.dump(ens, f, indent=1)
    return {"ensembles": ens}


@app.get("/api/agents")
def agents():
    out = [{"id": "heuristic", "label": "Heuristik (ohne Training)"}, {"id": "random", "label": "Zufall"}]
    for e in _ensembles():
        out.append({"id": e["id"], "label": e["label"], "ensemble": True, "strict_rules": True, "opponent_view": "full"})
    for ck in list_checkpoints(CK_DIR):
        lbl = ck["id"]
        if ck.get("iter") is not None:
            lbl += f"  (Iter {ck['iter']}"
            if ck.get("eval_score") is not None:
                lbl += f", Ø {ck['eval_score']:.1f} Pkt"
            lbl += ")"
        rules = ck.get("rules") or {"free_rounds": 3, "start_column_always": True}   # old checkpoints = rule-book variant
        strict = rules.get("free_rounds", 0) == 0 and not rules.get("start_column_always", False)
        if not strict:
            lbl += " · Regelheft-Variante"
        if ck.get("opponent_view", "full") == "none":
            lbl += " · blind"
        if ck.get("algo", "ppo") != "ppo":
            lbl += f" · {ck['algo'].upper()}"
        if ck.get("players"):
            lbl += f" · {ck['players']}P"
        out.append({"id": ck["id"], "label": lbl, "board": ck.get("board"), "players": ck.get("players"),
                    "iter": ck.get("iter"), "eval_score": ck.get("eval_score"), "rules": rules, "strict_rules": strict,
                    "opponent_view": ck.get("opponent_view", "full"), "algo": ck.get("algo", "ppo")})
    return out


# ---------------------------------------------------------------- games
def _game(gid: str) -> GameSession:
    g = _games.get(gid)
    if g is None:
        raise HTTPException(404, "game not found")
    return g


@app.post("/api/games")
def new_game(spec: NewGame):
    if not spec.players:
        raise HTTPException(400, "at least one player")
    board = get_board(spec.board)
    policies: Dict[str, BasePolicy] = {}
    players = []
    for p in spec.players:
        agent = p.agent or "heuristic"
        if p.type == "agent" or agent:
            try:
                policies[agent] = get_policy(spec.board, agent, spec.start_column_always)
            except FileNotFoundError as e:
                raise HTTPException(400, str(e))
        players.append({"name": p.name, "type": p.type, "agent": agent})
    if "heuristic" not in policies:
        policies["heuristic"] = get_policy(spec.board, "heuristic", spec.start_column_always)
    gid = uuid.uuid4().hex[:8]
    if spec.opponent_view not in ("full", "none"):
        raise HTTPException(400, "opponent_view must be 'full' or 'none'")
    _games[gid] = GameSession(board, players, policies, DEVICE, seed=spec.seed, max_rounds=spec.max_rounds,
                              free_rounds=spec.free_rounds, start_column_always=spec.start_column_always,
                              opponent_view=spec.opponent_view)
    # keep memory bounded
    if len(_games) > 50:
        oldest = sorted(_games, key=lambda k: _games[k].created)[0]
        del _games[oldest]
    return {"id": gid, "state": _games[gid].to_dict()}


@app.get("/api/games/{gid}")
def game_state(gid: str):
    return _game(gid).to_dict()


@app.delete("/api/games/{gid}")
def delete_game(gid: str):
    _games.pop(gid, None)
    return {"ok": True}


@app.post("/api/games/{gid}/roll")
def roll(gid: str):
    g = _game(gid)
    try:
        g.roll()
    except AssertionError as e:
        raise HTTPException(400, str(e))
    return g.to_dict()


@app.post("/api/games/{gid}/dice")
def set_dice(gid: str, spec: DiceSpec):
    g = _game(gid)
    try:
        g.set_dice(spec.colors, spec.numbers)
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))
    return g.to_dict()


@app.get("/api/games/{gid}/legal")
def legal(gid: str):
    return {"moves": _game(gid).legal_moves()}


@app.post("/api/games/{gid}/move")
def move(gid: str, spec: MoveSpec):
    g = _game(gid)
    try:
        entry = g.play_move(spec.cells, by="human")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"entry": entry, "state": g.to_dict()}


@app.post("/api/games/{gid}/extern")
def extern_turn(gid: str, spec: ExternSpec):
    g = _game(gid)
    try:
        entry = g.play_extern(spec.dice, spec.columns, spec.colors, spec.finished)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"entry": entry, "state": g.to_dict()}


@app.post("/api/games/{gid}/suggest")
def suggest(gid: str, spec: AgentSpec):
    g = _game(gid)
    if spec.agent and spec.agent not in g.policies:
        g.policies[spec.agent] = get_policy(g.board.source, spec.agent, g.env.start_column_always)
    return g.agent_distribution(spec=spec.agent, top_n=spec.top_n)


@app.post("/api/games/{gid}/agent_play")
def agent_play(gid: str, spec: AgentPlay):
    """Let the agent of the current seat move; optionally continue through all agent seats
    (and dice rolls) until a human has to decide or the game ends."""
    g = _game(gid)
    entries = []
    for _ in range(200):
        if g.phase == "finished":
            break
        if g.phase == "dice":
            if not spec.auto_roll:
                break
            g.roll()
            continue
        p = g.current_player()
        if g.players[p]["type"] == "extern":
            break  # public info about this player must be entered by hand
        if g.players[p]["type"] != "agent" and entries:
            break  # a human has to decide now
        dist = g.agent_distribution(p=p, top_n=1)
        if spec.greedy:
            if dist["suggestions"] and dist["suggestions"][0]["prob"] >= dist["pass_prob"]:
                cells = dist["suggestions"][0]["cells"]
            else:
                cells = None
        else:
            policy = g.policies[g.players[p].get("agent") or next(iter(g.policies))]
            a, _, _ = policy(g._obs(p))
            k = int(a[0])
            cells = list(g.board.move_cells[k]) if k < g.env.bt.K else None
        entries.append(g.play_move(cells, by="agent"))
        if not spec.until_human:
            break
    return {"entries": entries, "state": g.to_dict()}


@app.post("/api/games/{gid}/undo")
def undo(gid: str):
    g = _game(gid)
    g.undo()
    return g.to_dict()


# ---------------------------------------------------------------- training (several runs in parallel)
MAX_PARALLEL = int(os.environ.get("NOCHMAL_MAX_TRAININGS", "2"))
_trainings: Dict[str, Dict] = {}          # run name -> {"proc", "pid", "started", "cmd"}
_lock = threading.Lock()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _run_alive(name: str) -> bool:
    t = _trainings.get(name)
    if t is None:
        return False
    if t.get("proc") is not None:
        return t["proc"].poll() is None
    return bool(t.get("pid")) and _pid_alive(t["pid"])


def _discover_trainings() -> None:
    """Re-attach to training processes (pid files) after a server restart; drop finished ones."""
    if os.path.isdir(CK_DIR):
        for name in os.listdir(CK_DIR):
            pf = os.path.join(CK_DIR, name, "train.pid")
            if os.path.isfile(pf) and name not in _trainings:
                try:
                    pid = int(open(pf).read().strip())
                except ValueError:
                    continue
                if _pid_alive(pid):
                    _trainings[name] = {"proc": None, "pid": pid, "started": os.path.getmtime(pf), "cmd": "(re-attached)"}
    for name in list(_trainings):
        if not _run_alive(name):
            pf = os.path.join(CK_DIR, name, "train.pid")
            if os.path.isfile(pf):
                try:
                    os.remove(pf)
                except OSError:
                    pass
            t = _trainings.pop(name)
            if t.get("proc") is not None:
                _last_exit[name] = t["proc"].returncode


_last_exit: Dict[str, Optional[int]] = {}


def _running_runs() -> List[str]:
    with _lock:
        _discover_trainings()
        return sorted(_trainings)


def _training_alive() -> bool:
    return bool(_running_runs())


def _runs() -> List[Dict]:
    out = []
    if not os.path.isdir(CK_DIR):
        return out
    running = set(_running_runs())
    for name in sorted(os.listdir(CK_DIR)):
        d = os.path.join(CK_DIR, name)
        lp = os.path.join(d, "log.jsonl")
        if os.path.isdir(d) and os.path.isfile(lp):
            meta = {}
            mp = os.path.join(d, "run.json")
            if os.path.isfile(mp):
                with open(mp) as f:
                    meta = json.load(f)
            out.append({"run": name, "meta": meta, "modified": os.path.getmtime(lp), "running": name in running})
    out.sort(key=lambda r: -r["modified"])
    return out


def _tail(path: str, n: int = 4000) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", "replace")


@app.get("/api/training/runs")
def training_runs():
    return _runs()


@app.delete("/api/training/runs/{run}")
def training_delete(run: str):
    name = "".join(ch for ch in run if ch.isalnum() or ch in "-_")
    if not name or name != run:
        raise HTTPException(400, "invalid run name")
    if name in _running_runs():
        raise HTTPException(409, "run is active – stop it first")
    d = os.path.join(CK_DIR, name)
    if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "log.jsonl")):
        raise HTTPException(404, "run not found")
    import shutil
    shutil.rmtree(d)
    _policies.clear()
    return {"deleted": name}


@app.get("/api/training/log")
def training_log(run: str, since: int = 0, limit: int = 5000):
    lp = os.path.join(CK_DIR, run, "log.jsonl")
    if not os.path.isfile(lp):
        raise HTTPException(404, "run not found")
    recs = []
    with open(lp) as f:
        for i, line in enumerate(f):
            if i < since:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    total = since + len(recs)
    if len(recs) > limit:
        step = len(recs) / limit
        recs = [recs[int(i * step)] for i in range(limit)]
    meta = {}
    mp = os.path.join(CK_DIR, run, "run.json")
    if os.path.isfile(mp):
        with open(mp) as f:
            meta = json.load(f)
    return {"run": run, "records": recs, "total": total, "meta": meta, "running": run in _running_runs()}


@app.get("/api/training/status")
def training_status(run: Optional[str] = None):
    running = _running_runs()
    sel = run or (running[0] if running else None)
    tail = _tail(os.path.join(CK_DIR, sel, "train.out")) if sel else ""
    return {"running": bool(running), "runs": running, "max_parallel": MAX_PARALLEL, "run": sel, "tail": tail,
            "returncode": _last_exit.get(sel) if sel else None,
            "pipeline": _pipeline_status()}


def _spawn_training(spec: "TrainSpec") -> str:
    with _lock:
        _discover_trainings()
        if len(_trainings) >= MAX_PARALLEL:
            raise HTTPException(409, f"already {len(_trainings)} training run(s) active (limit {MAX_PARALLEL})")
        run = "".join(ch for ch in spec.run if ch.isalnum() or ch in "-_") or "run"
        if run in _trainings:
            raise HTTPException(409, f"run '{run}' is already active")
        out = os.path.join(CK_DIR, run)
        if not spec.resume and any(os.path.isfile(os.path.join(out, f)) for f in ("best.pt", "latest.pt")):
            raise HTTPException(409, f"run '{run}' already has checkpoints – choose another name, resume it, or delete it first")
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, "-m", "nochmal.train", "--board", spec.board, "--players", str(spec.players),
               "--games", str(spec.games), "--rounds", str(spec.rounds), "--iters", str(spec.iters),
               "--lr", str(spec.lr), "--ent", str(spec.ent), "--margin-weight", str(spec.margin_weight),
               "--eval-every", str(spec.eval_every), "--channels", str(spec.channels), "--blocks", str(spec.blocks),
               "--free-rounds", str(spec.free_rounds), "--opponent-view", spec.opponent_view, "--algo", spec.algo,
               "--out", out, "--device", os.environ.get("NOCHMAL_TRAIN_DEVICE", "auto")]
        if spec.start_column_always:
            cmd.append("--start-column-always")
        if spec.minibatch:
            cmd += ["--minibatch", str(spec.minibatch)]
        if spec.resume:
            cmd += ["--resume", spec.resume if os.path.isfile(spec.resume) else os.path.join(CK_DIR, spec.resume)]
        if spec.finetune_from:
            cmd += ["--finetune-from", spec.finetune_from if os.path.isfile(spec.finetune_from) else os.path.join(CK_DIR, spec.finetune_from)]
            if spec.kl_ref is not None:
                cmd += ["--kl-ref", str(spec.kl_ref)]
            if spec.l2_ref is not None:
                cmd += ["--l2-ref", str(spec.l2_ref)]
            if spec.warmup_iters is not None:
                cmd += ["--warmup-iters", str(spec.warmup_iters)]
            if spec.freeze_trunk_iters is not None:
                cmd += ["--freeze-trunk-iters", str(spec.freeze_trunk_iters)]
        if spec.opponent_pool:
            cmd += ["--opponent-pool", ",".join(spec.opponent_pool)]
        if spec.boards:
            cmd += ["--boards", spec.boards]
        if spec.random_boards:
            cmd += ["--random-boards", str(spec.random_boards), "--board-seed", str(spec.board_seed)]
        cmd += ["--eval-opponent", spec.eval_opponent]
        if spec.extra:
            cmd += spec.extra.split()
        logf = open(os.path.join(out, "train.out"), "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1", NOCHMAL_BOARDS=BOARDS_DIR, NOCHMAL_CHECKPOINTS=CK_DIR,
                   PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
        with open(os.path.join(out, "train.pid"), "w") as f:
            f.write(str(proc.pid))
        _trainings[run] = {"proc": proc, "pid": proc.pid, "started": time.time(), "cmd": " ".join(cmd)}
        return run


@app.post("/api/training/start")
def training_start(spec: TrainSpec):
    run = _spawn_training(spec)
    _policies.clear()
    return training_status(run)


def _stop_run(name: str) -> None:
    t = _trainings.get(name)
    if t is None:
        return
    if t.get("proc") is not None:
        t["proc"].send_signal(signal.SIGINT)
        try:
            t["proc"].wait(timeout=20)
        except subprocess.TimeoutExpired:
            t["proc"].kill()
    elif t.get("pid"):
        os.kill(t["pid"], signal.SIGINT)
        for _ in range(40):
            if not _pid_alive(t["pid"]):
                break
            time.sleep(0.5)
        else:
            os.kill(t["pid"], signal.SIGKILL)


@app.post("/api/training/stop")
def training_stop(run: Optional[str] = None):
    names = [run] if run else _running_runs()
    for n in names:
        _stop_run(n)
    _policies.clear()
    return training_status()


# ---------------------------------------------------------------- pipeline: finetune all -> arena -> champion
class PipelineSpec(BaseModel):
    bases: Optional[List[str]] = None        # checkpoint ids; default: every */best.pt
    iters: int = 60
    players: int = 2
    games: int = 2048
    use_pool: bool = True                    # opponents from the pool of all bases
    arena_players: List[int] = [2, 3]
    arena_games: int = 200
    suffix: str = "-ft"
    board: str = "classic"
    free_rounds: int = 0
    start_column_always: bool = False


_pipeline: Dict[str, object] = {"thread": None, "stage": "idle", "started": None, "runs": [], "log": [], "arena": None,
                                "champion": None, "error": None}


def _pipeline_status() -> Dict:
    return {k: v for k, v in _pipeline.items() if k != "thread"}


def _all_best_checkpoints() -> List[str]:
    return [ck["id"] for ck in list_checkpoints(CK_DIR) if ck["id"].endswith("best.pt") and "/" in ck["id"]]


def _champion_from_result(res: Dict) -> Optional[Dict]:
    """Best agent = highest mean win rate over the multi-player tables (tie: mean score)."""
    agg: Dict[str, List[float]] = {}
    scores: Dict[str, List[float]] = {}
    for P, entry in res.get("players", {}).items():
        for row in entry["table"]:
            if row["win_rate"] is not None:
                agg.setdefault(row["agent"], []).append(row["win_rate"])
            scores.setdefault(row["agent"], []).append(row["mean_score"])
    if not agg and not scores:
        return None
    def key(a):
        wr = sum(agg.get(a, [0])) / max(1, len(agg.get(a, [0])))
        return (wr, sum(scores[a]) / len(scores[a]))
    best = max(scores, key=key)
    return {"agent": best, "win_rate": key(best)[0], "mean_score": key(best)[1]}


def _pipeline_worker(spec: PipelineSpec) -> None:
    try:
        bases = spec.bases or _all_best_checkpoints()
        bases = [b for b in bases if b not in ("heuristic", "random")]
        _pipeline.update({"stage": "finetune", "runs": [], "log": [f"{len(bases)} Basis-Checkpoints"], "error": None})
        pending = list(bases)
        started: List[str] = []
        pool = bases if spec.use_pool else []
        while pending or any(r in _running_runs() for r in started):
            while pending and len(_running_runs()) < MAX_PARALLEL:
                base = pending.pop(0)
                run_name = base.split("/")[0] + spec.suffix
                ts = TrainSpec(run=run_name, board=spec.board, players=spec.players, games=spec.games, iters=spec.iters,
                               finetune_from=base, opponent_pool=[p for p in pool if p != base] or None,
                               free_rounds=spec.free_rounds, start_column_always=spec.start_column_always)
                try:
                    _spawn_training(ts)
                    started.append(run_name)
                    _pipeline["runs"] = list(started)
                    _pipeline["log"].append(f"gestartet: {run_name} (aus {base})")
                except HTTPException as e:
                    _pipeline["log"].append(f"nicht gestartet: {run_name}: {e.detail}")
            time.sleep(5)
        _pipeline.update({"stage": "arena"})
        agents = ["heuristic"] + bases + [f"{r}/best.pt" for r in started if os.path.isfile(os.path.join(CK_DIR, r, "best.pt"))]
        _pipeline["log"].append(f"Turnier mit {len(agents)} Agenten")
        from .arena import run_arena
        res = run_arena(agents, spec.arena_players, spec.arena_games, spec.board, "auto", CK_DIR,
                        spec.free_rounds, spec.start_column_always, progress=lambda m: _pipeline["log"].append(m))
        os.makedirs(ARENA_DIR, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + "-pipeline"
        with open(os.path.join(ARENA_DIR, name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        champ = _champion_from_result(res)
        if champ:
            champ["arena"] = name
            with open(os.path.join(CK_DIR, "champion.json"), "w") as f:
                json.dump(champ, f)
        _pipeline.update({"stage": "done", "arena": name, "champion": champ})
        _policies.clear()
    except Exception as e:  # pragma: no cover
        _pipeline.update({"stage": "error", "error": str(e)})


@app.post("/api/pipeline/finetune_all")
def pipeline_start(spec: PipelineSpec):
    th = _pipeline.get("thread")
    if th is not None and th.is_alive():
        raise HTTPException(409, "pipeline already running")
    _pipeline.update({"stage": "starting", "started": time.time(), "runs": [], "log": [], "arena": None, "champion": None})
    t = threading.Thread(target=_pipeline_worker, args=(spec,), daemon=True)
    _pipeline["thread"] = t
    t.start()
    return _pipeline_status()


@app.get("/api/pipeline/status")
def pipeline_status():
    return _pipeline_status()


@app.get("/api/champion")
def champion():
    cp = os.path.join(CK_DIR, "champion.json")
    if os.path.isfile(cp):
        with open(cp) as f:
            return json.load(f)
    # fall back: best agent of the most recent arena result
    res = arena_results()
    if res:
        c = _champion_from_result(res[0]["result"])
        if c:
            c["arena"] = res[0]["id"]
            return c
    return {"agent": None}


# ---------------------------------------------------------------- arena
ARENA_DIR = os.path.join(CK_DIR, "arena")
_arena: Dict[str, object] = {"proc": None, "out": None, "started": None, "log": None}


def _arena_alive() -> bool:
    return _arena["proc"] is not None and _arena["proc"].poll() is None


@app.get("/api/arena/status")
def arena_status():
    tail = ""
    if _arena["log"] and os.path.isfile(_arena["log"]):
        with open(_arena["log"], "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 3000))
            tail = f.read().decode("utf-8", "replace")
    return {"running": _arena_alive(), "out": _arena["out"], "started": _arena["started"], "tail": tail}


@app.post("/api/arena/start")
def arena_start(spec: ArenaSpec):
    if _arena_alive():
        raise HTTPException(409, "an arena run is already active")
    if len(spec.agents) < 1:
        raise HTTPException(400, "choose at least one agent")
    os.makedirs(ARENA_DIR, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(ARENA_DIR, name + ".json")
    cmd = [sys.executable, "-m", "nochmal.arena", "--agents", *spec.agents, "--players", *map(str, spec.players),
           "--games", str(spec.games), "--board", spec.board, "--free-rounds", str(spec.free_rounds),
           "--ck-dir", CK_DIR, "--out", out]
    if spec.start_column_always:
        cmd.append("--start-column-always")
    log = os.path.join(ARENA_DIR, name + ".log")
    logf = open(log, "ab")
    env = dict(os.environ, PYTHONUNBUFFERED="1", NOCHMAL_BOARDS=BOARDS_DIR)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
    _arena.update({"proc": proc, "out": out, "started": time.time(), "log": log})
    return arena_status()


@app.get("/api/arena/results")
def arena_results():
    out = []
    if os.path.isdir(ARENA_DIR):
        for fn in sorted(os.listdir(ARENA_DIR), reverse=True):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(ARENA_DIR, fn)) as f:
                        res = json.load(f)
                    out.append({"id": fn[:-5], "result": res})
                except (OSError, json.JSONDecodeError):
                    pass
    return out


# ---------------------------------------------------------------- independent evaluation
EVAL_DIR = os.path.join(CK_DIR, "evals")
_eval: Dict[str, object] = {"proc": None, "out": None, "started": None, "log": None}


class EvalSpec(BaseModel):
    agents: List[str]
    boards: List[str] = ["classic"]          # board ids and/or 'random:N:seed'
    players: int = 2
    games: int = 512
    seed: int = 123
    free_rounds: int = 0
    start_column_always: bool = False
    label: Optional[str] = None
    opponent: str = "heuristic"


def _eval_alive() -> bool:
    return _eval["proc"] is not None and _eval["proc"].poll() is None


@app.get("/api/eval/status")
def eval_status():
    return {"running": _eval_alive(), "out": _eval["out"], "started": _eval["started"],
            "tail": _tail(_eval["log"], 3000) if _eval["log"] else ""}


@app.post("/api/eval/start")
def eval_start(spec: EvalSpec):
    if _eval_alive():
        raise HTTPException(409, "an evaluation is already running")
    if not spec.agents:
        raise HTTPException(400, "choose at least one agent")
    os.makedirs(EVAL_DIR, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + ("-" + "".join(ch for ch in spec.label if ch.isalnum() or ch in "-_") if spec.label else "")
    out = os.path.join(EVAL_DIR, name + ".json")
    cmd = [sys.executable, "-m", "nochmal.evalrun", "--agents", *spec.agents, "--boards", *spec.boards,
           "--players", str(spec.players), "--games", str(spec.games), "--seed", str(spec.seed),
           "--free-rounds", str(spec.free_rounds), "--ck-dir", CK_DIR, "--out", out, "--opponent", spec.opponent]
    if spec.start_column_always:
        cmd.append("--start-column-always")
    log = os.path.join(EVAL_DIR, name + ".log")
    logf = open(log, "ab")
    env = dict(os.environ, PYTHONUNBUFFERED="1", NOCHMAL_BOARDS=BOARDS_DIR)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
    _eval.update({"proc": proc, "out": out, "started": time.time(), "log": log})
    return eval_status()


@app.get("/api/eval/results")
def eval_results():
    out = []
    if os.path.isdir(EVAL_DIR):
        for fn in sorted(os.listdir(EVAL_DIR), reverse=True):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(EVAL_DIR, fn)) as f:
                        out.append({"id": fn[:-5], "result": json.load(f)})
                except (OSError, json.JSONDecodeError):
                    pass
    return out


@app.delete("/api/eval/results/{eid}")
def eval_delete(eid: str):
    for ext in (".json", ".log"):
        p = os.path.join(EVAL_DIR, eid + ext)
        if os.path.isfile(p):
            os.remove(p)
    return {"deleted": eid}


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import uvicorn

    uvicorn.run("nochmal.server:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")),
                reload=False)


if __name__ == "__main__":
    main()
