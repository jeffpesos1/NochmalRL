"""Double DQN with dueling head for *Noch mal!* (``python -m nochmal.train --algo dqn``).

Value-based alternative to the policy-gradient algorithms: the network's per-move
outputs are interpreted as advantages A(s,a), the value head as V(s), and
Q(s,a) = V(s) + A(s,a) - mean_{legal a} A(s,a).  Illegal moves are masked out of
every max.  Transitions (one per player decision, next state = the same
player's next decision incl. the new dice) are stored in a replay buffer that
lives on the GPU.  Acting is epsilon-greedy over legal moves, targets use the
double-DQN rule with a soft-updated target network.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .agent import HeuristicPolicy, NNPolicy, RandomPolicy, SeatPolicy, load_model, save_checkpoint
from .board import load_board
from .env import NochmalEnv
from .features import Obs, apply_opponent_view
from .model import make_model

OBS_KEYS = ("marked", "opp_marked", "opp_best", "pool_c", "pool_n", "col_claimed", "color_claimed", "jokers_used", "scalars", "board")


def q_values(model, bt, obs: Obs):
    """Masked Q(s,a) (M,K+1) from the shared network (dueling combination)."""
    adv, v = model(bt, obs)
    legal = torch.isfinite(adv)
    adv0 = adv.masked_fill(~legal, 0.0)
    mean_adv = adv0.sum(1) / legal.sum(1).clamp(min=1)
    q = v.unsqueeze(1) + adv0 - mean_adv.unsqueeze(1)
    return q.masked_fill(~legal, float("-inf")), legal


class EpsGreedyPolicy(NNPolicy):
    """Acts greedily w.r.t. Q with probability 1-eps, otherwise uniformly over legal moves."""

    def __init__(self, bt, model, eps: float = 0.1, opponent_view: str = "full"):
        super().__init__(bt, model, greedy=True, opponent_view=opponent_view, label="dqn")
        self.eps = eps

    @torch.no_grad()
    def logits_and_values(self, obs: Obs):
        self.model.eval()
        q, _ = q_values(self.model, self.bt, apply_opponent_view(obs, self.opponent_view))
        return q.float(), q.max(1).values.float()

    @torch.no_grad()
    def __call__(self, obs: Obs, p=None):
        q, value = self.logits_and_values(obs)
        greedy = q.argmax(1)
        if self.eps > 0:
            legal = torch.isfinite(q)
            rnd = torch.multinomial(legal.to(torch.float32), 1).squeeze(1)
            explore = torch.rand(q.shape[0], device=q.device) < self.eps
            action = torch.where(explore, rnd, greedy)
        else:
            action = greedy
        logp = torch.zeros_like(value)
        return action, logp, value


class Replay:
    def __init__(self, capacity: int, proto: Obs, device):
        self.cap = capacity
        self.n = 0
        self.ptr = 0
        self.device = device
        self.obs = {k: torch.zeros((capacity,) + tuple(v.shape[1:]), dtype=v.dtype, device=device) for k, v in proto.items()}
        self.nxt = {k: torch.zeros_like(v) for k, v in self.obs.items()}
        self.action = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, device=device)

    def add(self, obs: Obs, nxt: Obs, action, reward, done):
        m = action.shape[0]
        idx = (torch.arange(m, device=self.device) + self.ptr) % self.cap
        for k in self.obs:
            self.obs[k][idx] = obs[k]
            self.nxt[k][idx] = nxt[k]
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        self.ptr = (self.ptr + m) % self.cap
        self.n = min(self.cap, self.n + m)

    def sample(self, batch: int):
        idx = torch.randint(0, self.n, (batch,), device=self.device)
        return ({k: v[idx] for k, v in self.obs.items()}, {k: v[idx] for k, v in self.nxt.items()},
                self.action[idx], self.reward[idx], self.done[idx])


def train_dqn(args) -> None:
    from .train import evaluate, pick_device  # shared helpers

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 64
    except Exception:
        pass
    use_amp = device.type == "cuda" and not args.no_amp
    use_compile = device.type == "cuda" and not args.no_compile
    from .train import build_board_pool
    board = load_board(args.board)
    train_boards, board_names = build_board_pool(args)
    eval_board = load_board(args.eval_board) if args.eval_board else board
    os.makedirs(args.out, exist_ok=True)
    rules = dict(free_rounds=args.free_rounds, start_column_always=args.start_column_always, max_rounds=args.max_rounds)
    view = args.opponent_view
    env = NochmalEnv(train_boards, args.games, args.players, device, margin_weight=args.margin_weight, seed=args.seed, **rules)
    eval_env = NochmalEnv(eval_board, max(64, args.eval_games), args.players, device, seed=args.seed + 1, **rules)
    bt = env.bt
    eval_bt = eval_env.bt

    model = make_model({"channels": args.channels, "blocks": args.blocks}).to(device)
    start_iter, best_score = 0, -1e9
    resumed_meta = {}
    if args.resume:
        model, resumed_meta = load_model(args.resume, device)
        start_iter = int(resumed_meta.get("iter", 0))
        best_score = float(resumed_meta.get("best_score", -1e9))
    elif args.finetune_from:
        model, src_meta = load_model(args.finetune_from, device)
        print(f"fine-tuning from {args.finetune_from} (iter {src_meta.get('iter')}, eval {src_meta.get('eval_score')})")
    ref_params = [p.detach().clone() for p in model.parameters()] if args.l2_ref > 0 else None
    trunk_params = list(model.stem.parameters()) + list(model.blocks.parameters())
    target = copy.deepcopy(model).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume and resumed_meta.get("optim") is not None:
        try:
            opt.load_state_dict(resumed_meta["optim"])
        except (ValueError, KeyError):
            pass
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} board={board.name!r} cells={bt.N} moves={bt.K} players={args.players} games={args.games} "
          f"params={n_params/1e6:.2f}M rules={rules} opponent_view={view}")
    print(f"algo=dqn buffer={args.buffer} steps/iter={args.dqn_steps} batch={args.dqn_batch} eps {args.eps_start}->{args.eps_end} "
          f"over {args.eps_iters} iters gamma={args.gamma} tau={args.tau} amp={use_amp} compile={use_compile}", flush=True)

    log_path = os.path.join(args.out, "log.jsonl")
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"args": vars(args), "rules": rules, "opponent_view": view, "algo": "dqn", "board": board.source,
                   "board_name": board.name, "params": n_params, "device": str(device), "started": time.time(),
                   "finetuned_from": args.finetune_from, "train_boards": board_names, "eval_board": eval_board.source}, f)
    heur = HeuristicPolicy(eval_bt)
    from .agent import make_policy as _mk
    ref_opp = heur if args.eval_opponent in ("heuristic", "greedy") else _mk(
        args.eval_opponent, eval_bt, os.environ.get("NOCHMAL_CHECKPOINTS", "checkpoints"), greedy=True)
    print(f"eval reference opponent: {args.eval_opponent}")
    base = evaluate(eval_env, heur, args.eval_games)
    rnd = evaluate(eval_env, RandomPolicy(eval_bt), args.eval_games)
    print(f"heuristic baseline: score={base['eval_score']:.2f}\nrandom baseline:    score={rnd['eval_score']:.2f}")
    with open(log_path, "a") as f:
        f.write(json.dumps({"type": "baseline", "heuristic_score": base["eval_score"], "random_score": rnd["eval_score"],
                            "time": time.time()}) + "\n")

    act_model = torch.compile(model, dynamic=False) if use_compile else model
    upd_model = torch.compile(model) if use_compile else model
    tgt_model = torch.compile(target) if use_compile else target
    policy = EpsGreedyPolicy(bt, act_model, eps=args.eps_start, opponent_view=view)
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else (lambda: torch.autocast("cpu", enabled=False))
    G, P, T = args.games, args.players, args.rounds
    replay = None
    steps_done = 0

    for it in range(start_iter + 1, start_iter + args.iters + 1):
        t0 = time.time()
        rel_it = it - start_iter
        frac = min(1.0, (rel_it - 1) / max(1, args.eps_iters))
        policy.eps = args.eps_start + (args.eps_end - args.eps_start) * frac
        lr_now = args.lr * (min(1.0, rel_it / args.warmup_iters) if args.warmup_iters > 0 else 1.0)
        for g_ in opt.param_groups:
            g_["lr"] = lr_now
        freeze = rel_it <= args.freeze_trunk_iters
        for p_ in trunk_params:
            p_.requires_grad_(not freeze)
        obs_l, act_l, rew_l, done_l = [], [], [], []
        ep_scores, ep_lens, ep_wins = [], [], []
        with torch.no_grad(), amp_ctx():
            for t in range(T + 1):
                rnd_before = env.round.clone()
                res = env.play_round(policy, record=True)
                obs_l.append(res.obs); act_l.append(res.action); rew_l.append(res.reward * args.reward_scale); done_l.append(res.done)
                if bool(res.done.any()):
                    fs = res.final_scores[res.done]
                    ep_scores.append(fs); ep_lens.append(rnd_before[res.done] + 1)
                    if P > 1:
                        key = fs * 100 + res.final_jokers_left[res.done]
                        ep_wins.append((key[:, 0] >= key.max(1).values).to(torch.float32))
        # transitions: (obs_t, a_t, r_t, obs_{t+1}, done_t) per stream
        flat = lambda lst: {k: torch.stack([o[k] for o in lst], 0).flatten(0, 2) for k in OBS_KEYS}
        cur = flat(obs_l[:T]); nxt = flat(obs_l[1:])
        if view == "none":
            cur, nxt = apply_opponent_view(cur, view), apply_opponent_view(nxt, view)
        a = torch.stack(act_l[:T]).flatten()
        r = torch.stack(rew_l[:T]).flatten()
        d = torch.stack(done_l[:T]).unsqueeze(-1).expand(-1, -1, P).to(torch.float32).flatten()
        if replay is None:
            replay = Replay(args.buffer, cur, device)
        replay.add(cur, nxt, a, r, d)
        rollout_time = time.time() - t0

        # ---------------------------------------------------------- updates
        model.train()
        losses, qmeans = [], []
        n_steps = args.dqn_steps if replay.n >= args.dqn_batch else 0
        for _ in range(n_steps):
            o, o2, a_b, r_b, d_b = replay.sample(args.dqn_batch)
            with torch.no_grad(), amp_ctx():
                q2_online, _ = q_values(upd_model, bt, o2)
                a_star = q2_online.float().argmax(1)
                q2_target, _ = q_values(tgt_model, bt, o2)
                y = r_b + args.gamma * (1.0 - d_b) * q2_target.float().gather(1, a_star.unsqueeze(1)).squeeze(1)
            with amp_ctx():
                q, _ = q_values(upd_model, bt, o)
            qa = q.float().gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(qa, y)
            if ref_params is not None:  # L2-SP: stay close to the starting weights
                loss = loss + args.l2_ref * sum(((p - r0) ** 2).sum() for p, r0 in zip(model.parameters(), ref_params))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
            opt.step()
            with torch.no_grad():
                for tp, mp in zip(target.parameters(), model.parameters()):
                    tp.lerp_(mp, args.tau)
            losses.append(float(loss)); qmeans.append(float(qa.mean()))
            steps_done += 1
        model.eval()
        dt = time.time() - t0
        n = a.shape[0]
        rec = {"type": "iter", "iter": it, "time": time.time(), "seconds": dt, "rollout_seconds": rollout_time,
               "samples": n, "sps": n / dt, "episodes_total": env.episodes,
               "policy_loss": 0.0, "value_loss": float(np.mean(losses)) if losses else float("nan"),
               "entropy": policy.eps, "kl": 0.0, "clipfrac": 0.0, "lr": lr_now, "trunk_frozen": bool(freeze),
               "q_mean": float(np.mean(qmeans)) if qmeans else float("nan"), "replay": replay.n, "grad_steps": steps_done}
        if ep_scores:
            fs = torch.cat(ep_scores)
            rec.update({"train_score": float(fs.mean()), "train_score_std": float(fs.std()) if fs.shape[0] > 1 else 0.0,
                        "train_score_max": float(fs.max()), "train_len": float(torch.cat(ep_lens).to(torch.float32).mean()),
                        "train_episodes": int(fs.shape[0])})
        if it % args.eval_every == 0 or it == start_iter + 1:
            greedy = EpsGreedyPolicy(eval_bt, act_model, eps=0.0, opponent_view=view)
            ev = evaluate(eval_env, greedy, args.eval_games)
            rec.update(ev)
            if P > 1:
                vs = evaluate(eval_env, SeatPolicy(eval_bt, [greedy] + [ref_opp] * (P - 1)), args.eval_games)
                rec["vs_heuristic_win"] = vs.get("eval_win", float("nan")); rec["vs_heuristic_score"] = vs["eval_score"]
            if ev["eval_score"] > best_score:
                best_score = ev["eval_score"]
                save_checkpoint(os.path.join(args.out, "best.pt"), model,
                                {"iter": it, "board": board.source, "players": P, "eval_score": ev["eval_score"],
                                 "best_score": best_score, "args": vars(args), "rules": rules, "opponent_view": view, "algo": "dqn",
                                 "finetuned_from": args.finetune_from})
                rec["new_best"] = True
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.out, "latest.pt"), model,
                            {"iter": it, "board": board.source, "players": P, "rules": rules, "opponent_view": view, "algo": "dqn",
                             "eval_score": rec.get("eval_score"), "best_score": best_score, "args": vars(args),
                             "optim": opt.state_dict(), "finetuned_from": args.finetune_from})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"it {it:5d} | score {rec.get('train_score', float('nan')):6.2f} | eval {rec.get('eval_score', float('nan')):6.2f}"
              f" | len {rec.get('train_len', float('nan')):5.1f} | eps {policy.eps:.3f} | td {rec['value_loss']:.4f}"
              f" | q {rec['q_mean']:.3f} | {rec['sps']/1000:.1f}k sps | {dt:.1f}s", flush=True)
    save_checkpoint(os.path.join(args.out, "latest.pt"), model,
                    {"iter": start_iter + args.iters, "board": board.source, "players": P, "best_score": best_score,
                     "args": vars(args), "rules": rules, "opponent_view": view, "algo": "dqn",
                     "optim": opt.state_dict(), "finetuned_from": args.finetune_from})
