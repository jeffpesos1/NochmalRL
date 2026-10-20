"""PPO self-play training for *Noch mal!* – fully on the GPU.

Example::

    python -m nochmal.train --board classic --players 2 --games 2048 --iters 3000 --out checkpoints/duo

Every iteration plays ``--rounds`` rounds in ``--games`` parallel games (all
seats controlled by the same network), then runs a few PPO epochs.  Metrics are
appended to ``<out>/log.jsonl`` (read by the web dashboard) and checkpoints are
written to ``<out>/latest.pt`` and ``<out>/best.pt``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch._dynamo as _dynamo
import torch.nn.functional as F

from .agent import HeuristicPolicy, NNPolicy, RandomPolicy, SeatPolicy, load_model, save_checkpoint
from .board import load_board
from .env import NochmalEnv
from .features import OPPONENT_VIEWS, Obs, apply_opponent_view
from .model import make_model


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------- board pools
def build_board_pool(args):
    """Training boards: --board, plus --boards (comma list) plus --random-boards N generated boards."""
    names = [args.board]
    boards = [load_board(args.board)]
    if getattr(args, "boards", None):
        from .board import official_board_ids
        spec = args.boards.split(",")
        if "official" in [x.strip() for x in spec]:
            spec = [x for x in spec if x.strip() != "official"] + official_board_ids()
        for n in spec:
            n = n.strip()
            if n and n not in names:
                names.append(n)
                boards.append(load_board(n))
    if getattr(args, "random_boards", 0) > 0:
        from .boardgen import random_boards
        rb = random_boards(args.random_boards, seed=args.board_seed)
        boards += rb
        names += [b.source for b in rb]
    return boards, names


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(env: NochmalEnv, policy, n_episodes: int, max_rounds_cap: int = 200) -> Dict[str, float]:
    """Play until ``n_episodes`` games finished; returns mean final score etc. of seat 0."""
    env.reset_games(torch.ones(env.G, dtype=torch.bool, device=env.device))
    scores: List[torch.Tensor] = []
    wins: List[torch.Tensor] = []
    lengths: List[torch.Tensor] = []
    rounds = 0
    while sum(int(s.shape[0]) for s in scores) < n_episodes and rounds < max_rounds_cap:
        rnd_before = env.round.clone()
        res = env.play_round(policy, record=False)
        rounds += 1
        if bool(res.done.any()):
            fs = res.final_scores[res.done]
            jl = res.final_jokers_left[res.done]
            scores.append(fs)
            lengths.append(rnd_before[res.done] + 1)
            if env.P > 1:
                key = fs * 100 + jl                                  # tie-break: more jokers left
                best = key.max(1, keepdim=True).values
                wins.append((key[:, 0] >= best[:, 0]).to(torch.float32))
    if not scores:
        return {"eval_score": float("nan"), "eval_win": float("nan"), "eval_len": float("nan")}
    fs = torch.cat(scores)
    out = {
        "eval_score": float(fs[:, 0].mean()),
        "eval_score_std": float(fs[:, 0].std()) if fs.shape[0] > 1 else 0.0,
        "eval_len": float(torch.cat(lengths).to(torch.float32).mean()),
        "eval_episodes": int(fs.shape[0]),
    }
    if env.P > 1:
        out["eval_win"] = float(torch.cat(wins).mean())
        out["eval_opp_score"] = float(fs[:, 1:].mean())
    return out


# ---------------------------------------------------------------- buffer utils
def flatten_obs(obs_list: List[Obs]) -> Obs:
    return {k: torch.stack([o[k] for o in obs_list], 0).flatten(0, 2) for k in obs_list[0]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO training for Noch mal!")
    ap.add_argument("--board", default="classic", help="training board (also the eval board unless --eval-board)")
    ap.add_argument("--boards", default=None,
                    help="comma separated board pool for training ('official' = all official sheets); each game plays its own board")
    ap.add_argument("--random-boards", type=int, default=0, help="add N *generated* random boards to the training pool (not official)")
    ap.add_argument("--board-seed", type=int, default=0, help="seed for the generated boards")
    ap.add_argument("--eval-board", default=None, help="board for the comparable eval (default: --board, i.e. classic)")
    ap.add_argument("--eval-opponent", default="heuristic",
                    help="reference opponent for the 'vs' eval metric: heuristic, random or a checkpoint id")
    ap.add_argument("--players", type=int, default=2, help="seats per game (1 = solo rules)")
    ap.add_argument("--games", type=int, default=2048, help="parallel games")
    ap.add_argument("--rounds", type=int, default=32, help="rounds per iteration (rollout length)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--algo", choices=("ppo", "a2c", "vpg", "dqn"), default="ppo",
                    help="ppo: clipped surrogate, several epochs | a2c: GAE advantage, 1 epoch, no clipping | "
                         "vpg: REINFORCE with value baseline (lambda=1), 1 epoch | dqn: double DQN with dueling head")
    ap.add_argument("--buffer", type=int, default=400_000, help="dqn: replay capacity (transitions, on the GPU)")
    ap.add_argument("--dqn-steps", type=int, default=24, help="dqn: gradient steps per iteration")
    ap.add_argument("--dqn-batch", type=int, default=8192)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-iters", type=int, default=60, help="dqn: iterations to anneal epsilon")
    ap.add_argument("--tau", type=float, default=0.005, help="dqn: soft target update rate")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--margin-weight", type=float, default=0.0, help="terminal reward weight for (own - mean opponent score)")
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (weights + optimizer, same run)")
    ap.add_argument("--finetune-from", default=None,
                    help="checkpoint to fine-tune (weights only, fresh optimizer; enables KL/L2-to-reference, warmup, head-first training)")
    ap.add_argument("--kl-ref", type=float, default=None, help="policy-gradient finetune: weight of KL(pi || reference policy) (default 0.02)")
    ap.add_argument("--l2-ref", type=float, default=None, help="dqn finetune: weight of ||theta - theta_ref||^2 (default 1e-3)")
    ap.add_argument("--warmup-iters", type=int, default=None, help="linear LR warmup iterations (finetune default 10)")
    ap.add_argument("--freeze-trunk-iters", type=int, default=None, help="train only the heads for the first n iterations (finetune default 5)")
    ap.add_argument("--opponent-pool", default=None,
                    help="comma separated checkpoints; opponent seats are played by a random pool member in some iterations")
    ap.add_argument("--pool-prob", type=float, default=0.5, help="probability per iteration that opponents come from the pool")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast (default on with CUDA)")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile for the update step")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--free-rounds", type=int, default=0, help="initial rounds in which everybody may use all dice (rule book: 3)")
    ap.add_argument("--start-column-always", action="store_true", help="column H may start a new group at any time (rule-book wording)")
    ap.add_argument("--opponent-view", choices=OPPONENT_VIEWS, default="full",
                    help="'full': the agent sees the opponents' sheets; 'none': blind variant")
    args = ap.parse_args(argv)

    finetune = bool(args.finetune_from)
    if finetune:
        if args.lr == 3e-4:
            args.lr = 1e-4
        args.kl_ref = 0.02 if args.kl_ref is None else args.kl_ref
        args.l2_ref = 1e-3 if args.l2_ref is None else args.l2_ref
        args.warmup_iters = 10 if args.warmup_iters is None else args.warmup_iters
        args.freeze_trunk_iters = 5 if args.freeze_trunk_iters is None else args.freeze_trunk_iters
        if args.eps_start == 1.0:
            args.eps_start = 0.2            # dqn: the policy is already good, explore less
    args.kl_ref = args.kl_ref or 0.0
    args.l2_ref = args.l2_ref or 0.0
    args.warmup_iters = args.warmup_iters or 0
    args.freeze_trunk_iters = args.freeze_trunk_iters or 0
    if args.algo == "dqn":
        from .dqn import train_dqn
        if args.gamma >= 1.0:
            args.gamma = 0.99
        return train_dqn(args)
    if args.algo in ("a2c", "vpg"):
        args.epochs = 1
    if args.algo == "vpg":
        args.lam = 1.0
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    try:  # several fixed batch shapes (rollout, passive seats, eval, update) -> allow more compiled variants
        _dynamo.config.cache_size_limit = 64
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and not args.no_amp
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
    start_iter = 0
    best_score = -1e9
    ref_model = None
    resumed_meta = {}
    if args.resume:
        model, resumed_meta = load_model(args.resume, device)
        start_iter = int(resumed_meta.get("iter", 0))
        best_score = float(resumed_meta.get("best_score", -1e9))
        print(f"resumed from {args.resume} at iteration {start_iter}")
    elif finetune:
        model, src_meta = load_model(args.finetune_from, device)
        print(f"fine-tuning from {args.finetune_from} (iter {src_meta.get('iter')}, eval {src_meta.get('eval_score')})")
    if args.kl_ref > 0:
        import copy
        ref_model = copy.deepcopy(model).eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume and resumed_meta.get("optim") is not None:
        try:
            opt.load_state_dict(resumed_meta["optim"])
            print("optimizer state restored")
        except (ValueError, KeyError) as e:
            print(f"optimizer state not restored: {e}")
    trunk_params = list(model.stem.parameters()) + list(model.blocks.parameters())
    pool_specs = [x for x in (args.opponent_pool or "").split(",") if x]
    pool_policies = []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} board={board.name!r} pool={len(train_boards)} boards eval_board={eval_board.name!r} "
          f"cells={bt.N} moves={bt.K} players={args.players} games={args.games} params={n_params/1e6:.2f}M "
          f"rules={rules} opponent_view={view}")

    log_path = os.path.join(args.out, "log.jsonl")
    meta_path = os.path.join(args.out, "run.json")
    with open(meta_path, "w") as f:
        json.dump({"args": vars(args), "rules": rules, "opponent_view": view, "algo": args.algo, "board": board.source,
                   "train_boards": board_names, "eval_board": eval_board.source,
                   "finetuned_from": args.finetune_from, "opponent_pool": pool_specs, "board_name": board.name, "params": n_params,
                   "device": str(device), "started": time.time()}, f)

    # baselines for the dashboard
    heur = HeuristicPolicy(eval_bt)
    from .agent import make_policy as _mk
    ref_opp = heur if args.eval_opponent in ("heuristic", "greedy") else _mk(
        args.eval_opponent, eval_bt, os.environ.get("NOCHMAL_CHECKPOINTS", "checkpoints"), greedy=True)
    print(f"eval reference opponent: {args.eval_opponent}")
    base = evaluate(eval_env, heur, args.eval_games)
    print(f"heuristic baseline: score={base['eval_score']:.2f}")
    rnd = evaluate(eval_env, RandomPolicy(eval_bt), args.eval_games)
    print(f"random baseline:    score={rnd['eval_score']:.2f}")
    with open(log_path, "a") as f:
        f.write(json.dumps({"type": "baseline", "heuristic_score": base["eval_score"], "random_score": rnd["eval_score"],
                            "time": time.time()}) + "\n")

    train_policy = NNPolicy(bt, model, greedy=False, opponent_view=view)
    train_policy_eager = NNPolicy(bt, model, greedy=False, opponent_view=view)
    G, P, T = args.games, args.players, args.rounds
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else (lambda: torch.autocast("cpu", enabled=False))
    # compiled copy of the model for the (fixed-shape) update minibatches; rollouts use the eager model
    use_compile = device.type == "cuda" and not args.no_compile
    update_model = torch.compile(model) if use_compile else model
    ref_update = (torch.compile(ref_model) if use_compile else ref_model) if ref_model is not None else None
    from .agent import make_policy
    for spec in pool_specs:
        pool_policies.append(make_policy(spec, bt, os.environ.get("NOCHMAL_CHECKPOINTS", "checkpoints"), greedy=False))
        print(f"opponent pool: {spec}")
    if use_compile:  # rollouts/evals have fixed batch shapes (G, G*(P-1), eval G) -> compiled too (~2.5x faster)
        train_policy.model = torch.compile(model, dynamic=False)
    print(f"algo={args.algo} epochs={args.epochs} amp={use_amp} compile={use_compile}", flush=True)

    for it in range(start_iter + 1, start_iter + args.iters + 1):
        t0 = time.time()
        rel_it = it - start_iter
        lr_now = args.lr * (min(1.0, rel_it / args.warmup_iters) if args.warmup_iters > 0 else 1.0)
        for g_ in opt.param_groups:
            g_["lr"] = lr_now
        freeze = rel_it <= args.freeze_trunk_iters
        for p_ in trunk_params:
            p_.requires_grad_(not freeze)
        use_pool = bool(pool_policies) and P > 1 and (torch.rand(1).item() < args.pool_prob)
        if use_pool:
            opp = pool_policies[int(torch.randint(len(pool_policies), (1,)).item())]
            # seat subsets have varying batch sizes -> use the eager model here (compiled would recompile per size)
            rollout_policy = SeatPolicy(bt, [train_policy_eager] + [opp] * (P - 1))
        else:
            rollout_policy = train_policy
        # ---------------------------------------------------------- rollout
        obs_l, act_l, logp_l, val_l, rew_l, done_l = [], [], [], [], [], []
        ep_scores, ep_lens, ep_wins = [], [], []
        with torch.no_grad(), amp_ctx():
            for t in range(T + 1):
                rnd_before = env.round.clone()
                res = env.play_round(rollout_policy, record=True)
                obs_l.append(res.obs); act_l.append(res.action); logp_l.append(res.logp)
                val_l.append(res.value); rew_l.append(res.reward * args.reward_scale); done_l.append(res.done)
                if bool(res.done.any()):
                    fs = res.final_scores[res.done]
                    ep_scores.append(fs)
                    ep_lens.append(rnd_before[res.done] + 1)
                    if P > 1:
                        key = fs * 100 + res.final_jokers_left[res.done]
                        ep_wins.append((key[:, 0] >= key.max(1).values).to(torch.float32))
        values = torch.stack(val_l)                     # (T+1,G,P)
        rewards = torch.stack(rew_l[:T])
        dones = torch.stack(done_l[:T]).unsqueeze(-1).to(torch.float32)  # (T,G,1)
        adv = torch.zeros(T, G, P, device=device)
        last = torch.zeros(G, P, device=device)
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + args.gamma * values[t + 1] * nonterm - values[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        returns = adv + values[:T]
        if use_pool:  # the learner sits in seat 0 only
            obs_l = [{k: v[:, :1] for k, v in o.items()} for o in obs_l]
            act_l = [a_[:, :1] for a_ in act_l]; logp_l = [l_[:, :1] for l_ in logp_l]
            adv = adv[:, :, :1]; returns = returns[:, :, :1]
        b_obs = flatten_obs(obs_l[:T])
        b_act = torch.stack(act_l[:T]).flatten()
        b_logp = torch.stack(logp_l[:T]).flatten()
        b_adv = adv.flatten()
        b_ret = returns.flatten()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_act.shape[0]
        rollout_time = time.time() - t0

        # ---------------------------------------------------------- update
        model.train()
        stats = {"pl": [], "vl": [], "ent": [], "kl": [], "clipfrac": []}
        mb_size = min(args.minibatch, n)
        n_mb = max(1, n // mb_size)          # full-size minibatches only (keeps compiled shapes static)
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for s in range(n_mb):
                idx = perm[s * mb_size: (s + 1) * mb_size]
                mb = apply_opponent_view({k: v[idx] for k, v in b_obs.items()}, view)
                with amp_ctx():
                    logits, value = update_model(bt, mb)
                logits = logits.float(); value = value.float()
                logp_all = torch.log_softmax(logits, -1)
                new_logp = logp_all.gather(1, b_act[idx].unsqueeze(1)).squeeze(1)
                probs = logp_all.exp()
                ent = -(probs * logp_all.masked_fill(probs == 0, 0)).sum(-1).mean()
                ratio = (new_logp - b_logp[idx]).exp()
                a = b_adv[idx]
                if args.algo == "ppo":
                    pl = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                else:  # a2c / vpg: plain policy gradient with the advantage estimate
                    pl = -(a * new_logp).mean()
                vl = F.mse_loss(value, b_ret[idx])
                loss = pl + args.vf * vl - args.ent * ent
                if ref_update is not None:
                    with torch.no_grad(), amp_ctx():
                        ref_logits, _ = ref_update(bt, mb)
                    ref_logp = torch.log_softmax(ref_logits.float(), -1)
                    kl_ref = (probs * (logp_all - ref_logp).masked_fill(probs == 0, 0)).sum(-1).mean()
                    loss = loss + args.kl_ref * kl_ref
                    stats.setdefault("kl_ref", []).append(float(kl_ref))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                opt.step()
                with torch.no_grad():
                    stats["pl"].append(float(pl)); stats["vl"].append(float(vl)); stats["ent"].append(float(ent))
                    stats["kl"].append(float((b_logp[idx] - new_logp).mean()))
                    stats["clipfrac"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
        model.eval()
        dt = time.time() - t0

        # ---------------------------------------------------------- logging
        rec = {
            "type": "iter", "iter": it, "time": time.time(), "seconds": dt, "rollout_seconds": rollout_time,
            "samples": n, "sps": n / dt, "episodes_total": env.episodes,
            "policy_loss": float(np.mean(stats["pl"])), "value_loss": float(np.mean(stats["vl"])),
            "entropy": float(np.mean(stats["ent"])), "kl": float(np.mean(stats["kl"])),
            "clipfrac": float(np.mean(stats["clipfrac"])),
            "lr": lr_now, "trunk_frozen": bool(freeze), "pool_iteration": bool(use_pool),
        }
        if stats.get("kl_ref"):
            rec["kl_ref"] = float(np.mean(stats["kl_ref"]))
        if ep_scores:
            fs = torch.cat(ep_scores)
            rec["train_score"] = float(fs.mean())
            rec["train_score_std"] = float(fs.std()) if fs.shape[0] > 1 else 0.0
            rec["train_score_max"] = float(fs.max())
            rec["train_len"] = float(torch.cat(ep_lens).to(torch.float32).mean())
            rec["train_episodes"] = int(fs.shape[0])
        if it % args.eval_every == 0 or it == start_iter + 1:
            greedy = NNPolicy(eval_bt, train_policy.model, greedy=True, opponent_view=view)
            ev = evaluate(eval_env, greedy, args.eval_games)
            rec.update(ev)
            if P > 1:
                vs = evaluate(eval_env, SeatPolicy(eval_bt, [greedy] + [ref_opp] * (P - 1)), args.eval_games)
                rec["vs_heuristic_win"] = vs.get("eval_win", float("nan"))
                rec["vs_heuristic_score"] = vs["eval_score"]
            score = ev["eval_score"]
            if score > best_score:
                best_score = score
                save_checkpoint(os.path.join(args.out, "best.pt"), model,
                                {"iter": it, "board": board.source, "players": P, "eval_score": score,
                                 "best_score": best_score, "args": vars(args), "rules": rules, "opponent_view": view, "algo": args.algo,
                                 "finetuned_from": args.finetune_from})
                rec["new_best"] = True
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.out, "latest.pt"), model,
                            {"iter": it, "board": board.source, "players": P, "rules": rules, "opponent_view": view, "algo": args.algo,
                             "eval_score": rec.get("eval_score"), "best_score": best_score, "args": vars(args),
                             "optim": opt.state_dict(), "finetuned_from": args.finetune_from})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = (f"it {it:5d} | score {rec.get('train_score', float('nan')):6.2f} | eval {rec.get('eval_score', float('nan')):6.2f}"
               f" | len {rec.get('train_len', float('nan')):5.1f} | ent {rec['entropy']:.3f} | vl {rec['value_loss']:.3f}"
               f" | kl {rec['kl']:.4f} | {rec['sps']/1000:.1f}k sps | {dt:.1f}s")
        print(msg, flush=True)
    save_checkpoint(os.path.join(args.out, "latest.pt"), model,
                    {"iter": start_iter + args.iters, "board": board.source, "players": P, "best_score": best_score,
                     "args": vars(args), "rules": rules, "opponent_view": view, "algo": args.algo,
                     "optim": opt.state_dict(), "finetuned_from": args.finetune_from})


if __name__ == "__main__":
    main()
