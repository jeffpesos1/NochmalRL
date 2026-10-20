"""Decision policies: neural network (trained), heuristic baseline and random.

All policies share the interface ``policy(obs, p) -> (action, logp, value)``
where ``obs`` is a batch of compact observations and ``p`` the seat index of
each decision (ignored by single-agent policies).  ``distribution(obs)``
returns the action probabilities for the UI.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import torch

from .board import Board
from .features import BoardTensors, Obs, apply_opponent_view
from .features import MAX_COLORS
from .model import PolicyNet, make_model


class BasePolicy:
    name = "base"

    def __init__(self, bt: BoardTensors):
        self.bt = bt

    def logits(self, obs: Obs) -> torch.Tensor:
        raise NotImplementedError

    def values(self, obs: Obs) -> torch.Tensor:
        return torch.zeros(obs["marked"].shape[0], device=obs["marked"].device)

    logit_scale = 1.0

    def distribution(self, obs: Obs) -> torch.Tensor:
        return torch.softmax(self.logits(obs) * self.logit_scale, -1)

    def __call__(self, obs: Obs, p: Optional[torch.Tensor] = None):
        logits = self.logits(obs)
        probs = torch.softmax(logits, -1)
        action = torch.multinomial(probs, 1).squeeze(1)
        logp = torch.log_softmax(logits, -1).gather(1, action.unsqueeze(1)).squeeze(1)
        return action, logp, self.values(obs)


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, bt: BoardTensors, pass_weight: float = 0.02):
        super().__init__(bt)
        self.pass_weight = pass_weight

    def logits(self, obs: Obs) -> torch.Tensor:
        legal, _ = self.bt.legal_moves(obs)
        lg = torch.zeros(legal.shape, device=legal.device)
        lg[:, -1] = float(torch.log(torch.tensor(self.pass_weight)))
        return lg.masked_fill(~legal, float("-inf"))


class HeuristicPolicy(BasePolicy):
    """Greedy hand-written evaluation of the per-move features (useful as baseline and as
    fallback when no trained checkpoint exists)."""

    name = "heuristic"
    # size, stars, jokers, completes_region, region_fill, completes_color, n_cols, col_pts, color_pts, ends_game, new_frontier, opp_overlap
    WEIGHTS = [1.0, 2.0, -1.2, 0.8, 0.3, 4.0, 0.5, 4.0, 4.0, 0.5, 0.6, -0.1]

    def __init__(self, bt: BoardTensors, temperature: float = 0.3):
        super().__init__(bt)
        self.temperature = temperature
        self.w = torch.tensor(self.WEIGHTS, device=bt.device)

    def logits(self, obs: Obs) -> torch.Tensor:
        legal, need = self.bt.legal_moves(obs)
        feats = self.bt.move_features(obs, need)
        score = (feats * self.w).sum(-1)                           # (M,K)
        pass_score = torch.full((score.shape[0], 1), -1.0, device=score.device)
        lg = torch.cat([score, pass_score], 1) / self.temperature
        return lg.masked_fill(~legal, float("-inf"))


class NNPolicy(BasePolicy):
    name = "nn"

    def __init__(self, bt: BoardTensors, model: PolicyNet, greedy: bool = False, temperature: float = 1.0,
                 label: str = "nn", opponent_view: str = "full"):
        super().__init__(bt)
        self.model = model.to(bt.device)
        self.greedy = greedy
        self.temperature = temperature
        self.name = label
        self.opponent_view = opponent_view      # 'full' = sees the opponents' sheets, 'none' = blind

    @torch.no_grad()
    def logits_and_values(self, obs: Obs):
        self.model.eval()
        logits, value = self.model(self.bt, apply_opponent_view(obs, self.opponent_view))
        return logits.float(), value.float()

    def logits(self, obs: Obs) -> torch.Tensor:
        lg, _ = self.logits_and_values(obs)
        return lg

    @torch.no_grad()
    def __call__(self, obs: Obs, p: Optional[torch.Tensor] = None):
        logits, value = self.logits_and_values(obs)
        if self.greedy:
            action = logits.argmax(-1)
        else:
            probs = torch.softmax(logits / self.temperature, -1)
            action = torch.multinomial(probs, 1).squeeze(1)
        logp = torch.log_softmax(logits, -1).gather(1, action.unsqueeze(1)).squeeze(1)
        return action, logp, value

    def values(self, obs: Obs) -> torch.Tensor:
        _, v = self.logits_and_values(obs)
        return v


class SeatPolicy(BasePolicy):
    """Dispatch to a different policy per seat (used for evaluation: NN vs heuristic)."""

    name = "seats"

    def __init__(self, bt: BoardTensors, policies: List[BasePolicy]):
        super().__init__(bt)
        self.policies = policies

    def __call__(self, obs: Obs, p: Optional[torch.Tensor] = None):
        M = obs["marked"].shape[0]
        d = obs["marked"].device
        if p is None:
            p = torch.zeros(M, dtype=torch.int64, device=d)
        action = torch.zeros(M, dtype=torch.int64, device=d)
        logp = torch.zeros(M, device=d)
        value = torch.zeros(M, device=d)
        for seat, pol in enumerate(self.policies):
            m = p == seat
            if bool(m.any()):
                sub = {k: v[m] for k, v in obs.items()}
                a, l, v = pol(sub)
                action[m], logp[m], value[m] = a, l, v
        return action, logp, value


class EnsemblePolicy(BasePolicy):
    """Combine several policies.

    mode 'mean': average the members' action distributions (soft voting), act greedily on the mean.
    mode 'vote': every member votes for its greedy move; most votes win, ties broken by the mean distribution.
    Spec string: ``ensemble:<mode>:<spec1>+<spec2>+...``
    """

    name = "ensemble"

    def __init__(self, bt: BoardTensors, members: List[BasePolicy], mode: str = "mean", label: str = "ensemble"):
        super().__init__(bt)
        assert mode in ("mean", "vote")
        self.members = members
        self.mode = mode
        self.name = label

    @torch.no_grad()
    def distribution(self, obs: Obs) -> torch.Tensor:
        dists = torch.stack([m.distribution(obs) for m in self.members], 0)        # (n,M,K+1)
        mean = dists.mean(0)
        if self.mode == "mean":
            return mean
        votes = torch.zeros_like(mean)
        votes.scatter_add_(1, dists.argmax(-1).t(), torch.ones(mean.shape[0], len(self.members), device=mean.device))
        return votes + 1e-3 * mean                                               # tie-break by mean probability

    def logits(self, obs: Obs) -> torch.Tensor:
        return torch.log(self.distribution(obs).clamp_min(1e-12))

    @torch.no_grad()
    def __call__(self, obs: Obs, p: Optional[torch.Tensor] = None):
        d = self.distribution(obs)
        action = d.argmax(-1)
        return action, torch.log(d.gather(1, action.unsqueeze(1)).squeeze(1).clamp_min(1e-12)), self.values(obs)

    def values(self, obs: Obs) -> torch.Tensor:
        return torch.stack([m.values(obs) for m in self.members], 0).mean(0)


def parse_ensemble(spec: str):
    """'ensemble:mean:a+b+c' -> ('mean', ['a','b','c']) or None."""
    if not spec.startswith("ensemble:"):
        return None
    _, mode, members = spec.split(":", 2)
    return mode, [m for m in members.split("+") if m]


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, model: PolicyNet, extra: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg, **extra}, path)


def load_model(path: str, device="cpu") -> tuple[PolicyNet, Dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = make_model(ck.get("model_cfg"))
    sd = ck["model"]
    # older checkpoints were trained with fewer input channels: pad the stem with zeros
    w = sd.get("stem.0.weight")
    tgt = model.stem[0].weight
    if w is not None and w.shape[1] < tgt.shape[1]:
        pad = torch.zeros(w.shape[0], tgt.shape[1] - w.shape[1], *w.shape[2:], dtype=w.dtype, device=w.device)
        sd = dict(sd)
        sd["stem.0.weight"] = torch.cat([w[:, : (MAX_COLORS + 2) + 2], pad, w[:, (MAX_COLORS + 2) + 2:]], 1)
    model.load_state_dict(sd)
    model.to(device).eval()
    meta = {k: v for k, v in ck.items() if k not in ("model",)}
    return model, meta


def list_checkpoints(ck_dir: str) -> List[Dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(ck_dir, "**", "*.pt"), recursive=True)):
        rel = os.path.relpath(p, ck_dir)
        info = {"id": rel, "path": p}
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            info.update({k: ck.get(k) for k in ("iter", "board", "players", "eval_score", "rules", "opponent_view", "algo", "finetuned_from") if k in ck})
        except Exception as e:  # pragma: no cover
            info["error"] = str(e)
        out.append(info)
    return out


def make_policy(spec: str, bt: BoardTensors, ck_dir: str = "checkpoints", greedy: bool = True,
                temperature: float = 1.0) -> BasePolicy:
    """``spec``: 'heuristic', 'random', a checkpoint id/path or 'ensemble:<mean|vote>:<spec>+<spec>...'."""
    ens = parse_ensemble(spec)
    if ens is not None:
        mode, members = ens
        return EnsemblePolicy(bt, [make_policy(m, bt, ck_dir, greedy=True, temperature=temperature) for m in members],
                              mode=mode, label=spec)
    if spec in ("heuristic", "greedy"):
        return HeuristicPolicy(bt)
    if spec == "random":
        return RandomPolicy(bt)
    path = spec if os.path.isfile(spec) else os.path.join(ck_dir, spec)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint {spec} not found")
    model, meta = load_model(path, bt.device)
    if meta.get("algo") == "dqn":
        from .dqn import EpsGreedyPolicy
        pol = EpsGreedyPolicy(bt, model, eps=0.0 if greedy else 0.05, opponent_view=meta.get("opponent_view", "full"))
        pol.name = os.path.basename(spec)
        pol.logit_scale = 20.0      # Q is in 0.1-point units; ~0.5 point temperature for the displayed distribution
        return pol
    return NNPolicy(bt, model, greedy=greedy, temperature=temperature, label=os.path.basename(spec),
                    opponent_view=meta.get("opponent_view", "full"))
