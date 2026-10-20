"""Board-agnostic policy/value network.

The network never sees a move index directly.  It produces

* a *per-cell* logit map (convolutional, so it works for any board size),
* a state-conditioned weight vector for the hand-made per-move features,
* a bias per move size, a pass logit and a state value.

The logit of a move is the sum of its cells' logits plus the weighted features.
Because moves are just sets of cells, the same weights apply to any board.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import MAX_COLORS, MAX_NUMBER, N_DYN_CHANNELS, N_MOVE_FEATS, N_SCALARS, BoardTensors, Obs

N_GLOBAL = (MAX_COLORS + 1) + (MAX_NUMBER + 1) + N_SCALARS
IN_CHANNELS = (MAX_COLORS + 2) + N_DYN_CHANNELS + N_GLOBAL
MAX_MOVE_SIZE = 8


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class PolicyNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, hidden: int = 256):
        super().__init__()
        self.cfg = {"channels": channels, "blocks": blocks, "hidden": hidden}
        self.stem = nn.Sequential(nn.Conv2d(IN_CHANNELS, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.cell_head = nn.Sequential(nn.Conv2d(channels, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.glob = nn.Sequential(nn.Linear(2 * channels + N_GLOBAL, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value_head = nn.Linear(hidden, 1)
        self.pass_head = nn.Linear(hidden, 1)
        self.feat_head = nn.Linear(hidden, N_MOVE_FEATS)
        self.size_bias = nn.Parameter(torch.zeros(MAX_MOVE_SIZE + 1))
        self.feat_bias = nn.Parameter(torch.zeros(N_MOVE_FEATS))
        nn.init.zeros_(self.cell_head[-1].weight)
        nn.init.zeros_(self.cell_head[-1].bias)
        nn.init.zeros_(self.feat_head.weight)
        nn.init.zeros_(self.feat_head.bias)
        nn.init.zeros_(self.pass_head.weight)
        nn.init.constant_(self.pass_head.bias, -2.0)

    def forward(self, bt: BoardTensors, obs: Obs, legal: torch.Tensor | None = None, feats: torch.Tensor | None = None):
        """Returns masked logits (M,K+1) and value (M,)."""
        if legal is None:
            legal, need = bt.legal_moves(obs)
            feats = bt.move_features(obs, need)
        x = bt.grid_inputs(obs, legal)                        # (M,C,H,W)
        glob_vec = x[:, -N_GLOBAL:, 0, 0]                      # the broadcast global vector
        h = self.blocks(self.stem(x))
        cell = self.cell_head(h).flatten(1)                   # (M,N)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3)), glob_vec], 1)
        g = self.glob(pooled)
        value = self.value_head(g).squeeze(-1)
        pass_logit = self.pass_head(g)                        # (M,1)
        w = self.feat_head(g) + self.feat_bias                # (M,F)
        move_logits = bt.move_logits(cell, obs)               # (M,K) sum of the cells' logits per move
        move_logits = move_logits + (feats * w.unsqueeze(1)).sum(-1) + self.size_bias[bt.move_size[bt._bid(obs)]]
        logits = torch.cat([move_logits, pass_logit], 1)
        logits = logits.masked_fill(~legal, float("-inf"))
        return logits, value


def make_model(cfg: dict | None = None) -> PolicyNet:
    cfg = cfg or {}
    return PolicyNet(**{k: cfg[k] for k in ("channels", "blocks", "hidden") if k in cfg})
