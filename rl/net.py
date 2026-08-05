"""Policy/value network.

One conv at full arena resolution, then a 2x downsample onto the placement grid
where the remaining convs run. That makes the placement head's resolution an
architectural fact (its output is exactly [4, PH, PW]) and cuts trunk FLOPs
~3.4x versus keeping every layer at full resolution.

Scalar state (elixir, clock, hand, tower health) is embedded and broadcast into
the spatial stream as a per-channel bias, so placement decisions are conditioned
on which cards are actually in hand.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cr.env import N_ACTIONS, N_CHANNELS, PH, PW, PLACE_STRIDE, SCALAR_DIM

# Finite, not -inf: Categorical.entropy() multiplies logits by probs, and
# 0 * -inf is NaN whereas 0 * -1e9 is 0.
NEG_INF = -1e9


def _init(m, gain=1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    return m


class ActorCritic(nn.Module):
    def __init__(self, width: int = 64, scalar_width: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            _init(nn.Conv2d(N_CHANNELS, 32, 3, padding=1), 2 ** 0.5), nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _init(nn.Conv2d(32, width, 3, padding=1), 2 ** 0.5),
            nn.GroupNorm(8, width), nn.ReLU(),
            _init(nn.Conv2d(width, width, 3, padding=1), 2 ** 0.5),
            nn.GroupNorm(8, width), nn.ReLU(),
        )
        self.scalar = nn.Sequential(
            _init(nn.Linear(SCALAR_DIM, scalar_width), 2 ** 0.5), nn.ReLU(),
            _init(nn.Linear(scalar_width, scalar_width), 2 ** 0.5), nn.ReLU(),
            nn.LayerNorm(scalar_width),
        )
        self.to_spatial = _init(nn.Linear(scalar_width, width), 1.0)

        self.joint_norm = nn.LayerNorm(width + scalar_width)
        self.place_head = _init(nn.Conv2d(width, 4, 1), 0.01)
        self.hold_head = _init(nn.Linear(width + scalar_width, 1), 0.01)
        self.value = nn.Sequential(
            _init(nn.Linear(width + scalar_width, 256), 2 ** 0.5), nn.ReLU(),
            _init(nn.Linear(256, 1), 1.0),
        )

    def forward(self, spatial, scalar):
        h = self.stem(spatial)                                 # [B, 32, H, W]
        h = F.avg_pool2d(h, PLACE_STRIDE)                      # [B, 32, PH, PW]
        h = self.blocks(h)                                     # [B, C, PH, PW]

        s = self.scalar(scalar)
        h = F.relu(h + self.to_spatial(s)[:, :, None, None])

        pooled = h.mean(dim=(2, 3))                            # [B, C]
        joint = self.joint_norm(torch.cat([pooled, s], dim=1))

        place = self.place_head(h).reshape(h.shape[0], 4 * PH * PW)
        logits = torch.cat([self.hold_head(joint), place], dim=1)
        return logits, self.value(joint).squeeze(-1)

    # ------------------------------------------------------------------ policy

    @staticmethod
    def _masked_dist(logits, mask):
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, spatial, scalar, mask, deterministic=False):
        logits, value = self(spatial, scalar)
        dist = self._masked_dist(logits, mask)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, spatial, scalar, mask, action):
        logits, value = self(spatial, scalar)
        dist = self._masked_dist(logits, mask)
        return dist.log_prob(action), dist.entropy(), value


def build(device="cpu", **kw) -> ActorCritic:
    return ActorCritic(**kw).to(device)


assert 1 + 4 * PH * PW == N_ACTIONS, "action head size must match the env"
