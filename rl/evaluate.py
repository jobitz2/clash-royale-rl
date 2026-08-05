"""Head-to-head evaluation against scripted bots and other checkpoints.

Seats are swapped every other game so any first-move or seat advantage in the
simulator cancels out instead of inflating the score.
"""

from __future__ import annotations

import numpy as np
import torch

from cr.bots import BOT_NAMES, make_bot
from cr.env import CRDuel, outcome_score

from .net import ActorCritic
from .rollout import stack_obs


class NetPolicy:
    """Adapts a network to the scripted-bot `act(env, p, mask)` interface."""

    def __init__(self, net: ActorCritic, device="cpu", deterministic=False):
        self.net = net
        self.device = device
        self.deterministic = deterministic

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, env, p, mask):
        sp, sc = stack_obs([env.obs(p)], self.device)
        mk = torch.as_tensor(mask[None], device=self.device)
        a, _, _ = self.net.act(sp, sc, mk, deterministic=self.deterministic)
        return int(a[0])


@torch.no_grad()
def play_match(pol_a, pol_b, seed=0, a_seat=0, decks=None, frame_skip=3):
    """One battle. Returns (score_for_a, info)."""
    env = CRDuel(decks=decks, frame_skip=frame_skip, seed=seed)
    env.reset(seed=seed)
    pol_a.reset()
    pol_b.reset()
    seat_a, seat_b = a_seat, 1 - a_seat
    done = False
    info = {}
    while not done:
        # Skip inference at ticks where holding is the only legal action.
        aa = pol_a.act(env, seat_a, env.mask(seat_a)) if env.can_act(seat_a) else 0
        ab = pol_b.act(env, seat_b, env.mask(seat_b)) if env.can_act(seat_b) else 0
        a0, a1 = (aa, ab) if seat_a == 0 else (ab, aa)
        rew, done, info = env.step(a0, a1)
    return outcome_score(info["winner"], seat_a), info, env


@torch.no_grad()
def head_to_head(pol_a, pol_b, games=20, seed=0, decks=None, frame_skip=3):
    scores = []
    for g in range(games):
        s, _, _ = play_match(pol_a, pol_b, seed=seed * 100003 + g,
                             a_seat=g % 2, decks=decks, frame_skip=frame_skip)
        scores.append(s)
    return float(np.mean(scores)), scores


@torch.no_grad()
def evaluate_vs_bots(net, games=20, device="cpu", bots=BOT_NAMES, seed=0,
                     decks=None, frame_skip=3, deterministic=False):
    """Win rate of `net` against each scripted bot. Absolute progress signal."""
    net.eval()
    pol = NetPolicy(net, device=device, deterministic=deterministic)
    out = {}
    for name in bots:
        bot = make_bot(name, seed=seed + 7)
        wr, _ = head_to_head(pol, bot, games=games, seed=seed + hash(name) % 9973,
                             decks=decks, frame_skip=frame_skip)
        out[name] = wr
    return out
