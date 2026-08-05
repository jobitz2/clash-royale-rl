"""Self-play rollout collection at decision points (semi-MDP).

Every environment holds one battle in which the learner occupies a randomised
seat and the other seat is played by a league opponent (a frozen snapshot or a
scripted bot). Only the learner's transitions are stored, so the data stays
on-policy for PPO.

Why decision points: for most of a match no card in hand is affordable, so the
only legal action is "hold". Storing those ticks would fill the batch with
states that have a single legal action -- zero policy gradient, zero entropy,
and a wasted network forward. Instead a transition is stored only where a real
choice exists, and the rewards of the forced ticks that follow are accumulated
into it. That turns the problem into a semi-MDP: a transition spanning `k` ticks
carries reward sum(gamma^j * r_j) and bootstraps with gamma^k, both tracked here
and consumed by `compute_gae`.

Learner inference is batched across environments; opponent inference is batched
per distinct opponent, and skipped entirely for opponents that are also at a
forced tick.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from cr.bots import make_bot
from cr.env import CRDuel, N_ACTIONS, N_CHANNELS, SCALAR_DIM, outcome_score
from cr import arena as A

from .league import Entry, League
from .net import ActorCritic


def stack_obs(obs_list, device):
    sp = np.stack([o["spatial"] for o in obs_list])
    sc = np.stack([o["scalar"] for o in obs_list])
    return (torch.as_tensor(sp, device=device),
            torch.as_tensor(sc, device=device))


class EpisodeStats:
    """Rolling window of finished-episode summaries."""

    def __init__(self, window=200):
        self.window = window
        self.keys = ("win", "draw", "len", "ret", "crowns_for", "crowns_against",
                     "decisions", "plays")
        self.data = {k: [] for k in self.keys}
        self.vs = defaultdict(list)

    def push(self, **kw):
        for k, v in kw.items():
            if k == "opp":
                self.vs[v].append(kw["win_score"])
                if len(self.vs[v]) > self.window:
                    del self.vs[v][0]
            elif k in self.data:
                self.data[k].append(v)
                if len(self.data[k]) > self.window:
                    del self.data[k][0]

    def summary(self):
        m = lambda l: float(np.mean(l)) if l else float("nan")
        d = self.data
        return {"winrate": m(d["win"]), "drawrate": m(d["draw"]),
                "ep_len": m(d["len"]), "ep_return": m(d["ret"]),
                "crowns_for": m(d["crowns_for"]),
                "crowns_against": m(d["crowns_against"]),
                "decisions": m(d["decisions"]), "plays": m(d["plays"])}

    def vs_table(self):
        return {k: float(np.mean(v)) for k, v in self.vs.items() if v}


class SelfPlayCollector:
    def __init__(self, net: ActorCritic, league: League, n_envs: int = 32,
                 horizon: int = 128, device="cpu", frame_skip: int = 3,
                 seed: int = 0, decks=None, reward=None, gamma: float = 0.997):
        self.net = net
        self.league = league
        self.n = n_envs
        self.horizon = horizon
        self.device = device
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.envs = [CRDuel(decks=decks, frame_skip=frame_skip,
                            seed=int(self.rng.integers(1 << 30)), reward=reward)
                     for _ in range(n_envs)]
        self.stats = EpisodeStats()

        self._opp_nets: dict[int, ActorCritic] = {}
        self._opp_bots: dict[int, object] = {}
        self.seat = np.zeros(n_envs, dtype=np.int64)
        self.opp: list[Entry] = [None] * n_envs

        # Per-env accumulator state (see module docstring).
        self.slot = np.full(n_envs, -1, dtype=np.int64)
        self.filled = np.zeros(n_envs, dtype=np.int64)
        self.disc = np.ones(n_envs, dtype=np.float64)
        self.ep_return = np.zeros(n_envs, dtype=np.float64)
        self.ep_ticks = np.zeros(n_envs, dtype=np.int64)
        self.ep_decisions = np.zeros(n_envs, dtype=np.int64)
        self.ep_plays = np.zeros(n_envs, dtype=np.int64)
        self._slot_start = np.zeros(n_envs, dtype=np.int64)  # tick a slot opened
        self.total_ticks = 0        # monotonic across episodes, for throughput
        self.finished: list[dict] = []   # episode records, drained by the trainer

        for i in range(n_envs):
            self._reset_env(i)
        self._alloc()
        self.last_legal = float("nan")
        self.last_skip = float("nan")

    # ------------------------------------------------------------- opponents

    def _opponent_policy(self, e: Entry):
        if e.kind == "bot":
            if e.uid not in self._opp_bots:
                self._opp_bots[e.uid] = make_bot(
                    e.name, seed=int(self.rng.integers(1 << 30)))
            return self._opp_bots[e.uid]
        net = self._opp_nets.get(e.uid)
        if net is None:
            net = ActorCritic().to(self.device)
            net.load_state_dict(e.params)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._opp_nets[e.uid] = net
        return net

    def prune_opponent_cache(self):
        live = {e.uid for e in self.league.entries}
        for d in (self._opp_nets, self._opp_bots):
            for uid in [u for u in d if u not in live]:
                del d[uid]

    def _reset_env(self, i):
        self.envs[i].reset(seed=int(self.rng.integers(1 << 30)))
        self.seat[i] = int(self.rng.integers(2))
        self.opp[i] = self.league.sample()
        if self.opp[i].kind == "bot":
            self._opponent_policy(self.opp[i]).reset()
        self.slot[i] = -1
        self.disc[i] = 1.0
        self.ep_return[i] = 0.0
        self.ep_ticks[i] = 0
        self.ep_decisions[i] = 0
        self.ep_plays[i] = 0

    # --------------------------------------------------------------- buffers

    def _alloc(self):
        T, N = self.horizon, self.n
        d = self.device
        self.b_spatial = torch.zeros((T, N, N_CHANNELS, A.H, A.W), device=d)
        self.b_scalar = torch.zeros((T, N, SCALAR_DIM), device=d)
        self.b_mask = torch.zeros((T, N, N_ACTIONS), dtype=torch.bool, device=d)
        self.b_action = torch.zeros((T, N), dtype=torch.long, device=d)
        self.b_logp = torch.zeros((T, N), device=d)
        self.b_value = torch.zeros((T, N), device=d)
        self.b_reward = torch.zeros((T, N), device=d)
        self.b_done = torch.zeros((T, N), device=d)
        self.b_gk = torch.ones((T, N), device=d)          # gamma^k per transition
        self.last_value = torch.zeros(N, device=d)

    # ---------------------------------------------------------------- collect

    @torch.no_grad()
    def _opponent_actions(self, active):
        """Batched actions for the opponent seat of every active env."""
        actions = {}
        groups: dict[int, list[int]] = defaultdict(list)
        for i in active:
            p = 1 - int(self.seat[i])
            if not self.envs[i].can_act(p):
                actions[i] = 0                      # forced hold, no forward
            else:
                groups[self.opp[i].uid].append(i)

        for uid, idxs in groups.items():
            entry = self.opp[idxs[0]]
            pol = self._opponent_policy(entry)
            if entry.kind == "bot":
                for i in idxs:
                    p = 1 - int(self.seat[i])
                    actions[i] = int(pol.act(self.envs[i], p, self.envs[i].mask(p)))
                continue
            ps = [1 - int(self.seat[i]) for i in idxs]
            sp, sc = stack_obs([self.envs[i].obs(p) for i, p in zip(idxs, ps)],
                               self.device)
            mk = torch.as_tensor(
                np.stack([self.envs[i].mask(p) for i, p in zip(idxs, ps)]),
                device=self.device)
            a, _, _ = pol.act(sp, sc, mk)
            for k, i in enumerate(idxs):
                actions[i] = int(a[k])
        return actions

    @torch.no_grad()
    def collect(self):
        self.net.eval()
        T = self.horizon
        gamma = self.gamma
        self.b_reward.zero_()
        self.b_done.zero_()
        self.b_gk.fill_(1.0)
        self.filled[:] = 0
        self.slot[:] = -1
        self.disc[:] = 1.0

        frozen = np.zeros(self.n, dtype=bool)
        legal_counts, skips = [], []

        while not frozen.all():
            active = [i for i in range(self.n) if not frozen[i]]

            # 1. Which active envs are at a learner decision point?
            deciding, holding = [], []
            for i in active:
                seat = int(self.seat[i])
                (deciding if self.envs[i].can_act(seat) else holding).append(i)

            # 2. One batched forward covers both storing a decision and, for
            #    envs that already filled the buffer, the bootstrap value.
            learner_action = {i: 0 for i in holding}
            if deciding:
                seats = [int(self.seat[i]) for i in deciding]
                sp, sc = stack_obs(
                    [self.envs[i].obs(p) for i, p in zip(deciding, seats)], self.device)
                mk = torch.as_tensor(
                    np.stack([self.envs[i].mask(p) for i, p in zip(deciding, seats)]),
                    device=self.device)
                action, logp, value = self.net.act(sp, sc, mk)
                acts = action.cpu().numpy()
                legal_counts.append(float(mk.sum(1).float().mean()))

                for k, i in enumerate(deciding):
                    if self.filled[i] >= T:
                        # Buffer full for this env and it has reached its next
                        # decision point: freeze here and bootstrap from V(s).
                        self.last_value[i] = value[k]
                        frozen[i] = True
                        continue
                    t = int(self.filled[i])
                    self.b_spatial[t, i] = sp[k]
                    self.b_scalar[t, i] = sc[k]
                    self.b_mask[t, i] = mk[k]
                    self.b_action[t, i] = int(acts[k])
                    self.b_logp[t, i] = logp[k]
                    self.b_value[t, i] = value[k]
                    if self.slot[i] >= 0:
                        skips.append(int(self.ep_ticks[i] - self._slot_start[i]))
                    self.slot[i] = t
                    self._slot_start[i] = int(self.ep_ticks[i])
                    self.disc[i] = 1.0
                    self.filled[i] = t + 1
                    self.ep_decisions[i] += 1
                    learner_action[i] = int(acts[k])

            stepping = [i for i in active if not frozen[i]]
            if not stepping:
                break

            # 3. Opponent acts, then everyone steps one tick.
            opp_action = self._opponent_actions(stepping)
            for i in stepping:
                seat = int(self.seat[i])
                al, ao = learner_action[i], opp_action[i]
                a0, a1 = (al, ao) if seat == 0 else (ao, al)
                rew, done, info = self.envs[i].step(a0, a1)
                r = rew[seat]
                self.ep_ticks[i] += 1
                self.total_ticks += 1
                self.ep_return[i] += r
                if info["played"][seat]:
                    self.ep_plays[i] += 1

                t = int(self.slot[i])
                if t >= 0:
                    self.b_reward[t, i] += self.disc[i] * r
                    self.disc[i] *= gamma
                    self.b_gk[t, i] *= gamma

                if done:
                    if t >= 0:
                        self.b_done[t, i] = 1.0
                    score = outcome_score(info["winner"], seat)
                    rec = {"uid": self.opp[i].uid, "opp": self.opp[i].label,
                           "win_score": score,
                           "win": 1.0 if score == 1.0 else 0.0,
                           "draw": 1.0 if score == 0.5 else 0.0,
                           "len": int(self.ep_ticks[i]),
                           "ret": float(self.ep_return[i]),
                           "crowns_for": int(info["crowns"][seat]),
                           "crowns_against": int(info["crowns"][1 - seat]),
                           "decisions": int(self.ep_decisions[i]),
                           "plays": int(self.ep_plays[i])}
                    self.finished.append(rec)
                    self.league.record(self.opp[i], score)
                    self.stats.push(**rec)
                    self._reset_env(i)

        self.last_legal = float(np.mean(legal_counts)) if legal_counts else float("nan")
        self.last_skip = float(np.mean(skips)) if skips else float("nan")
        return self.last_value

    # -------------------------------------------------------------------- GAE

    def compute_gae(self, last_value, lam=0.95):
        """Semi-MDP GAE: each transition discounts by its own gamma^k."""
        T = self.horizon
        adv = torch.zeros_like(self.b_reward)
        gae = torch.zeros(self.n, device=self.device)
        for t in reversed(range(T)):
            nonterm = 1.0 - self.b_done[t]
            nxt = last_value if t == T - 1 else self.b_value[t + 1]
            gk = self.b_gk[t]
            delta = self.b_reward[t] + gk * nxt * nonterm - self.b_value[t]
            gae = delta + gk * lam * nonterm * gae
            adv[t] = gae
        return adv, adv + self.b_value

    def batch(self, last_value, lam=0.95):
        adv, ret = self.compute_gae(last_value, lam)
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        return {
            "spatial": flat(self.b_spatial), "scalar": flat(self.b_scalar),
            "mask": flat(self.b_mask), "action": flat(self.b_action),
            "logp": flat(self.b_logp), "value": flat(self.b_value),
            "adv": flat(adv), "ret": flat(ret),
        }
