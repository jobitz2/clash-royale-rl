"""Two-player RL environment wrapping the battle engine.

Both players observe the board in a *canonical* frame in which they always push
toward +y, so a single network can play either side. Player 1's view is the
board rotated 180 degrees.

The action space is a pointer over the map: `1 + 4 * PH * PW` discrete actions
(index 0 is "hold"), i.e. pick one of the 4 hand slots and one placement cell.
Illegal combinations are masked out before the softmax rather than punished.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import arena as A
from .cards import CARDS, NUM_CARDS, SPELL
from .engine import Engine, NOT_TOWER

PLACE_STRIDE = 2
PH = A.H // PLACE_STRIDE          # 16
PW = A.W // PLACE_STRIDE          # 9
N_PLACE = PH * PW                 # 144
N_ACTIONS = 1 + 4 * N_PLACE       # 577

N_CHANNELS = 11
HP_SCALE = 2000.0
DPS_SCALE = 300.0
SCALAR_DIM = 2 + 1 + 1 + 1 + 6 + 4 * (NUM_CARDS + 2) + NUM_CARDS + 2


@dataclass
class RewardConfig:
    w_shape: float = 2.0       # potential-based tower-health shaping
    w_win: float = 2.0         # terminal outcome
    w_crown: float = 0.25      # bonus per crown swing
    w_leak: float = 0.02       # penalty per elixir wasted at cap
    w_step: float = 0.0        # per-decision cost (0 = off)


class CRDuel:
    """A single battle exposing symmetric observations/actions for both seats."""

    def __init__(self, decks=None, frame_skip: int = 3, seed: int | None = None,
                 reward: RewardConfig | None = None, perfect_info: bool = True,
                 max_steps: int = 1100):
        self.eng = Engine(decks=decks, seed=seed)
        self.frame_skip = frame_skip
        self.reward = reward or RewardConfig()
        self.perfect_info = perfect_info
        self.max_steps = max_steps
        self.steps = 0
        self._phi = [0.0, 0.0]
        self._leak = [0.0, 0.0]
        self._cache = {}

    # ------------------------------------------------------------------- frame

    @staticmethod
    def _to_canon(p, x, y):
        return (A.W - x, A.H - y) if p == 1 else (x, y)

    @staticmethod
    def _from_canon(p, x, y):
        return (A.W - x, A.H - y) if p == 1 else (x, y)

    # ------------------------------------------------------------------ observe

    def _spatial(self, p) -> np.ndarray:
        e = self.eng
        m = np.zeros((N_CHANNELS, A.H, A.W), dtype=np.float32)
        live = np.flatnonzero(e.alive)
        if live.size:
            cx, cy = self._to_canon(p, e.x[live], e.y[live])
            ix = np.clip(cx.astype(np.int32), 0, A.W - 1)
            iy = np.clip(cy.astype(np.int32), 0, A.H - 1)
            mine = e.owner[live] == p
            bld = e.is_building[live]
            air = e.is_air[live]
            hpn = np.minimum(e.hp[live] / HP_SCALE, 4.0)
            dpsn = np.minimum(
                np.divide(e.dmg[live], np.maximum(e.hit_interval[live], 1e-6)) / DPS_SCALE,
                4.0)

            def acc(ch, sel, vals):
                if sel.any():
                    np.add.at(m[ch], (iy[sel], ix[sel]), vals[sel])

            troop = ~bld
            acc(0, mine & troop, hpn)
            acc(1, mine & troop, dpsn)
            acc(2, mine & troop & air, np.ones_like(hpn))
            acc(3, ~mine & troop, hpn)
            acc(4, ~mine & troop, dpsn)
            acc(5, ~mine & troop & air, np.ones_like(hpn))
            acc(6, mine & bld, hpn)
            acc(7, ~mine & bld, hpn)
            acc(9, mine & troop, np.ones_like(hpn))
            acc(10, ~mine & troop, np.ones_like(hpn))

        dep = e.deploy_mask(p).astype(np.float32)
        m[8] = dep[::-1, ::-1] if p == 1 else dep
        return m

    def _scalar(self, p) -> np.ndarray:
        e = self.eng
        o, q = p, 1 - p
        v = [e.elixir[o] / A.ELIXIR_MAX,
             (e.elixir[q] / A.ELIXIR_MAX) if self.perfect_info else 0.0,
             A.elixir_rate(e.t) * A.ELIXIR_PERIOD / 3.0,
             e.time_left() / A.T_REGULAR,
             1.0 if e.sudden_death else 0.0]
        v += e.tower_hp_frac(o) + e.tower_hp_frac(q)
        for s in range(4):
            oh = np.zeros(NUM_CARDS, dtype=np.float32)
            cid = e.hands[o][s]
            oh[cid] = 1.0
            v += list(oh)
            v.append(CARDS[cid].cost / 10.0)
            v.append(1.0 if e.can_afford(o, s) else 0.0)
        nxt = np.zeros(NUM_CARDS, dtype=np.float32)
        n = e.next_in_cycle(o)
        if n >= 0:
            nxt[n] = 1.0
        v += list(nxt)
        v += [e.crowns[o] / 3.0, e.crowns[q] / 3.0]
        return np.asarray(v, dtype=np.float32)

    def _mask(self, p) -> np.ndarray:
        legal = self.eng.legal_mask(p)                    # [4, H, W] engine frame
        if p == 1:
            legal = legal[:, ::-1, ::-1]
        # any() over each stride x stride placement cell
        blk = legal.reshape(4, PH, PLACE_STRIDE, PW, PLACE_STRIDE).any(axis=(2, 4))
        out = np.empty(N_ACTIONS, dtype=bool)
        out[0] = True                                     # holding is always legal
        out[1:] = blk.reshape(-1)
        return out

    # Observations and masks are built lazily and memoised per tick: encoding
    # them for both seats on every tick costs more than the physics does, and
    # at forced no-op ticks nobody looks at them at all.
    def obs(self, p):
        key = ("obs", p)
        if key not in self._cache:
            self._cache[key] = {"spatial": self._spatial(p),
                                "scalar": self._scalar(p)}
        return self._cache[key]

    def mask(self, p):
        key = ("mask", p)
        if key not in self._cache:
            self._cache[key] = self._mask(p)
        return self._cache[key]

    def can_act(self, p) -> bool:
        """True if `p` has any real choice beyond holding.

        Affording a card implies a legal placement exists: a player's own half is
        never fully blocked and spells may target any tile. So affordability
        alone decides whether this tick is a decision point.
        """
        return any(self.eng.can_afford(p, s) for s in range(4))

    def observations(self):
        return (self.obs(0), self.obs(1)), (self.mask(0), self.mask(1))

    # ------------------------------------------------------------------ acting

    @staticmethod
    def decode(p, idx):
        """Action index -> (slot, engine-frame x, y). slot=-1 means hold."""
        if idx <= 0:
            return -1, 0.0, 0.0
        k = idx - 1
        slot, rem = divmod(k, N_PLACE)
        py, px = divmod(rem, PW)
        cx = (px + 0.5) * PLACE_STRIDE
        cy = (py + 0.5) * PLACE_STRIDE
        x, y = CRDuel._from_canon(p, cx, cy)
        return slot, float(x), float(y)

    @staticmethod
    def encode(p, slot, x, y) -> int:
        """(slot, engine-frame x, y) -> action index. Inverse of `decode`."""
        if slot < 0:
            return 0
        cx, cy = CRDuel._to_canon(p, x, y)
        px = int(np.clip(cx // PLACE_STRIDE, 0, PW - 1))
        py = int(np.clip(cy // PLACE_STRIDE, 0, PH - 1))
        return 1 + slot * N_PLACE + py * PW + px

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.eng.rng = np.random.default_rng(seed)
        self.eng.reset()
        self.steps = 0
        self._cache = {}
        for p in (0, 1):
            self._phi[p] = self._potential(p)
            self._leak[p] = 0.0
        return self

    def _potential(self, p) -> float:
        return self.eng.total_tower_frac(p) - self.eng.total_tower_frac(1 - p)

    def step(self, a0, a1):
        e = self.eng
        rc = self.reward
        crowns_before = list(e.crowns)
        self._cache = {}

        # Simultaneous resolution: randomise which seat is applied first so
        # neither systematically wins a contested tile or the last elixir.
        order = (0, 1) if e.rng.random() < 0.5 else (1, 0)
        acts = {0: a0, 1: a1}
        played = {0: False, 1: False}
        for p in order:
            slot, x, y = self.decode(p, int(acts[p]))
            if slot >= 0:
                played[p] = e.play(p, slot, x, y)

        for _ in range(self.frame_skip):
            e.step(A.DT)
            if e.done:
                break
        self.steps += 1

        rew = [0.0, 0.0]
        for p in (0, 1):
            phi = self._potential(p)
            rew[p] += rc.w_shape * (phi - self._phi[p])
            self._phi[p] = phi
            dc = (e.crowns[p] - crowns_before[p]) - (e.crowns[1 - p] - crowns_before[1 - p])
            rew[p] += rc.w_crown * dc
            leaked = e.elixir_leaked[p] - self._leak[p]
            self._leak[p] = e.elixir_leaked[p]
            rew[p] -= rc.w_leak * leaked
            rew[p] -= rc.w_step

        done = e.done or self.steps >= self.max_steps
        if done:
            w = e.winner
            for p in (0, 1):
                if w is None:
                    pass
                elif w == p:
                    rew[p] += rc.w_win
                else:
                    rew[p] -= rc.w_win

        info = {"winner": e.winner, "t": e.t, "crowns": list(e.crowns),
                "played": played, "timeout": self.steps >= self.max_steps}
        return rew, done, info


def outcome_score(winner, p) -> float:
    """1.0 win / 0.5 draw / 0.0 loss, for Elo bookkeeping."""
    if winner is None:
        return 0.5
    return 1.0 if winner == p else 0.0
