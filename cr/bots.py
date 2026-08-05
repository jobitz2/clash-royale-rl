"""Scripted opponents.

These are fixed reference points. Self-play Elo only measures progress relative
to the agent's own history, so a policy can drift in circles and still look like
it is improving; beating these bots is an absolute check that it is not.
"""

from __future__ import annotations

import numpy as np

from . import arena as A
from .cards import CARDS, SPELL, BUILDING
from .env import CRDuel


class Bot:
    name = "bot"

    def reset(self):
        pass

    def act(self, env: CRDuel, p: int, mask: np.ndarray) -> int:
        raise NotImplementedError


class NoOpBot(Bot):
    name = "noop"

    def act(self, env, p, mask):
        return 0


class RandomBot(Bot):
    """Uniform over legal actions, with a bias toward holding."""

    name = "random"

    def __init__(self, act_prob=0.25, seed=0):
        self.act_prob = act_prob
        self.rng = np.random.default_rng(seed)

    def act(self, env, p, mask):
        if self.rng.random() > self.act_prob:
            return 0
        legal = np.flatnonzero(mask)
        legal = legal[legal > 0]
        return int(self.rng.choice(legal)) if legal.size else 0


class RushBot(Bot):
    """Dumps the cheapest affordable card at its own side of a bridge."""

    name = "rush"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, p, mask):
        e = env.eng
        best, bestcost = -1, 99
        for s in range(4):
            if not e.can_afford(p, s):
                continue
            cd = CARDS[e.hands[p][s]]
            if cd.kind == SPELL:
                continue
            if cd.cost < bestcost:
                best, bestcost = s, cd.cost
        if best < 0:
            return 0
        bx = A.BRIDGE_X[0] if self.rng.random() < 0.5 else A.BRIDGE_X[1]
        by = A.RIVER_Y - 2.0 if p == 0 else A.RIVER_Y + 2.0
        idx = env.encode(p, best, bx, by)
        return idx if mask[idx] else 0


class HeuristicBot(Bot):
    """A reasonable human-ish policy: defend threats, counter-push, use spells
    on clusters, and commit a tank at the bridge when elixir allows."""

    name = "heuristic"

    def __init__(self, seed=0, spell_min_targets=3, push_elixir=8.0):
        self.rng = np.random.default_rng(seed)
        self.spell_min_targets = spell_min_targets
        self.push_elixir = push_elixir

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _enemy_units(e, p):
        live = np.flatnonzero(e.alive & (e.owner != p) & ~e.is_building)
        return live

    def _own_side(self, e, p, idx):
        if idx.size == 0:
            return idx
        return idx[e.y[idx] < A.RIVER_Y] if p == 0 else idx[e.y[idx] > A.RIVER_Y]

    def _best_spell(self, e, p, foes):
        """Slot and centre for a spell hitting the most enemies, or None."""
        if foes.size == 0:
            return None
        for s in range(4):
            if not e.can_afford(p, s):
                continue
            cd = CARDS[e.hands[p][s]]
            if cd.kind != SPELL:
                continue
            # Try each enemy as the blast centre; take the best count.
            bestn, bestc = 0, None
            for j in foes:
                d = np.hypot(e.x[foes] - e.x[j], e.y[foes] - e.y[j])
                n = int((d <= cd.spell_radius).sum())
                if n > bestn:
                    bestn, bestc = n, (float(e.x[j]), float(e.y[j]))
            if bestn >= self.spell_min_targets and bestc is not None:
                return s, bestc[0], bestc[1]
        return None

    def _cheapest_defender(self, e, p, need_air):
        best, bestcost = -1, 99
        for s in range(4):
            if not e.can_afford(p, s):
                continue
            cd = CARDS[e.hands[p][s]]
            if cd.kind == SPELL or cd.buildings_only:
                continue
            if need_air and not cd.hits_air:
                continue
            if cd.cost < bestcost:
                best, bestcost = s, cd.cost
        return best

    def _biggest_threat(self, e, p):
        best, bestcost = -1, -1
        for s in range(4):
            if not e.can_afford(p, s):
                continue
            cd = CARDS[e.hands[p][s]]
            if cd.kind == SPELL or cd.kind == BUILDING:
                continue
            if cd.cost > bestcost:
                best, bestcost = s, cd.cost
        return best

    # -- policy ----------------------------------------------------------
    def act(self, env, p, mask):
        e = env.eng
        foes = self._enemy_units(e, p)
        threats = self._own_side(e, p, foes)

        # 1. Spell a cluster if it is worth the elixir.
        sp = self._best_spell(e, p, foes)
        if sp is not None:
            idx = env.encode(p, sp[0], sp[1], sp[2])
            if mask[idx]:
                return idx

        # 2. Defend: drop the cheapest suitable troop between the threat and
        #    the tower it is walking toward.
        if threats.size:
            j = int(threats[np.argmax(e.hp[threats] * (1 + e.dmg[threats]))])
            need_air = bool(e.is_air[j])
            s = self._cheapest_defender(e, p, need_air)
            if s >= 0:
                dy = -1.8 if p == 0 else 1.8
                idx = env.encode(p, s, float(e.x[j]), float(e.y[j]) + dy)
                if mask[idx]:
                    return idx

        # 3. Push when elixir is plentiful.
        if e.elixir[p] >= self.push_elixir:
            s = self._biggest_threat(e, p)
            if s >= 0:
                bx = A.BRIDGE_X[0] if self.rng.random() < 0.5 else A.BRIDGE_X[1]
                by = A.RIVER_Y - 2.5 if p == 0 else A.RIVER_Y + 2.5
                idx = env.encode(p, s, bx, by)
                if mask[idx]:
                    return idx
        return 0


def make_bot(name: str, seed: int = 0) -> Bot:
    table = {
        "noop": lambda: NoOpBot(),
        "random": lambda: RandomBot(seed=seed),
        "rush": lambda: RushBot(seed=seed),
        "heuristic": lambda: HeuristicBot(seed=seed),
    }
    if name not in table:
        raise KeyError(f"unknown bot {name!r}; have {sorted(table)}")
    return table[name]()


BOT_NAMES = ("noop", "random", "rush", "heuristic")
