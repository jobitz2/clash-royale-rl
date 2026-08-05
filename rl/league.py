"""Opponent league with Elo bookkeeping and PFSP sampling.

Plain self-play against only the newest copy of yourself is unstable: policies
chase each other in cycles and can forget how to beat older styles. The league
keeps a pool of frozen historical snapshots plus the scripted bots, and samples
opponents with Prioritised Fictitious Self-Play -- weighting toward opponents
the learner is currently *losing* to, which is where the gradient is.

The `random` bot is Elo-anchored at 0 so ratings have an absolute scale and
"is it actually improving" is answerable, not just "is it beating itself".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from cr.bots import BOT_NAMES

K_LEARNER = 16.0
K_OPPONENT = 8.0
ANCHOR_BOT = "random"
ANCHOR_ELO = 0.0


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


@dataclass
class Entry:
    uid: int
    kind: str                  # "bot" or "snapshot"
    name: str
    elo: float = 0.0
    games: int = 0
    wins: float = 0.0
    anchored: bool = False
    iteration: int = -1
    permanent: bool = False    # never evicted
    params: dict | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.name if self.kind == "bot" else f"snap@{self.iteration}"


class League:
    def __init__(self, bots=BOT_NAMES, max_snapshots: int = 20,
                 p_bot: float = 0.15, p_latest: float = 0.35, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.max_snapshots = max_snapshots
        self.p_bot = p_bot
        self.p_latest = p_latest
        self.entries: list[Entry] = []
        self.learner_elo = 0.0
        self._next_uid = 0
        # Bumped whenever the pool's membership changes, so parallel workers
        # know when their clone of the league is stale.
        self.version = 0
        for b in bots:
            anchored = b == ANCHOR_BOT
            self.add_bot(b, elo=ANCHOR_ELO if anchored else 0.0, anchored=anchored)

    # ------------------------------------------------------------------ pool

    def _uid(self) -> int:
        self._next_uid += 1
        return self._next_uid - 1

    def add_bot(self, name, elo=0.0, anchored=False) -> Entry:
        e = Entry(self._uid(), "bot", name, elo=elo, anchored=anchored,
                  permanent=True)
        self.entries.append(e)
        self.version += 1
        return e

    def add_snapshot(self, params: dict, iteration: int, elo: float | None = None) -> Entry:
        e = Entry(self._uid(), "snapshot", f"snap{iteration}",
                  elo=self.learner_elo if elo is None else elo,
                  iteration=iteration,
                  permanent=not self.snapshots(),      # first snapshot anchors history
                  params=copy.deepcopy({k: v.detach().cpu().clone()
                                        for k, v in params.items()}))
        self.entries.append(e)
        self._evict()
        self.version += 1
        return e

    def snapshots(self) -> list[Entry]:
        return [e for e in self.entries if e.kind == "snapshot"]

    def bots(self) -> list[Entry]:
        return [e for e in self.entries if e.kind == "bot"]

    def _evict(self):
        snaps = [e for e in self.snapshots() if not e.permanent]
        while len(self.snapshots()) > self.max_snapshots and snaps:
            oldest = min(snaps, key=lambda e: e.iteration)
            self.entries.remove(oldest)
            snaps.remove(oldest)

    def latest(self) -> Entry | None:
        s = self.snapshots()
        return max(s, key=lambda e: e.iteration) if s else None

    # -------------------------------------------------------------- sampling

    def sample(self) -> Entry:
        """PFSP-weighted opponent draw, mixed with bot anchors and self-latest."""
        snaps = self.snapshots()
        r = self.rng.random()
        if not snaps or r < self.p_bot:
            bots = self.bots()
            return bots[self.rng.integers(len(bots))]
        if r < self.p_bot + self.p_latest:
            return self.latest()
        # (1 - p)^2 puts the mass on opponents we are losing to but not hopeless.
        w = np.array([max(1e-3, (1.0 - expected_score(self.learner_elo, e.elo)) ** 2)
                      for e in snaps], dtype=np.float64)
        w /= w.sum()
        return snaps[int(self.rng.choice(len(snaps), p=w))]

    # ---------------------------------------------------------------- ratings

    def record(self, entry: Entry, learner_score: float):
        """`learner_score` is 1 win / 0.5 draw / 0 loss from the learner's view."""
        exp = expected_score(self.learner_elo, entry.elo)
        self.learner_elo += K_LEARNER * (learner_score - exp)
        if not entry.anchored:
            entry.elo += K_OPPONENT * ((1.0 - learner_score) - (1.0 - exp))
        entry.games += 1
        entry.wins += 1.0 - learner_score

    # ------------------------------------------------------------------- misc

    def table(self) -> str:
        rows = sorted(self.entries, key=lambda e: -e.elo)
        out = [f"  learner        elo {self.learner_elo:8.1f}"]
        for e in rows:
            wr = (e.wins / e.games) if e.games else float("nan")
            tag = " (anchor)" if e.anchored else ""
            out.append(f"  {e.label:<14s} elo {e.elo:8.1f}  games {e.games:5d}"
                       f"  winrate_vs_learner {wr:5.2f}{tag}")
        return "\n".join(out)

    def state(self) -> dict:
        """Serialisable league state (snapshot params included)."""
        return {
            "learner_elo": self.learner_elo,
            "next_uid": self._next_uid,
            "version": self.version,
            "entries": [dict(vars(e)) for e in self.entries],
        }

    def load_state(self, st: dict):
        self.learner_elo = st["learner_elo"]
        self._next_uid = st["next_uid"]
        self.version = st.get("version", 0)
        self.entries = [Entry(**d) for d in st["entries"]]

    def elo_table(self) -> dict:
        """Compact rating snapshot for broadcasting to workers."""
        t = {e.uid: e.elo for e in self.entries}
        t["learner"] = self.learner_elo
        return t

    def apply_elo_table(self, t: dict):
        self.learner_elo = t["learner"]
        for e in self.entries:
            if e.uid in t:
                e.elo = t[e.uid]
