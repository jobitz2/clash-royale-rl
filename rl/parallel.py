"""Multiprocess rollout collection.

Simulation is pure Python/numpy and dominates wall-clock (~75% of an iteration),
so it is the thing worth parallelising. Each worker owns a shard of the
environments *and* a clone of the league, and runs the ordinary
`SelfPlayCollector` locally -- so the single-process path stays the reference
implementation and there is only one copy of the collection logic.

Per iteration the main process ships the learner's parameters plus a compact Elo
table (a few hundred KB) and receives back filled rollout buffers. The full
league -- including frozen snapshot weights -- is only re-broadcast when the pool
membership actually changes, which is every `--snapshot-every` iterations.

Elo is authoritative in the main process: workers report match outcomes and the
main process applies them, then broadcasts the updated table.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch

from cr.env import N_ACTIONS, N_CHANNELS, SCALAR_DIM
from cr import arena as A

from .league import League
from .net import ActorCritic
from .rollout import EpisodeStats, SelfPlayCollector

BUFFER_KEYS = ("spatial", "scalar", "mask", "action", "logp", "value",
               "reward", "done", "gk")


def _worker(conn, cfg):
    """Owns `cfg['n_envs']` environments; collects on demand."""
    torch.set_num_threads(1)
    net = ActorCritic()
    net.eval()
    league = League(max_snapshots=cfg["max_snapshots"], p_bot=cfg["p_bot"],
                    p_latest=cfg["p_latest"], seed=cfg["seed"])
    col = SelfPlayCollector(net, league, n_envs=cfg["n_envs"],
                            horizon=cfg["horizon"], device="cpu",
                            frame_skip=cfg["frame_skip"], seed=cfg["seed"],
                            decks=cfg["decks"], reward=cfg["reward"],
                            gamma=cfg["gamma"])
    try:
        while True:
            msg = conn.recv()
            cmd = msg["cmd"]
            if cmd == "close":
                break
            if cmd != "collect":
                continue
            net.load_state_dict(msg["params"])
            if msg["league"] is not None:
                league.load_state(msg["league"])
                col.prune_opponent_cache()
            league.apply_elo_table(msg["elo"])

            col.finished = []
            last_value = col.collect()
            conn.send({
                "buffers": {
                    "spatial": col.b_spatial.numpy(),
                    "scalar": col.b_scalar.numpy(),
                    "mask": col.b_mask.numpy(),
                    "action": col.b_action.numpy(),
                    "logp": col.b_logp.numpy(),
                    "value": col.b_value.numpy(),
                    "reward": col.b_reward.numpy(),
                    "done": col.b_done.numpy(),
                    "gk": col.b_gk.numpy(),
                },
                "last_value": last_value.numpy(),
                "finished": col.finished,
                "legal": col.last_legal,
                "skip": col.last_skip,
                "ticks": col.total_ticks,
            })
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        conn.close()


class ParallelCollector:
    """Drop-in replacement for `SelfPlayCollector` backed by worker processes."""

    def __init__(self, net: ActorCritic, league: League, n_envs=32, horizon=128,
                 workers=4, device="cpu", frame_skip=3, seed=0, decks=None,
                 reward=None, gamma=0.997, max_snapshots=20, p_bot=0.15,
                 p_latest=0.35):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.net = net
        self.league = league
        self.device = device
        self.horizon = horizon
        self.gamma = gamma
        self.stats = EpisodeStats()
        self.total_ticks = 0
        self.last_legal = float("nan")
        self.last_skip = float("nan")

        # Split environments as evenly as possible across workers.
        base, extra = divmod(n_envs, workers)
        self.shards = [base + (1 if w < extra else 0) for w in range(workers)]
        self.shards = [s for s in self.shards if s > 0]
        self.n = sum(self.shards)

        ctx = mp.get_context("spawn")
        self.procs, self.conns = [], []
        for w, shard in enumerate(self.shards):
            parent, child = ctx.Pipe()
            cfg = {"n_envs": shard, "horizon": horizon, "frame_skip": frame_skip,
                   "seed": seed + 1000 * (w + 1), "decks": decks, "reward": reward,
                   "gamma": gamma, "max_snapshots": max_snapshots,
                   "p_bot": p_bot, "p_latest": p_latest}
            p = ctx.Process(target=_worker, args=(child, cfg), daemon=True)
            p.start()
            child.close()
            self.procs.append(p)
            self.conns.append(parent)

        self._sent_version = -1
        self._alloc()

    def _alloc(self):
        T, N, d = self.horizon, self.n, self.device
        self.b_spatial = torch.zeros((T, N, N_CHANNELS, A.H, A.W), device=d)
        self.b_scalar = torch.zeros((T, N, SCALAR_DIM), device=d)
        self.b_mask = torch.zeros((T, N, N_ACTIONS), dtype=torch.bool, device=d)
        self.b_action = torch.zeros((T, N), dtype=torch.long, device=d)
        self.b_logp = torch.zeros((T, N), device=d)
        self.b_value = torch.zeros((T, N), device=d)
        self.b_reward = torch.zeros((T, N), device=d)
        self.b_done = torch.zeros((T, N), device=d)
        self.b_gk = torch.ones((T, N), device=d)
        self.last_value = torch.zeros(N, device=d)

    # ------------------------------------------------------------------ collect

    def prune_opponent_cache(self):
        """No-op: workers prune their own caches on league broadcast."""

    def collect(self):
        params = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        send_league = self.league.version != self._sent_version
        league_state = self.league.state() if send_league else None
        elo = self.league.elo_table()

        for conn in self.conns:
            conn.send({"cmd": "collect", "params": params,
                       "league": league_state, "elo": elo})
        self._sent_version = self.league.version

        replies = [conn.recv() for conn in self.conns]

        by_uid = {e.uid: e for e in self.league.entries}
        legal, skip, off = [], [], 0
        for shard, rep in zip(self.shards, replies):
            sl = slice(off, off + shard)
            b = rep["buffers"]
            self.b_spatial[:, sl] = torch.as_tensor(b["spatial"], device=self.device)
            self.b_scalar[:, sl] = torch.as_tensor(b["scalar"], device=self.device)
            self.b_mask[:, sl] = torch.as_tensor(b["mask"], device=self.device)
            self.b_action[:, sl] = torch.as_tensor(b["action"], device=self.device)
            self.b_logp[:, sl] = torch.as_tensor(b["logp"], device=self.device)
            self.b_value[:, sl] = torch.as_tensor(b["value"], device=self.device)
            self.b_reward[:, sl] = torch.as_tensor(b["reward"], device=self.device)
            self.b_done[:, sl] = torch.as_tensor(b["done"], device=self.device)
            self.b_gk[:, sl] = torch.as_tensor(b["gk"], device=self.device)
            self.last_value[sl] = torch.as_tensor(rep["last_value"],
                                                  device=self.device)
            off += shard

            for rec in rep["finished"]:
                entry = by_uid.get(rec["uid"])
                if entry is not None:
                    self.league.record(entry, rec["win_score"])
                self.stats.push(**rec)
            if rep["legal"] == rep["legal"]:
                legal.append(rep["legal"])
            if rep["skip"] == rep["skip"]:
                skip.append(rep["skip"])

        self.total_ticks = sum(rep["ticks"] for rep in replies)
        self.last_legal = float(np.mean(legal)) if legal else float("nan")
        self.last_skip = float(np.mean(skip)) if skip else float("nan")
        return self.last_value

    # Reuse the single-process GAE and flattening.
    compute_gae = SelfPlayCollector.compute_gae
    batch = SelfPlayCollector.batch

    # -------------------------------------------------------------------- close

    def close(self):
        for conn in self.conns:
            try:
                conn.send({"cmd": "close"})
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
