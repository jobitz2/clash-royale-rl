#!/usr/bin/env python
"""Train a Clash Royale agent by league self-play.

    python train.py --iters 2000 --n-envs 32

Each iteration: collect a rollout of league self-play, run PPO on it, and every
`--snapshot-every` iterations freeze a copy of the policy into the opponent pool.
Progress is reported two ways -- Elo against the league (relative) and win rate
against the scripted bots (absolute, so plateaus and cycling are visible).

Runs are resumable: `--resume runs/<name>/latest.pt` restores network, optimiser
and league state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from cr.cards import DEFAULT_DECK, avg_cost, deck_ids
from rl.evaluate import evaluate_vs_bots
from rl.league import League
from rl.net import ActorCritic
from rl.ppo import PPO, PPOConfig
from rl.rollout import SelfPlayCollector

LOG_FIELDS = [
    "iter", "steps", "ticks", "time", "sps", "learner_elo", "winrate", "drawrate",
    "ep_len", "ep_return", "crowns_for", "crowns_against", "decisions", "plays",
    "legal_actions", "skip_ticks", "pg_loss", "v_loss",
    "entropy", "kl", "clipfrac", "explained_var", "ent_coef", "lr",
    "pool", "eval_noop", "eval_random", "eval_rush", "eval_heuristic",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--frame-skip", type=int, default=3,
                   help="physics ticks per decision (3 = one action per 0.3s)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                   help="device for the PPO update; rollouts always run on cpu")
    p.add_argument("--workers", type=int, default=0,
                   help="rollout worker processes (0 = collect in-process)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true")
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.012)
    p.add_argument("--ent-coef-final", type=float, default=0.002)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--max-snapshots", type=int, default=20)
    p.add_argument("--p-bot", type=float, default=0.15,
                   help="fraction of episodes played against scripted bots")
    p.add_argument("--p-latest", type=float, default=0.35,
                   help="fraction played against the newest snapshot")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-games", type=int, default=12)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--name", default="cr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def make_run_dir(args) -> Path:
    if args.run_dir:
        d = Path(args.run_dir)
    else:
        d = Path("runs") / f"{args.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(path, net, ppo, league, it, steps, args):
    torch.save({
        "net": net.state_dict(),
        "opt": ppo.opt.state_dict(),
        "league": league.state(),
        "iter": it,
        "steps": steps,
        "args": vars(args),
    }, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    run = make_run_dir(args)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    log_path = run / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
    if new_log:
        log.writeheader()

    device = args.device
    net = ActorCritic().to(device)
    cfg = PPOConfig(lr=args.lr, gamma=args.gamma, lam=args.lam,
                    ent_coef=args.ent_coef, ent_coef_final=args.ent_coef_final,
                    minibatches=args.minibatches, epochs=args.epochs)
    ppo = PPO(net, cfg)
    league = League(max_snapshots=args.max_snapshots, p_bot=args.p_bot,
                    p_latest=args.p_latest, seed=args.seed)

    start_iter, total_steps = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        ppo.opt.load_state_dict(ck["opt"])
        league.load_state(ck["league"])
        start_iter, total_steps = ck["iter"] + 1, ck["steps"]
        print(f"resumed {args.resume} at iter {start_iter} ({total_steps} steps)")

    if args.workers > 0:
        from rl.parallel import ParallelCollector
        collector = ParallelCollector(
            net, league, n_envs=args.n_envs, horizon=args.horizon,
            workers=args.workers, device=device, frame_skip=args.frame_skip,
            seed=args.seed + 1, gamma=args.gamma,
            max_snapshots=args.max_snapshots, p_bot=args.p_bot,
            p_latest=args.p_latest)
    else:
        collector = SelfPlayCollector(
            net, league, n_envs=args.n_envs, horizon=args.horizon, device=device,
            frame_skip=args.frame_skip, seed=args.seed + 1, gamma=args.gamma)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"run dir      {run}")
    print(f"device       {device}   params {n_params:,}")
    print(f"deck         {', '.join(DEFAULT_DECK)}  (avg cost {avg_cost():.2f})")
    print(f"batch        {args.n_envs} envs x {args.horizon} steps = "
          f"{args.n_envs * args.horizon} transitions/iter")
    print()

    stop = {"now": False}

    def on_sigint(*_):
        if stop["now"]:
            raise KeyboardInterrupt
        stop["now"] = True
        print("\n[stopping after this iteration; press Ctrl-C again to abort]")

    signal.signal(signal.SIGINT, on_sigint)

    evals = {}
    t_start = time.time()
    for it in range(start_iter, args.iters):
        frac = it / max(1, args.iters - 1)
        ent = args.ent_coef + frac * (args.ent_coef_final - args.ent_coef)
        if args.anneal_lr:
            ppo.set_lr(args.lr * (1.0 - frac))

        t0 = time.time()
        last_value = collector.collect()
        batch = collector.batch(last_value, lam=args.lam)
        info = ppo.update(batch, ent_coef=ent)
        total_steps += args.n_envs * args.horizon
        dt = time.time() - t0

        if it % args.snapshot_every == 0 and it > start_iter:
            league.add_snapshot(net.state_dict(), it)
            collector.prune_opponent_cache()

        # Only rows where an evaluation actually ran carry eval columns. Copying
        # the last known values into every row makes the log look like it holds
        # far more measurements than it does.
        evals_now = {}
        if args.eval_every and (it % args.eval_every == 0 or it == args.iters - 1):
            evals_now = evaluate_vs_bots(net, games=args.eval_games, device=device,
                                         seed=args.seed + it,
                                         frame_skip=args.frame_skip)
            evals = evals_now

        st = collector.stats.summary()
        row = {
            "iter": it, "steps": total_steps, "ticks": collector.total_ticks,
            "time": round(time.time() - t_start, 1),
            "sps": round(args.n_envs * args.horizon / dt, 1),
            "learner_elo": round(league.learner_elo, 1),
            "pool": len(league.snapshots()),
            "lr": ppo.opt.param_groups[0]["lr"],
            "legal_actions": round(collector.last_legal, 2),
            "skip_ticks": round(collector.last_skip, 2),
        }
        row.update({k: (round(v, 4) if v == v else "") for k, v in st.items()})
        row.update({k: round(v, 4) for k, v in info.items() if k in LOG_FIELDS})
        row.update({f"eval_{k}": round(v, 3) for k, v in evals_now.items()})
        log.writerow({k: row.get(k, "") for k in LOG_FIELDS})
        log_f.flush()

        if it % 5 == 0 or it == args.iters - 1:
            ev = "  ".join(f"{k[:4]}={v:.2f}" for k, v in evals.items())
            print(f"it {it:5d} | dec {total_steps/1e3:6.1f}k | {row['sps']:6.0f} dps "
                  f"| elo {league.learner_elo:7.1f} | wr {st['winrate']:.2f} "
                  f"| plays {st['plays']:5.1f} | ret {st['ep_return']:6.2f} "
                  f"| ent {info['entropy']:.2f} | legal {collector.last_legal:5.0f} "
                  f"| ev {info['explained_var']:5.2f} "
                  f"| pool {len(league.snapshots()):2d}" + (f" | {ev}" if ev else ""))

        if it % args.save_every == 0 or it == args.iters - 1 or stop["now"]:
            save(run / "latest.pt", net, ppo, league, it, total_steps, args)

        if stop["now"]:
            print(f"stopped at iter {it}")
            break

    print("\nleague standings")
    print(league.table())
    print(f"\nsaved {run/'latest.pt'}")
    log_f.close()
    if hasattr(collector, "close"):
        collector.close()


if __name__ == "__main__":
    main()
