#!/usr/bin/env python
"""Score checkpoints against the scripted bots and against each other.

    python evaluate.py --ckpt runs/cr-.../latest.pt --games 40
    python evaluate.py --ckpt a.pt --vs b.pt --games 40
    python evaluate.py --bots-only --games 30        # baseline round-robin
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from cr.bots import BOT_NAMES, make_bot
from rl.evaluate import head_to_head, NetPolicy
from rl.league import expected_score
from rl.net import ActorCritic


def load_net(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic().to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck


def elo_gap(winrate: float) -> float:
    """Elo difference implied by a win rate (clamped away from 0 and 1)."""
    p = min(max(winrate, 1e-3), 1 - 1e-3)
    return -400.0 * np.log10(1.0 / p - 1.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--vs", default=None, help="second checkpoint or bot name")
    p.add_argument("--bots-only", action="store_true")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.bots_only:
        print(f"scripted-bot round robin, {args.games} games each "
              f"(row win rate vs column)\n")
        print(f"{'':12s}" + "".join(f"{n:>12s}" for n in BOT_NAMES))
        for a in BOT_NAMES:
            row = f"{a:12s}"
            for b in BOT_NAMES:
                if a == b:
                    row += f"{'---':>12s}"
                else:
                    wr, _ = head_to_head(make_bot(a, seed=1), make_bot(b, seed=2),
                                         games=args.games, seed=args.seed,
                                         frame_skip=args.frame_skip)
                    row += f"{wr:>12.2f}"
            print(row)
        return

    if not args.ckpt:
        raise SystemExit("need --ckpt (or --bots-only)")

    net, ck = load_net(args.ckpt, args.device)
    pol = NetPolicy(net, device=args.device, deterministic=args.deterministic)
    print(f"checkpoint {args.ckpt}")
    print(f"  trained iters {ck.get('iter', '?')}  steps {ck.get('steps', '?'):,}"
          f"  league elo {ck['league']['learner_elo']:.1f}\n")

    if args.vs:
        if args.vs in BOT_NAMES:
            opp, label = make_bot(args.vs, seed=3), args.vs
        else:
            onet, ock = load_net(args.vs, args.device)
            opp = NetPolicy(onet, device=args.device, deterministic=args.deterministic)
            label = f"{args.vs} (iter {ock.get('iter','?')})"
        wr, scores = head_to_head(pol, opp, games=args.games, seed=args.seed,
                                  frame_skip=args.frame_skip)
        se = float(np.std(scores) / max(1, np.sqrt(len(scores))))
        print(f"  vs {label}: win rate {wr:.3f} +/- {se:.3f}  "
              f"(implied elo gap {elo_gap(wr):+.0f})")
        return

    print(f"  {args.games} games vs each scripted bot (seats alternated)\n")
    for name in BOT_NAMES:
        wr, scores = head_to_head(pol, make_bot(name, seed=3), games=args.games,
                                  seed=args.seed + hash(name) % 9973,
                                  frame_skip=args.frame_skip)
        se = float(np.std(scores) / max(1, np.sqrt(len(scores))))
        print(f"  vs {name:<10s} win rate {wr:.3f} +/- {se:.3f}"
              f"   implied elo gap {elo_gap(wr):+7.0f}")


if __name__ == "__main__":
    main()
