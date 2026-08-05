#!/usr/bin/env python
"""Watch a battle render in the terminal.

    python watch.py --a runs/cr-.../latest.pt --b heuristic
    python watch.py --a heuristic --b random --fps 20
    python watch.py --a runs/x/latest.pt --b runs/y/latest.pt --games 3

`--a`/`--b` each take either a scripted bot name or a checkpoint path.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from cr import arena as A
from cr.bots import BOT_NAMES, make_bot
from cr.env import CRDuel, outcome_score
from cr.render import frame, hand_line
from rl.evaluate import NetPolicy
from rl.net import ActorCritic

CLEAR = "\033[H\033[J"


def load_policy(spec: str, device="cpu", deterministic=False):
    if spec in BOT_NAMES:
        return make_bot(spec, seed=0), spec
    ck = torch.load(spec, map_location=device, weights_only=False)
    net = ActorCritic().to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    label = f"{spec} (iter {ck.get('iter', '?')})"
    return NetPolicy(net, device=device, deterministic=deterministic), label


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="heuristic", help="bot name or checkpoint path")
    p.add_argument("--b", default="random", help="bot name or checkpoint path")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--fps", type=float, default=12.0, help="0 = no delay")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-render", action="store_true", help="scores only")
    return p.parse_args()


def main():
    args = parse_args()
    pol_a, label_a = load_policy(args.a, args.device, args.deterministic)
    pol_b, label_b = load_policy(args.b, args.device, args.deterministic)
    delay = 1.0 / args.fps if args.fps > 0 else 0.0

    total = 0.0
    for g in range(args.games):
        seat_a = g % 2                     # alternate seats between games
        seat_b = 1 - seat_a
        env = CRDuel(frame_skip=args.frame_skip, seed=args.seed + g)
        env.reset(seed=args.seed + g)
        pol_a.reset()
        pol_b.reset()
        done = False
        while not done:
            aa = pol_a.act(env, seat_a, env.mask(seat_a)) if env.can_act(seat_a) else 0
            ab = pol_b.act(env, seat_b, env.mask(seat_b)) if env.can_act(seat_b) else 0
            a0, a1 = (aa, ab) if seat_a == 0 else (ab, aa)
            rew, done, info = env.step(a0, a1)
            if not args.no_render:
                sys.stdout.write(CLEAR)
                print(f"game {g+1}/{args.games}   "
                      f"A={label_a} (seat {seat_a})   B={label_b} (seat {seat_b})")
                print(frame(env.eng))
                print(hand_line(env.eng, 1))
                print(hand_line(env.eng, 0))
                sys.stdout.flush()
                if delay:
                    time.sleep(delay)
        s = outcome_score(info["winner"], seat_a)
        total += s
        verdict = {1.0: "A wins", 0.5: "draw", 0.0: "B wins"}[s]
        print(f"\ngame {g+1}: {verdict}  crowns {info['crowns']}  t={info['t']:.1f}s"
              + ("  (timeout)" if info["timeout"] else ""))

    print(f"\nA score over {args.games} game(s): {total/args.games:.2f}")


if __name__ == "__main__":
    main()
