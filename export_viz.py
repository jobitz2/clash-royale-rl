#!/usr/bin/env python
"""Export training curves, the league ladder and battle replays as JSON for viz/.

    python export_viz.py --run runs/league-20260805-165311 --out viz/data.json

Replays are recorded from real matches played by the checkpoint, subsampled and
quantised so a few full battles fit in a page without a server.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cr import arena as A
from cr.bots import BOT_NAMES, make_bot
from cr.cards import CARDS
from cr.engine import KING, NOT_TOWER
from cr.env import CRDuel, outcome_score
from rl.evaluate import NetPolicy
from rl.net import ActorCritic

CURVE_KEYS = ("iter", "steps", "ticks", "time", "learner_elo", "winrate",
              "entropy", "explained_var", "plays", "ep_len", "legal_actions",
              "skip_ticks", "clipfrac", "pool", "sps")
EVAL_KEYS = tuple(f"eval_{b}" for b in BOT_NAMES)


def num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_curves(run: Path):
    rows = list(csv.DictReader(open(run / "log.csv")))
    out = {k: [] for k in CURVE_KEYS}
    for r in rows:
        for k in CURVE_KEYS:
            out[k].append(num(r.get(k)))
    evals = []
    for r in rows:
        if r.get("eval_random"):
            evals.append({"iter": int(num(r["iter"])),
                          **{b: num(r[f"eval_{b}"]) for b in BOT_NAMES}})
    return out, evals


def load_ladder(ckpt):
    """League standings from the checkpoint: bots plus every frozen snapshot."""
    lg = ckpt["league"]
    rows = []
    for e in lg["entries"]:
        rows.append({"label": e["name"] if e["kind"] == "bot"
                     else f"snap@{e['iteration']}",
                     "kind": e["kind"], "elo": round(e["elo"], 1),
                     "iteration": e["iteration"], "games": e["games"],
                     "winrate_vs_learner": (round(e["wins"] / e["games"], 3)
                                            if e["games"] else None),
                     "anchored": e["anchored"]})
    rows.sort(key=lambda r: -r["elo"])
    return {"learner_elo": round(lg["learner_elo"], 1), "entries": rows}


def record_match(pol_a, pol_b, label_a, label_b, seed, a_seat=0, stride=2,
                 frame_skip=3):
    """Play one match, capturing a subsampled quantised replay."""
    env = CRDuel(frame_skip=frame_skip, seed=seed)
    env.reset(seed=seed)
    pol_a.reset()
    pol_b.reset()
    eng = env.eng
    seat_a, seat_b = a_seat, 1 - a_seat

    frames, plays = [], []
    tick = 0
    done = False
    info = {}
    while not done:
        aa = pol_a.act(env, seat_a, env.mask(seat_a)) if env.can_act(seat_a) else 0
        ab = pol_b.act(env, seat_b, env.mask(seat_b)) if env.can_act(seat_b) else 0
        for seat, act in ((seat_a, aa), (seat_b, ab)):
            slot, px, py = CRDuel.decode(seat, int(act))
            if slot >= 0 and eng.can_afford(seat, slot):
                plays.append({"tick": tick, "seat": seat,
                              "card": int(eng.hands[seat][slot]),
                              "x": round(float(px), 2), "y": round(float(py), 2)})
        a0, a1 = (aa, ab) if seat_a == 0 else (ab, aa)
        rew, done, info = env.step(a0, a1)
        tick += 1

        if tick % stride == 0 or done:
            live = np.flatnonzero(eng.alive & (eng.tower == NOT_TOWER))
            units = []
            for j in live:
                j = int(j)
                units += [int(round(float(eng.x[j]) * 10)),
                          int(round(float(eng.y[j]) * 10)),
                          int(eng.owner[j]), int(eng.card[j]),
                          int(round(100 * float(eng.hp[j] / eng.max_hp[j])))]
            frames.append({
                "t": round(float(eng.t), 1),
                "e": [round(float(eng.elixir[0]), 1), round(float(eng.elixir[1]), 1)],
                "th": [[int(round(100 * f)) for f in eng.tower_hp_frac(0)],
                       [int(round(100 * f)) for f in eng.tower_hp_frac(1)]],
                "c": list(eng.crowns),
                "u": units,
            })

    return {
        "a": label_a, "b": label_b, "a_seat": seat_a, "seed": seed,
        "stride": stride, "frame_skip": frame_skip,
        "winner": info["winner"], "crowns": info["crowns"],
        "duration": round(info["t"], 1),
        "score_for_a": outcome_score(info["winner"], seat_a),
        "overtime": bool(eng.sudden_death),
        "frames": frames, "plays": plays,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("viz/data.json"))
    ap.add_argument("--replays", type=int, default=2,
                    help="matches to record against each chosen opponent")
    ap.add_argument("--vs", nargs="*", default=["heuristic", "rush"])
    ap.add_argument("--stride", type=int, default=2, help="record every N ticks")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ck = torch.load(args.run / "latest.pt", map_location=args.device,
                    weights_only=False)
    net = ActorCritic().to(args.device)
    net.load_state_dict(ck["net"])
    net.eval()
    pol = NetPolicy(net, device=args.device)

    curves, evals = load_curves(args.run)
    ladder = load_ladder(ck)

    replays = []
    for opp_name in args.vs:
        for g in range(args.replays):
            bot = make_bot(opp_name, seed=11 + g)
            r = record_match(pol, bot, f"agent@{ck['iter']}", opp_name,
                             seed=4242 + 97 * g, a_seat=g % 2,
                             stride=args.stride)
            replays.append(r)
            print(f"recorded vs {opp_name} seed={r['seed']} seat={r['a_seat']}: "
                  f"score={r['score_for_a']} crowns={r['crowns']} "
                  f"{r['duration']}s {len(r['frames'])} frames "
                  f"{len(r['plays'])} plays")

    data = {
        "meta": {
            "run": args.run.name,
            "iters": ck["iter"] + 1,
            "decisions": ck["steps"],
            "ticks": int(curves["ticks"][-1] or 0),
            "wall_minutes": round((curves["time"][-1] or 0) / 60, 1),
            "params": sum(p.numel() for p in net.parameters()),
            "dt": A.DT,
        },
        "arena": {
            "w": A.W, "h": A.H, "river_y": A.RIVER_Y,
            "bridge_x": list(A.BRIDGE_X), "bridge_half_w": A.BRIDGE_HALF_W,
            "king": [A.KING_X, A.KING_Y], "princess_x": list(A.PRINCESS_X),
            "princess_y": A.PRINCESS_Y,
            "t_single": A.T_SINGLE, "t_regular": A.T_REGULAR,
        },
        "cards": [{"name": c.name, "cost": c.cost, "kind": c.kind,
                   "air": c.is_air, "hits_air": c.hits_air,
                   "buildings_only": c.buildings_only} for c in CARDS],
        "curves": curves,
        "evals": evals,
        "ladder": ladder,
        "replays": replays,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":")))
    kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out}  ({kb:,.0f} KB)")


if __name__ == "__main__":
    main()
