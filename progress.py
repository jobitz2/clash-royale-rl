#!/usr/bin/env python
"""Summarise a training run from its log.csv.

    python progress.py runs/league-20260805-164721
    python progress.py runs/<run> --plot        # ASCII curves, no matplotlib

Reports the two things that matter separately: Elo (relative, what self-play
optimises) and win rate against the fixed scripted bots (absolute, ungameable).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

BOTS = ("noop", "random", "rush", "heuristic")


def load(run: Path):
    path = run / "log.csv" if run.is_dir() else run
    with open(path) as f:
        return list(csv.DictReader(f))


def num(row, key, default=float("nan")):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def sparkline(values, width=60, height=9):
    """Tiny ASCII plot so this works with no plotting dependency."""
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return ["  (not enough data)"]
    step = max(1, len(vals) // width)
    vals = vals[::step][:width]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    grid = [[" "] * len(vals) for _ in range(height)]
    for x, v in enumerate(vals):
        y = int((v - lo) / span * (height - 1))
        grid[height - 1 - y][x] = "*"
    out = []
    for i, rowchars in enumerate(grid):
        label = hi if i == 0 else (lo if i == height - 1 else None)
        tag = f"{label:9.2f} " if label is not None else " " * 10
        out.append(tag + "|" + "".join(rowchars))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--tail", type=int, default=12)
    args = ap.parse_args()

    rows = load(args.run)
    if not rows:
        raise SystemExit("no rows in log")
    last = rows[-1]

    print(f"run          {args.run}")
    print(f"iterations   {len(rows)}")
    print(f"decisions    {int(num(last, 'steps', 0)):,}")
    print(f"game ticks   {int(num(last, 'ticks', 0)):,} "
          f"(~{num(last, 'ticks', 0) * 0.1 / 3600:.1f} hours of simulated play)")
    print(f"wall clock   {num(last, 'time', 0) / 60:.1f} min")
    print(f"throughput   {num(last, 'sps', 0):.0f} decisions/s")
    print(f"league elo   {num(last, 'learner_elo'):.1f}   pool {last.get('pool', '?')}")
    print(f"entropy      {num(last, 'entropy'):.3f}   "
          f"explained_var {num(last, 'explained_var'):.3f}")

    evals = [r for r in rows if r.get("eval_random", "")]
    if evals:
        print("\nabsolute win rate vs scripted bots (the ungameable measure)")
        print(f"  {'iter':>6}  " + "  ".join(f"{b:>10s}" for b in BOTS))
        show = evals[:: max(1, len(evals) // 10)]
        if show[-1] is not evals[-1]:
            show.append(evals[-1])
        for r in show:
            cells = "  ".join(f"{num(r, 'eval_' + b):>10.2f}" for b in BOTS)
            print(f"  {r['iter']:>6}  {cells}")

    print(f"\nlast {args.tail} iterations")
    hdr = ("iter", "elo", "winrate", "ent", "expl_var", "plays", "dps")
    keys = ("iter", "learner_elo", "winrate", "entropy", "explained_var",
            "plays", "sps")
    print("  " + "".join(f"{h:>10s}" for h in hdr))
    for r in rows[-args.tail:]:
        print("  " + "".join(
            f"{r['iter']:>10s}" if k == "iter" else f"{num(r, k):>10.2f}"
            for h, k in zip(hdr, keys)))

    if args.plot:
        for key, title in (("learner_elo", "league Elo (relative)"),
                           ("eval_heuristic", "win rate vs heuristic bot"),
                           ("eval_rush", "win rate vs rush bot"),
                           ("explained_var", "critic explained variance"),
                           ("entropy", "policy entropy")):
            series = [num(r, key) for r in rows]
            if not [v for v in series if v == v]:
                continue
            print(f"\n{title}")
            for line in sparkline(series):
                print(line)


if __name__ == "__main__":
    main()
