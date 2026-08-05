"""Terminal renderer, so matches can be watched instead of only scored."""

from __future__ import annotations

import numpy as np

from . import arena as A
from .cards import CARDS
from .engine import KING, NOT_TOWER

RESET = "\033[0m"
BLUE = "\033[94m"
RED = "\033[91m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

# One glyph per card, plus towers.
GLYPH = {
    "Knight": "K", "Archers": "a", "Giant": "G", "Musketeer": "M",
    "Minions": "m", "Skeletons": "s", "MiniPekka": "P", "Valkyrie": "V",
    "BabyDragon": "d", "Cannon": "C", "Tesla": "T",
}


def frame(eng, width_pad: int = 2) -> str:
    grid = [[" " for _ in range(A.W)] for _ in range(A.H)]
    colour = [[None for _ in range(A.W)] for _ in range(A.H)]

    # River and bridges.
    ry = int(A.RIVER_Y)
    for x in range(A.W):
        on_bridge = any(abs(x + 0.5 - bx) <= A.BRIDGE_HALF_W for bx in A.BRIDGE_X)
        grid[ry][x] = "=" if on_bridge else "~"
        colour[ry][x] = YELLOW if on_bridge else CYAN

    live = np.flatnonzero(eng.alive)
    # Draw towers first so troops overlay them.
    for j in sorted(live, key=lambda k: eng.tower[k] == NOT_TOWER):
        j = int(j)
        ix = int(np.clip(eng.x[j], 0, A.W - 1))
        iy = int(np.clip(eng.y[j], 0, A.H - 1))
        if eng.tower[j] != NOT_TOWER:
            ch = "#" if eng.tower[j] == KING else "@"
        else:
            ch = GLYPH.get(CARDS[eng.card[j]].name, "?")
        grid[iy][ix] = ch
        colour[iy][ix] = BLUE if eng.owner[j] == 0 else RED

    pad = " " * width_pad
    lines = []
    # Player 1 is at the top, so print rows in descending y.
    for iy in range(A.H - 1, -1, -1):
        row = "".join(
            (colour[iy][ix] + grid[iy][ix] + RESET) if colour[iy][ix] else grid[iy][ix]
            for ix in range(A.W)
        )
        lines.append(f"{pad}|{row}|")

    top = pad + "+" + "-" * A.W + "+"
    hdr = _scoreboard(eng)
    return "\n".join([hdr, top] + lines + [top, _legend()])


def _bar(frac, n=10):
    filled = int(round(frac * n))
    return "#" * filled + "." * (n - filled)


def _scoreboard(eng) -> str:
    f0 = eng.tower_hp_frac(0)
    f1 = eng.tower_hp_frac(1)
    phase = "OT" if eng.sudden_death else ("2x" if eng.t >= A.T_SINGLE else "1x")
    left = f"{BLUE}P0{RESET} e={eng.elixir[0]:4.1f} K[{_bar(f0[0])}] L[{_bar(f0[1])}] R[{_bar(f0[2])}] crowns={eng.crowns[0]}"
    right = f"{RED}P1{RESET} e={eng.elixir[1]:4.1f} K[{_bar(f1[0])}] L[{_bar(f1[1])}] R[{_bar(f1[2])}] crowns={eng.crowns[1]}"
    clock = f"{DIM}t={eng.t:6.1f}s  left={eng.time_left():5.1f}s  {phase}{RESET}"
    return "\n".join([clock, right, left])


def _legend() -> str:
    parts = [f"{k}={v}" for v, k in sorted((v, k) for k, v in GLYPH.items())]
    return DIM + "  #=king @=princess  " + " ".join(parts) + RESET


def hand_line(eng, p) -> str:
    col = BLUE if p == 0 else RED
    items = []
    for s, cid in enumerate(eng.hands[p]):
        cd = CARDS[cid]
        ok = "*" if eng.can_afford(p, s) else " "
        items.append(f"{ok}{s}:{cd.name}({cd.cost})")
    nxt = eng.next_in_cycle(p)
    tail = f"  next={CARDS[nxt].name}" if nxt >= 0 else ""
    return f"{col}P{p} hand{RESET} " + "  ".join(items) + DIM + tail + RESET
