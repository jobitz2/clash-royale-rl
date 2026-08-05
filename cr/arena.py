"""Arena geometry and match constants.

Coordinates are continuous tiles. x grows left->right, y grows from player 0's
side (bottom, y=0) toward player 1's side (top, y=H). The river is the line
y = RIVER_Y, crossable by ground units only inside a bridge x-span.
"""

import numpy as np

W = 18          # arena width in tiles
H = 32          # arena height in tiles

RIVER_Y = 16.0
BRIDGE_HALF_W = 1.0
BRIDGE_X = (3.5, 14.5)          # bridge centre lines

# Tower placement, mirrored across RIVER_Y.
KING_Y = 2.5
PRINCESS_Y = 6.5
PRINCESS_X = (3.5, 14.5)
KING_X = 9.0

# Tournament-standard (level 11) tower stats.
KING_HP = 4008.0
KING_DMG = 109.0
KING_HIT = 1.0
KING_RANGE = 7.0
KING_RADIUS = 1.8

PRINCESS_HP = 2534.0
PRINCESS_DMG = 109.0
PRINCESS_HIT = 0.8
PRINCESS_RANGE = 7.5
PRINCESS_RADIUS = 1.2

# Match clock. Regular time is SINGLE + DOUBLE; if still tied we play OVERTIME
# at triple elixir under sudden-death (first tower lost ends it).
T_SINGLE = 120.0
T_DOUBLE = 60.0
T_OVERTIME = 120.0
T_REGULAR = T_SINGLE + T_DOUBLE

ELIXIR_MAX = 10.0
ELIXIR_START = 5.0
ELIXIR_PERIOD = 2.8            # seconds per elixir at 1x

DT = 0.1                       # physics timestep (seconds)

MAX_UNITS = 160                # per-side hard cap on simultaneous entities


def elixir_rate(t: float) -> float:
    """Elixir per second at match time `t` (seconds since start)."""
    if t < T_SINGLE:
        mult = 1.0
    elif t < T_REGULAR:
        mult = 2.0
    else:
        mult = 3.0
    return mult / ELIXIR_PERIOD


def mirror_y(y):
    """Map a y coordinate to the opposing player's frame of reference."""
    return H - y


def nearest_bridge_x(x: float) -> float:
    return BRIDGE_X[0] if abs(x - BRIDGE_X[0]) <= abs(x - BRIDGE_X[1]) else BRIDGE_X[1]


def on_bridge(x) -> bool:
    return (abs(x - BRIDGE_X[0]) <= BRIDGE_HALF_W) or (abs(x - BRIDGE_X[1]) <= BRIDGE_HALF_W)


def tower_positions(owner: int):
    """(king, princess_left, princess_right) centres for `owner`."""
    if owner == 0:
        return (
            (KING_X, KING_Y),
            (PRINCESS_X[0], PRINCESS_Y),
            (PRINCESS_X[1], PRINCESS_Y),
        )
    return (
        (KING_X, mirror_y(KING_Y)),
        (PRINCESS_X[0], mirror_y(PRINCESS_Y)),
        (PRINCESS_X[1], mirror_y(PRINCESS_Y)),
    )


def flip_state_xy(x, y):
    """Rotate a point 180 degrees so player 1 can reuse player 0's viewpoint."""
    return (W - x, H - y)
