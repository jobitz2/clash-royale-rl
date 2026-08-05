"""Clash Royale battle simulator.

Entities (troops, buildings and crown towers) live in one struct-of-arrays block
so the hot loops are numpy-vectorised. The engine is deliberately symmetric:
owner 0 pushes toward +y, owner 1 toward -y, and `Engine` never favours either.

Fidelity notes -- this is a faithful-but-simplified model, not a reimplementation.
Modelled: elixir economy and rate bands, 8-card cycling decks, hand of 4,
continuous movement, bridge-constrained ground pathing, aggro/sight retargeting,
buildings-only targeting, air/ground immunity, splash, spells with reduced tower
damage, king-tower activation, deployment-zone unlocking, crown-count and
sudden-death overtime. Not modelled: unit collision mass/push, charge attacks,
death spawns, shield/rage/clone mechanics, per-unit deploy jitter.
"""

from __future__ import annotations

import math

import numpy as np

from . import arena as A
from .cards import CARDS, SPELL, BUILDING, TROOP, deck_ids

SIGHT = 5.5              # radius within which troops divert onto enemy troops
RETARGET_PERIOD = 0.4    # seconds between target re-evaluations
SEPARATION = 0.55        # ground-unit soft body separation strength

NOT_TOWER, PRINCESS, KING = 0, 1, 2


class Engine:
    """One battle. Advance with `step(dt)`, issue plays with `play(...)`."""

    __slots__ = (
        "cap", "n", "rng", "t", "elixir", "decks", "hands", "cycles",
        "alive", "owner", "card", "tower", "x", "y", "hp", "max_hp",
        "dmg", "hit_interval", "hit_cd", "atk_range", "splash", "speed",
        "radius", "is_air", "hits_ground", "hits_air", "buildings_only",
        "is_building", "active", "deploy_cd", "lifetime", "target",
        "retarget_cd", "tower_idx", "king_active", "pending_spells",
        "tower_damage_dealt", "elixir_leaked", "crowns", "winner", "done",
        "sudden_death", "_deploy_mask_cache",
    )

    def __init__(self, decks=None, seed: int | None = None, cap: int = A.MAX_UNITS):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        d = deck_ids() if decks is None else decks
        if isinstance(d[0], (list, tuple)):
            self.decks = [list(d[0]), list(d[1])]
        else:
            self.decks = [list(d), list(d)]
        self._alloc()
        self.reset()

    # ------------------------------------------------------------------ setup

    def _alloc(self):
        c = self.cap
        f = lambda: np.zeros(c, dtype=np.float32)
        b = lambda: np.zeros(c, dtype=bool)
        self.alive = b()
        self.owner = np.zeros(c, dtype=np.int8)
        self.card = np.full(c, -1, dtype=np.int16)
        self.tower = np.zeros(c, dtype=np.int8)
        self.x, self.y = f(), f()
        self.hp, self.max_hp = f(), f()
        self.dmg, self.hit_interval, self.hit_cd = f(), f(), f()
        self.atk_range, self.splash = f(), f()
        self.speed, self.radius = f(), f()
        self.is_air, self.hits_ground, self.hits_air = b(), b(), b()
        self.buildings_only, self.is_building, self.active = b(), b(), b()
        self.deploy_cd, self.lifetime = f(), f()
        self.target = np.full(c, -1, dtype=np.int32)
        self.retarget_cd = f()

    def reset(self):
        self.alive[:] = False
        self.target[:] = -1
        self.n = 0
        self.t = 0.0
        self.elixir = [A.ELIXIR_START, A.ELIXIR_START]
        self.pending_spells = []
        self.tower_damage_dealt = [0.0, 0.0]
        self.elixir_leaked = [0.0, 0.0]
        self.crowns = [0, 0]
        self.king_active = [False, False]
        self.winner = None
        self.done = False
        self.sudden_death = False
        self._deploy_mask_cache = None

        # tower_idx[owner] = (king, left princess, right princess)
        self.tower_idx = [[], []]
        for o in (0, 1):
            (kx, ky), (lx, ly), (rx, ry) = A.tower_positions(o)
            self.tower_idx[o].append(self._spawn_tower(o, KING, kx, ky))
            self.tower_idx[o].append(self._spawn_tower(o, PRINCESS, lx, ly))
            self.tower_idx[o].append(self._spawn_tower(o, PRINCESS, rx, ry))

        # Shuffle decks, deal a hand of 4, rest is the cycle queue.
        self.hands = []
        self.cycles = []
        for o in (0, 1):
            order = list(self.decks[o])
            self.rng.shuffle(order)
            self.hands.append(order[:4])
            self.cycles.append(order[4:])
        return self

    def _free_slot(self) -> int:
        idx = np.flatnonzero(~self.alive)
        return int(idx[0]) if idx.size else -1

    def _spawn_tower(self, owner, kind, px, py) -> int:
        i = self._free_slot()
        self.alive[i] = True
        self.owner[i] = owner
        self.card[i] = -1
        self.tower[i] = kind
        self.x[i], self.y[i] = px, py
        if kind == KING:
            self.hp[i] = self.max_hp[i] = A.KING_HP
            self.dmg[i] = A.KING_DMG
            self.hit_interval[i] = A.KING_HIT
            self.atk_range[i] = A.KING_RANGE
            self.radius[i] = A.KING_RADIUS
            self.active[i] = False
        else:
            self.hp[i] = self.max_hp[i] = A.PRINCESS_HP
            self.dmg[i] = A.PRINCESS_DMG
            self.hit_interval[i] = A.PRINCESS_HIT
            self.atk_range[i] = A.PRINCESS_RANGE
            self.radius[i] = A.PRINCESS_RADIUS
            self.active[i] = True
        self.hits_ground[i] = True
        self.hits_air[i] = True
        self.is_building[i] = True
        self.speed[i] = 0.0
        self.lifetime[i] = np.inf
        self.hit_cd[i] = 0.0
        return i

    # ------------------------------------------------------------- deployment

    def hand(self, owner):
        return list(self.hands[owner])

    def next_in_cycle(self, owner):
        return self.cycles[owner][0] if self.cycles[owner] else -1

    def can_afford(self, owner, slot) -> bool:
        if slot < 0 or slot >= len(self.hands[owner]):
            return False
        return self.elixir[owner] >= CARDS[self.hands[owner][slot]].cost - 1e-6

    def deploy_mask(self, owner) -> np.ndarray:
        """Boolean [H, W] mask of tile centres `owner` may deploy a troop on."""
        if self._deploy_mask_cache is not None and owner in self._deploy_mask_cache:
            return self._deploy_mask_cache[owner]
        ys = np.arange(A.H, dtype=np.float32) + 0.5
        xs = np.arange(A.W, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)                    # [H, W]

        own_half = gy < A.RIVER_Y if owner == 0 else gy > A.RIVER_Y
        mask = own_half.copy()

        # Destroying an enemy princess tower opens that side of the enemy half.
        for k, ti in enumerate(self.tower_idx[1 - owner][1:]):
            if not self.alive[ti]:
                side = gx < A.W / 2 if k == 0 else gx >= A.W / 2
                enemy_half = gy > A.RIVER_Y if owner == 0 else gy < A.RIVER_Y
                mask |= side & enemy_half

        # No placing on the river except on a bridge.
        river = np.abs(gy - A.RIVER_Y) < 1.0
        bridge = (np.abs(gx - A.BRIDGE_X[0]) <= A.BRIDGE_HALF_W) | \
                 (np.abs(gx - A.BRIDGE_X[1]) <= A.BRIDGE_HALF_W)
        mask &= ~(river & ~bridge)

        # No placing inside a standing building's footprint.
        live = np.flatnonzero(self.alive & self.is_building)
        for j in live:
            r = self.radius[j] + 0.4
            mask &= ~((np.abs(gx - self.x[j]) < r) & (np.abs(gy - self.y[j]) < r))

        if self._deploy_mask_cache is None:
            self._deploy_mask_cache = {}
        self._deploy_mask_cache[owner] = mask
        return mask

    def spell_mask(self, owner) -> np.ndarray:
        """Spells may be cast on any tile."""
        return np.ones((A.H, A.W), dtype=bool)

    def legal_mask(self, owner) -> np.ndarray:
        """Boolean [4, H, W]: may `owner` play hand slot s at tile (y, x)?"""
        out = np.zeros((4, A.H, A.W), dtype=bool)
        dep = self.deploy_mask(owner)
        for s in range(len(self.hands[owner])):
            if not self.can_afford(owner, s):
                continue
            cd = CARDS[self.hands[owner][s]]
            out[s] = self.spell_mask(owner) if cd.kind == SPELL else dep
        return out

    def play(self, owner: int, slot: int, tx: float, ty: float) -> bool:
        """Attempt to play hand `slot` at (tx, ty). Returns True if it happened."""
        if self.done or not self.can_afford(owner, slot):
            return False
        cid = self.hands[owner][slot]
        cd = CARDS[cid]
        iy, ix = int(np.clip(ty, 0, A.H - 1e-3)), int(np.clip(tx, 0, A.W - 1e-3))
        if cd.kind == SPELL:
            if not self.spell_mask(owner)[iy, ix]:
                return False
        elif not self.deploy_mask(owner)[iy, ix]:
            return False

        self.elixir[owner] -= cd.cost
        # Cycle: played card leaves the hand, the queue refills it.
        self.cycles[owner].append(cid)
        self.hands[owner][slot] = self.cycles[owner].pop(0)

        if cd.kind == SPELL:
            self.pending_spells.append([owner, cid, float(tx), float(ty), cd.deploy_time])
        else:
            self._spawn_troop(owner, cid, float(tx), float(ty))
            self._deploy_mask_cache = None
        return True

    def _spawn_troop(self, owner, cid, tx, ty):
        cd = CARDS[cid]
        for k in range(cd.count):
            i = self._free_slot()
            if i < 0:
                return
            if cd.count == 1:
                px, py = tx, ty
            else:
                ang = 2 * np.pi * k / cd.count
                px = tx + cd.spread * np.cos(ang)
                py = ty + cd.spread * np.sin(ang)
            self.alive[i] = True
            self.owner[i] = owner
            self.card[i] = cid
            self.tower[i] = NOT_TOWER
            self.x[i] = np.clip(px, 0.2, A.W - 0.2)
            self.y[i] = np.clip(py, 0.2, A.H - 0.2)
            self.hp[i] = self.max_hp[i] = cd.hp
            self.dmg[i] = cd.dmg
            self.hit_interval[i] = cd.hit_interval
            self.hit_cd[i] = 0.0
            self.atk_range[i] = cd.attack_range
            self.splash[i] = cd.splash
            self.speed[i] = cd.speed
            self.radius[i] = cd.radius
            self.is_air[i] = cd.is_air
            self.hits_ground[i] = cd.hits_ground
            self.hits_air[i] = cd.hits_air
            self.buildings_only[i] = cd.buildings_only
            self.is_building[i] = cd.kind == BUILDING
            self.active[i] = True
            self.deploy_cd[i] = cd.deploy_time
            self.lifetime[i] = cd.lifetime if cd.lifetime > 0 else np.inf
            self.target[i] = -1
            self.retarget_cd[i] = 0.0

    # ------------------------------------------------------------------- step

    def step(self, dt: float = A.DT):
        if self.done:
            return
        self.t += dt

        rate = A.elixir_rate(self.t)
        for o in (0, 1):
            e = self.elixir[o] + rate * dt
            if e > A.ELIXIR_MAX:
                self.elixir_leaked[o] += e - A.ELIXIR_MAX
                e = A.ELIXIR_MAX
            self.elixir[o] = e

        self._resolve_spells(dt)

        live = np.flatnonzero(self.alive)
        if live.size:
            self.deploy_cd[live] = np.maximum(0.0, self.deploy_cd[live] - dt)
            self.hit_cd[live] = np.maximum(0.0, self.hit_cd[live] - dt)
            self.retarget_cd[live] = np.maximum(0.0, self.retarget_cd[live] - dt)
            finite = live[np.isfinite(self.lifetime[live])]
            if finite.size:
                self.lifetime[finite] -= dt
                expired = finite[self.lifetime[finite] <= 0]
                if expired.size:
                    self.hp[expired] = 0.0

            # Enemy tables are built once per tick and shared by every unit;
            # rebuilding them per unit dominated the profile.
            views = self._enemy_views(live)
            self._retarget(live, views)
            self._move(live, dt)
            self._attack(live, dt, views)

        self._reap()
        self._check_end()

    def _resolve_spells(self, dt):
        still = []
        for s in self.pending_spells:
            s[4] -= dt
            if s[4] > 0:
                still.append(s)
                continue
            owner, cid, sx, sy, _ = s
            cd = CARDS[cid]
            foes = np.flatnonzero(self.alive & (self.owner != owner))
            if foes.size:
                d = np.hypot(self.x[foes] - sx, self.y[foes] - sy)
                hit = foes[d <= cd.spell_radius + self.radius[foes]]
                for j in hit:
                    amount = cd.spell_damage
                    if self.tower[j] != NOT_TOWER:
                        amount *= cd.spell_tower_mult
                    self._damage(int(j), amount, owner)
        self.pending_spells = still

    def _enemy_views(self, live):
        """For each owner, the entities it may consider hostile, as flat arrays.

        Built once per tick. Every unit's targeting query then filters these
        small arrays instead of re-scanning the full capacity block.
        """
        views = {}
        for o in (0, 1):
            idx = live[self.owner[live] != o]
            views[o] = (idx, self.x[idx], self.y[idx], self.radius[idx],
                        self.is_air[idx], self.is_building[idx])
        return views

    def _visible(self, i, view):
        """Subset of `view` that unit `i` is allowed to shoot at."""
        idx, ex, ey, er, eair, ebld = view
        if idx.size == 0:
            return None
        if self.buildings_only[i]:
            sel = ebld
        elif self.hits_air[i] and self.hits_ground[i]:
            sel = None
        else:
            sel = eair if self.hits_air[i] else ~eair
        if sel is not None:
            if not sel.any():
                return None
            idx, ex, ey, er, ebld = idx[sel], ex[sel], ey[sel], er[sel], ebld[sel]
        return idx, ex, ey, er, ebld

    def _retarget(self, live, views):
        act = live[(self.deploy_cd[live] <= 0) & self.active[live]]
        if act.size == 0:
            return
        for i in act:
            i = int(i)
            cur = self.target[i]
            if cur >= 0 and self.alive[cur] and self.retarget_cd[i] > 0:
                continue
            self.retarget_cd[i] = RETARGET_PERIOD
            vis = self._visible(i, views[int(self.owner[i])])
            if vis is None:
                self.target[i] = -1
                continue
            idx, ex, ey, er, ebld = vis
            d = np.hypot(ex - self.x[i], ey - self.y[i])

            if self.is_building[i]:
                # Towers and defensive buildings only engage what is in range.
                inr = d <= self.atk_range[i] + er + self.radius[i]
                if not inr.any():
                    self.target[i] = -1
                    continue
                cand, dd, cbld = idx[inr], d[inr], ebld[inr]
                troops = ~cbld
                if troops.any():                     # prefer troops over buildings
                    cand, dd = cand[troops], dd[troops]
                self.target[i] = int(cand[np.argmin(dd)])
                continue

            if self.buildings_only[i]:
                self.target[i] = int(idx[np.argmin(d)])
                continue

            near = ~ebld & (d <= SIGHT)
            if near.any():
                self.target[i] = int(idx[near][np.argmin(d[near])])
            elif ebld.any():
                self.target[i] = int(idx[ebld][np.argmin(d[ebld])])
            else:
                self.target[i] = int(idx[np.argmin(d)])

    def _waypoint(self, i, tgt, ux, uy):
        """Next point unit `i` walks toward, honouring the river for ground units."""
        tx, ty = float(self.x[tgt]), float(self.y[tgt])
        if self.is_air[i]:
            return tx, ty
        same_side = (uy < A.RIVER_Y) == (ty < A.RIVER_Y)
        if same_side:
            return tx, ty
        # Choose the bridge minimising total path length, then walk the bank.
        best, bestd = A.BRIDGE_X[0], math.inf
        for bx in A.BRIDGE_X:
            dd = math.hypot(ux - bx, uy - A.RIVER_Y) + math.hypot(bx - tx, A.RIVER_Y - ty)
            if dd < bestd:
                best, bestd = bx, dd
        if abs(ux - best) > 0.45:
            bank = A.RIVER_Y - 1.2 if uy < A.RIVER_Y else A.RIVER_Y + 1.2
            return best, bank if abs(uy - A.RIVER_Y) > 1.2 else uy
        cross = A.RIVER_Y + 1.6 if uy < A.RIVER_Y else A.RIVER_Y - 1.6
        return best, cross

    def _move(self, live, dt):
        mob = live[(self.speed[live] > 0) & (self.deploy_cd[live] <= 0)]
        if mob.size == 0:
            return
        px_all, py_all = self.x, self.y
        for i in mob:
            i = int(i)
            tgt = int(self.target[i])
            if tgt < 0:
                continue
            ux, uy = float(px_all[i]), float(py_all[i])
            reach = float(self.atk_range[i] + self.radius[i] + self.radius[tgt])
            if math.hypot(float(px_all[tgt]) - ux, float(py_all[tgt]) - uy) <= reach:
                continue
            wx, wy = self._waypoint(i, tgt, ux, uy)
            dx, dy = wx - ux, wy - uy
            n = math.hypot(dx, dy)
            if n < 1e-6:
                continue
            stepd = float(self.speed[i]) * dt / n
            px_all[i] = ux + dx * stepd
            py_all[i] = uy + dy * stepd

        # Soft separation so ground troops do not perfectly stack.
        grd = mob[~self.is_air[mob]]
        if grd.size > 1:
            px, py = self.x[grd], self.y[grd]
            dx = px[:, None] - px[None, :]
            dy = py[:, None] - py[None, :]
            dist = np.hypot(dx, dy)
            rr = self.radius[grd][:, None] + self.radius[grd][None, :]
            np.fill_diagonal(dist, np.inf)
            over = (dist < rr) & (dist > 1e-6)
            if over.any():
                # Guard the non-overlapping entries (dist may be inf) before
                # dividing, otherwise the unused branch produces inf/nan.
                safe = np.where(over, dist, 1.0)
                w = np.where(over, (rr - safe) / safe, 0.0)
                self.x[grd] += SEPARATION * (w * dx).sum(1) * dt
                self.y[grd] += SEPARATION * (w * dy).sum(1) * dt

        np.clip(self.x, 0.2, A.W - 0.2, out=self.x)
        np.clip(self.y, 0.2, A.H - 0.2, out=self.y)

    def _attack(self, live, dt, views):
        act = live[(self.deploy_cd[live] <= 0) & self.active[live] &
                   (self.dmg[live] > 0) & (self.hit_cd[live] <= 0)]
        for i in act:
            i = int(i)
            tgt = int(self.target[i])
            if tgt < 0 or not self.alive[tgt]:
                continue
            reach = float(self.atk_range[i] + self.radius[i] + self.radius[tgt])
            tx, ty = float(self.x[tgt]), float(self.y[tgt])
            if math.hypot(tx - float(self.x[i]), ty - float(self.y[i])) > reach:
                continue
            self.hit_cd[i] = self.hit_interval[i]
            o = int(self.owner[i])
            if self.splash[i] > 0:
                vis = self._visible(i, views[o])
                if vis is not None:
                    idx, ex, ey, er, _ = vis
                    d = np.hypot(ex - tx, ey - ty)
                    for j in idx[d <= self.splash[i] + er]:
                        self._damage(int(j), float(self.dmg[i]), o)
            else:
                self._damage(tgt, float(self.dmg[i]), o)

    def _damage(self, j, amount, by_owner):
        if not self.alive[j] or amount <= 0:
            return
        amount = min(amount, float(self.hp[j]))
        self.hp[j] -= amount
        if self.tower[j] != NOT_TOWER:
            self.tower_damage_dealt[by_owner] += amount
            victim = int(self.owner[j])
            if not self.king_active[victim]:
                self._activate_king(victim)

    def _activate_king(self, owner):
        self.king_active[owner] = True
        self.active[self.tower_idx[owner][0]] = True

    def _reap(self):
        dead = np.flatnonzero(self.alive & (self.hp <= 0))
        if dead.size == 0:
            return
        for j in dead:
            j = int(j)
            self.alive[j] = False
            if self.is_building[j]:
                self._deploy_mask_cache = None
            if self.tower[j] != NOT_TOWER:
                victim = int(self.owner[j])
                self.crowns[1 - victim] += 1
                self._deploy_mask_cache = None
                if not self.king_active[victim]:
                    self._activate_king(victim)
                if self.tower[j] == KING:
                    self.winner = 1 - victim
                    self.done = True
                elif self.sudden_death and self.winner is None:
                    self.winner = 1 - victim
                    self.done = True
        stale = np.isin(self.target, dead)
        self.target[stale] = -1
        self.retarget_cd[stale] = 0.0

    def _check_end(self):
        if self.done:
            return
        if self.t >= A.T_REGULAR and not self.sudden_death:
            if self.crowns[0] != self.crowns[1]:
                self.winner = 0 if self.crowns[0] > self.crowns[1] else 1
                self.done = True
            else:
                self.sudden_death = True
        if self.sudden_death and self.t >= A.T_REGULAR + A.T_OVERTIME:
            # Lowest-HP-tower tiebreak, then draw.
            f = [self._lowest_tower_frac(0), self._lowest_tower_frac(1)]
            if abs(f[0] - f[1]) > 1e-4:
                self.winner = 0 if f[0] > f[1] else 1
            self.done = True

    def _lowest_tower_frac(self, owner) -> float:
        vals = [self.hp[i] / self.max_hp[i] for i in self.tower_idx[owner]
                if self.alive[i]]
        return min(vals) if vals else 0.0

    # ---------------------------------------------------------------- queries

    def tower_hp_frac(self, owner):
        """[king, left, right] HP fractions for `owner`."""
        return [float(self.hp[i] / self.max_hp[i]) if self.alive[i] else 0.0
                for i in self.tower_idx[owner]]

    def total_tower_frac(self, owner) -> float:
        """Crown-weighted health of `owner`'s towers in [0, 1]."""
        f = self.tower_hp_frac(owner)
        return (2.0 * f[0] + f[1] + f[2]) / 4.0

    def time_left(self) -> float:
        end = A.T_REGULAR + A.T_OVERTIME if self.sudden_death else A.T_REGULAR
        return max(0.0, end - self.t)
