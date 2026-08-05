"""Rules tests for the battle engine.

These pin the mechanics the agent is supposed to learn to exploit. If a card,
zone or targeting rule silently changes, the policy would quietly learn the wrong
game and only Elo -- which is measured against itself -- would look fine.
"""

import math

import numpy as np
import pytest

from cr import arena as A
from cr.cards import CARD_INDEX, CARDS, SPELL
from cr.engine import Engine, KING, PRINCESS
from cr.env import CRDuel


def run(eng, seconds, dt=A.DT):
    # round, not int: 0.6 / 0.1 is 5.999... in binary floating point.
    for _ in range(round(seconds / dt)):
        eng.step(dt)


def spawn(eng, owner, name, x, y):
    eng._spawn_troop(owner, CARD_INDEX[name], x, y)
    eng._deploy_mask_cache = None
    return int(np.flatnonzero(eng.alive & (eng.card == CARD_INDEX[name]))[-1])


# ------------------------------------------------------------------ economy

def test_elixir_regen_and_cap():
    eng = Engine(seed=0)
    assert eng.elixir[0] == pytest.approx(A.ELIXIR_START)
    run(eng, A.ELIXIR_PERIOD)
    assert eng.elixir[0] == pytest.approx(A.ELIXIR_START + 1.0, abs=0.05)
    run(eng, 60)
    assert eng.elixir[0] == pytest.approx(A.ELIXIR_MAX)
    assert eng.elixir_leaked[0] > 0, "overflow past the cap must be recorded"


def test_elixir_rate_bands():
    assert A.elixir_rate(0.0) == pytest.approx(1 / A.ELIXIR_PERIOD)
    assert A.elixir_rate(A.T_SINGLE + 1) == pytest.approx(2 / A.ELIXIR_PERIOD)
    assert A.elixir_rate(A.T_REGULAR + 1) == pytest.approx(3 / A.ELIXIR_PERIOD)


def test_playing_a_card_costs_elixir_and_cycles_the_hand():
    eng = Engine(seed=1)
    eng.elixir[0] = 10.0
    hand_before = eng.hand(0)
    nxt = eng.next_in_cycle(0)
    slot = next(s for s in range(4) if CARDS[hand_before[s]].kind != SPELL)
    cost = CARDS[hand_before[slot]].cost
    assert eng.play(0, slot, 9.0, 5.0)
    assert eng.elixir[0] == pytest.approx(10.0 - cost)
    assert eng.hands[0][slot] == nxt, "played slot refills from the cycle queue"
    assert hand_before[slot] in eng.cycles[0], "played card goes to the back"


def test_cannot_play_what_you_cannot_afford():
    eng = Engine(seed=2)
    eng.elixir[0] = 0.0
    assert not eng.play(0, 0, 9.0, 5.0)
    assert eng.elixir[0] == 0.0


# ---------------------------------------------------------------- deployment

def test_deploy_zone_is_own_half():
    eng = Engine(seed=3)
    mask = eng.deploy_mask(0)
    assert mask[10, 9], "open ground in own half is deployable"
    assert not mask[A.H - 10, 9], "enemy half is not"
    m1 = eng.deploy_mask(1)
    assert m1[A.H - 10, 9] and not m1[10, 9], "mirrored for player 1"


def test_river_blocks_deployment_except_on_bridges():
    eng = Engine(seed=3)
    mask = eng.deploy_mask(0)
    own_bank = int(A.RIVER_Y) - 1        # last row on player 0's side
    assert not mask[own_bank, 9], "mid-river is not deployable"
    assert mask[own_bank, int(A.BRIDGE_X[0])], "bridges are"


def test_destroying_a_princess_tower_opens_enemy_territory():
    eng = Engine(seed=4)
    assert not eng.deploy_mask(0)[A.H - 8, 3]
    left = eng.tower_idx[1][1]                  # player 1's left princess
    eng.hp[left] = 0.0
    eng._reap()
    assert eng.deploy_mask(0)[A.H - 8, 3], "that lane's enemy half unlocks"
    assert not eng.deploy_mask(0)[A.H - 8, 15], "the other lane stays closed"


def test_cannot_deploy_onto_a_tower_footprint():
    eng = Engine(seed=5)
    kx, ky = A.KING_X, A.KING_Y
    assert not eng.deploy_mask(0)[int(ky), int(kx)]


# ------------------------------------------------------------------ pathing

def test_ground_troop_crosses_at_a_bridge():
    eng = Engine(seed=6)
    i = spawn(eng, 0, "Giant", 9.0, 12.0)       # centre, away from both bridges
    eng.deploy_cd[i] = 0.0
    run(eng, 14)
    assert eng.alive[i]
    on_bridge_x = min(abs(float(eng.x[i]) - bx) for bx in A.BRIDGE_X)
    assert on_bridge_x < 2.0, "must funnel toward a bridge, not swim the middle"


def test_ground_troop_never_stands_mid_river_off_bridge():
    eng = Engine(seed=6)
    i = spawn(eng, 0, "Knight", 9.0, 14.0)
    eng.deploy_cd[i] = 0.0
    for _ in range(300):
        eng.step()
        if not eng.alive[i]:
            break
        if abs(float(eng.y[i]) - A.RIVER_Y) < 0.5:
            assert min(abs(float(eng.x[i]) - bx) for bx in A.BRIDGE_X) <= \
                A.BRIDGE_HALF_W + 0.6, "crossed the river off-bridge"


def test_air_troop_flies_straight_over_the_river():
    eng = Engine(seed=7)
    i = spawn(eng, 0, "Minions", 9.0, 13.0)
    eng.deploy_cd[i] = 0.0
    run(eng, 6)
    assert eng.alive[i]
    assert float(eng.y[i]) > A.RIVER_Y, "air units ignore the river"
    assert abs(float(eng.x[i]) - 9.0) < 3.0, "and do not detour to a bridge"


# ---------------------------------------------------------------- targeting

def test_buildings_only_unit_ignores_troops():
    eng = Engine(seed=8)
    giant = spawn(eng, 0, "Giant", 3.5, 12.0)
    knight = spawn(eng, 1, "Knight", 3.5, 13.0)
    eng.deploy_cd[giant] = eng.deploy_cd[knight] = 0.0
    eng.step()
    tgt = int(eng.target[giant])
    assert tgt != knight
    assert eng.is_building[tgt], "Giant targets buildings only"


def test_ground_only_attacker_cannot_hit_air():
    # Pin the units and silence the towers, so the only thing that can deal
    # damage here is the two units. Left free, they drift into a crown tower's
    # range and it -- not the Knight -- kills the minions.
    eng = Engine(seed=9)
    knight = spawn(eng, 0, "Knight", 9.0, 14.0)
    minions = spawn(eng, 1, "Minions", 9.0, 14.6)
    eng.deploy_cd[:] = 0.0
    eng.speed[:] = 0.0
    for o in (0, 1):
        for ti in eng.tower_idx[o]:
            eng.active[ti] = False
    hp0 = float(eng.hp[minions])
    run(eng, 3)
    assert int(eng.target[knight]) != minions, "air units are not valid targets"
    assert float(eng.hp[minions]) == pytest.approx(hp0), "Knight cannot hit air"
    assert float(eng.hp[knight]) < float(eng.max_hp[knight]), "but air hits back"


def test_troops_divert_onto_enemies_within_sight():
    eng = Engine(seed=10)
    knight = spawn(eng, 0, "Knight", 3.5, 10.0)
    foe = spawn(eng, 1, "Knight", 4.0, 11.0)
    eng.deploy_cd[knight] = eng.deploy_cd[foe] = 0.0
    eng.step()
    assert int(eng.target[knight]) == foe, "nearby troop takes priority over towers"


def test_tower_only_shoots_what_is_in_range():
    eng = Engine(seed=11)
    tower = eng.tower_idx[1][1]
    far = spawn(eng, 0, "Knight", 3.5, 12.0)
    eng.deploy_cd[far] = 0.0
    eng.step()
    d = math.hypot(float(eng.x[tower] - eng.x[far]), float(eng.y[tower] - eng.y[far]))
    assert d > A.PRINCESS_RANGE
    assert int(eng.target[tower]) == -1, "out-of-range troops are not targeted"


# ------------------------------------------------------------------- combat

def test_splash_damages_a_cluster():
    eng = Engine(seed=12)
    valk = spawn(eng, 0, "Valkyrie", 9.0, 8.0)
    a = spawn(eng, 1, "Skeletons", 9.0, 8.6)
    eng.deploy_cd[:] = 0.0
    hps = [float(h) for h in eng.hp[np.flatnonzero(
        eng.alive & (eng.owner == 1) & ~eng.is_building)]]
    run(eng, 3)
    remaining = np.flatnonzero(eng.alive & (eng.owner == 1) & ~eng.is_building)
    assert len(remaining) < len(hps), "splash should clear multiple skeletons"


def test_spells_do_reduced_damage_to_towers():
    eng = Engine(seed=13)
    eng.elixir[0] = 10.0
    cid = CARD_INDEX["Fireball"]
    eng.hands[0][0] = cid
    tower = eng.tower_idx[1][1]
    hp0 = float(eng.hp[tower])
    assert eng.play(0, 0, float(eng.x[tower]), float(eng.y[tower]))
    run(eng, 1.0)
    dealt = hp0 - float(eng.hp[tower])
    full = CARDS[cid].spell_damage
    assert dealt == pytest.approx(full * CARDS[cid].spell_tower_mult, rel=0.02)
    assert dealt < full


def test_spell_hits_troops_at_full_damage():
    eng = Engine(seed=14)
    eng.elixir[0] = 10.0
    eng.hands[0][0] = CARD_INDEX["Fireball"]
    foe = spawn(eng, 1, "Archers", 9.0, 20.0)
    hp0 = float(eng.hp[foe])
    assert eng.play(0, 0, 9.0, 20.0)
    run(eng, 0.6)
    assert not eng.alive[foe] or float(eng.hp[foe]) <= hp0 - 200


# ------------------------------------------------------- towers and outcomes

def test_king_starts_inactive_and_wakes_on_damage():
    eng = Engine(seed=15)
    king = eng.tower_idx[0][0]
    assert not eng.active[king]
    assert not eng.king_active[0]
    eng._damage(eng.tower_idx[0][1], 50.0, 1)
    assert eng.king_active[0] and eng.active[king]


def test_princess_tower_death_awards_a_crown():
    eng = Engine(seed=16)
    eng.hp[eng.tower_idx[1][1]] = 0.0
    eng._reap()
    assert eng.crowns[0] == 1
    assert not eng.done, "one princess tower does not end the match"


def test_king_tower_death_ends_the_match():
    eng = Engine(seed=17)
    eng.hp[eng.tower_idx[1][0]] = 0.0
    eng._reap()
    assert eng.done and eng.winner == 0


def test_crown_lead_wins_at_regular_time():
    eng = Engine(seed=18)
    eng.hp[eng.tower_idx[1][1]] = 0.0
    eng._reap()
    eng.t = A.T_REGULAR
    eng._check_end()
    assert eng.done and eng.winner == 0


def test_level_score_goes_to_overtime_then_hp_tiebreak():
    eng = Engine(seed=19)
    eng.t = A.T_REGULAR
    eng._check_end()
    assert eng.sudden_death and not eng.done, "tied crowns -> overtime"

    eng.hp[eng.tower_idx[1][1]] *= 0.5            # player 1 tower is weaker
    eng.t = A.T_REGULAR + A.T_OVERTIME
    eng._check_end()
    assert eng.done and eng.winner == 0


def test_sudden_death_first_tower_decides():
    eng = Engine(seed=20)
    eng.t = A.T_REGULAR
    eng._check_end()
    assert eng.sudden_death
    eng.hp[eng.tower_idx[0][2]] = 0.0
    eng._reap()
    assert eng.done and eng.winner == 1


def test_building_expires_after_its_lifetime():
    eng = Engine(seed=21)
    i = spawn(eng, 0, "Cannon", 9.0, 8.0)
    life = CARDS[CARD_INDEX["Cannon"]].lifetime
    run(eng, life - 2)
    assert eng.alive[i]
    run(eng, 3)
    assert not eng.alive[i], "buildings despawn when their lifetime ends"


def test_tower_health_fraction_is_crown_weighted():
    eng = Engine(seed=22)
    assert eng.total_tower_frac(0) == pytest.approx(1.0)
    eng.hp[eng.tower_idx[0][0]] = 0.0            # king counts double
    eng._reap()
    assert eng.total_tower_frac(0) == pytest.approx(0.5)


# --------------------------------------------------------------- determinism

def test_same_seed_gives_identical_matches():
    from cr.bots import make_bot
    from rl.evaluate import play_match

    def go():
        return play_match(make_bot("heuristic", seed=1),
                          make_bot("rush", seed=2), seed=99)[0]

    assert go() == go()
