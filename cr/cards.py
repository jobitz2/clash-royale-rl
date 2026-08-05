"""Card definitions.

Stats approximate tournament-standard (level 11) Clash Royale values. They are
deliberately data-only so decks can be swapped without touching the engine.

Speed bands (tiles/second): slow 0.7, medium 0.9, fast 1.2, very fast 1.5.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


TROOP = 0
SPELL = 1
BUILDING = 2


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    kind: int = TROOP

    # Spawn layout
    count: int = 1
    spread: float = 0.0          # radius units are scattered over on deploy

    # Combat
    hp: float = 0.0
    dmg: float = 0.0
    hit_interval: float = 1.0
    attack_range: float = 1.0
    splash: float = 0.0          # >0 damages everything within this radius of target

    # Movement / body
    speed: float = 0.0
    radius: float = 0.5
    is_air: bool = False

    # Targeting
    hits_ground: bool = True
    hits_air: bool = False
    buildings_only: bool = False

    deploy_time: float = 1.0     # seconds before the unit can act
    lifetime: float = 0.0        # >0 = despawns (buildings)

    # Spell-only
    spell_damage: float = 0.0
    spell_radius: float = 0.0
    spell_tower_mult: float = 1.0   # spells do reduced damage to crown towers

    @property
    def dps(self) -> float:
        return self.dmg / self.hit_interval if self.hit_interval else 0.0


CARDS: List[Card] = [
    Card("Knight", 3, TROOP, hp=1900, dmg=220, hit_interval=1.2, attack_range=1.2,
         speed=0.9, radius=0.6),
    Card("Archers", 3, TROOP, count=2, spread=0.6, hp=304, dmg=107,
         hit_interval=1.2, attack_range=5.0, speed=0.9, radius=0.45,
         hits_air=True),
    Card("Giant", 5, TROOP, hp=4091, dmg=261, hit_interval=1.5, attack_range=1.2,
         speed=0.7, radius=0.85, buildings_only=True),
    Card("Musketeer", 4, TROOP, hp=792, dmg=218, hit_interval=1.0,
         attack_range=6.0, speed=0.9, radius=0.5, hits_air=True),
    Card("Minions", 3, TROOP, count=3, spread=0.7, hp=203, dmg=93,
         hit_interval=1.0, attack_range=1.6, speed=1.2, radius=0.4,
         is_air=True, hits_air=True),
    Card("Skeletons", 1, TROOP, count=4, spread=0.7, hp=81, dmg=81,
         hit_interval=1.0, attack_range=0.9, speed=1.2, radius=0.35),
    Card("MiniPekka", 4, TROOP, hp=1361, dmg=712, hit_interval=1.8,
         attack_range=1.2, speed=1.2, radius=0.6),
    Card("Valkyrie", 4, TROOP, hp=2145, dmg=266, hit_interval=1.5,
         attack_range=1.3, splash=1.6, speed=0.9, radius=0.65),
    Card("BabyDragon", 4, TROOP, hp=1152, dmg=160, hit_interval=1.6,
         attack_range=3.5, splash=1.4, speed=0.9, radius=0.6,
         is_air=True, hits_air=True),
    Card("Cannon", 3, BUILDING, hp=896, dmg=212, hit_interval=0.9,
         attack_range=5.5, radius=0.7, lifetime=30.0, deploy_time=1.0),
    Card("Tesla", 4, BUILDING, hp=1017, dmg=190, hit_interval=1.1,
         attack_range=5.5, radius=0.7, lifetime=40.0, hits_air=True),
    Card("Fireball", 4, SPELL, spell_damage=689, spell_radius=2.5,
         spell_tower_mult=0.35, deploy_time=0.5),
    Card("Arrows", 3, SPELL, spell_damage=291, spell_radius=4.0,
         spell_tower_mult=0.35, deploy_time=0.4),
    Card("Zap", 2, SPELL, spell_damage=192, spell_radius=2.5,
         spell_tower_mult=0.35, deploy_time=0.2),
]

CARD_INDEX = {c.name: i for i, c in enumerate(CARDS)}
NUM_CARDS = len(CARDS)

# A coherent 8-card beatdown deck. Both players use it by default: a mirror
# match removes deck asymmetry as a confound while the policy is learning.
DEFAULT_DECK: Tuple[str, ...] = (
    "Giant", "Knight", "Archers", "Musketeer",
    "Minions", "Skeletons", "Fireball", "Arrows",
)


def deck_ids(names=DEFAULT_DECK) -> List[int]:
    return [CARD_INDEX[n] for n in names]


def avg_cost(names=DEFAULT_DECK) -> float:
    return sum(CARDS[CARD_INDEX[n]].cost for n in names) / len(names)
