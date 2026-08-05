# clash-royale-rl

A Clash Royale battle simulator plus a self-play reinforcement learning agent
that starts from zero, plays itself millions of times, and keeps getting better.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q       # 60 tests
PYTHONPATH=. .venv/bin/python train.py --workers 8 --device mps --iters 400
PYTHONPATH=. .venv/bin/python watch.py --a runs/<run>/latest.pt --b heuristic
PYTHONPATH=. .venv/bin/python evaluate.py --ckpt runs/<run>/latest.pt --games 40
```

## Read this first: what this does and does not do

**It learns to play a simulation of Clash Royale, not the real app.** The agent
trains against a model of the game implemented in `cr/`, and never touches
Supercell's servers or a phone. That is a deliberate choice, for three reasons:

1. **There is no API for playing.** Supercell's official API is read-only
   (player profiles, battle logs, cards). Nothing in it can place a card.
2. **Driving the real client is against the rules.** Supercell's terms forbid
   automating gameplay, and doing it risks a permanent ban on your account. This
   project does not include screen-capture or input-injection code, and adding it
   would put your account at risk.
3. **It could not learn anyway.** The agent below needs millions of matches. A
   phone plays about two matches a minute, so one million matches is roughly a
   year of continuous play. In simulation the same volume takes a few hours.

So the honest framing is: this is a *from-scratch RL project on a Clash Royale-like
game*, in the spirit of how AlphaZero learns chess. The strategy it discovers
(elixir trading, defending before counter-pushing, spell value, lane pressure) is
real strategy, and transfers to how you think about the actual game. The trained
weights do not transfer to a device.

## Does it actually improve?

Two independent measures, because each one alone can lie.

**Elo against the league** is the relative measure. It is what self-play
optimises, and on its own it is not evidence of anything — a policy can chase its
own tail in circles, beating last week's version while getting worse in absolute
terms.

**Win rate against fixed scripted bots** is the absolute measure. The bots never
change, so this cannot be gamed. `noop` < `random` < `rush` < `heuristic` is a
verified transitive skill ladder (`evaluate.py --bots-only`):

|            | noop | random | rush | heuristic |
|------------|------|--------|------|-----------|
| random     | 1.00 |   —    | 0.05 | 0.00 |
| rush       | 1.00 | 1.00   |  —   | 0.10 |
| heuristic  | 1.00 | 0.95   | 0.90 |  —   |

The `random` bot is Elo-anchored at 0 so the league's ratings have an absolute
scale rather than drifting as a group. Both numbers are written to
`runs/<name>/log.csv` every iteration.

## How the learning works

**PPO** (clipped surrogate, GAE) on an actor-critic conv net, trained by
**league self-play**.

**The league** (`rl/league.py`) is what turns self-play into steady improvement
rather than a cycle. Playing only the newest copy of yourself is unstable:
policies drift, and skills that beat older styles get forgotten. So the pool
holds frozen historical snapshots plus the scripted bots, and opponents are drawn
by **Prioritised Fictitious Self-Play** — weight `(1 - P(win))²`, which puts the
sampling mass on opponents the learner is currently *losing* to, because that is
where the gradient is. A fixed fraction of games go to the newest snapshot (a
stable learning signal) and to the scripted bots (so absolute skill is never
lost). The first snapshot is never evicted, as a permanent historical anchor.

**Learning from its mistakes** is concretely: every match ends in a win or loss
that flows back as a terminal reward, potential-based shaping on tower-health
difference gives per-tick credit for the damage trade, and PFSP then routes the
agent toward exactly the opponents currently beating it.

### Decision points, not ticks (the thing that made it work)

For most of a match no card in hand is affordable, so **the only legal action is
"hold"**. Storing those ticks fills the batch with states that have a single
legal action: no policy gradient, no entropy, and a wasted network forward. In the
first version ~75% of the batch was such states, mean entropy read 0.28, and KL
sat at 0.0 — it looked exactly like a collapsed policy but was actually a
degenerate action space.

So a transition is stored **only where a real choice exists**, and the rewards of
the forced ticks that follow are accumulated into it. That makes it a semi-MDP: a
transition spanning `k` ticks carries reward `Σ γʲrⱼ` and bootstraps with `γᵏ`,
tracked per transition in `rl/rollout.py` and consumed by `compute_gae`. Entropy
became meaningful (4.4 against `ln(93) ≈ 4.5` uniform), and both learner and
opponent skip inference entirely at forced ticks.

### Observation and action space

Both seats observe a **canonical frame** in which they always push toward `+y`
(player 1's view is the board rotated 180°), so one network plays either side.

- **Spatial** `[11, 32, 18]`: friendly/enemy troop HP, DPS and air presence,
  friendly/enemy buildings, unit-count density, and the legal deployment mask.
- **Scalar** `[91]`: elixir and rate band, clock, overtime flag, all six tower
  HP fractions, the 4-card hand (one-hot + cost + affordability), next card in
  cycle, crowns.
- **Actions** `577 = 1 + 4 × 16 × 9`: hold, or (hand slot × placement cell) on a
  2-tile grid. The head is a 1×1 conv over the placement grid, so the action space
  stays spatially structured instead of being flattened into a dense layer.
  Illegal combinations are masked before the softmax rather than punished.

## Layout

```
cr/                the game
  arena.py         geometry, clock, elixir bands, tower stats
  cards.py         card table (level-11 stats), default deck
  engine.py        the simulator: pathing, targeting, combat, win conditions
  env.py           RL wrapper: canonical obs, masked actions, reward
  bots.py          scripted baselines (noop/random/rush/heuristic)
  render.py        terminal renderer
rl/                the learner
  net.py           actor-critic conv net with a spatial placement head
  ppo.py           PPO update
  league.py        opponent pool, Elo, PFSP sampling
  rollout.py       semi-MDP decision-point collector (reference implementation)
  parallel.py      multiprocess collector (workers own env shards + league clones)
  evaluate.py      head-to-head scoring
train.py  evaluate.py  watch.py       CLIs
tests/             60 tests
```

## What the simulator models

Elixir economy and its 1x/2x/3x rate bands, 8-card cycling decks with a hand of
4, continuous unit movement, **ground pathing constrained to the two bridges**,
aggro/sight retargeting, buildings-only targeting (Giant walks past defenders),
air/ground immunity, splash, spells at reduced damage to crown towers,
king-tower activation, deployment-zone unlocking when a princess tower falls,
crown-count victory, and sudden-death overtime with an HP tiebreak.

Not modelled: unit collision mass and push, charge attacks, death spawns,
shield/rage/clone mechanics, card levels other than tournament standard.

Stats in `cr/cards.py` approximate level-11 tournament values and are data-only,
so decks and balance can be changed without touching the engine. The 14 cards
present cover a coherent beatdown deck; both players use the same one by default
because a mirror match removes deck asymmetry as a confound while the policy
learns.

## Performance

Measured on an M-series Mac (10 cores):

| | throughput |
|---|---|
| raw simulation, single env | ~4,980 ticks/s |
| in-process collector | ~246 decisions/s |
| 8 workers + MPS update | ~750 decisions/s |

Simulation is ~75% of an iteration and is pure Python/numpy, so it is what gets
parallelised: each worker owns a shard of environments *and* a clone of the
league, and runs the ordinary single-process collector. Only learner parameters
and a compact Elo table go out each iteration; frozen snapshot weights are
re-broadcast only when the pool changes. Elo stays authoritative in the main
process. The PPO update runs ~3x faster on MPS than CPU, so
`--workers 8 --device mps` is the recommended combination.

`--workers 0` keeps everything in-process, which is the reference implementation
and the one to debug against.

## Useful flags

```bash
--workers N          rollout worker processes (0 = in-process)
--device mps|cpu     device for the PPO update; rollouts always run on cpu
--iters / --n-envs / --horizon
--snapshot-every K   freeze a policy copy into the opponent pool every K iters
--p-bot / --p-latest fraction of games vs scripted bots / newest snapshot
--eval-every K       absolute win-rate check against every scripted bot
--resume runs/<r>/latest.pt      restores net, optimiser and league
```

Runs are resumable and Ctrl-C saves cleanly (press once to stop after the current
iteration).

## Where to take it next

- **Deck choice as part of the policy** — currently a fixed mirror deck. Letting
  the agent pick 8 of the 14 cards makes deck-building itself learnable.
- **Hidden information.** `perfect_info=True` shows the agent the opponent's
  elixir, which real players must infer. Turning it off makes the problem
  properly imperfect-information and is a one-line change in `cr/env.py`.
- **More cards**, especially death-spawn and charge units, which need engine work.
- **Learn from real battle logs.** Supercell's read-only API does expose battle
  logs; behaviour-cloning a prior from human games before self-play is the
  standard way to skip the slow early phase.
