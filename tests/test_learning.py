"""Tests for the env/action plumbing and the learning stack.

The point of these is that a silent mismatch between the action head, the legal
mask and the engine would still "train" -- the loss would go down while the agent
played nonsense. Each of these pins one link in that chain.
"""

import numpy as np
import pytest
import torch

from cr import arena as A
from cr.cards import CARD_INDEX, CARDS, SPELL
from cr.env import (CRDuel, N_ACTIONS, N_PLACE, PH, PLACE_STRIDE, PW,
                    RewardConfig, SCALAR_DIM, N_CHANNELS)
from rl.league import League, expected_score
from rl.net import ActorCritic
from rl.ppo import PPO, PPOConfig
from rl.rollout import SelfPlayCollector


# ------------------------------------------------------------ action mapping

def test_action_space_size_matches_the_grid():
    assert N_ACTIONS == 1 + 4 * PH * PW
    assert PH == A.H // PLACE_STRIDE and PW == A.W // PLACE_STRIDE


@pytest.mark.parametrize("p", [0, 1])
def test_encode_decode_roundtrip(p):
    for idx in range(1, N_ACTIONS):
        slot, x, y = CRDuel.decode(p, idx)
        assert 0 <= slot < 4
        assert CRDuel.encode(p, slot, x, y) == idx


def test_hold_action_decodes_to_no_play():
    assert CRDuel.decode(0, 0)[0] == -1
    assert CRDuel.encode(0, -1, 5.0, 5.0) == 0


def test_player1_actions_map_into_its_own_half():
    """A placement legal for player 1 must land on player 1's side."""
    env = CRDuel(seed=0)
    env.reset(seed=0)
    mask = env.mask(1)
    troop_slots = [s for s in range(4)
                   if CARDS[env.eng.hands[1][s]].kind != SPELL]
    assert troop_slots
    checked = 0
    for idx in np.flatnonzero(mask):
        if idx == 0:
            continue
        slot, x, y = CRDuel.decode(1, int(idx))
        if slot in troop_slots:
            assert y > A.RIVER_Y, "player 1 deploys on the far side of the river"
            checked += 1
    assert checked > 0


def test_mask_forbids_unaffordable_slots():
    env = CRDuel(seed=1)
    env.reset(seed=1)
    env.eng.elixir[0] = 0.0
    mask = env.mask(0)
    assert mask[0], "holding is always legal"
    assert mask.sum() == 1, "with no elixir there is nothing else to do"
    assert not env.can_act(0)


def test_can_act_agrees_with_the_mask():
    """`can_act` is the cheap gate used to skip inference; it must not lie."""
    env = CRDuel(seed=2)
    env.reset(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(400):
        for p in (0, 1):
            assert env.can_act(p) == (int(env.mask(p).sum()) > 1)
        a = []
        for p in (0, 1):
            if env.can_act(p) and rng.random() < 0.3:
                a.append(int(rng.choice(np.flatnonzero(env.mask(p)))))
            else:
                a.append(0)
        _, done, _ = env.step(a[0], a[1])
        if done:
            break


def test_illegal_action_is_refused_by_the_engine():
    env = CRDuel(seed=3)
    env.reset(seed=3)
    env.eng.elixir[0] = 10.0
    # A troop placed deep in the enemy half must be rejected.
    slot = next(s for s in range(4) if CARDS[env.eng.hands[0][s]].kind != SPELL)
    assert not env.eng.play(0, slot, 9.0, A.H - 4.0)


def test_playing_reduces_elixir_through_the_env_step():
    env = CRDuel(seed=4)
    env.reset(seed=4)
    env.eng.elixir[0] = 10.0
    slot = next(s for s in range(4) if CARDS[env.eng.hands[0][s]].kind != SPELL)
    idx = env.encode(0, slot, 9.0, 5.0)
    assert env.mask(0)[idx]
    before = env.eng.elixir[0]
    _, _, info = env.step(idx, 0)
    assert info["played"][0]
    assert env.eng.elixir[0] < before


# ------------------------------------------------------------- observations

def test_observation_shapes():
    env = CRDuel(seed=5)
    env.reset(seed=5)
    o = env.obs(0)
    assert o["spatial"].shape == (N_CHANNELS, A.H, A.W)
    assert o["scalar"].shape == (SCALAR_DIM,)
    assert np.isfinite(o["spatial"]).all() and np.isfinite(o["scalar"]).all()


def test_observation_is_seat_canonical():
    """Both seats see their own towers in the same place."""
    env = CRDuel(seed=6)
    env.reset(seed=6)
    o0, o1 = env.obs(0), env.obs(1)
    own0 = o0["spatial"][6]        # friendly buildings channel
    own1 = o1["spatial"][6]
    assert own0.sum() > 0 and own1.sum() > 0
    # Friendly towers must sit in the lower rows of each seat's own view.
    assert own0[: A.H // 2].sum() > own0[A.H // 2:].sum()
    assert own1[: A.H // 2].sum() > own1[A.H // 2:].sum()
    np.testing.assert_allclose(own0, own1, atol=1e-6)


def test_enemy_channels_are_mirror_images_at_reset():
    env = CRDuel(seed=7)
    env.reset(seed=7)
    e0 = env.obs(0)["spatial"][7]
    e1 = env.obs(1)["spatial"][7]
    np.testing.assert_allclose(e0, e1, atol=1e-6)


def test_obs_cache_refreshes_after_a_step():
    env = CRDuel(seed=8)
    env.reset(seed=8)
    first = env.obs(0)["scalar"].copy()
    for _ in range(30):
        env.step(0, 0)
    assert not np.allclose(first, env.obs(0)["scalar"]), "cache must not go stale"


# ------------------------------------------------------------------- rewards

def test_reward_is_zero_sum_on_shaping_and_outcome():
    env = CRDuel(seed=9, reward=RewardConfig(w_leak=0.0, w_step=0.0))
    env.reset(seed=9)
    rng = np.random.default_rng(1)
    total = [0.0, 0.0]
    for _ in range(2000):
        a = []
        for p in (0, 1):
            if env.can_act(p) and rng.random() < 0.3:
                a.append(int(rng.choice(np.flatnonzero(env.mask(p)))))
            else:
                a.append(0)
        rew, done, _ = env.step(a[0], a[1])
        total[0] += rew[0]
        total[1] += rew[1]
        if done:
            break
    assert total[0] + total[1] == pytest.approx(0.0, abs=1e-3)


def test_damaging_the_enemy_tower_is_rewarded():
    env = CRDuel(seed=10)
    env.reset(seed=10)
    env.eng.hp[env.eng.tower_idx[1][1]] *= 0.5
    rew, _, _ = env.step(0, 0)
    assert rew[0] > 0 and rew[1] < 0


def test_winner_gets_the_terminal_bonus():
    env = CRDuel(seed=11)
    env.reset(seed=11)
    env.eng.hp[env.eng.tower_idx[1][0]] = 1.0     # one hit from losing
    done = False
    for _ in range(2000):
        rew, done, info = env.step(0, 0)
        if done:
            break
    assert done and info["winner"] == 0
    assert rew[0] > 0 > rew[1]


# --------------------------------------------------------------------- model

def test_network_output_shapes():
    net = ActorCritic()
    sp = torch.zeros(3, N_CHANNELS, A.H, A.W)
    sc = torch.zeros(3, SCALAR_DIM)
    logits, value = net(sp, sc)
    assert logits.shape == (3, N_ACTIONS)
    assert value.shape == (3,)


def test_masked_actions_get_no_probability():
    net = ActorCritic()
    sp = torch.randn(2, N_CHANNELS, A.H, A.W)
    sc = torch.randn(2, SCALAR_DIM)
    mask = torch.zeros(2, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 5] = True
    with torch.no_grad():
        logits, _ = net(sp, sc)
        dist = net._masked_dist(logits, mask)
        probs = dist.probs
        assert probs[:, 0].min() > 0
        illegal = probs.clone()
        illegal[:, [0, 5]] = 0.0
        assert float(illegal.max()) < 1e-8
        assert torch.isfinite(dist.entropy()).all(), "entropy must not be NaN"


def test_sampled_actions_are_always_legal():
    net = ActorCritic()
    env = CRDuel(seed=12)
    env.reset(seed=12)
    mask = env.mask(0)
    sp = torch.as_tensor(env.obs(0)["spatial"][None])
    sc = torch.as_tensor(env.obs(0)["scalar"][None])
    mk = torch.as_tensor(mask[None])
    for _ in range(50):
        a, _, _ = net.act(sp, sc, mk)
        assert mask[int(a[0])], "sampled an illegal action"


def test_entropy_at_init_is_near_uniform():
    """Guards against an init that silently starts the policy near-deterministic."""
    net = ActorCritic()
    env = CRDuel(seed=13)
    env.reset(seed=13)
    mask = env.mask(0)
    sp = torch.as_tensor(env.obs(0)["spatial"][None])
    sc = torch.as_tensor(env.obs(0)["scalar"][None])
    with torch.no_grad():
        logits, _ = net(sp, sc)
        ent = float(net._masked_dist(logits, torch.as_tensor(mask[None])).entropy())
    assert ent == pytest.approx(np.log(mask.sum()), rel=0.02)


# --------------------------------------------------------------------- league

def test_expected_score_is_symmetric_and_monotone():
    assert expected_score(0, 0) == pytest.approx(0.5)
    assert expected_score(400, 0) == pytest.approx(10 / 11, rel=1e-3)
    assert expected_score(0, 400) == pytest.approx(1 / 11, rel=1e-3)


def test_winning_raises_learner_elo_and_lowers_the_opponent():
    lg = League(seed=0)
    opp = next(e for e in lg.entries if e.name == "rush")
    before_l, before_o = lg.learner_elo, opp.elo
    lg.record(opp, 1.0)
    assert lg.learner_elo > before_l and opp.elo < before_o


def test_anchor_bot_rating_never_moves():
    lg = League(seed=0)
    anchor = next(e for e in lg.entries if e.anchored)
    for s in (1.0, 0.0, 1.0):
        lg.record(anchor, s)
    assert anchor.elo == 0.0, "the absolute Elo reference must stay fixed"


def test_snapshot_pool_is_capped_but_keeps_an_anchor():
    lg = League(max_snapshots=3, seed=0)
    net = ActorCritic()
    for it in range(8):
        lg.add_snapshot(net.state_dict(), it)
    snaps = lg.snapshots()
    assert len(snaps) == 3
    assert any(e.iteration == 0 for e in snaps), "first snapshot is permanent"
    assert max(e.iteration for e in snaps) == 7, "newest is retained"


def test_pfsp_prefers_opponents_that_beat_the_learner():
    lg = League(max_snapshots=10, p_bot=0.0, p_latest=0.0, seed=0)
    net = ActorCritic()
    lg.add_snapshot(net.state_dict(), 0, elo=-800)     # much weaker
    lg.add_snapshot(net.state_dict(), 1, elo=800)      # much stronger
    lg.learner_elo = 0.0
    picks = [lg.sample().iteration for _ in range(400)]
    assert picks.count(1) > picks.count(0) * 3


def test_league_state_roundtrip():
    lg = League(seed=0)
    net = ActorCritic()
    lg.add_snapshot(net.state_dict(), 5)
    lg.record(lg.entries[2], 1.0)
    st = lg.state()
    lg2 = League(seed=1)
    lg2.load_state(st)
    assert lg2.learner_elo == lg.learner_elo
    assert [e.uid for e in lg2.entries] == [e.uid for e in lg.entries]
    assert lg2.latest().iteration == 5


# ------------------------------------------------------------ semi-MDP / GAE

def test_gae_uses_per_transition_gamma_k():
    """A transition spanning k ticks must bootstrap with gamma^k, not gamma."""
    net = ActorCritic()
    lg = League(seed=0)
    col = SelfPlayCollector(net, lg, n_envs=2, horizon=4, device="cpu", seed=0,
                            gamma=0.9)
    col.b_reward.zero_()
    col.b_done.zero_()
    col.b_value.zero_()
    col.b_gk.fill_(0.9 ** 3)                       # pretend every span is 3 ticks
    col.b_reward[-1] = 1.0
    adv, ret = col.compute_gae(torch.zeros(2), lam=1.0)
    # With zero baselines and reward only at the last step, advantage at step t
    # is (gamma^k * lam)^(T-1-t) * r.
    expected = (0.9 ** 3) ** 3
    assert float(adv[0, 0]) == pytest.approx(expected, rel=1e-5)


def test_done_flag_stops_credit_flowing_across_episodes():
    net = ActorCritic()
    lg = League(seed=0)
    col = SelfPlayCollector(net, lg, n_envs=1, horizon=3, device="cpu", seed=0,
                            gamma=0.99)
    col.b_reward.zero_()
    col.b_value.zero_()
    col.b_done.zero_()
    col.b_gk.fill_(0.99)
    col.b_reward[2, 0] = 1.0
    col.b_done[0, 0] = 1.0                         # episode ended at step 0
    adv, _ = col.compute_gae(torch.zeros(1), lam=1.0)
    assert float(adv[0, 0]) == pytest.approx(0.0), "reward must not leak backwards"
    assert float(adv[1, 0]) > 0


def test_collector_only_stores_real_decision_points():
    net = ActorCritic()
    lg = League(seed=0)
    col = SelfPlayCollector(net, lg, n_envs=4, horizon=16, device="cpu", seed=0)
    col.collect()
    assert bool(col.b_mask.sum(-1).min() > 1), "every stored state had a choice"
    chosen = col.b_action.reshape(-1)
    flat_mask = col.b_mask.reshape(-1, N_ACTIONS)
    assert bool(flat_mask[torch.arange(chosen.numel()), chosen].all()), \
        "every stored action was legal in its own mask"


def test_collector_accumulates_reward_over_skipped_ticks():
    net = ActorCritic()
    lg = League(seed=0)
    col = SelfPlayCollector(net, lg, n_envs=4, horizon=16, device="cpu", seed=0)
    col.collect()
    assert col.total_ticks > 4 * 16, "ticks must outnumber stored decisions"
    assert bool((col.b_gk < 1.0).all()), "each transition spans at least one tick"


# ----------------------------------------------------------------------- PPO

def test_ppo_update_runs_and_reports_finite_stats():
    torch.manual_seed(0)
    net = ActorCritic()
    lg = League(seed=0)
    col = SelfPlayCollector(net, lg, n_envs=4, horizon=16, device="cpu", seed=0)
    ppo = PPO(net, PPOConfig(minibatches=2, epochs=2))
    lv = col.collect()
    info = ppo.update(col.batch(lv))
    for k in ("pg_loss", "v_loss", "entropy", "kl", "clipfrac"):
        assert np.isfinite(info[k]), f"{k} was not finite"
    assert info["entropy"] > 0.5, "policy should not collapse in one update"


def test_advantage_normalisation_zeroes_a_constant_advantage():
    """Documents why the trivial-fit test below must disable normalisation."""
    torch.manual_seed(0)
    net = ActorCritic()
    ppo = PPO(net, PPOConfig(minibatches=1, epochs=2, norm_adv=True,
                             target_kl=1e9))
    n = 8
    mask = torch.ones(n, N_ACTIONS, dtype=torch.bool)
    batch = {
        "spatial": torch.zeros(n, N_CHANNELS, A.H, A.W),
        "scalar": torch.zeros(n, SCALAR_DIM),
        "mask": mask,
        "action": torch.zeros(n, dtype=torch.long),
        "logp": torch.zeros(n),
        "value": torch.zeros(n),
        "adv": torch.ones(n),          # no spread -> nothing to normalise to
        "ret": torch.zeros(n),
    }
    info = ppo.update(batch)
    assert abs(info["pg_loss"]) < 1e-6, "constant advantages give no policy signal"


def test_ppo_can_fit_a_trivial_target():
    """Sanity: with a positive advantage, the chosen action's logprob rises."""
    torch.manual_seed(0)
    net = ActorCritic()
    ppo = PPO(net, PPOConfig(minibatches=1, epochs=8, ent_coef=0.0,
                             target_kl=1e9, norm_adv=False))
    n = 16
    mask = torch.zeros(n, N_ACTIONS, dtype=torch.bool)
    mask[:, :6] = True
    batch = {
        "spatial": torch.zeros(n, N_CHANNELS, A.H, A.W),
        "scalar": torch.zeros(n, SCALAR_DIM),
        "mask": mask,
        "action": torch.full((n,), 3, dtype=torch.long),
        "logp": torch.full((n,), float(np.log(1 / 6))),
        "value": torch.zeros(n),
        "adv": torch.ones(n),
        "ret": torch.ones(n),
    }
    with torch.no_grad():
        before = net.evaluate(batch["spatial"], batch["scalar"], mask,
                              batch["action"])[0].mean()
    ppo.update(batch)
    with torch.no_grad():
        after = net.evaluate(batch["spatial"], batch["scalar"], mask,
                             batch["action"])[0].mean()
    assert float(after) > float(before), "advantaged action should get more likely"
