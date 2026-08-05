"""Learning stack: network, PPO, league self-play, evaluation."""

from .net import ActorCritic, build
from .ppo import PPO, PPOConfig
from .league import League, Entry, expected_score
from .rollout import SelfPlayCollector
from .evaluate import NetPolicy, evaluate_vs_bots, head_to_head, play_match

__all__ = [
    "ActorCritic", "build", "PPO", "PPOConfig", "League", "Entry",
    "expected_score", "SelfPlayCollector", "NetPolicy", "evaluate_vs_bots",
    "head_to_head", "play_match",
]
