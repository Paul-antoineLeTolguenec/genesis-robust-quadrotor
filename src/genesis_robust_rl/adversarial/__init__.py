"""Adversarial training module: protocols, agents, and training loop."""

from genesis_robust_rl.adversarial.curriculum import CurriculumScheduler
from genesis_robust_rl.adversarial.networks import ActorCritic
from genesis_robust_rl.adversarial.ppo_agent import PPOAgent
from genesis_robust_rl.adversarial.protocol import AdversarialAgent, DroneAgent, RolloutData
from genesis_robust_rl.adversarial.training_loop import TrainConfig, TrainingMode, train

__all__ = [
    "ActorCritic",
    "AdversarialAgent",
    "CurriculumScheduler",
    "DroneAgent",
    "PPOAgent",
    "RolloutData",
    "TrainConfig",
    "TrainingMode",
    "train",
]
