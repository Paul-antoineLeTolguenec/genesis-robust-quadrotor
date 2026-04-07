"""Agent protocol and rollout data container for adversarial training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class AdversarialAgent(Protocol):
    """Minimal interface for any agent in the minimax loop.

    Satisfied by PPOAgent (built-in) or any user-provided agent.
    act() is always called in inference mode; update() in training mode.
    """

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Select action from observation.

        Args:
            obs: Tensor[n_envs, obs_dim]

        Returns:
            action:   Tensor[n_envs, action_dim]
            log_prob: Tensor[n_envs]
            value:    Tensor[n_envs]
        """
        ...

    def update(self, rollout: RolloutData) -> dict[str, float]:
        """Run one gradient update on collected rollout data.

        Returns dict of scalar metrics (e.g. {"policy_loss": ..., "value_loss": ...}).
        """
        ...


# Semantic alias — same interface, distinct name for type clarity
DroneAgent = AdversarialAgent


@dataclass
class RolloutData:
    """Flat batch of rollout experience collected during one rollout phase."""

    obs: Tensor  # [T, n_envs, obs_dim]
    actions: Tensor  # [T, n_envs, action_dim]
    rewards: Tensor  # [T, n_envs]
    dones: Tensor  # [T, n_envs]
    log_probs: Tensor  # [T, n_envs]
    values: Tensor  # [T, n_envs]
    last_value: Tensor  # [n_envs] — bootstrap value for GAE
