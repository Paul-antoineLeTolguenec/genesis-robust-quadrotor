"""Shared actor-critic MLP network for PPO agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Shared-trunk MLP with Gaussian policy head and value head."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Policy head: mean
        self.policy_mean = nn.Linear(in_dim, action_dim)
        # Learnable log_std (per action dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Value head
        self.value_head = nn.Linear(in_dim, 1)

    def _forward_trunk(self, obs: Tensor) -> Tensor:
        return self.trunk(obs)

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for rollout collection.

        Returns (action, log_prob, value). Action sampled from Gaussian.
        """
        features = self._forward_trunk(obs)
        mean = self.policy_mean(features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for PPO update.

        Returns (log_prob, entropy, value) for the given (obs, action) pair.
        """
        features = self._forward_trunk(obs)
        mean = self.policy_mean(features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(features).squeeze(-1)
        return log_prob, entropy, value
