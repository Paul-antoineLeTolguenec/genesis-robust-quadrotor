"""Tests for agent protocol and RolloutData."""

from __future__ import annotations

import torch

from genesis_robust_rl.adversarial.ppo_agent import PPOAgent
from genesis_robust_rl.adversarial.protocol import AdversarialAgent, RolloutData


class TestProtocol:
    """Protocol satisfaction and RolloutData tests."""

    def test_ppo_agent_satisfies_protocol(self):
        agent = PPOAgent(obs_dim=4, action_dim=2)
        assert isinstance(agent, AdversarialAgent)

    def test_rollout_data_creation(self):
        t, n = 10, 4
        data = RolloutData(
            obs=torch.zeros(t, n, 6),
            actions=torch.zeros(t, n, 2),
            rewards=torch.zeros(t, n),
            dones=torch.zeros(t, n),
            log_probs=torch.zeros(t, n),
            values=torch.zeros(t, n),
            last_value=torch.zeros(n),
        )
        assert data.obs.shape == (t, n, 6)
        assert data.last_value.shape == (n,)

    def test_rollout_data_shapes_consistent(self):
        t, n = 5, 8
        data = RolloutData(
            obs=torch.randn(t, n, 3),
            actions=torch.randn(t, n, 1),
            rewards=torch.randn(t, n),
            dones=torch.zeros(t, n),
            log_probs=torch.randn(t, n),
            values=torch.randn(t, n),
            last_value=torch.randn(n),
        )
        assert data.rewards.shape[0] == data.obs.shape[0]
        assert data.rewards.shape[1] == data.obs.shape[1]
