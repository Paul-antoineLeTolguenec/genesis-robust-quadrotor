"""Tests for PPOAgent."""

from __future__ import annotations

import torch

from genesis_robust_rl.adversarial.ppo_agent import PPOAgent
from genesis_robust_rl.adversarial.protocol import RolloutData


def _make_rollout(t: int = 16, n: int = 4, obs_dim: int = 6, act_dim: int = 2) -> RolloutData:
    """Create a simple rollout with positive rewards."""
    return RolloutData(
        obs=torch.randn(t, n, obs_dim),
        actions=torch.randn(t, n, act_dim),
        rewards=torch.ones(t, n),
        dones=torch.zeros(t, n),
        log_probs=torch.randn(t, n),
        values=torch.zeros(t, n),
        last_value=torch.zeros(n),
    )


class TestPPOAgent:
    """PPO agent tests."""

    def test_act_shapes(self):
        agent = PPOAgent(obs_dim=6, action_dim=2)
        obs = torch.randn(4, 6)
        action, log_prob, value = agent.act(obs)
        assert action.shape == (4, 2)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)

    def test_update_returns_metrics(self):
        agent = PPOAgent(obs_dim=6, action_dim=2, n_epochs=2, mini_batch_size=16)
        rollout = _make_rollout()
        metrics = agent.update(rollout)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_update_loss_finite(self):
        agent = PPOAgent(obs_dim=6, action_dim=2, n_epochs=2, mini_batch_size=16)
        rollout = _make_rollout()
        metrics = agent.update(rollout)
        for k, v in metrics.items():
            assert torch.isfinite(torch.tensor(v)), f"{k} is not finite: {v}"

    def test_value_improves_after_update(self):
        """Value estimate should improve after training on constant reward."""
        agent = PPOAgent(obs_dim=4, action_dim=1, n_epochs=5, mini_batch_size=8, lr=1e-3)
        rollout = _make_rollout(t=32, n=4, obs_dim=4, act_dim=1)

        # Initial value estimate
        with torch.no_grad():
            _, _, val_before = agent.act(rollout.obs[0])

        # Train
        for _ in range(5):
            agent.update(rollout)

        # After training
        with torch.no_grad():
            _, _, val_after = agent.act(rollout.obs[0])

        # Value should be closer to discounted reward (positive)
        assert val_after.mean() > val_before.mean()

    def test_gradient_flows(self):
        """Parameters change after update."""
        agent = PPOAgent(obs_dim=4, action_dim=1, n_epochs=1, mini_batch_size=64)
        params_before = {name: p.clone() for name, p in agent.network.named_parameters()}

        rollout = _make_rollout(t=64, n=4, obs_dim=4, act_dim=1)
        agent.update(rollout)

        changed = False
        for name, p in agent.network.named_parameters():
            if not torch.equal(p, params_before[name]):
                changed = True
                break
        assert changed, "No parameters changed after update"
