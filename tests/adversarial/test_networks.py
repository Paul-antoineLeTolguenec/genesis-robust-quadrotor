"""Tests for ActorCritic network."""

from __future__ import annotations

import torch

from genesis_robust_rl.adversarial.networks import ActorCritic


class TestActorCritic:
    """Forward pass shape and behavior tests."""

    def test_act_shapes(self):
        net = ActorCritic(obs_dim=6, action_dim=3)
        obs = torch.randn(4, 6)
        action, log_prob, value = net.act(obs)
        assert action.shape == (4, 3)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)

    def test_evaluate_shapes(self):
        net = ActorCritic(obs_dim=6, action_dim=3)
        obs = torch.randn(4, 6)
        action = torch.randn(4, 3)
        log_prob, entropy, value = net.evaluate(obs, action)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_evaluate_log_prob_finite(self):
        net = ActorCritic(obs_dim=4, action_dim=2)
        obs = torch.randn(8, 4)
        action = torch.randn(8, 2)
        log_prob, _, _ = net.evaluate(obs, action)
        assert torch.isfinite(log_prob).all()

    def test_custom_hidden_sizes(self):
        net = ActorCritic(obs_dim=10, action_dim=4, hidden_sizes=(128, 128, 64))
        obs = torch.randn(2, 10)
        action, _, _ = net.act(obs)
        assert action.shape == (2, 4)

    def test_single_env(self):
        """Works with n_envs=1."""
        net = ActorCritic(obs_dim=3, action_dim=1)
        obs = torch.randn(1, 3)
        action, log_prob, value = net.act(obs)
        assert action.shape == (1, 1)
        assert log_prob.shape == (1,)
