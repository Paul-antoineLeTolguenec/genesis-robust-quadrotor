"""Tests for the training loop (DR, RARL, RAP modes)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from genesis_robust_rl.adversarial.ppo_agent import PPOAgent
from genesis_robust_rl.adversarial.training_loop import TrainConfig, TrainingMode, train
from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.config import PerturbationConfig
from genesis_robust_rl.perturbations.base import PhysicsPerturbation

# Reuse test infrastructure from envs
from tests.envs.conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


class DummyPhysics(PhysicsPerturbation):
    def apply(self, scene, drone, env_state):
        pass


def _make_perturbation(pid: str, n_envs: int = 4) -> DummyPhysics:
    return DummyPhysics(
        id=pid,
        n_envs=n_envs,
        dt=0.01,
        value_mode="fixed",
        frequency="per_step",
        scope="per_env",
        distribution="uniform",
        distribution_params={"low": -1.0, "high": 1.0},
        bounds=(-1.0, 1.0),
        nominal=0.0,
        dimension=(1,),
        observable=True,
    )


def _make_dr_env(n_envs: int = 4) -> DummyDroneEnv:
    """Plain DR env (no adversarial wrapper)."""
    return DummyDroneEnv(
        scene=_make_mock_scene(),
        drone=_make_mock_drone(n_envs),
        n_envs=n_envs,
        perturbation_cfg=PerturbationConfig(),
    )


def _make_adv_env(n_envs: int = 4) -> AdversarialEnv:
    """Adversarial env with one controllable perturbation."""
    p = _make_perturbation("p1", n_envs)
    cfg = PerturbationConfig(physics=[p])
    env = DummyDroneEnv(
        scene=_make_mock_scene(),
        drone=_make_mock_drone(n_envs),
        n_envs=n_envs,
        perturbation_cfg=cfg,
    )
    return AdversarialEnv(env=env, adversary_targets=["p1"])


class TestDRMode:
    """DR mode training tests."""

    def test_dr_runs_without_crash(self):
        env = _make_dr_env(n_envs=4)
        agent = PPOAgent(obs_dim=0, action_dim=4, n_epochs=1, mini_batch_size=8)
        config = TrainConfig(
            mode=TrainingMode.DR,
            total_timesteps=32,
            rollout_steps=8,
        )
        metrics = train(env, agent, config=config)
        assert "drone_reward" in metrics
        assert len(metrics["drone_reward"]) > 0

    def test_dr_rejects_adversary_agent(self):
        env = _make_dr_env()
        agent = PPOAgent(obs_dim=0, action_dim=4)
        adv = PPOAgent(obs_dim=1, action_dim=1)
        with pytest.raises(ValueError, match="does not accept"):
            train(env, agent, adversary_agent=adv, config=TrainConfig(mode=TrainingMode.DR))

    def test_dr_rejects_adversarial_env(self):
        env = _make_adv_env()
        agent = PPOAgent(obs_dim=0, action_dim=4)
        with pytest.raises(TypeError, match="RobustDroneEnv"):
            train(env, agent, config=TrainConfig(mode=TrainingMode.DR))


class TestRARLMode:
    """RARL mode training tests."""

    def test_rarl_runs_without_crash(self):
        env = _make_adv_env(n_envs=4)
        drone = PPOAgent(obs_dim=0, action_dim=4, n_epochs=1, mini_batch_size=8)
        adversary = PPOAgent(obs_dim=1, action_dim=1, n_epochs=1, mini_batch_size=8)
        config = TrainConfig(
            mode=TrainingMode.RARL,
            total_timesteps=32,
            rollout_steps=8,
        )
        metrics = train(env, drone, adversary_agent=adversary, config=config)
        assert "drone_reward" in metrics
        assert "adv_reward" in metrics

    def test_rarl_requires_adversary(self):
        env = _make_adv_env()
        agent = PPOAgent(obs_dim=0, action_dim=4)
        with pytest.raises(ValueError, match="requires adversary_agent"):
            train(env, agent, config=TrainConfig(mode=TrainingMode.RARL))

    def test_rarl_requires_adversarial_env(self):
        env = _make_dr_env()
        agent = PPOAgent(obs_dim=0, action_dim=4)
        adv = PPOAgent(obs_dim=1, action_dim=1)
        with pytest.raises(TypeError, match="AdversarialEnv"):
            train(env, agent, adversary_agent=adv, config=TrainConfig(mode=TrainingMode.RARL))


class TestRAPMode:
    """RAP mode training tests."""

    def test_rap_runs_without_crash(self):
        env = _make_adv_env(n_envs=4)
        drone = PPOAgent(obs_dim=0, action_dim=4, n_epochs=1, mini_batch_size=8)
        adversary = PPOAgent(obs_dim=1, action_dim=1, n_epochs=1, mini_batch_size=8)
        config = TrainConfig(
            mode=TrainingMode.RAP,
            total_timesteps=32,
            rollout_steps=8,
        )
        metrics = train(env, drone, adversary_agent=adversary, config=config)
        assert "drone_reward" in metrics


class TestCallbacks:
    """Callback integration tests."""

    def test_callback_called(self):
        env = _make_dr_env(n_envs=4)
        agent = PPOAgent(obs_dim=0, action_dim=4, n_epochs=1, mini_batch_size=8)
        callback = MagicMock()
        config = TrainConfig(
            mode=TrainingMode.DR,
            total_timesteps=16,
            rollout_steps=8,
            callbacks=[callback],
        )
        train(env, agent, config=config)
        assert callback.call_count >= 1

    def test_seed_reproducibility(self):
        """Same seed produces same rewards."""

        def _run(seed):
            torch.manual_seed(seed)
            env = _make_dr_env(n_envs=4)
            agent = PPOAgent(obs_dim=0, action_dim=4, n_epochs=1, mini_batch_size=8)
            config = TrainConfig(
                mode=TrainingMode.DR,
                total_timesteps=32,
                rollout_steps=8,
                seed=seed,
            )
            return train(env, agent, config=config)

        m1 = _run(42)
        m2 = _run(42)
        assert m1["drone_reward"] == m2["drone_reward"]


class TestRARLAlternation:
    """Verify RARL alternates drone/adversary updates correctly."""

    def test_rarl_alternates_updates(self):
        """Even rollouts update drone, odd rollouts update adversary."""
        env = _make_adv_env(n_envs=4)
        drone = PPOAgent(obs_dim=0, action_dim=4, n_epochs=1, mini_batch_size=8)
        adversary = PPOAgent(obs_dim=1, action_dim=1, n_epochs=1, mini_batch_size=8)

        drone_update_calls = []
        adv_update_calls = []

        _orig_drone_update = drone.update
        _orig_adv_update = adversary.update

        def _drone_update(rollout):
            drone_update_calls.append(len(drone_update_calls))
            return _orig_drone_update(rollout)

        def _adv_update(rollout):
            adv_update_calls.append(len(adv_update_calls))
            return _orig_adv_update(rollout)

        drone.update = _drone_update
        adversary.update = _adv_update

        config = TrainConfig(
            mode=TrainingMode.RARL,
            total_timesteps=64,
            rollout_steps=8,
        )
        train(env, drone, adversary_agent=adversary, config=config)

        # With 64 timesteps and 8 steps * 4 envs = 32 per rollout,
        # we get 2 rollouts. Rollout 0 (even) → drone, rollout 1 (odd) → adversary
        assert len(drone_update_calls) >= 1
        assert len(adv_update_calls) >= 1
        # Drone updated on even rollouts, adversary on odd
        assert len(drone_update_calls) >= len(adv_update_calls)
