"""Tests for AdversarialEnv reward computation."""

from __future__ import annotations

import torch
from torch import Tensor

from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.config import PerturbationConfig
from genesis_robust_rl.perturbations.base import EnvState, PhysicsPerturbation

from .conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


class DummyPhysics(PhysicsPerturbation):
    def apply(self, scene, drone, env_state):
        pass


class RewardDroneEnv(DummyDroneEnv):
    """DummyDroneEnv that returns non-zero reward for testing."""

    def _compute_reward(self, env_state: EnvState) -> tuple[Tensor, Tensor, Tensor]:
        reward = torch.ones(self.n_envs, device=self.device) * 3.0
        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        return reward, terminated, truncated


class CustomRewardAdversarialEnv(AdversarialEnv):
    """Subclass overriding _adversary_reward."""

    def _adversary_reward(self, obs, drone_reward, env_state):
        return drone_reward * 2.0  # non-zero-sum for testing


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
    )


class TestAdversarialReward:
    """Reward computation tests."""

    def test_default_zero_sum(self):
        """Default adv_reward == -drone_reward."""
        n_envs = 4
        p = _make_perturbation("p1", n_envs)
        cfg = PerturbationConfig(physics=[p])
        env = RewardDroneEnv(
            scene=_make_mock_scene(),
            drone=_make_mock_drone(n_envs),
            n_envs=n_envs,
            perturbation_cfg=cfg,
        )
        adv = AdversarialEnv(env=env, adversary_targets=["p1"])
        adv.reset()

        _, drone_r, adv_r, _, _, _ = adv.step(torch.zeros(n_envs, 4), torch.zeros(n_envs, 1))

        torch.testing.assert_close(adv_r, -drone_r)
        torch.testing.assert_close(drone_r, torch.full((n_envs,), 3.0))

    def test_custom_override(self):
        """Subclass override produces custom adversary reward."""
        n_envs = 4
        p = _make_perturbation("p1", n_envs)
        cfg = PerturbationConfig(physics=[p])
        env = RewardDroneEnv(
            scene=_make_mock_scene(),
            drone=_make_mock_drone(n_envs),
            n_envs=n_envs,
            perturbation_cfg=cfg,
        )
        adv = CustomRewardAdversarialEnv(env=env, adversary_targets=["p1"])
        adv.reset()

        _, drone_r, adv_r, _, _, _ = adv.step(torch.zeros(n_envs, 4), torch.zeros(n_envs, 1))

        torch.testing.assert_close(adv_r, drone_r * 2.0)

    def test_env_state_passed_to_reward(self):
        """_adversary_reward receives env._last_env_state (true state)."""
        n_envs = 4
        p = _make_perturbation("p1", n_envs)
        cfg = PerturbationConfig(physics=[p])
        env = RewardDroneEnv(
            scene=_make_mock_scene(),
            drone=_make_mock_drone(n_envs),
            n_envs=n_envs,
            perturbation_cfg=cfg,
        )

        received_state = []

        class SpyAdv(AdversarialEnv):
            def _adversary_reward(self, obs, drone_reward, env_state):
                received_state.append(env_state)
                return -drone_reward

        adv = SpyAdv(env=env, adversary_targets=["p1"])
        adv.reset()
        adv.step(torch.zeros(n_envs, 4), torch.zeros(n_envs, 1))

        assert len(received_state) == 1
        assert received_state[0] is not None
        assert received_state[0] is env._last_env_state
