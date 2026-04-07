"""Tests for AdversarialEnv.reset() delegation."""

from __future__ import annotations

import torch

from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.config import PerturbationConfig
from genesis_robust_rl.perturbations.base import PhysicsPerturbation

from .conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


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
    )


def _make_env(n_envs: int = 4) -> tuple[AdversarialEnv, DummyPhysics]:
    p = _make_perturbation("p1", n_envs)
    cfg = PerturbationConfig(physics=[p])
    env = DummyDroneEnv(
        scene=_make_mock_scene(),
        drone=_make_mock_drone(n_envs),
        n_envs=n_envs,
        perturbation_cfg=cfg,
    )
    adv = AdversarialEnv(env=env, adversary_targets=["p1"])
    return adv, p


class TestAdversarialReset:
    """Reset delegation tests."""

    def test_reset_returns_obs_and_info(self):
        adv, _ = _make_env()
        obs, info = adv.reset()
        assert isinstance(obs, torch.Tensor)
        assert isinstance(info, dict)

    def test_reset_obs_shape(self):
        adv, _ = _make_env(n_envs=8)
        obs, _ = adv.reset()
        expected_dim = adv.observation_space.shape[0]
        assert obs.shape == (8, expected_dim)

    def test_reset_privileged_obs_in_info(self):
        adv, _ = _make_env()
        _, info = adv.reset()
        assert "privileged_obs" in info
        assert isinstance(info["privileged_obs"], torch.Tensor)

    def test_current_value_is_nominal_after_reset(self):
        """After reset, _current_value == nominal for all targets."""
        adv, p = _make_env()
        # Set a non-nominal value first
        p._current_value = torch.full((4, 1), 0.5)
        adv.reset()
        assert p._current_value is not None
        torch.testing.assert_close(
            p._current_value,
            torch.zeros(4, 1),
            atol=1e-6,
            rtol=0,
        )

    def test_reset_with_seed(self):
        """reset(seed=...) does not crash."""
        adv, _ = _make_env()
        obs, info = adv.reset(seed=42)
        assert isinstance(obs, torch.Tensor)
