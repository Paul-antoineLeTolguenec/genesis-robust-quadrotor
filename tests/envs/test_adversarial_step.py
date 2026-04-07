"""Tests for AdversarialEnv.step() — sequence [A1]–[A4]."""

from __future__ import annotations

import pytest
import torch

from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.config import PerturbationConfig
from genesis_robust_rl.perturbations.base import PhysicsPerturbation

from .conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


class DummyPhysics(PhysicsPerturbation):
    def apply(self, scene, drone, env_state):
        pass


def _make_perturbation(
    pid: str, n_envs: int = 4, dimension: tuple[int, ...] = (1,)
) -> DummyPhysics:
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
        dimension=dimension,
    )


def _make_adv_env(
    n_envs: int = 4,
    perturbations: list | None = None,
    targets: list[str] | None = None,
) -> AdversarialEnv:
    perturbations = perturbations or []
    targets = targets or []
    cfg = PerturbationConfig(physics=perturbations)
    env = DummyDroneEnv(
        scene=_make_mock_scene(),
        drone=_make_mock_drone(n_envs),
        n_envs=n_envs,
        perturbation_cfg=cfg,
    )
    return AdversarialEnv(env=env, adversary_targets=targets)


class TestAdversarialStep:
    """Step sequence [A1]–[A4] tests."""

    def test_step_returns_6_tuple(self):
        """step() returns (obs, drone_reward, adv_reward, terminated, truncated, info)."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        adv_act = torch.zeros(4, 1)
        result = adv.step(drone_act, adv_act)
        assert len(result) == 6

        obs, drone_r, adv_r, term, trunc, info = result
        assert obs.shape[0] == 4
        assert drone_r.shape == (4,)
        assert adv_r.shape == (4,)
        assert term.shape == (4,)
        assert trunc.shape == (4,)
        assert isinstance(info, dict)

    def test_wrong_adversary_shape_raises(self):
        """adversary_action with wrong shape raises ValueError."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        bad_act = torch.zeros(4, 3)  # expected (4, 1)
        with pytest.raises(ValueError, match="shape"):
            adv.step(drone_act, bad_act)

    def test_wrong_adversary_batch_raises(self):
        """adversary_action with wrong batch dim raises ValueError."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        bad_act = torch.zeros(2, 1)  # expected n_envs=4
        with pytest.raises(ValueError, match="shape"):
            adv.step(drone_act, bad_act)

    def test_set_perturbation_values_called(self):
        """set_perturbation_values is called with correct slices before env.step."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        adv_act = torch.full((4, 1), 0.5)
        adv.step(drone_act, adv_act)

        # _current_value should reflect the adversary's action (possibly clipped)
        assert p._current_value is not None
        # Value was 0.0 (nominal) after reset, set to 0.5
        # No lipschitz_k set, so no clipping
        torch.testing.assert_close(p._current_value, torch.full((4, 1), 0.5))

    def test_multi_target_slicing(self):
        """Multiple targets get correct slices from flat adversary_action."""
        p1 = _make_perturbation("p1", dimension=(3,))
        p2 = _make_perturbation("p2", dimension=(1,))
        adv = _make_adv_env(perturbations=[p1, p2], targets=["p1", "p2"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        # adversary_dim = 3 + 1 = 4
        adv_act = torch.tensor([[1, 2, 3, 4]] * 4, dtype=torch.float32)
        adv.step(drone_act, adv_act)

        # set_value() does NOT clip to bounds — only Lipschitz (disabled here)
        torch.testing.assert_close(
            p1._current_value,
            torch.tensor([[1, 2, 3]] * 4, dtype=torch.float32),
        )
        torch.testing.assert_close(
            p2._current_value,
            torch.tensor([[4.0]] * 4, dtype=torch.float32),
        )

    def test_privileged_obs_contains_all(self):
        """info['privileged_obs'] contains ALL observable perturbations, not just targets."""
        p1 = _make_perturbation("p1")
        p2 = _make_perturbation("p2")
        adv = _make_adv_env(perturbations=[p1, p2], targets=["p1"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        adv_act = torch.zeros(4, 1)
        _, _, _, _, _, info = adv.step(drone_act, adv_act)

        priv = info["privileged_obs"]
        # Both p1 and p2 are observable=True by default → 2 dims
        assert priv.shape == (4, 2)

    def test_env_state_non_none_after_step(self):
        """env._last_env_state is guaranteed non-None after step."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        adv_act = torch.zeros(4, 1)
        adv.step(drone_act, adv_act)

        assert adv.env._last_env_state is not None

    def test_empty_targets_step_works(self):
        """Step with no adversary targets still works (adversary_dim=0)."""
        adv = _make_adv_env(targets=[])
        adv.reset()

        drone_act = torch.zeros(4, 4)
        adv_act = torch.zeros(4, 0)
        result = adv.step(drone_act, adv_act)
        assert len(result) == 6
