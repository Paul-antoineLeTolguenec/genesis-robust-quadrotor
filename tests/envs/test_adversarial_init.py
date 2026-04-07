"""Tests for AdversarialEnv constructor validation."""

from __future__ import annotations

import numpy as np
import pytest

from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.config import PerturbationConfig
from genesis_robust_rl.perturbations.base import (
    PerturbationMode,
    PhysicsPerturbation,
)

from .conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene

# ---------- Helpers: minimal concrete perturbations ----------


class DummyPhysicsPerturbation(PhysicsPerturbation):
    """Minimal per_env, stateless physics perturbation for testing."""

    def apply(self, scene, drone, env_state):
        pass


class StatefulPhysicsPerturbation(PhysicsPerturbation):
    """Stateful physics perturbation — should be rejected as adversary target."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_stateful = True

    def apply(self, scene, drone, env_state):
        pass


class GlobalPhysicsPerturbation(PhysicsPerturbation):
    """Global-scope physics perturbation — should be rejected as adversary target."""

    def apply(self, scene, drone, env_state):
        pass


# ---------- Fixtures ----------


def _make_perturbation(
    pid: str = "p1",
    n_envs: int = 4,
    scope: str = "per_env",
    dimension: tuple[int, ...] = (1,),
    bounds: tuple = (-1.0, 1.0),
    is_stateful: bool = False,
) -> PhysicsPerturbation:
    cls = StatefulPhysicsPerturbation if is_stateful else DummyPhysicsPerturbation
    if scope == "global":
        cls = GlobalPhysicsPerturbation
    return cls(
        id=pid,
        n_envs=n_envs,
        dt=0.01,
        value_mode="fixed",
        frequency="per_step",
        scope=scope,
        distribution="uniform",
        distribution_params={"low": bounds[0], "high": bounds[1]},
        bounds=bounds,
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
    scene = _make_mock_scene()
    drone = _make_mock_drone(n_envs)
    env = DummyDroneEnv(scene=scene, drone=drone, n_envs=n_envs, perturbation_cfg=cfg)
    return AdversarialEnv(env=env, adversary_targets=targets)


# ---------- Tests ----------


class TestAdversarialInit:
    """Constructor validation tests."""

    def test_stateful_target_raises(self):
        """Stateful perturbation in adversary_targets raises ValueError."""
        p = _make_perturbation("p1", is_stateful=True)
        with pytest.raises(ValueError, match="Stateful"):
            _make_adv_env(perturbations=[p], targets=["p1"])

    def test_global_scope_target_raises(self):
        """Global-scope perturbation in adversary_targets raises ValueError."""
        p = _make_perturbation("p1", scope="global")
        with pytest.raises(ValueError, match="Global-scope"):
            _make_adv_env(perturbations=[p], targets=["p1"])

    def test_unknown_target_raises(self):
        """Unknown perturbation ID raises KeyError."""
        with pytest.raises(KeyError, match="unknown_id"):
            _make_adv_env(targets=["unknown_id"])

    def test_adversary_action_space_shape_single(self):
        """Single target with dimension (1,) → adversary_dim = 1."""
        p = _make_perturbation("p1", dimension=(1,), bounds=(-2.0, 2.0))
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        assert adv.adversary_action_space.shape == (1,)
        np.testing.assert_allclose(adv.adversary_action_space.low, [-2.0])
        np.testing.assert_allclose(adv.adversary_action_space.high, [2.0])

    def test_adversary_action_space_shape_multi(self):
        """Two targets with dim (3,) and (1,) → adversary_dim = 4."""
        p1 = _make_perturbation("p1", dimension=(3,), bounds=(-5.0, 5.0))
        p2 = _make_perturbation("p2", dimension=(1,), bounds=(-1.0, 1.0))
        adv = _make_adv_env(perturbations=[p1, p2], targets=["p1", "p2"])
        assert adv.adversary_action_space.shape == (4,)
        np.testing.assert_allclose(adv.adversary_action_space.low, [-5, -5, -5, -1], atol=1e-6)
        np.testing.assert_allclose(adv.adversary_action_space.high, [5, 5, 5, 1], atol=1e-6)

    def test_empty_targets(self):
        """Empty adversary_targets → adversary_dim = 0."""
        adv = _make_adv_env(targets=[])
        assert adv.adversary_action_space.shape == (0,)

    def test_observation_space_delegated(self):
        """observation_space matches wrapped env."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        assert adv.observation_space == adv.env.observation_space

    def test_action_space_delegated(self):
        """action_space matches wrapped env."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        assert adv.action_space == adv.env.action_space

    def test_mode_switched_to_adversarial(self):
        """Wrapped env is switched to ADVERSARIAL mode at construction."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        assert adv.env._mode == PerturbationMode.ADVERSARIAL
        assert p.mode == PerturbationMode.ADVERSARIAL

    def test_params_mode_raises_not_implemented(self):
        """adversary_mode='params' raises NotImplementedError."""
        p = _make_perturbation("p1")
        cfg = PerturbationConfig(physics=[p])
        scene = _make_mock_scene()
        drone = _make_mock_drone(4)
        env = DummyDroneEnv(scene=scene, drone=drone, n_envs=4, perturbation_cfg=cfg)
        with pytest.raises(NotImplementedError, match="params"):
            AdversarialEnv(env=env, adversary_targets=["p1"], adversary_mode="params")

    def test_getattr_forwards_to_env(self):
        """Attribute forwarding to wrapped env works (gym compat)."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        # n_envs is an attribute of the wrapped env, not AdversarialEnv
        assert adv.n_envs == adv.env.n_envs
        assert adv.dt == adv.env.dt

    def test_already_adversarial_no_double_switch(self):
        """If env already in ADVERSARIAL mode, no error or double switch."""
        p = _make_perturbation("p1")
        cfg = PerturbationConfig(physics=[p])
        scene = _make_mock_scene()
        drone = _make_mock_drone(4)
        env = DummyDroneEnv(scene=scene, drone=drone, n_envs=4, perturbation_cfg=cfg)
        env.set_mode(PerturbationMode.ADVERSARIAL)
        adv = AdversarialEnv(env=env, adversary_targets=["p1"])
        assert adv.env._mode == PerturbationMode.ADVERSARIAL
