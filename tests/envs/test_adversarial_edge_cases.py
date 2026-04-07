"""Edge cases and Lipschitz enforcement tests for AdversarialEnv."""

from __future__ import annotations

import torch

from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.config import PerturbationConfig
from genesis_robust_rl.perturbations.base import PhysicsPerturbation

from .conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


class DummyPhysics(PhysicsPerturbation):
    def apply(self, scene, drone, env_state):
        pass


def _make_perturbation(
    pid: str,
    n_envs: int = 4,
    dimension: tuple[int, ...] = (1,),
    bounds: tuple = (-1.0, 1.0),
    lipschitz_k: float | None = None,
) -> DummyPhysics:
    return DummyPhysics(
        id=pid,
        n_envs=n_envs,
        dt=0.01,
        value_mode="fixed",
        frequency="per_step",
        scope="per_env",
        distribution="uniform",
        distribution_params={"low": bounds[0], "high": bounds[1]},
        bounds=bounds,
        nominal=0.0,
        dimension=dimension,
        lipschitz_k=lipschitz_k,
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


class TestEdgeCases:
    """Edge cases and invariant tests."""

    def test_single_target(self):
        """Single target perturbation works correctly."""
        p = _make_perturbation("p1")
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        result = adv.step(torch.zeros(4, 4), torch.full((4, 1), 0.3))
        assert len(result) == 6

    def test_multiple_targets_different_dims(self):
        """Multiple targets with different dimensions slice correctly."""
        p1 = _make_perturbation("p1", dimension=(2,))
        p2 = _make_perturbation("p2", dimension=(3,))
        p3 = _make_perturbation("p3", dimension=(1,))
        adv = _make_adv_env(perturbations=[p1, p2, p3], targets=["p1", "p2", "p3"])
        assert adv.adversary_action_space.shape == (6,)  # 2 + 3 + 1

        adv.reset()
        adv_act = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 4, dtype=torch.float32)
        adv.step(torch.zeros(4, 4), adv_act)

        torch.testing.assert_close(
            p1._current_value,
            torch.tensor([[0.1, 0.2]] * 4),
        )
        torch.testing.assert_close(
            p2._current_value,
            torch.tensor([[0.3, 0.4, 0.5]] * 4),
        )
        torch.testing.assert_close(
            p3._current_value,
            torch.tensor([[0.6]] * 4),
        )

    def test_lipschitz_enforcement(self):
        """Lipschitz clipping limits delta between consecutive steps."""
        # lipschitz_k=10.0, dt=0.01 → max_delta = 0.1 per step
        p = _make_perturbation("p1", lipschitz_k=10.0)
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        # After reset, _current_value = 0.0 (nominal)
        # Try to set to 1.0 — should be clipped to 0.1
        adv.step(torch.zeros(4, 4), torch.full((4, 1), 1.0))
        torch.testing.assert_close(p._current_value, torch.full((4, 1), 0.1), atol=1e-6, rtol=0)

        # Next step: try to set to 1.0 again — should clip to 0.2
        adv.step(torch.zeros(4, 4), torch.full((4, 1), 1.0))
        torch.testing.assert_close(p._current_value, torch.full((4, 1), 0.2), atol=1e-6, rtol=0)

    def test_lipschitz_reset_restarts_from_nominal(self):
        """After reset, Lipschitz clips relative to nominal again."""
        p = _make_perturbation("p1", lipschitz_k=10.0)
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        # Advance to 0.1
        adv.step(torch.zeros(4, 4), torch.full((4, 1), 1.0))
        torch.testing.assert_close(p._current_value, torch.full((4, 1), 0.1), atol=1e-6, rtol=0)

        # Reset and try again — should start from nominal (0.0)
        adv.reset()
        adv.step(torch.zeros(4, 4), torch.full((4, 1), 1.0))
        torch.testing.assert_close(p._current_value, torch.full((4, 1), 0.1), atol=1e-6, rtol=0)

    def test_no_lipschitz_no_clipping(self):
        """Without lipschitz_k, values are set directly (within bounds)."""
        p = _make_perturbation("p1", lipschitz_k=None, bounds=(-10.0, 10.0))
        adv = _make_adv_env(perturbations=[p], targets=["p1"])
        adv.reset()

        adv.step(torch.zeros(4, 4), torch.full((4, 1), 5.0))
        torch.testing.assert_close(p._current_value, torch.full((4, 1), 5.0))

    def test_partial_targets(self):
        """Only a subset of perturbations are adversary targets."""
        p1 = _make_perturbation("p1")
        p2 = _make_perturbation("p2")
        adv = _make_adv_env(perturbations=[p1, p2], targets=["p1"])

        assert adv.adversary_action_space.shape == (1,)
        adv.reset()

        adv.step(torch.zeros(4, 4), torch.full((4, 1), 0.5))
        # p1 controlled by adversary
        torch.testing.assert_close(p1._current_value, torch.full((4, 1), 0.5))
        # p2 stays at nominal (ADV mode, no sample, no set_value)
        torch.testing.assert_close(p2._current_value, torch.zeros(4, 1))
