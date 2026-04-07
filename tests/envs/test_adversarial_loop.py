"""Integration test: mini adversarial training loop."""

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


class TestAdversarialLoop:
    """Simulate a mini adversarial training loop."""

    def test_full_loop_no_crash(self):
        """10 steps, mid-loop reset, no crash, no NaN, shapes consistent."""
        n_envs = 4
        p1 = _make_perturbation("p1", n_envs, dimension=(3,))
        p2 = _make_perturbation("p2", n_envs, dimension=(1,))
        p3 = _make_perturbation("p3", n_envs, dimension=(2,))

        cfg = PerturbationConfig(physics=[p1, p2, p3])
        env = DummyDroneEnv(
            scene=_make_mock_scene(),
            drone=_make_mock_drone(n_envs),
            n_envs=n_envs,
            perturbation_cfg=cfg,
        )
        adv = AdversarialEnv(
            env=env,
            adversary_targets=["p1", "p2"],  # p3 not controlled
        )

        adversary_dim = adv.adversary_action_space.shape[0]
        assert adversary_dim == 4  # 3 + 1

        obs, info = adv.reset()
        obs_dim = adv.observation_space.shape[0]
        assert obs.shape == (n_envs, obs_dim)

        for step_i in range(10):
            drone_action = torch.rand(n_envs, 4) * 2 - 1
            adv_action = torch.rand(n_envs, adversary_dim) * 2 - 1

            obs, drone_r, adv_r, term, trunc, info = adv.step(drone_action, adv_action)

            # Shape checks
            assert obs.shape == (n_envs, obs_dim)
            assert drone_r.shape == (n_envs,)
            assert adv_r.shape == (n_envs,)
            assert term.shape == (n_envs,)
            assert trunc.shape == (n_envs,)
            assert "privileged_obs" in info

            # No NaN
            assert not torch.isnan(obs).any(), f"NaN in obs at step {step_i}"
            assert not torch.isnan(drone_r).any(), f"NaN in drone_r at step {step_i}"
            assert not torch.isnan(adv_r).any(), f"NaN in adv_r at step {step_i}"

            # Mid-loop reset at step 5
            if step_i == 5:
                obs, info = adv.reset()
                assert obs.shape == (n_envs, obs_dim)

    def test_loop_with_lipschitz(self):
        """Training loop with Lipschitz-constrained perturbation."""
        n_envs = 4
        p = DummyPhysics(
            id="lp",
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
            lipschitz_k=10.0,  # max_delta = 0.1 per step
        )

        cfg = PerturbationConfig(physics=[p])
        env = DummyDroneEnv(
            scene=_make_mock_scene(),
            drone=_make_mock_drone(n_envs),
            n_envs=n_envs,
            perturbation_cfg=cfg,
        )
        adv = AdversarialEnv(env=env, adversary_targets=["lp"])
        adv.reset()

        prev_val = torch.zeros(n_envs, 1)
        for _ in range(20):
            adv_act = torch.full((n_envs, 1), 1.0)  # always push to max
            adv.step(torch.zeros(n_envs, 4), adv_act)

            curr_val = p._current_value.clone()
            delta = (curr_val - prev_val).abs()
            assert (delta <= 0.1 + 1e-6).all(), f"Lipschitz violated: delta={delta}"
            prev_val = curr_val

    def test_non_controlled_perturbation_stays_nominal(self):
        """Non-target perturbations keep nominal value in ADV mode."""
        n_envs = 4
        p1 = _make_perturbation("controlled", n_envs)
        p2 = _make_perturbation("passive", n_envs)

        cfg = PerturbationConfig(physics=[p1, p2])
        env = DummyDroneEnv(
            scene=_make_mock_scene(),
            drone=_make_mock_drone(n_envs),
            n_envs=n_envs,
            perturbation_cfg=cfg,
        )
        adv = AdversarialEnv(env=env, adversary_targets=["controlled"])
        adv.reset()

        for _ in range(5):
            adv.step(torch.zeros(n_envs, 4), torch.full((n_envs, 1), 0.5))

        # passive perturbation should still be at nominal
        torch.testing.assert_close(p2._current_value, torch.zeros(n_envs, 1))
