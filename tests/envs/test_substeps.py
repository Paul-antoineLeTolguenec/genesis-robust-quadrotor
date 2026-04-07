"""Tests for variable delta_t (substeps, perturbation 3.5)."""

from unittest.mock import MagicMock

import torch

from genesis_robust_rl.envs.config import PerturbationConfig, SensorConfig
from genesis_robust_rl.perturbations.base import PerturbationMode
from tests.envs.conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


class TestFixedSubsteps:
    """Test fixed substeps behavior."""

    def test_default_substeps_1(self):
        scene, drone = _make_mock_scene(), _make_mock_drone(4)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=4,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            substeps=1,
        )
        assert env._get_substeps() == 1

    def test_fixed_substeps_5(self):
        scene, drone = _make_mock_scene(), _make_mock_drone(4)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=4,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            substeps=5,
        )
        assert env._get_substeps() == 5

    def test_scene_step_count_matches_substeps(self):
        scene, drone = _make_mock_scene(), _make_mock_drone(4)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=4,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            substeps=4,
        )
        env.reset()
        scene.step.reset_mock()
        env.step(torch.zeros(4, 4))
        assert scene.step.call_count == 4


class TestJitterSubsteps:
    """Test substeps_range (perturbation 3.5)."""

    def test_substeps_in_range(self):
        scene, drone = _make_mock_scene(), _make_mock_drone(4)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=4,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            substeps_range=(2, 5),
        )
        for _ in range(50):
            n = env._get_substeps()
            assert 2 <= n <= 5

    def test_substeps_range_overrides_fixed(self):
        scene, drone = _make_mock_scene(), _make_mock_drone(4)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=4,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            substeps=1,
            substeps_range=(3, 3),
        )
        assert env._get_substeps() == 3

    def test_dt_propagated_to_perturbations(self):
        """When substeps_range is active, p.dt must be updated before tick."""
        n = 4
        p = MagicMock()
        p.id = "test"
        p.observable = True
        p.scope = "per_env"
        p.nominal = 0.0
        p.dimension = (1,)
        p.n_envs = n
        p.mode = PerturbationMode.DOMAIN_RANDOMIZATION
        p._batch_shape.return_value = (n, 1)
        p._current_value = torch.zeros(n, 1)
        p.get_privileged_obs.return_value = torch.zeros(n, 1)
        p.is_stateful = False
        p.frequency = "per_episode"
        p.curriculum_scale = 1.0
        p.dt = 0.01
        p.apply = MagicMock()

        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
            dt=0.01,
            substeps_range=(3, 3),
        )
        env.reset()
        env.step(torch.zeros(n, 4))
        # After step, p.dt should be rl_dt = 3 * 0.01 = 0.03
        assert abs(p.dt - 0.03) < 1e-6
