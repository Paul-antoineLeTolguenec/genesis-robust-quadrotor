"""Tests for mode switch, privileged obs, curriculum, set_perturbation_values."""

from unittest.mock import MagicMock

import pytest
import torch

from genesis_robust_rl.envs.config import PerturbationConfig, SensorConfig
from genesis_robust_rl.perturbations.base import Perturbation, PerturbationMode
from tests.envs.conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


def _make_mock_perturbation(pid: str, n_envs: int = 4, observable: bool = True) -> MagicMock:
    """Create a mock perturbation with realistic attributes."""
    p = MagicMock(spec=Perturbation)
    p.id = pid
    p.observable = observable
    p.scope = "per_env"
    p.nominal = 0.0
    p.dimension = (1,)
    p.n_envs = n_envs
    p.mode = PerturbationMode.DOMAIN_RANDOMIZATION
    p._batch_shape.return_value = (n_envs, 1)
    p._current_value = torch.zeros(n_envs, 1)
    p.get_privileged_obs.return_value = torch.zeros(n_envs, 1)
    p.is_stateful = False
    p.frequency = "per_episode"
    p.curriculum_scale = 1.0
    return p


class TestModeSwitch:
    """Test DR <-> ADV mode switching."""

    def test_set_mode_dr_to_adv(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        env.set_mode(PerturbationMode.ADVERSARIAL)
        assert p.mode == PerturbationMode.ADVERSARIAL

    def test_set_mode_adv_to_dr(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        p.mode = PerturbationMode.ADVERSARIAL
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        env.set_mode(PerturbationMode.DOMAIN_RANDOMIZATION)
        assert p.mode == PerturbationMode.DOMAIN_RANDOMIZATION


class TestSetPerturbationValues:
    """Test adversarial set_perturbation_values."""

    def test_raises_in_dr_mode(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        with pytest.raises(ValueError, match="ADVERSARIAL"):
            env.set_perturbation_values({"p1": torch.zeros(n, 1)})

    def test_raises_unknown_id(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        p.mode = PerturbationMode.ADVERSARIAL
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        env.set_mode(PerturbationMode.ADVERSARIAL)
        with pytest.raises(KeyError, match="unknown"):
            env.set_perturbation_values({"unknown": torch.zeros(n, 1)})

    def test_calls_set_value(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        p.mode = PerturbationMode.ADVERSARIAL
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        env.set_mode(PerturbationMode.ADVERSARIAL)
        val = torch.ones(n, 1)
        env.set_perturbation_values({"p1": val})
        p.set_value.assert_called_once_with(val)


class TestUpdatePerturbationParams:
    """Test adversarial update_perturbation_params."""

    def test_raises_in_dr_mode(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        with pytest.raises(ValueError, match="ADVERSARIAL"):
            env.update_perturbation_params({"p1": {"mean": 0.5}})

    def test_calls_update_params(self):
        n = 4
        p = _make_mock_perturbation("p1", n)
        p.mode = PerturbationMode.ADVERSARIAL
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        env.set_mode(PerturbationMode.ADVERSARIAL)
        env.update_perturbation_params({"p1": {"mean": 0.5}})
        p.update_params.assert_called_once_with({"mean": 0.5})


class TestCurriculum:
    """Test set_curriculum_scale."""

    def test_global_scale(self):
        n = 4
        p1 = _make_mock_perturbation("p1", n)
        p2 = _make_mock_perturbation("p2", n)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p1, p2]),
            sensor_cfg=SensorConfig(),
        )
        env.set_curriculum_scale(0.5)
        assert p1.curriculum_scale == 0.5
        assert p2.curriculum_scale == 0.5

    def test_per_id_scale(self):
        n = 4
        p1 = _make_mock_perturbation("p1", n)
        p2 = _make_mock_perturbation("p2", n)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p1, p2]),
            sensor_cfg=SensorConfig(),
        )
        env.set_curriculum_scale({"p1": 0.3, "p2": 0.7})
        assert p1.curriculum_scale == 0.3
        assert p2.curriculum_scale == 0.7

    def test_unknown_id_raises(self):
        n = 4
        p1 = _make_mock_perturbation("p1", n)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p1]),
            sensor_cfg=SensorConfig(),
        )
        with pytest.raises(KeyError, match="unknown"):
            env.set_curriculum_scale({"unknown": 0.5})


class TestPrivilegedObs:
    """Test get_privileged_obs."""

    def test_empty_when_no_perturbations(self):
        n = 4
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
        )
        priv = env.get_privileged_obs()
        assert priv.shape == (n, 0)

    def test_skips_non_observable(self):
        n = 4
        p = _make_mock_perturbation("p1", n, observable=False)
        p.get_privileged_obs.return_value = None
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        priv = env.get_privileged_obs()
        assert priv.shape == (n, 0)

    def test_concatenates_observable(self):
        n = 4
        p1 = _make_mock_perturbation("p1", n)
        p1.get_privileged_obs.return_value = torch.ones(n, 1)
        p2 = _make_mock_perturbation("p2", n)
        p2.dimension = (3,)
        p2.get_privileged_obs.return_value = torch.ones(n, 3)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p1, p2]),
            sensor_cfg=SensorConfig(),
        )
        priv = env.get_privileged_obs()
        assert priv.shape == (n, 4)

    def test_global_broadcast(self):
        """Global-scope perturbation [1, dim] should broadcast to [n_envs, dim]."""
        n = 4
        p = _make_mock_perturbation("p1", n)
        p.scope = "global"
        p.get_privileged_obs.return_value = torch.ones(1, 1)
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(physics=[p]),
            sensor_cfg=SensorConfig(),
        )
        priv = env.get_privileged_obs()
        assert priv.shape == (n, 1)
        assert (priv == 1.0).all()
