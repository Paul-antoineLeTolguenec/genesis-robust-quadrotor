"""Shared fixtures for RobustDroneEnv tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from genesis_robust_rl.envs.config import PerturbationConfig, SensorConfig
from genesis_robust_rl.envs.robust_drone_env import RobustDroneEnv
from genesis_robust_rl.perturbations.base import EnvState

# ---------- Concrete subclass for testing ----------


class DummyDroneEnv(RobustDroneEnv):
    """Minimal concrete subclass of RobustDroneEnv for testing."""

    def policy_to_rpm(self, action: Tensor) -> Tensor:
        # Identity: action is already RPM-like [n_envs, 4]
        return action

    def _compute_reward(self, env_state: EnvState) -> tuple[Tensor, Tensor, Tensor]:
        reward = torch.zeros(self.n_envs, device=self.device)
        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        return reward, terminated, truncated


# ---------- Mock Genesis scene/drone ----------


def _make_mock_drone(n_envs: int, device: str = "cpu") -> MagicMock:
    """Create a mock drone with realistic return values."""
    drone = MagicMock()
    drone.get_pos.return_value = torch.zeros(n_envs, 3, device=device)
    drone.get_quat.return_value = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(n_envs, -1).clone()
    )
    drone.get_vel.return_value = torch.zeros(n_envs, 3, device=device)
    drone.get_ang_vel.return_value = torch.zeros(n_envs, 3, device=device)
    drone.get_rpm.return_value = torch.ones(n_envs, 4, device=device) * 3000.0
    drone.set_propellels_rpm = MagicMock()
    return drone


def _make_mock_scene() -> MagicMock:
    """Create a mock Genesis scene."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    return scene


# ---------- Fixtures ----------


@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param


@pytest.fixture
def mock_scene():
    return _make_mock_scene()


@pytest.fixture
def mock_drone(n_envs):
    return _make_mock_drone(n_envs)


@pytest.fixture
def dummy_env(mock_scene, mock_drone, n_envs):
    """DummyDroneEnv with no perturbations, no sensors."""
    return DummyDroneEnv(
        scene=mock_scene,
        drone=mock_drone,
        n_envs=n_envs,
        perturbation_cfg=PerturbationConfig(),
        sensor_cfg=SensorConfig(),
        dt=0.01,
    )


@pytest.fixture
def dummy_env_with_gyro(mock_scene, mock_drone, n_envs):
    """DummyDroneEnv with a gyroscope sensor."""
    from genesis_robust_rl.sensor_models import GyroscopeModel

    sensor_cfg = SensorConfig(gyroscope=GyroscopeModel(n_envs=n_envs))
    return DummyDroneEnv(
        scene=mock_scene,
        drone=mock_drone,
        n_envs=n_envs,
        perturbation_cfg=PerturbationConfig(),
        sensor_cfg=sensor_cfg,
        dt=0.01,
    )
