"""Tests for obs/action desync (perturbation 3.6)."""

import torch

from genesis_robust_rl.envs.config import DesyncConfig, PerturbationConfig, SensorConfig
from genesis_robust_rl.sensor_models import GyroscopeModel
from tests.envs.conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene


class TestDesyncBuffers:
    """Test desync delay buffer creation and reset."""

    def test_buffers_created_when_desync_cfg(self):
        n = 4
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(gyroscope=GyroscopeModel(n_envs=n)),
            desync_cfg=DesyncConfig(obs_delay_range=(0, 3), action_delay_range=(0, 2)),
        )
        assert env._obs_delay_buffer is not None
        assert env._action_delay_buffer is not None

    def test_no_buffers_without_desync_cfg(self):
        n = 4
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
        )
        assert env._obs_delay_buffer is None
        assert env._action_delay_buffer is None

    def test_delays_sampled_at_reset(self):
        n = 4
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(gyroscope=GyroscopeModel(n_envs=n)),
            desync_cfg=DesyncConfig(obs_delay_range=(1, 3), action_delay_range=(1, 2)),
        )
        env.reset()
        assert env._obs_delay is not None
        assert env._action_delay is not None
        # Delays should be within range
        assert (env._obs_delay >= 1).all() and (env._obs_delay <= 3).all()
        assert (env._action_delay >= 1).all() and (env._action_delay <= 2).all()


class TestDesyncColdStart:
    """Test zero-order hold cold start behavior."""

    def test_obs_cold_start_zeros(self):
        """First steps should return zeros when delay > 0."""
        n = 2
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        # Non-zero gyro reading
        drone.get_ang_vel.return_value = torch.ones(n, 3)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(gyroscope=GyroscopeModel(n_envs=n)),
            desync_cfg=DesyncConfig(
                obs_delay_range=(2, 2),
                action_delay_range=(0, 0),
            ),
        )
        env.reset()
        action = torch.zeros(n, 4)
        obs, _, _, _, _ = env.step(action)
        # With delay=2, first obs should be zeros (cold start)
        assert (obs == 0.0).all()


class TestDesyncCorrelation:
    """Test delay correlation parameter."""

    def test_full_correlation(self):
        """With correlation=1.0, obs and action delays should be fully correlated."""
        n = 16
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(gyroscope=GyroscopeModel(n_envs=n)),
            desync_cfg=DesyncConfig(
                obs_delay_range=(0, 10),
                action_delay_range=(0, 10),
                correlation=1.0,
            ),
        )
        env.reset()
        # With perfect correlation and same range, delays should be identical
        assert env._obs_delay is not None and env._action_delay is not None
        assert (env._obs_delay == env._action_delay).all()

    def test_zero_correlation_varies(self):
        """With correlation=0.0, delays should vary independently (statistical test)."""
        n = 128
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(gyroscope=GyroscopeModel(n_envs=n)),
            desync_cfg=DesyncConfig(
                obs_delay_range=(0, 10),
                action_delay_range=(0, 10),
                correlation=0.0,
            ),
        )
        env.reset()
        # With zero correlation, not all delays should match (with high probability)
        assert env._obs_delay is not None and env._action_delay is not None
        # At least some should differ
        assert not (env._obs_delay == env._action_delay).all()
