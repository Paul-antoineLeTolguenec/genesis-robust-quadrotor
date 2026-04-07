"""Tests for RobustDroneEnv.step() — sequence [0]-[11] from 04_interactions.md."""

import torch

from genesis_robust_rl.envs.config import PerturbationConfig, SensorConfig


class TestStepBasic:
    """Basic step behavior without perturbations."""

    def test_step_returns_5_tuple(self, dummy_env):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        result = dummy_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

    def test_step_obs_shape_no_sensors(self, dummy_env):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        obs, _, _, _, _ = dummy_env.step(action)
        assert obs.shape == (dummy_env.n_envs, 0)

    def test_step_obs_shape_with_gyro(self, dummy_env_with_gyro):
        dummy_env_with_gyro.reset()
        action = torch.zeros(dummy_env_with_gyro.n_envs, 4)
        obs, _, _, _, _ = dummy_env_with_gyro.step(action)
        assert obs.shape == (dummy_env_with_gyro.n_envs, 3)

    def test_step_reward_shape(self, dummy_env):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        _, reward, _, _, _ = dummy_env.step(action)
        assert reward.shape == (dummy_env.n_envs,)

    def test_step_terminated_truncated_shape(self, dummy_env):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        _, _, terminated, truncated, _ = dummy_env.step(action)
        assert terminated.shape == (dummy_env.n_envs,)
        assert truncated.shape == (dummy_env.n_envs,)

    def test_step_increments_count(self, dummy_env):
        dummy_env.reset()
        assert dummy_env._step_count == 0
        action = torch.zeros(dummy_env.n_envs, 4)
        dummy_env.step(action)
        assert dummy_env._step_count == 1
        dummy_env.step(action)
        assert dummy_env._step_count == 2

    def test_step_privileged_obs_in_info(self, dummy_env):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        _, _, _, _, info = dummy_env.step(action)
        assert "privileged_obs" in info


class TestStepSceneCalls:
    """Verify correct Genesis API calls during step."""

    def test_scene_step_called(self, dummy_env, mock_scene):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        dummy_env.step(action)
        # Default substeps=1 → 1 scene.step() call (+ 1 from reset via _build_env_state)
        # But scene.step() is called in the substep loop
        assert mock_scene.step.call_count >= 1

    def test_drone_set_rpm_called(self, dummy_env, mock_drone):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        dummy_env.step(action)
        mock_drone.set_propellels_rpm.assert_called()

    def test_substeps_multiple_scene_steps(self, mock_scene, mock_drone, n_envs):
        from tests.envs.conftest import DummyDroneEnv

        env = DummyDroneEnv(
            scene=mock_scene,
            drone=mock_drone,
            n_envs=n_envs,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            substeps=3,
        )
        env.reset()
        mock_scene.step.reset_mock()
        action = torch.zeros(n_envs, 4)
        env.step(action)
        assert mock_scene.step.call_count == 3


class TestStepMaxSteps:
    """Test truncation at max_steps."""

    def test_truncation_at_max_steps(self):
        from tests.envs.conftest import DummyDroneEnv, _make_mock_drone, _make_mock_scene

        n = 2
        scene, drone = _make_mock_scene(), _make_mock_drone(n)
        env = DummyDroneEnv(
            scene=scene,
            drone=drone,
            n_envs=n,
            perturbation_cfg=PerturbationConfig(),
            sensor_cfg=SensorConfig(),
            max_steps=3,
        )
        env.reset()
        action = torch.zeros(n, 4)
        for i in range(2):
            _, _, _, truncated, _ = env.step(action)
            assert not truncated.any(), f"Should not truncate at step {i + 1}"
        _, _, _, truncated, _ = env.step(action)
        assert truncated.all(), "Should truncate at max_steps"


class TestStepEnvState:
    """Verify _build_env_state and _last_env_state."""

    def test_last_env_state_updated(self, dummy_env):
        dummy_env.reset()
        action = torch.zeros(dummy_env.n_envs, 4)
        dummy_env.step(action)
        assert dummy_env._last_env_state is not None

    def test_vel_prev_updated(self, dummy_env, mock_drone):
        dummy_env.reset()
        # Set drone to return non-zero velocity
        mock_drone.get_vel.return_value = torch.ones(dummy_env.n_envs, 3)
        action = torch.zeros(dummy_env.n_envs, 4)
        dummy_env.step(action)
        assert (dummy_env._vel_prev == 1.0).all()
