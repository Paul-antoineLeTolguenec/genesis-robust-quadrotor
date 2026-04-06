"""SensorModel unit tests — USM1-USM4 + I4 pipeline."""

import pytest
import torch

from genesis_robust_rl.sensor_models import (
    AccelerometerModel,
    BarometerModel,
    GPSModel,
    GyroscopeModel,
    MagnetometerModel,
    OpticalFlowModel,
)

# ===================================================================
# USM1 — forward() output shape and dtype
# ===================================================================


@pytest.mark.unit
def test_sensor_forward_shape_dtype(sensor_model, mock_env_state, n_envs):
    """forward() must return [n_envs, dim] float32 with no NaN."""
    out = sensor_model.forward(mock_env_state)
    assert out.shape[0] == n_envs
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any(), "NaN in sensor_model.forward() output"
    assert torch.isfinite(out).all(), "Inf in sensor_model.forward() output"


# ===================================================================
# USM2 — forward() output range plausible
# ===================================================================


@pytest.mark.unit
def test_sensor_forward_range(sensor_model, mock_env_state):
    """Output must be within physical plausible range for a hovering drone."""
    out = sensor_model.forward(mock_env_state)
    if isinstance(sensor_model, GyroscopeModel):
        assert out.abs().max() < 50, "Gyro output > 50 rad/s"
    elif isinstance(sensor_model, AccelerometerModel):
        assert out.abs().max() < 200, "Accel output > 200 m/s²"
    elif isinstance(sensor_model, MagnetometerModel):
        assert out.abs().max() < 100, "Mag output > 100 µT"
    elif isinstance(sensor_model, BarometerModel):
        assert (out > -500).all() and (out < 5000).all(), "Baro outside [-500, 5000] m"
    elif isinstance(sensor_model, GPSModel):
        assert out.abs().max() < 1e6, "GPS output > 1e6 m"
    elif isinstance(sensor_model, OpticalFlowModel):
        assert out.abs().max() < 100, "Flow output > 100 m/s"


# ===================================================================
# USM3 — AccelerometerModel is stateless
# ===================================================================


@pytest.mark.unit
def test_accelerometer_stateless(mock_env_state, n_envs):
    """AccelerometerModel must not maintain _vel_prev; successive calls are independent."""
    model = AccelerometerModel(n_envs=n_envs)
    out1 = model.forward(mock_env_state)
    out2 = model.forward(mock_env_state)
    assert torch.allclose(out1, out2), "AccelerometerModel is unexpectedly stateful"
    assert not hasattr(model, "_vel_prev"), "AccelerometerModel should not have _vel_prev"


# ===================================================================
# USM4 — update_params() validates typed dataclass
# ===================================================================


@pytest.mark.unit
def test_sensor_update_params_validates(sensor_model):
    """update_params() with invalid type must raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        sensor_model.update_params({"invalid_field_xyz": 999.0})


# ===================================================================
# USM — Specific sensor output dimensions
# ===================================================================


@pytest.mark.unit
def test_gyroscope_output_dim(mock_env_state, n_envs):
    model = GyroscopeModel(n_envs=n_envs)
    assert model.forward(mock_env_state).shape == (n_envs, 3)


@pytest.mark.unit
def test_accelerometer_output_dim(mock_env_state, n_envs):
    model = AccelerometerModel(n_envs=n_envs)
    assert model.forward(mock_env_state).shape == (n_envs, 3)


@pytest.mark.unit
def test_magnetometer_output_dim(mock_env_state, n_envs):
    model = MagnetometerModel(n_envs=n_envs)
    assert model.forward(mock_env_state).shape == (n_envs, 3)


@pytest.mark.unit
def test_barometer_output_dim(mock_env_state, n_envs):
    model = BarometerModel(n_envs=n_envs)
    assert model.forward(mock_env_state).shape == (n_envs, 1)


@pytest.mark.unit
def test_gps_output_dim(mock_env_state, n_envs):
    model = GPSModel(n_envs=n_envs)
    assert model.forward(mock_env_state).shape == (n_envs, 3)


@pytest.mark.unit
def test_optical_flow_output_dim(mock_env_state, n_envs):
    model = OpticalFlowModel(n_envs=n_envs)
    assert model.forward(mock_env_state).shape == (n_envs, 2)


# ===================================================================
# USM — update_params() with valid params
# ===================================================================


@pytest.mark.unit
def test_gyroscope_update_params(n_envs):
    model = GyroscopeModel(n_envs=n_envs)
    new_C = torch.eye(3) * 1.01
    model.update_params({"C_misalign": new_C})
    assert torch.allclose(model._C_misalign, new_C)


@pytest.mark.unit
def test_magnetometer_update_params(n_envs):
    model = MagnetometerModel(n_envs=n_envs)
    new_b = torch.tensor([1.0, 2.0, 3.0])
    model.update_params({"b_hard": new_b})
    assert torch.allclose(model._b_hard, new_b)


# ===================================================================
# I4 — SensorModel → ObservationPerturbation pipeline
# ===================================================================


@pytest.mark.integration
def test_sensor_pipeline(sensor_model, obs_perturbation, mock_env_state, n_envs):
    """sensor_model.forward() → obs_perturbation.apply() must produce valid tensor."""
    raw = sensor_model.forward(mock_env_state)
    assert raw.shape[0] == n_envs
    assert raw.dtype == torch.float32
    # Build obs tensor with sensor output in the right slice
    obs = torch.zeros(n_envs, 32)
    obs_perturbation.tick(is_reset=True)
    obs_out = obs_perturbation.apply(obs)
    assert obs_out.shape == obs.shape
    assert not torch.isnan(obs_out).any(), "NaN in pipeline output"
