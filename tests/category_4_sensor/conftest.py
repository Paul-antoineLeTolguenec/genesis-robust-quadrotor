"""Fixtures for category 4 — sensor perturbations."""

import pytest
import torch

from genesis_robust_rl.perturbations.category_4_sensor import (
    AccelNoiseBiasDrift,
    BarometerDrift,
    ClockDrift,
    GyroBias,
    GyroDrift,
    GyroNoise,
    IMUVibration,
    MagnetometerInterference,
    ObsChannelMasking,
    OpticalFlowNoise,
    PositionDropout,
    PositionNoise,
    PositionOutlier,
    SensorCrossAxis,
    SensorQuantization,
    VelocityNoise,
)
from genesis_robust_rl.sensor_models import (
    AccelerometerModel,
    BarometerModel,
    GPSModel,
    GyroscopeModel,
    MagnetometerModel,
    OpticalFlowModel,
)

# ---------- Perturbation fixture ----------

OBS_DIM = 32  # total obs vector width for tests

# Slice assignments for testing (non-overlapping)
GYRO_SLICE = slice(0, 3)
ACCEL_SLICE = slice(3, 6)
POS_SLICE = slice(6, 9)
VEL_SLICE = slice(9, 12)
MAG_SLICE = slice(12, 15)
BARO_SLICE = slice(15, 16)
FLOW_SLICE = slice(16, 18)
QUANT_SLICE = slice(0, 6)
MASK_SLICE = slice(0, 10)
CLOCK_SLICE = slice(0, 6)


def _make_gyro_noise(n):
    return GyroNoise(obs_slice=GYRO_SLICE, n_envs=n, dt=0.01)


def _make_gyro_bias(n):
    return GyroBias(obs_slice=GYRO_SLICE, n_envs=n, dt=0.01)


def _make_gyro_drift(n):
    return GyroDrift(obs_slice=GYRO_SLICE, n_envs=n, dt=0.01)


def _make_accel_nbd(n):
    return AccelNoiseBiasDrift(obs_slice=ACCEL_SLICE, n_envs=n, dt=0.01)


def _make_cross_axis(n):
    return SensorCrossAxis(obs_slice=GYRO_SLICE, n_envs=n, dt=0.01)


def _make_position_noise(n):
    return PositionNoise(obs_slice=POS_SLICE, n_envs=n, dt=0.01)


def _make_position_dropout(n):
    return PositionDropout(obs_slice=POS_SLICE, n_envs=n, dt=0.01)


def _make_position_outlier(n):
    return PositionOutlier(obs_slice=POS_SLICE, n_envs=n, dt=0.01)


def _make_velocity_noise(n):
    return VelocityNoise(obs_slice=VEL_SLICE, n_envs=n, dt=0.01)


def _make_quantization(n):
    return SensorQuantization(obs_slice=QUANT_SLICE, n_envs=n, dt=0.01)


def _make_channel_masking(n):
    return ObsChannelMasking(obs_slice=MASK_SLICE, obs_dim=10, n_envs=n, dt=0.01)


def _make_mag_interference(n):
    return MagnetometerInterference(obs_slice=MAG_SLICE, n_envs=n, dt=0.01)


def _make_baro_drift(n):
    return BarometerDrift(obs_slice=BARO_SLICE, n_envs=n, dt=0.01)


def _make_flow_noise(n):
    return OpticalFlowNoise(obs_slice=FLOW_SLICE, n_envs=n, dt=0.01)


def _make_imu_vibration(n):
    p = IMUVibration(obs_slice=ACCEL_SLICE, n_envs=n, dt=0.01)
    p.set_rpm(torch.ones(n, 4) * 3000.0)
    return p


def _make_clock_drift(n):
    return ClockDrift(obs_slice=CLOCK_SLICE, obs_dim=6, n_envs=n, dt=0.01)


@pytest.fixture(
    params=[
        _make_gyro_noise,
        _make_gyro_bias,
        _make_gyro_drift,
        _make_accel_nbd,
        _make_cross_axis,
        _make_position_noise,
        _make_position_dropout,
        _make_position_outlier,
        _make_velocity_noise,
        _make_quantization,
        _make_channel_masking,
        _make_mag_interference,
        _make_baro_drift,
        _make_flow_noise,
        _make_imu_vibration,
        _make_clock_drift,
    ]
)
def perturbation(request, n_envs):
    """Parametrized fixture over all category-4 perturbation leaves."""
    return request.param(n_envs)


@pytest.fixture(
    params=[
        AccelNoiseBiasDrift,
        GyroDrift,
        PositionDropout,
        BarometerDrift,
        ClockDrift,
    ]
)
def perturbation_class(request):
    """Used by P3 memory test — stateful classes only."""
    return request.param


# ---------- SensorModel fixture ----------


@pytest.fixture(
    params=[
        GyroscopeModel,
        AccelerometerModel,
        MagnetometerModel,
        BarometerModel,
        GPSModel,
        OpticalFlowModel,
    ]
)
def sensor_model(request, n_envs):
    """Parametrized fixture over all SensorModel leaves."""
    return request.param(n_envs=n_envs)


# ---------- Obs perturbation for pipeline test ----------


@pytest.fixture
def obs_perturbation(n_envs):
    """Simple additive noise for I4 pipeline test."""
    return GyroNoise(obs_slice=GYRO_SLICE, n_envs=n_envs, dt=0.01)
