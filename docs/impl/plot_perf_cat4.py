"""Cat 4 — Sensor perturbations: overhead (%) vs n_envs on Genesis CF2X.

All 16 sensor perturbations subclass ``ObservationPerturbation``; their
``apply(obs)`` is invoked on a shared 12-wide obs buffer (wide enough for
``obs_channel_masking`` which uses a slice of width 6).

Run:
    uv run python docs/impl/plot_perf_cat4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_4_sensor import (  # noqa: E402
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

CAT = 4
DT = 0.01
OBS_SLICE_3 = slice(0, 3)
OBS_SLICE_2 = slice(0, 2)
OBS_SLICE_1 = slice(0, 1)
OBS_SLICE_6 = slice(0, 6)


def _mk_3(cls):
    def factory(scene, drone, n_envs: int):
        return cls(obs_slice=OBS_SLICE_3, n_envs=n_envs, dt=DT)

    return factory


def _mk_optical(scene, drone, n_envs: int) -> OpticalFlowNoise:
    return OpticalFlowNoise(obs_slice=OBS_SLICE_2, n_envs=n_envs, dt=DT)


def _mk_baro(scene, drone, n_envs: int) -> BarometerDrift:
    return BarometerDrift(obs_slice=OBS_SLICE_1, n_envs=n_envs, dt=DT)


def _mk_mask(scene, drone, n_envs: int) -> ObsChannelMasking:
    return ObsChannelMasking(obs_slice=OBS_SLICE_6, obs_dim=6, n_envs=n_envs, dt=DT)


def _mk_clock(scene, drone, n_envs: int) -> ClockDrift:
    return ClockDrift(obs_slice=OBS_SLICE_3, obs_dim=3, n_envs=n_envs, dt=DT)


SPECS: list[PertSpec] = [
    PertSpec("cat4_gyro_noise", "gyro_noise", "obs", _mk_3(GyroNoise)),
    PertSpec("cat4_gyro_bias", "gyro_bias", "obs", _mk_3(GyroBias)),
    PertSpec("cat4_gyro_drift", "gyro_drift", "obs", _mk_3(GyroDrift)),
    PertSpec(
        "cat4_accel_noise_bias_drift",
        "accel_noise_bias_drift",
        "obs",
        _mk_3(AccelNoiseBiasDrift),
    ),
    PertSpec("cat4_sensor_cross_axis", "sensor_cross_axis", "obs", _mk_3(SensorCrossAxis)),
    PertSpec("cat4_position_noise", "position_noise", "obs", _mk_3(PositionNoise)),
    PertSpec("cat4_position_dropout", "position_dropout", "obs", _mk_3(PositionDropout)),
    PertSpec("cat4_position_outlier", "position_outlier", "obs", _mk_3(PositionOutlier)),
    PertSpec("cat4_velocity_noise", "velocity_noise", "obs", _mk_3(VelocityNoise)),
    PertSpec("cat4_sensor_quantization", "sensor_quantization", "obs", _mk_3(SensorQuantization)),
    PertSpec("cat4_obs_channel_masking", "obs_channel_masking", "obs", _mk_mask),
    PertSpec(
        "cat4_magnetometer_interference",
        "magnetometer_interference",
        "obs",
        _mk_3(MagnetometerInterference),
    ),
    PertSpec("cat4_barometer_drift", "barometer_drift", "obs", _mk_baro),
    PertSpec("cat4_optical_flow_noise", "optical_flow_noise", "obs", _mk_optical),
    PertSpec("cat4_imu_vibration", "imu_vibration", "obs", _mk_3(IMUVibration)),
    PertSpec("cat4_clock_drift", "clock_drift", "obs", _mk_clock),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
