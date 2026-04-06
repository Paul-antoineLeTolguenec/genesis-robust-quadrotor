"""Generate curriculum and per-env diversity plots for category 4 sensor perturbations."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import plotly.io as pio
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

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

N_ENVS = 16
DT = 0.01
N_STEPS = 200
OBS_DIM = 32


def _save(fig: go.Figure, name: str) -> None:
    path = os.path.join(ASSETS_DIR, f"{name}.png")
    pio.write_image(fig, path, width=800, height=450)
    print(f"  saved: {path}")


# ===================================================================
# Helpers
# ===================================================================


def _curriculum_plot(
    title: str,
    filename: str,
    factory,
    scales: list[float] | None = None,
    metric_fn=None,
):
    """Plot metric vs curriculum_scale for a perturbation."""
    if scales is None:
        scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    if metric_fn is None:

        def metric_fn(p):
            return p._current_value.abs().mean().item() if p._current_value is not None else 0.0

    means = []
    for s in scales:
        vals = []
        for _ in range(50):
            p = factory()
            p.curriculum_scale = s
            p.tick(is_reset=True)
            for _ in range(20):
                p.tick(is_reset=False)
            vals.append(metric_fn(p))
        means.append(sum(vals) / len(vals))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scales, y=means, mode="lines+markers", name="mean |effect|"))
    fig.update_layout(
        title=title,
        xaxis_title="curriculum_scale",
        yaxis_title="Mean |perturbation effect|",
        template="plotly_white",
    )
    _save(fig, filename)


def _perenv_plot(title: str, filename: str, factory, n_steps: int = 100, metric_fn=None):
    """Plot per-env diversity: each env's trajectory over time."""
    if metric_fn is None:

        def metric_fn(p, i):
            if p._current_value is not None and p._current_value.ndim > 0:
                return p._current_value[i].abs().mean().item()
            return 0.0

    p = factory()
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)

    traces = {i: [] for i in range(min(N_ENVS, 8))}
    for _ in range(n_steps):
        p.tick(is_reset=False)
        obs = torch.randn(p.n_envs, OBS_DIM)
        p.apply(obs)
        for i in traces:
            traces[i].append(metric_fn(p, i))

    fig = go.Figure()
    for i, vals in traces.items():
        fig.add_trace(go.Scatter(y=vals, mode="lines", name=f"env {i}", opacity=0.7))
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="|perturbation|",
        template="plotly_white",
    )
    _save(fig, filename)


# ===================================================================
# 4.1 GyroNoise
# ===================================================================

print("Category 4 — Sensor plots")

# --- 4.1 GyroNoise ---
print("4.1 GyroNoise")
_curriculum_plot(
    "4.1 GyroNoise — Curriculum Effect",
    "cat4_gyro_noise_curriculum",
    lambda: GyroNoise(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.1 GyroNoise — Per-env Diversity",
    "cat4_gyro_noise_per_env",
    lambda: GyroNoise(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
)

# --- 4.2 GyroBias ---
print("4.2 GyroBias")
_curriculum_plot(
    "4.2 GyroBias — Curriculum Effect",
    "cat4_gyro_bias_curriculum",
    lambda: GyroBias(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.2 GyroBias — Per-env Diversity",
    "cat4_gyro_bias_per_env",
    lambda: GyroBias(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: (
        p._current_value[i].abs().mean().item() if p._current_value is not None else 0.0
    ),
)

# --- 4.3 GyroDrift ---
print("4.3 GyroDrift")
_curriculum_plot(
    "4.3 GyroDrift — Curriculum Effect",
    "cat4_gyro_drift_curriculum",
    lambda: GyroDrift(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.3 GyroDrift — Per-env Diversity",
    "cat4_gyro_drift_per_env",
    lambda: GyroDrift(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
)

# --- 4.4 AccelNoiseBiasDrift ---
print("4.4 AccelNoiseBiasDrift")
_curriculum_plot(
    "4.4 AccelNoiseBiasDrift — Curriculum Effect",
    "cat4_accel_nbd_curriculum",
    lambda: AccelNoiseBiasDrift(obs_slice=slice(3, 6), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.4 AccelNoiseBiasDrift — Per-env Diversity",
    "cat4_accel_nbd_per_env",
    lambda: AccelNoiseBiasDrift(obs_slice=slice(3, 6), n_envs=N_ENVS, dt=DT),
)

# --- 4.5 SensorCrossAxis ---
print("4.5 SensorCrossAxis")
_curriculum_plot(
    "4.5 SensorCrossAxis — Curriculum Effect",
    "cat4_cross_axis_curriculum",
    lambda: SensorCrossAxis(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p: (p._rot_matrices - torch.eye(3)).abs().mean().item(),
)
_perenv_plot(
    "4.5 SensorCrossAxis — Per-env Diversity",
    "cat4_cross_axis_per_env",
    lambda: SensorCrossAxis(obs_slice=slice(0, 3), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: (p._rot_matrices[i] - torch.eye(3)).abs().mean().item(),
)

# --- 4.6 PositionNoise ---
print("4.6 PositionNoise")
_curriculum_plot(
    "4.6 PositionNoise — Curriculum Effect",
    "cat4_position_noise_curriculum",
    lambda: PositionNoise(obs_slice=slice(6, 9), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.6 PositionNoise — Per-env Diversity",
    "cat4_position_noise_per_env",
    lambda: PositionNoise(obs_slice=slice(6, 9), n_envs=N_ENVS, dt=DT),
)

# --- 4.7 PositionDropout ---
print("4.7 PositionDropout")
_curriculum_plot(
    "4.7 PositionDropout — Curriculum Effect",
    "cat4_position_dropout_curriculum",
    lambda: PositionDropout(obs_slice=slice(6, 9), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p: p._current_value.mean().item() if p._current_value is not None else 0.0,
)
_perenv_plot(
    "4.7 PositionDropout — Per-env Diversity",
    "cat4_position_dropout_per_env",
    lambda: PositionDropout(obs_slice=slice(6, 9), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: p._current_value[i].item() if p._current_value is not None else 0.0,
)

# --- 4.8 PositionOutlier ---
print("4.8 PositionOutlier")
_curriculum_plot(
    "4.8 PositionOutlier — Curriculum Effect",
    "cat4_position_outlier_curriculum",
    lambda: PositionOutlier(obs_slice=slice(6, 9), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.8 PositionOutlier — Per-env Diversity",
    "cat4_position_outlier_per_env",
    lambda: PositionOutlier(obs_slice=slice(6, 9), n_envs=N_ENVS, dt=DT),
)

# --- 4.9 VelocityNoise ---
print("4.9 VelocityNoise")
_curriculum_plot(
    "4.9 VelocityNoise — Curriculum Effect",
    "cat4_velocity_noise_curriculum",
    lambda: VelocityNoise(obs_slice=slice(9, 12), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.9 VelocityNoise — Per-env Diversity",
    "cat4_velocity_noise_per_env",
    lambda: VelocityNoise(obs_slice=slice(9, 12), n_envs=N_ENVS, dt=DT),
)

# --- 4.10 SensorQuantization ---
print("4.10 SensorQuantization")
_curriculum_plot(
    "4.10 SensorQuantization — Curriculum Effect",
    "cat4_quantization_curriculum",
    lambda: SensorQuantization(obs_slice=slice(0, 6), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.10 SensorQuantization — Per-env Diversity",
    "cat4_quantization_per_env",
    lambda: SensorQuantization(obs_slice=slice(0, 6), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: (
        p._current_value[i].abs().mean().item() if p._current_value is not None else 0.0
    ),
)

# --- 4.11 ObsChannelMasking ---
print("4.11 ObsChannelMasking")
_curriculum_plot(
    "4.11 ObsChannelMasking — Curriculum Effect",
    "cat4_channel_masking_curriculum",
    lambda: ObsChannelMasking(obs_slice=slice(0, 10), obs_dim=10, n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p: (
        (1.0 - p._current_value).mean().item() if p._current_value is not None else 0.0
    ),
)
_perenv_plot(
    "4.11 ObsChannelMasking — Per-env Diversity",
    "cat4_channel_masking_per_env",
    lambda: ObsChannelMasking(obs_slice=slice(0, 10), obs_dim=10, n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: (
        (1.0 - p._current_value[i]).mean().item() if p._current_value is not None else 0.0
    ),
)

# --- 4.12 MagnetometerInterference ---
print("4.12 MagnetometerInterference")
_curriculum_plot(
    "4.12 MagnetometerInterference — Curriculum Effect",
    "cat4_mag_interference_curriculum",
    lambda: MagnetometerInterference(obs_slice=slice(12, 15), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.12 MagnetometerInterference — Per-env Diversity",
    "cat4_mag_interference_per_env",
    lambda: MagnetometerInterference(obs_slice=slice(12, 15), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: (
        p._current_value[i].abs().mean().item() if p._current_value is not None else 0.0
    ),
)

# --- 4.13 BarometerDrift ---
print("4.13 BarometerDrift")
_curriculum_plot(
    "4.13 BarometerDrift — Curriculum Effect",
    "cat4_baro_drift_curriculum",
    lambda: BarometerDrift(obs_slice=slice(15, 16), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.13 BarometerDrift — Per-env Diversity",
    "cat4_baro_drift_per_env",
    lambda: BarometerDrift(obs_slice=slice(15, 16), n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: (
        p._current_value[i].abs().item() if p._current_value is not None else 0.0
    ),
)

# --- 4.14 OpticalFlowNoise ---
print("4.14 OpticalFlowNoise")
_curriculum_plot(
    "4.14 OpticalFlowNoise — Curriculum Effect",
    "cat4_flow_noise_curriculum",
    lambda: OpticalFlowNoise(obs_slice=slice(16, 18), n_envs=N_ENVS, dt=DT),
)
_perenv_plot(
    "4.14 OpticalFlowNoise — Per-env Diversity",
    "cat4_flow_noise_per_env",
    lambda: OpticalFlowNoise(obs_slice=slice(16, 18), n_envs=N_ENVS, dt=DT),
)

# --- 4.15 IMUVibration ---
print("4.15 IMUVibration")


def _make_imu_vib():
    p = IMUVibration(obs_slice=slice(3, 6), n_envs=N_ENVS, dt=DT)
    p.set_rpm(torch.ones(N_ENVS, 4) * 3000.0)
    return p


_curriculum_plot(
    "4.15 IMUVibration — Curriculum Effect",
    "cat4_imu_vibration_curriculum",
    _make_imu_vib,
)
_perenv_plot(
    "4.15 IMUVibration — Per-env Diversity",
    "cat4_imu_vibration_per_env",
    _make_imu_vib,
)

# --- 4.16 ClockDrift ---
print("4.16 ClockDrift")
_curriculum_plot(
    "4.16 ClockDrift — Curriculum Effect",
    "cat4_clock_drift_curriculum",
    lambda: ClockDrift(obs_slice=slice(0, 6), obs_dim=6, n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p: p._phase_offset.abs().mean().item(),
)
_perenv_plot(
    "4.16 ClockDrift — Per-env Diversity",
    "cat4_clock_drift_per_env",
    lambda: ClockDrift(obs_slice=slice(0, 6), obs_dim=6, n_envs=N_ENVS, dt=DT),
    metric_fn=lambda p, i: p._phase_offset[i].abs().item(),
)

print("\nDone — all category 4 plots generated.")
