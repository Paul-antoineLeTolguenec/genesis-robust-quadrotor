"""Category 4 — Sensor perturbation plots.

Generates curriculum violins + specialised PNGs for 16 perturbations:
- 4.1  gyro_noise               (curriculum violin)
- 4.2  gyro_bias                (curriculum violin)
- 4.3  gyro_drift               (curriculum violin + OU trace)
- 4.4  accel_noise_bias_drift   (curriculum violin + composite trace)
- 4.5  sensor_cross_axis        (curriculum violin, angle magnitude)
- 4.6  position_noise           (curriculum violin)
- 4.7  position_dropout         (curriculum violin + dropout heatmap)
- 4.8  position_outlier         (curriculum violin + spike heatmap)
- 4.9  velocity_noise           (curriculum violin)
- 4.10 sensor_quantization      (curriculum violin)
- 4.11 obs_channel_masking      (curriculum violin + mask heatmap)
- 4.12 magnetometer_interference (curriculum violin)
- 4.13 barometer_drift          (curriculum violin + OU trace)
- 4.14 optical_flow_noise       (curriculum violin)
- 4.15 imu_vibration            (curriculum violin)
- 4.16 clock_drift              (curriculum violin + phase trace)

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_4.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _plot_framework import (  # noqa: E402
    CATEGORY_COLORS,
    HardwareMeta,
    apply_layout,
    collect_hardware_meta,
    log_and_plot,
    make_figure,
    stats_summary,
)

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
COLOR = CATEGORY_COLORS[CAT]
N_ENVS = 256
N_DRAWS = 40
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
TRACE_N_ENVS = 4
TRACE_STEPS = 400
EVENT_N_ENVS = 8
EVENT_N_STEPS = 300

# Common obs_slice for 3D sensors
OBS_SLICE_3 = slice(0, 3)
OBS_DIM_3 = 3
OBS_SLICE_2 = slice(0, 2)
OBS_DIM_2 = 2


# ===================================================================
# Shared: curriculum violin
# ===================================================================


def _curriculum_violin(
    hw: HardwareMeta,
    slug: str,
    make_fn: Callable[[int, float], object],
    extract_fn: Callable[[object], torch.Tensor],
    unit: str,
    label: str,
) -> None:
    rows: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(4000 + i)
        p = make_fn(N_ENVS, scale)
        values: list[float] = []
        for _ in range(N_DRAWS):
            p.sample()
            v = extract_fn(p)
            for m in v.tolist():
                values.append(float(m))
        samples_by_scale[f"{scale:.2f}"] = values
        for v in values:
            rows.append({"curriculum_scale": float(scale), "value": v})

    stats_by = {s: stats_summary(v) for s, v in samples_by_scale.items()}

    def fig_fn(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for scale_s, vals in samples_by_scale.items():
            fig.add_trace(
                go.Violin(
                    y=vals,
                    x=[scale_s] * len(vals),
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    line_color=COLOR,
                    fillcolor=COLOR,
                    opacity=0.5,
                    width=0.8,
                    showlegend=False,
                )
            )
        all_v = [v for vs in samples_by_scale.values() for v in vs]
        y_min, y_max = min(all_v), max(all_v)
        pad = 0.15 * (y_max - y_min + 1e-12)
        apply_layout(
            fig,
            title=f"Cat 4 — {label}: {unit} vs curriculum_scale",
            subtitle=f"n_envs={N_ENVS}, draws/env={N_DRAWS}",
            xaxis_title="curriculum_scale",
            yaxis_title=f"{label} [{unit}]",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_yaxes(range=[y_min - pad, y_max + pad])
        return fig

    log_and_plot(
        name=f"cat4_{slug}_curriculum",
        category=CAT,
        metric=f"{label} distribution",
        unit=unit,
        baseline_description=f"curriculum_scale=0 → nominal ({unit})",
        config={
            "n_envs": N_ENVS,
            "draws_per_env": N_DRAWS,
            "curriculum_scales": list(CURRICULUM_SCALES),
        },
        hardware=hw,
        csv_rows=rows,
        csv_columns=["curriculum_scale", "value"],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/curriculum] logged {len(rows)} samples")


# ===================================================================
# Shared: OU trace for stateful perturbations
# ===================================================================


def _ou_trace(
    hw: HardwareMeta,
    slug: str,
    make_fn: Callable[[int], object],
    extract_fn: Callable[[object], torch.Tensor],
    unit: str,
    label: str,
    seed: int,
    n_envs: int = TRACE_N_ENVS,
    n_steps: int = TRACE_STEPS,
) -> None:
    torch.manual_seed(seed)
    p = make_fn(n_envs)
    p.tick(is_reset=True)

    trace = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.tick(is_reset=False)
        trace[step] = extract_fn(p).clone()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "value": float(trace[step, e]),
                }
            )

    stats_per_env = {f"env_{e}": stats_summary(trace[:, e].tolist()) for e in range(n_envs)}

    def fig_fn(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=t, y=trace[:, e].tolist(), mode="lines", name=f"env {e}", line=dict(width=1.3)
                )
            )
        apply_layout(
            fig,
            title=f"Cat 4 — {label}: drift trace",
            subtitle=f"n_envs={n_envs}, n_steps={n_steps}, dt={DT}s",
            xaxis_title="time [s]",
            yaxis_title=f"{label} [{unit}]",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"cat4_{slug}_trace",
        category=CAT,
        metric=f"{label} trace",
        unit=unit,
        baseline_description="stateful drift at curriculum_scale=1.0",
        config={"n_envs": n_envs, "n_steps": n_steps, "dt_s": DT},
        hardware=hw,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "value"],
        stats_by_series=stats_per_env,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/trace] logged {len(rows)} rows")


# ===================================================================
# Shared: event heatmap (binary)
# ===================================================================


def _event_heatmap(
    hw: HardwareMeta,
    slug: str,
    make_fn: Callable[[int], object],
    event_fn: Callable[[object], torch.Tensor],
    event_color: str,
    active_label: str,
    inactive_label: str,
    seed: int,
    n_envs: int = EVENT_N_ENVS,
    n_steps: int = EVENT_N_STEPS,
) -> None:
    torch.manual_seed(seed)
    p = make_fn(n_envs)
    p.tick(is_reset=True)

    events = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.tick(is_reset=False)
        events[step] = event_fn(p).clone()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "active": int(events[step, e].item()),
                }
            )

    rates = [events[:, e].float().mean().item() for e in range(n_envs)]
    stats_rate = stats_summary(rates)

    def fig_fn(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Heatmap(
                z=events.float().numpy().T,
                x=[round(s * DT, 4) for s in range(n_steps)],
                y=[f"env {e}" for e in range(n_envs)],
                colorscale=[[0, "#ffffff"], [1, event_color]],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=("time=%{x}s<br>env=%{y}<br>active=%{z:.0f}<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title=f"Cat 4 — {slug}: event heatmap",
            subtitle=(
                f"{event_color.upper()}={active_label}, white={inactive_label}; "
                f"n_envs={n_envs}, n_steps={n_steps}"
            ),
            xaxis_title="time [s]",
            yaxis_title="environment",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name=f"cat4_{slug}_heatmap",
        category=CAT,
        metric=f"{slug} events",
        unit="boolean",
        baseline_description=f"1={active_label}, 0={inactive_label}",
        config={"n_envs": n_envs, "n_steps": n_steps, "dt_s": DT},
        hardware=hw,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "active"],
        stats_by_series={"event_rate": stats_rate},
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/heatmap] logged {len(rows)} rows")


# ===================================================================
# 4.1–4.2: Gyro noise / bias (curriculum violins)
# ===================================================================


def plot_gyro_noise(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "gyro_noise",
        make_fn=lambda n, s: GyroNoise(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="rad/s",
        label="noise σ (axis 0)",
    )


def plot_gyro_bias(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "gyro_bias",
        make_fn=lambda n, s: GyroBias(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="rad/s",
        label="bias (axis 0)",
    )


# ===================================================================
# 4.3 GyroDrift — curriculum violin + OU trace
# ===================================================================


def plot_gyro_drift(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "gyro_drift",
        make_fn=lambda n, s: GyroDrift(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="rad/s",
        label="OU drift (axis 0)",
    )
    _ou_trace(
        hw,
        "gyro_drift",
        make_fn=lambda n: GyroDrift(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=1.0,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="rad/s",
        label="gyro drift (axis 0)",
        seed=4300,
    )


# ===================================================================
# 4.4 AccelNoiseBiasDrift — curriculum violin + composite trace
# ===================================================================


def plot_accel_composite(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "accel_noise_bias_drift",
        make_fn=lambda n, s: AccelNoiseBiasDrift(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="m/s²",
        label="accel perturbation (axis 0)",
    )
    _ou_trace(
        hw,
        "accel_noise_bias_drift",
        make_fn=lambda n: AccelNoiseBiasDrift(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=1.0,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="m/s²",
        label="accel composite (axis 0)",
        seed=4400,
    )


# ===================================================================
# 4.5 SensorCrossAxis — curriculum violin (angle magnitude)
# ===================================================================


def plot_sensor_cross_axis(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "sensor_cross_axis",
        make_fn=lambda n, s: SensorCrossAxis(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0, 1],
        unit="dimensionless",
        label="rotation R[0,1]",
    )


# ===================================================================
# 4.6, 4.9, 4.14: White noise sensors
# ===================================================================


def plot_position_noise(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "position_noise",
        make_fn=lambda n, s: PositionNoise(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="m",
        label="position noise σ (axis 0)",
    )


def plot_velocity_noise(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "velocity_noise",
        make_fn=lambda n, s: VelocityNoise(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="m/s",
        label="velocity noise σ (axis 0)",
    )


def plot_optical_flow_noise(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "optical_flow_noise",
        make_fn=lambda n, s: OpticalFlowNoise(
            obs_slice=OBS_SLICE_2,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="m/s",
        label="optical flow noise σ (u)",
    )


# ===================================================================
# 4.7 PositionDropout — curriculum violin + dropout heatmap
# ===================================================================


def plot_position_dropout(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "position_dropout",
        make_fn=lambda n, s: PositionDropout(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="probability",
        label="dropout prob",
    )
    _event_heatmap(
        hw,
        "position_dropout",
        make_fn=lambda n: PositionDropout(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=1.0,
        ),
        event_fn=lambda p: (p._counter > 0).float(),
        event_color="#d62728",
        active_label="dropout",
        inactive_label="valid",
        seed=4700,
    )


# ===================================================================
# 4.8 PositionOutlier — curriculum violin + spike heatmap
# ===================================================================


def plot_position_outlier(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "position_outlier",
        make_fn=lambda n, s: PositionOutlier(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.norm(dim=-1),
        unit="m",
        label="outlier ‖spike‖₂",
    )
    _event_heatmap(
        hw,
        "position_outlier",
        make_fn=lambda n: PositionOutlier(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=1.0,
        ),
        event_fn=lambda p: (p._current_value.norm(dim=-1) > 0.01).float(),
        event_color="#ff7f0e",
        active_label="spike",
        inactive_label="clean",
        seed=4800,
    )


# ===================================================================
# 4.10 SensorQuantization
# ===================================================================


def plot_sensor_quantization(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "sensor_quantization",
        make_fn=lambda n, s: SensorQuantization(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="resolution",
        label="quantization step",
    )


# ===================================================================
# 4.11 ObsChannelMasking — curriculum violin + mask heatmap
# ===================================================================


def plot_obs_channel_masking(hw: HardwareMeta) -> None:
    obs_dim = 6

    _curriculum_violin(
        hw,
        "obs_channel_masking",
        make_fn=lambda n, s: ObsChannelMasking(
            obs_slice=slice(0, obs_dim),
            obs_dim=obs_dim,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.float().mean(dim=-1),
        unit="fraction",
        label="mask rate (mean channels masked)",
    )


# ===================================================================
# 4.12 MagnetometerInterference
# ===================================================================


def plot_magnetometer_interference(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "magnetometer_interference",
        make_fn=lambda n, s: MagnetometerInterference(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="µT",
        label="mag offset (axis 0)",
    )


# ===================================================================
# 4.13 BarometerDrift — curriculum violin + OU trace
# ===================================================================


def plot_barometer_drift(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "barometer_drift",
        make_fn=lambda n, s: BarometerDrift(
            obs_slice=slice(0, 1),
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="m",
        label="baro drift",
    )
    _ou_trace(
        hw,
        "barometer_drift",
        make_fn=lambda n: BarometerDrift(
            obs_slice=slice(0, 1),
            n_envs=n,
            dt=DT,
            curriculum_scale=1.0,
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="m",
        label="barometer drift",
        seed=4130,
    )


# ===================================================================
# 4.15 IMUVibration
# ===================================================================


def plot_imu_vibration(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "imu_vibration",
        make_fn=lambda n, s: IMUVibration(
            obs_slice=OBS_SLICE_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="gain",
        label="vibration gain",
    )


# ===================================================================
# 4.16 ClockDrift — curriculum violin + phase trace
# ===================================================================


def plot_clock_drift(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "clock_drift",
        make_fn=lambda n, s: ClockDrift(
            obs_slice=OBS_SLICE_3,
            obs_dim=OBS_DIM_3,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ppm",
        label="drift rate",
    )

    n_envs = TRACE_N_ENVS
    n_steps = TRACE_STEPS
    torch.manual_seed(4160)
    p = ClockDrift(
        obs_slice=OBS_SLICE_3,
        obs_dim=OBS_DIM_3,
        n_envs=n_envs,
        dt=DT,
        curriculum_scale=1.0,
    )
    p.tick(is_reset=True)

    phase = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.tick(is_reset=False)
        phase[step] = p._phase_offset.squeeze(-1).clone()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "phase_offset_s": float(phase[step, e]),
                }
            )

    stats_phase = {f"env_{e}": stats_summary(phase[:, e].tolist()) for e in range(n_envs)}

    def fig_fn(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=t, y=phase[:, e].tolist(), mode="lines", name=f"env {e}", line=dict(width=1.5)
                )
            )
        apply_layout(
            fig,
            title="Cat 4 — clock_drift: accumulated phase offset",
            subtitle=(f"phase += rate_ppm × 1e-6 × dt; n_envs={n_envs}, n_steps={n_steps}"),
            xaxis_title="time [s]",
            yaxis_title="phase offset [s]",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat4_clock_drift_trace",
        category=CAT,
        metric="accumulated phase offset",
        unit="s",
        baseline_description="linear accumulation: rate_ppm × 1e-6 per second",
        config={"n_envs": n_envs, "n_steps": n_steps, "dt_s": DT},
        hardware=hw,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "phase_offset_s"],
        stats_by_series=stats_phase,
        figure_fn=fig_fn,
    )
    print("  [clock_drift/trace] done")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    specs = [
        ("4.1 gyro_noise", plot_gyro_noise),
        ("4.2 gyro_bias", plot_gyro_bias),
        ("4.3 gyro_drift", plot_gyro_drift),
        ("4.4 accel_noise_bias_drift", plot_accel_composite),
        ("4.5 sensor_cross_axis", plot_sensor_cross_axis),
        ("4.6 position_noise", plot_position_noise),
        ("4.7 position_dropout", plot_position_dropout),
        ("4.8 position_outlier", plot_position_outlier),
        ("4.9 velocity_noise", plot_velocity_noise),
        ("4.10 sensor_quantization", plot_sensor_quantization),
        ("4.11 obs_channel_masking", plot_obs_channel_masking),
        ("4.12 magnetometer_interference", plot_magnetometer_interference),
        ("4.13 barometer_drift", plot_barometer_drift),
        ("4.14 optical_flow_noise", plot_optical_flow_noise),
        ("4.15 imu_vibration", plot_imu_vibration),
        ("4.16 clock_drift", plot_clock_drift),
    ]

    for label, fn in specs:
        print(f"\n--- {label} ---")
        fn(hardware)

    print("\nDone. Inspect docs/impl/{data,assets}/cat4_*.{csv,meta.json,png}")


if __name__ == "__main__":
    main()
