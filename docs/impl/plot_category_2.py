"""Category 2 — Motor perturbation plots.

Generates curriculum + specialised PNGs for 13 perturbations:
- 2.1  thrust_coeff_kf          (curriculum violin)
- 2.2  torque_coeff_km          (curriculum violin)
- 2.3  propeller_thrust_asymmetry (curriculum violin, motor 0)
- 2.4  motor_partial_failure    (curriculum violin, motor 0)
- 2.5  motor_kill               (kill count bar + kill mask heatmap)
- 2.6  motor_lag                (curriculum violin)
- 2.7  motor_rpm_noise          (curriculum violin)
- 2.8  motor_saturation         (curriculum violin)
- 2.9  motor_wear               (curriculum violin after 100 steps + decay trace)
- 2.10 rotor_imbalance          (curriculum violin + RPM modulation trace)
- 2.11 motor_back_emf           (curriculum violin)
- 2.12 motor_cold_start         (curriculum violin + warmup decay trace)
- 2.13 gyroscopic_effect        (curriculum violin)

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_2.py
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

from genesis_robust_rl.perturbations.category_2_motor import (  # noqa: E402
    GyroscopicEffect,
    MotorBackEMF,
    MotorColdStart,
    MotorKill,
    MotorLag,
    MotorPartialFailure,
    MotorRPMNoise,
    MotorSaturation,
    MotorWear,
    PropellerThrustAsymmetry,
    RotorImbalance,
    ThrustCoefficientKF,
    TorqueCoefficientKM,
)

CAT = 2
DT = 0.01
COLOR = CATEGORY_COLORS[CAT]
N_ENVS_CURRICULUM = 256
N_EPISODES_PER_ENV = 40
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


# ===================================================================
# Shared: curriculum violin (generic)
# ===================================================================


def _curriculum_violin(
    hardware: HardwareMeta,
    slug: str,
    make_fn: Callable[[int, float], object],
    extract_fn: Callable[[object], torch.Tensor],
    unit: str,
    label: str,
    n_envs: int = N_ENVS_CURRICULUM,
    n_draws: int = N_EPISODES_PER_ENV,
) -> None:
    rows: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(2000 + i)
        p = make_fn(n_envs, scale)
        values: list[float] = []
        for _ in range(n_draws):
            p.sample()
            v = extract_fn(p)
            for m in v.tolist():
                values.append(float(m))
        samples_by_scale[f"{scale:.2f}"] = values
        for v in values:
            rows.append({"curriculum_scale": float(scale), "value": v})

    stats_by_series = {s: stats_summary(v) for s, v in samples_by_scale.items()}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for scale_s, vals in samples_by_scale.items():
            fig.add_trace(
                go.Violin(
                    y=vals,
                    x=[scale_s] * len(vals),
                    name=f"scale={scale_s}",
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
        all_vals = [v for vs in samples_by_scale.values() for v in vs]
        y_min, y_max = min(all_vals), max(all_vals)
        pad = 0.15 * (y_max - y_min + 1e-12)
        apply_layout(
            fig,
            title=f"Cat 2 — {label}: {unit} vs curriculum_scale",
            subtitle=(f"n_envs={n_envs}, draws/env={n_draws}"),
            xaxis_title="curriculum_scale",
            yaxis_title=f"{label} [{unit}]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[y_min - pad, y_max + pad])
        return fig

    log_and_plot(
        name=f"cat2_{slug}_curriculum",
        category=CAT,
        metric=f"{label} distribution",
        unit=unit,
        baseline_description=f"curriculum_scale=0 collapses to nominal ({unit})",
        config={
            "n_envs": n_envs,
            "draws_per_env": n_draws,
            "curriculum_scales": list(CURRICULUM_SCALES),
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["curriculum_scale", "value"],
        stats_by_series=stats_by_series,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/curriculum] logged {len(rows)} samples")


# ===================================================================
# 2.1–2.4: Scalar / vector curriculum violins
# ===================================================================


def plot_thrust_coeff_kf(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "thrust_coeff_kf",
        make_fn=lambda n, s: ThrustCoefficientKF(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="N/RPM²",
        label="KF",
    )


def plot_torque_coeff_km(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "torque_coeff_km",
        make_fn=lambda n, s: TorqueCoefficientKM(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="N·m/RPM²",
        label="KM",
    )


def plot_propeller_thrust_asymmetry(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "propeller_thrust_asymmetry",
        make_fn=lambda n, s: PropellerThrustAsymmetry(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="ratio",
        label="thrust ratio (motor 0)",
    )


def plot_motor_partial_failure(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "motor_partial_failure",
        make_fn=lambda n, s: MotorPartialFailure(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="ratio",
        label="efficiency (motor 0)",
    )


# ===================================================================
# 2.5 MotorKill — kill count bar + heatmap
# ===================================================================


def plot_motor_kill(hw: HardwareMeta) -> None:
    n_envs = 256
    n_trials = 20

    rows_bar: list[dict] = []
    avg_killed_by_scale: dict[str, float] = {}

    for scale in CURRICULUM_SCALES:
        total = 0.0
        for trial in range(n_trials):
            torch.manual_seed(trial)
            mk = MotorKill(
                n_envs=n_envs,
                dt=DT,
                min_killed=0,
                max_killed=2,
                curriculum_scale=scale,
            )
            mk.sample()
            total += mk._current_value.sum().item()
        avg = total / (n_trials * n_envs)
        avg_killed_by_scale[f"{scale:.2f}"] = avg
        rows_bar.append({"curriculum_scale": float(scale), "avg_motors_killed": avg})

    stats_bar = {k: stats_summary([v]) for k, v in avg_killed_by_scale.items()}

    def fig_fn_bar(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Bar(
                x=[f"{s:.2f}" for s in CURRICULUM_SCALES],
                y=[avg_killed_by_scale[f"{s:.2f}"] for s in CURRICULUM_SCALES],
                marker_color=COLOR,
                name="avg killed",
            )
        )
        apply_layout(
            fig,
            title="Cat 2 — motor_kill: avg motors killed vs curriculum_scale",
            subtitle=f"n_envs={n_envs}, n_trials={n_trials}, max_killed=2",
            xaxis_title="curriculum_scale",
            yaxis_title="avg motors killed per env",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat2_motor_kill_curriculum",
        category=CAT,
        metric="avg motors killed",
        unit="count",
        baseline_description="curriculum_scale=0 → 0 motors killed",
        config={"n_envs": n_envs, "n_trials": n_trials, "max_killed": 2},
        hardware=hw,
        csv_rows=rows_bar,
        csv_columns=["curriculum_scale", "avg_motors_killed"],
        stats_by_series=stats_bar,
        figure_fn=fig_fn_bar,
    )
    print("  [motor_kill/curriculum] done")

    # Heatmap: kill mask at scale=1.0
    n_envs_hm = 16
    torch.manual_seed(42)
    mk = MotorKill(
        n_envs=n_envs_hm,
        dt=DT,
        min_killed=0,
        max_killed=2,
        curriculum_scale=1.0,
    )
    mk.sample()
    mask = mk._current_value.numpy()

    rows_hm: list[dict] = []
    for e in range(n_envs_hm):
        for m in range(4):
            rows_hm.append(
                {
                    "env_id": e,
                    "motor": m,
                    "killed": int(mask[e, m]),
                }
            )

    def fig_fn_hm(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Heatmap(
                z=mask,
                x=["motor 0", "motor 1", "motor 2", "motor 3"],
                y=[f"env {e}" for e in range(n_envs_hm)],
                colorscale=[[0, "#d4edda"], [1, "#f8d7da"]],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=("env=%{y}<br>%{x}<br>killed=%{z:.0f}<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title="Cat 2 — motor_kill: kill mask heatmap",
            subtitle=(
                f"green=alive, red=killed; n_envs={n_envs_hm}, curriculum_scale=1.0, max_killed=2"
            ),
            xaxis_title="motor",
            yaxis_title="environment",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name="cat2_motor_kill_heatmap",
        category=CAT,
        metric="kill mask",
        unit="boolean",
        baseline_description="1=killed, 0=alive",
        config={"n_envs": n_envs_hm, "curriculum_scale": 1.0, "max_killed": 2},
        hardware=hw,
        csv_rows=rows_hm,
        csv_columns=["env_id", "motor", "killed"],
        stats_by_series={
            "kill_rate": stats_summary(mask.mean(axis=1).tolist()),
        },
        figure_fn=fig_fn_hm,
    )
    print("  [motor_kill/heatmap] done")


# ===================================================================
# 2.6–2.8, 2.11, 2.13: Scalar curriculum violins
# ===================================================================


def plot_motor_lag(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "motor_lag",
        make_fn=lambda n, s: MotorLag(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="s",
        label="tau (time constant)",
    )


def plot_motor_rpm_noise(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "motor_rpm_noise",
        make_fn=lambda n, s: MotorRPMNoise(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="fraction",
        label="noise scale (motor 0)",
    )


def plot_motor_saturation(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "motor_saturation",
        make_fn=lambda n, s: MotorSaturation(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="RPM",
        label="rpm_max",
    )


def plot_motor_back_emf(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "motor_back_emf",
        make_fn=lambda n, s: MotorBackEMF(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="V·s/rad",
        label="Ke (back-EMF)",
    )


def plot_gyroscopic_effect(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "gyroscopic_effect",
        make_fn=lambda n, s: GyroscopicEffect(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="kg·m²",
        label="I_rotor",
    )


# ===================================================================
# 2.9 MotorWear — curriculum violin (after N steps) + decay trace
# ===================================================================


def plot_motor_wear(hw: HardwareMeta) -> None:
    n_envs = 256
    n_steps = 100

    rows_cur: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(2900 + i)
        w = MotorWear(n_envs=n_envs, dt=DT, curriculum_scale=scale)
        w.reset(torch.arange(n_envs))
        for _ in range(n_steps):
            w.step()
        eff = w._efficiency[:, 0].tolist()
        key = f"{scale:.2f}"
        samples_by_scale[key] = eff
        for v in eff:
            rows_cur.append({"curriculum_scale": float(scale), "efficiency": v})

    stats_cur = {s: stats_summary(v) for s, v in samples_by_scale.items()}

    def fig_fn_cur(hw_: HardwareMeta) -> go.Figure:
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
        apply_layout(
            fig,
            title=f"Cat 2 — motor_wear: efficiency after {n_steps} steps (motor 0)",
            subtitle=f"n_envs={n_envs}, floor=0.8",
            xaxis_title="curriculum_scale",
            yaxis_title="efficiency [ratio]",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_yaxes(range=[0.75, 1.05])
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="floor (0.8)",
        )
        return fig

    log_and_plot(
        name="cat2_motor_wear_curriculum",
        category=CAT,
        metric=f"efficiency after {n_steps} steps",
        unit="ratio",
        baseline_description="monotonic decay, floored at 0.8",
        config={"n_envs": n_envs, "n_steps": n_steps},
        hardware=hw,
        csv_rows=rows_cur,
        csv_columns=["curriculum_scale", "efficiency"],
        stats_by_series=stats_cur,
        figure_fn=fig_fn_cur,
    )
    print("  [motor_wear/curriculum] done")

    # Decay trace
    n_envs_trace = 8
    torch.manual_seed(2910)
    w = MotorWear(n_envs=n_envs_trace, dt=DT, curriculum_scale=1.0)
    w.reset(torch.arange(n_envs_trace))
    history = torch.zeros(n_steps, n_envs_trace)
    for step in range(n_steps):
        w.step()
        history[step] = w._efficiency[:, 0].clone()

    rows_trace: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs_trace):
            rows_trace.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "efficiency": float(history[step, e]),
                }
            )

    stats_trace = {f"env_{e}": stats_summary(history[:, e].tolist()) for e in range(n_envs_trace)}

    def fig_fn_trace(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs_trace):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=history[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.5),
                )
            )
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="floor (0.8)",
        )
        apply_layout(
            fig,
            title="Cat 2 — motor_wear: efficiency decay trace (motor 0)",
            subtitle=(f"n_envs={n_envs_trace}, n_steps={n_steps}, curriculum_scale=1.0"),
            xaxis_title="time [s]",
            yaxis_title="efficiency [ratio]",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_yaxes(range=[0.75, 1.05])
        return fig

    log_and_plot(
        name="cat2_motor_wear_trace",
        category=CAT,
        metric="efficiency decay",
        unit="ratio",
        baseline_description="linear decay from 1.0, floor at 0.8",
        config={
            "n_envs": n_envs_trace,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
        },
        hardware=hw,
        csv_rows=rows_trace,
        csv_columns=["step", "env_id", "time_s", "efficiency"],
        stats_by_series=stats_trace,
        figure_fn=fig_fn_trace,
    )
    print("  [motor_wear/trace] done")


# ===================================================================
# 2.10 RotorImbalance — curriculum violin + RPM modulation trace
# ===================================================================


def plot_rotor_imbalance(hw: HardwareMeta) -> None:
    n_envs = 256
    rows_cur: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(2100 + i)
        ri = RotorImbalance(n_envs=n_envs, dt=DT, curriculum_scale=scale)
        ri.reset(torch.arange(n_envs))
        mag = ri._magnitude[:, 0].tolist()
        key = f"{scale:.2f}"
        samples_by_scale[key] = mag
        for v in mag:
            rows_cur.append({"curriculum_scale": float(scale), "magnitude": v})

    stats_cur = {s: stats_summary(v) for s, v in samples_by_scale.items()}

    def fig_fn_cur(hw_: HardwareMeta) -> go.Figure:
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
        apply_layout(
            fig,
            title="Cat 2 — rotor_imbalance: magnitude vs curriculum_scale (motor 0)",
            subtitle=f"n_envs={n_envs}",
            xaxis_title="curriculum_scale",
            yaxis_title="imbalance magnitude [ratio]",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat2_rotor_imbalance_curriculum",
        category=CAT,
        metric="imbalance magnitude",
        unit="ratio",
        baseline_description="curriculum_scale=0 → 0 imbalance",
        config={"n_envs": n_envs},
        hardware=hw,
        csv_rows=rows_cur,
        csv_columns=["curriculum_scale", "magnitude"],
        stats_by_series=stats_cur,
        figure_fn=fig_fn_cur,
    )
    print("  [rotor_imbalance/curriculum] done")

    # RPM modulation trace
    n_envs_trace = 4
    n_steps = 200
    torch.manual_seed(2110)
    ri = RotorImbalance(n_envs=n_envs_trace, dt=DT, curriculum_scale=1.0)
    ri.reset(torch.arange(n_envs_trace))
    rpm_cmd = torch.ones(n_envs_trace, 4) * 15000.0

    rpm_history = torch.zeros(n_steps, n_envs_trace)
    for step in range(n_steps):
        rpm_out = ri.apply(rpm_cmd)
        rpm_history[step] = rpm_out[:, 0].clone()
        ri.step()

    rows_trace: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs_trace):
            rows_trace.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "rpm_motor0": float(rpm_history[step, e]),
                }
            )

    stats_trace = {
        f"env_{e}": stats_summary(rpm_history[:, e].tolist()) for e in range(n_envs_trace)
    }

    def fig_fn_trace(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs_trace):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=rpm_history[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.2),
                )
            )
        fig.add_hline(
            y=15000.0,
            line_dash="dash",
            line_color="grey",
            annotation_text="commanded (15000 RPM)",
        )
        apply_layout(
            fig,
            title="Cat 2 — rotor_imbalance: RPM modulation trace (motor 0)",
            subtitle=(f"commanded=15000 RPM, n_envs={n_envs_trace}, n_steps={n_steps}"),
            xaxis_title="time [s]",
            yaxis_title="RPM (motor 0)",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat2_rotor_imbalance_trace",
        category=CAT,
        metric="RPM modulation",
        unit="RPM",
        baseline_description="rpm × (1 + mag × sin(phase)), phase advancing at rotor ω",
        config={
            "n_envs": n_envs_trace,
            "n_steps": n_steps,
            "rpm_commanded": 15000.0,
        },
        hardware=hw,
        csv_rows=rows_trace,
        csv_columns=["step", "env_id", "time_s", "rpm_motor0"],
        stats_by_series=stats_trace,
        figure_fn=fig_fn_trace,
    )
    print("  [rotor_imbalance/trace] done")


# ===================================================================
# 2.12 MotorColdStart — curriculum violin + warmup decay trace
# ===================================================================


def plot_motor_cold_start(hw: HardwareMeta) -> None:
    n_envs = 256
    rows_cur: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(2120 + i)
        cs = MotorColdStart(n_envs=n_envs, dt=DT, curriculum_scale=scale)
        cs.reset(torch.arange(n_envs))
        overhead = cs._initial_overhead[:, 0].tolist()
        key = f"{scale:.2f}"
        samples_by_scale[key] = overhead
        for v in overhead:
            rows_cur.append({"curriculum_scale": float(scale), "overhead": v})

    stats_cur = {s: stats_summary(v) for s, v in samples_by_scale.items()}

    def fig_fn_cur(hw_: HardwareMeta) -> go.Figure:
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
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="grey",
            annotation_text="nominal (warm)",
        )
        apply_layout(
            fig,
            title="Cat 2 — motor_cold_start: initial overhead vs curriculum_scale",
            subtitle=f"n_envs={n_envs}, motor 0",
            xaxis_title="curriculum_scale",
            yaxis_title="warmup factor [ratio]",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat2_motor_cold_start_curriculum",
        category=CAT,
        metric="initial warmup overhead",
        unit="ratio",
        baseline_description="curriculum_scale=0 → overhead=1.0 (no cold start)",
        config={"n_envs": n_envs},
        hardware=hw,
        csv_rows=rows_cur,
        csv_columns=["curriculum_scale", "overhead"],
        stats_by_series=stats_cur,
        figure_fn=fig_fn_cur,
    )
    print("  [motor_cold_start/curriculum] done")

    # Warmup decay trace
    n_envs_trace = 8
    n_steps = 100
    torch.manual_seed(2130)
    cs = MotorColdStart(
        n_envs=n_envs_trace,
        dt=DT,
        warmup_tau=0.5,
        curriculum_scale=1.0,
    )
    cs.reset(torch.arange(n_envs_trace))
    history = torch.zeros(n_steps, n_envs_trace)
    for step in range(n_steps):
        history[step] = cs._warmup_factor[:, 0].clone()
        cs.step()

    rows_trace: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs_trace):
            rows_trace.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "warmup_factor": float(history[step, e]),
                }
            )

    stats_trace = {f"env_{e}": stats_summary(history[:, e].tolist()) for e in range(n_envs_trace)}

    def fig_fn_trace(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs_trace):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=history[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.5),
                )
            )
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="grey",
            annotation_text="nominal (warm)",
        )
        apply_layout(
            fig,
            title="Cat 2 — motor_cold_start: warmup decay trace (motor 0)",
            subtitle=(f"τ=0.5 s, exp decay → 1.0; n_envs={n_envs_trace}, n_steps={n_steps}"),
            xaxis_title="time [s]",
            yaxis_title="warmup factor [ratio]",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat2_motor_cold_start_trace",
        category=CAT,
        metric="warmup factor decay",
        unit="ratio",
        baseline_description="exp decay: factor = 1 + (overhead − 1) × exp(−t/τ)",
        config={
            "n_envs": n_envs_trace,
            "n_steps": n_steps,
            "warmup_tau": 0.5,
        },
        hardware=hw,
        csv_rows=rows_trace,
        csv_columns=["step", "env_id", "time_s", "warmup_factor"],
        stats_by_series=stats_trace,
        figure_fn=fig_fn_trace,
    )
    print("  [motor_cold_start/trace] done")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    specs = [
        ("2.1 thrust_coeff_kf", plot_thrust_coeff_kf),
        ("2.2 torque_coeff_km", plot_torque_coeff_km),
        ("2.3 propeller_thrust_asymmetry", plot_propeller_thrust_asymmetry),
        ("2.4 motor_partial_failure", plot_motor_partial_failure),
        ("2.5 motor_kill", plot_motor_kill),
        ("2.6 motor_lag", plot_motor_lag),
        ("2.7 motor_rpm_noise", plot_motor_rpm_noise),
        ("2.8 motor_saturation", plot_motor_saturation),
        ("2.9 motor_wear", plot_motor_wear),
        ("2.10 rotor_imbalance", plot_rotor_imbalance),
        ("2.11 motor_back_emf", plot_motor_back_emf),
        ("2.12 motor_cold_start", plot_motor_cold_start),
        ("2.13 gyroscopic_effect", plot_gyroscopic_effect),
    ]

    for label, fn in specs:
        print(f"\n--- {label} ---")
        fn(hardware)

    print("\nDone. Inspect docs/impl/{data,assets}/cat2_*.{csv,meta.json,png}")


if __name__ == "__main__":
    main()
