"""Generate technical plots for category 2 (motor perturbations) documentation.

Outputs PNG files to docs/impl/assets/ via Plotly + kaleido.
Run: uv run python docs/impl/plot_category_2.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).parents[2]
ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SAMPLES = 5000
N_ENVS = 16
N_EPISODES = 30
N_STEPS = 100


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _curriculum_histogram(
    name: str,
    title: str,
    xaxis: str,
    nominal: float,
    low: float,
    high: float,
    bounds: tuple[float, float],
) -> None:
    """Plot curriculum_scale effect as overlaid histograms."""
    torch.manual_seed(0)
    raw = torch.empty(N_SAMPLES).uniform_(low, high).numpy()

    fig = go.Figure()
    for scale, color in zip(SCALES, COLORS):
        values = nominal + (raw - nominal) * scale
        values = np.clip(values, bounds[0], bounds[1])
        fig.add_trace(go.Histogram(
            x=values, nbinsx=60, name=f"scale={scale}",
            opacity=0.65, marker_color=color,
        ))

    fig.update_layout(
        title=f"{title} -- curriculum_scale effect",
        xaxis_title=xaxis, yaxis_title="count",
        barmode="overlay", legend_title="curriculum_scale",
        template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=nominal, line_dash="dash", line_color="grey",
                  annotation_text="nominal")
    out = ASSETS / f"cat2_{name}_curriculum.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")


def _per_env_scatter(
    name: str,
    title: str,
    yaxis: str,
    nominal: float | list[float],
    low: float,
    high: float,
    bounds: tuple[float, float],
    dim: int = 1,
) -> None:
    """Plot per-env sampled values across episodes."""
    torch.manual_seed(42)
    if dim == 1:
        samples = torch.empty(N_ENVS, N_EPISODES).uniform_(low, high).numpy()
        nom = nominal if isinstance(nominal, float) else nominal[0]
    else:
        samples = torch.empty(N_ENVS, N_EPISODES, dim).uniform_(low, high).numpy()
        nom = nominal if isinstance(nominal, float) else nominal[0]

    fig = go.Figure()
    if dim == 1:
        for env_id in range(N_ENVS):
            fig.add_trace(go.Scatter(
                x=list(range(N_EPISODES)), y=samples[env_id],
                mode="lines+markers", name=f"env {env_id}",
                marker=dict(size=3), line=dict(width=1.2),
            ))
        fig.add_hline(y=nom, line_dash="dash", line_color="grey",
                      annotation_text="nominal")
    else:
        # Show motor 0 across envs for clarity
        for env_id in range(min(N_ENVS, 8)):
            fig.add_trace(go.Scatter(
                x=list(range(N_EPISODES)), y=samples[env_id, :, 0],
                mode="lines+markers", name=f"env {env_id} (motor 0)",
                marker=dict(size=3), line=dict(width=1.2),
            ))
        fig.add_hline(y=nom, line_dash="dash", line_color="grey",
                      annotation_text="nominal")

    fig.update_layout(
        title=f"{title} -- per-env diversity (scale=1.0)",
        xaxis_title="episode", yaxis_title=yaxis,
        legend_title="env", template="plotly_white",
        width=700, height=420,
    )
    out = ASSETS / f"cat2_{name}_per_env.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")


def _curriculum_histogram_vec4(
    name: str,
    title: str,
    xaxis: str,
    nominal: float,
    low: float,
    high: float,
    bounds: tuple[float, float],
) -> None:
    """Curriculum histogram for vector(4) perturbations -- shows motor 0."""
    _curriculum_histogram(name, title + " (motor 0)", xaxis, nominal, low, high, bounds)


# -----------------------------------------------------------------------
# 2.1 ThrustCoefficientKF
# -----------------------------------------------------------------------


def plot_thrust_coeff_kf() -> None:
    _curriculum_histogram(
        "thrust_coeff_kf", "ThrustCoefficientKF", "KF [N/RPM^2]",
        nominal=3.16e-10, low=2.53e-10, high=3.79e-10,
        bounds=(1.58e-10, 4.74e-10),
    )
    _per_env_scatter(
        "thrust_coeff_kf", "ThrustCoefficientKF", "KF [N/RPM^2]",
        nominal=3.16e-10, low=2.53e-10, high=3.79e-10,
        bounds=(1.58e-10, 4.74e-10),
    )


# -----------------------------------------------------------------------
# 2.2 TorqueCoefficientKM
# -----------------------------------------------------------------------


def plot_torque_coeff_km() -> None:
    _curriculum_histogram(
        "torque_coeff_km", "TorqueCoefficientKM", "KM [N.m/RPM^2]",
        nominal=3.16e-12, low=2.53e-12, high=3.79e-12,
        bounds=(1.58e-12, 4.74e-12),
    )
    _per_env_scatter(
        "torque_coeff_km", "TorqueCoefficientKM", "KM [N.m/RPM^2]",
        nominal=3.16e-12, low=2.53e-12, high=3.79e-12,
        bounds=(1.58e-12, 4.74e-12),
    )


# -----------------------------------------------------------------------
# 2.3 PropellerThrustAsymmetry
# -----------------------------------------------------------------------


def plot_propeller_thrust_asymmetry() -> None:
    _curriculum_histogram_vec4(
        "propeller_thrust_asymmetry", "PropellerThrustAsymmetry", "ratio",
        nominal=1.0, low=0.85, high=1.15, bounds=(0.7, 1.3),
    )
    _per_env_scatter(
        "propeller_thrust_asymmetry", "PropellerThrustAsymmetry", "ratio",
        nominal=[1.0, 1.0, 1.0, 1.0], low=0.85, high=1.15,
        bounds=(0.7, 1.3), dim=4,
    )


# -----------------------------------------------------------------------
# 2.4 MotorPartialFailure
# -----------------------------------------------------------------------


def plot_motor_partial_failure() -> None:
    _curriculum_histogram_vec4(
        "motor_partial_failure", "MotorPartialFailure", "efficiency",
        nominal=1.0, low=0.3, high=1.0, bounds=(0.0, 1.0),
    )
    _per_env_scatter(
        "motor_partial_failure", "MotorPartialFailure", "efficiency",
        nominal=[1.0, 1.0, 1.0, 1.0], low=0.3, high=1.0,
        bounds=(0.0, 1.0), dim=4,
    )


# -----------------------------------------------------------------------
# 2.5 MotorKill
# -----------------------------------------------------------------------


def plot_motor_kill() -> None:
    """Curriculum: show average number of killed motors vs scale."""
    from genesis_robust_rl.perturbations.category_2_motor import MotorKill

    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_envs = 256
    n_trials = 20
    avg_killed = []

    for s in scales:
        total = 0.0
        for trial in range(n_trials):
            torch.manual_seed(trial)
            mk = MotorKill(n_envs=n_envs, dt=0.01, min_killed=0, max_killed=2)
            mk.curriculum_scale = s
            mk.sample()
            total += mk._current_value.sum().item()
        avg_killed.append(total / (n_trials * n_envs))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(s) for s in scales], y=avg_killed,
                         marker_color=COLORS[:len(scales)]))
    fig.update_layout(
        title="MotorKill -- avg killed motors vs curriculum_scale",
        xaxis_title="curriculum_scale", yaxis_title="avg motors killed per env",
        template="plotly_white", width=700, height=420,
    )
    out = ASSETS / "cat2_motor_kill_curriculum.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")

    # Per-env: show kill mask heatmap
    torch.manual_seed(42)
    mk = MotorKill(n_envs=N_ENVS, dt=0.01, min_killed=0, max_killed=2)
    mk.curriculum_scale = 1.0
    mk.sample()
    mask = mk._current_value.numpy()

    fig = go.Figure(data=go.Heatmap(
        z=mask, x=["motor 0", "motor 1", "motor 2", "motor 3"],
        y=[f"env {i}" for i in range(N_ENVS)],
        colorscale=[[0, "#d4edda"], [1, "#f8d7da"]],
        zmin=0, zmax=1,
        text=[["killed" if v == 1 else "alive" for v in row] for row in mask],
        texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        title="MotorKill -- per-env kill mask (scale=1.0)",
        template="plotly_white", width=500, height=500,
    )
    out = ASSETS / "cat2_motor_kill_per_env.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")


# -----------------------------------------------------------------------
# 2.6 MotorLag
# -----------------------------------------------------------------------


def plot_motor_lag() -> None:
    _curriculum_histogram(
        "motor_lag", "MotorLag", "tau [s]",
        nominal=0.033, low=0.02, high=0.05, bounds=(0.01, 0.1),
    )
    _per_env_scatter(
        "motor_lag", "MotorLag", "tau [s]",
        nominal=0.033, low=0.02, high=0.05, bounds=(0.01, 0.1),
    )


# -----------------------------------------------------------------------
# 2.7 MotorRPMNoise
# -----------------------------------------------------------------------


def plot_motor_rpm_noise() -> None:
    _curriculum_histogram(
        "motor_rpm_noise", "MotorRPMNoise", "noise scale (fraction)",
        nominal=0.0, low=0.0, high=0.03, bounds=(0.0, 0.05),
    )
    _per_env_scatter(
        "motor_rpm_noise", "MotorRPMNoise", "noise scale (fraction)",
        nominal=0.0, low=0.0, high=0.03, bounds=(0.0, 0.05),
    )


# -----------------------------------------------------------------------
# 2.8 MotorSaturation
# -----------------------------------------------------------------------


def plot_motor_saturation() -> None:
    rpm_max_nom = 21666.0
    _curriculum_histogram(
        "motor_saturation", "MotorSaturation", "rpm_max [RPM]",
        nominal=rpm_max_nom, low=rpm_max_nom * 0.8, high=rpm_max_nom,
        bounds=(rpm_max_nom * 0.7, rpm_max_nom),
    )
    _per_env_scatter(
        "motor_saturation", "MotorSaturation", "rpm_max [RPM]",
        nominal=rpm_max_nom, low=rpm_max_nom * 0.8, high=rpm_max_nom,
        bounds=(rpm_max_nom * 0.7, rpm_max_nom),
    )


# -----------------------------------------------------------------------
# 2.9 MotorWear (stateful -- state evolution)
# -----------------------------------------------------------------------


def plot_motor_wear() -> None:
    """Curriculum: show efficiency after 100 steps at various scales."""
    from genesis_robust_rl.perturbations.category_2_motor import MotorWear

    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_envs = 256

    fig = go.Figure()
    for s, color in zip(scales, COLORS):
        torch.manual_seed(0)
        w = MotorWear(n_envs=n_envs, dt=0.01)
        w.curriculum_scale = s
        w.reset(torch.arange(n_envs))
        for _ in range(N_STEPS):
            w.step()
        eff = w._efficiency[:, 0].numpy()
        fig.add_trace(go.Histogram(
            x=eff, nbinsx=50, name=f"scale={s}",
            opacity=0.65, marker_color=color,
        ))

    fig.update_layout(
        title=f"MotorWear -- efficiency after {N_STEPS} steps (motor 0)",
        xaxis_title="efficiency", yaxis_title="count",
        barmode="overlay", legend_title="curriculum_scale",
        template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey",
                  annotation_text="nominal")
    out = ASSETS / "cat2_motor_wear_curriculum.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")

    # Per-env: state evolution over 100 steps
    torch.manual_seed(42)
    n_envs_plot = 8
    w = MotorWear(n_envs=n_envs_plot, dt=0.01)
    w.curriculum_scale = 1.0
    w.reset(torch.arange(n_envs_plot))
    history = []
    for _ in range(N_STEPS):
        w.step()
        history.append(w._efficiency[:, 0].clone().numpy())

    history = np.array(history)  # [steps, n_envs]
    fig = go.Figure()
    for env_id in range(n_envs_plot):
        fig.add_trace(go.Scatter(
            x=list(range(N_STEPS)), y=history[:, env_id],
            mode="lines", name=f"env {env_id}", line=dict(width=1.5),
        ))
    fig.update_layout(
        title="MotorWear -- efficiency decay over episode (motor 0, scale=1.0)",
        xaxis_title="step", yaxis_title="efficiency",
        legend_title="env", template="plotly_white", width=700, height=420,
    )
    fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                  annotation_text="floor (0.8)")
    out = ASSETS / "cat2_motor_wear_per_env.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")


# -----------------------------------------------------------------------
# 2.10 RotorImbalance (stateful -- magnitude + phase)
# -----------------------------------------------------------------------


def plot_rotor_imbalance() -> None:
    """Curriculum: magnitude histogram at various scales."""
    from genesis_robust_rl.perturbations.category_2_motor import RotorImbalance

    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_envs = 256

    fig = go.Figure()
    for s, color in zip(scales, COLORS):
        torch.manual_seed(0)
        ri = RotorImbalance(n_envs=n_envs, dt=0.01)
        ri.curriculum_scale = s
        ri.reset(torch.arange(n_envs))
        mag = ri._magnitude[:, 0].numpy()
        fig.add_trace(go.Histogram(
            x=mag, nbinsx=50, name=f"scale={s}",
            opacity=0.65, marker_color=color,
        ))

    fig.update_layout(
        title="RotorImbalance -- magnitude after reset (motor 0)",
        xaxis_title="imbalance magnitude", yaxis_title="count",
        barmode="overlay", legend_title="curriculum_scale",
        template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="grey",
                  annotation_text="nominal")
    out = ASSETS / "cat2_rotor_imbalance_curriculum.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")

    # Per-env: RPM modulation over steps
    torch.manual_seed(42)
    n_envs_plot = 4
    ri = RotorImbalance(n_envs=n_envs_plot, dt=0.01)
    ri.curriculum_scale = 1.0
    ri.reset(torch.arange(n_envs_plot))
    rpm_cmd = torch.ones(n_envs_plot, 4) * 15000.0

    history = []
    for _ in range(N_STEPS):
        rpm_out = ri.apply(rpm_cmd)
        history.append(rpm_out[:, 0].clone().numpy())
        ri.step()

    history = np.array(history)
    fig = go.Figure()
    for env_id in range(n_envs_plot):
        fig.add_trace(go.Scatter(
            x=list(range(N_STEPS)), y=history[:, env_id],
            mode="lines", name=f"env {env_id}", line=dict(width=1.2),
        ))
    fig.add_hline(y=15000.0, line_dash="dash", line_color="grey",
                  annotation_text="commanded")
    fig.update_layout(
        title="RotorImbalance -- RPM modulation (motor 0, scale=1.0)",
        xaxis_title="step", yaxis_title="RPM (motor 0)",
        legend_title="env", template="plotly_white", width=700, height=420,
    )
    out = ASSETS / "cat2_rotor_imbalance_per_env.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")


# -----------------------------------------------------------------------
# 2.11 MotorBackEMF
# -----------------------------------------------------------------------


def plot_motor_back_emf() -> None:
    _curriculum_histogram(
        "motor_back_emf", "MotorBackEMF", "Ke [V.s/rad]",
        nominal=0.01, low=0.007, high=0.013, bounds=(0.005, 0.015),
    )
    _per_env_scatter(
        "motor_back_emf", "MotorBackEMF", "Ke [V.s/rad]",
        nominal=0.01, low=0.007, high=0.013, bounds=(0.005, 0.015),
    )


# -----------------------------------------------------------------------
# 2.12 MotorColdStart (stateful -- warmup decay)
# -----------------------------------------------------------------------


def plot_motor_cold_start() -> None:
    """Curriculum: initial overhead histogram at various scales."""
    from genesis_robust_rl.perturbations.category_2_motor import MotorColdStart

    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_envs = 256

    fig = go.Figure()
    for s, color in zip(scales, COLORS):
        torch.manual_seed(0)
        cs = MotorColdStart(n_envs=n_envs, dt=0.01)
        cs.curriculum_scale = s
        cs.reset(torch.arange(n_envs))
        overhead = cs._initial_overhead[:, 0].numpy()
        fig.add_trace(go.Histogram(
            x=overhead, nbinsx=50, name=f"scale={s}",
            opacity=0.65, marker_color=color,
        ))

    fig.update_layout(
        title="MotorColdStart -- initial overhead at reset (motor 0)",
        xaxis_title="warmup factor", yaxis_title="count",
        barmode="overlay", legend_title="curriculum_scale",
        template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey",
                  annotation_text="nominal")
    out = ASSETS / "cat2_motor_cold_start_curriculum.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")

    # Per-env: warmup factor decay over steps
    torch.manual_seed(42)
    n_envs_plot = 8
    cs = MotorColdStart(n_envs=n_envs_plot, dt=0.01, warmup_tau=0.5)
    cs.curriculum_scale = 1.0
    cs.reset(torch.arange(n_envs_plot))
    history = []
    for _ in range(N_STEPS):
        history.append(cs._warmup_factor[:, 0].clone().numpy())
        cs.step()

    history = np.array(history)
    fig = go.Figure()
    for env_id in range(n_envs_plot):
        fig.add_trace(go.Scatter(
            x=list(range(N_STEPS)), y=history[:, env_id],
            mode="lines", name=f"env {env_id}", line=dict(width=1.5),
        ))
    fig.update_layout(
        title="MotorColdStart -- warmup decay over episode (motor 0, scale=1.0)",
        xaxis_title="step", yaxis_title="warmup factor",
        legend_title="env", template="plotly_white", width=700, height=420,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey",
                  annotation_text="nominal (warm)")
    out = ASSETS / "cat2_motor_cold_start_per_env.png"
    pio.write_image(fig, out, scale=2)
    print(f"Saved: {out.name}")


# -----------------------------------------------------------------------
# 2.13 GyroscopicEffect
# -----------------------------------------------------------------------


def plot_gyroscopic_effect() -> None:
    _curriculum_histogram(
        "gyroscopic_effect", "GyroscopicEffect", "I_rotor [kg.m^2]",
        nominal=3.0e-5, low=2.0e-5, high=4.0e-5, bounds=(1.5e-5, 4.5e-5),
    )
    _per_env_scatter(
        "gyroscopic_effect", "GyroscopicEffect", "I_rotor [kg.m^2]",
        nominal=3.0e-5, low=2.0e-5, high=4.0e-5, bounds=(1.5e-5, 4.5e-5),
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


if __name__ == "__main__":
    print("Generating category 2 plots...")
    plot_thrust_coeff_kf()
    plot_torque_coeff_km()
    plot_propeller_thrust_asymmetry()
    plot_motor_partial_failure()
    plot_motor_kill()
    plot_motor_lag()
    plot_motor_rpm_noise()
    plot_motor_saturation()
    plot_motor_wear()
    plot_rotor_imbalance()
    plot_motor_back_emf()
    plot_motor_cold_start()
    plot_gyroscopic_effect()
    print("Done -- all category 2 plots saved to docs/impl/assets/")
