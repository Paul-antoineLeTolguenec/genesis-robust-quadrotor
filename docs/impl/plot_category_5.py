"""Category 5 — Wind perturbation plots.

Generates curriculum + specialised PNGs for:
- 5.1 constant_wind             (curriculum violin + per-env bar)
- 5.2 turbulence                (curriculum traces + spectrum)
- 5.3 wind_gust                 (gust event heatmap + duration histogram)
- 5.4 wind_shear                (curriculum violin + altitude sweep)
- 5.5 adversarial_wind          (curriculum violin + Lipschitz trace)
- 5.6 blade_vortex_interaction  (curriculum traces + spectrum)
- 5.7 ground_effect_boundary    (altitude sweep + per-env bar)
- 5.8 payload_sway              (curriculum traces + phase-space)
- 5.9 proximity_disturbance     (force vs distance + per-env bar)

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

from genesis_robust_rl.perturbations.base import EnvState  # noqa: E402
from genesis_robust_rl.perturbations.category_5_wind import (  # noqa: E402
    AdversarialWind,
    BladeVortexInteraction,
    ConstantWind,
    GroundEffectBoundary,
    PayloadSway,
    ProximityDisturbance,
    Turbulence,
    WindGust,
    WindShear,
)

CAT = 5
DT = 0.01
COLOR = CATEGORY_COLORS[CAT]
N_ENVS_CURRICULUM = 256
N_EPISODES_PER_ENV = 40
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
TRACE_STEPS = 400
TRACE_N_ENVS = 4
SPECTRUM_STEPS = 2048
PER_ENV_N = 16


def _make_env_state(n_envs: int, altitude: float = 0.5) -> EnvState:
    return EnvState(
        pos=torch.tensor([[0.0, 0.0, altitude]]).expand(n_envs, 3).clone(),
        quat=torch.tensor([[1.0, 0, 0, 0]]).expand(n_envs, 4).clone(),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 14000.0,
        dt=DT,
        step=0,
    )


# ===================================================================
# Shared: curriculum violin for scalar/vector magnitude
# ===================================================================


def _curriculum_violin_magnitude(
    hardware: HardwareMeta,
    slug: str,
    pert_cls: type,
    unit: str,
    label: str,
    vector: bool = True,
) -> None:
    rows: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(5000 + i)
        p = pert_cls(n_envs=N_ENVS_CURRICULUM, dt=DT, curriculum_scale=scale)
        values: list[float] = []
        for _ in range(N_EPISODES_PER_ENV):
            v = p.sample()
            if vector:
                mag = v.norm(dim=-1).squeeze()
            else:
                mag = v.squeeze()
            for m in mag.tolist():
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
        pad = 0.15 * (y_max - y_min + 1e-6)
        apply_layout(
            fig,
            title=f"Cat 5 — {label}: {unit} vs curriculum_scale",
            subtitle=(
                f"distribution=uniform, n_envs={N_ENVS_CURRICULUM}, draws/env={N_EPISODES_PER_ENV}"
            ),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title=f"{label} [{unit}]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[max(0, y_min - pad), y_max + pad])
        return fig

    log_and_plot(
        name=f"cat5_{slug}_curriculum",
        category=CAT,
        metric=f"{label} distribution",
        unit=unit,
        baseline_description=f"curriculum_scale=0 collapses to nominal ({unit})",
        config={
            "distribution": "uniform",
            "n_envs": N_ENVS_CURRICULUM,
            "draws_per_env": N_EPISODES_PER_ENV,
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
# Shared: OU trace plot
# ===================================================================


def _ou_trace(
    hardware: HardwareMeta,
    slug: str,
    pert_cls: type,
    unit: str,
    label: str,
    seed: int,
    axis_idx: int = 0,
    axis_name: str = "X",
) -> None:
    n_envs = TRACE_N_ENVS
    n_steps = TRACE_STEPS
    torch.manual_seed(seed)
    p = pert_cls(n_envs=n_envs, dt=DT, curriculum_scale=1.0)
    p.tick(is_reset=True)

    trace = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.tick(is_reset=False)
        trace[step] = p._current_value[:, axis_idx].clone()

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
    stats_per_env["all"] = stats_summary(trace.reshape(-1).tolist())

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=trace[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.3),
                )
            )
        apply_layout(
            fig,
            title=f"Cat 5 — {label}: OU trace ({axis_name}-axis)",
            subtitle=(f"n_envs={n_envs}, n_steps={n_steps}, dt={DT}s, curriculum_scale=1.0"),
            xaxis_title="time [s]",
            yaxis_title=f"{label} [{unit}]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"cat5_{slug}_trace",
        category=CAT,
        metric=f"{label} trace",
        unit=unit,
        baseline_description="OU trace at curriculum_scale=1.0",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "dt_s": DT,
            "axis": axis_name,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "value"],
        stats_by_series=stats_per_env,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/trace] logged {len(rows)} rows")


# ===================================================================
# Shared: OU spectrum
# ===================================================================


def _ou_spectrum(
    hardware: HardwareMeta,
    slug: str,
    pert_cls: type,
    unit: str,
    label: str,
    seed: int,
    axis_idx: int = 0,
) -> None:
    n_envs = TRACE_N_ENVS
    n_steps = SPECTRUM_STEPS
    torch.manual_seed(seed)
    p = pert_cls(n_envs=n_envs, dt=DT, curriculum_scale=1.0)
    p.tick(is_reset=True)

    signal = np.zeros((n_steps, n_envs))
    for step in range(n_steps):
        p.tick(is_reset=False)
        signal[step] = p._current_value[:, axis_idx].detach().numpy()

    fs = 1.0 / DT
    freqs = np.fft.rfftfreq(n_steps, d=DT)
    power = np.zeros_like(freqs)
    for e in range(n_envs):
        x = signal[:, e] - signal[:, e].mean()
        X = np.fft.rfft(x)
        power += (np.abs(X) ** 2) / n_steps
    power /= n_envs

    variance = signal.var()
    white_ref = np.full_like(freqs, variance / (fs / 2))

    rows: list[dict] = []
    for f, p_ou, p_w in zip(freqs.tolist(), power.tolist(), white_ref.tolist()):
        rows.append({"frequency_hz": f, "ou_power": p_ou, "white_noise_ref": p_w})

    stats_power = stats_summary(power[1:].tolist())

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        mask = freqs > 0
        fig.add_trace(
            go.Scatter(
                x=freqs[mask].tolist(),
                y=power[mask].tolist(),
                mode="lines",
                name="OU process",
                line=dict(color=COLOR, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freqs[mask].tolist(),
                y=white_ref[mask].tolist(),
                mode="lines",
                name="white-noise reference",
                line=dict(color="#d62728", dash="dash", width=1),
            )
        )
        apply_layout(
            fig,
            title=f"Cat 5 — {label}: power spectrum",
            subtitle=(
                f"1/f² roll-off expected; n_steps={n_steps}, n_envs={n_envs}, fs={fs:.0f} Hz"
            ),
            xaxis_title="frequency [Hz]",
            yaxis_title=f"power [{unit}²/Hz]",
            xaxis_type="log",
            yaxis_type="log",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"cat5_{slug}_spectrum",
        category=CAT,
        metric="power spectrum",
        unit=f"{unit}²/Hz",
        baseline_description="OU vs white-noise flat reference",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "dt_s": DT,
            "fs_hz": fs,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["frequency_hz", "ou_power", "white_noise_ref"],
        stats_by_series={"ou_power_no_dc": stats_power},
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/spectrum] logged {len(rows)} rows")


# ===================================================================
# 5.1 ConstantWind — curriculum violin + per-env bar
# ===================================================================


def plot_constant_wind(hardware: HardwareMeta) -> None:
    _curriculum_violin_magnitude(hardware, "constant_wind", ConstantWind, "m/s", "wind ‖v‖₂")

    torch.manual_seed(5100)
    p = ConstantWind(n_envs=PER_ENV_N, dt=DT)
    p.tick(is_reset=True)
    v = p._current_value

    rows: list[dict] = []
    for e in range(PER_ENV_N):
        for ax, name in enumerate(["vx", "vy", "vz"]):
            rows.append({"env_id": e, "axis": name, "velocity_ms": float(v[e, ax])})

    stats_mag = stats_summary(v.norm(dim=-1).tolist())

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for ax, name in enumerate(["X", "Y", "Z"]):
            fig.add_trace(
                go.Bar(
                    x=list(range(PER_ENV_N)),
                    y=v[:, ax].numpy().tolist(),
                    name=name,
                )
            )
        apply_layout(
            fig,
            title="Cat 5 — constant_wind: per-env wind velocity",
            subtitle=f"n_envs={PER_ENV_N}, curriculum_scale=1.0",
            xaxis_title="environment",
            yaxis_title="wind velocity [m/s]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_layout(barmode="group")
        return fig

    log_and_plot(
        name="cat5_constant_wind_per_env",
        category=CAT,
        metric="per-env wind velocity",
        unit="m/s",
        baseline_description="per-axis wind sampled once per episode",
        config={"n_envs": PER_ENV_N, "curriculum_scale": 1.0},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["env_id", "axis", "velocity_ms"],
        stats_by_series={"magnitude": stats_mag},
        figure_fn=fig_fn,
    )
    print("  [constant_wind/per_env] done")


# ===================================================================
# 5.2 Turbulence — trace + spectrum
# ===================================================================


def plot_turbulence(hardware: HardwareMeta) -> None:
    _ou_trace(
        hardware,
        "turbulence",
        Turbulence,
        "m/s",
        "turbulence wind",
        seed=5200,
        axis_idx=0,
        axis_name="X",
    )
    _ou_spectrum(
        hardware,
        "turbulence",
        Turbulence,
        "m/s",
        "turbulence wind",
        seed=5201,
        axis_idx=0,
    )


# ===================================================================
# 5.3 WindGust — gust event heatmap + duration histogram
# ===================================================================


def _simulate_gusts(n_envs: int, n_steps: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    torch.manual_seed(seed)
    p = WindGust(
        n_envs=n_envs,
        dt=DT,
        curriculum_scale=1.0,
        distribution_params={
            "prob_low": 0.03,
            "prob_high": 0.08,
            "mag_low": 1.0,
            "mag_high": 10.0,
            "duration_low": 3,
            "duration_high": 15,
        },
    )
    p.tick(is_reset=True)

    mag = torch.zeros(n_steps, n_envs)
    prev_counter = torch.zeros(n_envs, dtype=torch.long)
    durations: list[int] = []

    for step in range(n_steps):
        p.tick(is_reset=False)
        mag[step] = p._current_value.norm(dim=-1).clone()
        counter = p._gust_counter.clone()
        for e in range(n_envs):
            if prev_counter[e] == 0 and counter[e] > 0:
                durations.append(int(counter[e].item()))
        prev_counter = counter

    return mag, durations


def plot_wind_gust(hardware: HardwareMeta) -> None:
    n_envs, n_steps = 8, 400
    mag, durations = _simulate_gusts(n_envs, n_steps, seed=5300)

    active = (mag > 0.01).float()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "force_magnitude_N": float(mag[step, e]),
                    "active": int(active[step, e].item()),
                }
            )

    gust_rates = [active[:, e].mean().item() for e in range(n_envs)]
    stats_rate = stats_summary(gust_rates)

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Heatmap(
                z=active.numpy().T,
                x=[round(s * DT, 4) for s in range(n_steps)],
                y=[f"env {e}" for e in range(n_envs)],
                colorscale=[[0, "#ffffff"], [1, COLOR]],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=("time=%{x}s<br>env=%{y}<br>active=%{z:.0f}<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title="Cat 5 — wind_gust: gust event heatmap",
            subtitle=(
                f"purple=active gust, white=calm; n_envs={n_envs}, "
                f"n_steps={n_steps}, curriculum_scale=1.0"
            ),
            xaxis_title="time [s]",
            yaxis_title="environment",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name="cat5_wind_gust_heatmap",
        category=CAT,
        metric="gust events (binary)",
        unit="boolean",
        baseline_description="1=gust active, 0=calm",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "force_magnitude_N", "active"],
        stats_by_series={"gust_active_rate": stats_rate},
        figure_fn=fig_fn,
    )
    print(f"  [wind_gust/heatmap] logged {len(rows)} rows")

    if not durations:
        print("  [wind_gust/duration_hist] no gust events — skipping")
        return

    dur_rows: list[dict] = [{"duration_steps": d} for d in durations]
    stats_dur = stats_summary([float(d) for d in durations])

    def fig_fn_dur(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        dur_arr = np.array(durations)
        bins = np.arange(0.5, dur_arr.max() + 1.5, 1.0)
        counts, edges = np.histogram(dur_arr, bins=bins)
        centers = ((edges[:-1] + edges[1:]) / 2).astype(int)
        fig.add_trace(
            go.Bar(
                x=centers.tolist(),
                y=counts.tolist(),
                marker_color=COLOR,
                name="observed",
                width=0.7,
            )
        )
        expected = len(durations) / max(int(edges[-1] - edges[0]), 1)
        fig.add_hline(
            y=expected,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"expected (uniform) ≈ {expected:.1f}",
        )
        apply_layout(
            fig,
            title="Cat 5 — wind_gust: gust duration histogram",
            subtitle=(f"n_events={len(durations)}, duration ∈ [3, 15] steps"),
            xaxis_title="gust duration [steps]",
            yaxis_title="count",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_wind_gust_duration_hist",
        category=CAT,
        metric="gust duration",
        unit="steps",
        baseline_description="duration sampled uniform ∈ [3, 15]",
        config={"n_events": len(durations)},
        hardware=hardware,
        csv_rows=dur_rows,
        csv_columns=["duration_steps"],
        stats_by_series={"duration": stats_dur},
        figure_fn=fig_fn_dur,
    )
    print(f"  [wind_gust/duration_hist] logged {len(durations)} events")


# ===================================================================
# 5.4 WindShear — curriculum violin + altitude sweep
# ===================================================================


def plot_wind_shear(hardware: HardwareMeta) -> None:
    _curriculum_violin_magnitude(
        hardware,
        "wind_shear",
        WindShear,
        "(m/s)/m",
        "shear gradient",
        vector=False,
    )

    torch.manual_seed(5400)
    n_pts = 100
    altitudes = torch.linspace(0.01, 3.0, n_pts)
    gradients = [0.3, 0.75, 1.5]
    rows: list[dict] = []
    traces_data: dict[str, list[float]] = {}

    for grad in gradients:
        p = WindShear(n_envs=n_pts, dt=DT)
        p._current_value = torch.full((n_pts,), grad)
        p._wind_direction = torch.tensor([[1.0, 0.0, 0.0]]).expand(n_pts, 3).clone()
        env_state = _make_env_state(n_pts)
        env_state.pos[:, 2] = altitudes
        wrench = p._compute_wrench(env_state)
        forces = wrench[:, 0].tolist()
        key = f"grad={grad}"
        traces_data[key] = forces
        for alt, f in zip(altitudes.tolist(), forces):
            rows.append(
                {
                    "altitude_m": round(alt, 3),
                    "gradient": grad,
                    "force_x_N": round(f, 6),
                }
            )

    stats_by = {k: stats_summary(v) for k, v in traces_data.items()}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for grad in gradients:
            key = f"grad={grad}"
            fig.add_trace(
                go.Scatter(
                    x=altitudes.numpy().tolist(),
                    y=traces_data[key],
                    mode="lines",
                    name=f"gradient={grad} (m/s)/m",
                    line=dict(width=2),
                )
            )
        apply_layout(
            fig,
            title="Cat 5 — wind_shear: force vs altitude",
            subtitle=("F = drag × (grad × z)²; drag_coeff=0.1, wind direction=[1,0,0]"),
            xaxis_title="altitude [m]",
            yaxis_title="F_x [N]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_wind_shear_altitude_sweep",
        category=CAT,
        metric="wind shear force vs altitude",
        unit="N",
        baseline_description="quadratic force = drag_coeff × (gradient × altitude)²",
        config={"gradients": gradients, "n_points": n_pts, "drag_coeff": 0.1},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["altitude_m", "gradient", "force_x_N"],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print("  [wind_shear/altitude_sweep] done")


# ===================================================================
# 5.5 AdversarialWind — curriculum violin + Lipschitz trace
# ===================================================================


def plot_adversarial_wind(hardware: HardwareMeta) -> None:
    _curriculum_violin_magnitude(hardware, "adversarial_wind", AdversarialWind, "m/s", "wind ‖v‖₂")

    torch.manual_seed(5500)
    n_steps = 300
    p = AdversarialWind(n_envs=1, dt=DT, curriculum_scale=1.0)
    p.tick(is_reset=True)

    targets: list[float] = []
    actual: list[float] = []
    for step in range(n_steps):
        t = torch.tensor([[5.0 * np.sin(step * 0.05), 0.0, 0.0]])
        p.set_value(t)
        targets.append(float(t[0, 0]))
        actual.append(float(p._current_value[0, 0]))

    rows: list[dict] = []
    for step in range(n_steps):
        rows.append(
            {
                "step": step,
                "time_s": round(step * DT, 4),
                "target_vx": round(targets[step], 4),
                "actual_vx": round(actual[step], 4),
            }
        )

    stats_target = stats_summary(targets)
    stats_actual = stats_summary(actual)

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=targets,
                mode="lines",
                name="target",
                line=dict(color="#d62728", dash="dash", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=actual,
                mode="lines",
                name="Lipschitz-constrained",
                line=dict(color=COLOR, width=2),
            )
        )
        apply_layout(
            fig,
            title="Cat 5 — adversarial_wind: Lipschitz enforcement",
            subtitle=(f"target=5·sin(0.05·t), lipschitz_k=5.0 m/s², n_steps={n_steps}"),
            xaxis_title="time [s]",
            yaxis_title="wind velocity X [m/s]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_adversarial_wind_lipschitz",
        category=CAT,
        metric="Lipschitz-constrained tracking",
        unit="m/s",
        baseline_description="smooth ramp to target, bounded by lipschitz_k×dt per step",
        config={"n_steps": n_steps, "lipschitz_k": 5.0, "dt_s": DT},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "time_s", "target_vx", "actual_vx"],
        stats_by_series={"target": stats_target, "actual": stats_actual},
        figure_fn=fig_fn,
    )
    print("  [adversarial_wind/lipschitz] done")


# ===================================================================
# 5.6 BladeVortexInteraction — trace + spectrum
# ===================================================================


def plot_bvi(hardware: HardwareMeta) -> None:
    _ou_trace(
        hardware,
        "blade_vortex",
        BladeVortexInteraction,
        "fraction",
        "BVI thrust perturbation",
        seed=5600,
        axis_idx=0,
        axis_name="rotor 0",
    )
    _ou_spectrum(
        hardware,
        "blade_vortex",
        BladeVortexInteraction,
        "fraction",
        "BVI thrust perturbation",
        seed=5601,
        axis_idx=0,
    )


# ===================================================================
# 5.7 GroundEffectBoundary — altitude sweep + per-env
# ===================================================================


def plot_ground_effect(hardware: HardwareMeta) -> None:
    n_pts = 100
    altitudes = torch.linspace(0.01, 0.5, n_pts)
    diameters = [0.06, 0.092, 0.15]
    rows: list[dict] = []
    traces_data: dict[str, list[float]] = {}

    for diam in diameters:
        p = GroundEffectBoundary(n_envs=n_pts, dt=DT)
        p._current_value = torch.full((n_pts,), diam)
        env_state = _make_env_state(n_pts)
        env_state.pos[:, 2] = altitudes
        wrench = p._compute_wrench(env_state)
        forces = wrench[:, 2].tolist()
        key = f"D={diam}m"
        traces_data[key] = forces
        for alt, f in zip(altitudes.tolist(), forces):
            rows.append(
                {
                    "altitude_m": round(alt, 3),
                    "diameter_m": diam,
                    "delta_Fz_N": round(f, 6),
                }
            )

    stats_by = {k: stats_summary(v) for k, v in traces_data.items()}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for diam in diameters:
            key = f"D={diam}m"
            fig.add_trace(
                go.Scatter(
                    x=altitudes.numpy().tolist(),
                    y=traces_data[key],
                    mode="lines",
                    name=key,
                    line=dict(width=2),
                )
            )
        apply_layout(
            fig,
            title="Cat 5 — ground_effect: ΔF_z vs altitude",
            subtitle=("Cheeseman-Bennett model: k_ge = 1/(1−(R/4h)²); nominal_thrust=0.265 N"),
            xaxis_title="altitude [m]",
            yaxis_title="ΔF_z [N]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_ground_effect_altitude_sweep",
        category=CAT,
        metric="ground effect force vs altitude",
        unit="N",
        baseline_description="Cheeseman-Bennett model, zeroed above 3×diameter",
        config={"diameters": diameters, "n_points": n_pts},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["altitude_m", "diameter_m", "delta_Fz_N"],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print("  [ground_effect/altitude_sweep] done")

    torch.manual_seed(5700)
    p = GroundEffectBoundary(n_envs=PER_ENV_N, dt=DT)
    p.tick(is_reset=True)
    p.tick(is_reset=False)
    diam_vals = p._current_value.flatten().tolist()
    rows_env: list[dict] = [
        {"env_id": e, "diameter_m": round(d, 4)} for e, d in enumerate(diam_vals)
    ]
    stats_diam = stats_summary(diam_vals)

    def fig_fn_env(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Bar(
                x=list(range(PER_ENV_N)),
                y=diam_vals,
                marker_color=COLOR,
                name="diameter",
            )
        )
        apply_layout(
            fig,
            title="Cat 5 — ground_effect: per-env rotor diameter",
            subtitle=f"n_envs={PER_ENV_N}, curriculum_scale=1.0",
            xaxis_title="environment",
            yaxis_title="rotor diameter [m]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_ground_effect_per_env",
        category=CAT,
        metric="rotor diameter per env",
        unit="m",
        baseline_description="sampled uniformly ∈ [0.07, 0.15] m",
        config={"n_envs": PER_ENV_N},
        hardware=hardware,
        csv_rows=rows_env,
        csv_columns=["env_id", "diameter_m"],
        stats_by_series={"diameter": stats_diam},
        figure_fn=fig_fn_env,
    )
    print("  [ground_effect/per_env] done")


# ===================================================================
# 5.8 PayloadSway — trace + phase-space
# ===================================================================


def plot_payload_sway(hardware: HardwareMeta) -> None:
    n_envs = TRACE_N_ENVS
    n_steps = TRACE_STEPS
    torch.manual_seed(5800)
    p = PayloadSway(n_envs=n_envs, dt=DT, curriculum_scale=1.0)
    p.tick(is_reset=True)

    theta_trace = torch.zeros(n_steps, n_envs)
    theta_dot_trace = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.tick(is_reset=False)
        theta_trace[step] = p._theta[:, 0].clone()
        theta_dot_trace[step] = p._theta_dot[:, 0].clone()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "theta_x_rad": float(theta_trace[step, e]),
                    "theta_dot_x_rads": float(theta_dot_trace[step, e]),
                }
            )

    stats_theta = {f"env_{e}": stats_summary(theta_trace[:, e].tolist()) for e in range(n_envs)}

    def fig_fn_trace(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=theta_trace[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.3),
                )
            )
        apply_layout(
            fig,
            title="Cat 5 — payload_sway: pendulum angle θ_x trace",
            subtitle=(f"n_envs={n_envs}, n_steps={n_steps}, dt={DT}s, symplectic Euler"),
            xaxis_title="time [s]",
            yaxis_title="θ_x [rad]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_payload_sway_trace",
        category=CAT,
        metric="pendulum angle trace",
        unit="rad",
        baseline_description="symplectic Euler ODE: θ̈ = −(g/L)sin(θ)",
        config={"n_envs": n_envs, "n_steps": n_steps, "dt_s": DT},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "theta_x_rad", "theta_dot_x_rads"],
        stats_by_series=stats_theta,
        figure_fn=fig_fn_trace,
    )
    print(f"  [payload_sway/trace] logged {len(rows)} rows")

    def fig_fn_phase(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=theta_trace[:, e].tolist(),
                    y=theta_dot_trace[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.2),
                )
            )
        apply_layout(
            fig,
            title="Cat 5 — payload_sway: phase-space portrait (θ, θ̇)",
            subtitle=(
                f"n_envs={n_envs}, n_steps={n_steps}, elliptical orbits ≈ energy conservation"
            ),
            xaxis_title="θ_x [rad]",
            yaxis_title="θ̇_x [rad/s]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_payload_sway_phase_space",
        category=CAT,
        metric="phase-space portrait",
        unit="rad × rad/s",
        baseline_description="(θ, θ̇) orbits from symplectic Euler",
        config={"n_envs": n_envs, "n_steps": n_steps},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "theta_x_rad", "theta_dot_x_rads"],
        stats_by_series=stats_theta,
        figure_fn=fig_fn_phase,
    )
    print("  [payload_sway/phase_space] done")


# ===================================================================
# 5.9 ProximityDisturbance — force vs distance + per-env
# ===================================================================


def plot_proximity(hardware: HardwareMeta) -> None:
    n_pts = 100
    f_maxes = [0.1, 0.25, 0.5]
    x_positions = torch.linspace(1.5, 2.0, n_pts)
    distances = (2.0 - x_positions).tolist()

    rows: list[dict] = []
    traces_data: dict[str, list[float]] = {}

    for f_max in f_maxes:
        p = ProximityDisturbance(n_envs=n_pts, dt=DT, surface_distance=2.0)
        p._current_value = torch.full((n_pts,), f_max)
        env_state = _make_env_state(n_pts)
        env_state.pos[:, 0] = x_positions
        env_state.pos[:, 1] = 0.0
        env_state.pos[:, 2] = 1.0
        wrench = p._compute_wrench(env_state)
        forces = wrench[:, 0].tolist()
        key = f"F_max={f_max}N"
        traces_data[key] = forces
        for d, f in zip(distances, forces):
            rows.append(
                {
                    "distance_to_wall_m": round(d, 4),
                    "f_max_N": f_max,
                    "force_x_N": round(f, 6),
                }
            )

    stats_by = {k: stats_summary(v) for k, v in traces_data.items()}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for f_max in f_maxes:
            key = f"F_max={f_max}N"
            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=traces_data[key],
                    mode="lines",
                    name=key,
                    line=dict(width=2),
                )
            )
        apply_layout(
            fig,
            title="Cat 5 — proximity_disturbance: F_x vs distance to wall",
            subtitle=("F = F_max × (1 − d/d_max)² for d < d_max; d_max=0.3 m, surface=2.0 m"),
            xaxis_title="distance to wall [m]",
            yaxis_title="F_x [N]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_proximity_force_vs_distance",
        category=CAT,
        metric="proximity force vs distance",
        unit="N",
        baseline_description="quadratic repulsive force within d_max radius",
        config={"f_maxes": f_maxes, "d_max": 0.3, "surface_distance": 2.0},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["distance_to_wall_m", "f_max_N", "force_x_N"],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print("  [proximity/force_vs_distance] done")

    torch.manual_seed(5900)
    p = ProximityDisturbance(n_envs=PER_ENV_N, dt=DT)
    p.tick(is_reset=True)
    p.tick(is_reset=False)
    fmax_vals = p._current_value.flatten().tolist()
    rows_env: list[dict] = [{"env_id": e, "f_max_N": round(f, 4)} for e, f in enumerate(fmax_vals)]
    stats_fmax = stats_summary(fmax_vals)

    def fig_fn_env(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Bar(
                x=list(range(PER_ENV_N)),
                y=fmax_vals,
                marker_color=COLOR,
                name="F_max",
            )
        )
        apply_layout(
            fig,
            title="Cat 5 — proximity_disturbance: per-env F_max",
            subtitle=f"n_envs={PER_ENV_N}, curriculum_scale=1.0",
            xaxis_title="environment",
            yaxis_title="F_max [N]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat5_proximity_per_env",
        category=CAT,
        metric="F_max per env",
        unit="N",
        baseline_description="sampled uniformly ∈ [0, 0.5] N",
        config={"n_envs": PER_ENV_N},
        hardware=hardware,
        csv_rows=rows_env,
        csv_columns=["env_id", "f_max_N"],
        stats_by_series={"f_max": stats_fmax},
        figure_fn=fig_fn_env,
    )
    print("  [proximity/per_env] done")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    print("\n--- 5.1 constant_wind ---")
    plot_constant_wind(hardware)

    print("\n--- 5.2 turbulence ---")
    plot_turbulence(hardware)

    print("\n--- 5.3 wind_gust ---")
    plot_wind_gust(hardware)

    print("\n--- 5.4 wind_shear ---")
    plot_wind_shear(hardware)

    print("\n--- 5.5 adversarial_wind ---")
    plot_adversarial_wind(hardware)

    print("\n--- 5.6 blade_vortex_interaction ---")
    plot_bvi(hardware)

    print("\n--- 5.7 ground_effect_boundary ---")
    plot_ground_effect(hardware)

    print("\n--- 5.8 payload_sway ---")
    plot_payload_sway(hardware)

    print("\n--- 5.9 proximity_disturbance ---")
    plot_proximity(hardware)

    print("\nDone. Inspect docs/impl/{data,assets}/cat5_*.{csv,meta.json,png}")


if __name__ == "__main__":
    main()
