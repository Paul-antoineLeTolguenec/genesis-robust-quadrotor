"""Category 8 — External disturbances plots.

Generates curriculum + OU trace/spectrum/autocorrelation PNGs for:
- body_force_disturbance
- body_torque_disturbance

Both perturbations support two regimes:
- constant: value sampled per episode (uniform/gaussian)
- ou_process: stateful 3D Ornstein-Uhlenbeck driven wrench

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_8.py
"""

from __future__ import annotations

import os
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
from genesis_robust_rl.perturbations.category_8_external import (  # noqa: E402
    BodyForceDisturbance,
    BodyTorqueDisturbance,
)

CAT = 8
DT = 0.01
N_ENVS_SAMPLE = 256
N_EPISODES_PER_ENV = 40
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
TRACE_STEPS = 400
TRACE_N_ENVS = 4
SPECTRUM_STEPS = 2048
AUTOCORR_LAG = 300


def _make_env_state(n_envs: int) -> EnvState:
    return EnvState(
        pos=torch.tensor([[0.0, 0.0, 0.5]]).expand(n_envs, 3).clone(),
        quat=torch.tensor([[1.0, 0, 0, 0]]).expand(n_envs, 4).clone(),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 14000.0,
        dt=DT,
        step=0,
    )


def _make_constant(pert_cls: type, n_envs: int, curriculum_scale: float):
    return pert_cls(
        n_envs=n_envs,
        dt=DT,
        distribution="uniform",
        curriculum_scale=curriculum_scale,
    )


def _make_ou(pert_cls: type, n_envs: int, curriculum_scale: float = 1.0):
    return pert_cls(
        n_envs=n_envs,
        dt=DT,
        distribution="ou_process",
        curriculum_scale=curriculum_scale,
    )


def generate_constant_curriculum(
    hardware: HardwareMeta, slug: str, pert_cls: type, unit: str
) -> None:
    rows: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(8000 + i)
        p = _make_constant(pert_cls, N_ENVS_SAMPLE, scale)
        values: list[float] = []
        for _ in range(N_EPISODES_PER_ENV):
            v = p.sample()
            mag = v.norm(dim=-1)
            for m in mag.tolist():
                values.append(float(m))
        samples_by_scale[f"{scale:.2f}"] = values
        for v in values:
            rows.append({"curriculum_scale": float(scale), "wrench_magnitude": v})

    stats_by_series = {s: stats_summary(v) for s, v in samples_by_scale.items()}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for scale_s, values in samples_by_scale.items():
            fig.add_trace(
                go.Violin(
                    y=values,
                    x=[scale_s] * len(values),
                    name=f"scale={scale_s}",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    line_color=CATEGORY_COLORS[CAT],
                    fillcolor=CATEGORY_COLORS[CAT],
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
            title=f"Cat 8 — {slug}: constant-mode ‖wrench‖₂ vs curriculum_scale",
            subtitle=(
                f"distribution=uniform, n_envs={N_ENVS_SAMPLE}, draws/env={N_EPISODES_PER_ENV}"
            ),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title=f"‖wrench‖₂ [{unit}]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[max(0, y_min - pad), y_max + pad])
        return fig

    log_and_plot(
        name=f"cat8_{slug}_constant_curriculum",
        category=CAT,
        metric="wrench magnitude (constant mode)",
        unit=unit,
        baseline_description="curriculum_scale=0 collapses draws to nominal ‖0‖=0",
        config={
            "distribution": "uniform",
            "n_envs": N_ENVS_SAMPLE,
            "draws_per_env": N_EPISODES_PER_ENV,
            "curriculum_scales": list(CURRICULUM_SCALES),
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["curriculum_scale", "wrench_magnitude"],
        stats_by_series=stats_by_series,
        figure_fn=fig_fn,
    )
    print(f"[{slug}/const_curriculum] logged {len(rows)} samples")


def _simulate_ou_trace(pert_cls: type, n_envs: int, n_steps: int, seed: int):
    torch.manual_seed(seed)
    p = _make_ou(pert_cls, n_envs, curriculum_scale=1.0)
    env_state = _make_env_state(n_envs)
    env_ids = torch.arange(n_envs)
    p.tick(is_reset=True, env_ids=env_ids)
    trace = torch.empty(n_steps, n_envs, 3)
    for step in range(n_steps):
        p.tick(is_reset=False)
        trace[step] = p._compute_wrench(env_state).detach().clone()
    return trace


def generate_ou_trace(hardware: HardwareMeta, slug: str, pert_cls: type, unit: str) -> None:
    trace = _simulate_ou_trace(pert_cls, TRACE_N_ENVS, TRACE_STEPS, seed=8100)
    mag = trace.norm(dim=-1)

    rows: list[dict] = []
    for step in range(TRACE_STEPS):
        for env in range(TRACE_N_ENVS):
            rows.append(
                {
                    "step": step,
                    "env_id": env,
                    "time_s": step * DT,
                    "wrench_magnitude": float(mag[step, env]),
                }
            )

    stats_per_env = {f"env_{e}": stats_summary(mag[:, e].tolist()) for e in range(TRACE_N_ENVS)}
    stats_per_env["all_envs"] = stats_summary(mag.reshape(-1).tolist())

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [i * DT for i in range(TRACE_STEPS)]
        for env in range(TRACE_N_ENVS):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=mag[:, env].tolist(),
                    mode="lines",
                    name=f"env {env}",
                    line=dict(width=1.3),
                )
            )
        apply_layout(
            fig,
            title=f"Cat 8 — {slug}: OU-mode ‖wrench‖₂ trace",
            subtitle=(f"θ=1.0, σ per-env, n_envs={TRACE_N_ENVS}, n_steps={TRACE_STEPS}, dt={DT}s"),
            xaxis_title="time [s]",
            yaxis_title=f"‖wrench‖₂ [{unit}]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"cat8_{slug}_ou_trace",
        category=CAT,
        metric="wrench magnitude (OU mode)",
        unit=unit,
        baseline_description="OU starts at 0 and evolves as θ(μ−w)dt + σ√dt·ε",
        config={
            "distribution": "ou_process",
            "n_envs": TRACE_N_ENVS,
            "n_steps": TRACE_STEPS,
            "dt_s": DT,
            "theta": 1.0,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "wrench_magnitude"],
        stats_by_series=stats_per_env,
        figure_fn=fig_fn,
    )
    print(f"[{slug}/ou_trace] logged {len(rows)} rows")


def generate_ou_spectrum(hardware: HardwareMeta, slug: str, pert_cls: type, unit: str) -> None:
    trace = _simulate_ou_trace(pert_cls, TRACE_N_ENVS, SPECTRUM_STEPS, seed=8200)
    signal = trace[:, :, 0].numpy()
    fs = 1.0 / DT

    freqs = np.fft.rfftfreq(SPECTRUM_STEPS, d=DT)
    power = np.zeros_like(freqs)
    for env in range(TRACE_N_ENVS):
        x = signal[:, env] - signal[:, env].mean()
        X = np.fft.rfft(x)
        power += (np.abs(X) ** 2) / SPECTRUM_STEPS
    power /= TRACE_N_ENVS

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
                line=dict(color=CATEGORY_COLORS[CAT], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freqs[mask].tolist(),
                y=white_ref[mask].tolist(),
                mode="lines",
                name="white-noise reference (flat)",
                line=dict(color="#d62728", dash="dash", width=1),
            )
        )
        apply_layout(
            fig,
            title=f"Cat 8 — {slug}: OU-mode power spectrum",
            subtitle=(
                f"OU expected 1/f² roll-off; white-noise flat. "
                f"n_steps={SPECTRUM_STEPS}, n_envs={TRACE_N_ENVS}, fs={fs:.0f} Hz"
            ),
            xaxis_title="frequency [Hz]",
            yaxis_title=f"power [{unit}²/Hz]",
            xaxis_type="log",
            yaxis_type="log",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"cat8_{slug}_ou_spectrum",
        category=CAT,
        metric="power spectrum (OU mode)",
        unit=f"{unit}²/Hz",
        baseline_description="OU vs white-noise flat reference at same variance",
        config={
            "distribution": "ou_process",
            "n_envs": TRACE_N_ENVS,
            "n_steps": SPECTRUM_STEPS,
            "dt_s": DT,
            "fs_hz": fs,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["frequency_hz", "ou_power", "white_noise_ref"],
        stats_by_series={"ou_power_no_dc": stats_power},
        figure_fn=fig_fn,
    )
    print(f"[{slug}/ou_spectrum] logged {len(rows)} rows")


def generate_ou_autocorr(hardware: HardwareMeta, slug: str, pert_cls: type, unit: str) -> None:
    trace = _simulate_ou_trace(pert_cls, TRACE_N_ENVS, SPECTRUM_STEPS, seed=8300)
    signal = trace[:, :, 0].numpy()

    # Vectorized autocorrelation via FFT: ifft(|fft(x)|²) — O(N log N) per env
    acf = np.zeros(AUTOCORR_LAG + 1)
    n = signal.shape[0]
    pad = 2 * n
    for env in range(TRACE_N_ENVS):
        x = signal[:, env] - signal[:, env].mean()
        f = np.fft.rfft(x, n=pad)
        raw = np.fft.irfft(f * np.conj(f), n=pad)[: AUTOCORR_LAG + 1].real
        acf += raw / raw[0]
    acf /= TRACE_N_ENVS

    lags_s = [lag * DT for lag in range(AUTOCORR_LAG + 1)]
    rows: list[dict] = []
    for lag, v in enumerate(acf.tolist()):
        rows.append({"lag_steps": lag, "lag_s": lag * DT, "autocorr": float(v)})

    stats_acf = stats_summary(acf[1:].tolist())

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Scatter(
                x=lags_s,
                y=acf.tolist(),
                mode="lines+markers",
                name="empirical ACF",
                line=dict(color=CATEGORY_COLORS[CAT], width=2),
                marker=dict(size=4),
            )
        )
        theta = 1.0
        theo = [float(np.exp(-theta * t)) for t in lags_s]
        fig.add_trace(
            go.Scatter(
                x=lags_s,
                y=theo,
                mode="lines",
                name=f"theoretical exp(-θ·τ), θ={theta}",
                line=dict(color="#d62728", dash="dash", width=1),
            )
        )
        apply_layout(
            fig,
            title=f"Cat 8 — {slug}: OU-mode autocorrelation",
            subtitle=(
                f"empirical ACF vs theoretical exp(-θ·τ); "
                f"n_envs={TRACE_N_ENVS}, n_steps={SPECTRUM_STEPS}, dt={DT}s"
            ),
            xaxis_title="lag τ [s]",
            yaxis_title="autocorrelation (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"cat8_{slug}_ou_autocorr",
        category=CAT,
        metric="autocorrelation (OU mode)",
        unit="dimensionless",
        baseline_description="empirical vs theoretical exp(-θ·τ) with θ=1.0",
        config={
            "distribution": "ou_process",
            "n_envs": TRACE_N_ENVS,
            "n_steps": SPECTRUM_STEPS,
            "dt_s": DT,
            "max_lag_steps": AUTOCORR_LAG,
            "theta": 1.0,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["lag_steps", "lag_s", "autocorr"],
        stats_by_series={"acf_excluding_zero": stats_acf},
        figure_fn=fig_fn,
    )
    print(f"[{slug}/ou_autocorr] logged {len(rows)} rows")


def run_perf() -> None:  # pragma: no cover
    raise NotImplementedError(
        "Perf runner placeholder. Model on docs/impl/plot_pilot_mass_shift.py."
    )


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    specs = [
        ("body_force_disturbance", BodyForceDisturbance, "N"),
        ("body_torque_disturbance", BodyTorqueDisturbance, "N·m"),
    ]
    for slug, cls, unit in specs:
        print(f"--- {slug} ---")
        generate_constant_curriculum(hardware, slug, cls, unit)
        generate_ou_trace(hardware, slug, cls, unit)
        generate_ou_spectrum(hardware, slug, cls, unit)
        generate_ou_autocorr(hardware, slug, cls, unit)

    print("\nDone. Inspect docs/impl/{data,assets}/cat8_*.{csv,meta.json,png}")


if __name__ == "__main__":
    if os.getenv("RUN_GENESIS") == "1":
        run_perf()
    main()
