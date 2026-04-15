"""Category 6 — Action perturbations plots.

Generates curriculum + per_env + effect PNGs for:
- action_noise (σ per-step)
- action_deadzone (threshold per-episode)
- action_saturation (limit per-episode)
- actuator_hysteresis (width per-episode, stateful direction)
- esc_low_pass_filter (cutoff freq per-episode, IIR)

The "effect" plot is the hero for Cat 6 — input vs output on synthetic actions,
adapted per perturbation: ramp for deadzone/saturation, sine for hysteresis
(+ Lissajous), step for IIR rise, constant-zero + noise cloud for action_noise.

CSV + meta JSON logged BEFORE each PNG via shared framework.

Run:
    uv run python docs/impl/plot_category_6.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

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

from genesis_robust_rl.perturbations.category_6_action import (  # noqa: E402
    ActionDeadzone,
    ActionNoise,
    ActionSaturation,
    ActuatorHysteresis,
    ESCLowPassFilter,
)

CAT = 6
DT = 0.01
N_ENVS_SAMPLE = 256
N_DRAWS = 40
PER_ENV_N_ENVS = 16
PER_ENV_N_EPISODES = 30
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
ACTION_DIM = 4
EFFECT_STEPS = 400


# ---------------------------------------------------------------------------
# Shared curriculum + per_env helpers
# ---------------------------------------------------------------------------


def _curriculum_samples(
    factory, param_name: str, unit: str, n_envs: int, n_draws: int, scales, seed_base: int
):
    """factory(n_envs, scale) -> perturbation with scalar _current_value.
    Returns samples_by_scale dict + rows list ready for log_and_plot."""
    samples_by_scale: dict[str, list[float]] = {}
    rows: list[dict] = []
    for i, scale in enumerate(scales):
        torch.manual_seed(seed_base + i)
        p = factory(n_envs, scale)
        values: list[float] = []
        for _ in range(n_draws):
            p.sample()
            v = p._current_value  # [n_envs]
            for x in v.tolist():
                values.append(float(x))
        samples_by_scale[f"{scale:.2f}"] = values
        for v in values:
            rows.append({"curriculum_scale": float(scale), param_name: v})
    return samples_by_scale, rows


def _violin_figure(samples_by_scale, *, title, subtitle, xaxis_title, yaxis_title, hw):
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
        title=title,
        subtitle=subtitle,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hardware_footer=hw.footer_line(),
    )
    fig.update_yaxes(range=[y_min - pad, y_max + pad])
    return fig


def _per_env_samples(factory, param_name: str, seed: int):
    """factory(n_envs, scale=1.0). Returns [n_episodes, n_envs] tensor + rows."""
    torch.manual_seed(seed)
    p = factory(PER_ENV_N_ENVS, 1.0)
    samples = torch.empty(PER_ENV_N_EPISODES, PER_ENV_N_ENVS)
    for ep in range(PER_ENV_N_EPISODES):
        p.sample()
        samples[ep] = p._current_value.clone()
    rows: list[dict] = []
    for ep in range(PER_ENV_N_EPISODES):
        for e in range(PER_ENV_N_ENVS):
            rows.append({"episode": ep, "env_id": e, param_name: float(samples[ep, e])})
    return samples, rows


def _per_env_figure(samples, *, title, yaxis_title, hw):
    fig = make_figure()
    for e in range(PER_ENV_N_ENVS):
        fig.add_trace(
            go.Scatter(
                x=list(range(PER_ENV_N_EPISODES)),
                y=samples[:, e].tolist(),
                mode="lines+markers",
                name=f"env {e}",
                line=dict(width=1),
                marker=dict(size=4),
                opacity=0.6,
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(PER_ENV_N_EPISODES)),
            y=samples.mean(dim=1).tolist(),
            mode="lines",
            name="mean across envs",
            line=dict(width=3, color="black"),
        )
    )
    apply_layout(
        fig,
        title=title,
        subtitle=(f"n_envs={PER_ENV_N_ENVS}, n_episodes={PER_ENV_N_EPISODES}, curriculum=1.0"),
        xaxis_title="episode index",
        yaxis_title=yaxis_title,
        hardware_footer=hw.footer_line(),
    )
    return fig


def _commit_curriculum(
    hardware, slug, param_name, unit, factory, dist_spec, title_metric, seed_base
):
    samples_by_scale, rows = _curriculum_samples(
        factory, param_name, unit, N_ENVS_SAMPLE, N_DRAWS, CURRICULUM_SCALES, seed_base
    )
    stats_by = {k: stats_summary(v) for k, v in samples_by_scale.items()}

    def fig_fn(hw):
        return _violin_figure(
            samples_by_scale,
            title=f"Cat 6 — {slug}: {title_metric} vs curriculum_scale",
            subtitle=(f"{dist_spec} — n_envs={N_ENVS_SAMPLE}, draws/env={N_DRAWS}"),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title=f"{title_metric} [{unit}]",
            hw=hw,
        )

    log_and_plot(
        name=f"cat6_{slug}_curriculum",
        category=CAT,
        metric=title_metric,
        unit=unit,
        baseline_description="curriculum=0 collapses samples to nominal",
        config={
            "distribution_spec": dist_spec,
            "n_envs": N_ENVS_SAMPLE,
            "draws_per_env": N_DRAWS,
            "curriculum_scales": list(CURRICULUM_SCALES),
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["curriculum_scale", param_name],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print(f"[{slug}/curriculum] logged {len(rows)} samples")


def _commit_per_env(hardware, slug, param_name, unit, factory, title_metric, seed):
    samples, rows = _per_env_samples(factory, param_name, seed)
    stats_by = {f"env_{e}": stats_summary(samples[:, e].tolist()) for e in range(PER_ENV_N_ENVS)}
    stats_by["all"] = stats_summary(samples.reshape(-1).tolist())

    def fig_fn(hw):
        return _per_env_figure(
            samples,
            title=f"Cat 6 — {slug}: per-env {title_metric} across episodes",
            yaxis_title=f"{title_metric} [{unit}]",
            hw=hw,
        )

    log_and_plot(
        name=f"cat6_{slug}_per_env",
        category=CAT,
        metric=title_metric,
        unit=unit,
        baseline_description="one sample per (episode, env)",
        config={"n_envs": PER_ENV_N_ENVS, "n_episodes": PER_ENV_N_EPISODES},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["episode", "env_id", param_name],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print(f"[{slug}/per_env] logged {len(rows)} samples")


# ---------------------------------------------------------------------------
# action_noise
# ---------------------------------------------------------------------------


def _noise_factory(n_envs, scale):
    return ActionNoise(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM, curriculum_scale=scale)


def generate_action_noise(hardware: HardwareMeta) -> None:
    _commit_curriculum(
        hardware,
        "action_noise",
        "noise_sigma",
        "dimensionless",
        _noise_factory,
        "uniform σ ∈ [0.0, 0.05]",
        "noise σ",
        seed_base=6100,
    )
    _commit_per_env(
        hardware, "action_noise", "noise_sigma", "dimensionless", _noise_factory, "noise σ", 6200
    )
    # effect: input = zeros, output = noise cloud via time trace
    torch.manual_seed(6300)
    p = _noise_factory(PER_ENV_N_ENVS, 1.0)
    p.tick(is_reset=True, env_ids=torch.arange(PER_ENV_N_ENVS))
    input_action = torch.zeros(PER_ENV_N_ENVS, ACTION_DIM)
    out = torch.empty(EFFECT_STEPS, PER_ENV_N_ENVS, ACTION_DIM)
    for step in range(EFFECT_STEPS):
        p.tick(is_reset=False)
        out[step] = p.apply(input_action)
    # Track channel 0 only
    out_ch0 = out[:, :, 0]  # [steps, envs]

    rows: list[dict] = []
    for step in range(EFFECT_STEPS):
        for e in range(min(4, PER_ENV_N_ENVS)):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "input_action": 0.0,
                    "output_action": float(out_ch0[step, e]),
                }
            )
    stats_effect = stats_summary(out_ch0.reshape(-1).tolist())

    def fig_fn(hw):
        fig = make_figure()
        for e in range(4):
            fig.add_trace(
                go.Scatter(
                    x=list(range(EFFECT_STEPS)),
                    y=out_ch0[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1),
                )
            )
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#888",
            annotation_text="input = 0",
            annotation_position="top right",
        )
        apply_layout(
            fig,
            title="Cat 6 — action_noise: effect (input=0 vs noisy output, channel 0)",
            subtitle=f"σ ~ U(0, 0.05), curriculum=1.0, n_envs_shown=4, steps={EFFECT_STEPS}",
            xaxis_title="step",
            yaxis_title="action value (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat6_action_noise_effect",
        category=CAT,
        metric="noisy output vs zero input",
        unit="dimensionless",
        baseline_description="input constant at 0; output is pure noise centered at 0",
        config={
            "n_envs_shown": 4,
            "n_steps": EFFECT_STEPS,
            "action_dim": ACTION_DIM,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "input_action", "output_action"],
        stats_by_series={"output_all_envs_channel0": stats_effect},
        figure_fn=fig_fn,
    )
    print(f"[action_noise/effect] logged {len(rows)} rows")


# ---------------------------------------------------------------------------
# action_deadzone — ramp input, flat band output
# ---------------------------------------------------------------------------


def _deadzone_factory(n_envs, scale):
    return ActionDeadzone(n_envs=n_envs, dt=DT, curriculum_scale=scale)


def generate_action_deadzone(hardware: HardwareMeta) -> None:
    _commit_curriculum(
        hardware,
        "action_deadzone",
        "threshold",
        "dimensionless",
        _deadzone_factory,
        "uniform threshold ∈ [0.0, 0.05]",
        "deadzone threshold",
        seed_base=6400,
    )
    _commit_per_env(
        hardware,
        "action_deadzone",
        "threshold",
        "dimensionless",
        _deadzone_factory,
        "deadzone threshold",
        6500,
    )

    # effect: input ramp [-1, 1], output shows flat band
    torch.manual_seed(6600)
    n = 4
    p = ActionDeadzone(n_envs=n, dt=DT, curriculum_scale=1.0)
    # Force max thresholds to amplify visible effect
    p._current_value = torch.tensor([0.01, 0.02, 0.035, 0.05])
    ramp = torch.linspace(-0.2, 0.2, EFFECT_STEPS)  # small range so threshold is visible
    outputs = torch.empty(EFFECT_STEPS, n)
    for step in range(EFFECT_STEPS):
        action = ramp[step].expand(n, ACTION_DIM).clone()
        out = p.apply(action)
        outputs[step] = out[:, 0]

    rows: list[dict] = []
    thresholds = p._current_value.tolist()
    for step in range(EFFECT_STEPS):
        for e in range(n):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "threshold": thresholds[e],
                    "input_action": float(ramp[step]),
                    "output_action": float(outputs[step, e]),
                }
            )

    def fig_fn(hw):
        fig = make_figure()
        fig.add_trace(
            go.Scatter(
                x=ramp.tolist(),
                y=ramp.tolist(),
                mode="lines",
                name="pass-through (y=x)",
                line=dict(color="#888", dash="dash", width=1),
            )
        )
        for e in range(n):
            fig.add_trace(
                go.Scatter(
                    x=ramp.tolist(),
                    y=outputs[:, e].tolist(),
                    mode="lines",
                    name=f"threshold={thresholds[e]:.3f}",
                    line=dict(width=2),
                )
            )
        apply_layout(
            fig,
            title="Cat 6 — action_deadzone: output vs input (ramp)",
            subtitle=f"n_envs_shown={n}, thresholds vary per env, curriculum=1.0",
            xaxis_title="input action (dimensionless)",
            yaxis_title="output action (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat6_action_deadzone_effect",
        category=CAT,
        metric="deadzone output vs input ramp",
        unit="dimensionless",
        baseline_description="|a| < threshold → 0; else pass-through",
        config={"n_envs_shown": n, "thresholds_used": thresholds, "action_dim": ACTION_DIM},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "threshold", "input_action", "output_action"],
        stats_by_series={"output_all": stats_summary(outputs.reshape(-1).tolist())},
        figure_fn=fig_fn,
    )
    print(f"[action_deadzone/effect] logged {len(rows)} rows")


# ---------------------------------------------------------------------------
# action_saturation — ramp input, clipping at ±limit
# ---------------------------------------------------------------------------


def _saturation_factory(n_envs, scale):
    return ActionSaturation(n_envs=n_envs, dt=DT, curriculum_scale=scale)


def generate_action_saturation(hardware: HardwareMeta) -> None:
    _commit_curriculum(
        hardware,
        "action_saturation",
        "limit",
        "dimensionless",
        _saturation_factory,
        "uniform limit ∈ [0.7, 1.0]",
        "saturation limit",
        seed_base=6700,
    )
    _commit_per_env(
        hardware,
        "action_saturation",
        "limit",
        "dimensionless",
        _saturation_factory,
        "saturation limit",
        6800,
    )

    # effect: input ramp [-1.2, 1.2], output clips at ±limit
    torch.manual_seed(6900)
    n = 4
    p = ActionSaturation(n_envs=n, dt=DT, curriculum_scale=1.0)
    p._current_value = torch.tensor([0.5, 0.7, 0.85, 1.0])
    ramp = torch.linspace(-1.2, 1.2, EFFECT_STEPS)
    outputs = torch.empty(EFFECT_STEPS, n)
    for step in range(EFFECT_STEPS):
        action = ramp[step].expand(n, ACTION_DIM).clone()
        out = p.apply(action)
        outputs[step] = out[:, 0]

    rows: list[dict] = []
    limits = p._current_value.tolist()
    for step in range(EFFECT_STEPS):
        for e in range(n):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "limit": limits[e],
                    "input_action": float(ramp[step]),
                    "output_action": float(outputs[step, e]),
                }
            )

    def fig_fn(hw):
        fig = make_figure()
        fig.add_trace(
            go.Scatter(
                x=ramp.tolist(),
                y=ramp.tolist(),
                mode="lines",
                name="pass-through (y=x)",
                line=dict(color="#888", dash="dash", width=1),
            )
        )
        for e in range(n):
            fig.add_trace(
                go.Scatter(
                    x=ramp.tolist(),
                    y=outputs[:, e].tolist(),
                    mode="lines",
                    name=f"limit={limits[e]:.2f}",
                    line=dict(width=2),
                )
            )
        apply_layout(
            fig,
            title="Cat 6 — action_saturation: output vs input (ramp)",
            subtitle=f"n_envs_shown={n}, limits vary per env, curriculum=1.0",
            xaxis_title="input action (dimensionless)",
            yaxis_title="output action (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat6_action_saturation_effect",
        category=CAT,
        metric="saturation output vs input ramp",
        unit="dimensionless",
        baseline_description="clamp(a, -limit, limit) per env",
        config={"n_envs_shown": n, "limits_used": limits, "action_dim": ACTION_DIM},
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "limit", "input_action", "output_action"],
        stats_by_series={"output_all": stats_summary(outputs.reshape(-1).tolist())},
        figure_fn=fig_fn,
    )
    print(f"[action_saturation/effect] logged {len(rows)} rows")


# ---------------------------------------------------------------------------
# actuator_hysteresis — sine input + Lissajous output-vs-input
# ---------------------------------------------------------------------------


def _hysteresis_factory(n_envs, scale):
    return ActuatorHysteresis(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM, curriculum_scale=scale)


def generate_actuator_hysteresis(hardware: HardwareMeta) -> None:
    _commit_curriculum(
        hardware,
        "actuator_hysteresis",
        "width",
        "dimensionless",
        _hysteresis_factory,
        "uniform width ∈ [0.0, 0.03]",
        "hysteresis width",
        seed_base=6920,
    )
    _commit_per_env(
        hardware,
        "actuator_hysteresis",
        "width",
        "dimensionless",
        _hysteresis_factory,
        "hysteresis width",
        6940,
    )

    # effect: sine input + output trace + Lissajous
    torch.manual_seed(6960)
    n = 1
    p = ActuatorHysteresis(n_envs=n, dt=DT, action_dim=ACTION_DIM, curriculum_scale=1.0)
    p._current_value = torch.tensor([0.3])
    p._width = torch.full((n, ACTION_DIM), 0.6)  # amplified ×20 for visible loop
    p._last_action = torch.zeros(n, ACTION_DIM)
    input_signal = torch.tensor([math.sin(2 * math.pi * s / 100) for s in range(EFFECT_STEPS)])
    outputs = torch.empty(EFFECT_STEPS)
    for step in range(EFFECT_STEPS):
        action = input_signal[step].expand(n, ACTION_DIM).clone()
        out = p.apply(action)
        outputs[step] = out[0, 0]

    rows: list[dict] = []
    for step in range(EFFECT_STEPS):
        rows.append(
            {
                "step": step,
                "input_action": float(input_signal[step]),
                "output_action": float(outputs[step]),
            }
        )

    def fig_fn(hw):
        fig = make_figure()
        # Lissajous: input vs output
        fig.add_trace(
            go.Scatter(
                x=input_signal.tolist(),
                y=outputs.tolist(),
                mode="lines",
                name="output(input) — hysteresis loop",
                line=dict(color=CATEGORY_COLORS[CAT], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=input_signal.tolist(),
                y=input_signal.tolist(),
                mode="lines",
                name="pass-through (y=x)",
                line=dict(color="#888", dash="dash", width=1),
            )
        )
        apply_layout(
            fig,
            title="Cat 6 — actuator_hysteresis: Lissajous output vs input (sine)",
            subtitle=(
                f"input=sin(2π·step/100), width=0.06 (amplified to visualise loop), "
                f"steps={EFFECT_STEPS}"
            ),
            xaxis_title="input action (dimensionless)",
            yaxis_title="output action (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat6_actuator_hysteresis_effect",
        category=CAT,
        metric="Lissajous output vs input",
        unit="dimensionless",
        baseline_description="width amplified to 0.06 for visibility; nominal ∈ [0, 0.03]",
        config={
            "n_envs_shown": n,
            "width_amplified": 0.06,
            "input_signal": "sin(2π·step/100)",
            "width_scale_factor": 20,
            "n_steps": EFFECT_STEPS,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "input_action", "output_action"],
        stats_by_series={"output": stats_summary(outputs.tolist())},
        figure_fn=fig_fn,
    )
    print(f"[actuator_hysteresis/effect] logged {len(rows)} rows")


# ---------------------------------------------------------------------------
# esc_low_pass_filter — step input, IIR rise
# ---------------------------------------------------------------------------


def _esc_factory(n_envs, scale):
    return ESCLowPassFilter(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM, curriculum_scale=scale)


def generate_esc_low_pass_filter(hardware: HardwareMeta) -> None:
    _commit_curriculum(
        hardware,
        "esc_low_pass_filter",
        "cutoff_hz",
        "Hz",
        _esc_factory,
        "uniform f_c ∈ [10, 50] Hz",
        "cutoff frequency",
        seed_base=6980,
    )
    _commit_per_env(
        hardware,
        "esc_low_pass_filter",
        "cutoff_hz",
        "Hz",
        _esc_factory,
        "cutoff frequency",
        6990,
    )

    # effect: step input at step=50, output shows IIR rise
    torch.manual_seed(6999)
    n = 4
    p = ESCLowPassFilter(n_envs=n, dt=DT, action_dim=ACTION_DIM, curriculum_scale=1.0)
    # Force specific cutoffs to see the curve family
    cutoffs = torch.tensor([2.0, 5.0, 10.0, 20.0])
    p._current_value = cutoffs.clone()
    p._alpha = (2.0 * math.pi * cutoffs * DT).clamp(0.0, 1.0).unsqueeze(1)
    p._filtered = torch.zeros(n, ACTION_DIM)

    step_in = torch.zeros(EFFECT_STEPS)
    step_in[50:] = 1.0
    outputs = torch.empty(EFFECT_STEPS, n)
    for step in range(EFFECT_STEPS):
        action = step_in[step].expand(n, ACTION_DIM).clone()
        out = p.apply(action)
        outputs[step] = out[:, 0]

    rows: list[dict] = []
    for step in range(EFFECT_STEPS):
        for e in range(n):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "cutoff_hz": float(cutoffs[e]),
                    "input_action": float(step_in[step]),
                    "output_action": float(outputs[step, e]),
                }
            )

    def fig_fn(hw):
        fig = make_figure()
        t = list(range(EFFECT_STEPS))
        fig.add_trace(
            go.Scatter(
                x=t,
                y=step_in.tolist(),
                mode="lines",
                name="input (step at t=50)",
                line=dict(color="#888", dash="dash", width=1.5),
            )
        )
        for e in range(n):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=outputs[:, e].tolist(),
                    mode="lines",
                    name=f"f_c={cutoffs[e]:.0f} Hz",
                    line=dict(width=2),
                )
            )
        apply_layout(
            fig,
            title="Cat 6 — esc_low_pass_filter: step response (IIR rise)",
            subtitle=f"input=step at step=50, n_envs_shown={n}, dt={DT}s",
            xaxis_title="step",
            yaxis_title="action value (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        fig.update_xaxes(range=[40, 200])
        return fig

    log_and_plot(
        name="cat6_esc_low_pass_filter_effect",
        category=CAT,
        metric="step response",
        unit="dimensionless",
        baseline_description="y += (x − y)·α, α = 2π·f_c·dt clamped to [0, 1]",
        config={
            "n_envs_shown": n,
            "cutoffs_hz_used": cutoffs.tolist(),
            "input_signal": "unit step at step=50",
            "n_steps": EFFECT_STEPS,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=[
            "step",
            "env_id",
            "cutoff_hz",
            "input_action",
            "output_action",
        ],
        stats_by_series={"output_all": stats_summary(outputs.reshape(-1).tolist())},
        figure_fn=fig_fn,
    )
    print(f"[esc_low_pass_filter/effect] logged {len(rows)} rows")


# ---------------------------------------------------------------------------
# Genesis perf stub
# ---------------------------------------------------------------------------


def run_perf() -> None:  # pragma: no cover
    raise NotImplementedError(
        "Perf runner placeholder. Model on docs/impl/plot_pilot_mass_shift.py."
    )


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")
    generate_action_noise(hardware)
    generate_action_deadzone(hardware)
    generate_action_saturation(hardware)
    generate_actuator_hysteresis(hardware)
    generate_esc_low_pass_filter(hardware)
    print("\nDone. Inspect docs/impl/{data,assets}/cat6_*.{csv,meta.json,png}")


if __name__ == "__main__":
    if os.getenv("RUN_GENESIS") == "1":
        run_perf()
    main()
