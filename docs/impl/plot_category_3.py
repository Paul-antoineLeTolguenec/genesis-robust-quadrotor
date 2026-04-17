"""Category 3 — Temporal / Latency perturbation plots.

Generates curriculum + temporal-specific PNGs for:
- 3.1 obs_fixed_delay       (curriculum violin + heatmap env×episode)
- 3.2 obs_variable_delay    (curriculum violin + heatmap step×env)
- 3.3 action_fixed_delay    (curriculum violin + heatmap env×episode)
- 3.4 action_variable_delay (curriculum violin + heatmap step×env)
- 3.7 packet_loss           (curriculum violin + drop event heatmap + cumulative rate)
- 3.8 computation_overload  (curriculum violin + stall heatmap + duration histogram)

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_3.py
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

from genesis_robust_rl.perturbations.category_3_temporal import (  # noqa: E402
    ActionFixedDelay,
    ActionVariableDelay,
    ComputationOverload,
    ObsFixedDelay,
    ObsVariableDelay,
    PacketLoss,
)

CAT = 3
DT = 0.01
N_ENVS_CURRICULUM = 256
N_EPISODES_PER_ENV = 40
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
COLOR = CATEGORY_COLORS[CAT]

# Heatmap params
HEATMAP_ENVS_FIXED = 16
HEATMAP_EPISODES_FIXED = 30
HEATMAP_ENVS_VARIABLE = 8
HEATMAP_STEPS_VARIABLE = 200

# PacketLoss / ComputationOverload trace params
EVENT_N_ENVS = 8
EVENT_N_STEPS = 300


# ===================================================================
# Shared helpers
# ===================================================================


def _make_delay_pert(cls: type, n_envs: int, curriculum_scale: float, **extra):
    """Instantiate a delay perturbation (obs or action)."""
    if cls in (ObsFixedDelay, ObsVariableDelay):
        return cls(
            obs_slice=slice(0, 3),
            obs_dim=3,
            n_envs=n_envs,
            dt=DT,
            curriculum_scale=curriculum_scale,
            **extra,
        )
    return cls(n_envs=n_envs, dt=DT, curriculum_scale=curriculum_scale, **extra)


# ===================================================================
# Curriculum violin — shared for all 6 perturbations
# ===================================================================


def generate_curriculum_violin(
    hardware: HardwareMeta,
    slug: str,
    pert_cls: type,
    unit: str,
    label: str,
) -> None:
    rows: list[dict] = []
    samples_by_scale: dict[str, list[float]] = {}

    for i, scale in enumerate(CURRICULUM_SCALES):
        torch.manual_seed(3000 + i)
        p = _make_delay_pert(pert_cls, N_ENVS_CURRICULUM, scale)
        values: list[float] = []
        for _ in range(N_EPISODES_PER_ENV):
            v = p.sample().squeeze(-1)
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
        pad = 0.15 * (y_max - y_min + 1e-6)
        apply_layout(
            fig,
            title=f"Cat 3 — {label}: {unit} vs curriculum_scale",
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
        name=f"cat3_{slug}_curriculum",
        category=CAT,
        metric=f"{label} distribution",
        unit=unit,
        baseline_description=f"curriculum_scale=0 collapses to nominal=0 ({unit})",
        config={
            "distribution": "uniform",
            "n_envs": N_ENVS_CURRICULUM,
            "draws_per_env": N_EPISODES_PER_ENV,
            "curriculum_scales": list(CURRICULUM_SCALES),
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["curriculum_scale", "value"],
        stats_by_series=stats_by_series,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/curriculum] logged {len(rows)} samples")


# ===================================================================
# Heatmaps — fixed delays (env × episode)
# ===================================================================


def generate_fixed_delay_heatmap(
    hardware: HardwareMeta,
    slug: str,
    pert_cls: type,
    unit: str,
    label: str,
) -> None:
    n_envs = HEATMAP_ENVS_FIXED
    n_episodes = HEATMAP_EPISODES_FIXED
    torch.manual_seed(3100)
    p = _make_delay_pert(pert_cls, n_envs, curriculum_scale=1.0)

    grid = torch.zeros(n_envs, n_episodes)
    rows: list[dict] = []

    for ep in range(n_episodes):
        p.tick(is_reset=True)
        for e in range(n_envs):
            d = p._delay[e].item()
            grid[e, ep] = d
            rows.append({"env_id": e, "episode": ep, "delay_steps": int(d)})

    all_delays = grid.reshape(-1).tolist()
    stats_all = stats_summary(all_delays)
    stats_per_env = {f"env_{e}": stats_summary(grid[e].tolist()) for e in range(n_envs)}
    stats_per_env["all"] = stats_all

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Heatmap(
                z=grid.numpy(),
                x=list(range(n_episodes)),
                y=[f"env {e}" for e in range(n_envs)],
                colorscale=[[0, "#ffffff"], [1, COLOR]],
                colorbar=dict(title=f"delay [{unit}]"),
                hovertemplate=("episode=%{x}<br>env=%{y}<br>delay=%{z} steps<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title=f"Cat 3 — {label}: delay heatmap (env × episode)",
            subtitle=(
                f"curriculum_scale=1.0, n_envs={n_envs}, "
                f"n_episodes={n_episodes}, per_episode resampling"
            ),
            xaxis_title="episode",
            yaxis_title="environment",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name=f"cat3_{slug}_heatmap",
        category=CAT,
        metric=f"{label} per env×episode",
        unit=unit,
        baseline_description="integer delay sampled uniformly at each episode reset",
        config={
            "n_envs": n_envs,
            "n_episodes": n_episodes,
            "curriculum_scale": 1.0,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["env_id", "episode", "delay_steps"],
        stats_by_series=stats_per_env,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/heatmap] logged {len(rows)} rows")


# ===================================================================
# Heatmaps — variable delays (step × env)
# ===================================================================


def generate_variable_delay_heatmap(
    hardware: HardwareMeta,
    slug: str,
    pert_cls: type,
    unit: str,
    label: str,
) -> None:
    n_envs = HEATMAP_ENVS_VARIABLE
    n_steps = HEATMAP_STEPS_VARIABLE
    torch.manual_seed(3200)
    p = _make_delay_pert(pert_cls, n_envs, curriculum_scale=1.0)
    p.tick(is_reset=True)

    grid = torch.zeros(n_steps, n_envs)
    rows: list[dict] = []

    for step in range(n_steps):
        p.tick(is_reset=False)
        for e in range(n_envs):
            d = p._delay[e].item()
            grid[step, e] = d
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "delay_steps": int(d),
                }
            )

    all_delays = grid.reshape(-1).tolist()
    stats_all = stats_summary(all_delays)
    stats_per_env = {f"env_{e}": stats_summary(grid[:, e].tolist()) for e in range(n_envs)}
    stats_per_env["all"] = stats_all

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Heatmap(
                z=grid.numpy().T,
                x=[round(s * DT, 4) for s in range(n_steps)],
                y=[f"env {e}" for e in range(n_envs)],
                colorscale=[[0, "#ffffff"], [1, COLOR]],
                colorbar=dict(title=f"delay [{unit}]"),
                hovertemplate=("time=%{x}s<br>env=%{y}<br>delay=%{z} steps<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title=f"Cat 3 — {label}: delay heatmap (env × step)",
            subtitle=(
                f"curriculum_scale=1.0, n_envs={n_envs}, n_steps={n_steps}, per_step resampling"
            ),
            xaxis_title="time [s]",
            yaxis_title="environment",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name=f"cat3_{slug}_heatmap",
        category=CAT,
        metric=f"{label} per step×env",
        unit=unit,
        baseline_description="integer delay resampled uniformly each step (i.i.d.)",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "delay_steps"],
        stats_by_series=stats_per_env,
        figure_fn=fig_fn,
    )
    print(f"  [{slug}/heatmap] logged {len(rows)} rows")


# ===================================================================
# PacketLoss — drop event heatmap + cumulative rate
# ===================================================================


def _simulate_packet_loss(
    n_envs: int, n_steps: int, curriculum_scale: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    p = PacketLoss(n_envs=n_envs, dt=DT, curriculum_scale=curriculum_scale)
    p.tick(is_reset=True)
    drops = torch.zeros(n_steps, n_envs, dtype=torch.bool)
    probs = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.tick(is_reset=False)
        drops[step] = p._drop_mask.clone()
        probs[step] = p._drop_prob.clone()
    return drops, probs


def generate_packet_loss_heatmap(hardware: HardwareMeta) -> None:
    n_envs, n_steps = EVENT_N_ENVS, EVENT_N_STEPS
    drops, probs = _simulate_packet_loss(n_envs, n_steps, 1.0, seed=3700)

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "dropped": int(drops[step, e].item()),
                    "drop_prob": round(float(probs[step, e].item()), 4),
                }
            )

    drop_rates = [drops[:, e].float().mean().item() for e in range(n_envs)]
    stats_rate = stats_summary(drop_rates)

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        z = drops.float().numpy().T
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[round(s * DT, 4) for s in range(n_steps)],
                y=[f"env {e}" for e in range(n_envs)],
                colorscale=[[0, "#ffffff"], [1, "#d62728"]],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=("time=%{x}s<br>env=%{y}<br>dropped=%{z:.0f}<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title="Cat 3 — packet_loss: drop event heatmap",
            subtitle=(
                f"red=dropped, white=passed; n_envs={n_envs}, "
                f"n_steps={n_steps}, curriculum_scale=1.0"
            ),
            xaxis_title="time [s]",
            yaxis_title="environment",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name="cat3_packet_loss_heatmap",
        category=CAT,
        metric="drop events (binary)",
        unit="boolean",
        baseline_description="1=dropped (ZOH active), 0=passed",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "dropped", "drop_prob"],
        stats_by_series={"drop_rate_per_env": stats_rate},
        figure_fn=fig_fn,
    )
    print(f"  [packet_loss/heatmap] logged {len(rows)} rows")


def generate_packet_loss_cumulative(hardware: HardwareMeta) -> None:
    n_envs, n_steps = EVENT_N_ENVS, EVENT_N_STEPS
    drops, _ = _simulate_packet_loss(n_envs, n_steps, 1.0, seed=3700)

    cumsum = drops.float().cumsum(dim=0)
    steps_range = torch.arange(1, n_steps + 1).unsqueeze(1).float()
    cum_rate = (cumsum / steps_range).numpy()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "cumulative_drop_rate": round(float(cum_rate[step, e]), 6),
                }
            )

    final_rates = [float(cum_rate[-1, e]) for e in range(n_envs)]
    stats_final = stats_summary(final_rates)

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=cum_rate[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.3),
                )
            )
        apply_layout(
            fig,
            title="Cat 3 — packet_loss: cumulative drop rate",
            subtitle=(
                f"running average of drop events; n_envs={n_envs}, "
                f"n_steps={n_steps}, curriculum_scale=1.0"
            ),
            xaxis_title="time [s]",
            yaxis_title="cumulative drop rate (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat3_packet_loss_cumulative_rate",
        category=CAT,
        metric="cumulative drop rate",
        unit="dimensionless",
        baseline_description="running mean of Bernoulli drops; converges to p_drop",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "cumulative_drop_rate"],
        stats_by_series={"final_drop_rate": stats_final},
        figure_fn=fig_fn,
    )
    print(f"  [packet_loss/cumulative] logged {len(rows)} rows")


# ===================================================================
# ComputationOverload — stall heatmap + duration histogram
# ===================================================================


def _simulate_overload(
    n_envs: int, n_steps: int, curriculum_scale: float, seed: int
) -> tuple[torch.Tensor, list[int]]:
    """Returns (stall_grid[n_steps, n_envs], stall_durations)."""
    torch.manual_seed(seed)
    p = ComputationOverload(n_envs=n_envs, dt=DT, curriculum_scale=curriculum_scale)
    p.tick(is_reset=True)

    stall = torch.zeros(n_steps, n_envs, dtype=torch.bool)
    prev_counter = torch.zeros(n_envs, dtype=torch.long)
    durations: list[int] = []

    for step in range(n_steps):
        p.tick(is_reset=False)
        counter = p._skip_counter.clone()
        stall[step] = counter > 0

        for e in range(n_envs):
            if prev_counter[e] == 0 and counter[e] > 0:
                durations.append(int(counter[e].item()))

        prev_counter = counter

    return stall, durations


def generate_overload_heatmap(hardware: HardwareMeta) -> None:
    n_envs, n_steps = EVENT_N_ENVS, EVENT_N_STEPS
    stall, _ = _simulate_overload(n_envs, n_steps, 1.0, seed=3800)

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "stalled": int(stall[step, e].item()),
                }
            )

    stall_rates = [stall[:, e].float().mean().item() for e in range(n_envs)]
    stats_rate = stats_summary(stall_rates)

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        z = stall.float().numpy().T
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[round(s * DT, 4) for s in range(n_steps)],
                y=[f"env {e}" for e in range(n_envs)],
                colorscale=[[0, "#ffffff"], [1, "#ff7f0e"]],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=("time=%{x}s<br>env=%{y}<br>stalled=%{z:.0f}<extra></extra>"),
            )
        )
        apply_layout(
            fig,
            title="Cat 3 — computation_overload: stall event heatmap",
            subtitle=(
                f"orange=stalled, white=normal; n_envs={n_envs}, "
                f"n_steps={n_steps}, curriculum_scale=1.0"
            ),
            xaxis_title="time [s]",
            yaxis_title="environment",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(type="category")
        return fig

    log_and_plot(
        name="cat3_computation_overload_heatmap",
        category=CAT,
        metric="stall events (binary)",
        unit="boolean",
        baseline_description="1=stalled (ZOH active), 0=normal",
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "stalled"],
        stats_by_series={"stall_rate_per_env": stats_rate},
        figure_fn=fig_fn,
    )
    print(f"  [computation_overload/heatmap] logged {len(rows)} rows")


def generate_overload_duration_histogram(hardware: HardwareMeta) -> None:
    n_envs, n_steps = EVENT_N_ENVS, EVENT_N_STEPS
    _, durations = _simulate_overload(n_envs, n_steps, 1.0, seed=3800)

    rows: list[dict] = [{"duration_steps": d} for d in durations]

    if not durations:
        print("  [computation_overload/duration_hist] no stall events — skipping")
        return

    stats_dur = stats_summary([float(d) for d in durations])

    def fig_fn(hw: HardwareMeta) -> go.Figure:
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
        n_dur_values = int(edges[-1] - edges[0])
        expected = len(durations) / max(n_dur_values, 1)
        fig.add_hline(
            y=expected,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"expected (uniform) ≈ {expected:.1f}",
        )
        apply_layout(
            fig,
            title="Cat 3 — computation_overload: stall duration histogram",
            subtitle=(
                f"n_events={len(durations)}, n_envs={n_envs}, "
                f"n_steps={n_steps}, duration ∈ [1, 5] steps"
            ),
            xaxis_title="stall duration [steps]",
            yaxis_title="count",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat3_computation_overload_duration_hist",
        category=CAT,
        metric="stall duration",
        unit="steps",
        baseline_description=("stall duration sampled uniformly ∈ [duration_low, duration_high]"),
        config={
            "n_envs": n_envs,
            "n_steps": n_steps,
            "curriculum_scale": 1.0,
            "distribution_params": {
                "prob_low": 0.0,
                "prob_high": 0.1,
                "duration_low": 1,
                "duration_high": 5,
            },
            "dt_s": DT,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["duration_steps"],
        stats_by_series={"duration": stats_dur},
        figure_fn=fig_fn,
    )
    print(f"  [computation_overload/duration_hist] logged {len(durations)} events")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    delay_specs = [
        ("obs_fixed_delay", ObsFixedDelay, "steps", "obs delay (fixed)"),
        ("obs_variable_delay", ObsVariableDelay, "steps", "obs delay (variable)"),
        ("action_fixed_delay", ActionFixedDelay, "steps", "action delay (fixed)"),
        (
            "action_variable_delay",
            ActionVariableDelay,
            "steps",
            "action delay (variable)",
        ),
    ]

    for slug, cls, unit, label in delay_specs:
        print(f"\n--- {slug} ---")
        generate_curriculum_violin(hardware, slug, cls, unit, label)
        if "fixed" in slug:
            generate_fixed_delay_heatmap(hardware, slug, cls, unit, label)
        else:
            generate_variable_delay_heatmap(hardware, slug, cls, unit, label)

    print("\n--- packet_loss ---")
    generate_curriculum_violin(
        hardware, "packet_loss", PacketLoss, "probability", "drop probability"
    )
    generate_packet_loss_heatmap(hardware)
    generate_packet_loss_cumulative(hardware)

    print("\n--- computation_overload ---")
    generate_curriculum_violin(
        hardware,
        "computation_overload",
        ComputationOverload,
        "probability",
        "skip probability",
    )
    generate_overload_heatmap(hardware)
    generate_overload_duration_histogram(hardware)

    print("\nDone. Inspect docs/impl/{data,assets}/cat3_*.{csv,meta.json,png}")


if __name__ == "__main__":
    main()
