"""Pilot plot generator — Cat 1 ``mass_shift``.

Validates the shared framework (CSV + meta JSON + unified Plotly template)
before rolling it out to the remaining 68 perturbations.

Produces three figures under ``docs/impl/assets/`` and their paired raw data
under ``docs/impl/data/``:

  * ``cat1_mass_shift_curriculum.{png,csv,meta.json}``
      distribution of sampled Δm across curriculum_scale ∈ {0, 0.25, 0.5, 0.75, 1.0}
  * ``cat1_mass_shift_per_env.{png,csv,meta.json}``
      per-env Δm over N_EPISODES, showing inter-env diversity
  * ``cat1_mass_shift_perf.{png,csv,meta.json}``
      overhead (%) vs n_envs on real Genesis CF2X, median + IQR, 5 rounds × 100 steps

Run:
    uv run python docs/impl/plot_pilot_mass_shift.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import torch

# Allow running as a script: `python docs/impl/plot_pilot_mass_shift.py`
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

from genesis_robust_rl.perturbations.category_1_physics import MassShift  # noqa: E402

CAT = 1
SLUG = "cat1_mass_shift"
METRIC_UNIT = "kg"

# Shared perturbation config (matches defaults in MassShift)
DIST_PARAMS = {"low": -0.05, "high": 0.1}
BOUNDS = (-0.5, 1.0)
NOMINAL = 0.0

# ---------------------------------------------------------------------------
# 1. Curriculum plot — distribution of sampled Δm per curriculum_scale
# ---------------------------------------------------------------------------

CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
CURRICULUM_N_ENVS = 256
CURRICULUM_N_DRAWS = 40  # draws per env → 256 × 40 = 10 240 samples per scale


def _sample_curriculum(scale: float, n_envs: int, n_draws: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    p = MassShift(
        setter_fn=lambda v, idx: None,
        n_envs=n_envs,
        dt=0.005,
        distribution_params=DIST_PARAMS,
        bounds=BOUNDS,
        nominal=NOMINAL,
        curriculum_scale=scale,
    )
    out = torch.empty(n_draws, n_envs)
    for i in range(n_draws):
        out[i] = p.sample()
    return out.reshape(-1)  # (n_draws * n_envs,)


def generate_curriculum(hardware: HardwareMeta) -> None:
    samples_by_scale: dict[str, torch.Tensor] = {}
    for idx, s in enumerate(CURRICULUM_SCALES):
        samples_by_scale[f"{s:.2f}"] = _sample_curriculum(
            s, CURRICULUM_N_ENVS, CURRICULUM_N_DRAWS, seed=1000 + idx
        )

    rows: list[dict] = []
    for scale_s, samples in samples_by_scale.items():
        for v in samples.tolist():
            rows.append({"curriculum_scale": float(scale_s), "delta_mass_kg": v})

    stats_by_series = {s: stats_summary(v.tolist()) for s, v in samples_by_scale.items()}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for scale_s, samples in samples_by_scale.items():
            fig.add_trace(
                go.Violin(
                    y=samples.tolist(),
                    x=[scale_s] * len(samples),
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
        # Nominal reference line (inside the data-adapted range)
        fig.add_hline(
            y=NOMINAL,
            line_dash="dash",
            line_color="#888",
            line_width=1,
            annotation_text=f"nominal = {NOMINAL} kg",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#666"),
        )
        # Distribution bounds (low/high of the uniform draw)
        fig.add_hline(
            y=DIST_PARAMS["low"],
            line_dash="dot",
            line_color="#2ca02c",
            line_width=1,
            annotation_text=f"draw low = {DIST_PARAMS['low']} kg",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#2ca02c"),
        )
        fig.add_hline(
            y=DIST_PARAMS["high"],
            line_dash="dot",
            line_color="#2ca02c",
            line_width=1,
            annotation_text=f"draw high = {DIST_PARAMS['high']} kg",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#2ca02c"),
        )
        # Auto-zoom to the sampled range with margin
        all_samples = torch.cat(list(samples_by_scale.values()))
        y_min = float(all_samples.min())
        y_max = float(all_samples.max())
        y_pad = 0.2 * (y_max - y_min + 1e-6)
        apply_layout(
            fig,
            title="Cat 1 — mass_shift: sampled Δm vs curriculum_scale",
            subtitle=(
                f"distribution=uniform [{DIST_PARAMS['low']}, {DIST_PARAMS['high']}] kg, "
                f"hard clip [{BOUNDS[0]}, {BOUNDS[1]}] kg — "
                f"n_envs={CURRICULUM_N_ENVS}, draws/env={CURRICULUM_N_DRAWS}"
            ),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title="sampled Δm [kg]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad])
        return fig

    log_and_plot(
        name=f"{SLUG}_curriculum",
        category=CAT,
        metric="sampled Δm",
        unit=METRIC_UNIT,
        baseline_description="curriculum_scale = 0.0 collapses all samples to nominal",
        config={
            "distribution": "uniform",
            "distribution_params": DIST_PARAMS,
            "bounds": list(BOUNDS),
            "nominal": NOMINAL,
            "n_envs": CURRICULUM_N_ENVS,
            "n_draws_per_env": CURRICULUM_N_DRAWS,
            "curriculum_scales": list(CURRICULUM_SCALES),
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["curriculum_scale", "delta_mass_kg"],
        stats_by_series=stats_by_series,
        figure_fn=fig_fn,
    )
    print(f"[curriculum] logged {len(rows)} samples across {len(CURRICULUM_SCALES)} scales")


# ---------------------------------------------------------------------------
# 2. Per-env plot — Δm per env across N_EPISODES
# ---------------------------------------------------------------------------

PER_ENV_N_ENVS = 16
PER_ENV_N_EPISODES = 30


def generate_per_env(hardware: HardwareMeta) -> None:
    torch.manual_seed(2000)
    p = MassShift(
        setter_fn=lambda v, idx: None,
        n_envs=PER_ENV_N_ENVS,
        dt=0.005,
        distribution_params=DIST_PARAMS,
        bounds=BOUNDS,
        nominal=NOMINAL,
        curriculum_scale=1.0,
    )
    samples = torch.empty(PER_ENV_N_EPISODES, PER_ENV_N_ENVS)
    for ep in range(PER_ENV_N_EPISODES):
        samples[ep] = p.sample()

    rows: list[dict] = []
    for ep in range(PER_ENV_N_EPISODES):
        for e in range(PER_ENV_N_ENVS):
            rows.append({"episode": ep, "env_id": e, "delta_mass_kg": float(samples[ep, e])})

    stats_per_env = {
        f"env_{e}": stats_summary(samples[:, e].tolist()) for e in range(PER_ENV_N_ENVS)
    }
    stats_per_env["all"] = stats_summary(samples.reshape(-1).tolist())

    def fig_fn(hw: HardwareMeta) -> go.Figure:
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
        mean_series = samples.mean(dim=1).tolist()
        fig.add_trace(
            go.Scatter(
                x=list(range(PER_ENV_N_EPISODES)),
                y=mean_series,
                mode="lines",
                name="mean across envs",
                line=dict(width=3, color="black"),
            )
        )
        apply_layout(
            fig,
            title="Cat 1 — mass_shift: per-env Δm across episodes",
            subtitle=(
                f"n_envs={PER_ENV_N_ENVS}, n_episodes={PER_ENV_N_EPISODES}, "
                f"scope=per_env, curriculum=1.0"
            ),
            xaxis_title="episode index",
            yaxis_title="sampled Δm [kg]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"{SLUG}_per_env",
        category=CAT,
        metric="sampled Δm",
        unit=METRIC_UNIT,
        baseline_description="one sample per (episode, env) — illustrates DR inter-env diversity",
        config={
            "distribution_params": DIST_PARAMS,
            "bounds": list(BOUNDS),
            "nominal": NOMINAL,
            "n_envs": PER_ENV_N_ENVS,
            "n_episodes": PER_ENV_N_EPISODES,
            "curriculum_scale": 1.0,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=["episode", "env_id", "delta_mass_kg"],
        stats_by_series=stats_per_env,
        figure_fn=fig_fn,
    )
    print(
        f"[per_env] logged {len(rows)} samples "
        f"({PER_ENV_N_ENVS} envs × {PER_ENV_N_EPISODES} episodes)"
    )


# ---------------------------------------------------------------------------
# 3. Perf plot — overhead (%) vs n_envs on real Genesis CF2X
# ---------------------------------------------------------------------------

PERF_N_ENVS = (1, 4, 16, 64, 128)
PERF_WARMUP = 30
PERF_ROUNDS = 5
PERF_STEPS = 100
PERF_RESET_EVERY = 20


def _measure_loop_times(scene, loop_fn, *, warmup: int, rounds: int, steps: int) -> list[float]:
    for i in range(warmup):
        loop_fn()
        if (i + 1) % PERF_RESET_EVERY == 0:
            scene.reset()
    per_step_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for i in range(steps):
            loop_fn()
            if (i + 1) % PERF_RESET_EVERY == 0:
                scene.reset()
        t1 = time.perf_counter()
        per_step_times.append((t1 - t0) / steps)
    return per_step_times


def _build_genesis_scene(n_envs: int):
    import genesis as gs  # imported lazily — heavy

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.005),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            batch_dofs_info=True,
            batch_links_info=True,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(gs.morphs.URDF(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.5)))
    scene.build(n_envs=n_envs)
    return scene, drone


def _make_env_state(n_envs: int):
    from genesis_robust_rl.perturbations.base import EnvState

    return EnvState(
        pos=torch.rand(n_envs, 3).clamp(min=0.3) * 2,
        quat=torch.tensor([[1.0, 0, 0, 0]]).expand(n_envs, 4).clone(),
        vel=torch.randn(n_envs, 3) * 0.5,
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.randn(n_envs, 3) * 0.1,
        rpm=torch.ones(n_envs, 4) * 14000.0,
        dt=0.005,
        step=0,
    )


def generate_perf(hardware: HardwareMeta) -> None:
    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")

    rows: list[dict] = []
    per_n_envs_overheads: dict[int, list[float]] = {}

    for n_envs in PERF_N_ENVS:
        print(f"[perf] building scene n_envs={n_envs} …")
        scene, drone = _build_genesis_scene(n_envs)

        baseline_times = _measure_loop_times(
            scene,
            lambda: scene.step(),
            warmup=PERF_WARMUP,
            rounds=PERF_ROUNDS,
            steps=PERF_STEPS,
        )

        pert = MassShift(
            setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            n_envs=n_envs,
            dt=0.005,
            distribution_params=DIST_PARAMS,
            bounds=BOUNDS,
            nominal=NOMINAL,
        )
        env_state = _make_env_state(n_envs)
        pert.tick(is_reset=True, env_ids=torch.arange(n_envs))

        def loop():
            pert.tick(is_reset=False)
            pert.apply(scene, drone, env_state)
            scene.step()

        perturbed_times = _measure_loop_times(
            scene,
            loop,
            warmup=PERF_WARMUP,
            rounds=PERF_ROUNDS,
            steps=PERF_STEPS,
        )

        overheads = [(tp - tb) / tb * 100.0 for tp, tb in zip(perturbed_times, baseline_times)]
        per_n_envs_overheads[n_envs] = overheads

        for r, (tb, tp, oh) in enumerate(zip(baseline_times, perturbed_times, overheads)):
            rows.append(
                {
                    "n_envs": n_envs,
                    "round": r,
                    "baseline_s_per_step": tb,
                    "perturbed_s_per_step": tp,
                    "overhead_pct": oh,
                }
            )

        print(
            f"[perf]   baseline={statistics.fmean(baseline_times) * 1e6:.1f} µs  "
            f"overhead_median={statistics.median(overheads):+.2f}%"
        )

    stats_by_n = {f"n_envs_{n}": stats_summary(per_n_envs_overheads[n]) for n in PERF_N_ENVS}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        ns = list(PERF_N_ENVS)
        medians = [stats_by_n[f"n_envs_{n}"].median for n in ns]
        q1s = [stats_by_n[f"n_envs_{n}"].q1 for n in ns]
        q3s = [stats_by_n[f"n_envs_{n}"].q3 for n in ns]
        # IQR band
        fig.add_trace(
            go.Scatter(
                x=ns + ns[::-1],
                y=q3s + q1s[::-1],
                fill="toself",
                fillcolor="rgba(31,119,180,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="IQR",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ns,
                y=medians,
                mode="lines+markers",
                name="median",
                line=dict(color=CATEGORY_COLORS[CAT], width=2),
                marker=dict(size=8),
            )
        )
        fig.add_hline(
            y=5.0,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="5% budget",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#d62728"),
        )
        apply_layout(
            fig,
            title="Cat 1 — mass_shift: overhead (%) vs n_envs",
            subtitle=(
                f"Genesis CF2X, warmup={PERF_WARMUP}, {PERF_ROUNDS} rounds × {PERF_STEPS} steps"
                f" — baseline = scene.step() only"
            ),
            xaxis_title="n_envs",
            yaxis_title="overhead [%]",
            xaxis_type="log",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"{SLUG}_perf",
        category=CAT,
        metric="overhead",
        unit="%",
        baseline_description="scene.step() only — no tick/apply call",
        config={
            "n_envs_range": list(PERF_N_ENVS),
            "warmup": PERF_WARMUP,
            "rounds": PERF_ROUNDS,
            "steps_per_round": PERF_STEPS,
            "reset_every": PERF_RESET_EVERY,
            "genesis_backend": "cpu",
            "drone_urdf": "urdf/drones/cf2x.urdf",
            "dt_s": 0.005,
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=[
            "n_envs",
            "round",
            "baseline_s_per_step",
            "perturbed_s_per_step",
            "overhead_pct",
        ],
        stats_by_series=stats_by_n,
        figure_fn=fig_fn,
    )
    print(f"[perf] logged {len(rows)} rows across {len(PERF_N_ENVS)} n_envs settings")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    # Pure-torch plots first (fast, no Genesis init).
    generate_curriculum(hardware)
    generate_per_env(hardware)

    # Perf plot needs Genesis — runs last, long.
    generate_perf(hardware)
    print("\nDone. Inspect:")
    print("  docs/impl/data/cat1_mass_shift_{curriculum,per_env,perf}.{csv,meta.json}")
    print("  docs/impl/assets/cat1_mass_shift_{curriculum,per_env,perf}.png")


if __name__ == "__main__":
    main()
