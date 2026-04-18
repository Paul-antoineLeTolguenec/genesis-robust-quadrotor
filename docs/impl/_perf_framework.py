"""Shared Genesis overhead-measurement framework.

Used by all ``plot_perf_cat<N>.py`` scripts. Enforces the canonical methodology
from ``tests/integration/test_overhead_genesis.py``:

  BASELINE:  scene.step() only
  PERTURBED: pert.tick(is_reset=False) + pert.apply(...) + scene.step()
  overhead (%) = (median_perturbed - median_baseline) / median_baseline * 100

For each perturbation, sweeps n_envs ∈ {1, 4, 16, 64, 128}, rebuilding the Genesis
scene at each n_envs (Genesis cannot change n_envs after build). Writes one
``<slug>_perf.csv`` + ``<slug>_perf.meta.json`` + ``<slug>_perf.png`` per
perturbation via ``docs/impl/_plot_framework.log_and_plot``.

The four apply signatures on Perturbation subclasses are handled by ``PertKind``:

  * ``physics``  →  ``pert.apply(scene, drone, env_state)``
  * ``motor``    →  ``pert.apply(rpm_cmd)``
  * ``obs``      →  ``pert.apply(obs)``
  * ``action``   →  ``pert.apply(action)``

Run from a script that calls ``run_perf_sweep(...)``. Genesis is initialised
exactly once per process.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

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

# ---------------------------------------------------------------------------
# Canonical measurement constants (match test_overhead_genesis.py)
# ---------------------------------------------------------------------------

N_ENVS_RANGE: tuple[int, ...] = (1, 4, 16, 64, 128)
WARMUP = 30
ROUNDS = 5
STEPS_PER_ROUND = 100
RESET_EVERY = 20
DT = 0.005
HOVER_RPM = 14000.0

PertKind = Literal["physics", "motor", "obs", "action"]

_GENESIS_INITIALISED = False


def init_genesis(backend: str = "cpu") -> None:
    """Idempotent Genesis init — may only be called once per process."""
    global _GENESIS_INITIALISED
    if _GENESIS_INITIALISED:
        return
    import genesis as gs

    gs_backend = gs.cpu if backend == "cpu" else gs.cuda
    gs.init(backend=gs_backend, logging_level="warning")
    _GENESIS_INITIALISED = True


def build_scene(n_envs: int) -> tuple[Any, Any]:
    """Build a Genesis scene with a Crazyflie CF2X drone."""
    import genesis as gs

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=DT),
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


def make_env_state(n_envs: int) -> Any:
    """Realistic pre-allocated EnvState for measurements."""
    from genesis_robust_rl.perturbations.base import EnvState

    return EnvState(
        pos=torch.rand(n_envs, 3).clamp(min=0.3) * 2,
        quat=torch.tensor([[1.0, 0, 0, 0]]).expand(n_envs, 4).clone(),
        vel=torch.randn(n_envs, 3) * 0.5,
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.randn(n_envs, 3) * 0.1,
        rpm=torch.ones(n_envs, 4) * HOVER_RPM,
        dt=DT,
        step=0,
    )


# ---------------------------------------------------------------------------
# Perturbation spec
# ---------------------------------------------------------------------------


@dataclass
class PertSpec:
    """One perturbation to measure.

    Attributes:
        slug: filename stem, e.g. ``cat8_body_force_disturbance`` (framework
            appends ``_perf``).
        display_name: human-readable name for the plot title.
        kind: which apply signature to use.
        factory: ``factory(scene, drone, n_envs) -> Perturbation`` — must
            bind any setter lambdas to ``drone`` before returning.
    """

    slug: str
    display_name: str
    kind: PertKind
    factory: Callable[[Any, Any, int], Any]


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------


def _make_loop_for_kind(
    kind: PertKind,
    pert: Any,
    scene: Any,
    drone: Any,
    env_state: Any,
    rpm_cmd: torch.Tensor,
    obs_buf: torch.Tensor,
    action_buf: torch.Tensor,
) -> Callable[[], None]:
    """Build the per-step loop for the given apply signature.

    All four branches call ``pert.tick(is_reset=False)`` first, then ``apply``,
    then ``scene.step()`` — identical pattern to the P6 test.
    """
    if kind == "physics":

        def loop() -> None:
            pert.tick(is_reset=False)
            pert.apply(scene, drone, env_state)
            scene.step()

    elif kind == "motor":

        def loop() -> None:
            pert.tick(is_reset=False)
            pert.apply(rpm_cmd)
            scene.step()

    elif kind == "obs":

        def loop() -> None:
            pert.tick(is_reset=False)
            pert.apply(obs_buf)
            scene.step()

    elif kind == "action":

        def loop() -> None:
            pert.tick(is_reset=False)
            pert.apply(action_buf)
            scene.step()

    else:
        raise ValueError(f"unknown kind: {kind}")

    return loop


def _measure_loop_times(
    scene: Any,
    loop_fn: Callable[[], None],
    *,
    warmup: int,
    rounds: int,
    steps: int,
) -> list[float]:
    """Return per-step wall-clock times (seconds) for ``rounds`` batches of ``steps``."""
    for i in range(warmup):
        loop_fn()
        if (i + 1) % RESET_EVERY == 0:
            scene.reset()
    per_step_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for i in range(steps):
            loop_fn()
            if (i + 1) % RESET_EVERY == 0:
                scene.reset()
        t1 = time.perf_counter()
        per_step_times.append((t1 - t0) / steps)
    return per_step_times


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_perf_sweep(
    *,
    category: int,
    perturbations: list[PertSpec],
    n_envs_range: tuple[int, ...] = N_ENVS_RANGE,
) -> None:
    """Measure overhead for all ``perturbations`` across ``n_envs_range``.

    Pattern per n_envs:
      1. Build a fresh Genesis scene + CF2X drone.
      2. Measure baseline (scene.step() only).
      3. For each perturbation: instantiate via spec.factory, reset, measure.

    After all measurements are collected, emit one set of CSV + meta JSON + PNG
    per perturbation via ``_plot_framework.log_and_plot``.
    """
    init_genesis(backend="cpu")
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    results: dict[str, dict[int, dict[str, list[float]]]] = {
        spec.slug: {} for spec in perturbations
    }

    for n_envs in n_envs_range:
        print(f"\n=== n_envs={n_envs} ===")
        scene, drone = build_scene(n_envs)
        env_state = make_env_state(n_envs)
        rpm_cmd = torch.ones(n_envs, 4) * HOVER_RPM
        # Width 12 covers every obs-slice used by the catalog (max is slice(0, 6)
        # for obs_channel_masking).
        obs_buf = torch.randn(n_envs, 12)
        action_buf = torch.ones(n_envs, 4) * HOVER_RPM

        baseline_times = _measure_loop_times(
            scene,
            scene.step,
            warmup=WARMUP,
            rounds=ROUNDS,
            steps=STEPS_PER_ROUND,
        )
        print(f"  baseline: median={statistics.median(baseline_times) * 1e6:.1f} µs/step")

        for spec in perturbations:
            pert = spec.factory(scene, drone, n_envs)
            pert.tick(is_reset=True, env_ids=torch.arange(n_envs))

            loop = _make_loop_for_kind(
                spec.kind,
                pert,
                scene,
                drone,
                env_state,
                rpm_cmd,
                obs_buf,
                action_buf,
            )

            perturbed_times = _measure_loop_times(
                scene,
                loop,
                warmup=WARMUP,
                rounds=ROUNDS,
                steps=STEPS_PER_ROUND,
            )

            overheads = [(tp - tb) / tb * 100.0 for tp, tb in zip(perturbed_times, baseline_times)]
            results[spec.slug][n_envs] = {
                "baseline": baseline_times,
                "perturbed": perturbed_times,
                "overhead": overheads,
            }
            print(
                f"  {spec.display_name:<34s} "
                f"perturbed={statistics.median(perturbed_times) * 1e6:6.1f} µs  "
                f"overhead_median={statistics.median(overheads):+6.2f}%"
            )

    # ---- Emit plots + CSV + meta per perturbation ----------------------------
    for spec in perturbations:
        _emit_perf_artifacts(
            spec=spec,
            category=category,
            per_n_envs=results[spec.slug],
            n_envs_range=n_envs_range,
            hardware=hardware,
        )


def _emit_perf_artifacts(
    *,
    spec: PertSpec,
    category: int,
    per_n_envs: dict[int, dict[str, list[float]]],
    n_envs_range: tuple[int, ...],
    hardware: HardwareMeta,
) -> None:
    """Write CSV + meta JSON + PNG for one perturbation's perf sweep."""
    rows: list[dict[str, Any]] = []
    for n_envs in n_envs_range:
        data = per_n_envs[n_envs]
        for r, (tb, tp, oh) in enumerate(
            zip(data["baseline"], data["perturbed"], data["overhead"])
        ):
            rows.append(
                {
                    "n_envs": n_envs,
                    "round": r,
                    "baseline_s_per_step": tb,
                    "perturbed_s_per_step": tp,
                    "overhead_pct": oh,
                }
            )

    stats_by_n = {f"n_envs_{n}": stats_summary(per_n_envs[n]["overhead"]) for n in n_envs_range}

    def fig_fn(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        ns = list(n_envs_range)
        medians = [stats_by_n[f"n_envs_{n}"].median for n in ns]
        q1s = [stats_by_n[f"n_envs_{n}"].q1 for n in ns]
        q3s = [stats_by_n[f"n_envs_{n}"].q3 for n in ns]

        color = CATEGORY_COLORS[category]
        fig.add_trace(
            go.Scatter(
                x=ns + ns[::-1],
                y=q3s + q1s[::-1],
                fill="toself",
                fillcolor=_rgba(color, 0.2),
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
                line=dict(color=color, width=2),
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
            title=f"Cat {category} — {spec.display_name}: overhead (%) vs n_envs",
            subtitle=(
                f"Genesis CF2X, warmup={WARMUP}, {ROUNDS} rounds × {STEPS_PER_ROUND} steps"
                " — baseline = scene.step() only"
            ),
            xaxis_title="n_envs",
            yaxis_title="overhead [%]",
            xaxis_type="log",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name=f"{spec.slug}_perf",
        category=category,
        metric="overhead",
        unit="%",
        baseline_description="scene.step() only — no tick/apply call",
        config={
            "n_envs_range": list(n_envs_range),
            "warmup": WARMUP,
            "rounds": ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "reset_every": RESET_EVERY,
            "genesis_backend": "cpu",
            "drone_urdf": "urdf/drones/cf2x.urdf",
            "dt_s": DT,
            "apply_kind": spec.kind,
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
    print(f"  wrote {spec.slug}_perf.{{csv,meta.json,png}}")


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` + alpha to a Plotly ``rgba(...)`` string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
