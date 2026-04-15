"""Category 7 — Payload & configuration plots.

Generates curriculum + per_env PNGs for:
- payload_mass (scalar Δm)
- payload_com_offset (3D Δr, plot L2 magnitude)
- asymmetric_prop_guard_drag (per-arm drag ratios, add bar chart)

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_7.py
"""

from __future__ import annotations

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

from genesis_robust_rl.perturbations.category_7_payload import (  # noqa: E402
    AsymmetricPropGuardDrag,
    PayloadCOMOffset,
    PayloadMass,
)

CAT = 7
DT = 0.01
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
CURRICULUM_N_ENVS = 256
CURRICULUM_N_DRAWS = 40
PER_ENV_N_ENVS = 16
PER_ENV_N_EPISODES = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_scalar(pert_cls, n_envs, n_draws, scale, seed):
    torch.manual_seed(seed)
    p = pert_cls(
        setter_fn=lambda v, idx: None,
        n_envs=n_envs,
        dt=DT,
        curriculum_scale=scale,
    )
    out = torch.empty(n_draws, n_envs)
    for i in range(n_draws):
        out[i] = p.sample()
    return out


def _sample_vector_magnitude(pert_cls, n_envs, n_draws, scale, seed):
    torch.manual_seed(seed)
    p = pert_cls(
        setter_fn=lambda v, idx: None,
        n_envs=n_envs,
        dt=DT,
        curriculum_scale=scale,
    )
    out = torch.empty(n_draws, n_envs, 3)
    for i in range(n_draws):
        out[i] = p.sample()
    return out  # [n_draws, n_envs, 3]


def _sample_ratios(n_envs, n_draws, scale, seed):
    """Per-arm drag ratios (4 arms) for AsymmetricPropGuardDrag."""
    torch.manual_seed(seed)
    p = AsymmetricPropGuardDrag(
        n_envs=n_envs,
        dt=DT,
        curriculum_scale=scale,
    )
    out = torch.empty(n_draws, n_envs, 4)
    for i in range(n_draws):
        out[i] = p.sample()
    return out


# ---------------------------------------------------------------------------
# payload_mass — scalar curriculum + per_env
# ---------------------------------------------------------------------------


def generate_payload_mass(hardware: HardwareMeta) -> None:
    samples_by_scale: dict[str, list[float]] = {}
    rows_curr: list[dict] = []
    for i, scale in enumerate(CURRICULUM_SCALES):
        s = _sample_scalar(PayloadMass, CURRICULUM_N_ENVS, CURRICULUM_N_DRAWS, scale, 7100 + i)
        flat = s.reshape(-1).tolist()
        samples_by_scale[f"{scale:.2f}"] = flat
        for v in flat:
            rows_curr.append({"curriculum_scale": float(scale), "payload_mass_kg": v})

    stats_curr = {k: stats_summary(v) for k, v in samples_by_scale.items()}

    def fig_curriculum(hw: HardwareMeta) -> go.Figure:
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
            title="Cat 7 — payload_mass: sampled Δm vs curriculum_scale",
            subtitle=(
                f"distribution=uniform [0.0, 0.5] kg, "
                f"n_envs={CURRICULUM_N_ENVS}, draws/env={CURRICULUM_N_DRAWS}"
            ),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title="payload Δm [kg]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[max(0, y_min - pad), y_max + pad])
        return fig

    log_and_plot(
        name="cat7_payload_mass_curriculum",
        category=CAT,
        metric="payload mass shift",
        unit="kg",
        baseline_description="curriculum=0 collapses samples to nominal 0.0 kg",
        config={
            "distribution": "uniform",
            "bounds_kg": [0.0, 0.5],
            "n_envs": CURRICULUM_N_ENVS,
            "draws_per_env": CURRICULUM_N_DRAWS,
        },
        hardware=hardware,
        csv_rows=rows_curr,
        csv_columns=["curriculum_scale", "payload_mass_kg"],
        stats_by_series=stats_curr,
        figure_fn=fig_curriculum,
    )
    print(f"[payload_mass/curriculum] logged {len(rows_curr)} samples")

    # per_env
    torch.manual_seed(7200)
    s_pe = _sample_scalar(PayloadMass, PER_ENV_N_ENVS, PER_ENV_N_EPISODES, 1.0, 7200)
    rows_pe: list[dict] = []
    for ep in range(PER_ENV_N_EPISODES):
        for e in range(PER_ENV_N_ENVS):
            rows_pe.append({"episode": ep, "env_id": e, "payload_mass_kg": float(s_pe[ep, e])})
    stats_pe = {f"env_{e}": stats_summary(s_pe[:, e].tolist()) for e in range(PER_ENV_N_ENVS)}
    stats_pe["all"] = stats_summary(s_pe.reshape(-1).tolist())

    def fig_per_env(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for e in range(PER_ENV_N_ENVS):
            fig.add_trace(
                go.Scatter(
                    x=list(range(PER_ENV_N_EPISODES)),
                    y=s_pe[:, e].tolist(),
                    mode="lines+markers",
                    name=f"env {e}",
                    line=dict(width=1),
                    marker=dict(size=4),
                    opacity=0.6,
                    showlegend=False,
                )
            )
        mean_series = s_pe.mean(dim=1).tolist()
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
            title="Cat 7 — payload_mass: per-env Δm across episodes",
            subtitle=(f"n_envs={PER_ENV_N_ENVS}, n_episodes={PER_ENV_N_EPISODES}, curriculum=1.0"),
            xaxis_title="episode index",
            yaxis_title="payload Δm [kg]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat7_payload_mass_per_env",
        category=CAT,
        metric="payload mass shift",
        unit="kg",
        baseline_description="one sample per (episode, env)",
        config={"n_envs": PER_ENV_N_ENVS, "n_episodes": PER_ENV_N_EPISODES},
        hardware=hardware,
        csv_rows=rows_pe,
        csv_columns=["episode", "env_id", "payload_mass_kg"],
        stats_by_series=stats_pe,
        figure_fn=fig_per_env,
    )
    print(f"[payload_mass/per_env] logged {len(rows_pe)} samples")


# ---------------------------------------------------------------------------
# payload_com_offset — L2 magnitude
# ---------------------------------------------------------------------------


def generate_payload_com_offset(hardware: HardwareMeta) -> None:
    samples_by_scale: dict[str, list[float]] = {}
    rows_curr: list[dict] = []
    for i, scale in enumerate(CURRICULUM_SCALES):
        v = _sample_vector_magnitude(
            PayloadCOMOffset, CURRICULUM_N_ENVS, CURRICULUM_N_DRAWS, scale, 7300 + i
        )  # [n_draws, n_envs, 3]
        mag = v.norm(dim=-1)  # [n_draws, n_envs]
        flat = mag.reshape(-1).tolist()
        samples_by_scale[f"{scale:.2f}"] = flat
        # Log each axis in CSV for reproducibility
        for d in range(v.shape[0]):
            for e in range(v.shape[1]):
                rows_curr.append(
                    {
                        "curriculum_scale": float(scale),
                        "draw": d,
                        "env_id": e,
                        "offset_x_m": float(v[d, e, 0]),
                        "offset_y_m": float(v[d, e, 1]),
                        "offset_z_m": float(v[d, e, 2]),
                        "offset_magnitude_m": float(mag[d, e]),
                    }
                )

    stats_curr = {k: stats_summary(v) for k, v in samples_by_scale.items()}

    def fig_curriculum(hw: HardwareMeta) -> go.Figure:
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
            title="Cat 7 — payload_com_offset: ‖Δr‖₂ vs curriculum_scale",
            subtitle=(
                f"per-axis uniform [-0.1, +0.1] m, "
                f"n_envs={CURRICULUM_N_ENVS}, draws/env={CURRICULUM_N_DRAWS}"
            ),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title="‖Δr‖₂ [m]",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[max(0, y_min - pad), y_max + pad])
        return fig

    log_and_plot(
        name="cat7_payload_com_offset_curriculum",
        category=CAT,
        metric="CoM offset magnitude",
        unit="m",
        baseline_description="curriculum=0 collapses samples to nominal (0,0,0)",
        config={
            "distribution": "uniform",
            "per_axis_bounds_m": [-0.1, 0.1],
            "n_envs": CURRICULUM_N_ENVS,
            "draws_per_env": CURRICULUM_N_DRAWS,
        },
        hardware=hardware,
        csv_rows=rows_curr,
        csv_columns=[
            "curriculum_scale",
            "draw",
            "env_id",
            "offset_x_m",
            "offset_y_m",
            "offset_z_m",
            "offset_magnitude_m",
        ],
        stats_by_series=stats_curr,
        figure_fn=fig_curriculum,
    )
    print(f"[payload_com_offset/curriculum] logged {len(rows_curr)} samples")

    # per_env
    v_pe = _sample_vector_magnitude(PayloadCOMOffset, PER_ENV_N_ENVS, PER_ENV_N_EPISODES, 1.0, 7400)
    mag_pe = v_pe.norm(dim=-1)  # [n_episodes, n_envs]
    rows_pe: list[dict] = []
    for ep in range(PER_ENV_N_EPISODES):
        for e in range(PER_ENV_N_ENVS):
            rows_pe.append(
                {
                    "episode": ep,
                    "env_id": e,
                    "offset_x_m": float(v_pe[ep, e, 0]),
                    "offset_y_m": float(v_pe[ep, e, 1]),
                    "offset_z_m": float(v_pe[ep, e, 2]),
                    "offset_magnitude_m": float(mag_pe[ep, e]),
                }
            )
    stats_pe = {f"env_{e}": stats_summary(mag_pe[:, e].tolist()) for e in range(PER_ENV_N_ENVS)}
    stats_pe["all"] = stats_summary(mag_pe.reshape(-1).tolist())

    def fig_per_env(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for e in range(PER_ENV_N_ENVS):
            fig.add_trace(
                go.Scatter(
                    x=list(range(PER_ENV_N_EPISODES)),
                    y=mag_pe[:, e].tolist(),
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
                y=mag_pe.mean(dim=1).tolist(),
                mode="lines",
                name="mean across envs",
                line=dict(width=3, color="black"),
            )
        )
        apply_layout(
            fig,
            title="Cat 7 — payload_com_offset: per-env ‖Δr‖₂ across episodes",
            subtitle=(f"n_envs={PER_ENV_N_ENVS}, n_episodes={PER_ENV_N_EPISODES}, curriculum=1.0"),
            xaxis_title="episode index",
            yaxis_title="‖Δr‖₂ [m]",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat7_payload_com_offset_per_env",
        category=CAT,
        metric="CoM offset magnitude",
        unit="m",
        baseline_description="one 3D sample per (episode, env); plot L2 magnitude",
        config={"n_envs": PER_ENV_N_ENVS, "n_episodes": PER_ENV_N_EPISODES},
        hardware=hardware,
        csv_rows=rows_pe,
        csv_columns=[
            "episode",
            "env_id",
            "offset_x_m",
            "offset_y_m",
            "offset_z_m",
            "offset_magnitude_m",
        ],
        stats_by_series=stats_pe,
        figure_fn=fig_per_env,
    )
    print(f"[payload_com_offset/per_env] logged {len(rows_pe)} samples")


# ---------------------------------------------------------------------------
# asymmetric_prop_guard_drag — curriculum (ratio magnitude) + per_env + bar chart
# ---------------------------------------------------------------------------


def generate_asymmetric_prop_guard_drag(hardware: HardwareMeta) -> None:
    # curriculum: distribution of per-env mean ratio (max|ratio - 1|) vs scale
    samples_by_scale: dict[str, list[float]] = {}
    rows_curr: list[dict] = []
    for i, scale in enumerate(CURRICULUM_SCALES):
        r = _sample_ratios(CURRICULUM_N_ENVS, CURRICULUM_N_DRAWS, scale, 7500 + i)
        # Deviation from nominal 1.0: L∞ across 4 arms
        dev = (r - 1.0).abs().max(dim=-1).values  # [n_draws, n_envs]
        flat = dev.reshape(-1).tolist()
        samples_by_scale[f"{scale:.2f}"] = flat
        for d in range(r.shape[0]):
            for e in range(r.shape[1]):
                rows_curr.append(
                    {
                        "curriculum_scale": float(scale),
                        "draw": d,
                        "env_id": e,
                        "ratio_arm_0": float(r[d, e, 0]),
                        "ratio_arm_1": float(r[d, e, 1]),
                        "ratio_arm_2": float(r[d, e, 2]),
                        "ratio_arm_3": float(r[d, e, 3]),
                        "max_deviation_from_nominal": float(dev[d, e]),
                    }
                )

    stats_curr = {k: stats_summary(v) for k, v in samples_by_scale.items()}

    def fig_curriculum(hw: HardwareMeta) -> go.Figure:
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
            title="Cat 7 — asymmetric_prop_guard_drag: max |ratio−1| vs curriculum_scale",
            subtitle=(
                f"per-arm drag ratio uniform [0.8, 1.2], "
                f"n_envs={CURRICULUM_N_ENVS}, draws/env={CURRICULUM_N_DRAWS}"
            ),
            xaxis_title="curriculum_scale (dimensionless)",
            yaxis_title="max |ratio − 1| across 4 arms (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        fig.update_yaxes(range=[max(0, y_min - pad), y_max + pad])
        return fig

    log_and_plot(
        name="cat7_asymmetric_prop_guard_drag_curriculum",
        category=CAT,
        metric="max arm drag deviation",
        unit="dimensionless",
        baseline_description="curriculum=0 collapses to nominal ratios = 1",
        config={
            "distribution": "uniform",
            "bounds": [0.8, 1.2],
            "n_envs": CURRICULUM_N_ENVS,
            "draws_per_env": CURRICULUM_N_DRAWS,
        },
        hardware=hardware,
        csv_rows=rows_curr,
        csv_columns=[
            "curriculum_scale",
            "draw",
            "env_id",
            "ratio_arm_0",
            "ratio_arm_1",
            "ratio_arm_2",
            "ratio_arm_3",
            "max_deviation_from_nominal",
        ],
        stats_by_series=stats_curr,
        figure_fn=fig_curriculum,
    )
    print(f"[asym_drag/curriculum] logged {len(rows_curr)} samples")

    # per_env
    r_pe = _sample_ratios(PER_ENV_N_ENVS, PER_ENV_N_EPISODES, 1.0, 7600)
    dev_pe = (r_pe - 1.0).abs().max(dim=-1).values  # [n_episodes, n_envs]
    rows_pe: list[dict] = []
    for ep in range(PER_ENV_N_EPISODES):
        for e in range(PER_ENV_N_ENVS):
            rows_pe.append(
                {
                    "episode": ep,
                    "env_id": e,
                    "ratio_arm_0": float(r_pe[ep, e, 0]),
                    "ratio_arm_1": float(r_pe[ep, e, 1]),
                    "ratio_arm_2": float(r_pe[ep, e, 2]),
                    "ratio_arm_3": float(r_pe[ep, e, 3]),
                    "max_deviation_from_nominal": float(dev_pe[ep, e]),
                }
            )
    stats_pe = {f"env_{e}": stats_summary(dev_pe[:, e].tolist()) for e in range(PER_ENV_N_ENVS)}
    stats_pe["all"] = stats_summary(dev_pe.reshape(-1).tolist())

    def fig_per_env(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for e in range(PER_ENV_N_ENVS):
            fig.add_trace(
                go.Scatter(
                    x=list(range(PER_ENV_N_EPISODES)),
                    y=dev_pe[:, e].tolist(),
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
                y=dev_pe.mean(dim=1).tolist(),
                mode="lines",
                name="mean across envs",
                line=dict(width=3, color="black"),
            )
        )
        apply_layout(
            fig,
            title="Cat 7 — asymmetric_prop_guard_drag: per-env max |ratio−1|",
            subtitle=(f"n_envs={PER_ENV_N_ENVS}, n_episodes={PER_ENV_N_EPISODES}, curriculum=1.0"),
            xaxis_title="episode index",
            yaxis_title="max |ratio − 1| (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat7_asymmetric_prop_guard_drag_per_env",
        category=CAT,
        metric="max arm drag deviation",
        unit="dimensionless",
        baseline_description="one sample of 4-arm ratios per (episode, env)",
        config={"n_envs": PER_ENV_N_ENVS, "n_episodes": PER_ENV_N_EPISODES},
        hardware=hardware,
        csv_rows=rows_pe,
        csv_columns=[
            "episode",
            "env_id",
            "ratio_arm_0",
            "ratio_arm_1",
            "ratio_arm_2",
            "ratio_arm_3",
            "max_deviation_from_nominal",
        ],
        stats_by_series=stats_pe,
        figure_fn=fig_per_env,
    )
    print(f"[asym_drag/per_env] logged {len(rows_pe)} samples")

    # bar chart: per-arm mean ± σ across 100 envs
    r_bar = _sample_ratios(100, 1, 1.0, 7700)[0]  # [100, 4]
    means = r_bar.mean(dim=0).tolist()  # [4]
    stds = r_bar.std(dim=0).tolist()
    rows_bar: list[dict] = []
    for arm in range(4):
        rows_bar.append({"arm_id": arm, "mean_ratio": means[arm], "std_ratio": stds[arm]})
    stats_bar = {f"arm_{arm}": stats_summary(r_bar[:, arm].tolist()) for arm in range(4)}

    def fig_bar(hw: HardwareMeta) -> go.Figure:
        fig = make_figure()
        fig.add_trace(
            go.Bar(
                x=[0, 1, 2, 3],
                y=means,
                error_y=dict(type="data", array=stds, visible=True, thickness=2),
                marker_color=CATEGORY_COLORS[CAT],
                name="mean ratio ± σ",
                width=0.6,
            )
        )
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#888",
            annotation_text="nominal = 1.0",
            annotation_position="top right",
        )
        apply_layout(
            fig,
            title="Cat 7 — asymmetric_prop_guard_drag: per-arm drag ratio",
            subtitle="n_samples=100, distribution=uniform [0.8, 1.2], curriculum=1.0",
            xaxis_title="arm index",
            yaxis_title="drag ratio r_j (dimensionless)",
            hardware_footer=hw.footer_line(),
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["arm 0", "arm 1", "arm 2", "arm 3"],
        )
        fig.update_yaxes(range=[0.7, 1.3])
        return fig

    log_and_plot(
        name="cat7_asymmetric_prop_guard_drag_bar",
        category=CAT,
        metric="per-arm drag ratio",
        unit="dimensionless",
        baseline_description="mean ± σ across 100 envs",
        config={"n_envs_sampled": 100, "n_arms": 4},
        hardware=hardware,
        csv_rows=rows_bar,
        csv_columns=["arm_id", "mean_ratio", "std_ratio"],
        stats_by_series=stats_bar,
        figure_fn=fig_bar,
    )
    print(f"[asym_drag/bar] logged {len(rows_bar)} arms")


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
    generate_payload_mass(hardware)
    generate_payload_com_offset(hardware)
    generate_asymmetric_prop_guard_drag(hardware)
    print("\nDone. Inspect docs/impl/{data,assets}/cat7_*.{csv,meta.json,png}")


if __name__ == "__main__":
    if os.getenv("RUN_GENESIS") == "1":
        run_perf()
    main()
