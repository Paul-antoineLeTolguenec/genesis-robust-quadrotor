"""Category 1 — Physics perturbation plots.

Generates curriculum violins + specialised PNGs for 14 perturbations
(1.2–1.15; mass_shift 1.1 is covered by plot_pilot_mass_shift.py).

- 1.2  com_shift                 (curriculum violin, L2 magnitude)
- 1.3  inertia_tensor            (curriculum violin, axis 0)
- 1.4  motor_armature            (curriculum violin)
- 1.5  friction_ratio            (curriculum violin)
- 1.6  position_gain_kp          (curriculum violin)
- 1.7  velocity_gain_kv          (curriculum violin)
- 1.8  joint_stiffness           (curriculum violin)
- 1.9  joint_damping             (curriculum violin)
- 1.10 aero_drag_coeff           (curriculum violin, axis 0)
- 1.11 ground_effect             (altitude sweep)
- 1.12 chassis_geometry_asymmetry (curriculum violin, arm 0)
- 1.13 propeller_blade_damage    (curriculum violin, prop 0)
- 1.14 structural_flexibility    (curriculum violin, k)
- 1.15 battery_voltage_sag       (curriculum violin + SoC trace)

CSV + meta JSON logged BEFORE each PNG via the shared framework.

Run:
    uv run python docs/impl/plot_category_1.py
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

from genesis_robust_rl.perturbations.base import EnvState  # noqa: E402
from genesis_robust_rl.perturbations.category_1_physics import (  # noqa: E402
    AeroDragCoeff,
    BatteryVoltageSag,
    ChassisGeometryAsymmetry,
    COMShift,
    FrictionRatio,
    GroundEffect,
    InertiaTensor,
    JointDamping,
    JointStiffness,
    MotorArmature,
    PositionGainKp,
    PropellerBladeDamage,
    StructuralFlexibility,
    VelocityGainKv,
)

CAT = 1
DT = 0.01
COLOR = CATEGORY_COLORS[CAT]
N_ENVS = 256
N_DRAWS = 40
CURRICULUM_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)

# Dummy setter for GenesisSetterPerturbation (plots don't call apply)
_NOOP_SETTER = lambda val, idx: None  # noqa: E731


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
        torch.manual_seed(1000 + i)
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
            title=f"Cat 1 — {label}: {unit} vs curriculum_scale",
            subtitle=f"n_envs={N_ENVS}, draws/env={N_DRAWS}",
            xaxis_title="curriculum_scale",
            yaxis_title=f"{label} [{unit}]",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_yaxes(range=[y_min - pad, y_max + pad])
        return fig

    log_and_plot(
        name=f"cat1_{slug}_curriculum",
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
# 1.2–1.9: Simple curriculum violins
# ===================================================================


def plot_com_shift(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "com_shift",
        make_fn=lambda n, s: COMShift(setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.norm(dim=-1),
        unit="m",
        label="‖ΔCoM‖₂",
    )


def plot_inertia_tensor(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "inertia_tensor",
        make_fn=lambda n, s: InertiaTensor(
            mass_setter_fn=_NOOP_SETTER,
            com_setter_fn=_NOOP_SETTER,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="ratio",
        label="inertia scale (axis 0)",
    )


def plot_motor_armature(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "motor_armature",
        make_fn=lambda n, s: MotorArmature(
            setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="armature ratio",
    )


def plot_friction_ratio(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "friction_ratio",
        make_fn=lambda n, s: FrictionRatio(
            setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="friction ratio",
    )


def plot_position_gain_kp(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "position_gain_kp",
        make_fn=lambda n, s: PositionGainKp(
            setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="Kp ratio",
    )


def plot_velocity_gain_kv(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "velocity_gain_kv",
        make_fn=lambda n, s: VelocityGainKv(
            setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="Kv ratio",
    )


def plot_joint_stiffness(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "joint_stiffness",
        make_fn=lambda n, s: JointStiffness(
            setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="stiffness ratio",
    )


def plot_joint_damping(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "joint_damping",
        make_fn=lambda n, s: JointDamping(
            setter_fn=_NOOP_SETTER, n_envs=n, dt=DT, curriculum_scale=s
        ),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="damping ratio",
    )


# ===================================================================
# 1.10 AeroDragCoeff
# ===================================================================


def plot_aero_drag_coeff(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "aero_drag_coeff",
        make_fn=lambda n, s: AeroDragCoeff(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="ratio",
        label="Cd multiplier (axis 0)",
    )


# ===================================================================
# 1.11 GroundEffect — altitude sweep
# ===================================================================


def _make_env_state(n: int, altitude: float = 0.5) -> EnvState:
    return EnvState(
        pos=torch.tensor([[0.0, 0.0, altitude]]).expand(n, 3).clone(),
        quat=torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4).clone(),
        vel=torch.zeros(n, 3),
        ang_vel=torch.zeros(n, 3),
        acc=torch.zeros(n, 3),
        rpm=torch.ones(n, 4) * 14000.0,
        dt=DT,
        step=0,
    )


def plot_ground_effect(hw: HardwareMeta) -> None:
    n_pts = 100
    altitudes = torch.linspace(0.01, 0.5, n_pts)
    radii = [0.05, 0.1, 0.15]
    rows: list[dict] = []
    traces_data: dict[str, list[float]] = {}

    for r in radii:
        p = GroundEffect(n_envs=n_pts, dt=DT, rotor_radius=r)
        p.tick(is_reset=True)
        p.tick(is_reset=False)
        env_state = _make_env_state(n_pts)
        env_state.pos[:, 2] = altitudes
        wrench = p._compute_wrench(env_state)
        forces = wrench[:, 2].tolist()
        key = f"R={r}m"
        traces_data[key] = forces
        for alt, f in zip(altitudes.tolist(), forces):
            rows.append(
                {
                    "altitude_m": round(alt, 3),
                    "rotor_radius_m": r,
                    "delta_Fz_N": round(f, 6),
                }
            )

    stats_by = {k: stats_summary(v) for k, v in traces_data.items()}

    def fig_fn(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        for r in radii:
            key = f"R={r}m"
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
            title="Cat 1 — ground_effect: ΔF_z vs altitude",
            subtitle=("Cheeseman-Bennett: k_ge = 1/(1−(R/4h)²); nominal_thrust=4.905 N"),
            xaxis_title="altitude [m]",
            yaxis_title="ΔF_z [N]",
            hardware_footer=hw_.footer_line(),
        )
        return fig

    log_and_plot(
        name="cat1_ground_effect_altitude_sweep",
        category=CAT,
        metric="ground effect force vs altitude",
        unit="N",
        baseline_description="Cheeseman-Bennett model, k_ge clamped to [1, 2]",
        config={"radii": radii, "n_points": n_pts, "nominal_thrust": 4.905},
        hardware=hw,
        csv_rows=rows,
        csv_columns=["altitude_m", "rotor_radius_m", "delta_Fz_N"],
        stats_by_series=stats_by,
        figure_fn=fig_fn,
    )
    print("  [ground_effect/altitude_sweep] done")


# ===================================================================
# 1.12–1.14: Vector curriculum violins
# ===================================================================


def plot_chassis_geometry(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "chassis_geometry_asymmetry",
        make_fn=lambda n, s: ChassisGeometryAsymmetry(
            mass_setter_fn=_NOOP_SETTER,
            com_setter_fn=_NOOP_SETTER,
            n_envs=n,
            dt=DT,
            curriculum_scale=s,
        ),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="m",
        label="arm deviation (arm 0)",
    )


def plot_propeller_blade_damage(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "propeller_blade_damage",
        make_fn=lambda n, s: PropellerBladeDamage(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="ratio",
        label="blade efficiency (prop 0)",
    )


def plot_structural_flexibility(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "structural_flexibility",
        make_fn=lambda n, s: StructuralFlexibility(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value[:, 0],
        unit="N·m/rad",
        label="stiffness k",
    )


# ===================================================================
# 1.15 BatteryVoltageSag — curriculum violin + SoC trace
# ===================================================================


def plot_battery_voltage_sag(hw: HardwareMeta) -> None:
    _curriculum_violin(
        hw,
        "battery_voltage_sag",
        make_fn=lambda n, s: BatteryVoltageSag(n_envs=n, dt=DT, curriculum_scale=s),
        extract_fn=lambda p: p._current_value.squeeze(-1),
        unit="ratio",
        label="voltage ratio",
    )

    # SoC trace
    n_envs = 8
    n_steps = 500
    torch.manual_seed(1150)
    p = BatteryVoltageSag(n_envs=n_envs, dt=DT, curriculum_scale=1.0)
    p.reset(torch.arange(n_envs))

    soc_trace = torch.zeros(n_steps, n_envs)
    vratio_trace = torch.zeros(n_steps, n_envs)
    for step in range(n_steps):
        p.step()
        soc_trace[step] = p._soc.clone()
        vratio_trace[step] = p._current_value.squeeze(-1).clone()

    rows: list[dict] = []
    for step in range(n_steps):
        for e in range(n_envs):
            rows.append(
                {
                    "step": step,
                    "env_id": e,
                    "time_s": round(step * DT, 4),
                    "soc": float(soc_trace[step, e]),
                    "voltage_ratio": float(vratio_trace[step, e]),
                }
            )

    stats_soc = {f"env_{e}": stats_summary(soc_trace[:, e].tolist()) for e in range(n_envs)}

    def fig_fn(hw_: HardwareMeta) -> go.Figure:
        fig = make_figure()
        t = [round(s * DT, 4) for s in range(n_steps)]
        for e in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=vratio_trace[:, e].tolist(),
                    mode="lines",
                    name=f"env {e}",
                    line=dict(width=1.5),
                )
            )
        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="min voltage (0.7)",
        )
        apply_layout(
            fig,
            title="Cat 1 — battery_voltage_sag: voltage ratio over time",
            subtitle=(f"v_ratio = 0.7 + 0.3×SoC; n_envs={n_envs}, n_steps={n_steps}"),
            xaxis_title="time [s]",
            yaxis_title="voltage ratio [dimensionless]",
            hardware_footer=hw_.footer_line(),
        )
        fig.update_yaxes(range=[0.65, 1.05])
        return fig

    log_and_plot(
        name="cat1_battery_voltage_sag_trace",
        category=CAT,
        metric="voltage ratio trajectory",
        unit="ratio",
        baseline_description="SoC decays linearly, v_ratio = 0.7 + 0.3×SoC",
        config={"n_envs": n_envs, "n_steps": n_steps, "dt_s": DT},
        hardware=hw,
        csv_rows=rows,
        csv_columns=["step", "env_id", "time_s", "soc", "voltage_ratio"],
        stats_by_series=stats_soc,
        figure_fn=fig_fn,
    )
    print("  [battery_voltage_sag/trace] done")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    hardware = collect_hardware_meta(backend="cpu")
    print(f"hardware: {hardware.footer_line()}")

    specs = [
        ("1.2 com_shift", plot_com_shift),
        ("1.3 inertia_tensor", plot_inertia_tensor),
        ("1.4 motor_armature", plot_motor_armature),
        ("1.5 friction_ratio", plot_friction_ratio),
        ("1.6 position_gain_kp", plot_position_gain_kp),
        ("1.7 velocity_gain_kv", plot_velocity_gain_kv),
        ("1.8 joint_stiffness", plot_joint_stiffness),
        ("1.9 joint_damping", plot_joint_damping),
        ("1.10 aero_drag_coeff", plot_aero_drag_coeff),
        ("1.11 ground_effect", plot_ground_effect),
        ("1.12 chassis_geometry_asymmetry", plot_chassis_geometry),
        ("1.13 propeller_blade_damage", plot_propeller_blade_damage),
        ("1.14 structural_flexibility", plot_structural_flexibility),
        ("1.15 battery_voltage_sag", plot_battery_voltage_sag),
    ]

    for label, fn in specs:
        print(f"\n--- {label} ---")
        fn(hardware)

    print("\nDone. Inspect docs/impl/{data,assets}/cat1_*.{csv,meta.json,png}")


if __name__ == "__main__":
    main()
