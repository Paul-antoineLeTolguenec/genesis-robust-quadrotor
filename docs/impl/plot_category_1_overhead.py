"""Generate overhead (%) vs n_envs plots for all Cat 1 perturbations.

Uses real Genesis Crazyflie CF2X scene — requires Genesis installed.
Run: uv run python docs/impl/plot_category_1_overhead.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import torch

import genesis as gs

from genesis_robust_rl.perturbations.base import EnvState
from genesis_robust_rl.perturbations.category_1_physics import (
    AeroDragCoeff,
    BatteryVoltageSag,
    ChassisGeometryAsymmetry,
    COMShift,
    FrictionRatio,
    GroundEffect,
    InertiaTensor,
    JointDamping,
    JointStiffness,
    MassShift,
    MotorArmature,
    PositionGainKp,
    PropellerBladeDamage,
    StructuralFlexibility,
    VelocityGainKv,
)

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

WARMUP = 30
STEPS_PER_ROUND = 100
ROUNDS = 5
RESET_EVERY = 20
N_ENVS_RANGE = [1, 4, 16, 64, 128]


def _make_env_state(n_envs: int) -> EnvState:
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


def _measure_median(scene, loop_fn, warmup: int, rounds: int, steps: int) -> float:
    for i in range(warmup):
        loop_fn()
        if (i + 1) % RESET_EVERY == 0:
            scene.reset()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for i in range(steps):
            loop_fn()
            if (i + 1) % RESET_EVERY == 0:
                scene.reset()
        t1 = time.perf_counter()
        times.append((t1 - t0) / steps)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Perturbation factories — returns (perturbation, env_state) for a given scene
# ---------------------------------------------------------------------------

def _make_perturbations(scene, drone, n_envs: int) -> dict:
    """Instantiate all 15 Cat 1 perturbations for a given scene."""
    envs_idx = torch.arange(n_envs)

    return {
        "MassShift": MassShift(
            setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            n_envs=n_envs, dt=0.005,
        ),
        "COMShift": COMShift(
            setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
            n_envs=n_envs, dt=0.005,
        ),
        "InertiaTensor": InertiaTensor(
            mass_setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            com_setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
            n_envs=n_envs, dt=0.005,
        ),
        "MotorArmature": MotorArmature(
            setter_fn=lambda v, idx: None,  # armature setter not available on CF2X
            n_envs=n_envs, dt=0.005,
        ),
        "FrictionRatio": FrictionRatio(
            setter_fn=lambda v, idx: None,  # friction not available on CF2X
            n_envs=n_envs, dt=0.005,
        ),
        "PositionGainKp": PositionGainKp(
            setter_fn=lambda v, idx: None,  # kp not available on CF2X
            n_envs=n_envs, dt=0.005,
        ),
        "VelocityGainKv": VelocityGainKv(
            setter_fn=lambda v, idx: None,  # kv not available on CF2X
            n_envs=n_envs, dt=0.005,
        ),
        "JointStiffness": JointStiffness(
            setter_fn=lambda v, idx: None,
            n_envs=n_envs, dt=0.005,
        ),
        "JointDamping": JointDamping(
            setter_fn=lambda v, idx: None,
            n_envs=n_envs, dt=0.005,
        ),
        "AeroDragCoeff": AeroDragCoeff(n_envs=n_envs, dt=0.005),
        "GroundEffect": GroundEffect(n_envs=n_envs, dt=0.005),
        "ChassisGeometryAsymmetry": ChassisGeometryAsymmetry(
            mass_setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            com_setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
            n_envs=n_envs, dt=0.005,
        ),
        "PropellerBladeDamage": PropellerBladeDamage(n_envs=n_envs, dt=0.005),
        "StructuralFlexibility": StructuralFlexibility(n_envs=n_envs, dt=0.005),
        "BatteryVoltageSag": BatteryVoltageSag(n_envs=n_envs, dt=0.005),
    }


# ---------------------------------------------------------------------------
# Main: measure overhead for each perturbation across n_envs
# ---------------------------------------------------------------------------

def main() -> None:
    gs.init(backend=gs.cpu, logging_level="warning")

    # Results: {perturbation_name: [overhead_pct_per_n_envs]}
    results: dict[str, list[float]] = {}
    baselines: list[float] = []

    for n_envs in N_ENVS_RANGE:
        print(f"\n--- n_envs={n_envs} ---")

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
        drone = scene.add_entity(
            gs.morphs.URDF(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.5)),
        )
        scene.build(n_envs=n_envs)

        # Baseline: scene.step() only
        t_base = _measure_median(
            scene, lambda: scene.step(), WARMUP, ROUNDS, STEPS_PER_ROUND
        )
        baselines.append(t_base * 1e6)
        print(f"  baseline: {t_base*1e6:.0f} µs/step")

        # Perturbations
        perturbations = _make_perturbations(scene, drone, n_envs)
        env_state = _make_env_state(n_envs)

        for name, p in perturbations.items():
            p.tick(is_reset=True, env_ids=torch.arange(n_envs))

            def make_loop(pert, es):
                def loop():
                    pert.tick(is_reset=False)
                    pert.apply(scene, drone, es)
                    scene.step()
                return loop

            t_p = _measure_median(
                scene, make_loop(p, env_state), WARMUP, ROUNDS, STEPS_PER_ROUND
            )
            overhead_pct = (t_p - t_base) / t_base * 100

            results.setdefault(name, []).append(overhead_pct)
            print(f"  {name}: {t_p*1e6:.0f} µs  overhead={overhead_pct:+.1f}%")

    # ---------------------------------------------------------------------------
    # Generate one plot per perturbation
    # ---------------------------------------------------------------------------
    slug_map = {
        "MassShift": "mass_shift",
        "COMShift": "com_shift",
        "InertiaTensor": "inertia_tensor",
        "MotorArmature": "motor_armature",
        "FrictionRatio": "friction_ratio",
        "PositionGainKp": "position_gain_kp",
        "VelocityGainKv": "velocity_gain_kv",
        "JointStiffness": "joint_stiffness",
        "JointDamping": "joint_damping",
        "AeroDragCoeff": "aero_drag_coeff",
        "GroundEffect": "ground_effect",
        "ChassisGeometryAsymmetry": "chassis_asymmetry",
        "PropellerBladeDamage": "blade_damage",
        "StructuralFlexibility": "structural_flexibility",
        "BatteryVoltageSag": "battery_voltage_sag",
    }

    for name, overheads in results.items():
        slug = slug_map[name]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=N_ENVS_RANGE,
            y=overheads,
            mode="lines+markers",
            name="overhead (%)",
            marker=dict(size=8, color="#EF553B"),
            line=dict(width=2, color="#EF553B"),
        ))
        fig.update_layout(
            title=f"{name} — overhead vs n_envs (CPU, Crazyflie CF2X)",
            xaxis_title="n_envs",
            yaxis_title="overhead (%)",
            xaxis_type="log",
            template="plotly_white",
            width=700,
            height=420,
        )
        fig.add_annotation(
            x=N_ENVS_RANGE[-1],
            y=overheads[-1],
            text=f"{overheads[-1]:+.1f}%",
            showarrow=True,
            arrowhead=2,
        )

        fname = f"cat1_{slug}_perf.png"
        pio.write_image(fig, ASSETS / fname, scale=2)
        print(f"Saved: {fname}")

    # ---------------------------------------------------------------------------
    # Summary plot: all perturbations on one chart
    # ---------------------------------------------------------------------------
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    ]

    fig = go.Figure()
    for i, (name, overheads) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=N_ENVS_RANGE,
            y=overheads,
            mode="lines+markers",
            name=name,
            marker=dict(size=5),
            line=dict(width=1.5, color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        title="Cat 1 — all perturbations overhead vs n_envs (CPU, CF2X)",
        xaxis_title="n_envs",
        yaxis_title="overhead (%)",
        xaxis_type="log",
        template="plotly_white",
        width=900,
        height=520,
        legend=dict(font=dict(size=9)),
    )

    pio.write_image(fig, ASSETS / "cat1_all_overhead_summary.png", scale=2)
    print("Saved: cat1_all_overhead_summary.png")


if __name__ == "__main__":
    main()
