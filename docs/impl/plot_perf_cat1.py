"""Cat 1 — Physics perturbations: overhead (%) vs n_envs on Genesis CF2X.

Replaces ``plot_category_1_overhead.py`` (which used the old PNG-only pipeline).
Measures the overhead of the 15 Cat 1 perturbations using the shared framework
(CSV + meta JSON + unified Plotly template + hardware footer).

Perturbation kinds:
  * GenesisSetterPerturbation (11) — receive a setter lambda bound to the
    drone. For shifts not exposed on the CF2X (armature, friction, dof gains,
    joint stiffness/damping), we use a no-op lambda so only our perturbation
    logic (tick + apply dispatch) is timed.
  * ExternalWrenchPerturbation (4) — no setter; physics kind.

Run:
    uv run python docs/impl/plot_perf_cat1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

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
    MassShift,
    MotorArmature,
    PositionGainKp,
    PropellerBladeDamage,
    StructuralFlexibility,
    VelocityGainKv,
)

CAT = 1
DT = 0.005


def _noop(v, idx) -> None:
    """No-op setter for CF2X shifts that Genesis does not expose for this model."""


def _make_mass_shift(scene, drone, n_envs: int) -> MassShift:
    return MassShift(
        setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
        n_envs=n_envs,
        dt=DT,
    )


def _make_com_shift(scene, drone, n_envs: int) -> COMShift:
    return COMShift(
        setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
        n_envs=n_envs,
        dt=DT,
    )


def _make_inertia_tensor(scene, drone, n_envs: int) -> InertiaTensor:
    return InertiaTensor(
        mass_setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
        com_setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
        n_envs=n_envs,
        dt=DT,
    )


def _make_chassis_asymmetry(scene, drone, n_envs: int) -> ChassisGeometryAsymmetry:
    return ChassisGeometryAsymmetry(
        mass_setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
        com_setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
        n_envs=n_envs,
        dt=DT,
    )


def _make_setter_noop(cls):
    """Factory for GenesisSetter classes whose CF2X setter is a no-op."""

    def factory(scene, drone, n_envs: int):
        return cls(setter_fn=_noop, n_envs=n_envs, dt=DT)

    return factory


def _make_external_wrench(cls):
    """Factory for ExternalWrench classes (no setter needed)."""

    def factory(scene, drone, n_envs: int):
        return cls(n_envs=n_envs, dt=DT)

    return factory


SPECS: list[PertSpec] = [
    # Real Genesis setters
    PertSpec("cat1_mass_shift", "mass_shift", "physics", _make_mass_shift),
    PertSpec("cat1_com_shift", "com_shift", "physics", _make_com_shift),
    PertSpec("cat1_inertia_tensor", "inertia_tensor", "physics", _make_inertia_tensor),
    PertSpec(
        "cat1_chassis_geometry_asymmetry",
        "chassis_geometry_asymmetry",
        "physics",
        _make_chassis_asymmetry,
    ),
    # No-op setters (CF2X does not expose these)
    PertSpec(
        "cat1_motor_armature",
        "motor_armature",
        "physics",
        _make_setter_noop(MotorArmature),
    ),
    PertSpec(
        "cat1_friction_ratio",
        "friction_ratio",
        "physics",
        _make_setter_noop(FrictionRatio),
    ),
    PertSpec(
        "cat1_position_gain_kp",
        "position_gain_kp",
        "physics",
        _make_setter_noop(PositionGainKp),
    ),
    PertSpec(
        "cat1_velocity_gain_kv",
        "velocity_gain_kv",
        "physics",
        _make_setter_noop(VelocityGainKv),
    ),
    PertSpec(
        "cat1_joint_stiffness",
        "joint_stiffness",
        "physics",
        _make_setter_noop(JointStiffness),
    ),
    PertSpec(
        "cat1_joint_damping",
        "joint_damping",
        "physics",
        _make_setter_noop(JointDamping),
    ),
    # ExternalWrench
    PertSpec(
        "cat1_aero_drag_coeff",
        "aero_drag_coeff",
        "physics",
        _make_external_wrench(AeroDragCoeff),
    ),
    PertSpec(
        "cat1_ground_effect",
        "ground_effect",
        "physics",
        _make_external_wrench(GroundEffect),
    ),
    PertSpec(
        "cat1_propeller_blade_damage",
        "propeller_blade_damage",
        "physics",
        _make_external_wrench(PropellerBladeDamage),
    ),
    PertSpec(
        "cat1_structural_flexibility",
        "structural_flexibility",
        "physics",
        _make_external_wrench(StructuralFlexibility),
    ),
    PertSpec(
        "cat1_battery_voltage_sag",
        "battery_voltage_sag",
        "physics",
        _make_external_wrench(BatteryVoltageSag),
    ),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
