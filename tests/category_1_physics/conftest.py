"""Fixtures for category 1 — physics perturbations."""
import pytest

from genesis_robust_rl.perturbations.category_1_physics import (
    AeroDragCoeff,
    BatteryVoltageSag,
    ChassisGeometryAsymmetry,
    COMShift,
    FrictionRatio,
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


@pytest.fixture(
    params=[
        lambda n, scene: MassShift(setter_fn=scene.drone.set_links_mass_shift, n_envs=n, dt=0.01),
        lambda n, scene: COMShift(setter_fn=scene.drone.set_links_COM_shift, n_envs=n, dt=0.01),
        lambda n, scene: InertiaTensor(
            mass_setter_fn=scene.drone.set_links_mass_shift,
            com_setter_fn=scene.drone.set_links_COM_shift,
            n_envs=n,
            dt=0.01,
        ),
        lambda n, scene: MotorArmature(setter_fn=scene.drone.set_dofs_armature, n_envs=n, dt=0.01),
        lambda n, scene: FrictionRatio(setter_fn=scene.drone.set_geoms_friction_ratio, n_envs=n, dt=0.01),
        lambda n, scene: PositionGainKp(setter_fn=scene.drone.set_dofs_kp, n_envs=n, dt=0.01),
        lambda n, scene: VelocityGainKv(setter_fn=scene.drone.set_dofs_kv, n_envs=n, dt=0.01),
        lambda n, scene: JointStiffness(setter_fn=scene.drone.set_dofs_stiffness, n_envs=n, dt=0.01),
        lambda n, scene: JointDamping(setter_fn=scene.drone.set_dofs_damping, n_envs=n, dt=0.01),
        lambda n, scene: ChassisGeometryAsymmetry(
            mass_setter_fn=scene.drone.set_links_mass_shift,
            com_setter_fn=scene.drone.set_links_COM_shift,
            n_envs=n,
            dt=0.01,
        ),
        lambda n, scene: PropellerBladeDamage(n_envs=n, dt=0.01),
        lambda n, scene: StructuralFlexibility(n_envs=n, dt=0.01),
        lambda n, scene: BatteryVoltageSag(n_envs=n, dt=0.01),
    ]
)
def perturbation(request, n_envs, mock_scene):
    """Parametrized fixture over all category-1 perturbation leaves."""
    return request.param(n_envs, mock_scene)


@pytest.fixture(
    params=[
        MassShift,
        COMShift,
        InertiaTensor,
        MotorArmature,
        FrictionRatio,
        PositionGainKp,
        VelocityGainKv,
        JointStiffness,
        JointDamping,
        ChassisGeometryAsymmetry,
        PropellerBladeDamage,
        StructuralFlexibility,
        BatteryVoltageSag,
    ]
)
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
