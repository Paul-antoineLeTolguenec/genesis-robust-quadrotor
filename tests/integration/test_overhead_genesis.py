"""P6 — Relative overhead test: real perturbations vs bare scene.step().

Methodology:
  BASELINE:  scene.step() only (standard Genesis usage, no perturbation)
  PERTURBED: tick() + apply() + scene.step() (our full perturbation stack)

  Measures the TOTAL cost of our perturbation layer including:
  - tick() orchestration (sample, curriculum, clamp)
  - apply() physics computation (_compute_wrench, Genesis setters)

Thresholds:
  - FAIL if overhead > 200% (hard limit)
  - WARNING if overhead > 100% (logged, does not fail)

Uses a real Crazyflie CF2X URDF drone with batched environments.
Run locally: uv run pytest tests/integration/test_overhead_genesis.py -v -s
"""

from __future__ import annotations

import logging
import statistics
import time
import warnings

import pytest
import torch

try:
    import genesis as gs

    _GENESIS_AVAILABLE = True
except ImportError:
    _GENESIS_AVAILABLE = False

pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(not _GENESIS_AVAILABLE, reason="Genesis not installed"),
]

logger = logging.getLogger(__name__)

MAX_OVERHEAD_FAIL = 2.0  # 200% — hard fail
MAX_OVERHEAD_WARN = 1.0  # 100% — warning
WARMUP = 30
STEPS_PER_ROUND = 100
ROUNDS = 5
RESET_EVERY = 20
N_ENVS = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def genesis_env():
    """Create a Genesis scene with Crazyflie CF2X drone."""
    gs.init(backend=gs.cpu, logging_level="warning")
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
    scene.build(n_envs=N_ENVS)
    yield {"scene": scene, "drone": drone, "n_envs": N_ENVS}


@pytest.fixture(scope="module")
def env_state():
    """Pre-allocated EnvState reused across measurements."""
    from genesis_robust_rl.perturbations.base import EnvState

    return EnvState(
        pos=torch.rand(N_ENVS, 3).clamp(min=0.3) * 2,
        quat=torch.tensor([[1.0, 0, 0, 0]]).expand(N_ENVS, 4).clone(),
        vel=torch.randn(N_ENVS, 3) * 0.5,
        ang_vel=torch.randn(N_ENVS, 3) * 0.1,
        acc=torch.randn(N_ENVS, 3) * 0.1,
        rpm=torch.ones(N_ENVS, 4) * 14000.0,
        dt=0.005,
        step=0,
    )


@pytest.fixture(scope="module")
def all_perturbations(genesis_env):
    """Instantiate all 15 Cat 1 perturbations with real Genesis setters."""
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

    drone = genesis_env["drone"]
    n = genesis_env["n_envs"]
    dt = 0.005

    # Setter-based perturbations use real Genesis setters where available,
    # no-op lambdas where the CF2X doesn't expose the setter
    def noop(v, idx):  # noqa: ARG001
        return None

    return {
        "MassShift": MassShift(
            setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            n_envs=n,
            dt=dt,
        ),
        "COMShift": COMShift(
            setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
            n_envs=n,
            dt=dt,
        ),
        "InertiaTensor": InertiaTensor(
            mass_setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            com_setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
            n_envs=n,
            dt=dt,
        ),
        "MotorArmature": MotorArmature(setter_fn=noop, n_envs=n, dt=dt),
        "FrictionRatio": FrictionRatio(setter_fn=noop, n_envs=n, dt=dt),
        "PositionGainKp": PositionGainKp(setter_fn=noop, n_envs=n, dt=dt),
        "VelocityGainKv": VelocityGainKv(setter_fn=noop, n_envs=n, dt=dt),
        "JointStiffness": JointStiffness(setter_fn=noop, n_envs=n, dt=dt),
        "JointDamping": JointDamping(setter_fn=noop, n_envs=n, dt=dt),
        "AeroDragCoeff": AeroDragCoeff(n_envs=n, dt=dt),
        "GroundEffect": GroundEffect(n_envs=n, dt=dt),
        "ChassisGeometryAsymmetry": ChassisGeometryAsymmetry(
            mass_setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
            com_setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
            n_envs=n,
            dt=dt,
        ),
        "PropellerBladeDamage": PropellerBladeDamage(n_envs=n, dt=dt),
        "StructuralFlexibility": StructuralFlexibility(n_envs=n, dt=dt),
        "BatteryVoltageSag": BatteryVoltageSag(n_envs=n, dt=dt),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _measure_median(scene, loop_fn: callable) -> float:
    """Warmup + measure median of ROUNDS rounds."""
    for i in range(WARMUP):
        loop_fn()
        if (i + 1) % RESET_EVERY == 0:
            scene.reset()

    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for i in range(STEPS_PER_ROUND):
            loop_fn()
            if (i + 1) % RESET_EVERY == 0:
                scene.reset()
        t1 = time.perf_counter()
        times.append((t1 - t0) / STEPS_PER_ROUND)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
PERTURBATION_NAMES = [
    "MassShift",
    "COMShift",
    "InertiaTensor",
    "MotorArmature",
    "FrictionRatio",
    "PositionGainKp",
    "VelocityGainKv",
    "JointStiffness",
    "JointDamping",
    "AeroDragCoeff",
    "GroundEffect",
    "ChassisGeometryAsymmetry",
    "PropellerBladeDamage",
    "StructuralFlexibility",
    "BatteryVoltageSag",
]


@pytest.mark.parametrize("name", PERTURBATION_NAMES)
def test_perturbation_overhead(genesis_env, env_state, all_perturbations, name):
    """Each real perturbation must have overhead < 200%. Warn if > 100%."""
    scene = genesis_env["scene"]
    drone = genesis_env["drone"]
    p = all_perturbations[name]
    scene.reset()

    # Init perturbation
    p.tick(is_reset=True, env_ids=torch.arange(genesis_env["n_envs"]))

    # Baseline: bare scene.step()
    t_base = _measure_median(scene, lambda: scene.step())

    # Perturbed: tick + apply + step
    def perturbed_loop():
        p.tick(is_reset=False)
        p.apply(scene, drone, env_state)
        scene.step()

    t_pert = _measure_median(scene, perturbed_loop)

    overhead = (t_pert - t_base) / t_base
    overhead_pct = overhead * 100
    delta_us = (t_pert - t_base) * 1e6

    print(
        f"\n  {name}: base={t_base * 1e6:.0f}µs  pert={t_pert * 1e6:.0f}µs  "
        f"delta={delta_us:.0f}µs  overhead={overhead_pct:+.1f}%"
    )

    # Warning at 100%
    if overhead > MAX_OVERHEAD_WARN:
        warnings.warn(
            f"{name}: overhead {overhead_pct:.1f}% exceeds 100% warning threshold "
            f"(base={t_base * 1e6:.0f}µs, delta={delta_us:.0f}µs). "
            f"Consider optimizing _compute_wrench() or pre-allocating tensors.",
            UserWarning,
            stacklevel=1,
        )

    # Hard fail at 200%
    assert overhead < MAX_OVERHEAD_FAIL, (
        f"{name}: overhead {overhead_pct:.1f}% exceeds 200% limit "
        f"(base={t_base * 1e6:.0f}µs, pert={t_pert * 1e6:.0f}µs, delta={delta_us:.0f}µs)"
    )


def test_overhead_summary(genesis_env, env_state, all_perturbations):
    """Print a summary table of all perturbation overheads."""
    scene = genesis_env["scene"]
    drone = genesis_env["drone"]
    n_envs = genesis_env["n_envs"]
    scene.reset()

    t_base = _measure_median(scene, lambda: scene.step())

    print(f"\n{'=' * 70}")
    print(f"  Cat 1 Overhead Summary (n_envs={n_envs}, CPU, CF2X)")
    print(f"  Baseline: {t_base * 1e6:.0f} µs/step")
    print(f"{'=' * 70}")
    print(f"  {'Perturbation':<28s} {'Time':>8s} {'Delta':>8s} {'Overhead':>10s}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 10}")

    warn_count = 0
    for name in PERTURBATION_NAMES:
        p = all_perturbations[name]
        p.tick(is_reset=True, env_ids=torch.arange(n_envs))

        def make_loop(pert):
            def loop():
                pert.tick(is_reset=False)
                pert.apply(scene, drone, env_state)
                scene.step()

            return loop

        t_p = _measure_median(scene, make_loop(p))
        overhead = (t_p - t_base) / t_base * 100
        delta = (t_p - t_base) * 1e6
        flag = " ⚠" if overhead > 100 else ""
        print(f"  {name:<28s} {t_p * 1e6:>7.0f}µs {delta:>+7.0f}µs {overhead:>+9.1f}%{flag}")
        if overhead > 100:
            warn_count += 1

    print(f"{'=' * 70}")
    if warn_count:
        print(f"  ⚠ {warn_count} perturbation(s) above 100% warning threshold")
    print()
