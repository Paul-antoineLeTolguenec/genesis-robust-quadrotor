"""Performance tests for category 1 — physics perturbations (CPU).

Measures tick() + apply() overhead at n_envs=1 and n_envs=512.
Fails if overhead exceeds defined thresholds.
"""
import time
from typing import Callable

import pytest
import torch
from unittest.mock import MagicMock

from genesis_robust_rl.perturbations.category_1_physics import (
    COMShift,
    InertiaTensor,
    JointDamping,
    JointStiffness,
    MassShift,
    MotorArmature,
    PositionGainKp,
    VelocityGainKv,
)
from genesis_robust_rl.perturbations.base import Perturbation
from tests.conftest import EnvState

WARMUP = 200
STEPS = 2000

# Thresholds (CPU, no Genesis)
MAX_TICK_MS_PER_STEP = 0.1   # tick() budget: sampling + curriculum scaling
MAX_APPLY_MS_PER_STEP = 0.05  # apply() budget: setter dispatch only


def _make_mass_shift(n_envs: int) -> MassShift:
    return MassShift(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_com_shift(n_envs: int) -> COMShift:
    return COMShift(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_inertia_tensor(n_envs: int) -> InertiaTensor:
    return InertiaTensor(mass_setter_fn=MagicMock(), com_setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_motor_armature(n_envs: int) -> MotorArmature:
    return MotorArmature(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_position_gain_kp(n_envs: int) -> PositionGainKp:
    return PositionGainKp(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_velocity_gain_kv(n_envs: int) -> VelocityGainKv:
    return VelocityGainKv(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_joint_stiffness(n_envs: int) -> JointStiffness:
    return JointStiffness(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_joint_damping(n_envs: int) -> JointDamping:
    return JointDamping(setter_fn=MagicMock(), n_envs=n_envs, dt=0.01)


def _make_env_state(n_envs: int) -> EnvState:
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


def _run_tick_perf(factory: Callable[[int], Perturbation], n_envs: int) -> float:
    """Run tick() perf benchmark and return ms/step."""
    p = factory(n_envs)
    for _ in range(WARMUP):
        p.tick(is_reset=True)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=True)
    return (time.perf_counter() - start) * 1000 / STEPS


def _run_apply_perf(factory: Callable[[int], Perturbation], n_envs: int) -> float:
    """Run apply() perf benchmark and return ms/step."""
    p = factory(n_envs)
    scene, drone = MagicMock(), MagicMock()
    env_state = _make_env_state(n_envs)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.apply(scene, drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.apply(scene, drone, env_state)
    return (time.perf_counter() - start) * 1000 / STEPS


# ---------------------------------------------------------------------------
# MassShift perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_mass_shift_tick_overhead_cpu(n_envs: int) -> None:
    """MassShift tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_mass_shift, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"MassShift tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_mass_shift_apply_overhead_cpu(n_envs: int) -> None:
    """MassShift apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_mass_shift, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"MassShift apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# COMShift perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_com_shift_tick_overhead_cpu(n_envs: int) -> None:
    """COMShift tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_com_shift, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"COMShift tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_com_shift_apply_overhead_cpu(n_envs: int) -> None:
    """COMShift apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_com_shift, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"COMShift apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# InertiaTensor perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_inertia_tensor_tick_overhead_cpu(n_envs: int) -> None:
    """InertiaTensor tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_inertia_tensor, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"InertiaTensor tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_inertia_tensor_apply_overhead_cpu(n_envs: int) -> None:
    """InertiaTensor apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_inertia_tensor, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"InertiaTensor apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# MotorArmature perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_motor_armature_tick_overhead_cpu(n_envs: int) -> None:
    """MotorArmature tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_motor_armature, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"MotorArmature tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_motor_armature_apply_overhead_cpu(n_envs: int) -> None:
    """MotorArmature apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_motor_armature, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"MotorArmature apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# PositionGainKp perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_position_gain_kp_tick_overhead_cpu(n_envs: int) -> None:
    """PositionGainKp tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_position_gain_kp, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"PositionGainKp tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_position_gain_kp_apply_overhead_cpu(n_envs: int) -> None:
    """PositionGainKp apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_position_gain_kp, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"PositionGainKp apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# VelocityGainKv perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_velocity_gain_kv_tick_overhead_cpu(n_envs: int) -> None:
    """VelocityGainKv tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_velocity_gain_kv, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"VelocityGainKv tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_velocity_gain_kv_apply_overhead_cpu(n_envs: int) -> None:
    """VelocityGainKv apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_velocity_gain_kv, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"VelocityGainKv apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# JointStiffness perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_joint_stiffness_tick_overhead_cpu(n_envs: int) -> None:
    """JointStiffness tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_joint_stiffness, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"JointStiffness tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_joint_stiffness_apply_overhead_cpu(n_envs: int) -> None:
    """JointStiffness apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_joint_stiffness, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"JointStiffness apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )


# ---------------------------------------------------------------------------
# JointDamping perf
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_joint_damping_tick_overhead_cpu(n_envs: int) -> None:
    """JointDamping tick(is_reset=True) must stay under MAX_TICK_MS_PER_STEP on CPU."""
    elapsed_ms = _run_tick_perf(_make_joint_damping, n_envs)
    assert elapsed_ms < MAX_TICK_MS_PER_STEP, (
        f"JointDamping tick() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_TICK_MS_PER_STEP} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_joint_damping_apply_overhead_cpu(n_envs: int) -> None:
    """JointDamping apply() must stay under MAX_APPLY_MS_PER_STEP on CPU."""
    elapsed_ms = _run_apply_perf(_make_joint_damping, n_envs)
    assert elapsed_ms < MAX_APPLY_MS_PER_STEP, (
        f"JointDamping apply() too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms/step "
        f"(limit {MAX_APPLY_MS_PER_STEP} ms)"
    )
