"""Performance tests for category 3 — temporal perturbations (CPU).

Measures tick() + apply() overhead at n_envs=1 and n_envs=512.
"""

import time

import pytest
import torch

from genesis_robust_rl.perturbations.category_3_temporal import (
    ActionFixedDelay,
    ActionVariableDelay,
    ComputationOverload,
    ObsFixedDelay,
    ObsVariableDelay,
    PacketLoss,
)

WARMUP = 200
STEPS = 2000

MAX_TICK_MS_PER_STEP = 0.1
MAX_APPLY_MS_PER_STEP = 0.1  # slightly higher than cat1 due to buffer ops


def _run_tick_perf(factory, n_envs: int) -> float:
    p = factory(n_envs)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
    return (time.perf_counter() - start) * 1000 / STEPS


def _run_apply_perf_obs(factory, n_envs: int, obs_dim: int = 32) -> float:
    p = factory(n_envs)
    obs = torch.randn(n_envs, obs_dim)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.apply(obs)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.apply(obs)
    return (time.perf_counter() - start) * 1000 / STEPS


def _run_apply_perf_action(factory, n_envs: int, action_dim: int = 4) -> float:
    p = factory(n_envs)
    action = torch.randn(n_envs, action_dim)
    p.tick(is_reset=True)
    # For PacketLoss/ComputationOverload, need step() to set masks
    for _ in range(WARMUP):
        p.tick(is_reset=False)
        p.apply(action)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(action)
    return (time.perf_counter() - start) * 1000 / STEPS


# ---------------------------------------------------------------------------
# ObsFixedDelay
# ---------------------------------------------------------------------------


def _make_obs_fixed(n: int) -> ObsFixedDelay:
    return ObsFixedDelay(obs_slice=slice(0, 6), obs_dim=6, n_envs=n, dt=0.01)


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_obs_fixed_delay_tick_cpu(n_envs: int) -> None:
    elapsed = _run_tick_perf(_make_obs_fixed, n_envs)
    assert elapsed < MAX_TICK_MS_PER_STEP, (
        f"ObsFixedDelay tick() too slow: {elapsed:.4f} ms (limit {MAX_TICK_MS_PER_STEP})"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_obs_fixed_delay_apply_cpu(n_envs: int) -> None:
    elapsed = _run_apply_perf_obs(_make_obs_fixed, n_envs)
    assert elapsed < MAX_APPLY_MS_PER_STEP, (
        f"ObsFixedDelay apply() too slow: {elapsed:.4f} ms (limit {MAX_APPLY_MS_PER_STEP})"
    )


# ---------------------------------------------------------------------------
# ObsVariableDelay
# ---------------------------------------------------------------------------


def _make_obs_var(n: int) -> ObsVariableDelay:
    return ObsVariableDelay(obs_slice=slice(0, 6), obs_dim=6, n_envs=n, dt=0.01)


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_obs_variable_delay_tick_cpu(n_envs: int) -> None:
    elapsed = _run_tick_perf(_make_obs_var, n_envs)
    assert elapsed < MAX_TICK_MS_PER_STEP, (
        f"ObsVariableDelay tick() too slow: {elapsed:.4f} ms"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_obs_variable_delay_apply_cpu(n_envs: int) -> None:
    elapsed = _run_apply_perf_obs(_make_obs_var, n_envs)
    assert elapsed < MAX_APPLY_MS_PER_STEP, (
        f"ObsVariableDelay apply() too slow: {elapsed:.4f} ms"
    )


# ---------------------------------------------------------------------------
# ActionFixedDelay
# ---------------------------------------------------------------------------


def _make_act_fixed(n: int) -> ActionFixedDelay:
    return ActionFixedDelay(n_envs=n, dt=0.01, action_dim=4)


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_action_fixed_delay_tick_cpu(n_envs: int) -> None:
    elapsed = _run_tick_perf(_make_act_fixed, n_envs)
    assert elapsed < MAX_TICK_MS_PER_STEP, (
        f"ActionFixedDelay tick() too slow: {elapsed:.4f} ms"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_action_fixed_delay_apply_cpu(n_envs: int) -> None:
    elapsed = _run_apply_perf_action(_make_act_fixed, n_envs)
    assert elapsed < MAX_APPLY_MS_PER_STEP, (
        f"ActionFixedDelay apply() too slow: {elapsed:.4f} ms"
    )


# ---------------------------------------------------------------------------
# ActionVariableDelay
# ---------------------------------------------------------------------------


def _make_act_var(n: int) -> ActionVariableDelay:
    return ActionVariableDelay(n_envs=n, dt=0.01, action_dim=4)


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_action_variable_delay_tick_cpu(n_envs: int) -> None:
    elapsed = _run_tick_perf(_make_act_var, n_envs)
    assert elapsed < MAX_TICK_MS_PER_STEP, (
        f"ActionVariableDelay tick() too slow: {elapsed:.4f} ms"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_action_variable_delay_apply_cpu(n_envs: int) -> None:
    elapsed = _run_apply_perf_action(_make_act_var, n_envs)
    assert elapsed < MAX_APPLY_MS_PER_STEP, (
        f"ActionVariableDelay apply() too slow: {elapsed:.4f} ms"
    )


# ---------------------------------------------------------------------------
# PacketLoss
# ---------------------------------------------------------------------------


def _make_packet_loss(n: int) -> PacketLoss:
    return PacketLoss(n_envs=n, dt=0.01, action_dim=4)


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_packet_loss_tick_cpu(n_envs: int) -> None:
    elapsed = _run_tick_perf(_make_packet_loss, n_envs)
    assert elapsed < MAX_TICK_MS_PER_STEP, (
        f"PacketLoss tick() too slow: {elapsed:.4f} ms"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_packet_loss_apply_cpu(n_envs: int) -> None:
    elapsed = _run_apply_perf_action(_make_packet_loss, n_envs)
    assert elapsed < MAX_APPLY_MS_PER_STEP, (
        f"PacketLoss apply() too slow: {elapsed:.4f} ms"
    )


# ---------------------------------------------------------------------------
# ComputationOverload
# ---------------------------------------------------------------------------


def _make_computation_overload(n: int) -> ComputationOverload:
    return ComputationOverload(n_envs=n, dt=0.01, action_dim=4)


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_computation_overload_tick_cpu(n_envs: int) -> None:
    elapsed = _run_tick_perf(_make_computation_overload, n_envs)
    assert elapsed < MAX_TICK_MS_PER_STEP, (
        f"ComputationOverload tick() too slow: {elapsed:.4f} ms"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_computation_overload_apply_cpu(n_envs: int) -> None:
    elapsed = _run_apply_perf_action(_make_computation_overload, n_envs)
    assert elapsed < MAX_APPLY_MS_PER_STEP, (
        f"ComputationOverload apply() too slow: {elapsed:.4f} ms"
    )
