"""Performance tests for category 8 — external disturbances.

P1 CPU: tick + apply overhead at n_envs=1 and n_envs=512.
"""

import time
from unittest.mock import MagicMock

import pytest
import torch

from genesis_robust_rl.perturbations.category_8_external import (
    BodyForceDisturbance,
    BodyTorqueDisturbance,
)
from tests.conftest import EnvState

WARMUP = 50
STEPS = 500
MAX_MS = 0.5  # per step


def _make_env_state(n_envs: int) -> EnvState:
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 2.0,
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.randn(n_envs, 3) * 0.2,
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


def _make_scene() -> MagicMock:
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    scene.drone = MagicMock()
    return scene


# -- 8.1 BodyForceDisturbance --


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_body_force_uniform_perf(n_envs: int) -> None:
    scene = _make_scene()
    p = BodyForceDisturbance(n_envs=n_envs, dt=0.01)
    env_state = _make_env_state(n_envs)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_MS, f"BodyForceDisturbance(uniform) too slow: {elapsed_ms:.3f} ms/step"


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_body_force_ou_perf(n_envs: int) -> None:
    scene = _make_scene()
    p = BodyForceDisturbance(n_envs=n_envs, dt=0.01, distribution="ou_process")
    env_state = _make_env_state(n_envs)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_MS, f"BodyForceDisturbance(ou) too slow: {elapsed_ms:.3f} ms/step"


# -- 8.2 BodyTorqueDisturbance --


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_body_torque_uniform_perf(n_envs: int) -> None:
    scene = _make_scene()
    p = BodyTorqueDisturbance(n_envs=n_envs, dt=0.01)
    env_state = _make_env_state(n_envs)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_MS, f"BodyTorqueDisturbance(uniform) too slow: {elapsed_ms:.3f} ms/step"


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_body_torque_ou_perf(n_envs: int) -> None:
    scene = _make_scene()
    p = BodyTorqueDisturbance(n_envs=n_envs, dt=0.01, distribution="ou_process")
    env_state = _make_env_state(n_envs)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_MS, f"BodyTorqueDisturbance(ou) too slow: {elapsed_ms:.3f} ms/step"
