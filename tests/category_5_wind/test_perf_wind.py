"""P1 CPU performance tests for category 5 — wind perturbations."""

import time

import pytest
import torch

from genesis_robust_rl.perturbations.category_5_wind import (
    AdversarialWind,
    BladeVortexInteraction,
    ConstantWind,
    GroundEffectBoundary,
    PayloadSway,
    ProximityDisturbance,
    Turbulence,
    WindGust,
    WindShear,
)
from tests.conftest import EnvState

PERF_WARMUP = 50
PERF_STEPS = 500
MAX_MS_PER_STEP = 1.0  # CPU threshold (ms)


def _make_env_state(n_envs: int) -> EnvState:
    return EnvState(
        pos=torch.rand(n_envs, 3) * 2.0,
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 2.0,
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.randn(n_envs, 3) * 0.2,
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


@pytest.fixture(
    params=[
        ("constant_wind", ConstantWind, 1),
        ("constant_wind", ConstantWind, 512),
        ("turbulence", Turbulence, 1),
        ("turbulence", Turbulence, 512),
        ("wind_gust", WindGust, 1),
        ("wind_gust", WindGust, 512),
        ("wind_shear", WindShear, 1),
        ("wind_shear", WindShear, 512),
        ("adversarial_wind", AdversarialWind, 1),
        ("adversarial_wind", AdversarialWind, 512),
        ("blade_vortex", BladeVortexInteraction, 1),
        ("blade_vortex", BladeVortexInteraction, 512),
        ("ground_effect_boundary", GroundEffectBoundary, 1),
        ("ground_effect_boundary", GroundEffectBoundary, 512),
        ("payload_sway", PayloadSway, 1),
        ("payload_sway", PayloadSway, 512),
        ("proximity_disturbance", ProximityDisturbance, 1),
        ("proximity_disturbance", ProximityDisturbance, 512),
    ],
    ids=lambda p: f"{p[0]}_n{p[2]}",
)
def perf_setup(request):
    name, cls, n_envs = request.param
    p = cls(n_envs=n_envs, dt=0.01)
    env_state = _make_env_state(n_envs)
    return name, p, env_state, n_envs


@pytest.mark.perf
def test_tick_apply_overhead(perf_setup):
    """tick() + apply() must complete within CPU threshold per step."""
    from unittest.mock import MagicMock

    name, p, env_state, n_envs = perf_setup

    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    drone = MagicMock()

    p.tick(is_reset=True)

    # Warmup
    for _ in range(PERF_WARMUP):
        p.tick(is_reset=False)
        p.apply(scene, drone, env_state)

    # Measure
    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        p.tick(is_reset=False)
        p.apply(scene, drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS

    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"{name} n_envs={n_envs}: {elapsed_ms:.3f} ms/step (limit {MAX_MS_PER_STEP} ms)"
    )
