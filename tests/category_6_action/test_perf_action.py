"""P1 CPU performance tests for category 6 — action perturbations."""

import time

import pytest
import torch

from genesis_robust_rl.perturbations.category_6_action import (
    ActionDeadzone,
    ActionNoise,
    ActionSaturation,
    ActuatorHysteresis,
    ESCLowPassFilter,
)

PERF_WARMUP = 50
PERF_STEPS = 500
MAX_MS_PER_STEP = 1.0
ACTION_DIM = 4


@pytest.fixture(
    params=[
        ("action_noise", ActionNoise, 1),
        ("action_noise", ActionNoise, 512),
        ("action_deadzone", ActionDeadzone, 1),
        ("action_deadzone", ActionDeadzone, 512),
        ("action_saturation", ActionSaturation, 1),
        ("action_saturation", ActionSaturation, 512),
        ("actuator_hysteresis", ActuatorHysteresis, 1),
        ("actuator_hysteresis", ActuatorHysteresis, 512),
        ("esc_low_pass_filter", ESCLowPassFilter, 1),
        ("esc_low_pass_filter", ESCLowPassFilter, 512),
    ],
    ids=lambda p: f"{p[0]}_n{p[2]}",
)
def perf_setup(request):
    name, cls, n_envs = request.param
    p = cls(n_envs=n_envs, dt=0.01)
    return name, p, n_envs


@pytest.mark.perf
def test_tick_apply_overhead(perf_setup):
    """tick() + apply() within CPU threshold."""
    name, p, n_envs = perf_setup
    action = torch.randn(n_envs, ACTION_DIM)

    p.tick(is_reset=True)

    for _ in range(PERF_WARMUP):
        p.tick(is_reset=False)
        p.apply(action)

    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        p.tick(is_reset=False)
        p.apply(action)
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS

    assert elapsed_ms < MAX_MS_PER_STEP, f"{name} n_envs={n_envs}: {elapsed_ms:.3f} ms/step"
