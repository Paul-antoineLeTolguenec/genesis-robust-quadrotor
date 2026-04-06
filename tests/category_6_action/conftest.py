"""Fixtures for category 6 — action perturbations."""

import pytest

from genesis_robust_rl.perturbations.category_6_action import (
    ActionDeadzone,
    ActionNoise,
    ActionSaturation,
    ActuatorHysteresis,
    ESCLowPassFilter,
)


@pytest.fixture(
    params=[
        lambda n: ActionNoise(n_envs=n, dt=0.01),
        lambda n: ActionDeadzone(n_envs=n, dt=0.01),
        lambda n: ActionSaturation(n_envs=n, dt=0.01),
        lambda n: ActuatorHysteresis(n_envs=n, dt=0.01),
        lambda n: ESCLowPassFilter(n_envs=n, dt=0.01),
    ]
)
def perturbation(request, n_envs):
    """Parametrized fixture over all category-6 perturbation leaves."""
    return request.param(n_envs)


@pytest.fixture(
    params=[
        ActionNoise,
        ActionDeadzone,
        ActionSaturation,
        ActuatorHysteresis,
        ESCLowPassFilter,
    ]
)
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
