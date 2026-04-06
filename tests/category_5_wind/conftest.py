"""Fixtures for category 5 — wind and external forces perturbations."""

import pytest

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


@pytest.fixture(
    params=[
        lambda n: ConstantWind(n_envs=n, dt=0.01),
        lambda n: Turbulence(n_envs=n, dt=0.01),
        lambda n: WindGust(n_envs=n, dt=0.01),
        lambda n: WindShear(n_envs=n, dt=0.01),
        lambda n: AdversarialWind(n_envs=n, dt=0.01),
        lambda n: BladeVortexInteraction(n_envs=n, dt=0.01),
        lambda n: GroundEffectBoundary(n_envs=n, dt=0.01),
        lambda n: PayloadSway(n_envs=n, dt=0.01),
        lambda n: ProximityDisturbance(n_envs=n, dt=0.01),
    ]
)
def perturbation(request, n_envs):
    """Parametrized fixture over all category-5 perturbation leaves."""
    return request.param(n_envs)


@pytest.fixture(
    params=[
        ConstantWind,
        Turbulence,
        WindGust,
        WindShear,
        AdversarialWind,
        BladeVortexInteraction,
        GroundEffectBoundary,
        PayloadSway,
        ProximityDisturbance,
    ]
)
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
