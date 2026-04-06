"""Fixtures for category 8 — external disturbances."""

import pytest

from genesis_robust_rl.perturbations.category_8_external import (
    BodyForceDisturbance,
    BodyTorqueDisturbance,
)


@pytest.fixture(
    params=[
        # Stateless (uniform)
        lambda n, _scene: BodyForceDisturbance(n_envs=n, dt=0.01),
        lambda n, _scene: BodyTorqueDisturbance(n_envs=n, dt=0.01),
        # Stateful (ou_process)
        lambda n, _scene: BodyForceDisturbance(n_envs=n, dt=0.01, distribution="ou_process"),
        lambda n, _scene: BodyTorqueDisturbance(n_envs=n, dt=0.01, distribution="ou_process"),
    ],
    ids=[
        "BodyForceDisturbance_uniform",
        "BodyTorqueDisturbance_uniform",
        "BodyForceDisturbance_ou",
        "BodyTorqueDisturbance_ou",
    ],
)
def perturbation(request, n_envs, mock_scene):
    """Parametrized fixture over all category-8 perturbation variants."""
    return request.param(n_envs, mock_scene)


@pytest.fixture(params=[BodyForceDisturbance, BodyTorqueDisturbance])
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
