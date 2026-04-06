"""Fixtures for category 7 — payload & configuration perturbations."""

import pytest

from genesis_robust_rl.perturbations.category_7_payload import (
    AsymmetricPropGuardDrag,
    PayloadCOMOffset,
    PayloadMass,
)


@pytest.fixture(
    params=[
        lambda n, scene: PayloadMass(setter_fn=scene.drone.set_links_mass_shift, n_envs=n, dt=0.01),
        lambda n, scene: PayloadCOMOffset(
            setter_fn=scene.drone.set_links_COM_shift, n_envs=n, dt=0.01
        ),
        lambda n, scene: AsymmetricPropGuardDrag(n_envs=n, dt=0.01),
    ]
)
def perturbation(request, n_envs, mock_scene):
    """Parametrized fixture over all category-7 perturbation leaves."""
    return request.param(n_envs, mock_scene)


@pytest.fixture(params=[PayloadMass, PayloadCOMOffset, AsymmetricPropGuardDrag])
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
