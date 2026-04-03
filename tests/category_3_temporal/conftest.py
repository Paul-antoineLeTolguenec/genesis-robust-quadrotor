"""Fixtures for category 3 — temporal / latency perturbations."""

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


@pytest.fixture(
    params=[
        lambda n: ObsFixedDelay(obs_slice=slice(0, 6), obs_dim=6, n_envs=n, dt=0.01),
        lambda n: ObsVariableDelay(obs_slice=slice(0, 6), obs_dim=6, n_envs=n, dt=0.01),
        lambda n: ActionFixedDelay(n_envs=n, dt=0.01, action_dim=4),
        lambda n: ActionVariableDelay(n_envs=n, dt=0.01, action_dim=4),
        lambda n: PacketLoss(n_envs=n, dt=0.01, action_dim=4),
        lambda n: ComputationOverload(n_envs=n, dt=0.01, action_dim=4),
    ]
)
def perturbation(request, n_envs):
    """Parametrized fixture over all category-3 perturbation leaves."""
    return request.param(n_envs)


@pytest.fixture(
    params=[
        ObsFixedDelay,
        ObsVariableDelay,
        ActionFixedDelay,
        ActionVariableDelay,
        PacketLoss,
        ComputationOverload,
    ]
)
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
