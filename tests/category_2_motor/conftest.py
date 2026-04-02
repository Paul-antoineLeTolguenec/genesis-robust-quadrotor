"""Fixtures for category 2 — motor perturbations (all 13 classes)."""

import pytest

from genesis_robust_rl.perturbations.category_2_motor import (
    GyroscopicEffect,
    MotorBackEMF,
    MotorColdStart,
    MotorKill,
    MotorLag,
    MotorPartialFailure,
    MotorRPMNoise,
    MotorSaturation,
    MotorWear,
    PropellerThrustAsymmetry,
    RotorImbalance,
    ThrustCoefficientKF,
    TorqueCoefficientKM,
)


@pytest.fixture(
    params=[
        # ExternalWrenchPerturbation leaves (need mock_scene but don't use it)
        lambda n, scene: ThrustCoefficientKF(n_envs=n, dt=0.01),
        lambda n, scene: TorqueCoefficientKM(n_envs=n, dt=0.01),
        lambda n, scene: PropellerThrustAsymmetry(n_envs=n, dt=0.01),
        lambda n, scene: MotorPartialFailure(n_envs=n, dt=0.01),
        lambda n, scene: MotorBackEMF(n_envs=n, dt=0.01),
        lambda n, scene: GyroscopicEffect(n_envs=n, dt=0.01),
        # MotorCommandPerturbation stateless leaves
        lambda n, scene: MotorKill(n_envs=n, dt=0.01),
        lambda n, scene: MotorRPMNoise(n_envs=n, dt=0.01),
        lambda n, scene: MotorSaturation(n_envs=n, dt=0.01),
        # MotorCommandPerturbation stateful leaves
        lambda n, scene: MotorLag(n_envs=n, dt=0.01),
        lambda n, scene: MotorWear(n_envs=n, dt=0.01),
        lambda n, scene: RotorImbalance(n_envs=n, dt=0.01),
        lambda n, scene: MotorColdStart(n_envs=n, dt=0.01),
    ]
)
def perturbation(request, n_envs, mock_scene):
    """Parametrized fixture over all 13 category-2 perturbation leaves."""
    return request.param(n_envs, mock_scene)


@pytest.fixture(
    params=[
        ThrustCoefficientKF,
        TorqueCoefficientKM,
        PropellerThrustAsymmetry,
        MotorPartialFailure,
        MotorBackEMF,
        GyroscopicEffect,
        MotorKill,
        MotorRPMNoise,
        MotorSaturation,
        MotorLag,
        MotorWear,
        RotorImbalance,
        MotorColdStart,
    ]
)
def perturbation_class(request):
    """Used by P3 memory test."""
    return request.param
