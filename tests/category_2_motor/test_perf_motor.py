"""P1 — GPU performance test for category 2 wrench perturbations."""

import time

import pytest
import torch

from genesis_robust_rl.perturbations.category_2_motor import (
    GyroscopicEffect,
    MotorBackEMF,
    MotorPartialFailure,
    PropellerThrustAsymmetry,
    ThrustCoefficientKF,
    TorqueCoefficientKM,
)
from tests.conftest import EnvState

PERF_N_ENVS = 512
PERF_WARMUP = 100
PERF_STEPS = 1000
MAX_MS_PER_STEP = 0.5


def _make_env_state(n_envs: int, device: str = "cpu") -> EnvState:
    return EnvState(
        pos=torch.zeros(n_envs, 3, device=device),
        quat=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device=device,
        ).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3, device=device) * 2.0,
        ang_vel=torch.randn(n_envs, 3, device=device) * 0.1,
        acc=torch.zeros(n_envs, 3, device=device),
        rpm=torch.ones(n_envs, 4, device=device) * 3000.0,
        dt=0.01,
        step=0,
    )


@pytest.mark.perf
@pytest.mark.parametrize(
    "cls",
    [
        ThrustCoefficientKF,
        TorqueCoefficientKM,
        PropellerThrustAsymmetry,
        MotorPartialFailure,
        MotorBackEMF,
        GyroscopicEffect,
    ],
)
def test_compute_wrench_overhead(cls: type) -> None:
    """_compute_wrench() overhead < 0.5 ms/step on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    p = cls(n_envs=PERF_N_ENVS, dt=0.01)
    p.tick(is_reset=True)
    env_state = _make_env_state(PERF_N_ENVS, device="cuda")

    for _ in range(PERF_WARMUP):
        p._compute_wrench(env_state)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        p._compute_wrench(env_state)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS

    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"{cls.__name__}._compute_wrench() too slow: "
        f"{elapsed_ms:.3f} ms/step (limit {MAX_MS_PER_STEP} ms)"
    )
