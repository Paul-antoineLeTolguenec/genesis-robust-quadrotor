"""Tests for ExternalWrenchPerturbation leaves (category 2 — motor).

Covers: U1-U11 unit tests, I1, I3 integration tests.
Perturbations tested: ThrustCoefficientKF, TorqueCoefficientKM,
PropellerThrustAsymmetry, MotorPartialFailure, MotorBackEMF, GyroscopicEffect.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PerturbationMode,
    PhysicsPerturbation,
)
from genesis_robust_rl.perturbations.category_2_motor import (
    GyroscopicEffect,
    MotorBackEMF,
    MotorPartialFailure,
    PropellerThrustAsymmetry,
    ThrustCoefficientKF,
    TorqueCoefficientKM,
)
from tests.conftest import EnvState, assert_lipschitz

# -- Wrench-only fixture (subset of the 13) --

_WRENCH_FACTORIES = [
    lambda n: ThrustCoefficientKF(n_envs=n, dt=0.01),
    lambda n: TorqueCoefficientKM(n_envs=n, dt=0.01),
    lambda n: PropellerThrustAsymmetry(n_envs=n, dt=0.01),
    lambda n: MotorPartialFailure(n_envs=n, dt=0.01),
    lambda n: MotorBackEMF(n_envs=n, dt=0.01),
    lambda n: GyroscopicEffect(n_envs=n, dt=0.01),
]


@pytest.fixture(params=_WRENCH_FACTORIES)
def wrench_pert(request, n_envs):
    return request.param(n_envs)


# ---------------------------------------------------------------------------
# U1 — sample() output within bounds (1000 draws)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(wrench_pert, n_envs):
    lo, hi = wrench_pert.bounds
    for _ in range(1000):
        v = wrench_pert.sample()
        assert v.shape[0] in (n_envs, 1)
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.6e}, max={v.max():.6e}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(wrench_pert):
    wrench_pert.curriculum_scale = 0.0
    v = wrench_pert.sample()
    nominal = torch.tensor(wrench_pert.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(wrench_pert):
    wrench_pert.curriculum_scale = 1.0
    samples = torch.stack([wrench_pert.sample() for _ in range(500)])
    nominal = torch.tensor(wrench_pert.nominal, dtype=samples.dtype)
    assert not torch.allclose(
        samples,
        nominal.expand_as(samples),
        atol=0.0,
        rtol=1e-3,
    ), "curriculum_scale=1 produced only nominal values"


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_reset_samples_per_episode(wrench_pert, n_envs):
    env_ids = torch.arange(n_envs)
    wrench_pert.tick(is_reset=True, env_ids=env_ids)
    if wrench_pert.frequency == "per_episode":
        assert wrench_pert._current_value is not None


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step_advances(wrench_pert):
    wrench_pert.tick(is_reset=True)
    wrench_pert.tick(is_reset=False)
    if wrench_pert.frequency == "per_step":
        assert wrench_pert._current_value is not None


# ---------------------------------------------------------------------------
# U5 — set_value() enforces Lipschitz
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(wrench_pert):
    wrench_pert.tick(
        is_reset=True,
        env_ids=torch.arange(wrench_pert.n_envs),
    )
    assert_lipschitz(wrench_pert, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() enforces Lipschitz on distribution params
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_params_lipschitz(wrench_pert):
    if wrench_pert.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")


# ---------------------------------------------------------------------------
# U7 — get_privileged_obs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_privileged_obs_observable(
    wrench_pert,
    n_envs,
    mock_env_state,
    mock_scene,
):
    wrench_pert.observable = True
    wrench_pert.tick(is_reset=True)
    if wrench_pert.frequency == "per_step":
        wrench_pert.tick(is_reset=False)
    wrench_pert.apply(mock_scene, mock_scene.drone, mock_env_state)
    obs = wrench_pert.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_get_privileged_obs_not_observable(wrench_pert):
    wrench_pert.observable = False
    assert wrench_pert.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful persistence (all wrench are stateless -> skip)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(wrench_pert, n_envs):
    if not wrench_pert.is_stateful:
        pytest.skip("stateless perturbation")


# ---------------------------------------------------------------------------
# U9 — output shape and dtype (via apply())
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_shape_dtype(
    wrench_pert,
    n_envs,
    mock_env_state,
    mock_scene,
):
    wrench_pert.tick(is_reset=True, env_ids=torch.arange(n_envs))
    if wrench_pert.frequency == "per_step":
        wrench_pert.tick(is_reset=False)
    wrench_pert.apply(mock_scene, mock_scene.drone, mock_env_state)
    if wrench_pert.wrench_type == "torque":
        solver_fn = mock_scene.rigid_solver.apply_links_external_torque
    else:
        solver_fn = mock_scene.rigid_solver.apply_links_external_force
    assert solver_fn.called
    wrench = solver_fn.call_args[0][0]
    assert wrench.shape == (n_envs, 3)
    assert wrench.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset (skipped for stateless physics)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(wrench_pert, n_envs):
    if not wrench_pert.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(wrench_pert, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply()")


# ---------------------------------------------------------------------------
# U11 — adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(wrench_pert):
    wrench_pert.tick(is_reset=True)
    wrench_pert.mode = PerturbationMode.ADVERSARIAL
    with patch.object(
        type(wrench_pert),
        "sample",
        wraps=wrench_pert.sample,
    ) as mock_s:
        wrench_pert.tick(is_reset=False)
        assert mock_s.call_count == 0


# ---------------------------------------------------------------------------
# I1 — apply() returns None
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_returns_none(
    wrench_pert,
    mock_env_state,
    mock_scene,
    n_envs,
):
    wrench_pert.tick(is_reset=True)
    if wrench_pert.frequency == "per_step":
        wrench_pert.tick(is_reset=False)
    result = wrench_pert.apply(
        mock_scene,
        mock_scene.drone,
        mock_env_state,
    )
    assert result is None


# ---------------------------------------------------------------------------
# I3 — apply() calls solver with correct tensor
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrench_apply_calls_solver(
    wrench_pert,
    mock_scene,
    mock_env_state,
    n_envs,
):
    assert isinstance(wrench_pert, ExternalWrenchPerturbation)
    wrench_pert.tick(is_reset=True)
    if wrench_pert.frequency == "per_step":
        wrench_pert.tick(is_reset=False)
    wrench_pert.apply(mock_scene, mock_scene.drone, mock_env_state)
    called = (
        mock_scene.rigid_solver.apply_links_external_force.called
        or mock_scene.rigid_solver.apply_links_external_torque.called
    )
    assert called


# ===========================================================================
# Physics-model specific tests
# ===========================================================================


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_thrust_kf_only_z(n_envs: int) -> None:
    """ThrustCoefficientKF force must have only Z nonzero."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = ThrustCoefficientKF(n_envs=n_envs, dt=0.01)
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    assert torch.allclose(wrench[:, :2], torch.zeros(n_envs, 2), atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_torque_km_only_z(n_envs: int) -> None:
    """TorqueCoefficientKM torque must have only Z nonzero."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    p = TorqueCoefficientKM(n_envs=n_envs, dt=0.01)
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    wrench = scene.rigid_solver.apply_links_external_torque.call_args[0][0]
    assert torch.allclose(wrench[:, :2], torch.zeros(n_envs, 2), atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_partial_failure_reduces_thrust(n_envs: int) -> None:
    """MotorPartialFailure with efficiency < 1 must produce negative dF_z."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = MotorPartialFailure(
        n_envs=n_envs,
        dt=0.01,
        distribution="uniform",
        distribution_params={"low": 0.3, "high": 0.5},
    )
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    assert (wrench[:, 2] < 0).all()


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_nominal_produces_zero_wrench(n_envs: int) -> None:
    """At nominal values, all wrench classes produce zero force/torque."""
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    classes = [
        ThrustCoefficientKF,
        TorqueCoefficientKM,
        PropellerThrustAsymmetry,
        MotorPartialFailure,
        MotorBackEMF,
        GyroscopicEffect,
    ]
    for cls in classes:
        p = cls(n_envs=n_envs, dt=0.01)
        nominal = torch.tensor(p.nominal, dtype=torch.float32)
        p._current_value = nominal.expand(p._batch_shape())
        wrench = p._compute_wrench(env_state)
        assert torch.allclose(wrench, torch.zeros(n_envs, 3), atol=1e-10), (
            f"{cls.__name__} at nominal should produce zero wrench"
        )


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_gyroscopic_zero_angvel(n_envs: int) -> None:
    """GyroscopicEffect must produce zero torque at zero ang_vel."""
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p = GyroscopicEffect(n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)
    wrench = p._compute_wrench(env_state)
    assert torch.allclose(wrench, torch.zeros(n_envs, 3), atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_back_emf_zero_rpm(n_envs: int) -> None:
    """MotorBackEMF must produce zero torque at zero RPM."""
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.zeros(n_envs, 4),
        dt=0.01,
        step=0,
    )
    p = MotorBackEMF(n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)
    p._current_value = torch.full(p._batch_shape(), 0.015)
    wrench = p._compute_wrench(env_state)
    assert torch.allclose(wrench, torch.zeros(n_envs, 3), atol=1e-10)
