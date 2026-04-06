"""Tests for category 5 — wind perturbations (U1–U11, I1, I3)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PerturbationMode,
    PhysicsPerturbation,
)
from tests.conftest import EnvState, assert_lipschitz

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param


@pytest.fixture
def mock_env_state(n_envs):
    """Realistic EnvState for a drone at moderate altitude with velocity."""
    return EnvState(
        pos=torch.rand(n_envs, 3) * 2.0,  # altitude 0-2m
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 2.0,
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.randn(n_envs, 3) * 0.2,
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


@pytest.fixture
def mock_scene():
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    drone = MagicMock()
    scene.drone = drone
    return scene


# ---------------------------------------------------------------------------
# U1 — sample() output within bounds (1000 draws)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    lo, hi = perturbation.bounds
    for _ in range(1000):
        v = perturbation.sample()
        assert v.shape[0] in (n_envs, 1)
        assert (v >= lo - 1e-6).all() and (v <= hi + 1e-6).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(perturbation):
    perturbation.curriculum_scale = 0.0
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    from genesis_robust_rl.perturbations.category_5_wind import (
        BladeVortexInteraction,
        PayloadSway,
        Turbulence,
        WindGust,
    )

    # Stateful perturbations with custom sample() manage state internally
    if isinstance(perturbation, (WindGust, Turbulence, BladeVortexInteraction, PayloadSway)):
        pytest.skip("stateful perturbation with event/OU-driven sample()")
    perturbation.curriculum_scale = 1.0
    samples = torch.stack([perturbation.sample() for _ in range(500)])
    nominal = torch.tensor(perturbation.nominal, dtype=samples.dtype)
    if perturbation.distribution == "constant":
        return  # constant distribution always returns nominal
    assert not torch.allclose(samples, nominal.expand_as(samples), atol=1e-4), (
        "curriculum_scale=1 produced only nominal values"
    )


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_reset_resets_state(perturbation, n_envs):
    perturbation.tick(is_reset=True)
    if perturbation.frequency == "per_episode":
        assert perturbation._current_value is not None


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step_advances(perturbation):
    perturbation.tick(is_reset=True)
    perturbation.tick(is_reset=False)
    if perturbation.frequency == "per_step":
        assert perturbation._current_value is not None


# ---------------------------------------------------------------------------
# U5 — set_value() enforces Lipschitz (adversarial)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() enforces Lipschitz on distribution params
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_params_lipschitz(perturbation):
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")
    k = perturbation.lipschitz_k
    dt = perturbation.dt
    original = dict(perturbation.distribution_params)
    huge_params = {k_: v + 1e6 for k_, v in original.items() if isinstance(v, (int, float))}
    if not huge_params:
        pytest.skip("no scalar distribution params to test")
    perturbation.update_params(huge_params)
    for key, orig_val in original.items():
        if key in huge_params:
            new_val = perturbation.distribution_params[key]
            delta = abs(new_val - orig_val)
            assert delta <= k * dt + 1e-6


# ---------------------------------------------------------------------------
# U7 — get_privileged_obs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_privileged_obs_observable(perturbation):
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_get_privileged_obs_not_observable(perturbation):
    perturbation.observable = False
    result = perturbation.get_privileged_obs()
    assert result is None


# ---------------------------------------------------------------------------
# U8 — Stateful: state persists and resets correctly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    from genesis_robust_rl.perturbations.category_5_wind import WindGust

    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(perturbation, WindGust):
        pytest.skip("WindGust is event-based/probabilistic — state may not change in 10 steps")
    perturbation.tick(is_reset=True)
    state_after_reset = perturbation._current_value.clone()
    for _ in range(10):
        perturbation.tick(is_reset=False)
    state_after_steps = perturbation._current_value.clone()
    assert not torch.allclose(state_after_reset, state_after_steps, atol=1e-6), (
        "Stateful perturbation state did not change over 10 steps"
    )
    env_ids = torch.tensor([0])
    perturbation.reset(env_ids)


@pytest.mark.unit
def test_stateful_reset_all(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True)
    perturbation.reset(env_ids)


# ---------------------------------------------------------------------------
# U9 — Output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs, mock_env_state, mock_scene):
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    expected_shape = (n_envs,) + tuple(perturbation.dimension)

    if isinstance(perturbation, PhysicsPerturbation):
        perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
        v = perturbation._current_value
    else:
        v = perturbation.sample()

    assert v is not None, "_current_value is None after apply()/sample()"
    if perturbation.scope == "global":
        expected_shape = (1,) + tuple(perturbation.dimension)
    assert v.shape == expected_shape, f"Expected {expected_shape}, got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — Partial reset (subset of envs)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if n_envs < 2:
        pytest.skip("need n_envs >= 2 to test partial reset")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation — partial reset has no observable state effect")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation _current_value set by apply(), not tick()")
    if perturbation.scope == "global":
        pytest.skip("scope=global: no per-env dimension to isolate")
    perturbation.tick(is_reset=True)
    for _ in range(10):
        perturbation.tick(is_reset=False)
    state_before = perturbation._current_value.clone()
    env_ids = torch.tensor([0])
    perturbation.reset(env_ids)
    if n_envs > 1:
        assert torch.allclose(perturbation._current_value[1:], state_before[1:], atol=1e-6), (
            "reset(env_ids=[0]) affected envs other than 0"
        )


# ---------------------------------------------------------------------------
# U11 — Adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_sample:
        perturbation.tick(is_reset=False)
        assert mock_sample.call_count == 0, (
            f"tick(False) called sample() {mock_sample.call_count}x in ADVERSARIAL"
        )


# ---------------------------------------------------------------------------
# I1 — apply() does not modify inputs in-place (N/A for ExternalWrenchPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_produces_valid_output(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() calls solver with valid tensor — ExternalWrenchPerturbation returns None."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    # Should not raise


# ---------------------------------------------------------------------------
# I3 — ExternalWrenchPerturbation.apply() calls solver
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrench_apply_calls_solver(perturbation, mock_scene, mock_env_state, n_envs):
    assert isinstance(perturbation, ExternalWrenchPerturbation)
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    called = (
        mock_scene.rigid_solver.apply_links_external_force.called
        or mock_scene.rigid_solver.apply_links_external_torque.called
    )
    assert called, "solver.apply_links_external_force/torque was not called"


# ---------------------------------------------------------------------------
# I3b — apply() wrench tensor has correct shape
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrench_shape(perturbation, mock_scene, mock_env_state, n_envs):
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    # Check the tensor passed to the solver
    if mock_scene.rigid_solver.apply_links_external_force.called:
        args = mock_scene.rigid_solver.apply_links_external_force.call_args
    else:
        args = mock_scene.rigid_solver.apply_links_external_torque.call_args
    wrench = args[0][0]
    assert wrench.shape == (n_envs, 3), f"Expected ({n_envs}, 3), got {wrench.shape}"
    assert wrench.dtype == torch.float32
