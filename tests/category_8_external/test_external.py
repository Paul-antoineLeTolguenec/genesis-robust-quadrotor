"""Tests for category 8 — external disturbances.

Covers: U1–U11 unit tests, I1–I3 integration tests.
Both BodyForceDisturbance and BodyTorqueDisturbance are tested in both
stateless (uniform) and stateful (ou_process) variants.
"""

from unittest.mock import patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PerturbationMode,
    PhysicsPerturbation,
)
from tests.conftest import assert_lipschitz

# ---------------------------------------------------------------------------
# U1 — sample() output within bounds (1000 draws)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    lo, hi = perturbation.bounds
    for _ in range(1000):
        v = perturbation.sample()
        assert v.shape[0] in (n_envs, 1)
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(perturbation):
    if perturbation.is_stateful:
        pytest.skip("OU mode: curriculum_scale affects sigma at reset, not sample()")
    perturbation.curriculum_scale = 0.0
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    if perturbation.is_stateful:
        pytest.skip("OU mode: curriculum_scale affects sigma at reset, not sample()")
    perturbation.curriculum_scale = 1.0
    samples = torch.stack([perturbation.sample() for _ in range(500)])
    nominal = torch.tensor(perturbation.nominal, dtype=samples.dtype)
    assert not torch.allclose(samples, nominal.expand_as(samples), atol=1e-4), (
        "curriculum_scale=1 produced only nominal values"
    )


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_reset_samples_per_episode(perturbation, n_envs):
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True, env_ids=env_ids)
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
# U5 — set_value() enforces Lipschitz
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
    huge = {key: val + 1e6 for key, val in original.items() if isinstance(val, (int, float))}
    if not huge:
        pytest.skip("no scalar distribution params")
    # Initialize _params_prev so Lipschitz clipping kicks in
    perturbation.update_params(dict(original))
    perturbation.update_params(huge)
    for key, orig_val in original.items():
        if key in huge:
            delta = abs(perturbation.distribution_params[key] - orig_val)
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
    assert perturbation.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful persistence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    state_after_reset = perturbation._current_value.clone()
    for _ in range(10):
        perturbation.tick(is_reset=False)
    assert not torch.allclose(state_after_reset, perturbation._current_value, atol=1e-6), (
        "Stateful perturbation state did not change over 10 steps"
    )
    perturbation.reset(torch.tensor([0]))


@pytest.mark.unit
def test_stateful_reset_all(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    perturbation.reset(torch.arange(n_envs))


# ---------------------------------------------------------------------------
# U9 — output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs, mock_env_state, mock_scene):
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    if isinstance(perturbation, PhysicsPerturbation):
        perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
        v = perturbation._current_value
    else:
        v = perturbation.sample()
    assert v is not None
    if perturbation.scope == "global":
        expected = (1,) + tuple(perturbation.dimension)
    else:
        expected = (n_envs,) + tuple(perturbation.dimension)
    assert v.shape == expected, f"Expected {expected}, got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply()")
    if perturbation.scope == "global":
        pytest.skip("scope=global")
    perturbation.tick(is_reset=True)
    for _ in range(10):
        perturbation.tick(is_reset=False)
    state_before = perturbation._current_value.clone()
    perturbation.reset(torch.tensor([0]))
    assert torch.allclose(perturbation._current_value[1:], state_before[1:], atol=1e-6)


# ---------------------------------------------------------------------------
# U11 — adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_s:
        perturbation.tick(is_reset=False)
        assert mock_s.call_count == 0, (
            f"sample() called {mock_s.call_count} time(s) in ADVERSARIAL mode"
        )


# ---------------------------------------------------------------------------
# I1 — apply() returns None
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_returns_none(perturbation, mock_env_state, mock_scene, n_envs):
    perturbation.tick(is_reset=True)
    result = perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert result is None


# ---------------------------------------------------------------------------
# I2 — setter_fn called (skip — Cat 8 are ExternalWrenchPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_physics_setter_called_correctly(perturbation, mock_scene, mock_env_state, n_envs):
    if not hasattr(perturbation, "setter_fn"):
        pytest.skip("not a GenesisSetterPerturbation")


# ---------------------------------------------------------------------------
# I3 — ExternalWrenchPerturbation.apply() calls solver
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrench_apply_calls_solver(perturbation, mock_scene, mock_env_state, n_envs):
    if not isinstance(perturbation, ExternalWrenchPerturbation):
        pytest.skip("not an ExternalWrenchPerturbation")
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    called = (
        mock_scene.rigid_solver.apply_links_external_force.called
        or mock_scene.rigid_solver.apply_links_external_torque.called
    )
    assert called, "solver.apply_links_external_force/torque was not called"


# ---------------------------------------------------------------------------
# I3b — force vs torque dispatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrench_dispatch_force_vs_torque(perturbation, mock_scene, mock_env_state, n_envs):
    """Verify that force perturbations call apply_links_external_force
    and torque perturbations call apply_links_external_torque."""
    if not isinstance(perturbation, ExternalWrenchPerturbation):
        pytest.skip("not an ExternalWrenchPerturbation")
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    if perturbation.wrench_type == "force":
        assert mock_scene.rigid_solver.apply_links_external_force.called
    else:
        assert mock_scene.rigid_solver.apply_links_external_torque.called
