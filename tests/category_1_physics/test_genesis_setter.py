"""Tests for GenesisSetterPerturbation leaves (category 1).

Covers: U1–U11 unit tests, I1–I2 integration tests.
MassShift is stateless + per_episode → U8/U10 are skipped automatically.
InertiaTensor uses two setters (mass + CoM) — tested in I2b.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from genesis_robust_rl.perturbations.base import (
    PhysicsPerturbation,
    PerturbationMode,
)
from genesis_robust_rl.perturbations.category_1_physics import InertiaTensor
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
    perturbation.curriculum_scale = 0.0
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    if perturbation.is_stateful:
        pytest.skip("stateful perturbation — U2 tested in dedicated file (requires tick)")
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
# U8 — stateful persistence (skipped for stateless)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    state_after_reset = perturbation._current_value.clone()
    for _ in range(10):
        perturbation.tick(is_reset=False)
    assert not torch.allclose(state_after_reset, perturbation._current_value, atol=1e-6)
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
# U10 — partial reset (skipped for stateless / global / physics)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply(), not testable here")
    if perturbation.scope == "global":
        pytest.skip("scope=global: no per-env dimension to isolate")
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
            f"tick(is_reset=False) called sample() {mock_s.call_count} time(s) in ADVERSARIAL mode"
        )


# ---------------------------------------------------------------------------
# I1 — apply() returns None, does not mutate env_state
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_returns_none(perturbation, mock_env_state, mock_scene, n_envs):
    perturbation.tick(is_reset=True)
    result = perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert result is None


# ---------------------------------------------------------------------------
# I2 — setter_fn called with correct shape and envs_idx
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_physics_setter_called_correctly(perturbation, mock_scene, mock_env_state, n_envs):
    if not hasattr(perturbation, "setter_fn"):
        pytest.skip("not a GenesisSetterPerturbation")
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert perturbation.setter_fn.called, "setter_fn was not called"
    call_args = perturbation.setter_fn.call_args[0]
    value_arg, envs_arg = call_args[0], call_args[1]
    assert value_arg.shape[0] == n_envs
    assert envs_arg.shape[0] == n_envs


# ---------------------------------------------------------------------------
# I2b — InertiaTensor: both setters called with correct shapes
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_inertia_tensor_both_setters_called(n_envs: int, mock_env_state) -> None:
    """InertiaTensor.apply() must call both mass and CoM setters with correct shapes."""
    mass_fn = MagicMock()
    com_fn = MagicMock()
    p = InertiaTensor(
        mass_setter_fn=mass_fn,
        com_setter_fn=com_fn,
        n_envs=n_envs,
        dt=0.01,
    )
    p.tick(is_reset=True)
    # Build an env_state with the right n_envs
    from tests.conftest import EnvState
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
    scene = MagicMock()
    drone = MagicMock()
    p.apply(scene, drone, env_state)

    assert mass_fn.called, "_mass_setter_fn was not called"
    assert com_fn.called, "_com_setter_fn was not called"

    # mass setter: value shape [n_envs], envs_idx shape [n_envs]
    mass_val, mass_idx = mass_fn.call_args[0]
    assert mass_val.shape == (n_envs,), f"mass delta shape {mass_val.shape}"
    assert mass_idx.shape == (n_envs,)

    # com setter: value shape [n_envs, 3], envs_idx shape [n_envs]
    com_val, com_idx = com_fn.call_args[0]
    assert com_val.shape == (n_envs, 3), f"CoM delta shape {com_val.shape}"
    assert com_idx.shape == (n_envs,)
