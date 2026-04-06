"""Tests for category 6 — action perturbations (U1–U11, I1)."""

from unittest.mock import patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    PerturbationMode,
)
from genesis_robust_rl.perturbations.category_6_action import (
    ESCLowPassFilter,
)
from tests.conftest import assert_lipschitz

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ACTION_DIM = 4


@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param


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
def test_tick_reset(perturbation, n_envs):
    perturbation.tick(is_reset=True)
    if perturbation.frequency == "per_episode":
        assert perturbation._current_value is not None


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step(perturbation):
    perturbation.tick(is_reset=True)
    perturbation.tick(is_reset=False)
    if perturbation.frequency == "per_step":
        assert perturbation._current_value is not None


# ---------------------------------------------------------------------------
# U5 — set_value() Lipschitz
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() Lipschitz
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_params_lipschitz(perturbation):
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")


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
# U8 — Stateful persistence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    # For stateful action perturbations, apply actions to evolve state
    for i in range(10):
        perturbation.tick(is_reset=False)
        action = torch.randn(n_envs, ACTION_DIM) * (i + 1) * 0.1
        perturbation.apply(action)
    # State should have been affected
    if isinstance(perturbation, ESCLowPassFilter):
        assert not torch.allclose(
            perturbation._filtered,
            torch.zeros_like(perturbation._filtered),
            atol=1e-6,
        )


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
def test_output_shape_dtype(perturbation, n_envs):
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    v = perturbation.sample()
    assert v is not None
    assert v.dtype == torch.float32
    expected_n = 1 if perturbation.scope == "global" else n_envs
    assert v.shape[0] == expected_n


# ---------------------------------------------------------------------------
# U10 — Partial reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    for _ in range(10):
        perturbation.tick(is_reset=False)
        perturbation.apply(torch.randn(n_envs, ACTION_DIM))
    # Reset only env 0
    env_ids = torch.tensor([0])
    perturbation.reset(env_ids)


# ---------------------------------------------------------------------------
# U11 — Adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_sample:
        perturbation.tick(is_reset=False)
        assert mock_sample.call_count == 0


# ---------------------------------------------------------------------------
# I1 — apply() returns valid tensor
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_returns_valid_tensor(perturbation, n_envs):
    perturbation.tick(is_reset=True)
    action = torch.randn(n_envs, ACTION_DIM)
    result = perturbation.apply(action)
    assert result is not None
    assert result.shape == (n_envs, ACTION_DIM)
    assert result.dtype == torch.float32
    assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_action_noise_adds_noise(n_envs):
    """ActionNoise with non-zero sigma should change the action."""
    from genesis_robust_rl.perturbations.category_6_action import ActionNoise

    p = ActionNoise(n_envs=n_envs, dt=0.01)
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)
    p.tick(is_reset=False)
    action = torch.ones(n_envs, 4)
    p.apply(action)  # warm up
    # With non-zero noise, result should differ from input
    differs = False
    for _ in range(10):
        p.tick(is_reset=False)
        r = p.apply(action)
        if not torch.allclose(r, action, atol=1e-6):
            differs = True
            break
    assert differs, "ActionNoise did not add any noise"


@pytest.mark.unit
def test_deadzone_zeros_small_actions(n_envs):
    """ActionDeadzone should zero out small action components."""
    from genesis_robust_rl.perturbations.category_6_action import ActionDeadzone

    p = ActionDeadzone(n_envs=n_envs, dt=0.01, distribution_params={"low": 0.05, "high": 0.05})
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)
    action = torch.full((n_envs, 4), 0.01)  # below threshold
    result = p.apply(action)
    assert torch.allclose(result, torch.zeros_like(result))


@pytest.mark.unit
def test_saturation_clips_action(n_envs):
    """ActionSaturation should clip to reduced range."""
    from genesis_robust_rl.perturbations.category_6_action import ActionSaturation

    p = ActionSaturation(n_envs=n_envs, dt=0.01, distribution_params={"low": 0.5, "high": 0.5})
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)
    action = torch.ones(n_envs, 4) * 0.8
    result = p.apply(action)
    assert (result <= 0.5 + 1e-6).all()


@pytest.mark.unit
def test_esc_filter_smooths_step(n_envs):
    """ESCLowPassFilter should smooth a step input."""
    p = ESCLowPassFilter(
        n_envs=n_envs,
        dt=0.01,
        distribution_params={"low": 10.0, "high": 10.0},
    )
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)
    # Step input
    action = torch.ones(n_envs, 4)
    # First application should not reach full value
    result = p.apply(action)
    assert (result < action).any(), "Filter did not smooth step input"
    # After many applications, should converge
    for _ in range(200):
        result = p.apply(action)
    assert torch.allclose(result, action, atol=0.05)
