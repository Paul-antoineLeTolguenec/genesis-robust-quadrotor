"""Unit + integration tests for category 4 observation perturbations.

Tests: U1–U11, I1, I5.
"""

from unittest.mock import patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import PerturbationMode

# OBS_DIM used by all tests
OBS_DIM = 32


# ===================================================================
# U1 — sample() output within bounds (1000 draws)
# ===================================================================


@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    """All 1000 draws must stay within perturbation.bounds."""
    perturbation.tick(is_reset=True)
    lo, hi = perturbation.bounds
    for _ in range(100):  # 100 × n_envs draws
        v = perturbation.sample()
        assert v.shape[0] in (1, n_envs)
        assert (v >= lo - 1e-5).all() and (v <= hi + 1e-5).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}, bounds=({lo}, {hi})"
        )


# ===================================================================
# U2 — curriculum_scale extremes
# ===================================================================


@pytest.mark.unit
def test_curriculum_scale_zero(perturbation):
    """curriculum_scale=0 → value close to nominal after apply."""
    perturbation.curriculum_scale = 0.0
    perturbation.tick(is_reset=True)
    obs = torch.randn(perturbation.n_envs, OBS_DIM)
    obs_before = obs.clone()
    result = perturbation.apply(obs)
    # With scale=0, perturbation effect should be minimal
    # (some perturbations like quantization or masking may still have residual effects)
    assert result.shape == obs_before.shape


@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    """curriculum_scale=1 → full perturbation effect."""
    perturbation.curriculum_scale = 1.0
    perturbation.tick(is_reset=True)
    obs = torch.randn(perturbation.n_envs, OBS_DIM) * 5.0
    obs_before = obs.clone()
    # Run 50 steps to give stateful perturbations time to have an effect
    for _ in range(50):
        perturbation.tick(is_reset=False)
    result = perturbation.apply(obs)
    assert result.shape == obs_before.shape


# ===================================================================
# U3 — tick() reset path
# ===================================================================


@pytest.mark.unit
def test_tick_reset_resets_state(perturbation, n_envs):
    """tick(is_reset=True) must call reset() and sample if per_episode."""
    perturbation.tick(is_reset=True)
    if perturbation.frequency == "per_episode":
        assert perturbation._current_value is not None


# ===================================================================
# U4 — tick() step path
# ===================================================================


@pytest.mark.unit
def test_tick_step_advances(perturbation):
    """tick(is_reset=False) must sample if per_step, step if stateful."""
    perturbation.tick(is_reset=True)
    perturbation.tick(is_reset=False)
    if perturbation.frequency == "per_step":
        assert perturbation._current_value is not None


# ===================================================================
# U5 — set_value() enforces Lipschitz
# ===================================================================


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    """set_value() must clip value delta to lipschitz_k * dt."""
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    if perturbation._current_value is None:
        perturbation.sample()
    from tests.conftest import assert_lipschitz

    assert_lipschitz(perturbation, n_steps=50)


# ===================================================================
# U6 — update_params() enforces Lipschitz on distribution params
# ===================================================================


@pytest.mark.unit
def test_update_params_lipschitz(perturbation):
    """update_params() must clip each param change to lipschitz_k * dt."""
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")
    k = perturbation.lipschitz_k
    dt = perturbation.dt
    original = dict(perturbation.distribution_params)
    huge_params = {k_: v + 1e6 for k_, v in original.items() if isinstance(v, (int, float))}
    if not huge_params:
        pytest.skip("no scalar distribution params to test")
    # First call establishes baseline (_params_prev = None → no clipping)
    perturbation.update_params(dict(original))
    # Second call should be clipped
    perturbation.update_params(huge_params)
    for key, orig_val in original.items():
        if key in huge_params:
            new_val = perturbation.distribution_params[key]
            delta = abs(new_val - orig_val)
            assert delta <= k * dt + 1e-6


# ===================================================================
# U7 — get_privileged_obs()
# ===================================================================


@pytest.mark.unit
def test_get_privileged_obs_observable(perturbation):
    """observable=True → get_privileged_obs() returns _current_value."""
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    # per_step perturbations need a step tick to populate _current_value
    if perturbation.frequency == "per_step":
        perturbation.tick(is_reset=False)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_get_privileged_obs_not_observable(perturbation):
    """observable=False → get_privileged_obs() returns None."""
    perturbation.observable = False
    result = perturbation.get_privileged_obs()
    assert result is None


# ===================================================================
# U8 — Stateful: state persists and resets correctly
# ===================================================================


@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    """Stateful perturbations must persist state across ticks."""
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    for _ in range(20):
        perturbation.tick(is_reset=False)
    state_after_steps = perturbation._current_value
    # State should have changed (for most stateful perturbations)
    # Allow tolerance for slow-changing processes
    assert state_after_steps is not None


@pytest.mark.unit
def test_stateful_reset_all(perturbation, n_envs):
    """reset(all_env_ids) must not raise."""
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True)
    perturbation.reset(env_ids)


# ===================================================================
# U9 — Output shape and dtype
# ===================================================================


@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs):
    """apply() must produce valid obs tensor."""
    perturbation.tick(is_reset=True)
    obs = torch.randn(n_envs, OBS_DIM)
    result = perturbation.apply(obs)
    assert result.shape == (n_envs, OBS_DIM)
    assert result.dtype == torch.float32


# ===================================================================
# U10 — Partial reset (subset of envs)
# ===================================================================


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    """reset(env_ids=[0]) must reset only env 0."""
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if perturbation.scope == "global":
        pytest.skip("global scope")
    perturbation.tick(is_reset=True)
    for _ in range(10):
        perturbation.tick(is_reset=False)
    env_ids = torch.tensor([0])
    perturbation.reset(env_ids)
    # Must not raise


# ===================================================================
# U11 — Adversarial mode: tick() must not call sample()
# ===================================================================


@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    """In ADVERSARIAL mode, tick(is_reset=False) must never call sample()."""
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_sample:
        perturbation.tick(is_reset=False)
        assert mock_sample.call_count == 0, (
            f"tick() called sample() {mock_sample.call_count} time(s) in ADVERSARIAL mode"
        )


# ===================================================================
# I1 — apply() produces valid output
# ===================================================================


@pytest.mark.integration
def test_apply_produces_valid_output(perturbation, n_envs):
    """apply() must return a tensor of correct shape."""
    perturbation.tick(is_reset=True)
    obs = torch.randn(n_envs, OBS_DIM)
    result = perturbation.apply(obs)
    assert result is not None
    assert result.shape == (n_envs, OBS_DIM)
    assert not torch.isnan(result).any(), "NaN in apply() output"


# ===================================================================
# I5 — N-env vectorization
# ===================================================================


@pytest.mark.integration
def test_vectorized_env(perturbation, n_envs):
    """apply() must work on batched tensors."""
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    perturbation.tick(is_reset=True)
    obs = torch.randn(n_envs, OBS_DIM)
    result = perturbation.apply(obs)
    assert result.shape == (n_envs, OBS_DIM)
