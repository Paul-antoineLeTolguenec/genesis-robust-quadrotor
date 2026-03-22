"""Tests for 1.14 StructuralFlexibility.

Covers U1–U11, I1, I3. Stateless, per_episode, per_env, dimension=(2,).
Custom _draw() samples [k, b] independently from separate uniform ranges.
apply() calls apply_links_external_torque (not force).
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    PerturbationMode,
    PhysicsPerturbation,
)
from genesis_robust_rl.perturbations.category_1_physics import StructuralFlexibility
from tests.conftest import EnvState, assert_lipschitz

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param


@pytest.fixture
def mock_scene():
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    return scene


@pytest.fixture
def perturbation(n_envs):
    return StructuralFlexibility(n_envs=n_envs, dt=0.01)


@pytest.fixture
def mock_env_state(n_envs):
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


# ---------------------------------------------------------------------------
# U1 — sample() within bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    """Both k and b values must stay within bounds (0.0, 500.0)."""
    lo, hi = perturbation.bounds
    for _ in range(1000):
        v = perturbation.sample()
        assert v.shape == (n_envs, 2)
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(perturbation):
    """curriculum_scale=0 → [k, b] = nominal [0, 0] → zero torque."""
    perturbation.curriculum_scale = 0.0
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    """curriculum_scale=1 → [k, b] sampled from full range — non-zero."""
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
    """tick(is_reset=True) must set _current_value [n_envs, 2]."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    assert perturbation._current_value is not None
    assert perturbation._current_value.shape == (n_envs, 2)


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step_does_not_resample(perturbation):
    """tick(is_reset=False) must not resample for per_episode perturbation."""
    perturbation.tick(is_reset=True)
    v_before = perturbation._current_value.clone()
    perturbation.tick(is_reset=False)
    assert torch.allclose(perturbation._current_value, v_before)


# ---------------------------------------------------------------------------
# U5 — Lipschitz (skip — lipschitz_k=None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() Lipschitz (skip)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_params_lipschitz(perturbation):
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")


# ---------------------------------------------------------------------------
# U7 — get_privileged_obs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_privileged_obs_observable(perturbation, mock_env_state, mock_scene):
    """observable=True → get_privileged_obs() returns [k, b] vector after tick."""
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)
    # _current_value must still be [k, b], not the torque (apply() preserves it)
    assert obs.shape[-1] == 2


@pytest.mark.unit
def test_get_privileged_obs_not_observable(perturbation):
    """observable=False → get_privileged_obs() returns None."""
    perturbation.observable = False
    assert perturbation.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful (skipped — stateless)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(perturbation):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")


@pytest.mark.unit
def test_stateful_reset_all(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")


# ---------------------------------------------------------------------------
# U9 — output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs, mock_env_state, mock_scene):
    """_current_value after tick() + apply() must be [n_envs, 2] float32 ([k, b])."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    v = perturbation._current_value
    assert v is not None
    assert v.shape == (n_envs, 2), f"Expected ({n_envs}, 2), got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset (skipped — stateless PhysicsPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply()")


# ---------------------------------------------------------------------------
# U11 — adversarial mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    """In ADVERSARIAL mode, tick(is_reset=False) must never call sample()."""
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_s:
        perturbation.tick(is_reset=False)
        assert mock_s.call_count == 0


# ---------------------------------------------------------------------------
# I1 — apply() returns None
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_returns_none(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() must return None."""
    perturbation.tick(is_reset=True)
    result = perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert result is None


# ---------------------------------------------------------------------------
# I3 — apply() calls apply_links_external_torque (not force)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_calls_torque_not_force(perturbation, mock_scene, mock_env_state, n_envs):
    """apply() must call apply_links_external_torque (not apply_links_external_force)."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert mock_scene.rigid_solver.apply_links_external_torque.called, (
        "apply_links_external_torque was not called"
    )
    assert not mock_scene.rigid_solver.apply_links_external_force.called, (
        "apply_links_external_force should NOT be called for torque perturbation"
    )


@pytest.mark.integration
def test_torque_shape(perturbation, mock_scene, mock_env_state, n_envs):
    """Torque argument must have shape [n_envs, 3]."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    call_args = mock_scene.rigid_solver.apply_links_external_torque.call_args[0]
    torque_arg = call_args[0]
    assert torque_arg.shape == (n_envs, 3), (
        f"Expected torque shape ({n_envs}, 3), got {torque_arg.shape}"
    )


@pytest.mark.integration
def test_zero_kb_zero_torque(n_envs):
    """At k=0, b=0 (nominal): torque must be exactly zero."""
    scene = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    p = StructuralFlexibility(n_envs=n_envs, dt=0.01, curriculum_scale=0.0)
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.ones(n_envs, 3) * 0.5,
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    torque = scene.rigid_solver.apply_links_external_torque.call_args[0][0]
    assert torch.allclose(torque, torch.zeros_like(torque), atol=1e-8), (
        f"Expected zero torque at k=b=0, got: {torque}"
    )


@pytest.mark.integration
def test_kb_preserved_after_apply(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() must NOT overwrite _current_value ([k, b] must remain)."""
    perturbation.tick(is_reset=True)
    kb_before = perturbation._current_value.clone()
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert torch.allclose(perturbation._current_value, kb_before), (
        "apply() overwrote _current_value — [k, b] vector was lost"
    )


# ---------------------------------------------------------------------------
# Perf — tick and apply overhead (CPU)
# ---------------------------------------------------------------------------


WARMUP = 200
STEPS = 2000
MAX_TICK_MS = 0.1
MAX_APPLY_MS = 0.05


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_tick_overhead_cpu(n_envs: int) -> None:
    """StructuralFlexibility tick() must stay under MAX_TICK_MS on CPU."""
    p = StructuralFlexibility(n_envs=n_envs, dt=0.01)
    for _ in range(WARMUP):
        p.tick(is_reset=True)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=True)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_TICK_MS, (
        f"StructuralFlexibility tick() too slow at n_envs={n_envs}: "
        f"{elapsed_ms:.4f} ms/step (limit {MAX_TICK_MS} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_apply_overhead_cpu(n_envs: int) -> None:
    """StructuralFlexibility apply() must stay under MAX_APPLY_MS on CPU."""
    scene = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    p = StructuralFlexibility(n_envs=n_envs, dt=0.01)
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
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_APPLY_MS, (
        f"StructuralFlexibility apply() too slow at n_envs={n_envs}: "
        f"{elapsed_ms:.4f} ms/step (limit {MAX_APPLY_MS} ms)"
    )
