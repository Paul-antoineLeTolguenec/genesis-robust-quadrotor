"""Tests for 1.13 PropellerBladeDamage.

Covers U1–U11, I1, I3. Stateless, per_episode, per_env, dimension=(4,).
Supports uniform and beta distributions.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PhysicsPerturbation,
    PerturbationMode,
)
from genesis_robust_rl.perturbations.category_1_physics import PropellerBladeDamage
from tests.conftest import assert_lipschitz, EnvState


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
    scene.rigid_solver.apply_links_external_force = MagicMock()
    return scene


@pytest.fixture
def perturbation(n_envs):
    return PropellerBladeDamage(n_envs=n_envs, dt=0.01)


@pytest.fixture
def mock_env_state(n_envs):
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.zeros(n_envs, 3),
        ang_vel=torch.zeros(n_envs, 3),
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
    """All 1000 draws must stay within [0.5, 1.0]."""
    lo, hi = perturbation.bounds
    for _ in range(1000):
        v = perturbation.sample()
        assert v.shape == (n_envs, 4)
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(perturbation):
    """curriculum_scale=0 → value equals nominal [1,1,1,1]."""
    perturbation.curriculum_scale = 0.0
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    """curriculum_scale=1 → samples differ from nominal with non-zero variance."""
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
    """tick(is_reset=True) must set _current_value for per_episode perturbation."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    assert perturbation._current_value is not None
    assert perturbation._current_value.shape == (n_envs, 4)


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
# U5 — Lipschitz (lipschitz_k=None → skip)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    """lipschitz_k=None → no constraint — assert_lipschitz is a no-op."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() Lipschitz (skipped — lipschitz_k=None)
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
    """observable=True → get_privileged_obs() returns efficiency vector after tick."""
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)
    # _current_value is the efficiency vector, NOT the force (overridden apply)
    assert obs.shape[-1] == 4


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
    """_current_value after tick() must be [n_envs, 4] float32 (efficiency vector)."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    # apply() must NOT overwrite _current_value — verify shape stays (n_envs, 4)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    v = perturbation._current_value
    assert v is not None
    assert v.shape == (n_envs, 4), f"Expected ({n_envs}, 4), got {v.shape}"
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
# U11 — adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    """In ADVERSARIAL mode, tick(is_reset=False) must never call sample()."""
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_s:
        perturbation.tick(is_reset=False)
        assert mock_s.call_count == 0, (
            f"tick(is_reset=False) called sample() {mock_s.call_count} time(s) in ADVERSARIAL mode"
        )


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
# I3 — apply() calls solver.apply_links_external_force
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrench_apply_calls_solver(perturbation, mock_scene, mock_env_state, n_envs):
    """apply() must call apply_links_external_force."""
    assert isinstance(perturbation, ExternalWrenchPerturbation)
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert mock_scene.rigid_solver.apply_links_external_force.called, (
        "apply_links_external_force was not called"
    )


@pytest.mark.integration
def test_force_shape(perturbation, mock_scene, mock_env_state, n_envs):
    """The force passed to apply_links_external_force must have shape [n_envs, 3]."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    call_args = mock_scene.rigid_solver.apply_links_external_force.call_args[0]
    force_arg = call_args[0]
    assert force_arg.shape == (n_envs, 3), (
        f"Expected force shape ({n_envs}, 3), got {force_arg.shape}"
    )


@pytest.mark.integration
def test_nominal_efficiency_zero_force(n_envs):
    """At nominal efficiency [1,1,1,1]: ΔF_z must be exactly zero."""
    scene = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = PropellerBladeDamage(n_envs=n_envs, dt=0.01)
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
    p._current_value = torch.ones(n_envs, 4)  # force nominal efficiency
    p.apply(scene, scene.drone, env_state)
    force = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    assert torch.allclose(force, torch.zeros_like(force), atol=1e-10), (
        f"Expected zero force at nominal efficiency, got: {force}"
    )


@pytest.mark.integration
def test_efficiency_preserved_after_apply(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() must NOT overwrite _current_value (efficiency ratios must remain)."""
    perturbation.tick(is_reset=True)
    efficiency_before = perturbation._current_value.clone()
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert torch.allclose(perturbation._current_value, efficiency_before), (
        "apply() overwrote _current_value — efficiency vector was lost"
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
    """PropellerBladeDamage tick() must stay under MAX_TICK_MS on CPU."""
    p = PropellerBladeDamage(n_envs=n_envs, dt=0.01)
    for _ in range(WARMUP):
        p.tick(is_reset=True)
    import time
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=True)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_TICK_MS, (
        f"PropellerBladeDamage tick() too slow at n_envs={n_envs}: "
        f"{elapsed_ms:.4f} ms/step (limit {MAX_TICK_MS} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_apply_overhead_cpu(n_envs: int) -> None:
    """PropellerBladeDamage apply() must stay under MAX_APPLY_MS on CPU."""
    import time
    scene = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = PropellerBladeDamage(n_envs=n_envs, dt=0.01)
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
    for _ in range(WARMUP):
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_APPLY_MS, (
        f"PropellerBladeDamage apply() too slow at n_envs={n_envs}: "
        f"{elapsed_ms:.4f} ms/step (limit {MAX_APPLY_MS} ms)"
    )
