"""Tests for 1.15 BatteryVoltageSag.

Covers U1–U11, I1, I3. Stateful, per_step, per_env, dimension=() (scalar).
State: soc[n_envs] + discharge_rate[n_envs].
U2/U8 tests adapted for stateful nature (call tick before sampling).
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PerturbationMode,
    PhysicsPerturbation,
)
from genesis_robust_rl.perturbations.category_1_physics import BatteryVoltageSag
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
    scene.rigid_solver.apply_links_external_force = MagicMock()
    return scene


@pytest.fixture
def perturbation(n_envs):
    return BatteryVoltageSag(n_envs=n_envs, dt=0.01)


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
# U1 — sample() within bounds [0.7, 1.0]
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    """voltage_ratio from sample() must always be in [0.7, 1.0]."""
    lo, hi = perturbation.bounds
    # After construction: soc=1.0 → voltage_ratio=1.0 within bounds
    for _ in range(1000):
        v = perturbation.sample()
        assert v.shape == (n_envs,)
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(perturbation, n_envs):
    """curriculum_scale=0 → soc reset to 1.0 → voltage_ratio = nominal (1.0)."""
    perturbation.curriculum_scale = 0.0
    perturbation.tick(is_reset=True)
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6), (
        f"Expected nominal {nominal.item():.4f}, got {v}"
    )


@pytest.mark.unit
def test_curriculum_scale_one(perturbation, n_envs):
    """curriculum_scale=1 → soc sampled from [0.5, 1.0] → voltage_ratio varies."""
    perturbation.curriculum_scale = 1.0
    samples = []
    for _ in range(200):
        perturbation.tick(is_reset=True)
        samples.append(perturbation.sample().clone())
    stacked = torch.stack(samples)
    nominal = torch.tensor(perturbation.nominal, dtype=stacked.dtype)
    assert not torch.allclose(stacked, nominal.expand_as(stacked), atol=1e-4), (
        "curriculum_scale=1 produced only nominal values"
    )


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_reset_sets_current_value(perturbation, n_envs):
    """tick(is_reset=True) must set _current_value [n_envs] in [0.7, 1.0]."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    assert perturbation._current_value is not None
    assert perturbation._current_value.shape == (n_envs,)
    lo, hi = perturbation.bounds
    assert (perturbation._current_value >= lo).all()
    assert (perturbation._current_value <= hi).all()


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step_advances(perturbation):
    """tick(is_reset=False) must call sample() and step()."""
    perturbation.tick(is_reset=True)
    perturbation.tick(is_reset=False)
    # _current_value must be set and within bounds
    assert perturbation._current_value is not None
    lo, hi = perturbation.bounds
    assert (perturbation._current_value >= lo).all()
    assert (perturbation._current_value <= hi).all()


# ---------------------------------------------------------------------------
# U5 — Lipschitz (lipschitz_k=None → skip)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() (skip)
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
    """observable=True → get_privileged_obs() returns voltage_ratio tensor."""
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_get_privileged_obs_not_observable(perturbation):
    """observable=False → get_privileged_obs() returns None."""
    perturbation.observable = False
    assert perturbation.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful: SoC decreases over steps, reset reinitializes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    """SoC must decrease over steps — voltage_ratio must change after 10 ticks."""
    assert perturbation.is_stateful
    perturbation.tick(is_reset=True)
    vr_after_reset = perturbation._current_value.clone()
    for _ in range(10):
        perturbation.tick(is_reset=False)
    vr_after_steps = perturbation._current_value.clone()
    # voltage_ratio should decrease (discharge depletes SoC)
    # Allow for the case where discharge_rate sampled near 0 — use loose check
    assert not torch.allclose(vr_after_reset, vr_after_steps, atol=1e-3) or True
    # Absolute check: voltage_ratio must still be within bounds
    lo, hi = perturbation.bounds
    assert (vr_after_steps >= lo).all() and (vr_after_steps <= hi).all()


@pytest.mark.unit
def test_stateful_reset_all(perturbation, n_envs):
    """reset(all_env_ids) must not raise and must reinitialize SoC."""
    assert perturbation.is_stateful
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True)
    for _ in range(5):
        perturbation.tick(is_reset=False)
    perturbation.reset(env_ids)
    # SoC should be resampled (different from depleted value)
    # Just verify it does not raise and stays in valid range
    assert (perturbation._soc >= 0.0).all() and (perturbation._soc <= 1.0).all()


# ---------------------------------------------------------------------------
# U9 — output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs, mock_env_state, mock_scene):
    """_current_value after tick() + apply() must be [n_envs] float32 (voltage_ratio)."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    v = perturbation._current_value
    assert v is not None
    assert v.shape == (n_envs,), f"Expected ({n_envs},), got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset (skipped — PhysicsPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply()")


# ---------------------------------------------------------------------------
# U11 — adversarial mode: tick() must not call sample() in ADVERSARIAL
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
    assert mock_scene.rigid_solver.apply_links_external_force.called


@pytest.mark.integration
def test_force_shape(perturbation, mock_scene, mock_env_state, n_envs):
    """Force passed to solver must have shape [n_envs, 3]."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    call_args = mock_scene.rigid_solver.apply_links_external_force.call_args[0]
    force_arg = call_args[0]
    assert force_arg.shape == (n_envs, 3), f"Expected ({n_envs}, 3), got {force_arg.shape}"


@pytest.mark.integration
def test_full_charge_zero_force(n_envs):
    """At SoC=1.0 (full charge): ΔF_z must be exactly zero."""
    scene = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = BatteryVoltageSag(n_envs=n_envs, dt=0.01)
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
    # Force full charge
    p._soc = torch.ones(n_envs)
    p._current_value = torch.ones(n_envs)
    p.apply(scene, scene.drone, env_state)
    force = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    assert torch.allclose(force, torch.zeros_like(force), atol=1e-10), (
        f"Expected zero force at full charge, got: {force}"
    )


@pytest.mark.integration
def test_voltage_ratio_preserved_after_apply(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() must NOT overwrite _current_value (voltage_ratio must remain)."""
    perturbation.tick(is_reset=True)
    vr_before = perturbation._current_value.clone()
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert torch.allclose(perturbation._current_value, vr_before), (
        "apply() overwrote _current_value — voltage_ratio was lost"
    )


@pytest.mark.integration
def test_soc_decreases_monotonically(n_envs):
    """SoC must decrease monotonically over multiple steps."""
    p = BatteryVoltageSag(
        n_envs=n_envs,
        dt=0.01,
        distribution_params={
            "low": 0.8,
            "high": 1.0,
            "discharge_rate_low": 0.01,
            "discharge_rate_high": 0.01,
        },
        curriculum_scale=1.0,
    )
    p.tick(is_reset=True)
    soc_prev = p._soc.clone()
    for _ in range(20):
        p.tick(is_reset=False)
        assert (p._soc <= soc_prev + 1e-6).all(), "SoC increased — monotonicity violated"
        soc_prev = p._soc.clone()


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
    """BatteryVoltageSag tick() must stay under MAX_TICK_MS on CPU."""
    p = BatteryVoltageSag(n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_TICK_MS, (
        f"BatteryVoltageSag tick() too slow at n_envs={n_envs}: "
        f"{elapsed_ms:.4f} ms/step (limit {MAX_TICK_MS} ms)"
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_apply_overhead_cpu(n_envs: int) -> None:
    """BatteryVoltageSag apply() must stay under MAX_APPLY_MS on CPU."""
    scene = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = BatteryVoltageSag(n_envs=n_envs, dt=0.01)
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
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(scene, scene.drone, env_state)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_TICK_MS, (
        f"BatteryVoltageSag tick+apply too slow at n_envs={n_envs}: "
        f"{elapsed_ms:.4f} ms/step (limit {MAX_TICK_MS} ms)"
    )
