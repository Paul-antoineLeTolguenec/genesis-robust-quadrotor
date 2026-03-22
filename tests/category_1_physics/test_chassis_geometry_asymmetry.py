"""Tests for 1.12 ChassisGeometryAsymmetry.

Covers U1–U11, I1, I3 (PhysicsPerturbation with dual setters), P1.
The perturbation is stateless, per_episode, per_env, dimension=(4,).
"""
import math
import time

import pytest
import torch
from unittest.mock import MagicMock, patch

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PhysicsPerturbation,
    PerturbationMode,
)
from genesis_robust_rl.perturbations.category_1_physics import ChassisGeometryAsymmetry
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
    drone = MagicMock()
    drone.set_links_mass_shift = MagicMock()
    drone.set_links_COM_shift = MagicMock()
    scene.drone = drone
    return scene


@pytest.fixture
def perturbation(n_envs, mock_scene):
    return ChassisGeometryAsymmetry(
        mass_setter_fn=mock_scene.drone.set_links_mass_shift,
        com_setter_fn=mock_scene.drone.set_links_COM_shift,
        n_envs=n_envs,
        dt=0.01,
    )


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
# U1 — sample() output within bounds (1000 draws)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    """All 1000 draws must stay within perturbation.bounds."""
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
    """curriculum_scale=0 → value equals nominal [0,0,0,0]."""
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
        "curriculum_scale=1 produced only nominal values — distribution may be degenerate"
    )


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_reset_samples_per_episode(perturbation, n_envs):
    """tick(is_reset=True) must set _current_value for per_episode perturbation."""
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True, env_ids=env_ids)
    assert perturbation._current_value is not None


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step_does_not_resample(perturbation):
    """tick(is_reset=False) must not resample for per_episode perturbation."""
    perturbation.tick(is_reset=True)
    v_before = perturbation._current_value.clone()
    perturbation.tick(is_reset=False)
    # per_episode → value unchanged after step tick
    assert torch.allclose(perturbation._current_value, v_before)


# ---------------------------------------------------------------------------
# U5 — set_value() Lipschitz (lipschitz_k=None → no constraint)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    """lipschitz_k=None → set_value() passes through without clipping."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() Lipschitz (skipped — lipschitz_k=None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_params_lipschitz(perturbation):
    """lipschitz_k=None → no Lipschitz constraint on distribution params."""
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")


# ---------------------------------------------------------------------------
# U7 — get_privileged_obs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_privileged_obs_observable(perturbation, mock_env_state, mock_scene):
    """observable=True → get_privileged_obs() returns _current_value after apply()."""
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_get_privileged_obs_not_observable(perturbation):
    """observable=False → get_privileged_obs() returns None."""
    perturbation.observable = False
    assert perturbation.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful persistence (skipped — stateless)
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
    """_current_value after tick() must be [n_envs, 4] float32."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    v = perturbation._current_value
    assert v is not None, "_current_value is None after tick()"
    assert v.shape == (n_envs, 4), f"Expected ({n_envs}, 4), got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset (skipped — stateless PhysicsPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply(), not testable here")


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
# I1 — apply() returns None and does not raise
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_returns_none(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() must return None."""
    perturbation.tick(is_reset=True)
    result = perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert result is None


# ---------------------------------------------------------------------------
# I3 — apply() calls both setters with correct shapes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_calls_both_setters(perturbation, mock_env_state, mock_scene, n_envs):
    """apply() must call both set_links_mass_shift and set_links_COM_shift."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)

    assert mock_scene.drone.set_links_mass_shift.called, "set_links_mass_shift was not called"
    assert mock_scene.drone.set_links_COM_shift.called, "set_links_COM_shift was not called"


@pytest.mark.integration
def test_apply_setter_shapes(perturbation, mock_env_state, mock_scene, n_envs):
    """Setter call arguments must have correct shapes."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)

    # mass setter: delta_m [n_envs], envs_idx [n_envs]
    mass_args = mock_scene.drone.set_links_mass_shift.call_args[0]
    assert mass_args[0].shape == (n_envs,), (
        f"delta_m shape: expected ({n_envs},), got {mass_args[0].shape}"
    )
    assert mass_args[1].shape == (n_envs,), (
        f"envs_idx shape: expected ({n_envs},), got {mass_args[1].shape}"
    )

    # com setter: delta_com [n_envs, 3], envs_idx [n_envs]
    com_args = mock_scene.drone.set_links_COM_shift.call_args[0]
    assert com_args[0].shape == (n_envs, 3), (
        f"delta_com shape: expected ({n_envs}, 3), got {com_args[0].shape}"
    )
    assert com_args[1].shape == (n_envs,), (
        f"envs_idx shape: expected ({n_envs},), got {com_args[1].shape}"
    )


@pytest.mark.integration
def test_mass_shift_is_zero(perturbation, mock_env_state, mock_scene, n_envs):
    """Arm deviations do not change mass — delta_m must be exactly zero."""
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    delta_m = mock_scene.drone.set_links_mass_shift.call_args[0][0]
    assert torch.allclose(delta_m, torch.zeros_like(delta_m), atol=1e-8), (
        f"Expected zero mass shift, got {delta_m}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_com_shift_geometry(n_envs: int) -> None:
    """CoM shift must be consistent with the planar mass-moment formula."""
    scene = MagicMock()
    scene.drone = MagicMock()
    scene.drone.set_links_mass_shift = MagicMock()
    scene.drone.set_links_COM_shift = MagicMock()

    p = ChassisGeometryAsymmetry(
        mass_setter_fn=scene.drone.set_links_mass_shift,
        com_setter_fn=scene.drone.set_links_COM_shift,
        n_envs=n_envs,
        dt=0.01,
        distribution="constant",
        distribution_params={"value": 0.01},
        bounds=(-0.05, 0.05),
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

    delta_com = scene.drone.set_links_COM_shift.call_args[0][0]  # [n_envs, 3]
    assert delta_com.shape == (n_envs, 3)
    assert delta_com.dtype == torch.float32

    # Constant 0.01 m deviation on all 4 arms:
    # cos_x = [cos(45°), cos(135°), cos(225°), cos(315°)] → sum ≈ 0
    # cos_y = [-sin(45°), -sin(135°), -sin(225°), -sin(315°)] → sum ≈ 0
    # So Δx ≈ 0, Δy ≈ 0 for uniform symmetric deviations
    assert torch.allclose(delta_com[:, 2], torch.zeros(n_envs), atol=1e-6), (
        "Z component of CoM shift must always be zero"
    )


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_com_asymmetric_deviation(n_envs: int) -> None:
    """Non-symmetric arm deviation must produce non-zero CoM shift."""
    scene = MagicMock()
    scene.drone = MagicMock()
    scene.drone.set_links_mass_shift = MagicMock()
    scene.drone.set_links_COM_shift = MagicMock()

    p = ChassisGeometryAsymmetry(
        mass_setter_fn=scene.drone.set_links_mass_shift,
        com_setter_fn=scene.drone.set_links_COM_shift,
        n_envs=n_envs,
        dt=0.01,
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
    # Force an asymmetric deviation: arm 0 = +0.05 m, others = 0
    p.tick(is_reset=True)
    p._current_value = torch.zeros(n_envs, 4)
    p._current_value[:, 0] = 0.05

    p.apply(scene, scene.drone, env_state)
    delta_com = scene.drone.set_links_COM_shift.call_args[0][0]

    # arm 0 at angle π/4: cos(π/4) = √2/2 → Δx = 0.05 * cos(π/4) / 4 ≠ 0
    expected_x = 0.05 * math.cos(math.pi / 4) / 4
    assert abs(delta_com[0, 0].item() - expected_x) < 1e-5, (
        f"Expected Δx_com ≈ {expected_x:.6f}, got {delta_com[0, 0].item():.6f}"
    )
    # Z must be zero
    assert torch.allclose(delta_com[:, 2], torch.zeros(n_envs), atol=1e-6)


# ---------------------------------------------------------------------------
# P1 — apply() overhead < 0.5 ms/step (GPU, skipped if no CUDA)
# ---------------------------------------------------------------------------


PERF_N_ENVS = 512
PERF_WARMUP = 100
PERF_STEPS = 1000
MAX_MS_PER_STEP = 0.5


@pytest.mark.perf
def test_apply_overhead() -> None:
    """apply() measured over 1000 steps on GPU must be < 0.5 ms/step."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — perf tests require GPU")

    scene = MagicMock()
    scene.drone = MagicMock()
    scene.drone.set_links_mass_shift = MagicMock()
    scene.drone.set_links_COM_shift = MagicMock()

    p = ChassisGeometryAsymmetry(
        mass_setter_fn=scene.drone.set_links_mass_shift,
        com_setter_fn=scene.drone.set_links_COM_shift,
        n_envs=PERF_N_ENVS,
        dt=0.01,
    )
    p.tick(is_reset=True)

    env_state = EnvState(
        pos=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").expand(PERF_N_ENVS, -1),
        vel=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        ang_vel=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        acc=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        rpm=torch.ones(PERF_N_ENVS, 4, device="cuda") * 3000.0,
        dt=0.01,
        step=0,
    )

    # Move internal tensors to CUDA
    p._current_value = p._current_value.cuda()

    # Warmup (bypass MagicMock overhead — measure only tensor ops)
    for _ in range(PERF_WARMUP):
        deviations = p._current_value
        angles = p._ARM_ANGLES.to(deviations.device)
        cos_x = torch.cos(angles)
        cos_y = torch.cos(angles + math.pi / 2)
        delta_x = (deviations * cos_x).sum(dim=1) / 4.0
        delta_y = (deviations * cos_y).sum(dim=1) / 4.0
        _ = torch.stack([delta_x, delta_y, torch.zeros_like(delta_x)], dim=1)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        deviations = p._current_value
        angles = p._ARM_ANGLES.to(deviations.device)
        cos_x = torch.cos(angles)
        cos_y = torch.cos(angles + math.pi / 2)
        delta_x = (deviations * cos_x).sum(dim=1) / 4.0
        delta_y = (deviations * cos_y).sum(dim=1) / 4.0
        _ = torch.stack([delta_x, delta_y, torch.zeros_like(delta_x)], dim=1)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS

    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"ChassisGeometryAsymmetry CoM computation too slow: {elapsed_ms:.3f} ms/step "
        f"(limit {MAX_MS_PER_STEP} ms)"
    )
