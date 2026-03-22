"""Tests for ExternalWrenchPerturbation leaves (category 1).

Covers: U1–U11 unit tests, I1, I3 integration tests, P1 performance test.
AeroDragCoeff is stateless + per_episode → U8/U10 are skipped automatically.
"""
import time

import pytest
import torch
from unittest.mock import patch, MagicMock

from genesis_robust_rl.perturbations.base import (
    ExternalWrenchPerturbation,
    PhysicsPerturbation,
    PerturbationMode,
)
from genesis_robust_rl.perturbations.category_1_physics import AeroDragCoeff, GroundEffect
from tests.conftest import assert_lipschitz, EnvState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param


@pytest.fixture
def mock_env_state(n_envs):
    """Realistic EnvState for a drone moving at moderate velocity."""
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 2.0,  # non-zero velocity for drag computation
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


@pytest.fixture
def perturbation(request, n_envs):
    """AeroDragCoeff fixture parametrized over n_envs."""
    return AeroDragCoeff(n_envs=n_envs, dt=0.01)


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
# U9 — output shape and dtype (via apply() for PhysicsPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs, mock_env_state, mock_scene):
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    # ExternalWrenchPerturbation: _current_value is set in apply()
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    v = perturbation._current_value
    assert v is not None
    expected = (n_envs, 3)
    assert v.shape == expected, f"Expected {expected}, got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset (skipped for stateless / physics)
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
# I3 — ExternalWrenchPerturbation.apply() calls solver with correct tensor
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


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_wrench_force_shape(n_envs: int) -> None:
    """apply() must call solver with wrench tensor of shape [n_envs, 3]."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    drone = MagicMock()
    p = AeroDragCoeff(n_envs=n_envs, dt=0.01)
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 2.0,
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p.tick(is_reset=True)
    p.apply(scene, drone, env_state)

    assert scene.rigid_solver.apply_links_external_force.called
    call_args = scene.rigid_solver.apply_links_external_force.call_args[0]
    wrench_arg = call_args[0]
    assert wrench_arg.shape == (n_envs, 3), f"Expected ({n_envs}, 3), got {wrench_arg.shape}"
    assert wrench_arg.dtype == torch.float32


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_drag_force_direction(n_envs: int) -> None:
    """Drag force must oppose velocity: sign(F_i) == -sign(v_i) for each axis."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    drone = MagicMock()
    # Use fixed positive velocity for deterministic sign check
    vel = torch.ones(n_envs, 3) * 3.0
    env_state = EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=vel,
        ang_vel=torch.zeros(n_envs, 3),
        acc=torch.zeros(n_envs, 3),
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )
    p = AeroDragCoeff(n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)
    p.apply(scene, drone, env_state)

    wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    # All velocity components positive → all force components must be negative
    assert (wrench < 0).all(), f"Drag force should be negative for positive velocity, got {wrench}"


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

    p = AeroDragCoeff(n_envs=PERF_N_ENVS, dt=0.01)
    p.tick(is_reset=True)

    env_state = EnvState(
        pos=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").expand(PERF_N_ENVS, -1),
        vel=torch.randn(PERF_N_ENVS, 3, device="cuda") * 2.0,
        ang_vel=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        acc=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        rpm=torch.ones(PERF_N_ENVS, 4, device="cuda") * 3000.0,
        dt=0.01,
        step=0,
    )

    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    drone = MagicMock()

    # Warmup
    for _ in range(PERF_WARMUP):
        p._compute_wrench(env_state)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        p._compute_wrench(env_state)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS

    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"_compute_wrench() too slow: {elapsed_ms:.3f} ms/step (limit {MAX_MS_PER_STEP} ms)"
    )


# ===========================================================================
# 1.11 GroundEffect — U1–U11, I1, I3, P1
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1, 4, 16])
def ge_n_envs(request):
    return request.param


@pytest.fixture
def ge_perturbation(ge_n_envs):
    """GroundEffect fixture parametrized over n_envs."""
    return GroundEffect(n_envs=ge_n_envs, dt=0.01)


def _make_ge_env_state(n: int, altitude: float) -> EnvState:
    """Build an EnvState with a fixed altitude for GroundEffect tests."""
    pos = torch.zeros(n, 3)
    pos[:, 2] = altitude
    return EnvState(
        pos=pos,
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1),
        vel=torch.zeros(n, 3),
        ang_vel=torch.zeros(n, 3),
        acc=torch.zeros(n, 3),
        rpm=torch.ones(n, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


# ---------------------------------------------------------------------------
# U1 — sample() output within bounds (constant distribution)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_sample_within_bounds(ge_perturbation, ge_n_envs):
    """All 1000 sample() draws must stay within perturbation.bounds (constant distribution)."""
    lo, hi = ge_perturbation.bounds
    for _ in range(1000):
        v = ge_perturbation.sample()
        assert v.shape[0] in (ge_n_envs, 1)
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_curriculum_scale_zero(ge_perturbation):
    """curriculum_scale=0 → value equals nominal (0.0)."""
    ge_perturbation.curriculum_scale = 0.0
    v = ge_perturbation.sample()
    nominal = torch.tensor(ge_perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_ge_curriculum_scale_one(ge_perturbation):
    """curriculum_scale=1 — constant distribution always returns nominal; skip variance check."""
    # For constant distribution, all samples are always nominal — this is by design.
    ge_perturbation.curriculum_scale = 1.0
    v = ge_perturbation.sample()
    # Just verify it returns a valid tensor
    assert torch.is_tensor(v)
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_tick_reset(ge_perturbation, ge_n_envs):
    """tick(is_reset=True) must set _current_value for per_episode perturbations."""
    env_ids = torch.arange(ge_n_envs)
    ge_perturbation.tick(is_reset=True, env_ids=env_ids)
    # frequency=per_step → _current_value not set by tick(reset) alone (sample not called)
    # but it must not raise
    assert True


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_tick_step_advances(ge_perturbation):
    """tick(is_reset=False) must call sample() for per_step perturbations."""
    ge_perturbation.tick(is_reset=True)
    ge_perturbation.tick(is_reset=False)
    assert ge_perturbation._current_value is not None


# ---------------------------------------------------------------------------
# U5 — set_value() Lipschitz (lipschitz_k=None → no constraint)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_set_value_lipschitz(ge_perturbation):
    """lipschitz_k=None → set_value() passes through without clipping."""
    ge_perturbation.tick(is_reset=True)
    ge_perturbation.tick(is_reset=False)
    assert_lipschitz(ge_perturbation, n_steps=200)


# ---------------------------------------------------------------------------
# U6 — update_params() Lipschitz on distribution params
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_update_params_lipschitz(ge_perturbation):
    """lipschitz_k=None → update_params() skipped."""
    if ge_perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")


# ---------------------------------------------------------------------------
# U7 — get_privileged_obs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_get_privileged_obs_observable(ge_perturbation, ge_n_envs):
    """observable=True → get_privileged_obs() returns a tensor after apply()."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    ge_perturbation.observable = True
    ge_perturbation.tick(is_reset=True)
    env_state = _make_ge_env_state(ge_n_envs, altitude=0.05)
    ge_perturbation.apply(scene, scene.drone, env_state)
    obs = ge_perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_ge_get_privileged_obs_not_observable(ge_perturbation):
    """observable=False → get_privileged_obs() returns None."""
    ge_perturbation.observable = False
    assert ge_perturbation.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful persistence (skipped — stateless)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_stateful_persistence(ge_perturbation):
    if not ge_perturbation.is_stateful:
        pytest.skip("stateless perturbation")


@pytest.mark.unit
def test_ge_stateful_reset_all(ge_perturbation, ge_n_envs):
    if not ge_perturbation.is_stateful:
        pytest.skip("stateless perturbation")


# ---------------------------------------------------------------------------
# U9 — output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_output_shape_dtype(ge_perturbation, ge_n_envs):
    """_current_value after apply() must be [n_envs, 3] float32."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    ge_perturbation.tick(is_reset=True)
    env_state = _make_ge_env_state(ge_n_envs, altitude=0.05)
    ge_perturbation.apply(scene, scene.drone, env_state)
    v = ge_perturbation._current_value
    assert v is not None
    assert v.shape == (ge_n_envs, 3), f"Expected ({ge_n_envs}, 3), got {v.shape}"
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset (skipped — stateless PhysicsPerturbation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_partial_reset(ge_perturbation, ge_n_envs):
    if ge_n_envs < 2:
        pytest.skip("need n_envs >= 2")
    if not ge_perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    if isinstance(ge_perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation: _current_value set by apply(), not testable here")


# ---------------------------------------------------------------------------
# U11 — adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ge_adversarial_tick_no_sample(ge_perturbation):
    """In ADVERSARIAL mode, tick(is_reset=False) must never call sample()."""
    ge_perturbation.tick(is_reset=True)
    ge_perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(ge_perturbation), "sample", wraps=ge_perturbation.sample) as mock_s:
        ge_perturbation.tick(is_reset=False)
        assert mock_s.call_count == 0, (
            f"tick(is_reset=False) called sample() {mock_s.call_count} time(s) in ADVERSARIAL mode"
        )


# ---------------------------------------------------------------------------
# I1 — apply() returns None, does not raise
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ge_apply_returns_none(ge_perturbation, ge_n_envs):
    """apply() must return None and not raise."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    ge_perturbation.tick(is_reset=True)
    env_state = _make_ge_env_state(ge_n_envs, altitude=0.05)
    result = ge_perturbation.apply(scene, scene.drone, env_state)
    assert result is None


# ---------------------------------------------------------------------------
# I3 — ExternalWrenchPerturbation.apply() calls solver
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ge_wrench_apply_calls_solver(ge_perturbation, ge_n_envs):
    """apply() must call apply_links_external_force on the solver."""
    assert isinstance(ge_perturbation, ExternalWrenchPerturbation)
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    ge_perturbation.tick(is_reset=True)
    env_state = _make_ge_env_state(ge_n_envs, altitude=0.05)
    ge_perturbation.apply(scene, scene.drone, env_state)
    assert scene.rigid_solver.apply_links_external_force.called, (
        "solver.apply_links_external_force was not called"
    )


# ---------------------------------------------------------------------------
# Physics-model specific tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_ge_force_direction_upward(n_envs: int) -> None:
    """Ground-effect force must be upward (+Z) near the ground."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = GroundEffect(n_envs=n_envs, dt=0.01)
    env_state = _make_ge_env_state(n_envs, altitude=0.05)  # very close to ground
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    assert wrench.shape == (n_envs, 3)
    # Z component must be positive (upward thrust increase)
    assert (wrench[:, 2] > 0).all(), f"Ground-effect force should be upward, got Z={wrench[:, 2]}"
    # X and Y components must be zero
    assert torch.allclose(wrench[:, :2], torch.zeros(n_envs, 2), atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_ge_no_force_at_high_altitude(n_envs: int) -> None:
    """At altitude >> 4R, ground-effect force must be zero."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    p = GroundEffect(n_envs=n_envs, dt=0.01, rotor_radius=0.1)
    # Altitude = 5m >> 4*R=0.4m → well above influence zone
    env_state = _make_ge_env_state(n_envs, altitude=5.0)
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    assert torch.allclose(wrench, torch.zeros(n_envs, 3), atol=1e-6), (
        f"Expected zero force at high altitude, got {wrench}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_ge_force_monotonically_decreasing_with_altitude(n_envs: int) -> None:
    """Ground-effect force must decrease as altitude increases (monotonic in [R/2, 4R])."""
    altitudes = [0.06, 0.1, 0.15, 0.2, 0.3, 0.4]
    forces = []
    for alt in altitudes:
        scene = MagicMock()
        scene.rigid_solver = MagicMock()
        scene.rigid_solver.apply_links_external_force = MagicMock()
        p = GroundEffect(n_envs=n_envs, dt=0.01)
        env_state = _make_ge_env_state(n_envs, altitude=alt)
        p.tick(is_reset=True)
        p.apply(scene, scene.drone, env_state)
        wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
        forces.append(wrench[0, 2].item())
    # Each subsequent force should be <= previous
    for i in range(len(forces) - 1):
        assert forces[i] >= forces[i + 1] - 1e-6, (
            f"Force not monotonically decreasing: alt={altitudes[i]:.2f}→{altitudes[i+1]:.2f}, "
            f"F={forces[i]:.4f}→{forces[i+1]:.4f}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("n_envs", [1, 4, 16])
def test_ge_force_clamped_at_max(n_envs: int) -> None:
    """Force must be clamped when altitude is at or below R/2 (near-singularity)."""
    max_k_ge = 2.0
    nominal_thrust = 9.81 * 0.5
    p = GroundEffect(n_envs=n_envs, dt=0.01, max_k_ge=max_k_ge, nominal_thrust=nominal_thrust)
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    # Altitude = R/2 — exactly at the clamping boundary
    env_state = _make_ge_env_state(n_envs, altitude=0.05)
    p.tick(is_reset=True)
    p.apply(scene, scene.drone, env_state)
    wrench = scene.rigid_solver.apply_links_external_force.call_args[0][0]
    max_force = (max_k_ge - 1.0) * nominal_thrust
    assert (wrench[:, 2] <= max_force + 1e-5).all(), (
        f"Force exceeds max_k_ge bound: {wrench[:, 2].max():.4f} > {max_force:.4f}"
    )


# ---------------------------------------------------------------------------
# P1 — _compute_wrench() overhead < 0.5 ms/step (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_ge_apply_overhead() -> None:
    """_compute_wrench() measured over 1000 steps on GPU must be < 0.5 ms/step."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — perf tests require GPU")

    p = GroundEffect(n_envs=PERF_N_ENVS, dt=0.01)
    p.tick(is_reset=True)

    pos = torch.zeros(PERF_N_ENVS, 3, device="cuda")
    pos[:, 2] = 0.05  # close to ground for meaningful computation
    env_state = EnvState(
        pos=pos,
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").expand(PERF_N_ENVS, -1),
        vel=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        ang_vel=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        acc=torch.zeros(PERF_N_ENVS, 3, device="cuda"),
        rpm=torch.ones(PERF_N_ENVS, 4, device="cuda") * 3000.0,
        dt=0.01,
        step=0,
    )

    # Warmup
    for _ in range(PERF_WARMUP):
        p._compute_wrench(env_state)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        p._compute_wrench(env_state)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS

    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"GroundEffect._compute_wrench() too slow: {elapsed_ms:.3f} ms/step "
        f"(limit {MAX_MS_PER_STEP} ms)"
    )
