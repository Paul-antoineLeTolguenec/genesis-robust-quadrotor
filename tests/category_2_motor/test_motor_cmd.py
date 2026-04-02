"""Tests for MotorCommandPerturbation leaves (category 2 — stateless + stateful).

Covers: U1-U11 unit tests, I1, perturbation-specific tests.
"""

import time
from unittest.mock import patch

import pytest
import torch

from genesis_robust_rl.perturbations.base import (
    MotorCommandPerturbation,
    PerturbationMode,
)
from genesis_robust_rl.perturbations.category_2_motor import (
    MotorColdStart,
    MotorKill,
    MotorLag,
    MotorRPMNoise,
    MotorSaturation,
    MotorWear,
    RotorImbalance,
)
from tests.conftest import assert_lipschitz

# -- MotorCommand-only fixture (7 classes) --

_MOTOR_CMD_FACTORIES = [
    lambda n: MotorKill(n_envs=n, dt=0.01),
    lambda n: MotorRPMNoise(n_envs=n, dt=0.01),
    lambda n: MotorSaturation(n_envs=n, dt=0.01),
    lambda n: MotorLag(n_envs=n, dt=0.01),
    lambda n: MotorWear(n_envs=n, dt=0.01),
    lambda n: RotorImbalance(n_envs=n, dt=0.01),
    lambda n: MotorColdStart(n_envs=n, dt=0.01),
]


@pytest.fixture(params=_MOTOR_CMD_FACTORIES)
def motor_pert(request, n_envs):
    return request.param(n_envs)


# ---------------------------------------------------------------------------
# U1 — sample() output within bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_within_bounds(motor_pert, n_envs):
    lo, hi = motor_pert.bounds
    motor_pert.tick(is_reset=True)
    for _ in range(200):
        if motor_pert.is_stateful:
            motor_pert.tick(is_reset=True)
            v = motor_pert._current_value
        else:
            v = motor_pert.sample()
        assert v is not None
        assert (v >= lo - 1e-6).all() and (v <= hi + 1e-6).all(), (
            f"Out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )


# ---------------------------------------------------------------------------
# U2 — curriculum_scale extremes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_curriculum_scale_zero(motor_pert):
    motor_pert.curriculum_scale = 0.0
    if motor_pert.is_stateful:
        motor_pert.tick(is_reset=True)
        v = motor_pert._current_value
    else:
        v = motor_pert.sample()
    nominal = torch.tensor(motor_pert.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)


@pytest.mark.unit
def test_curriculum_scale_one(motor_pert):
    motor_pert.curriculum_scale = 1.0
    # State-driven perturbations at bounds can't show variation at reset
    if motor_pert.nominal == motor_pert.bounds[0]:
        pytest.skip("nominal at lower bound")
    if motor_pert.nominal == motor_pert.bounds[1]:
        pytest.skip("nominal at upper bound")
    samples = []
    for _ in range(100):
        if motor_pert.is_stateful:
            motor_pert.tick(is_reset=True)
            samples.append(motor_pert._current_value.clone())
        else:
            samples.append(motor_pert.sample())
    stacked = torch.stack(samples)
    nominal = torch.tensor(motor_pert.nominal, dtype=stacked.dtype)
    assert not torch.allclose(stacked, nominal.expand_as(stacked), atol=1e-4), (
        "curriculum_scale=1 produced only nominal values"
    )


# ---------------------------------------------------------------------------
# U3 — tick() reset path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_reset_sets_value(motor_pert, n_envs):
    motor_pert.tick(is_reset=True, env_ids=torch.arange(n_envs))
    if motor_pert.frequency == "per_episode" or motor_pert.is_stateful:
        assert motor_pert._current_value is not None


# ---------------------------------------------------------------------------
# U4 — tick() step path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tick_step_advances(motor_pert):
    motor_pert.tick(is_reset=True)
    motor_pert.tick(is_reset=False)
    assert motor_pert._current_value is not None


# ---------------------------------------------------------------------------
# U5 — set_value() Lipschitz
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_value_lipschitz(motor_pert):
    motor_pert.tick(
        is_reset=True,
        env_ids=torch.arange(motor_pert.n_envs),
    )
    assert_lipschitz(motor_pert, n_steps=50)


# ---------------------------------------------------------------------------
# U6 — update_params() Lipschitz (all have lipschitz_k=None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_params_lipschitz(motor_pert):
    if motor_pert.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")


# ---------------------------------------------------------------------------
# U7 — get_privileged_obs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_privileged_obs_observable(motor_pert):
    motor_pert.observable = True
    motor_pert.tick(is_reset=True)
    if motor_pert.frequency == "per_step":
        motor_pert.tick(is_reset=False)
    obs = motor_pert.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)


@pytest.mark.unit
def test_get_privileged_obs_not_observable(motor_pert):
    motor_pert.observable = False
    assert motor_pert.get_privileged_obs() is None


# ---------------------------------------------------------------------------
# U8 — stateful persistence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_persistence(motor_pert, n_envs):
    if not motor_pert.is_stateful:
        pytest.skip("stateless perturbation")
    motor_pert.tick(is_reset=True)
    rpm_cmd = torch.ones(n_envs, 4) * 3000.0
    out_t0 = motor_pert.apply(rpm_cmd).clone()
    for _ in range(10):
        motor_pert.tick(is_reset=False)
    out_t10 = motor_pert.apply(rpm_cmd).clone()
    if isinstance(motor_pert, RotorImbalance):
        assert motor_pert._phase.abs().max().item() > 1.0
    elif isinstance(motor_pert, MotorLag):
        assert not torch.allclose(out_t0, out_t10, atol=1e-3)
    else:
        assert not torch.allclose(out_t0, out_t10, atol=1e-6)


# ---------------------------------------------------------------------------
# U9 — apply() output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_output_shape(motor_pert, n_envs):
    motor_pert.tick(is_reset=True)
    if motor_pert.frequency == "per_step":
        motor_pert.tick(is_reset=False)
    rpm_cmd = torch.ones(n_envs, 4) * 10000.0
    result = motor_pert.apply(rpm_cmd)
    assert result.shape == (n_envs, 4)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# U10 — partial reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_reset(motor_pert, n_envs):
    if not motor_pert.is_stateful:
        pytest.skip("stateless perturbation")
    if n_envs < 2:
        pytest.skip("need n_envs >= 2")
    motor_pert.tick(is_reset=True)
    rpm_cmd = torch.ones(n_envs, 4) * 3000.0
    for _ in range(20):
        motor_pert.apply(rpm_cmd)
        motor_pert.tick(is_reset=False)
    cv_before = motor_pert._current_value[1:].clone()
    motor_pert.reset(torch.tensor([0]))
    cv_after = motor_pert._current_value[1:].clone()
    assert torch.allclose(cv_before, cv_after, atol=1e-6), "Partial reset affected non-reset envs"


# ---------------------------------------------------------------------------
# U11 — adversarial mode: tick() must not call sample()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adversarial_tick_no_sample(motor_pert):
    motor_pert.tick(is_reset=True)
    motor_pert.mode = PerturbationMode.ADVERSARIAL
    with patch.object(
        type(motor_pert),
        "sample",
        wraps=motor_pert.sample,
    ) as mock_s:
        motor_pert.tick(is_reset=False)
        assert mock_s.call_count == 0


# ---------------------------------------------------------------------------
# I1 — apply() does not modify input rpm_cmd in-place
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_no_inplace_mutation(motor_pert, n_envs):
    motor_pert.tick(is_reset=True)
    if motor_pert.frequency == "per_step":
        motor_pert.tick(is_reset=False)
    rpm_cmd = torch.ones(n_envs, 4) * 10000.0
    rpm_orig = rpm_cmd.clone()
    _ = motor_pert.apply(rpm_cmd)
    assert torch.allclose(rpm_cmd, rpm_orig), "apply() modified input"


# ---------------------------------------------------------------------------
# I2 — inheritance check
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_inheritance(motor_pert):
    assert isinstance(motor_pert, MotorCommandPerturbation)


# ===========================================================================
# Perturbation-specific tests
# ===========================================================================


class TestMotorKill:
    @pytest.fixture(params=[1, 4, 16])
    def n_envs(self, request):
        return request.param

    @pytest.fixture
    def kill_pert(self, n_envs):
        return MotorKill(n_envs=n_envs, dt=0.01, max_killed=2)

    @pytest.mark.unit
    def test_killed_motors_zero_rpm(self, kill_pert, n_envs):
        kill_pert.tick(is_reset=True)
        rpm_cmd = torch.ones(n_envs, 4) * 15000.0
        result = kill_pert.apply(rpm_cmd)
        mask = kill_pert._current_value
        killed = result[mask == 1.0]
        if killed.numel() > 0:
            assert (killed == 0.0).all()

    @pytest.mark.unit
    def test_kill_mask_binary(self, kill_pert, n_envs):
        for _ in range(100):
            kill_pert.sample()
            mask = kill_pert._current_value
            assert ((mask == 0.0) | (mask == 1.0)).all()

    @pytest.mark.unit
    def test_kill_count_in_range(self, kill_pert, n_envs):
        for _ in range(200):
            kill_pert.sample()
            n_killed = kill_pert._current_value.sum(dim=1)
            assert (n_killed >= kill_pert.min_killed).all()
            assert (n_killed <= kill_pert.max_killed).all()

    @pytest.mark.unit
    def test_curriculum_zero_no_kills(self, n_envs):
        p = MotorKill(n_envs=n_envs, dt=0.01, max_killed=2)
        p.curriculum_scale = 0.0
        for _ in range(100):
            p.sample()
            assert (p._current_value == 0.0).all()


class TestMotorRPMNoise:
    @pytest.fixture(params=[1, 4, 16])
    def n_envs(self, request):
        return request.param

    @pytest.mark.unit
    def test_noise_is_relative(self, n_envs):
        p = MotorRPMNoise(n_envs=n_envs, dt=0.01)
        p.tick(is_reset=True)
        p._current_value = torch.full((n_envs, 4), 0.01)
        rpm_low = torch.ones(n_envs, 4) * 1000.0
        rpm_high = torch.ones(n_envs, 4) * 20000.0
        torch.manual_seed(42)
        diffs_lo = [(p.apply(rpm_low) - rpm_low).abs().mean() for _ in range(200)]
        diffs_hi = [(p.apply(rpm_high) - rpm_high).abs().mean() for _ in range(200)]
        assert torch.stack(diffs_hi).mean() > torch.stack(diffs_lo).mean() * 5


class TestMotorSaturation:
    @pytest.fixture(params=[1, 4, 16])
    def n_envs(self, request):
        return request.param

    @pytest.mark.unit
    def test_rpm_clamped(self, n_envs):
        p = MotorSaturation(n_envs=n_envs, dt=0.01)
        p.tick(is_reset=True)
        rpm_cmd = torch.ones(n_envs, 4) * 25000.0
        result = p.apply(rpm_cmd)
        rpm_max = p._current_value.unsqueeze(-1)
        assert (result <= rpm_max + 1e-6).all()

    @pytest.mark.unit
    def test_negative_rpm_clamped(self, n_envs):
        p = MotorSaturation(n_envs=n_envs, dt=0.01)
        p.tick(is_reset=True)
        result = p.apply(torch.ones(n_envs, 4) * -5000.0)
        assert (result >= 0.0).all()


class TestMotorLag:
    @pytest.mark.unit
    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_lag_converges(self, n_envs):
        p = MotorLag(n_envs=n_envs, dt=0.01)
        p.tick(is_reset=True)
        rpm_cmd = torch.ones(n_envs, 4) * 5000.0
        prev_err = float("inf")
        for _ in range(500):
            out = p.apply(rpm_cmd)
            err = (rpm_cmd - out).abs().max().item()
            assert err <= prev_err + 1e-6
            prev_err = err
        assert prev_err < 100.0


class TestMotorWear:
    @pytest.mark.unit
    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_monotonically_decreasing(self, n_envs):
        p = MotorWear(
            n_envs=n_envs,
            dt=0.01,
            distribution_params={"low": 0.001, "high": 0.001},
            curriculum_scale=1.0,
        )
        p.tick(is_reset=True)
        eff_prev = p._efficiency.clone()
        for _ in range(50):
            p.tick(is_reset=False)
            assert (p._efficiency <= eff_prev + 1e-6).all()
            eff_prev = p._efficiency.clone()

    @pytest.mark.unit
    def test_efficiency_floor(self):
        p = MotorWear(
            n_envs=4,
            dt=0.01,
            distribution_params={"low": 0.01, "high": 0.01},
            curriculum_scale=1.0,
        )
        p.tick(is_reset=True)
        for _ in range(200):
            p.tick(is_reset=False)
        assert (p._efficiency >= 0.8 - 1e-6).all()


class TestRotorImbalance:
    @pytest.mark.unit
    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_modulation_bounded(self, n_envs):
        p = RotorImbalance(
            n_envs=n_envs,
            dt=0.01,
            distribution_params={"low": 0.02, "high": 0.03},
            curriculum_scale=1.0,
        )
        p.tick(is_reset=True)
        rpm_cmd = torch.ones(n_envs, 4) * 3000.0
        for _ in range(100):
            out = p.apply(rpm_cmd)
            p.tick(is_reset=False)
            ratio = out / rpm_cmd
            assert (ratio >= 1.0 - p.bounds[1] - 1e-6).all()
            assert (ratio <= 1.0 + p.bounds[1] + 1e-6).all()

    @pytest.mark.unit
    def test_imu_noise_amplitude(self):
        p = RotorImbalance(n_envs=4, dt=0.01)
        p.tick(is_reset=True)
        p.apply(torch.ones(4, 4) * 3000.0)
        assert p.imu_noise_amplitude.shape == (4, 4)


class TestMotorColdStart:
    @pytest.mark.unit
    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_warmup_decays(self, n_envs):
        p = MotorColdStart(
            n_envs=n_envs,
            dt=0.01,
            warmup_tau=0.1,
            distribution_params={"low": 1.2, "high": 1.3},
            curriculum_scale=1.0,
        )
        p.tick(is_reset=True)
        d0 = (p._warmup_factor - 1.0).abs().max().item()
        for _ in range(200):
            p.tick(is_reset=False)
        d1 = (p._warmup_factor - 1.0).abs().max().item()
        assert d1 < d0

    @pytest.mark.unit
    def test_warmup_converges(self):
        p = MotorColdStart(
            n_envs=4,
            dt=0.01,
            warmup_tau=0.1,
            distribution_params={"low": 1.3, "high": 1.3},
            curriculum_scale=1.0,
        )
        p.tick(is_reset=True)
        for _ in range(500):
            p.tick(is_reset=False)
        assert torch.allclose(p._warmup_factor, torch.ones(4, 4), atol=1e-3)


# ---------------------------------------------------------------------------
# Perf — tick + apply overhead (CPU)
# ---------------------------------------------------------------------------

WARMUP = 200
STEPS = 2000
MAX_TICK_MS = 0.1


@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
@pytest.mark.parametrize(
    "make_pert",
    [
        lambda n: MotorKill(n_envs=n, dt=0.01),
        lambda n: MotorRPMNoise(n_envs=n, dt=0.01),
        lambda n: MotorSaturation(n_envs=n, dt=0.01),
        lambda n: MotorLag(n_envs=n, dt=0.01),
        lambda n: MotorWear(n_envs=n, dt=0.01),
        lambda n: RotorImbalance(n_envs=n, dt=0.01),
        lambda n: MotorColdStart(n_envs=n, dt=0.01),
    ],
    ids=["kill", "noise", "sat", "lag", "wear", "imbalance", "cold"],
)
def test_tick_apply_overhead_cpu(n_envs, make_pert):
    """tick() + apply() must stay under MAX_TICK_MS on CPU."""
    p = make_pert(n_envs)
    rpm_cmd = torch.ones(n_envs, 4) * 3000.0
    p.tick(is_reset=True)
    # per_step perturbations need a step tick to populate _current_value
    if p.frequency == "per_step":
        p.tick(is_reset=False)
    for _ in range(WARMUP):
        p.tick(is_reset=False)
        p.apply(rpm_cmd)
    start = time.perf_counter()
    for _ in range(STEPS):
        p.tick(is_reset=False)
        p.apply(rpm_cmd)
    elapsed_ms = (time.perf_counter() - start) * 1000 / STEPS
    assert elapsed_ms < MAX_TICK_MS, f"{p.id} too slow at n_envs={n_envs}: {elapsed_ms:.4f} ms"
