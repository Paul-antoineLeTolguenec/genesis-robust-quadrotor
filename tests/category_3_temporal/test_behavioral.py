"""Behavioral tests for category 3 — temporal perturbations.

Tests the actual ZOH, delay, and stall behavior beyond shape checks.
"""

import pytest
import torch

from genesis_robust_rl.perturbations.category_3_temporal import (
    ActionFixedDelay,
    ComputationOverload,
    ObsFixedDelay,
    PacketLoss,
)

# ---------------------------------------------------------------------------
# ObsFixedDelay — delay=0 is passthrough, delay=N returns N-step-old obs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_obs_delay_zero_passthrough() -> None:
    """delay=0 must return the just-pushed obs (passthrough)."""
    p = ObsFixedDelay(obs_slice=slice(0, 3), obs_dim=3, n_envs=2, dt=0.01)
    p.tick(is_reset=True)
    # Force delay=0
    p._delay[:] = 0
    obs = torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 6.0, 99.0]])
    result = p.apply(obs)
    assert torch.allclose(result[:, :3], obs[:, :3], atol=1e-6)


@pytest.mark.unit
def test_obs_delay_returns_old_value() -> None:
    """delay=2 must return obs from 2 steps ago."""
    p = ObsFixedDelay(obs_slice=slice(0, 3), obs_dim=3, n_envs=1, dt=0.01)
    p.tick(is_reset=True)
    p._delay[:] = 2

    obs_t0 = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    obs_t1 = torch.tensor([[2.0, 2.0, 2.0, 0.0]])
    obs_t2 = torch.tensor([[3.0, 3.0, 3.0, 0.0]])

    p.apply(obs_t0)  # buffer: [t0]
    p.apply(obs_t1)  # buffer: [t0, t1]
    result = p.apply(obs_t2)  # buffer: [t0, t1, t2], read delay=2 → t0
    assert torch.allclose(result[:, :3], obs_t0[:, :3], atol=1e-6)


# ---------------------------------------------------------------------------
# ActionFixedDelay — same logic for actions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_action_delay_zero_passthrough() -> None:
    """delay=0 must return the just-pushed action."""
    p = ActionFixedDelay(n_envs=2, dt=0.01, action_dim=4)
    p.tick(is_reset=True)
    p._delay[:] = 0
    action = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    result = p.apply(action)
    assert torch.allclose(result, action, atol=1e-6)


@pytest.mark.unit
def test_action_delay_returns_old_value() -> None:
    """delay=1 must return the action from 1 step ago."""
    p = ActionFixedDelay(n_envs=1, dt=0.01, action_dim=4)
    p.tick(is_reset=True)
    p._delay[:] = 1

    act_t0 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    act_t1 = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

    p.apply(act_t0)  # buffer: [t0]
    result = p.apply(act_t1)  # buffer: [t0, t1], read delay=1 → t0
    assert torch.allclose(result, act_t0, atol=1e-6)


# ---------------------------------------------------------------------------
# PacketLoss — ZOH behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_packet_loss_zoh_on_drop() -> None:
    """When drop mask is True, apply() must return last valid action."""
    p = PacketLoss(n_envs=2, dt=0.01, action_dim=4)
    p.tick(is_reset=True)

    # First action passes through for all envs
    action_1 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    p._drop_mask[:] = False
    result_1 = p.apply(action_1)
    assert torch.allclose(result_1, action_1, atol=1e-6)

    # Now drop env 0, pass env 1
    action_2 = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    p._drop_mask[0] = True
    p._drop_mask[1] = False
    result_2 = p.apply(action_2)
    # env 0: dropped → ZOH → action_1[0]
    assert torch.allclose(result_2[0], action_1[0], atol=1e-6)
    # env 1: passed → action_2[1]
    assert torch.allclose(result_2[1], action_2[1], atol=1e-6)


@pytest.mark.unit
def test_packet_loss_last_action_not_updated_on_drop() -> None:
    """_last_action must NOT be updated for dropped envs."""
    p = PacketLoss(n_envs=1, dt=0.01, action_dim=4)
    p.tick(is_reset=True)

    # Pass through first action
    action_1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    p._drop_mask[:] = False
    p.apply(action_1)
    assert torch.allclose(p._last_action, action_1, atol=1e-6)

    # Drop second action
    p._drop_mask[:] = True
    action_2 = torch.tensor([[99.0, 99.0, 99.0, 99.0]])
    p.apply(action_2)
    # _last_action should still be action_1
    assert torch.allclose(p._last_action, action_1, atol=1e-6)


# ---------------------------------------------------------------------------
# ComputationOverload — stall duration correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_computation_overload_stall_duration() -> None:
    """A stall of duration D must produce exactly D steps of freeze."""
    p = ComputationOverload(
        n_envs=1,
        dt=0.01,
        action_dim=4,
        distribution_params={
            "prob_low": 1.0,
            "prob_high": 1.0,  # always trigger
            "duration_low": 3,
            "duration_high": 3,  # fixed duration
        },
        bounds=(0.0, 1.0),
    )
    p.tick(is_reset=True)
    p.sample()  # sets _skip_prob=1.0

    # Pass the initial action
    action_init = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    p._skip_counter[:] = 0
    p._drop_mask = torch.zeros(1, dtype=torch.bool)
    result = p.apply(action_init)
    assert torch.allclose(result, action_init, atol=1e-6)

    # Trigger a stall (step will trigger because prob=1.0)
    p.step()
    # _skip_counter should be 3 (duration=3, just triggered)
    assert p._skip_counter.item() == 3, f"Expected 3, got {p._skip_counter.item()}"

    action_new = torch.tensor([[99.0, 99.0, 99.0, 99.0]])
    # Step 1 of stall: action should be held
    result = p.apply(action_new)
    assert torch.allclose(result, action_init, atol=1e-6), "Step 1 of stall should hold"

    # Advance: decrement to 2
    p.step()
    assert p._skip_counter.item() == 2
    result = p.apply(action_new)
    assert torch.allclose(result, action_init, atol=1e-6), "Step 2 of stall should hold"

    # Advance: decrement to 1
    p.step()
    assert p._skip_counter.item() == 1
    result = p.apply(action_new)
    assert torch.allclose(result, action_init, atol=1e-6), "Step 3 of stall should hold"


@pytest.mark.unit
def test_computation_overload_resumes_after_stall() -> None:
    """After stall counter reaches 0, new actions pass through (until re-trigger)."""
    p = ComputationOverload(
        n_envs=1,
        dt=0.01,
        action_dim=4,
        distribution_params={
            "prob_low": 0.0,
            "prob_high": 0.0,  # never re-trigger
            "duration_low": 1,
            "duration_high": 1,
        },
        bounds=(0.0, 1.0),
    )
    p.tick(is_reset=True)
    p.sample()

    # Manually set a stall of duration 1
    action_old = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    p._last_action = action_old.clone()
    p._skip_counter[:] = 1
    p._current_value[:] = 1.0

    # During stall: held
    action_new = torch.tensor([[9.0, 9.0, 9.0, 9.0]])
    result = p.apply(action_new)
    assert torch.allclose(result, action_old, atol=1e-6)

    # step() decrements counter to 0
    p.step()
    assert p._skip_counter.item() == 0

    # Now action passes through
    result = p.apply(action_new)
    assert torch.allclose(result, action_new, atol=1e-6)
