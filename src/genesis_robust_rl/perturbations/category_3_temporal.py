"""Category 3 — Temporal / Latency perturbations.

Implemented perturbations:
  3.1  ObsFixedDelay          — fixed per-episode observation delay via DelayBuffer
  3.2  ObsVariableDelay       — variable per-step observation delay via DelayBuffer
  3.3  ActionFixedDelay       — fixed per-episode action delay via DelayBuffer
  3.4  ActionVariableDelay    — variable per-step action delay via DelayBuffer
  3.7  PacketLoss             — Bernoulli action dropout with zero-order hold
  3.8  ComputationOverload    — stochastic multi-step action freeze
"""

from __future__ import annotations

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    ActionPerturbation,
    DelayBuffer,
    ObservationPerturbation,
    register,
)

# ===================================================================
# 3.1 — ObsFixedDelay (ObservationPerturbation)
# ===================================================================


@register("obs_fixed_delay")
class ObsFixedDelay(ObservationPerturbation):
    """3.1 — Fixed per-episode observation delay via circular buffer.

    At episode reset, samples an integer delay d ∈ [low, high] per env.
    Each step, the observation slice is pushed into a DelayBuffer and
    the value read back d steps later is substituted.

    delay=0 is a pass-through (no latency added).

    Args:
        obs_slice: slice selecting the observation channels to delay.
        obs_dim: width of the obs slice (number of features).
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        max_delay: maximum delay in steps (DelayBuffer capacity).
        distribution_params: {"low", "high"} for integer delay sampling.
        bounds: hard clamp on delay value.
    """

    def __init__(
        self,
        obs_slice: slice,
        obs_dim: int,
        n_envs: int,
        dt: float = 0.01,
        max_delay: int = 10,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 10.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="obs_fixed_delay",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed",
            frequency="per_episode",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0, "high": max_delay},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self.obs_dim = obs_dim
        # Buffer capacity = max_delay + 1 so delay=max_delay is valid
        self._buffer = DelayBuffer(n_envs, max_delay + 1, obs_dim)
        self._delay: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._nominal_t = torch.tensor(nominal, dtype=torch.float32)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset delay buffer and delay values for selected envs."""
        self._buffer.reset(env_ids)
        self._delay[env_ids] = 0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Draw integer delay per env, store in _delay and _current_value."""
        raw = self._draw().squeeze(-1)  # [n_envs]
        scaled = self._nominal_t + (raw - self._nominal_t) * self.curriculum_scale
        scaled.clamp_(self.bounds[0], self.bounds[1])
        self._delay = scaled.round().long()
        self._current_value = self._delay.float().unsqueeze(-1)
        return self._current_value

    def step(self) -> None:
        """No-op — buffer operations happen in apply()."""

    def apply(self, obs: Tensor) -> Tensor:
        """Push obs slice into buffer and read delayed value.

        Args:
            obs: observation tensor [n_envs, obs_dim_total].

        Returns:
            Modified obs with the selected slice delayed.
        """
        out = obs.clone()
        self._buffer.push(out[:, self.obs_slice])
        out[:, self.obs_slice] = self._buffer.read(self._delay)
        return out


# ===================================================================
# 3.2 — ObsVariableDelay (ObservationPerturbation)
# ===================================================================


@register("obs_variable_delay")
class ObsVariableDelay(ObservationPerturbation):
    """3.2 — Variable per-step observation delay via circular buffer.

    Same mechanism as ObsFixedDelay but the integer delay is resampled
    every step, producing time-varying latency.

    Args:
        obs_slice: slice selecting the observation channels to delay.
        obs_dim: width of the obs slice (number of features).
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        max_delay: maximum delay in steps (DelayBuffer capacity).
        distribution_params: {"low", "high"} for integer delay sampling.
        bounds: hard clamp on delay value.
    """

    def __init__(
        self,
        obs_slice: slice,
        obs_dim: int,
        n_envs: int,
        dt: float = 0.01,
        max_delay: int = 5,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 5.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="obs_variable_delay",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0, "high": max_delay},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self.obs_dim = obs_dim
        # Buffer capacity = max_delay + 1 so delay=max_delay is valid
        self._buffer = DelayBuffer(n_envs, max_delay + 1, obs_dim)
        self._delay: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._nominal_t = torch.tensor(nominal, dtype=torch.float32)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset delay buffer and delay values for selected envs."""
        self._buffer.reset(env_ids)
        self._delay[env_ids] = 0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Draw integer delay per env, store in _delay and _current_value."""
        raw = self._draw().squeeze(-1)  # [n_envs]
        scaled = self._nominal_t + (raw - self._nominal_t) * self.curriculum_scale
        scaled.clamp_(self.bounds[0], self.bounds[1])
        self._delay = scaled.round().long()
        self._current_value = self._delay.float().unsqueeze(-1)
        return self._current_value

    def step(self) -> None:
        """No-op — buffer operations happen in apply()."""

    def apply(self, obs: Tensor) -> Tensor:
        """Push obs slice into buffer and read delayed value.

        Args:
            obs: observation tensor [n_envs, obs_dim_total].

        Returns:
            Modified obs with the selected slice delayed.
        """
        out = obs.clone()
        self._buffer.push(out[:, self.obs_slice])
        out[:, self.obs_slice] = self._buffer.read(self._delay)
        return out


# ===================================================================
# 3.3 — ActionFixedDelay (ActionPerturbation)
# ===================================================================


@register("action_fixed_delay")
class ActionFixedDelay(ActionPerturbation):
    """3.3 — Fixed per-episode action delay via circular buffer.

    At episode reset, samples an integer delay d ∈ [low, high] per env.
    Each step, the action is pushed into a DelayBuffer and the value
    read back d steps later is returned to the simulator.

    delay=0 is a pass-through (no latency added).

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: action vector width (default 4 for quadrotor).
        max_delay: maximum delay in steps (DelayBuffer capacity).
        distribution_params: {"low", "high"} for integer delay sampling.
        bounds: hard clamp on delay value.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        max_delay: int = 5,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 5.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            id="action_fixed_delay",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed",
            frequency="per_episode",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0, "high": max_delay},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self.action_dim = action_dim
        # Buffer capacity = max_delay + 1 so delay=max_delay is valid
        self._buffer = DelayBuffer(n_envs, max_delay + 1, action_dim)
        self._delay: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._nominal_t = torch.tensor(nominal, dtype=torch.float32)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset delay buffer and delay values for selected envs."""
        self._buffer.reset(env_ids)
        self._delay[env_ids] = 0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Draw integer delay per env, store in _delay and _current_value."""
        raw = self._draw().squeeze(-1)  # [n_envs]
        scaled = self._nominal_t + (raw - self._nominal_t) * self.curriculum_scale
        scaled.clamp_(self.bounds[0], self.bounds[1])
        self._delay = scaled.round().long()
        self._current_value = self._delay.float().unsqueeze(-1)
        return self._current_value

    def step(self) -> None:
        """No-op — buffer operations happen in apply()."""

    def apply(self, action: Tensor) -> Tensor:
        """Push action into buffer and read delayed value.

        Args:
            action: action tensor [n_envs, action_dim].

        Returns:
            Delayed action tensor [n_envs, action_dim].
        """
        self._buffer.push(action)
        return self._buffer.read(self._delay)


# ===================================================================
# 3.4 — ActionVariableDelay (ActionPerturbation)
# ===================================================================


@register("action_variable_delay")
class ActionVariableDelay(ActionPerturbation):
    """3.4 — Variable per-step action delay via circular buffer.

    Same mechanism as ActionFixedDelay but the integer delay is resampled
    every step, producing time-varying latency.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: action vector width (default 4 for quadrotor).
        max_delay: maximum delay in steps (DelayBuffer capacity).
        distribution_params: {"low", "high"} for integer delay sampling.
        bounds: hard clamp on delay value.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        max_delay: int = 5,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 5.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            id="action_variable_delay",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0, "high": max_delay},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self.action_dim = action_dim
        # Buffer capacity = max_delay + 1 so delay=max_delay is valid
        self._buffer = DelayBuffer(n_envs, max_delay + 1, action_dim)
        self._delay: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._nominal_t = torch.tensor(nominal, dtype=torch.float32)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset delay buffer and delay values for selected envs."""
        self._buffer.reset(env_ids)
        self._delay[env_ids] = 0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Draw integer delay per env, store in _delay and _current_value."""
        raw = self._draw().squeeze(-1)  # [n_envs]
        scaled = self._nominal_t + (raw - self._nominal_t) * self.curriculum_scale
        scaled.clamp_(self.bounds[0], self.bounds[1])
        self._delay = scaled.round().long()
        self._current_value = self._delay.float().unsqueeze(-1)
        return self._current_value

    def step(self) -> None:
        """No-op — buffer operations happen in apply()."""

    def apply(self, action: Tensor) -> Tensor:
        """Push action into buffer and read delayed value.

        Args:
            action: action tensor [n_envs, action_dim].

        Returns:
            Delayed action tensor [n_envs, action_dim].
        """
        self._buffer.push(action)
        return self._buffer.read(self._delay)


# ===================================================================
# 3.7 — PacketLoss (ActionPerturbation)
# ===================================================================


@register("packet_loss")
class PacketLoss(ActionPerturbation):
    """3.7 — Bernoulli action dropout with zero-order hold.

    Each step, independently per env, the action packet is dropped with
    probability p (sampled per step from the configured distribution).
    When dropped, the previous action is held (zero-order hold).

    _current_value exposes the binary drop mask [n_envs, 1]:
      1.0 = packet dropped (ZOH active), 0.0 = packet passed through.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: action vector width (default 4 for quadrotor).
        distribution_params: {"low", "high"} for drop probability sampling.
        bounds: hard clamp on drop probability.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.3),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            id="packet_loss",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.3},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self.action_dim = action_dim
        self._last_action: Tensor = torch.zeros(n_envs, action_dim)
        self._drop_prob: Tensor = torch.zeros(n_envs)
        self._drop_mask: Tensor = torch.zeros(n_envs, dtype=torch.bool)
        self._nominal_t = torch.tensor(nominal, dtype=torch.float32)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset held action and drop state for selected envs."""
        self._last_action[env_ids] = 0.0
        self._drop_prob[env_ids] = 0.0
        self._drop_mask[env_ids] = False
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Sample drop probability per env from the configured distribution."""
        raw = self._draw().squeeze(-1)  # [n_envs]
        scaled = self._nominal_t + (raw - self._nominal_t) * self.curriculum_scale
        self._drop_prob = scaled.clamp(self.bounds[0], self.bounds[1])
        return self._drop_prob.unsqueeze(-1)

    def step(self) -> None:
        """Draw Bernoulli drop mask from current drop probabilities."""
        self._drop_mask = torch.bernoulli(self._drop_prob).bool()
        self._current_value = self._drop_mask.unsqueeze(-1).float()

    def apply(self, action: Tensor) -> Tensor:
        """Apply packet loss: dropped envs get ZOH, others pass through.

        Args:
            action: action tensor [n_envs, action_dim].

        Returns:
            Action tensor with dropped packets replaced by last action.
        """
        mask = self._drop_mask.unsqueeze(-1).expand_as(action)
        result = torch.where(mask, self._last_action, action)
        # Update last action for envs that passed through
        self._last_action = torch.where(mask, self._last_action, action)
        return result


# ===================================================================
# 3.8 — ComputationOverload (ActionPerturbation)
# ===================================================================


@register("computation_overload")
class ComputationOverload(ActionPerturbation):
    """3.8 — Stochastic multi-step action freeze (computation overload).

    Simulates computation overload events where the controller stalls for
    a random duration. During a stall, the last valid action is held
    (zero-order hold).

    Each step, for envs not currently stalled:
      - draw Bernoulli(skip_prob) to trigger a new stall event
      - if triggered, sample a random duration ∈ [duration_low, duration_high]
    For envs currently stalled:
      - decrement the skip counter

    _current_value exposes the stall indicator [n_envs, 1]:
      1.0 = currently stalled, 0.0 = normal operation.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: action vector width (default 4 for quadrotor).
        distribution_params: {"prob_low", "prob_high", "duration_low", "duration_high"}.
        bounds: hard clamp on skip probability.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.1),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            id="computation_overload",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params
            or {"prob_low": 0.0, "prob_high": 0.1, "duration_low": 1, "duration_high": 5},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self.action_dim = action_dim
        self._skip_counter: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._last_action: Tensor = torch.zeros(n_envs, action_dim)
        self._skip_prob: Tensor = torch.zeros(n_envs)
        self._nominal_t = torch.tensor(nominal, dtype=torch.float32)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset skip counter, held action, and skip prob for selected envs."""
        self._skip_counter[env_ids] = 0
        self._last_action[env_ids] = 0.0
        self._skip_prob[env_ids] = 0.0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def _draw(self) -> Tensor:
        """Draw skip probability from uniform [prob_low, prob_high].

        Overrides base _draw() because distribution_params has non-standard keys.
        """
        p = self.distribution_params
        shape = self._batch_shape()
        return torch.empty(shape).uniform_(float(p["prob_low"]), float(p["prob_high"]))

    def sample(self) -> Tensor:
        """Sample skip probability per env."""
        raw = self._draw().squeeze(-1)  # [n_envs]
        scaled = self._nominal_t + (raw - self._nominal_t) * self.curriculum_scale
        self._skip_prob = scaled.clamp(self.bounds[0], self.bounds[1])
        return self._skip_prob.unsqueeze(-1)

    def step(self) -> None:
        """Advance stall logic: decrement active counters, then trigger new events."""
        p = self.distribution_params
        dur_lo = int(p["duration_low"])
        dur_hi = int(p["duration_high"])

        # Decrement active counters first
        active = self._skip_counter > 0
        self._skip_counter.sub_(active.long())

        # Envs eligible for a new stall event (counter just reached 0 or was already 0)
        idle = self._skip_counter == 0
        # Draw Bernoulli trigger for idle envs
        trigger = torch.bernoulli(self._skip_prob).bool() & idle
        # Sample random duration for triggered envs
        durations = torch.randint(dur_lo, dur_hi + 1, (self.n_envs,), dtype=torch.long)
        self._skip_counter = torch.where(trigger, durations, self._skip_counter)

        # Update privileged obs
        self._current_value = (self._skip_counter > 0).unsqueeze(-1).float()

    def apply(self, action: Tensor) -> Tensor:
        """Apply computation overload: stalled envs get ZOH, others pass through.

        Args:
            action: action tensor [n_envs, action_dim].

        Returns:
            Action tensor with stalled envs holding their last valid action.
        """
        stalled = (self._current_value.squeeze(-1) > 0.5)
        mask = stalled.unsqueeze(-1).expand_as(action)
        result = torch.where(mask, self._last_action, action)
        # Update last action for envs that are NOT stalled
        self._last_action = torch.where(mask, self._last_action, action)
        return result
