"""Category 6 — Action perturbations.

Implemented perturbations:
  6.1  ActionNoise            — additive noise on action vector
  6.2  ActionDeadzone         — zero-out small actions below threshold
  6.3  ActionSaturation       — clip action to reduced range
  6.4  ActuatorHysteresis     — direction-dependent offset (stateful)
  6.5  ESCLowPassFilter       — first-order IIR low-pass filter (stateful)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    ActionPerturbation,
    register,
)

# ---------------------------------------------------------------------------
# 6.1 Action noise
# ---------------------------------------------------------------------------


@register("action_noise")
class ActionNoise(ActionPerturbation):
    """6.1 — Additive noise on the action vector.

    At each step, samples noise from a gaussian or uniform distribution
    and adds it to the action tensor. The noise standard deviation (or
    half-width for uniform) is the perturbation value controlled by
    curriculum_scale.

    apply(action) = action + noise,  noise ~ N(0, σ²) or U(-σ, σ)

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: action vector dimension (default 4).
        noise_type: "gaussian" or "uniform".
        distribution_params: {"low", "high"} — noise std range.
        bounds: noise std clamp [min, max].
        nominal: 0.0 (no noise).
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        noise_type: str = "gaussian",
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.1),
        nominal: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 0.05}

        self._noise_type = noise_type

        super().__init__(
            id="action_noise",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )
        self._current_value = torch.zeros(n_envs)

    def apply(self, action: Tensor) -> Tensor:
        """Add noise to action vector.

        Args:
            action: [n_envs, action_dim] action tensor.

        Returns:
            Noisy action tensor [n_envs, action_dim].
        """
        assert self._current_value is not None, "call tick() before apply()"
        sigma = self._current_value.unsqueeze(1)  # [n_envs, 1]
        if self._noise_type == "gaussian":
            noise = torch.randn_like(action) * sigma
        else:
            noise = (torch.rand_like(action) * 2 - 1) * sigma
        return action + noise


# ---------------------------------------------------------------------------
# 6.2 Action deadzone
# ---------------------------------------------------------------------------


@register("action_deadzone")
class ActionDeadzone(ActionPerturbation):
    """6.2 — Zero-out action channels below a threshold.

    Samples a deadzone width per env per episode. Any action component
    with |a_i| < threshold is set to zero.

    apply(action) = where(|action| < threshold, 0, action)

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution_params: {"low", "high"} — threshold range.
        bounds: threshold clamp [min, max].
        nominal: 0.0 (no deadzone).
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.1),
        nominal: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 0.05}

        super().__init__(
            id="action_deadzone",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )

    def apply(self, action: Tensor) -> Tensor:
        """Zero out action channels below deadzone threshold.

        Args:
            action: [n_envs, action_dim] action tensor.

        Returns:
            Action with deadzone applied [n_envs, action_dim].
        """
        assert self._current_value is not None, "call tick() before apply()"
        threshold = self._current_value.unsqueeze(1)  # [n_envs, 1]
        return torch.where(action.abs() < threshold, 0.0, action)


# ---------------------------------------------------------------------------
# 6.3 Action saturation (reduced range)
# ---------------------------------------------------------------------------


@register("action_saturation")
class ActionSaturation(ActionPerturbation):
    """6.3 — Clip action to a reduced range.

    Samples a saturation limit per env per episode. The action is clipped
    to [-limit, +limit] where limit ∈ [0.5, 1.0] of the nominal range.

    apply(action) = clamp(action, -limit, +limit)

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution_params: {"low", "high"} — limit range.
        bounds: limit clamp [min, max].
        nominal: 1.0 (full range).
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 1.0),
        nominal: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.7, "high": 1.0}

        super().__init__(
            id="action_saturation",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )

    def apply(self, action: Tensor) -> Tensor:
        """Clip action to reduced range.

        Args:
            action: [n_envs, action_dim] action tensor.

        Returns:
            Saturated action [n_envs, action_dim].
        """
        assert self._current_value is not None, "call tick() before apply()"
        limit = self._current_value.unsqueeze(1)  # [n_envs, 1]
        return action.clamp(-limit, limit)


# ---------------------------------------------------------------------------
# 6.4 Actuator hysteresis
# ---------------------------------------------------------------------------


@register("actuator_hysteresis")
class ActuatorHysteresis(ActionPerturbation):
    """6.4 — Direction-dependent actuator hysteresis.

    Tracks the direction of RPM command changes. When the command direction
    reverses, a dead-band offset is applied proportional to the hysteresis
    width. The output is shifted by ±width/2 depending on direction.

    State variables:
      - _last_action [n_envs, 4]: previous action for direction detection
      - _width [n_envs, 4]: sampled hysteresis width per motor

    apply(action):
      direction = sign(action - _last_action)
      output = action + direction × width / 2
      _last_action = action

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: number of motors (default 4).
        distribution_params: {"low", "high"} — hysteresis width range.
        bounds: width clamp [min, max].
        nominal: 0.0 (no hysteresis).
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.05),
        nominal: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 0.03}

        self._action_dim = action_dim
        self._last_action: Tensor = torch.zeros(n_envs, action_dim)
        self._width: Tensor = torch.zeros(n_envs, action_dim)

        super().__init__(
            id="actuator_hysteresis",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        self._current_value = torch.zeros(n_envs)

    def reset(self, env_ids: Tensor) -> None:
        """Reset hysteresis state and sample width for selected envs."""
        n = len(env_ids)
        self._last_action[env_ids] = 0.0
        p = self.distribution_params
        width_raw = torch.empty(n).uniform_(p["low"], p["high"])
        self._width[env_ids] = (
            (width_raw * self.curriculum_scale).unsqueeze(1).expand(n, self._action_dim)
        )
        self._current_value[env_ids] = width_raw * self.curriculum_scale

    def step(self) -> None:
        """No-op — state is updated in apply()."""

    def apply(self, action: Tensor) -> Tensor:
        """Apply hysteresis offset based on command direction.

        Args:
            action: [n_envs, action_dim] action tensor.

        Returns:
            Action with hysteresis offset [n_envs, action_dim].
        """
        assert self._current_value is not None, "call tick() before apply()"
        direction = (action - self._last_action).sign()
        offset = direction * self._width * 0.5
        self._last_action.copy_(action)
        return action + offset


# ---------------------------------------------------------------------------
# 6.5 ESC low-pass filtering
# ---------------------------------------------------------------------------


@register("esc_low_pass_filter")
class ESCLowPassFilter(ActionPerturbation):
    """6.5 — First-order IIR low-pass filter on action commands.

    Models the ESC (Electronic Speed Controller) bandwidth limitation.
    The filtered output tracks the commanded action with a time constant
    determined by the cutoff frequency:

        y += (x − y) × α,  where α = 2π × f_c × dt

    At infinite cutoff: α → large → y = x (pass-through).
    At low cutoff: α → small → heavy smoothing.

    State variables:
      - _filtered [n_envs, 4]: current filtered output
      - _alpha [n_envs, 1]: per-env filter coefficient

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        action_dim: number of motors (default 4).
        distribution_params: {"low", "high"} — cutoff frequency range [Hz].
        bounds: cutoff freq clamp [min, max] Hz.
        nominal: 50.0 Hz (minimal filtering).
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        action_dim: int = 4,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (5.0, 50.0),
        nominal: float = 50.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 10.0, "high": 50.0}

        self._action_dim = action_dim
        self._filtered: Tensor = torch.zeros(n_envs, action_dim)
        self._alpha: Tensor = torch.ones(n_envs, 1)
        self._output_buf: Tensor = torch.zeros(n_envs, action_dim)

        super().__init__(
            id="esc_low_pass_filter",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        self._current_value = torch.full((n_envs,), float(nominal))

    def reset(self, env_ids: Tensor) -> None:
        """Reset filter state for selected envs."""
        self._filtered[env_ids] = 0.0
        # Recompute alpha from current cutoff frequency
        if self._current_value is not None:
            fc = self._current_value[env_ids]
            self._alpha[env_ids] = (2.0 * math.pi * fc * self.dt).clamp(0.0, 1.0).unsqueeze(1)

    def sample(self) -> Tensor:
        """Sample cutoff frequency and update cached alpha."""
        result = super().sample()
        # Cache alpha = clamp(2π × f_c × dt, 0, 1) for all envs
        fc = self._current_value  # [n_envs]
        self._alpha = (2.0 * math.pi * fc * self.dt).clamp(0.0, 1.0).unsqueeze(1)
        return result

    def step(self) -> None:
        """No-op — filtering happens in apply()."""

    def apply(self, action: Tensor) -> Tensor:
        """Apply first-order IIR low-pass filter.

        y += (x − y) × α,  where α is cached from sample()/reset().

        Args:
            action: [n_envs, action_dim] commanded action.

        Returns:
            Filtered action [n_envs, action_dim].
        """
        assert self._current_value is not None, "call tick() before apply()"
        self._filtered.add_((action - self._filtered) * self._alpha)
        self._output_buf.copy_(self._filtered)
        return self._output_buf
