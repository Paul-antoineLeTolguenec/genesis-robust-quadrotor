"""Category 8 — External disturbances.

Implemented perturbations:
  8.1  BodyForceDisturbance  — per-env external force (constant or OU-driven)
  8.2  BodyTorqueDisturbance — per-env external torque (constant or OU-driven)
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    EnvState,
    ExternalWrenchPerturbation,
    OUProcess,
    register,
)

# ---------------------------------------------------------------------------
# Shared base for body force/torque disturbances
# ---------------------------------------------------------------------------


class _BaseBodyDisturbance(ExternalWrenchPerturbation):
    """Shared logic for body force and torque disturbances.

    Supports two regimes:
      * **constant** (distribution ∈ {uniform, gaussian}): stateless, value sampled
        per episode or per step via the standard base class `sample()`.
      * **OU** (distribution = "ou_process"): stateful, value evolves as a 3D
        Ornstein-Uhlenbeck process per environment. In OU mode, only `step()`
        advances the state — `frequency` is set to `"per_episode"` so that
        `tick()` does not redundantly call `sample()` on the step path.

    Formula (OU mode):
        dw = θ(μ − w)dt + σ√dt·ε     with per-env σ sampled at reset.
    """

    def __init__(
        self,
        *,
        registry_id: str,
        wrench_type: Literal["force", "torque"],
        n_envs: int,
        dt: float,
        distribution: str,
        distribution_params: dict,
        bounds: tuple[float, float],
        nominal: list[float],
        lipschitz_k: float | None,
        frame: str,
        link_idx: int,
        duration_mode: str,
        **kwargs: Any,
    ) -> None:
        is_ou = distribution == "ou_process"

        # OU state — pre-allocated unconditionally (lightweight)
        self._ou = OUProcess(n_envs, 3)
        self._sigma: Tensor = torch.zeros(n_envs)
        # Pre-allocated column view for sigma broadcast in step()
        self._sigma_col: Tensor = self._sigma.unsqueeze(1)  # [n_envs, 1] view

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            id=registry_id,
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic" if is_ou else "fixed"),
            # OU uses per_episode so tick() does NOT call sample() on step path
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            wrench_type=wrench_type,
            **kwargs,
        )
        self.is_stateful = is_ou
        self._current_value = torch.zeros(n_envs, 3)

    # -- lifecycle overrides (OU mode) --

    def reset(self, env_ids: Tensor) -> None:
        """Reset OU state and sample per-env sigma (OU mode)."""
        if self.is_stateful:
            self._ou.reset(env_ids)
            p = self.distribution_params
            self._sigma[env_ids] = (
                torch.empty(len(env_ids))
                .uniform_(float(p["sigma_low"]), float(p["sigma_high"]))
                .mul_(self.curriculum_scale)
            )
            self._current_value[env_ids] = 0.0

    def step(self) -> None:
        """Advance OU process (OU mode only)."""
        if not self.is_stateful:
            return
        p = self.distribution_params
        theta = float(p["theta"])
        mu = float(p["mu"])
        state = self._ou.state
        noise = torch.randn_like(state)
        state.add_(theta * (mu - state) * self.dt + self._sigma_col * math.sqrt(self.dt) * noise)
        self._current_value.copy_(state).clamp_(self.bounds[0], self.bounds[1])

    def sample(self) -> Tensor:
        """Sample wrench value.

        OU mode: return clamped OU state (state advanced by step()).
        Constant mode: standard draw from distribution.
        """
        if self.is_stateful:
            self._current_value.copy_(self._ou.state).clamp_(self.bounds[0], self.bounds[1])
            return self._current_value
        return super().sample()

    # -- wrench computation --

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Return current wrench value [n_envs, 3] directly."""
        assert self._current_value is not None, "call tick() before apply()"
        return self._current_value


# ---------------------------------------------------------------------------
# 8.1 Body force disturbance
# ---------------------------------------------------------------------------


@register("body_force_disturbance")
class BodyForceDisturbance(_BaseBodyDisturbance):
    """8.1 — Per-env external body force disturbance.

    Applies a 3D force via ``apply_links_external_force``. Supports constant
    (uniform/gaussian) and OU-driven regimes.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform", "gaussian", or "ou_process".
        distribution_params: distribution-specific parameters.
        bounds: hard clamp per axis [min_N, max_N] — default [-5.0, +5.0].
        nominal: unperturbed force ([0, 0, 0] N).
        lipschitz_k: max |ΔF| per step in adversarial mode.
        frame: reference frame — "world" (default) or "local".
        link_idx: body link index (default 0).
        duration_mode: "continuous" (default) or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-5.0, 5.0),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = 0.5,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]
        if distribution_params is None:
            if distribution == "ou_process":
                distribution_params = {
                    "theta": 1.0,
                    "sigma_low": 0.5,
                    "sigma_high": 2.0,
                    "mu": 0.0,
                }
            else:
                distribution_params = {"low": -5.0, "high": 5.0}
        super().__init__(
            registry_id="body_force_disturbance",
            wrench_type="force",
            n_envs=n_envs,
            dt=dt,
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            lipschitz_k=lipschitz_k,
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 8.2 Body torque disturbance
# ---------------------------------------------------------------------------


@register("body_torque_disturbance")
class BodyTorqueDisturbance(_BaseBodyDisturbance):
    """8.2 — Per-env external body torque disturbance.

    Applies a 3D torque via ``apply_links_external_torque``. Supports constant
    (uniform/gaussian) and OU-driven regimes.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform", "gaussian", or "ou_process".
        distribution_params: distribution-specific parameters.
        bounds: hard clamp per axis [min_Nm, max_Nm] — default [-1.0, +1.0].
        nominal: unperturbed torque ([0, 0, 0] N·m).
        lipschitz_k: max |Δτ| per step in adversarial mode.
        frame: reference frame — "world" (default) or "local".
        link_idx: body link index (default 0).
        duration_mode: "continuous" (default) or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-1.0, 1.0),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = 0.05,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]
        if distribution_params is None:
            if distribution == "ou_process":
                distribution_params = {
                    "theta": 1.0,
                    "sigma_low": 0.05,
                    "sigma_high": 0.3,
                    "mu": 0.0,
                }
            else:
                distribution_params = {"low": -1.0, "high": 1.0}
        super().__init__(
            registry_id="body_torque_disturbance",
            wrench_type="torque",
            n_envs=n_envs,
            dt=dt,
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            lipschitz_k=lipschitz_k,
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            **kwargs,
        )
