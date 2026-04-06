"""Category 5 — Wind and external forces perturbations.

Implemented perturbations:
  5.1  ConstantWind               — per-env constant wind force (per episode)
  5.2  Turbulence                 — stochastic wind via OU process
  5.3  WindGust                   — random impulse wind events
  5.4  WindShear                  — altitude-dependent wind gradient
  5.5  AdversarialWind            — adversary-controlled wind force
  5.6  BladeVortexInteraction     — rotor wake ingestion (OU per rotor)
  5.7  GroundEffectBoundary       — dynamic ground effect transition
  5.8  PayloadSway                — suspended load pendulum dynamics
  5.9  ProximityDisturbance       — wall/ceiling aerodynamic effect
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    EnvState,
    ExternalWrenchPerturbation,
    OUProcess,
    register,
)

# ---------------------------------------------------------------------------
# Shared drag helper
# ---------------------------------------------------------------------------

_DEFAULT_DRAG_COEFF = 0.1  # effective Cd×A×ρ/2 [N·s²/m²]


def _drag_force(v_wind: Tensor, v_drone: Tensor, drag_coeff: float) -> Tensor:
    """Compute aerodynamic drag force from relative wind velocity.

    F = -drag_coeff × sign(v_rel) × v_rel²   (per axis, [n_envs, 3])
    Convention: force on the drone from the wind, so positive v_rel → positive force.
    """
    v_rel = v_wind - v_drone  # [n_envs, 3]
    return drag_coeff * v_rel.sign() * v_rel.pow(2)


# ---------------------------------------------------------------------------
# 5.1 Constant wind
# ---------------------------------------------------------------------------


@register("constant_wind")
class ConstantWind(ExternalWrenchPerturbation):
    """5.1 — Per-env constant wind velocity sampled per episode.

    Samples a 3D wind velocity vector per environment. The aerodynamic force
    applied to the drone at each step is:

        F = drag_coeff × sign(v_wind − v_drone) × (v_wind − v_drone)²

    The perturbation value (_current_value) is the wind velocity (m/s),
    preserved for privileged observation.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        drag_coeff: effective drag coefficient Cd×A×ρ/2 [N·s²/m²].
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: per-axis clamp [min, max] m/s.
        nominal: unperturbed wind velocity [0, 0, 0] m/s.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        drag_coeff: float = _DEFAULT_DRAG_COEFF,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-10.0, 10.0),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": -5.0, "high": 5.0}
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]

        self._drag_coeff = drag_coeff

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="constant_wind",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute wind drag force from constant wind velocity."""
        assert self._current_value is not None, "call tick() before apply()"
        return _drag_force(self._current_value, env_state.vel, self._drag_coeff)


# ---------------------------------------------------------------------------
# 5.2 Turbulence (stochastic wind)
# ---------------------------------------------------------------------------


@register("turbulence")
class Turbulence(ExternalWrenchPerturbation):
    """5.2 — Stochastic wind via Ornstein-Uhlenbeck process.

    Wind velocity evolves as a 3D OU process per environment:

        dv = θ(μ − v)dt + σ√dt·ε

    The OU sigma is sampled per-env per episode from [sigma_low, sigma_high].
    The drag force applied each step uses the same model as ConstantWind.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        drag_coeff: effective drag coefficient.
        theta: OU mean-reversion rate.
        mu: OU long-term mean (per axis, default 0).
        distribution_params: {"sigma_low", "sigma_high", "theta"}.
        bounds: per-axis wind velocity clamp [min, max] m/s.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        drag_coeff: float = _DEFAULT_DRAG_COEFF,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-5.0, 5.0),
        nominal: list[float] | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {
                "sigma_low": 0.5,
                "sigma_high": 2.0,
                "theta": 1.0,
                "mu": 0.0,
            }
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]

        self._drag_coeff = drag_coeff
        self._ou = OUProcess(n_envs, 3)
        self._sigma: Tensor = torch.ones(n_envs)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="turbulence",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        self._current_value = torch.zeros(n_envs, 3)

    def reset(self, env_ids: Tensor) -> None:
        """Reset OU state and sample per-env sigma."""
        self._ou.reset(env_ids)
        p = self.distribution_params
        n = len(env_ids)
        sigma_raw = torch.empty(n).uniform_(p["sigma_low"], p["sigma_high"])
        self._sigma[env_ids] = sigma_raw * self.curriculum_scale
        self._current_value[env_ids] = 0.0

    def sample(self) -> Tensor:
        """Return current wind velocity from OU state (no distribution draw)."""
        self._current_value = self._ou.state.clamp(self.bounds[0], self.bounds[1])
        return self._current_value

    def step(self) -> None:
        """Advance OU process with per-env sigma.

        Inline OU update: dx = θ(μ−x)dt + σ_i·√dt·ε
        Uses per-env sigma stored in self._sigma.
        """
        p = self.distribution_params
        theta = float(p["theta"])
        mu = float(p["mu"])
        state = self._ou.state
        noise = torch.randn_like(state)
        # Per-env sigma broadcast: [n_envs, 1] × [n_envs, 3]
        sigma = self._sigma.unsqueeze(1)
        state.add_(theta * (mu - state) * self.dt + sigma * math.sqrt(self.dt) * noise)
        self._current_value = state.clamp(self.bounds[0], self.bounds[1])

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute drag force from OU wind velocity."""
        assert self._current_value is not None, "call tick() before apply()"
        return _drag_force(self._current_value, env_state.vel, self._drag_coeff)


# ---------------------------------------------------------------------------
# 5.3 Wind gust (impulse)
# ---------------------------------------------------------------------------


@register("wind_gust")
class WindGust(ExternalWrenchPerturbation):
    """5.3 — Random wind gust impulse events.

    At each step, a gust may trigger with sampled probability. When active,
    a force impulse is applied for a sampled duration (in steps). The gust
    direction is random (uniform on S²), magnitude sampled from bounds.

    State variables:
      - gust_counter [n_envs]: remaining steps for active gust (0 = inactive)
      - gust_force [n_envs, 3]: force vector during active gust
      - gust_prob [n_envs]: per-env trigger probability (sampled per episode)

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution_params: {"prob_low", "prob_high", "mag_low", "mag_high",
                               "duration_low", "duration_high"}.
        bounds: per-axis force clamp [min, max] N.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-15.0, 15.0),
        nominal: list[float] | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "pulse",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {
                "prob_low": 0.0,
                "prob_high": 0.05,
                "mag_low": 1.0,
                "mag_high": 10.0,
                "duration_low": 1,
                "duration_high": 20,
            }
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]

        # Pre-allocate state tensors
        self._gust_counter: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._gust_force: Tensor = torch.zeros(n_envs, 3)
        self._gust_prob: Tensor = torch.zeros(n_envs)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="wind_gust",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        self._current_value = torch.zeros(n_envs, 3)

    def reset(self, env_ids: Tensor) -> None:
        """Reset gust state and sample per-env trigger probability."""
        p = self.distribution_params
        n = len(env_ids)
        self._gust_counter[env_ids] = 0
        self._gust_force[env_ids] = 0.0
        prob_raw = torch.empty(n).uniform_(p["prob_low"], p["prob_high"])
        self._gust_prob[env_ids] = prob_raw * self.curriculum_scale
        self._current_value[env_ids] = 0.0

    def sample(self) -> Tensor:
        """Return current gust force (no distribution draw — event-based)."""
        return self._current_value

    def step(self) -> None:
        """Advance gust counters; trigger new gusts stochastically."""
        p = self.distribution_params
        # Decrement active gusts
        active = self._gust_counter > 0
        self._gust_counter[active] -= 1

        # Check for new gust triggers (only on inactive envs)
        inactive = ~active
        if inactive.any():
            trigger = torch.rand(self.n_envs) < self._gust_prob
            new_gust = inactive & trigger
            n_new = new_gust.sum().item()
            if n_new > 0:
                # Sample duration
                dur = torch.randint(
                    int(p["duration_low"]),
                    int(p["duration_high"]) + 1,
                    (n_new,),
                )
                self._gust_counter[new_gust] = dur

                # Sample magnitude and random direction on S²
                mag = torch.empty(n_new).uniform_(p["mag_low"], p["mag_high"])
                direction = torch.randn(n_new, 3)
                direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
                self._gust_force[new_gust] = direction * mag.unsqueeze(1)

        # Zero force for expired gusts
        expired = self._gust_counter == 0
        self._gust_force[expired] = 0.0

        # Clamp to bounds
        self._current_value = self._gust_force.clamp(self.bounds[0], self.bounds[1])

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Return current gust force vector."""
        assert self._current_value is not None, "call tick() before apply()"
        return self._current_value


# ---------------------------------------------------------------------------
# 5.4 Wind shear (altitude-dependent)
# ---------------------------------------------------------------------------


@register("wind_shear")
class WindShear(ExternalWrenchPerturbation):
    """5.4 — Altitude-dependent wind gradient.

    Wind velocity increases linearly with altitude:

        v_wind = gradient × altitude × wind_direction

    The gradient (m/s per m altitude) is sampled per episode. Wind direction
    is sampled uniformly on the horizontal plane (XY). The drag force is then
    computed from the relative wind velocity.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        drag_coeff: effective drag coefficient.
        distribution_params: {"low", "high"} — gradient range.
        bounds: gradient clamp [min, max] (m/s)/m.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        drag_coeff: float = _DEFAULT_DRAG_COEFF,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 2.0),
        nominal: float = 0.0,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 1.5}

        self._drag_coeff = drag_coeff
        # Wind direction per env (horizontal, sampled at reset)
        self._wind_direction: Tensor = torch.zeros(n_envs, 3)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="wind_shear",
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
        self._current_value = torch.full((n_envs,), float(nominal))

    def reset(self, env_ids: Tensor) -> None:
        """Sample wind direction (horizontal) for selected envs."""
        n = len(env_ids)
        angle = torch.empty(n).uniform_(0.0, 2.0 * math.pi)
        self._wind_direction[env_ids, 0] = angle.cos()
        self._wind_direction[env_ids, 1] = angle.sin()
        self._wind_direction[env_ids, 2] = 0.0

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute wind force from altitude-dependent shear profile."""
        assert self._current_value is not None, "call tick() before apply()"
        gradient = self._current_value  # [n_envs] or [1]
        if gradient.dim() == 0:
            gradient = gradient.unsqueeze(0)
        altitude = env_state.pos[:, 2].abs()  # [n_envs]
        wind_speed = gradient.squeeze(-1) * altitude  # [n_envs]
        v_wind = self._wind_direction * wind_speed.unsqueeze(1)  # [n_envs, 3]
        return _drag_force(v_wind, env_state.vel, self._drag_coeff)


# ---------------------------------------------------------------------------
# 5.5 Adversarial wind
# ---------------------------------------------------------------------------


@register("adversarial_wind")
class AdversarialWind(ExternalWrenchPerturbation):
    """5.5 — Adversary-controlled wind force with Lipschitz constraint.

    Same physical model as 5.1 (ConstantWind) but with mode=dynamic and
    Lipschitz enforcement. The adversarial agent sets the wind velocity
    each step via set_value(), bounded by lipschitz_k.

    In DR mode, functions identically to ConstantWind with per_step sampling.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        drag_coeff: effective drag coefficient.
        lipschitz_k: max wind velocity change per step (m/s per second).
        bounds: per-axis wind velocity clamp [min, max] m/s.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        drag_coeff: float = _DEFAULT_DRAG_COEFF,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-10.0, 10.0),
        nominal: list[float] | None = None,
        lipschitz_k: float = 5.0,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": -5.0, "high": 5.0}
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]

        self._drag_coeff = drag_coeff

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="adversarial_wind",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._current_value = torch.zeros(n_envs, 3)
        self._params_prev = dict(self.distribution_params)

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute wind drag force from adversary-controlled velocity."""
        assert self._current_value is not None, "call tick() before apply()"
        return _drag_force(self._current_value, env_state.vel, self._drag_coeff)


# ---------------------------------------------------------------------------
# 5.6 Blade-vortex interaction (rotor wake ingestion)
# ---------------------------------------------------------------------------


@register("blade_vortex_interaction")
class BladeVortexInteraction(ExternalWrenchPerturbation):
    """5.6 — Stochastic rotor wake ingestion via OU process per rotor.

    Models blade-vortex interaction as an OU process on thrust perturbation
    for each of the 4 rotors. The perturbation is a fractional thrust change:

        ΔT_j = x_j × T_nominal_j    where x_j ∈ [-0.1, +0.1]

    The net additional force is:

        F_z = Σ_j ΔT_j = Σ_j x_j × KF × ω_j²

    Applied as upward force in world frame.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KF: nominal thrust coefficient [N/RPM²].
        distribution_params: {"sigma_low", "sigma_high", "theta", "mu"}.
        bounds: per-rotor fractional bounds [-0.1, +0.1].
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KF: float = 3.16e-10,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-0.1, 0.1),
        nominal: list[float] | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {
                "sigma_low": 0.01,
                "sigma_high": 0.05,
                "theta": 2.0,
                "mu": 0.0,
            }
        if nominal is None:
            nominal = [0.0, 0.0, 0.0, 0.0]

        self._KF = KF
        self._ou = OUProcess(n_envs, 4)
        self._sigma: Tensor = torch.ones(n_envs)
        self._force_buf: Tensor = torch.zeros(n_envs, 3)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="blade_vortex_interaction",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(4,),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        self._current_value = torch.zeros(n_envs, 4)

    def reset(self, env_ids: Tensor) -> None:
        """Reset OU state and sample per-env sigma."""
        self._ou.reset(env_ids)
        p = self.distribution_params
        n = len(env_ids)
        sigma_raw = torch.empty(n).uniform_(p["sigma_low"], p["sigma_high"])
        self._sigma[env_ids] = sigma_raw * self.curriculum_scale
        self._current_value[env_ids] = 0.0

    def sample(self) -> Tensor:
        """Return current OU state clamped to bounds."""
        self._current_value = self._ou.state.clamp(self.bounds[0], self.bounds[1])
        return self._current_value

    def step(self) -> None:
        """Advance OU process with per-env sigma for rotor wake perturbation.

        Inline OU update: dx = θ(μ−x)dt + σ_i·√dt·ε
        """
        p = self.distribution_params
        theta = float(p["theta"])
        mu = float(p["mu"])
        state = self._ou.state
        noise = torch.randn_like(state)
        sigma = self._sigma.unsqueeze(1)  # [n_envs, 1]
        state.add_(theta * (mu - state) * self.dt + sigma * math.sqrt(self.dt) * noise)
        self._current_value = state.clamp(self.bounds[0], self.bounds[1])

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute net thrust perturbation force from BVI.

        F_z = Σ_j x_j × KF × ω_j²   (upward world +Z)
        """
        assert self._current_value is not None, "call tick() before apply()"
        x = self._current_value  # [n_envs, 4] fractional perturbation
        rpm = env_state.rpm  # [n_envs, 4]
        delta_thrust = (x * self._KF * rpm.pow(2)).sum(dim=1)  # [n_envs]
        self._force_buf.zero_()
        self._force_buf[:, 2] = delta_thrust
        return self._force_buf


# ---------------------------------------------------------------------------
# 5.7 Ground effect boundary transition
# ---------------------------------------------------------------------------


@register("ground_effect_boundary")
class GroundEffectBoundary(ExternalWrenchPerturbation):
    """5.7 — Dynamic ground effect in/out transition.

    Distinct from 1.11 (GroundEffect): this captures the rapid thrust
    variation when the drone crosses the ground-effect boundary. Uses
    Cheeseman-Bennett with a perturbable rotor diameter that sets the
    transition zone (sampled per episode).

    Transition zone: altitude ∈ [1×D_rotor, 3×D_rotor].
    Below: full ground effect. Above: no effect. Between: smooth interpolation.

    The thrust multiplier gradient is sampled as the perturbation value.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        nominal_thrust: hover thrust [N].
        rotor_diameter_nom: nominal rotor diameter [m].
        distribution_params: {"low", "high"} — rotor diameter perturbation range.
        bounds: diameter bounds [min, max] m.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        nominal_thrust: float = 9.81 * 0.027,
        rotor_diameter_nom: float = 0.092,
        max_k_ge: float = 2.0,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.05, 0.2),
        nominal: float = 0.092,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.07, "high": 0.15}

        self._nominal_thrust = nominal_thrust
        self._max_k_ge = max_k_ge
        self._force_buf: Tensor = torch.zeros(n_envs, 3)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="ground_effect_boundary",
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
        self._current_value = torch.full((n_envs,), float(nominal))

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute ground-effect force from altitude and sampled rotor diameter.

        k_ge = 1 / (1 − (R/(4h))²),  R = diameter/2
        ΔF_z = (k_ge − 1) × nominal_thrust
        """
        assert self._current_value is not None, "call tick() before apply()"
        diameter = self._current_value  # [n_envs] or scalar
        if diameter.dim() == 0:
            diameter = diameter.unsqueeze(0)
        diameter = diameter.squeeze(-1)  # ensure [n_envs]
        R = diameter / 2.0  # [n_envs]
        h = env_state.pos[:, 2].abs()  # [n_envs]

        # Safe altitude: clamp to avoid singularity
        h_safe = torch.max(h, R / 2.0)

        # Cheeseman-Bennett
        ratio = R / (4.0 * h_safe)
        k_ge = 1.0 / (1.0 - ratio.pow(2))
        k_ge = k_ge.clamp(1.0, self._max_k_ge)

        # Zero effect above transition zone (h > 3 × diameter)
        above = h > 3.0 * diameter
        k_ge.masked_fill_(above, 1.0)

        delta_f = (k_ge - 1.0) * self._nominal_thrust
        self._force_buf.zero_()
        self._force_buf[:, 2] = delta_f
        return self._force_buf


# ---------------------------------------------------------------------------
# 5.8 Payload sway (suspended load pendulum dynamics)
# ---------------------------------------------------------------------------


@register("payload_sway")
class PayloadSway(ExternalWrenchPerturbation):
    """5.8 — Suspended load pendulum dynamics.

    Models a point mass payload hanging from the drone by a rigid cable.
    The pendulum state (θ_x, θ_y, θ̇_x, θ̇_y) evolves via symplectic Euler:

        θ̈ = −(g/L)sin(θ) + a_drone/L   (per axis, small-angle OK)

    The reaction force on the drone is:

        F_reaction = −m_payload × (L × θ̈ + a_drone)

    Simplified: tension + centripetal terms projected back.

    State: θ [n_envs, 2], θ_dot [n_envs, 2], cable_length [n_envs], payload_mass [n_envs].

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution_params: {"length_low", "length_high", "mass_low", "mass_high"}.
        bounds: per-axis reaction force clamp [min, max] N.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-5.0, 5.0),
        nominal: list[float] | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {
                "length_low": 0.1,
                "length_high": 1.0,
                "mass_low": 0.01,
                "mass_high": 0.5,
            }
        if nominal is None:
            nominal = [0.0, 0.0, 0.0, 0.0]

        self._g = 9.81
        # Pendulum state
        self._theta: Tensor = torch.zeros(n_envs, 2)
        self._theta_dot: Tensor = torch.zeros(n_envs, 2)
        self._cable_length: Tensor = torch.ones(n_envs) * 0.5
        self._payload_mass: Tensor = torch.zeros(n_envs)
        self._force_buf: Tensor = torch.zeros(n_envs, 3)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="payload_sway",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(4,),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        self._current_value = torch.zeros(n_envs, 4)

    def reset(self, env_ids: Tensor) -> None:
        """Reset pendulum state and sample cable length / payload mass."""
        p = self.distribution_params
        n = len(env_ids)
        # Small initial angle perturbation to break symmetry
        self._theta[env_ids] = torch.randn(n, 2) * 0.1
        self._theta_dot[env_ids] = 0.0

        length_raw = torch.empty(n).uniform_(p["length_low"], p["length_high"])
        mass_raw = torch.empty(n).uniform_(p["mass_low"], p["mass_high"])
        self._cable_length[env_ids] = length_raw
        self._payload_mass[env_ids] = mass_raw * self.curriculum_scale
        self._current_value[env_ids] = 0.0

    def sample(self) -> Tensor:
        """Return current pendulum state [θ_x, θ_y, θ̇_x, θ̇_y]."""
        self._current_value[:, :2] = self._theta
        self._current_value[:, 2:] = self._theta_dot
        return self._current_value

    def step(self) -> None:
        """Integrate pendulum dynamics via symplectic Euler."""
        g = self._g
        L = self._cable_length.unsqueeze(1)  # [n_envs, 1]

        # θ̈ = −(g/L)sin(θ)
        theta_ddot = -(g / L) * self._theta.sin()  # [n_envs, 2]

        # Symplectic Euler: update velocity first, then position (in-place)
        self._theta_dot.add_(theta_ddot * self.dt)
        self._theta.add_(self._theta_dot * self.dt)

        # Clamp angles to avoid instability
        self._theta.clamp_(-math.pi / 2, math.pi / 2)

        self._current_value[:, :2] = self._theta
        self._current_value[:, 2:] = self._theta_dot

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute reaction force from pendulum on drone body.

        Horizontal reaction: F_xy = −m × g × sin(θ)
        Vertical: F_z = −m × g × (cos(θ) − 1)
        """
        assert self._current_value is not None, "call tick() before apply()"
        m = self._payload_mass  # [n_envs]
        theta = self._theta  # [n_envs, 2]

        # Horizontal reaction force (X, Y)
        f_xy = m.unsqueeze(1) * self._g * theta.sin()  # [n_envs, 2]

        # Vertical correction (tension deviation from mg)
        cos_avg = theta.cos().mean(dim=1)  # [n_envs]
        f_z = -m * self._g * (cos_avg - 1.0)  # [n_envs]

        self._force_buf[:, :2] = -f_xy
        self._force_buf[:, 2] = f_z
        self._force_buf.clamp_(self.bounds[0], self.bounds[1])
        return self._force_buf


# ---------------------------------------------------------------------------
# 5.9 Proximity aerodynamic disturbance (wall / ceiling effect)
# ---------------------------------------------------------------------------


@register("proximity_disturbance")
class ProximityDisturbance(ExternalWrenchPerturbation):
    """5.9 — Wall/ceiling aerodynamic proximity effect.

    Models the aerodynamic disturbance when the drone is near a surface.
    The force magnitude decays with distance:

        F = F_max × (1 − d/d_max)²     for d < d_max, else 0

    Direction: repulsive (away from surface). The surface positions are
    configurable; by default, walls at ±surface_distance on X and Y axes.

    The perturbation value (_current_value) is F_max (sampled per episode).

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        d_max: maximum influence distance [m].
        surface_distance: distance of walls from origin [m].
        distribution_params: {"low", "high"} — F_max range [N].
        bounds: F_max clamp [min, max] N.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        d_max: float = 0.3,
        surface_distance: float = 2.0,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.5),
        nominal: float = 0.0,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 0.5}

        self._d_max = d_max
        self._surface_distance = surface_distance
        self._force_buf: Tensor = torch.zeros(n_envs, 3)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="proximity_disturbance",
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
        self._current_value = torch.full((n_envs,), float(nominal))

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute repulsive force from nearby surfaces.

        Considers ±X, ±Y walls and ceiling (+Z). Force decays quadratically
        with distance, direction is away from the nearest surface.
        Vectorized over axes (no Python loop).
        """
        assert self._current_value is not None, "call tick() before apply()"
        f_max = self._current_value  # [n_envs] or scalar
        if f_max.dim() == 0:
            f_max = f_max.unsqueeze(0)
        f_max = f_max.view(-1)  # [n_envs]

        pos = env_state.pos  # [n_envs, 3]
        sd = self._surface_distance
        d_max = self._d_max
        self._force_buf.zero_()

        # XY walls: vectorized [n_envs, 2]
        pos_xy = pos[:, :2]  # [n_envs, 2]
        d_pos = (sd - pos_xy).clamp(min=0.0)  # distance to +wall
        d_neg = (sd + pos_xy).clamp(min=0.0)  # distance to -wall

        f_max_2d = f_max.unsqueeze(1)  # [n_envs, 1]
        f_pos = f_max_2d * (1.0 - d_pos / d_max).pow(2) * (d_pos < d_max).float()
        f_neg = f_max_2d * (1.0 - d_neg / d_max).pow(2) * (d_neg < d_max).float()
        self._force_buf[:, :2] = f_neg - f_pos  # net repulsive

        # Ceiling effect (+Z surface)
        d_ceil = (sd - pos[:, 2]).clamp(min=0.0)
        f_ceil = f_max * (1.0 - d_ceil / d_max).pow(2) * (d_ceil < d_max).float()
        self._force_buf[:, 2] = -f_ceil

        return self._force_buf
