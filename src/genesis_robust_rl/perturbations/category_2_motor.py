"""Category 2 — Motor dynamics perturbations.

Implemented perturbations:
  2.1  ThrustCoefficientKF          — thrust coefficient KF perturbation (force)
  2.2  TorqueCoefficientKM          — torque coefficient KM perturbation (torque)
  2.3  PropellerThrustAsymmetry     — per-propeller thrust ratio (force)
  2.4  MotorPartialFailure          — per-motor efficiency fraction (force)
  2.5  MotorKill                    — binary motor kill mask
  2.6  MotorLag                     — first-order IIR lag on RPM
  2.7  MotorRPMNoise                — multiplicative Gaussian RPM noise
  2.8  MotorSaturation              — hard RPM clamp per env
  2.9  MotorWear                    — motor efficiency decay over episode
  2.10 RotorImbalance               — sinusoidal RPM modulation (vibration)
  2.11 MotorBackEMF                 — back-EMF brake torque delta (torque)
  2.12 MotorColdStart               — exponential KF warmup at episode start
  2.13 GyroscopicEffect             — gyroscopic precession torque delta (torque)
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    EnvState,
    ExternalWrenchPerturbation,
    MotorCommandPerturbation,
    register,
)

# ---------------------------------------------------------------------------
# Physical constants (Crazyflie CF2X defaults)
# ---------------------------------------------------------------------------

_KF_NOM = 3.16e-10  # N/RPM²
_KM_NOM = 3.16e-12  # N·m/RPM²
_SPIN_SIGNS = [1.0, -1.0, 1.0, -1.0]  # CW/CCW standard X config
_RPM_TO_RADS = 2.0 * math.pi / 60.0
_RPM_MAX_NOM: float = 21666.0  # CF2X typical max RPM


# ===================================================================
# 2.1 — ThrustCoefficientKF (ExternalWrenchPerturbation)
# ===================================================================


@register("thrust_coeff_kf")
class ThrustCoefficientKF(ExternalWrenchPerturbation):
    """2.1 — Thrust coefficient KF perturbation via external force.

    Perturbs the thrust coefficient KF (N/RPM^2) and injects the thrust
    delta as an upward force on the drone body:

        dF_z = sum_i (KF_pert - KF_nom) * RPM_i^2

    Only the Z component is nonzero (world frame, upward).

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KF_nom: nominal thrust coefficient (N/RPM^2).
        distribution: sampling distribution name.
        distribution_params: distribution parameters.
        bounds: hard clamp on KF value.
        nominal: unperturbed KF value.
        lipschitz_k: Lipschitz constraint (None for per_episode).
        frame: reference frame for force application.
        link_idx: body link index.
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KF_nom: float = _KF_NOM,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (1.58e-10, 4.74e-10),
        nominal: float = _KF_NOM,
        lipschitz_k: float | None = None,
        frame: Literal["local", "world"] = "world",
        link_idx: int = 0,
        duration_mode: Literal["continuous", "pulse"] = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 2.53e-10, "high": 3.79e-10}

        self._KF_nom = KF_nom

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="force",
            preserve_current_value=True,
            id="thrust_coeff_kf",
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
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._wrench_buf = torch.zeros(n_envs, 3)

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute thrust delta from KF perturbation.

        Returns:
            Tensor[n_envs, 3] -- force with only Z component nonzero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        kf_pert = self._current_value.view(-1)
        rpm = env_state.rpm
        delta_fz = (kf_pert - self._KF_nom) * rpm.pow(2).sum(dim=1)

        buf = self._wrench_buf
        buf.zero_()
        buf[:, 2] = delta_fz
        return buf


# ===================================================================
# 2.2 — TorqueCoefficientKM (ExternalWrenchPerturbation)
# ===================================================================


@register("torque_coeff_km")
class TorqueCoefficientKM(ExternalWrenchPerturbation):
    """2.2 — Torque coefficient KM perturbation via external torque.

    Perturbs the reactive torque coefficient KM (N.m/RPM^2) and injects
    the torque delta about the Z axis:

        dt_z = sum_i spin_i * (KM_pert - KM_nom) * RPM_i^2

    where spin_i = +1/-1 for CW/CCW propellers.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KM_nom: nominal torque coefficient (N.m/RPM^2).
        spin_signs: per-propeller CW/CCW signs [+1, -1, +1, -1].
        distribution: sampling distribution name.
        distribution_params: distribution parameters.
        bounds: hard clamp on KM value.
        nominal: unperturbed KM value.
        lipschitz_k: Lipschitz constraint.
        frame: reference frame.
        link_idx: body link index.
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KM_nom: float = _KM_NOM,
        spin_signs: list[float] | None = None,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (1.58e-12, 4.74e-12),
        nominal: float = _KM_NOM,
        lipschitz_k: float | None = None,
        frame: Literal["local", "world"] = "world",
        link_idx: int = 0,
        duration_mode: Literal["continuous", "pulse"] = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 2.53e-12, "high": 3.79e-12}
        if spin_signs is None:
            spin_signs = list(_SPIN_SIGNS)

        self._KM_nom = KM_nom
        self._spin = torch.tensor(spin_signs, dtype=torch.float32)
        self._spin_cached: Tensor | None = None

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="torque",
            preserve_current_value=True,
            id="torque_coeff_km",
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
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._wrench_buf = torch.zeros(n_envs, 3)

    def _get_spin(self, device: torch.device) -> Tensor:
        """Return spin signs on the correct device (cached)."""
        if self._spin_cached is None or self._spin_cached.device != device:
            self._spin_cached = self._spin.to(device)
        return self._spin_cached

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute reactive torque delta from KM perturbation.

        Returns:
            Tensor[n_envs, 3] -- torque with only Z component nonzero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        km_pert = self._current_value.view(-1)
        rpm = env_state.rpm
        spin = self._get_spin(rpm.device)
        delta_tz = (km_pert - self._KM_nom) * (spin * rpm.pow(2)).sum(dim=1)

        buf = self._wrench_buf
        buf.zero_()
        buf[:, 2] = delta_tz
        return buf


# ===================================================================
# 2.3 — PropellerThrustAsymmetry (ExternalWrenchPerturbation)
# ===================================================================


@register("propeller_thrust_asymmetry")
class PropellerThrustAsymmetry(ExternalWrenchPerturbation):
    """2.3 — Per-propeller thrust ratio multiplier via external force.

    Each propeller has a ratio_i multiplier relative to the nominal KF.
    The net thrust correction is:

        dF_z = sum_i (ratio_i - 1) * KF_nom * RPM_i^2

    ratio = [n_envs, 4], one per propeller.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KF_nom: nominal thrust coefficient (N/RPM^2).
        distribution: sampling distribution name.
        distribution_params: distribution parameters.
        bounds: hard clamp on ratio values.
        nominal: unperturbed ratios [1, 1, 1, 1].
        lipschitz_k: Lipschitz constraint.
        frame: reference frame.
        link_idx: body link index.
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KF_nom: float = _KF_NOM,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.7, 1.3),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = None,
        frame: Literal["local", "world"] = "world",
        link_idx: int = 0,
        duration_mode: Literal["continuous", "pulse"] = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.85, "high": 1.15}
        if nominal is None:
            nominal = [1.0, 1.0, 1.0, 1.0]

        self._KF_nom = KF_nom

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="force",
            preserve_current_value=True,
            id="propeller_thrust_asymmetry",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(4,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._wrench_buf = torch.zeros(n_envs, 3)

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute thrust asymmetry correction.

        Returns:
            Tensor[n_envs, 3] -- force with only Z component nonzero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        rpm = env_state.rpm
        delta_fz = ((self._current_value - 1.0) * self._KF_nom * rpm.pow(2)).sum(dim=1)

        buf = self._wrench_buf
        buf.zero_()
        buf[:, 2] = delta_fz
        return buf


# ===================================================================
# 2.4 — MotorPartialFailure (ExternalWrenchPerturbation)
# ===================================================================


@register("motor_partial_failure")
class MotorPartialFailure(ExternalWrenchPerturbation):
    """2.4 — Per-motor efficiency fraction via external force.

    Each motor has efficiency_i in [0, 1] (0 = full failure, 1 = nominal).
    The thrust correction compensates for the reduced motor output:

        dF_z = sum_i (efficiency_i - 1) * KF_nom * RPM_i^2

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KF_nom: nominal thrust coefficient (N/RPM^2).
        distribution: sampling distribution name.
        distribution_params: distribution parameters.
        bounds: hard clamp on efficiency [0, 1].
        nominal: unperturbed efficiencies [1, 1, 1, 1].
        lipschitz_k: Lipschitz constraint.
        frame: reference frame.
        link_idx: body link index.
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KF_nom: float = _KF_NOM,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 1.0),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = None,
        frame: Literal["local", "world"] = "world",
        link_idx: int = 0,
        duration_mode: Literal["continuous", "pulse"] = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.3, "high": 1.0}
        if nominal is None:
            nominal = [1.0, 1.0, 1.0, 1.0]

        self._KF_nom = KF_nom

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="force",
            preserve_current_value=True,
            id="motor_partial_failure",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(4,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._wrench_buf = torch.zeros(n_envs, 3)

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute motor failure thrust correction.

        Returns:
            Tensor[n_envs, 3] -- force with only Z component nonzero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        rpm = env_state.rpm
        delta_fz = ((self._current_value - 1.0) * self._KF_nom * rpm.pow(2)).sum(dim=1)

        buf = self._wrench_buf
        buf.zero_()
        buf[:, 2] = delta_fz
        return buf


# ===================================================================
# 2.5 — MotorKill (MotorCommandPerturbation, stateless)
# ===================================================================


@register("motor_kill")
class MotorKill(MotorCommandPerturbation):
    """2.5 — Binary motor kill mask.

    Randomly kills 0..max_killed motors per env per episode.
    apply() zeroes out RPM for killed motors.

    _current_value holds the kill mask [n_envs, 4] (1=killed, 0=alive).
    Custom sample() -- does not use _draw().

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        min_killed: minimum number of killed motors per env.
        max_killed: maximum number of killed motors per env.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        min_killed: int = 0,
        max_killed: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            id="motor_kill",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution="uniform",
            distribution_params={"low": 0.0, "high": 1.0},
            bounds=(0.0, 1.0),
            nominal=0.0,
            dimension=(4,),
            lipschitz_k=None,
            **kwargs,
        )
        self.min_killed = min_killed
        self.max_killed = max_killed
        self._kill_mask = torch.zeros(n_envs, 4)

    def sample(self) -> Tensor:
        """Sample a binary kill mask respecting curriculum_scale.

        At curriculum_scale=0, n_killed=0 for all envs (nominal).
        At curriculum_scale=1, n_killed ~ U[min_killed, max_killed].
        """
        n = self.n_envs
        mask = self._kill_mask
        mask.zero_()

        eff_max = int(
            round(self.min_killed + (self.max_killed - self.min_killed) * self.curriculum_scale)
        )
        eff_min = int(round(self.min_killed * self.curriculum_scale))
        eff_min = max(0, min(eff_min, 4))
        eff_max = max(eff_min, min(eff_max, 4))

        if eff_max > 0:
            n_killed = torch.randint(eff_min, eff_max + 1, (n,), dtype=torch.long)
            rand_perm = torch.rand(n, 4).argsort(dim=1)
            for k in range(1, eff_max + 1):
                env_mask = n_killed >= k
                if env_mask.any():
                    motor_idx = rand_perm[env_mask, k - 1]
                    mask[env_mask, motor_idx] = 1.0

        self._current_value = mask.clone()
        return self._current_value

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Zero out RPM for killed motors: rpm * (1 - kill_mask)."""
        assert self._current_value is not None, "call tick() before apply()"
        return rpm_cmd * (1.0 - self._current_value)


# ===================================================================
# 2.6 — MotorLag (MotorCommandPerturbation, stateful)
# ===================================================================


@register("motor_lag")
class MotorLag(MotorCommandPerturbation):
    """2.6 — First-order IIR lag on RPM commands.

    Models motor response delay as a low-pass filter:

        rpm_actual += (rpm_cmd - rpm_actual) * dt / tau

    where tau is the motor time constant (seconds).

    State: _rpm_actual [n_envs, 4] -- filtered RPM output.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        tau_nom: float = 0.033,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.01, 0.1),
        nominal: float = 0.033,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.02, "high": 0.05}

        super().__init__(
            id="motor_lag",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
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
        self._tau_nom = tau_nom
        self._rpm_actual: Tensor = torch.zeros(n_envs, 4)
        self._current_value = torch.full((n_envs,), nominal)

    def reset(self, env_ids: Tensor) -> None:
        """Zero filtered RPM for selected envs."""
        self._rpm_actual[env_ids] = 0.0

    def step(self) -> None:
        """No-op -- filtering happens in apply()."""

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Apply first-order lag filter to RPM commands.

        Args:
            rpm_cmd: commanded RPM [n_envs, 4].

        Returns:
            Filtered RPM [n_envs, 4].
        """
        assert self._current_value is not None, "call tick() before apply()"
        tau = self._current_value.unsqueeze(-1)  # [n_envs, 1]
        alpha = (self.dt / tau).clamp(0.0, 1.0)
        self._rpm_actual.add_(alpha * (rpm_cmd - self._rpm_actual))
        return self._rpm_actual.clone()


# ===================================================================
# 2.7 — MotorRPMNoise (MotorCommandPerturbation, stateless)
# ===================================================================


@register("motor_rpm_noise")
class MotorRPMNoise(MotorCommandPerturbation):
    """2.7 — Multiplicative Gaussian RPM noise.

    Each step, draws relative noise scale from [0, noise_frac_max],
    then applies: rpm_actual = rpm_cmd + randn * scale * rpm_cmd.

    _current_value holds the noise scale [n_envs, 4] resampled per step.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        noise_frac_max: max noise as fraction of commanded RPM.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        noise_frac_max: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            id="motor_rpm_noise",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution="uniform",
            distribution_params={"low": 0.0, "high": 0.03},
            bounds=(0.0, noise_frac_max),
            nominal=0.0,
            dimension=(4,),
            lipschitz_k=None,
            **kwargs,
        )
        self.noise_frac_max = noise_frac_max

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Add relative Gaussian noise: rpm + randn * scale * rpm."""
        assert self._current_value is not None, "call tick() before apply()"
        noise = torch.randn_like(rpm_cmd) * self._current_value
        return rpm_cmd + noise * rpm_cmd


# ===================================================================
# 2.8 — MotorSaturation (MotorCommandPerturbation, stateless)
# ===================================================================


@register("motor_saturation")
class MotorSaturation(MotorCommandPerturbation):
    """2.8 — Hard RPM saturation clamp.

    Clamps RPM to [0, rpm_max] where rpm_max is sampled per env per episode.

    _current_value holds rpm_max [n_envs] (scalar per env).

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        rpm_max_nom: nominal (unperturbed) max RPM.
        rpm_max_low_frac: lowest rpm_max as fraction of nominal.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        rpm_max_nom: float = _RPM_MAX_NOM,
        rpm_max_low_frac: float = 0.7,
        **kwargs: Any,
    ) -> None:
        lo = rpm_max_nom * rpm_max_low_frac
        hi = rpm_max_nom
        default_sample_lo = rpm_max_nom * 0.8
        super().__init__(
            id="motor_saturation",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution="uniform",
            distribution_params={"low": default_sample_lo, "high": hi},
            bounds=(lo, hi),
            nominal=rpm_max_nom,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )
        self.rpm_max_nom = rpm_max_nom

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Clamp RPM: clamp(rpm_cmd, 0, rpm_max)."""
        assert self._current_value is not None, "call tick() before apply()"
        rpm_max = self._current_value.unsqueeze(-1)
        return rpm_cmd.clamp(min=0.0, max=None).clamp(max=rpm_max)


# ===================================================================
# 2.9 — MotorWear (MotorCommandPerturbation, stateful)
# ===================================================================


@register("motor_wear")
class MotorWear(MotorCommandPerturbation):
    """2.9 — Motor efficiency decay over an episode.

    Efficiency starts at 1.0 and decreases monotonically each step:

        efficiency = max(efficiency - rate, 0.8)

    The decay rate is sampled per motor per env at reset.

    State: _efficiency [n_envs, 4], _rate [n_envs, 4].
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.8, 1.0),
        nominal: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0001, "high": 0.001}

        super().__init__(
            id="motor_wear",
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
        self._efficiency: Tensor = torch.ones(n_envs, 4)
        self._rate: Tensor = torch.zeros(n_envs, 4)
        self._current_value = self._efficiency.clone()

    def reset(self, env_ids: Tensor) -> None:
        """Reset efficiency to 1.0 and resample decay rate."""
        self._efficiency[env_ids] = 1.0
        p = self.distribution_params
        n = len(env_ids)
        rate_raw = torch.empty(n, 4).uniform_(p["low"], p["high"])
        self._rate[env_ids] = rate_raw * self.curriculum_scale
        self._current_value = self._efficiency

    def sample(self) -> Tensor:
        """Return current efficiency (state-driven, not distribution-drawn)."""
        self._current_value = self._efficiency
        return self._current_value

    def set_value(self, value: Tensor) -> None:
        """Adversarial mode: sync _efficiency with _current_value."""
        super().set_value(value)
        if self._current_value is not None:
            self._efficiency = self._current_value

    def step(self) -> None:
        """Advance efficiency decay by one step."""
        self._efficiency.sub_(self._rate).clamp_(min=0.8)
        self._current_value = self._efficiency

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Scale RPM by current motor efficiency.

        Args:
            rpm_cmd: commanded RPM [n_envs, 4].

        Returns:
            Degraded RPM [n_envs, 4].
        """
        return rpm_cmd * self._current_value


# ===================================================================
# 2.10 — RotorImbalance (MotorCommandPerturbation, stateful)
# ===================================================================


@register("rotor_imbalance")
class RotorImbalance(MotorCommandPerturbation):
    """2.10 — Sinusoidal RPM modulation from rotor mass imbalance.

    Models vibration from unbalanced propellers:

        rpm_out = rpm_cmd * (1 + magnitude * sin(phase))

    Phase advances each step proportional to current RPM.

    State: _phase [n_envs, 4], _last_rpm [n_envs, 4].
    Exposes imu_noise_amplitude for Cat 4 vibration coupling.
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.05),
        nominal: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 0.03}

        super().__init__(
            id="rotor_imbalance",
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
        self._phase: Tensor = torch.zeros(n_envs, 4)
        self._magnitude: Tensor = torch.zeros(n_envs, 4)
        self._last_rpm: Tensor = torch.zeros(n_envs, 4)
        self._current_value = self._magnitude

    def reset(self, env_ids: Tensor) -> None:
        """Randomize phase and resample magnitude for selected envs."""
        n = len(env_ids)
        self._phase[env_ids] = torch.empty(n, 4).uniform_(0.0, 2.0 * math.pi)
        p = self.distribution_params
        mag_raw = torch.empty(n, 4).uniform_(p["low"], p["high"])
        nominal_t = torch.tensor(self.nominal, dtype=torch.float32)
        scaled = nominal_t + (mag_raw - nominal_t) * self.curriculum_scale
        self._magnitude[env_ids] = scaled.clamp(self.bounds[0], self.bounds[1])
        self._last_rpm[env_ids] = 0.0
        self._current_value = self._magnitude

    def sample(self) -> Tensor:
        """Return current magnitude (state-driven)."""
        self._current_value = self._magnitude
        return self._current_value

    def set_value(self, value: Tensor) -> None:
        """Adversarial mode: sync _magnitude with _current_value."""
        super().set_value(value)
        if self._current_value is not None:
            self._magnitude = self._current_value

    def step(self) -> None:
        """Advance rotor phase based on last known RPM."""
        omega = self._last_rpm * _RPM_TO_RADS
        self._phase.add_(omega * self.dt)

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Apply sinusoidal RPM modulation from rotor imbalance.

        Args:
            rpm_cmd: commanded RPM [n_envs, 4].

        Returns:
            Modulated RPM [n_envs, 4].
        """
        self._last_rpm = rpm_cmd.detach().clone()
        modulation = 1.0 + self._magnitude * torch.sin(self._phase)
        return rpm_cmd * modulation

    @property
    def imu_noise_amplitude(self) -> Tensor:
        """Vibration amplitude for Cat 4 sensor coupling.

        Returns:
            [n_envs, 4] -- vibration proportional to RPM * imbalance.
        """
        omega = self._last_rpm * _RPM_TO_RADS
        return self._magnitude * omega


# ===================================================================
# 2.11 — MotorBackEMF (ExternalWrenchPerturbation)
# ===================================================================


@register("motor_back_emf")
class MotorBackEMF(ExternalWrenchPerturbation):
    """2.11 — Back-EMF brake torque delta via external torque.

    Models the back-EMF braking effect: each motor generates a brake
    torque t_brake = Ke^2 * w_motor / R that opposes rotation. The
    perturbation injects only the DELTA between perturbed and nominal:

        w_motor_i = RPM_i * 2pi/60
        dt_z = sum_i spin_i * (Ke_pert^2 - Ke_nom^2) * w_i / R

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        Ke_nom: nominal back-EMF constant (V.s/rad).
        R_motor: motor winding resistance (ohm, not perturbed).
        spin_signs: per-propeller CW/CCW signs.
        distribution: sampling distribution name.
        distribution_params: distribution parameters.
        bounds: hard clamp on Ke value.
        nominal: unperturbed Ke value.
        lipschitz_k: Lipschitz constraint.
        frame: reference frame.
        link_idx: body link index.
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        Ke_nom: float = 0.01,
        R_motor: float = 1.0,
        spin_signs: list[float] | None = None,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.005, 0.015),
        nominal: float = 0.01,
        lipschitz_k: float | None = None,
        frame: Literal["local", "world"] = "world",
        link_idx: int = 0,
        duration_mode: Literal["continuous", "pulse"] = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.007, "high": 0.013}
        if spin_signs is None:
            spin_signs = list(_SPIN_SIGNS)

        self._Ke_nom = Ke_nom
        self._Ke_nom_sq = Ke_nom**2
        self._R_motor = R_motor
        self._spin = torch.tensor(spin_signs, dtype=torch.float32)
        self._spin_cached: Tensor | None = None

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="torque",
            preserve_current_value=True,
            id="motor_back_emf",
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
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._wrench_buf = torch.zeros(n_envs, 3)

    def _get_spin(self, device: torch.device) -> Tensor:
        """Return spin signs on the correct device (cached)."""
        if self._spin_cached is None or self._spin_cached.device != device:
            self._spin_cached = self._spin.to(device)
        return self._spin_cached

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute back-EMF brake torque delta.

        Returns:
            Tensor[n_envs, 3] -- torque with only Z component nonzero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        ke_pert = self._current_value.view(-1)
        rpm = env_state.rpm
        spin = self._get_spin(rpm.device)

        omega = rpm * _RPM_TO_RADS
        ke_pert_sq = (ke_pert * ke_pert).unsqueeze(1)  # [n_envs, 1]
        delta_per_motor = (ke_pert_sq - self._Ke_nom_sq) * omega / self._R_motor
        delta_tz = (spin * delta_per_motor).sum(dim=1)

        buf = self._wrench_buf
        buf.zero_()
        buf[:, 2] = delta_tz
        return buf


# ===================================================================
# 2.12 — MotorColdStart (MotorCommandPerturbation, stateful)
# ===================================================================


@register("motor_cold_start")
class MotorColdStart(MotorCommandPerturbation):
    """2.12 — Exponential KF warmup at episode start.

    Models cold motor behavior where effective thrust coefficient is
    initially higher than nominal and decays exponentially to 1.0:

        warmup_factor = 1 + (initial_overhead - 1) * exp(-t / tau)

    The RPM is scaled by warmup_factor.

    State: _warmup_factor [n_envs, 4].
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        warmup_tau: float = 0.5,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (1.0, 1.5),
        nominal: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 1.0, "high": 1.3}

        super().__init__(
            id="motor_cold_start",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
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
        self._warmup_tau = warmup_tau
        self._warmup_factor: Tensor = torch.ones(n_envs, 4)
        self._initial_overhead: Tensor = torch.ones(n_envs, 4)
        self._current_value = self._warmup_factor

    def reset(self, env_ids: Tensor) -> None:
        """Resample initial overhead and reset warmup factor."""
        n = len(env_ids)
        p = self.distribution_params
        raw = torch.empty(n, 4).uniform_(p["low"], p["high"])
        nominal_t = torch.tensor(self.nominal, dtype=torch.float32)
        scaled = nominal_t + (raw - nominal_t) * self.curriculum_scale
        self._initial_overhead[env_ids] = scaled.clamp(self.bounds[0], self.bounds[1])
        self._warmup_factor[env_ids] = self._initial_overhead[env_ids]
        self._current_value = self._warmup_factor

    def sample(self) -> Tensor:
        """Return current warmup factor (state-driven)."""
        self._current_value = self._warmup_factor
        return self._current_value

    def set_value(self, value: Tensor) -> None:
        """Adversarial mode: sync _warmup_factor with _current_value."""
        super().set_value(value)
        if self._current_value is not None:
            self._warmup_factor = self._current_value

    def step(self) -> None:
        """Exponential decay of warmup factor toward 1.0."""
        decay = math.exp(-self.dt / self._warmup_tau)
        self._warmup_factor.sub_(1.0).mul_(decay).add_(1.0)
        self._current_value = self._warmup_factor

    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Scale RPM by warmup factor.

        Args:
            rpm_cmd: commanded RPM [n_envs, 4].

        Returns:
            Scaled RPM [n_envs, 4].
        """
        return rpm_cmd * self._current_value


# ===================================================================
# 2.13 — GyroscopicEffect (ExternalWrenchPerturbation)
# ===================================================================


@register("gyroscopic_effect")
class GyroscopicEffect(ExternalWrenchPerturbation):
    """2.13 — Gyroscopic precession torque delta via external torque.

    Models the gyroscopic torque from spinning rotors:

        Omega_rotor = sum_i spin_i * w_rotor_i  (net angular velocity)
        t_gyro = I_rotor * Omega_rotor * (rotor_axis x w_body)

    where rotor_axis = [0, 0, 1]. Injects DELTA between perturbed
    and nominal I_rotor.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        I_rotor_nom: nominal rotor inertia (kg.m^2).
        spin_signs: per-propeller CW/CCW signs.
        distribution: sampling distribution name.
        distribution_params: distribution parameters.
        bounds: hard clamp on I_rotor value.
        nominal: unperturbed I_rotor value.
        lipschitz_k: Lipschitz constraint.
        frame: reference frame.
        link_idx: body link index.
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        I_rotor_nom: float = 3.0e-5,
        spin_signs: list[float] | None = None,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (1.5e-5, 4.5e-5),
        nominal: float = 3.0e-5,
        lipschitz_k: float | None = None,
        frame: Literal["local", "world"] = "world",
        link_idx: int = 0,
        duration_mode: Literal["continuous", "pulse"] = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 2.0e-5, "high": 4.0e-5}
        if spin_signs is None:
            spin_signs = list(_SPIN_SIGNS)

        self._I_rotor_nom = I_rotor_nom
        self._spin = torch.tensor(spin_signs, dtype=torch.float32)
        self._spin_cached: Tensor | None = None

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="torque",
            preserve_current_value=True,
            id="gyroscopic_effect",
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
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._wrench_buf = torch.zeros(n_envs, 3)

    def _get_spin(self, device: torch.device) -> Tensor:
        """Return spin signs on the correct device (cached)."""
        if self._spin_cached is None or self._spin_cached.device != device:
            self._spin_cached = self._spin.to(device)
        return self._spin_cached

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute gyroscopic torque delta.

        Cross product [0,0,1] x w_body = [-w_y, w_x, 0].

        Returns:
            Tensor[n_envs, 3] -- gyroscopic torque correction.
        """
        assert self._current_value is not None, "call tick() before apply()"
        i_pert = self._current_value.view(-1)
        rpm = env_state.rpm
        ang_vel = env_state.ang_vel
        spin = self._get_spin(rpm.device)

        omega_net = (spin * rpm * _RPM_TO_RADS).sum(dim=1)
        scale = (i_pert - self._I_rotor_nom) * omega_net  # [n_envs]

        buf = self._wrench_buf
        buf[:, 0] = -ang_vel[:, 1] * scale
        buf[:, 1] = ang_vel[:, 0] * scale
        buf[:, 2] = 0.0
        return buf
