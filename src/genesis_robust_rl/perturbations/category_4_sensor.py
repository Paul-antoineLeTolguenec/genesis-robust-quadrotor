"""Category 4 — Sensor observation perturbations.

Intermediate classes (not registered):
  AdditiveNoise       — stateless additive noise/bias on obs slice
  OUDrift             — stateful OU-process drift on obs slice
  ObsDropout          — stateful dropout with zero-order hold

Implemented perturbations:
  4.1  GyroNoise                — i.i.d. Gaussian noise on gyro channels
  4.2  GyroBias                 — constant per-episode bias on gyro channels
  4.3  GyroDrift                — OU-process drift on gyro channels
  4.4  AccelNoiseBiasDrift      — composite noise + bias + OU drift on accel channels
  4.5  SensorCrossAxis          — rotation matrix misalignment on obs slice
  4.6  PositionNoise            — i.i.d. Gaussian noise on GPS position
  4.7  PositionDropout          — event-driven dropout with last-value hold
  4.8  PositionOutlier          — Bernoulli multipath spikes on GPS
  4.9  VelocityNoise            — i.i.d. Gaussian noise on velocity estimate
  4.10 SensorQuantization       — discretization of obs channels
  4.11 ObsChannelMasking        — Bernoulli channel dropout (zero mask)
  4.12 MagnetometerInterference — additive offset on magnetometer reading
  4.13 BarometerDrift           — OU drift + Gaussian noise on baro altitude
  4.14 OpticalFlowNoise         — i.i.d. Gaussian noise on optical flow
  4.15 IMUVibration             — RPM-correlated Gaussian noise on accel
  4.16 ClockDrift               — accumulated phase offset (fractional delay)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    ObservationPerturbation,
    OUProcess,
    register,
)

# ===================================================================
# Intermediate classes (not registered)
# ===================================================================


class AdditiveNoise(ObservationPerturbation):
    """Stateless additive noise/bias on an obs slice.

    apply() adds _current_value (sampled noise or bias) to the targeted
    observation channels. Used by 4.2, 4.12 (constant offset perturbations).
    """

    def apply(self, obs: Tensor) -> Tensor:
        """Add _current_value to the targeted obs slice."""
        out = obs.clone()
        if self._current_value is None:
            return out
        out[:, self.obs_slice] = out[:, self.obs_slice] + self._current_value
        return out


class WhiteNoise(ObservationPerturbation):
    """Stateless white noise on an obs slice.

    _current_value stores the noise std σ (sampled from distribution).
    apply() generates N(0, σ²) noise and adds it to the obs slice.
    Bounds are on the std, not the noise value.
    Used by 4.1, 4.6, 4.9, 4.14.
    """

    def __init__(self, noise_dim: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._noise_dim = noise_dim

    def apply(self, obs: Tensor) -> Tensor:
        """Generate N(0, σ²) noise using _current_value as std, add to obs slice."""
        out = obs.clone()
        if self._current_value is None:
            return out
        # _current_value = sampled std [n_envs, dim] or [n_envs, 1]
        noise = torch.randn(self.n_envs, self._noise_dim) * self._current_value
        out[:, self.obs_slice] = out[:, self.obs_slice] + noise
        return out


class OUDrift(ObservationPerturbation):
    """Stateful OU-process drift on an obs slice.

    Maintains an OUProcess whose state is added to obs each step.
    Used by 4.3, 4.13, 4.16.
    """

    def __init__(
        self,
        obs_slice: slice,
        ou_theta: float = 0.15,
        ou_sigma: float = 0.01,
        ou_mu: float = 0.0,
        drift_dim: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_slice=obs_slice, **kwargs)
        self.is_stateful = True
        self._ou_theta = ou_theta
        self._ou_sigma = ou_sigma
        self._ou_mu = ou_mu
        self._ou = OUProcess(self.n_envs, drift_dim)
        self._current_value = torch.zeros(self.n_envs, drift_dim)

    def reset(self, env_ids: Tensor) -> None:
        """Reset OU process state for selected envs."""
        self._ou.reset(env_ids)
        self._current_value[env_ids] = 0.0

    def step(self) -> None:
        """Advance OU process one step."""
        self._ou.step(self._ou_theta, self._ou_sigma, self._ou_mu, self.dt)
        self._current_value = self._ou.state.clone()

    def apply(self, obs: Tensor) -> Tensor:
        """Add OU drift state to obs slice."""
        out = obs.clone()
        out[:, self.obs_slice] = out[:, self.obs_slice] + self._current_value
        return out


class ObsDropout(ObservationPerturbation):
    """Stateful observation dropout with zero-order hold.

    When a dropout event fires, the obs slice is replaced with the last
    valid reading for a random duration. Used by 4.7.
    """

    def __init__(
        self,
        obs_slice: slice,
        slice_dim: int,
        duration_low: int = 1,
        duration_high: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_slice=obs_slice, **kwargs)
        self.is_stateful = True
        self._slice_dim = slice_dim
        self._duration_low = duration_low
        self._duration_high = duration_high
        self._counter: Tensor = torch.zeros(self.n_envs, dtype=torch.long)
        self._last_valid: Tensor = torch.zeros(self.n_envs, slice_dim)
        self._drop_prob: Tensor = torch.zeros(self.n_envs)
        self._current_value = torch.zeros(self.n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset dropout state for selected envs."""
        self._counter[env_ids] = 0
        self._last_valid[env_ids] = 0.0
        self._drop_prob[env_ids] = 0.0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Sample dropout probability per env."""
        raw = self._draw().squeeze(-1)
        nominal_t = torch.tensor(self.nominal, dtype=torch.float32)
        scaled = nominal_t + (raw - nominal_t) * self.curriculum_scale
        self._drop_prob = scaled.clamp(self.bounds[0], self.bounds[1])
        return self._drop_prob.unsqueeze(-1)

    def step(self) -> None:
        """Decrement active counters, trigger new dropout events."""
        # Decrement active counters
        active = self._counter > 0
        self._counter.sub_(active.long())

        # Trigger new events for idle envs
        idle = self._counter == 0
        trigger = torch.bernoulli(self._drop_prob).bool() & idle
        durations = torch.randint(
            self._duration_low,
            self._duration_high + 1,
            (self.n_envs,),
            dtype=torch.long,
        )
        self._counter = torch.where(trigger, durations, self._counter)
        self._current_value = (self._counter > 0).unsqueeze(-1).float()

    def apply(self, obs: Tensor) -> Tensor:
        """Replace obs slice with last valid when dropout is active."""
        out = obs.clone()
        dropping = self._counter > 0
        # Update last valid for non-dropping envs
        not_dropping = ~dropping
        self._last_valid[not_dropping] = out[not_dropping][:, self.obs_slice]
        # Replace with last valid for dropping envs
        out[dropping, self.obs_slice] = self._last_valid[dropping]
        return out


# ===================================================================
# 4.1 — GyroNoise
# ===================================================================


@register("gyro_noise")
class GyroNoise(WhiteNoise):
    """4.1 — i.i.d. Gaussian white noise on gyroscope channels.

    Per step, samples a noise std σ from [0, max_std], then draws
    N(0, σ²) noise and adds it to the gyro obs slice.
    _current_value stores σ (the std), not the noise itself.

    Args:
        obs_slice: slice selecting gyro channels in observation vector.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: clamp on noise std [0, max_std] rad/s.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.05),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            noise_dim=3,
            obs_slice=obs_slice,
            id="gyro_noise",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.05},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


# ===================================================================
# 4.2 — GyroBias
# ===================================================================


@register("gyro_bias")
class GyroBias(AdditiveNoise):
    """4.2 — Constant per-episode bias on gyroscope channels.

    Sampled once per episode, constant additive offset on gyro reading.

    Args:
        obs_slice: slice selecting gyro channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: bias range per axis [-max, +max] rad/s.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-0.1, 0.1),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="gyro_bias",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed",
            frequency="per_episode",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": -0.1, "high": 0.1},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


# ===================================================================
# 4.3 — GyroDrift
# ===================================================================


@register("gyro_drift")
class GyroDrift(OUDrift):
    """4.3 — OU-process drift on gyroscope channels.

    Stateful: advances an Ornstein-Uhlenbeck process each step.
    The OU state is added to the gyro obs slice.

    Args:
        obs_slice: slice selecting gyro channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        ou_theta: mean-reversion rate.
        ou_sigma: noise scale.
        bounds: drift magnitude clamp.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        ou_theta: float = 0.15,
        ou_sigma: float = 0.01,
        ou_mu: float = 0.0,
        bounds: tuple[float, float] = (-0.2, 0.2),
        nominal: float = 0.0,
        lipschitz_k: float | None = 0.001,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            ou_theta=ou_theta,
            ou_sigma=ou_sigma,
            ou_mu=ou_mu,
            drift_dim=3,
            id="gyro_drift",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution="constant",
            distribution_params={"value": 0.0},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )

    def step(self) -> None:
        """Advance OU and clamp drift to bounds."""
        super().step()
        self._ou.state.clamp_(self.bounds[0], self.bounds[1])
        self._current_value = self._ou.state.clone()


# ===================================================================
# 4.4 — AccelNoiseBiasDrift
# ===================================================================


@register("accel_noise_bias_drift")
class AccelNoiseBiasDrift(ObservationPerturbation):
    """4.4 — Composite noise + bias + OU drift on accelerometer channels.

    Combines three effects identical to 4.1/4.2/4.3 but for linear acceleration:
      - White noise: i.i.d. Gaussian, std sampled per step
      - Bias: constant offset sampled per episode
      - Drift: OU process advancing each step

    All three are summed and added to the accel obs slice.

    Args:
        obs_slice: slice selecting accelerometer channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        noise_std: white noise standard deviation.
        bias_range: per-axis bias bounds.
        ou_theta, ou_sigma, ou_mu: drift OU parameters.
        bounds: hard clamp on total perturbation.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        noise_std: float = 0.1,
        bias_range: float = 1.0,
        ou_theta: float = 0.15,
        ou_sigma: float = 0.02,
        ou_mu: float = 0.0,
        bounds: tuple[float, float] = (-2.0, 2.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = 0.005,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="accel_noise_bias_drift",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution="constant",
            distribution_params={"value": 0.0},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self._noise_std = noise_std
        self._bias_range = bias_range
        self._ou = OUProcess(n_envs, 3)
        self._ou_theta = ou_theta
        self._ou_sigma = ou_sigma
        self._ou_mu = ou_mu
        self._bias: Tensor = torch.zeros(n_envs, 3)
        self._current_value = torch.zeros(n_envs, 3)

    def reset(self, env_ids: Tensor) -> None:
        """Reset bias and OU drift for selected envs."""
        self._ou.reset(env_ids)
        self._bias[env_ids] = (
            torch.empty(len(env_ids), 3).uniform_(-self._bias_range, self._bias_range)
            * self.curriculum_scale
        )
        self._current_value[env_ids] = 0.0

    def step(self) -> None:
        """Advance OU drift."""
        self._ou.step(self._ou_theta, self._ou_sigma, self._ou_mu, self.dt)

    def sample(self) -> Tensor:
        """Compute total perturbation: noise + bias + drift."""
        noise = torch.randn(self.n_envs, 3) * self._noise_std * self.curriculum_scale
        total = noise + self._bias + self._ou.state
        total.clamp_(self.bounds[0], self.bounds[1])
        self._current_value = total
        return total

    def apply(self, obs: Tensor) -> Tensor:
        """Add combined noise+bias+drift to accel channels."""
        out = obs.clone()
        out[:, self.obs_slice] = out[:, self.obs_slice] + self._current_value
        return out


# ===================================================================
# 4.5 — SensorCrossAxis
# ===================================================================


@register("sensor_cross_axis")
class SensorCrossAxis(ObservationPerturbation):
    """4.5 — Sensor cross-axis sensitivity (misalignment).

    Applies a small rotation matrix (close to identity) to a 3D obs slice
    each step. The rotation angles are sampled per episode.

    Args:
        obs_slice: slice selecting the 3D sensor channels to misalign.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: misalignment angle range in radians.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "gaussian",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-1.0, 1.0),  # rotation matrix elements
        angle_max: float = 0.0873,  # ±5° max misalignment angle
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="sensor_cross_axis",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed",
            frequency="per_episode",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"mean": 0.0, "std": 0.02},
            bounds=bounds,
            nominal=nominal,
            dimension=(3, 3),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._angle_max = angle_max
        # Pre-allocate rotation matrices [n_envs, 3, 3]
        self._rot_matrices: Tensor = torch.eye(3).unsqueeze(0).expand(n_envs, -1, -1).clone()
        self._current_value = self._rot_matrices.view(n_envs, 3, 3)

    def _angles_to_rotation(self, angles: Tensor) -> Tensor:
        """Convert small rotation angles [n_envs, 3] to rotation matrices [n_envs, 3, 3].

        Uses Rodrigues' small-angle approximation: R ≈ I + [θ]×.
        """
        n = angles.shape[0]
        R = torch.eye(3, device=angles.device).unsqueeze(0).expand(n, -1, -1).clone()
        # Skew-symmetric cross-product matrix
        R[:, 0, 1] -= angles[:, 2]
        R[:, 0, 2] += angles[:, 1]
        R[:, 1, 0] += angles[:, 2]
        R[:, 1, 2] -= angles[:, 0]
        R[:, 2, 0] -= angles[:, 1]
        R[:, 2, 1] += angles[:, 0]
        return R

    def sample(self) -> Tensor:
        """Sample rotation angles and compute rotation matrices."""
        shape = (self.n_envs, 3) if self.scope != "global" else (1, 3)
        angles = torch.randn(shape) * float(self.distribution_params.get("std", 0.02))
        angles = angles * self.curriculum_scale
        angles = angles.clamp(-self._angle_max, self._angle_max)
        self._rot_matrices = self._angles_to_rotation(angles)
        self._current_value = self._rot_matrices
        n = self.n_envs if self.scope != "global" else 1
        return self._current_value.view(n, 9)

    def apply(self, obs: Tensor) -> Tensor:
        """Multiply 3D obs slice by per-env rotation matrix."""
        out = obs.clone()
        v = out[:, self.obs_slice].unsqueeze(-1)  # [n_envs, 3, 1]
        rot = self._rot_matrices
        if rot.shape[0] == 1:
            rot = rot.expand(v.shape[0], -1, -1)
        rotated = torch.bmm(rot, v).squeeze(-1)  # [n_envs, 3]
        out[:, self.obs_slice] = rotated
        return out


# ===================================================================
# 4.6 — PositionNoise
# ===================================================================


@register("position_noise")
class PositionNoise(WhiteNoise):
    """4.6 — i.i.d. Gaussian noise on GPS/mocap position channels.

    Samples noise std σ from [0, max_std], then draws N(0, σ²) per axis.

    Args:
        obs_slice: slice selecting position channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: noise std range [0, max_std] m.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.5),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            noise_dim=3,
            obs_slice=obs_slice,
            id="position_noise",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.5},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


# ===================================================================
# 4.7 — PositionDropout
# ===================================================================


@register("position_dropout")
class PositionDropout(ObsDropout):
    """4.7 — Position sensor dropout with last-value hold.

    Event-driven: each step, independently per env, a dropout fires
    with probability p. During dropout, the position obs slice is
    replaced with the last valid reading.

    Args:
        obs_slice: slice selecting GPS position channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: dropout probability range.
        duration_low, duration_high: dropout duration range in steps.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.3),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        duration_low: int = 1,
        duration_high: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            slice_dim=3,
            duration_low=duration_low,
            duration_high=duration_high,
            id="position_dropout",
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


# ===================================================================
# 4.8 — PositionOutlier
# ===================================================================


@register("position_outlier")
class PositionOutlier(ObservationPerturbation):
    """4.8 — GPS multipath outlier spikes.

    Each step, independently per env, a spike fires with probability p.
    The spike magnitude is uniform in [mag_low, mag_high] with random direction.

    Args:
        obs_slice: slice selecting GPS position channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        spike_prob: probability of spike per step.
        mag_low, mag_high: spike magnitude range (m).
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        spike_prob: float = 0.01,
        mag_low: float = 0.5,
        mag_high: float = 5.0,
        bounds: tuple[float, float] = (-5.0, 5.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="position_outlier",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution="constant",
            distribution_params={"value": 0.0},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._spike_prob = spike_prob
        self._mag_low = mag_low
        self._mag_high = mag_high
        self._current_value = torch.zeros(n_envs, 3)

    def sample(self) -> Tensor:
        """Generate spike perturbation: Bernoulli trigger × random direction × magnitude."""
        trigger = torch.bernoulli(
            torch.full((self.n_envs,), self._spike_prob * self.curriculum_scale)
        ).bool()
        # Random direction (uniform on sphere)
        direction = torch.randn(self.n_envs, 3)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        # Random magnitude
        magnitude = torch.empty(self.n_envs, 1).uniform_(self._mag_low, self._mag_high)
        spike = direction * magnitude * trigger.unsqueeze(-1).float()
        spike.clamp_(self.bounds[0], self.bounds[1])
        self._current_value = spike
        return spike

    def apply(self, obs: Tensor) -> Tensor:
        """Add spike to position obs slice."""
        out = obs.clone()
        out[:, self.obs_slice] = out[:, self.obs_slice] + self._current_value
        return out


# ===================================================================
# 4.9 — VelocityNoise
# ===================================================================


@register("velocity_noise")
class VelocityNoise(WhiteNoise):
    """4.9 — i.i.d. Gaussian noise on velocity estimation channels.

    Samples noise std σ from [0, max_std], then draws N(0, σ²) per axis.

    Args:
        obs_slice: slice selecting velocity channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: noise std range [0, max_std] m/s.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.3),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            noise_dim=3,
            obs_slice=obs_slice,
            id="velocity_noise",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.3},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


# ===================================================================
# 4.10 — SensorQuantization
# ===================================================================


@register("sensor_quantization")
class SensorQuantization(ObservationPerturbation):
    """4.10 — Discretization of observation channels.

    Applies round(obs / resolution) * resolution to the obs slice.
    Resolution is sampled per episode.

    Args:
        obs_slice: slice selecting channels to quantize.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: resolution range [min_res, max_res].
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (1e-4, 1e-2),
        nominal: float = 1e-6,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="sensor_quantization",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed",
            frequency="per_episode",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 1e-4, "high": 1e-2},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )

    def apply(self, obs: Tensor) -> Tensor:
        """Quantize the targeted obs channels."""
        out = obs.clone()
        if self._current_value is None:
            return out
        res = self._current_value  # [n_envs, 1]
        # Avoid division by zero with tiny resolution
        res_safe = res.clamp(min=1e-8)
        sliced = out[:, self.obs_slice]
        out[:, self.obs_slice] = torch.round(sliced / res_safe) * res_safe
        return out


# ===================================================================
# 4.11 — ObsChannelMasking
# ===================================================================


@register("obs_channel_masking")
class ObsChannelMasking(ObservationPerturbation):
    """4.11 — Partial observation masking (sensor channel dropout).

    Applies a Bernoulli binary mask to the obs slice. Masked channels
    are zeroed out. The mask probability is the perturbation value.

    Args:
        obs_slice: slice selecting obs channels to mask.
        obs_dim: number of channels in the slice.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: masking probability range [0, 1].
    """

    def __init__(
        self,
        obs_slice: slice,
        obs_dim: int,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 1.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        frequency: str = "per_episode",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="obs_channel_masking",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed" if frequency == "per_episode" else "dynamic",
            frequency=frequency,
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.5},
            bounds=bounds,
            nominal=nominal,
            dimension=(obs_dim,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._obs_dim = obs_dim

    def sample(self) -> Tensor:
        """Sample mask probability and generate binary mask."""
        prob_raw = self._draw()
        nominal_t = torch.tensor(self.nominal, dtype=torch.float32).expand(prob_raw.shape)
        prob = nominal_t + (prob_raw - nominal_t) * self.curriculum_scale
        prob = prob.clamp(self.bounds[0], self.bounds[1])
        # Generate binary mask: 1 = keep, 0 = masked
        threshold = prob[:, :1].expand(-1, self._obs_dim)
        mask = (torch.rand(self.n_envs, self._obs_dim) > threshold).float()
        self._current_value = mask
        return mask

    def apply(self, obs: Tensor) -> Tensor:
        """Zero out masked channels."""
        out = obs.clone()
        if self._current_value is None:
            return out
        out[:, self.obs_slice] = out[:, self.obs_slice] * self._current_value
        return out


# ===================================================================
# 4.12 — MagnetometerInterference
# ===================================================================


@register("magnetometer_interference")
class MagnetometerInterference(AdditiveNoise):
    """4.12 — Additive interference offset on magnetometer reading.

    Constant per-episode or per-step offset in µT on the magnetometer
    obs slice, simulating hard-iron or environmental magnetic interference.

    Args:
        obs_slice: slice selecting magnetometer channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: interference magnitude per axis [-max, +max] µT.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-50.0, 50.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        frequency: str = "per_episode",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="magnetometer_interference",
            n_envs=n_envs,
            dt=dt,
            value_mode="fixed" if frequency == "per_episode" else "dynamic",
            frequency=frequency,
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": -10.0, "high": 10.0},
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


# ===================================================================
# 4.13 — BarometerDrift
# ===================================================================


@register("barometer_drift")
class BarometerDrift(ObservationPerturbation):
    """4.13 — Barometer OU drift + Gaussian noise on altitude channel.

    Composite: OU-process drift plus per-step white noise on the
    barometric altitude observation channel.

    Args:
        obs_slice: slice selecting the altitude channel.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        noise_std: white noise standard deviation (m).
        ou_theta, ou_sigma, ou_mu: drift OU parameters.
        bounds: drift magnitude clamp (m).
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        noise_std: float = 0.1,
        ou_theta: float = 0.05,
        ou_sigma: float = 0.02,
        ou_mu: float = 0.0,
        bounds: tuple[float, float] = (-2.0, 2.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="barometer_drift",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution="constant",
            distribution_params={"value": 0.0},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self._noise_std = noise_std
        self._ou = OUProcess(n_envs, 1)
        self._ou_theta = ou_theta
        self._ou_sigma = ou_sigma
        self._ou_mu = ou_mu
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset OU drift for selected envs."""
        self._ou.reset(env_ids)
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def step(self) -> None:
        """Advance OU drift."""
        self._ou.step(self._ou_theta, self._ou_sigma, self._ou_mu, self.dt)
        self._ou.state.clamp_(self.bounds[0], self.bounds[1])

    def sample(self) -> Tensor:
        """Compute drift + noise."""
        noise = torch.randn(self.n_envs, 1) * self._noise_std * self.curriculum_scale
        total = self._ou.state + noise
        total.clamp_(self.bounds[0], self.bounds[1])
        self._current_value = total
        return total

    def apply(self, obs: Tensor) -> Tensor:
        """Add drift + noise to altitude channel."""
        out = obs.clone()
        out[:, self.obs_slice] = out[:, self.obs_slice] + self._current_value
        return out


# ===================================================================
# 4.14 — OpticalFlowNoise
# ===================================================================


@register("optical_flow_noise")
class OpticalFlowNoise(WhiteNoise):
    """4.14 — i.i.d. Gaussian noise on optical flow channels.

    Samples noise std σ from [0, max_std], then draws N(0, σ²) per axis.

    Args:
        obs_slice: slice selecting optical flow channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: noise std range [0, max_std] m/s.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.5),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            noise_dim=2,
            obs_slice=obs_slice,
            id="optical_flow_noise",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.5},
            bounds=bounds,
            nominal=nominal,
            dimension=(2,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


# ===================================================================
# 4.15 — IMUVibration
# ===================================================================


@register("imu_vibration")
class IMUVibration(ObservationPerturbation):
    """4.15 — RPM-correlated Gaussian noise on accelerometer channels.

    Noise std scales with mean(RPM²): σ_noise = gain × mean(rpm²).
    Requires access to env_state.rpm, passed through a special method.

    Args:
        obs_slice: slice selecting accel channels.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: vibration gain range.
    """

    def __init__(
        self,
        obs_slice: slice,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.1),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="imu_vibration",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": 0.0, "high": 0.05},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self._rpm_squared_mean: Tensor = torch.zeros(n_envs, 1)
        self._current_value = torch.zeros(n_envs, 1)

    def set_rpm(self, rpm: Tensor) -> None:
        """Update current RPM for vibration computation. Called by env each step."""
        self._rpm_squared_mean = (rpm**2).mean(dim=-1, keepdim=True)

    def apply(self, obs: Tensor) -> Tensor:
        """Add RPM-correlated vibration noise to accel channels.

        Noise std = gain × mean(rpm²), where gain = _current_value[:, :1].
        """
        out = obs.clone()
        if self._current_value is None:
            return out
        gain = self._current_value[:, :1]  # [n_envs, 1]
        noise_std = gain * self._rpm_squared_mean
        noise = torch.randn(self.n_envs, 3) * noise_std
        out[:, self.obs_slice] = out[:, self.obs_slice] + noise
        return out


# ===================================================================
# 4.16 — ClockDrift
# ===================================================================


@register("clock_drift")
class ClockDrift(ObservationPerturbation):
    """4.16 — Clock drift between onboard modules.

    Accumulates a phase offset over time, simulating clock desynchronization.
    The drift rate (in ppm) is sampled per episode. The phase offset grows
    linearly and is applied as a fractional-step interpolation on the obs slice.

    Effectively: obs_out = (1-α) × obs_current + α × obs_previous,
    where α = frac(phase_offset / dt).

    Args:
        obs_slice: slice selecting affected channels.
        obs_dim: width of the affected slice.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        bounds: drift rate range in ppm.
    """

    def __init__(
        self,
        obs_slice: slice,
        obs_dim: int,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-100.0, 100.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs_slice=obs_slice,
            id="clock_drift",
            n_envs=n_envs,
            dt=dt,
            value_mode="dynamic",
            frequency="per_step",
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params or {"low": -50.0, "high": 50.0},
            bounds=bounds,
            nominal=nominal,
            dimension=(1,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )
        self.is_stateful = True
        self._obs_dim = obs_dim
        self._rate_ppm: Tensor = torch.zeros(n_envs)
        self._phase_offset: Tensor = torch.zeros(n_envs)
        self._prev_obs: Tensor = torch.zeros(n_envs, obs_dim)
        self._current_value = torch.zeros(n_envs, 1)

    def reset(self, env_ids: Tensor) -> None:
        """Reset phase offset and previous obs for selected envs."""
        self._phase_offset[env_ids] = 0.0
        self._prev_obs[env_ids] = 0.0
        self._rate_ppm[env_ids] = 0.0
        self._current_value[env_ids] = 0.0  # type: ignore[index]

    def sample(self) -> Tensor:
        """Sample drift rate in ppm."""
        raw = self._draw().squeeze(-1)
        nominal_t = torch.tensor(self.nominal, dtype=torch.float32)
        scaled = nominal_t + (raw - nominal_t) * self.curriculum_scale
        self._rate_ppm = scaled.clamp(self.bounds[0], self.bounds[1])
        self._current_value = self._phase_offset.unsqueeze(-1)
        return self._current_value

    def step(self) -> None:
        """Accumulate phase offset: phase += rate_ppm * 1e-6 * dt."""
        self._phase_offset += self._rate_ppm * 1e-6 * self.dt
        self._current_value = self._phase_offset.unsqueeze(-1)

    def apply(self, obs: Tensor) -> Tensor:
        """Apply fractional-step interpolation based on accumulated phase."""
        out = obs.clone()
        # Fractional interpolation weight
        alpha = (self._phase_offset.abs() / max(self.dt, 1e-8)).frac().unsqueeze(-1)
        current = out[:, self.obs_slice]
        interpolated = (1.0 - alpha) * current + alpha * self._prev_obs
        out[:, self.obs_slice] = interpolated
        # Save current as previous for next step
        self._prev_obs = current.clone()
        return out
