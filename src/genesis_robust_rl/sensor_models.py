"""Phenomenological sensor forward models for quadrotor simulation.

Each model converts the true Genesis state into a raw sensor reading.
No fusion assumption — outputs are raw readings only.

Pipeline: env_state → SensorModel.forward(env_state) → ObservationPerturbation.apply(raw)

Models:
  GyroscopeModel     — body angular velocity (rad/s), dim=3
  AccelerometerModel  — specific force in body frame (m/s²), dim=3
  MagnetometerModel   — magnetic field in body frame (µT), dim=3
  BarometerModel      — barometric altitude estimate (m), dim=1
  GPSModel            — world-frame position (m), dim=3
  OpticalFlowModel    — apparent horizontal velocity (m/s), dim=2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import EnvState

# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------


def _quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert quaternion (w,x,y,z) to rotation matrix [n_envs, 3, 3]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (y2 + z2)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (x2 + z2)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (x2 + y2)

    # Stack into [n_envs, 3, 3]
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SensorModel(ABC):
    """Abstract base for phenomenological sensor forward models.

    Converts the true Genesis state into a raw sensor reading.
    Perturbations corrupt the output via ObservationPerturbation.apply().
    """

    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs

    @abstractmethod
    def forward(self, env_state: EnvState) -> Tensor:
        """Produce raw sensor reading [n_envs, dim]."""

    def update_params(self, new_params: dict) -> None:
        """Update internal sensor parameters. Validates against typed dataclass."""
        raise ValueError(f"{type(self).__name__} does not accept params: {list(new_params.keys())}")

    def reset(self, env_ids: Tensor) -> None:
        """Reset internal state for selected envs. No-op by default."""


# ---------------------------------------------------------------------------
# Typed param dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GyroParams:
    """Gyroscope model parameters."""

    C_misalign: Tensor = field(default_factory=lambda: torch.eye(3))


@dataclass(frozen=True)
class AccelParams:
    """Accelerometer model parameters."""

    C_misalign: Tensor = field(default_factory=lambda: torch.eye(3))
    gravity: float = 9.81


@dataclass(frozen=True)
class MagParams:
    """Magnetometer model parameters."""

    B_earth: Tensor = field(
        default_factory=lambda: torch.tensor([20.0, 0.0, -40.0])
    )  # µT, typical mid-latitude
    M_soft: Tensor = field(default_factory=lambda: torch.eye(3))
    b_hard: Tensor = field(default_factory=lambda: torch.zeros(3))


@dataclass(frozen=True)
class BaroParams:
    """Barometer model parameters."""

    ref_alt: float = 0.0  # reference altitude offset (m)


@dataclass(frozen=True)
class GPSParams:
    """GPS model parameters."""

    offset: Tensor = field(default_factory=lambda: torch.zeros(3))


@dataclass(frozen=True)
class FlowParams:
    """Optical flow model parameters."""

    min_alt: float = 0.05  # minimum altitude for valid flow (m)


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


class GyroscopeModel(SensorModel):
    """Gyroscope forward model — body angular velocity (rad/s).

    Output: C_misalign @ ang_vel_true, shape [n_envs, 3].
    Perturbation classes (4.1–4.3, 4.5) add noise/bias/drift/misalignment.
    """

    def __init__(self, n_envs: int, params: GyroParams | None = None) -> None:
        super().__init__(n_envs)
        p = params or GyroParams()
        self._C_misalign: Tensor = p.C_misalign.clone()

    def forward(self, env_state: EnvState) -> Tensor:
        """Return C_misalign @ ang_vel [n_envs, 3]."""
        return torch.einsum("ij,nj->ni", self._C_misalign, env_state.ang_vel)

    def update_params(self, new_params: dict) -> None:
        if "C_misalign" in new_params:
            val = new_params["C_misalign"]
            if not isinstance(val, Tensor) or val.shape != (3, 3):
                raise ValueError("C_misalign must be Tensor[3,3]")
            self._C_misalign = val
        unknown = set(new_params) - {"C_misalign"}
        if unknown:
            raise ValueError(f"Unknown params: {unknown}")


class AccelerometerModel(SensorModel):
    """Accelerometer forward model — specific force in body frame (m/s²).

    Output: C_misalign @ (acc - R^T @ [0,0,-g]), shape [n_envs, 3].
    Stateless — reads env_state.acc directly. No reset() override.
    """

    def __init__(self, n_envs: int, params: AccelParams | None = None) -> None:
        super().__init__(n_envs)
        p = params or AccelParams()
        self._C_misalign: Tensor = p.C_misalign.clone()
        self._gravity = p.gravity

    def forward(self, env_state: EnvState) -> Tensor:
        """Return specific force in body frame [n_envs, 3]."""
        # Gravity in world frame: [0, 0, -g]
        g_world = torch.zeros(env_state.pos.shape[0], 3, device=env_state.pos.device)
        g_world[:, 2] = -self._gravity
        # R^T @ g_world = project gravity to body frame
        R = _quat_to_rotmat(env_state.quat)  # [n_envs, 3, 3]
        g_body = torch.bmm(R.transpose(1, 2), g_world.unsqueeze(-1)).squeeze(-1)
        # Specific force = acc - g_body
        accel_true = env_state.acc - g_body
        return torch.einsum("ij,nj->ni", self._C_misalign, accel_true)

    def update_params(self, new_params: dict) -> None:
        if "C_misalign" in new_params:
            val = new_params["C_misalign"]
            if not isinstance(val, Tensor) or val.shape != (3, 3):
                raise ValueError("C_misalign must be Tensor[3,3]")
            self._C_misalign = val
        if "gravity" in new_params:
            self._gravity = float(new_params["gravity"])
        unknown = set(new_params) - {"C_misalign", "gravity"}
        if unknown:
            raise ValueError(f"Unknown params: {unknown}")


class MagnetometerModel(SensorModel):
    """Magnetometer forward model — magnetic field in body frame (µT).

    Output: M_soft @ (R^T @ B_earth) + b_hard, shape [n_envs, 3].
    """

    def __init__(self, n_envs: int, params: MagParams | None = None) -> None:
        super().__init__(n_envs)
        p = params or MagParams()
        self._B_earth: Tensor = p.B_earth.clone()
        self._M_soft: Tensor = p.M_soft.clone()
        self._b_hard: Tensor = p.b_hard.clone()

    def forward(self, env_state: EnvState) -> Tensor:
        """Return magnetic field in body frame [n_envs, 3]."""
        R = _quat_to_rotmat(env_state.quat)
        # R^T @ B_earth for each env
        B_body = torch.einsum("nij,j->ni", R.transpose(1, 2), self._B_earth)
        # M_soft @ B_body + b_hard
        return torch.einsum("ij,nj->ni", self._M_soft, B_body) + self._b_hard

    def update_params(self, new_params: dict) -> None:
        valid_keys = {"B_earth", "M_soft", "b_hard"}
        if "B_earth" in new_params:
            val = new_params["B_earth"]
            if not isinstance(val, Tensor) or val.shape != (3,):
                raise ValueError("B_earth must be Tensor[3]")
            self._B_earth = val
        if "M_soft" in new_params:
            val = new_params["M_soft"]
            if not isinstance(val, Tensor) or val.shape != (3, 3):
                raise ValueError("M_soft must be Tensor[3,3]")
            self._M_soft = val
        if "b_hard" in new_params:
            val = new_params["b_hard"]
            if not isinstance(val, Tensor) or val.shape != (3,):
                raise ValueError("b_hard must be Tensor[3]")
            self._b_hard = val
        unknown = set(new_params) - valid_keys
        if unknown:
            raise ValueError(f"Unknown params: {unknown}")


class BarometerModel(SensorModel):
    """Barometer forward model — altitude estimate from pressure (m).

    Simplified ISA model: returns altitude from env_state.pos[:, 2].
    """

    def __init__(self, n_envs: int, params: BaroParams | None = None) -> None:
        super().__init__(n_envs)
        p = params or BaroParams()
        self._ref_alt = p.ref_alt

    def forward(self, env_state: EnvState) -> Tensor:
        """Return barometric altitude estimate [n_envs, 1]."""
        return (env_state.pos[:, 2:3] - self._ref_alt).float()

    def update_params(self, new_params: dict) -> None:
        if "ref_alt" in new_params:
            self._ref_alt = float(new_params["ref_alt"])
        unknown = set(new_params) - {"ref_alt"}
        if unknown:
            raise ValueError(f"Unknown params: {unknown}")


class GPSModel(SensorModel):
    """GPS forward model — world-frame position (m).

    Output: pos + offset, shape [n_envs, 3].
    """

    def __init__(self, n_envs: int, params: GPSParams | None = None) -> None:
        super().__init__(n_envs)
        p = params or GPSParams()
        self._offset: Tensor = p.offset.clone()

    def forward(self, env_state: EnvState) -> Tensor:
        """Return GPS position estimate [n_envs, 3]."""
        return (env_state.pos + self._offset).float()

    def update_params(self, new_params: dict) -> None:
        if "offset" in new_params:
            val = new_params["offset"]
            if not isinstance(val, Tensor) or val.shape != (3,):
                raise ValueError("offset must be Tensor[3]")
            self._offset = val
        unknown = set(new_params) - {"offset"}
        if unknown:
            raise ValueError(f"Unknown params: {unknown}")


class OpticalFlowModel(SensorModel):
    """Optical flow forward model — horizontal velocity / altitude (m/s).

    Simplified divergence model: flow = vel_horizontal / max(alt, min_alt).
    Output: [n_envs, 2].
    """

    def __init__(self, n_envs: int, params: FlowParams | None = None) -> None:
        super().__init__(n_envs)
        p = params or FlowParams()
        self._min_alt = p.min_alt

    def forward(self, env_state: EnvState) -> Tensor:
        """Return optical flow estimate [n_envs, 2]."""
        alt = env_state.pos[:, 2:3].clamp(min=self._min_alt)
        vel_xy = env_state.vel[:, :2]
        return (vel_xy / alt).float()

    def update_params(self, new_params: dict) -> None:
        if "min_alt" in new_params:
            self._min_alt = float(new_params["min_alt"])
        unknown = set(new_params) - {"min_alt"}
        if unknown:
            raise ValueError(f"Unknown params: {unknown}")
