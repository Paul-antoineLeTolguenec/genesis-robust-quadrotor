"""Configuration dataclasses for RobustDroneEnv.

PerturbationConfig  — groups perturbations by hook; validates unique IDs.
SensorConfig        — maps sensor names to SensorModel instances.
DesyncConfig        — obs/action desync delay parameters (3.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from genesis_robust_rl.perturbations.base import (
    ActionPerturbation,
    MotorCommandPerturbation,
    ObservationPerturbation,
    PhysicsPerturbation,
)
from genesis_robust_rl.sensor_models import (
    AccelerometerModel,
    BarometerModel,
    GPSModel,
    GyroscopeModel,
    MagnetometerModel,
    OpticalFlowModel,
)


@dataclass
class PerturbationConfig:
    """Groups instantiated perturbations by application hook.

    Validates that all perturbation IDs are unique across all lists at
    construction time.
    """

    physics: list[PhysicsPerturbation] = field(default_factory=list)
    motor: list[MotorCommandPerturbation] = field(default_factory=list)
    sensors: list[ObservationPerturbation] = field(default_factory=list)
    actions: list[ActionPerturbation] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Cache the flattened list for hot-path performance
        self._all: list = [
            *self.physics,
            *self.motor,
            *self.sensors,
            *self.actions,
        ]
        ids: list[str] = []
        for p in self._all:
            if p.id in ids:
                raise ValueError(f"Duplicate perturbation ID: {p.id!r}")
            ids.append(p.id)
        # Pre-build id->perturbation map for O(1) lookup
        self._id_map: dict[str, object] = {p.id: p for p in self._all}

    def all_perturbations(self) -> list:
        """Return all perturbations in declaration order (cached)."""
        return self._all


@dataclass
class SensorConfig:
    """Maps sensor names to instantiated SensorModel objects.

    The env calls forward() on each non-None model after every scene.step().
    Results are concatenated in declaration order.
    """

    gyroscope: GyroscopeModel | None = None
    accelerometer: AccelerometerModel | None = None
    magnetometer: MagnetometerModel | None = None
    barometer: BarometerModel | None = None
    gps: GPSModel | None = None
    optical_flow: OpticalFlowModel | None = None

    # Fixed output dims per sensor type
    _SENSOR_DIMS: dict[str, int] = field(
        default_factory=lambda: {
            "gyroscope": 3,
            "accelerometer": 3,
            "magnetometer": 3,
            "barometer": 1,
            "gps": 3,
            "optical_flow": 2,
        },
        repr=False,
    )

    def active_sensors(self) -> list[tuple[str, object]]:
        """Return (name, model) pairs for non-None sensors in declaration order."""
        result = []
        sensor_names = (
            "gyroscope",
            "accelerometer",
            "magnetometer",
            "barometer",
            "gps",
            "optical_flow",
        )
        for name in sensor_names:
            model = getattr(self, name)
            if model is not None:
                result.append((name, model))
        return result

    def total_obs_dim(self) -> int:
        """Total observation dimension from all active sensors."""
        return sum(self._SENSOR_DIMS[name] for name, _ in self.active_sensors())


@dataclass(frozen=True)
class DesyncConfig:
    """Configuration for obs/action desync (perturbation 3.6).

    Not a Perturbation subclass. Instantiated internally by RobustDroneEnv
    as two DelayBuffers.
    """

    obs_delay_range: tuple[int, int]
    action_delay_range: tuple[int, int]
    correlation: float = 1.0

    def __post_init__(self) -> None:
        if self.obs_delay_range[0] < 0 or self.obs_delay_range[1] < self.obs_delay_range[0]:
            raise ValueError(f"Invalid obs_delay_range: {self.obs_delay_range}")
        act = self.action_delay_range
        if act[0] < 0 or act[1] < act[0]:
            raise ValueError(f"Invalid action_delay_range: {self.action_delay_range}")
        if not 0.0 <= self.correlation <= 1.0:
            raise ValueError(f"correlation must be in [0, 1], got {self.correlation}")
