"""Base classes for the perturbation engine.

Hierarchy:
  OUProcess, DelayBuffer           — utility components (composition)
  Perturbation (ABC)               — base: config, lifecycle, Lipschitz, curriculum
    PhysicsPerturbation (ABC)      — apply(scene, drone, env_state)
      GenesisSetterPerturbation    — wraps a Genesis API setter
      ExternalWrenchPerturbation   — force/torque via solver API
    MotorCommandPerturbation (ABC) — apply(rpm_cmd) -> rpm_cmd
    ObservationPerturbation (ABC)  — apply(obs) -> obs
    ActionPerturbation (ABC)       — apply(action) -> action
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------


@dataclass
class EnvState:
    """Drone state snapshot passed to physics hooks at each env step."""

    pos: Tensor  # [n_envs, 3]  world position (m)
    quat: Tensor  # [n_envs, 4]  orientation quaternion (w, x, y, z)
    vel: Tensor  # [n_envs, 3]  body linear velocity (m/s)
    ang_vel: Tensor  # [n_envs, 3]  body angular velocity (rad/s)
    acc: Tensor  # [n_envs, 3]  body linear acceleration (m/s²)
    rpm: Tensor  # [n_envs, 4]  propeller RPM
    dt: float
    step: int


# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------


class PerturbationMode(str, Enum):
    """Controls whether perturbation values are sampled (DR) or set externally (adversarial)."""

    DOMAIN_RANDOMIZATION = "domain_randomization"
    ADVERSARIAL = "adversarial"


# ---------------------------------------------------------------------------
# Utility components
# ---------------------------------------------------------------------------


class OUProcess:
    """Ornstein-Uhlenbeck process state for a batch of environments.

    All ops use torch exclusively — no numpy, no Python loops over envs.
    """

    def __init__(self, n_envs: int, dim: int) -> None:
        self.state: Tensor = torch.zeros(n_envs, dim)

    def reset(self, env_ids: Tensor) -> None:
        """Reset state to zero for selected envs."""
        self.state[env_ids] = 0.0

    def step(self, theta: float, sigma: float, mu: float, dt: float) -> None:
        """Advance OU process: dx = θ(μ−x)dt + σ√dt·ε."""
        noise = torch.randn_like(self.state)
        self.state += theta * (mu - self.state) * dt + sigma * math.sqrt(dt) * noise


class DelayBuffer:
    """Per-env circular delay buffer.

    Stores the last `max_delay` steps for a batch of environments.
    Fully vectorized — no Python loops over the env dimension.

    Canonical index:
      push: buffer[env, write_ptr[env]] = x[env]; write_ptr = (write_ptr + 1) % max_delay
      read: idx = (write_ptr - delay - 1) % max_delay
    delay=0 → returns the just-pushed value (pass-through).
    Cold start: uninitialized positions read as zero.
    """

    def __init__(self, n_envs: int, max_delay: int, dim: int) -> None:
        self.buffer: Tensor = torch.zeros(n_envs, max_delay, dim)
        self.write_ptr: Tensor = torch.zeros(n_envs, dtype=torch.long)
        self._max_delay = max_delay

    def reset(self, env_ids: Tensor) -> None:
        """Zero buffer and reset write pointer for selected envs."""
        self.buffer[env_ids] = 0.0
        self.write_ptr[env_ids] = 0

    def push(self, x: Tensor) -> None:
        """Write x into the buffer at the current write position."""
        envs = torch.arange(x.shape[0], device=x.device)
        self.buffer[envs, self.write_ptr] = x
        self.write_ptr = (self.write_ptr + 1) % self._max_delay

    def read(self, delay: Tensor) -> Tensor:
        """Read value delayed by `delay` steps (integer, clamped to [0, max_delay-1])."""
        delay = delay.clamp(0, self._max_delay - 1)
        envs = torch.arange(self.buffer.shape[0], device=self.buffer.device)
        idx = (self.write_ptr - delay - 1) % self._max_delay
        return self.buffer[envs, idx]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PerturbationRegistry:
    """Config-driven perturbation instantiation utility.

    Each Perturbation leaf is decorated with @perturbation_registry.register("id")
    at class definition time. The registry is a standalone utility; the env takes
    instantiated Perturbation objects, not registry keys.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Perturbation]] = {}

    def register(self, id: str) -> Callable[[type], type]:
        """Decorator: register a Perturbation subclass under its catalog id."""

        def decorator(cls: type) -> type:
            if id in self._registry:
                raise ValueError(f"Duplicate registry key: {id!r}")
            self._registry[id] = cls
            cls._registry_id = id  # type: ignore[attr-defined]
            return cls

        return decorator

    def get(self, id: str) -> type["Perturbation"]:
        """Retrieve a class by key. Raises KeyError if not found."""
        if id not in self._registry:
            raise KeyError(f"Unknown perturbation id: {id!r}")
        return self._registry[id]

    def list(self) -> list[str]:
        """Return all registered perturbation IDs, sorted alphabetically."""
        return sorted(self._registry.keys())

    def build(self, id: str, n_envs: int, dt: float, **kwargs: Any) -> "Perturbation":
        """Instantiate a perturbation from its registered key + kwargs."""
        cls = self.get(id)
        return cls(n_envs=n_envs, dt=dt, **kwargs)

    def build_from_config(self, config: dict) -> "Perturbation":
        """Build from a plain dict: {"id": ..., "n_envs": ..., "dt": ..., ...kwargs}."""
        config = dict(config)
        id = config.pop("id")
        n_envs = config.pop("n_envs")
        dt = config.pop("dt")
        return self.build(id, n_envs, dt, **config)


# Global singleton
perturbation_registry = PerturbationRegistry()

# Backward-compatible alias
PERTURBATION_REGISTRY = perturbation_registry._registry


def register(id: str) -> Callable[[type], type]:
    """Shortcut decorator using the global registry singleton."""
    return perturbation_registry.register(id)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Perturbation(ABC):
    """Base class for all perturbations.

    Encapsulates catalog config, lifecycle (tick/reset/step), domain-randomization
    sampling with curriculum scaling, adversarial value injection with Lipschitz
    enforcement, and privileged observation exposure.
    """

    def __init__(
        self,
        id: str,
        n_envs: int,
        dt: float,
        value_mode: Literal["fixed", "dynamic"],
        frequency: Literal["per_episode", "per_step"],
        scope: Literal["per_env", "global"],
        distribution: str,
        distribution_params: dict,
        bounds: tuple[float, float],
        nominal: Any,
        dimension: tuple[int, ...],
        lipschitz_k: float | None = None,
        curriculum_scale: float = 1.0,
        observable: bool = True,
        risk: Literal["low", "medium", "high"] = "low",
        priority: int = 1,
    ) -> None:
        self.id = id
        self.n_envs = n_envs
        self.dt = dt
        self.value_mode = value_mode
        self.frequency = frequency
        self.scope = scope
        self.distribution = distribution
        self.distribution_params = dict(distribution_params)
        self.bounds = bounds
        self.nominal = nominal
        self.dimension = dimension
        self.lipschitz_k = lipschitz_k
        self.curriculum_scale = curriculum_scale
        self.observable = observable
        self.risk = risk
        self.priority = priority

        # Runtime mode — DR by default, switch to ADVERSARIAL for adversary control
        self.mode: PerturbationMode = PerturbationMode.DOMAIN_RANDOMIZATION

        # Internal state
        self._current_value: Tensor | None = None
        self._params_prev: dict | None = None
        self.is_stateful: bool = False  # stateful leaves set this to True

    # --- Lifecycle ---

    def reset(self, env_ids: Tensor) -> None:
        """Reset internal state for selected envs. No-op for stateless leaves."""

    def tick(self, is_reset: bool, env_ids: Tensor | None = None) -> None:
        """Orchestrate sampling and state advancement for one env step.

        In ADVERSARIAL mode, tick(is_reset=False) never calls sample().
        step() is always called on stateful perturbations regardless of mode.
        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs)
        if is_reset:
            self.reset(env_ids)
            if (
                self.mode == PerturbationMode.DOMAIN_RANDOMIZATION
                and self.frequency == "per_episode"
            ):
                self.sample()
        else:
            if self.mode == PerturbationMode.DOMAIN_RANDOMIZATION and self.frequency == "per_step":
                self.sample()
            if self.is_stateful:
                self.step()

    def step(self) -> None:
        """Advance internal stateful dynamics. No-op in base; override in stateful leaves."""

    # --- Value interface ---

    def _batch_shape(self) -> tuple[int, ...]:
        n = 1 if self.scope == "global" else self.n_envs
        return (n,) + self.dimension

    def _draw(self) -> Tensor:
        """Draw a raw sample from the configured distribution."""
        shape = self._batch_shape()
        p = self.distribution_params
        if self.distribution == "uniform":
            return torch.empty(shape).uniform_(p["low"], p["high"])
        if self.distribution == "gaussian":
            return torch.normal(mean=torch.full(shape, float(p["mean"])), std=float(p["std"]))
        if self.distribution == "constant":
            return torch.full(shape, float(p["value"]))
        if self.distribution == "beta":
            lo, hi = self.bounds
            raw = torch.distributions.Beta(float(p["alpha"]), float(p["beta"])).sample(shape)
            return lo + raw * (hi - lo)
        raise ValueError(f"Unknown distribution: {self.distribution!r}")

    def sample(self) -> Tensor:
        """DR mode: draw from distribution, apply curriculum_scale, clip to bounds."""
        raw = self._draw()
        nominal = torch.tensor(self.nominal, dtype=torch.float32).expand(raw.shape)
        value = nominal + (raw - nominal) * self.curriculum_scale
        value = value.clamp(self.bounds[0], self.bounds[1]).to(torch.float32)
        self._current_value = value
        return value

    def set_value(self, value: Tensor) -> None:
        """Adversarial mode: Lipschitz-clip value delta and store."""
        if self.lipschitz_k is not None and self._current_value is not None:
            max_d = self.lipschitz_k * self.dt
            value = self._current_value + (value - self._current_value).clamp(-max_d, max_d)
        self._current_value = value

    def update_params(self, new_params: dict) -> None:
        """Adversarial mode: Lipschitz-clip each distribution param (L∞ element-wise)."""
        if self.lipschitz_k is not None and self._params_prev is not None:
            max_d = self.lipschitz_k * self.dt
            clipped: dict = {}
            for key, val in new_params.items():
                if key not in self._params_prev:
                    clipped[key] = val
                    continue
                prev = self._params_prev[key]
                if isinstance(val, Tensor) and isinstance(prev, Tensor):
                    clipped[key] = prev + (val - prev).clamp(-max_d, max_d)
                elif isinstance(val, (int, float)) and isinstance(prev, (int, float)):
                    clipped[key] = float(torch.tensor(val).clamp(prev - max_d, prev + max_d))
                else:
                    clipped[key] = val
            new_params = clipped
        self._params_prev = dict(new_params)
        self.distribution_params = new_params

    def get_privileged_obs(self) -> Tensor | None:
        """Return _current_value for privileged obs vector, or None if not observable."""
        if not self.observable:
            return None
        return self._current_value


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


class PhysicsPerturbation(Perturbation, ABC):
    """Perturbation applied via Genesis scene/drone API (called at env step [6])."""

    @abstractmethod
    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        """Apply perturbation side-effects to the Genesis scene."""


class GenesisSetterPerturbation(PhysicsPerturbation):
    """Wraps a single Genesis API setter: setter_fn(value, envs_idx).

    The setter_fn is bound at construction (caller passes e.g. drone.set_links_mass_shift).
    apply() calls setter_fn with _current_value and a full envs_idx tensor.

    Performance: Genesis setters persist the value server-side until overwritten,
    so ``apply()`` skips the call whenever the current value has not changed
    since the last apply. Change detection uses tensor identity AND torch's
    ``._version`` counter, so in-place mutations (``value[...] = x``) also
    invalidate the cache. Callers of :meth:`set_value` must still treat the
    contract as "pass a fresh tensor"; the version counter is a safety net,
    not a license for in-place mutation.

    ``_envs_idx_cache`` is allocated lazily on the first apply(), on the same
    device as ``_current_value`` — this keeps the perturbation portable between
    CPU and CUDA Genesis backends.
    """

    def __init__(self, setter_fn: Callable[[Tensor, Tensor], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.setter_fn = setter_fn
        self._last_applied: Tensor | None = None
        self._last_applied_version: int = -1
        self._envs_idx_cache: Tensor | None = None

    def reset(self, env_ids: Tensor) -> None:
        super().reset(env_ids)
        self._last_applied = None
        self._last_applied_version = -1

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        assert self._current_value is not None, "call tick() before apply()"
        cur = self._current_value
        if self._last_applied is cur and self._last_applied_version == cur._version:
            return
        if self._envs_idx_cache is None or self._envs_idx_cache.device != cur.device:
            self._envs_idx_cache = torch.arange(self.n_envs, device=cur.device)
        self.setter_fn(cur, self._envs_idx_cache)
        self._last_applied = cur
        self._last_applied_version = cur._version


class ExternalWrenchPerturbation(PhysicsPerturbation, ABC):
    """Applies a force or torque via the Genesis external wrench API.

    Stateless leaves override only _compute_wrench().
    Stateful leaves also set is_stateful=True and override step().
    """

    def __init__(
        self,
        frame: Literal["local", "world"],
        link_idx: int,
        duration_mode: Literal["continuous", "pulse"],
        wrench_type: Literal["force", "torque"] = "force",
        preserve_current_value: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.frame = frame
        self.link_idx = link_idx
        self.duration_mode = duration_mode
        self.wrench_type = wrench_type
        self._preserve_current_value = preserve_current_value
        # Pre-allocated buffers for apply()
        self._envs_idx: Tensor = torch.arange(self.n_envs)

    @abstractmethod
    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute wrench tensor [n_envs, 3]."""

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        wrench = self._compute_wrench(env_state)
        if not self._preserve_current_value:
            self._current_value = wrench
        local = self.frame == "local"
        envs_idx = self._envs_idx.to(wrench.device)
        if self.wrench_type == "torque":
            scene.rigid_solver.apply_links_external_torque(
                wrench,
                self.link_idx,
                envs_idx,
                local=local,
            )
        else:
            scene.rigid_solver.apply_links_external_force(
                wrench,
                self.link_idx,
                envs_idx,
                local=local,
            )


class MotorCommandPerturbation(Perturbation, ABC):
    """Perturbation applied to the RPM command tensor before thrust physics."""

    @abstractmethod
    def apply(self, rpm_cmd: Tensor) -> Tensor:
        """Transform rpm_cmd [n_envs, 4] → [n_envs, 4]."""


class ObservationPerturbation(Perturbation, ABC):
    """Perturbation applied to the observation tensor."""

    def __init__(self, obs_slice: slice, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.obs_slice = obs_slice

    @abstractmethod
    def apply(self, obs: Tensor) -> Tensor:
        """Transform obs [n_envs, obs_dim] → [n_envs, obs_dim]."""


class ActionPerturbation(Perturbation, ABC):
    """Perturbation applied to the action tensor."""

    @abstractmethod
    def apply(self, action: Tensor) -> Tensor:
        """Transform action [n_envs, action_dim] → [n_envs, action_dim]."""
