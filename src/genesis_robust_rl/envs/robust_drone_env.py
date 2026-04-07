"""RobustDroneEnv — Gymnasium-compatible base environment for robust drone RL.

Composes Genesis simulation, perturbation engine, and sensor models into a
standard gym.Env interface. Supports domain randomization and adversarial modes.

Subclasses must implement:
  - policy_to_rpm(action) -> Tensor[n_envs, 4]
  - _compute_reward(env_state) -> (reward, terminated, truncated)
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from genesis_robust_rl.envs.config import DesyncConfig, PerturbationConfig, SensorConfig
from genesis_robust_rl.perturbations.base import (
    DelayBuffer,
    EnvState,
    Perturbation,
    PerturbationMode,
)


class RobustDroneEnv(gym.Env):
    """Base Gymnasium environment for robust quadrotor control.

    Orchestrates Genesis simulation, perturbation hooks, and sensor models
    following the sequences defined in docs/04_interactions.md.
    """

    def __init__(
        self,
        scene: Any,
        drone: Any,
        n_envs: int,
        perturbation_cfg: PerturbationConfig | None = None,
        sensor_cfg: SensorConfig | None = None,
        dt: float = 0.01,
        substeps: int = 1,
        substeps_range: tuple[int, int] | None = None,
        desync_cfg: DesyncConfig | None = None,
        max_steps: int = 1000,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.scene = scene
        self.drone = drone
        self.n_envs = n_envs
        self.dt = dt
        self.substeps = substeps
        self.substeps_range = substeps_range
        self.desync_cfg = desync_cfg
        self.max_steps = max_steps
        self.device = torch.device(device)

        self._perturbation_cfg = perturbation_cfg or PerturbationConfig()
        self._sensor_cfg = sensor_cfg or SensorConfig()

        # Runtime mode tracking
        self._mode = PerturbationMode.DOMAIN_RANDOMIZATION

        # Pre-allocated state
        self._vel_prev = torch.zeros(n_envs, 3, device=self.device)
        self._step_count = 0
        self._last_env_state: EnvState | None = None
        self._all_env_ids = torch.arange(n_envs, device=self.device)

        # Observation and action spaces (before delay buffers which need dims)
        obs_dim = self._sensor_cfg.total_obs_dim() + self._extra_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Desync delay buffers (3.6) — after action_space is defined
        self._obs_delay_buffer: DelayBuffer | None = None
        self._action_delay_buffer: DelayBuffer | None = None
        self._obs_delay: Tensor | None = None
        self._action_delay: Tensor | None = None
        if desync_cfg is not None:
            obs_max = max(desync_cfg.obs_delay_range[1], 1)
            action_max = max(desync_cfg.action_delay_range[1], 1)
            act_dim = self.action_space.shape[0]
            self._obs_delay_buffer = DelayBuffer(
                n_envs,
                obs_max,
                obs_dim,
            )
            self._action_delay_buffer = DelayBuffer(
                n_envs,
                action_max,
                act_dim,
            )

    # ------------------------------------------------------------------
    # Abstract methods (subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def policy_to_rpm(self, action: Tensor) -> Tensor:
        """Convert policy action [n_envs, action_dim] to RPM command [n_envs, 4]."""

    @abstractmethod
    def _compute_reward(self, env_state: EnvState) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (reward, terminated, truncated), each [n_envs]."""

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        env_ids: Tensor | None = None,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Tensor, dict]:
        """Reset environments following docs/04_interactions.md section 2.

        Steps [1]-[7]: scene reset, vel_prev zero, nominal restore, sensor reset,
        perturbation tick, build obs, privileged obs.
        """
        if seed is not None:
            torch.manual_seed(seed)

        full_reset = env_ids is None
        ids = self._all_env_ids if full_reset else env_ids

        # [1] Scene reset
        if full_reset:
            self.scene.reset()
        else:
            self.scene.reset(ids)

        # [2] Reset _vel_prev
        if full_reset:
            self._vel_prev.zero_()
        else:
            self._vel_prev[ids] = 0.0

        # Reset step count on full reset
        if full_reset:
            self._step_count = 0

        # [3] Restore _current_value = nominal for each perturbation
        for p in self._perturbation_cfg.all_perturbations():
            self._reset_current_value(p, ids, full_reset)

        # [4] Reset sensor models
        for _, model in self._sensor_cfg.active_sensors():
            model.reset(ids)

        # [5] tick(is_reset=True) on all perturbations
        for p in self._perturbation_cfg.all_perturbations():
            p.tick(is_reset=True, env_ids=ids)

        # Reset desync delay buffers
        if self._obs_delay_buffer is not None:
            self._obs_delay_buffer.reset(ids)
            self._action_delay_buffer.reset(ids)  # type: ignore[union-attr]
            self._sample_desync_delays(ids)

        # [6] Build initial obs
        env_state = self._build_env_state(self.dt * self.substeps)
        self._last_env_state = env_state
        obs = self._build_obs(env_state)

        # [7] Build info
        info: dict[str, Any] = {"privileged_obs": self.get_privileged_obs()}

        return obs, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """Execute one RL step following docs/04_interactions.md section 3.

        Steps [0]-[11]: desync action delay, action perturbations, policy_to_rpm,
        motor perturbations, substeps, build env_state, physics perturbations,
        sensor models, sensor perturbations, desync obs delay, tick, reward,
        privileged obs.
        """
        # [0] Desync action delay (3.6)
        if self._action_delay_buffer is not None and self._action_delay is not None:
            action_to_use = self._action_delay_buffer.read(self._action_delay)
            self._action_delay_buffer.push(action)
        else:
            action_to_use = action

        # [1] Action perturbations (hook: on_action)
        action_perturbed = self._apply_action_perturbations(action_to_use)

        # [2] Policy to RPM
        rpm_cmd = self.policy_to_rpm(action_perturbed)

        # [3] Motor perturbations (hook: pre_physics)
        rpm_actual = self._apply_motor_perturbations(rpm_cmd)

        # [4] Substeps
        n_substeps = self._get_substeps()
        for _ in range(n_substeps):
            self.drone.set_propellels_rpm(rpm_actual)
            self.scene.step()

        # [5] Build env_state
        rl_dt = n_substeps * self.dt
        env_state = self._build_env_state(rl_dt)
        self._last_env_state = env_state

        # [6] Physics perturbations (hook: post_physics)
        self._apply_physics_perturbations(env_state)

        # [7] + [8] Sensor models + sensor perturbations + extra obs
        obs = self._build_obs(env_state)

        # [8b] Desync obs delay (3.6)
        if self._obs_delay_buffer is not None and self._obs_delay is not None:
            obs_delayed = self._obs_delay_buffer.read(self._obs_delay)
            self._obs_delay_buffer.push(obs)
            obs = obs_delayed

        # [9] tick(is_reset=False) on all perturbations
        if self.substeps_range is not None:
            for p in self._perturbation_cfg.all_perturbations():
                p.dt = rl_dt
        for p in self._perturbation_cfg.all_perturbations():
            p.tick(is_reset=False, env_ids=self._all_env_ids)

        # [10] Reward
        reward, terminated, truncated = self._compute_reward(env_state)

        # Increment step count
        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = torch.ones(self.n_envs, device=self.device, dtype=torch.bool)

        # [11] Privileged obs
        info: dict[str, Any] = {"privileged_obs": self.get_privileged_obs()}

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal: build env state
    # ------------------------------------------------------------------

    def _build_env_state(self, rl_dt: float) -> EnvState:
        """Read Genesis state and compute acceleration from velocity delta."""
        pos = self.drone.get_pos()
        quat = self.drone.get_quat()
        vel = self.drone.get_vel()
        ang_vel = self.drone.get_ang_vel()

        acc = (vel - self._vel_prev) / rl_dt
        self._vel_prev = vel.clone() if isinstance(vel, Tensor) else vel

        rpm = (
            self.drone.get_rpm()
            if hasattr(self.drone, "get_rpm")
            else torch.zeros(self.n_envs, 4, device=self.device)
        )

        return EnvState(
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            acc=acc,
            rpm=rpm,
            dt=rl_dt,
            step=self._step_count,
        )

    # ------------------------------------------------------------------
    # Internal: build observation (shared by reset and step)
    # ------------------------------------------------------------------

    def _build_obs(self, env_state: EnvState) -> Tensor:
        """Run sensor models, apply sensor perturbations, append extra obs."""
        # [7] Sensor models
        raw_obs = self._run_sensor_models(env_state)

        # [8] Sensor perturbations
        obs = self._apply_sensor_perturbations(raw_obs)

        # Append extra obs if subclass provides it
        extra = self._extra_obs()
        if extra is not None:
            obs = torch.cat([obs, extra], dim=1)

        return obs

    def _run_sensor_models(self, env_state: EnvState) -> Tensor:
        """Call forward() on each active sensor, concatenate results."""
        readings = []
        for _, model in self._sensor_cfg.active_sensors():
            readings.append(model.forward(env_state))
        if not readings:
            return torch.zeros(self.n_envs, 0, device=self.device)
        return torch.cat(readings, dim=1)

    # ------------------------------------------------------------------
    # Internal: perturbation hooks
    # ------------------------------------------------------------------

    def _apply_physics_perturbations(self, env_state: EnvState) -> None:
        """Apply physics perturbations (hook: post_physics)."""
        for p in self._perturbation_cfg.physics:
            p.apply(self.scene, self.drone, env_state)

    def _apply_motor_perturbations(self, rpm_cmd: Tensor) -> Tensor:
        """Chain motor perturbations (hook: pre_physics)."""
        for p in self._perturbation_cfg.motor:
            rpm_cmd = p.apply(rpm_cmd)
        return rpm_cmd

    def _apply_sensor_perturbations(self, raw_obs: Tensor) -> Tensor:
        """Chain sensor perturbations (hook: on_observation)."""
        for p in self._perturbation_cfg.sensors:
            raw_obs = p.apply(raw_obs)
        return raw_obs

    def _apply_action_perturbations(self, action: Tensor) -> Tensor:
        """Chain action perturbations (hook: on_action)."""
        for p in self._perturbation_cfg.actions:
            action = p.apply(action)
        return action

    # ------------------------------------------------------------------
    # Internal: substeps (3.5)
    # ------------------------------------------------------------------

    def _get_substeps(self) -> int:
        """Return number of scene.step() calls per RL step."""
        if self.substeps_range is not None:
            lo, hi = self.substeps_range
            return int(torch.randint(lo, hi + 1, (1,)).item())
        return self.substeps

    # ------------------------------------------------------------------
    # Internal: desync (3.6)
    # ------------------------------------------------------------------

    def _sample_desync_delays(self, env_ids: Tensor) -> None:
        """Sample correlated obs/action delays for the given envs."""
        assert self.desync_cfg is not None
        n = env_ids.shape[0]
        cfg = self.desync_cfg

        # Sample two uniform values in [0, 1]
        u1 = torch.rand(n, device=self.device)

        # Correlate via Cholesky: u2 = rho * u1 + sqrt(1 - rho^2) * u_indep
        rho = cfg.correlation
        u_indep = torch.rand(n, device=self.device)
        u2 = rho * u1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * u_indep
        u2 = u2.clamp(0.0, 1.0)

        # Quantize to integer delays
        obs_lo, obs_hi = cfg.obs_delay_range
        act_lo, act_hi = cfg.action_delay_range
        obs_delays = (obs_lo + u1 * (obs_hi - obs_lo + 1)).long().clamp(obs_lo, obs_hi)
        act_delays = (act_lo + u2 * (act_hi - act_lo + 1)).long().clamp(act_lo, act_hi)

        if self._obs_delay is None:
            self._obs_delay = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
            self._action_delay = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

        self._obs_delay[env_ids] = obs_delays
        self._action_delay[env_ids] = act_delays  # type: ignore[index]

    # ------------------------------------------------------------------
    # Internal: nominal reset helper
    # ------------------------------------------------------------------

    def _reset_current_value(
        self,
        p: Perturbation,
        env_ids: Tensor,
        full_reset: bool,
    ) -> None:
        """Set _current_value to nominal for the given envs."""
        shape = p._batch_shape()
        nominal_t = torch.as_tensor(
            p.nominal,
            dtype=torch.float32,
            device=self.device,
        ).expand(shape)

        if full_reset or p.scope == "global":
            p._current_value = nominal_t.clone()
        else:
            if p._current_value is None:
                full_shape = (p.n_envs, *p.dimension)
                p._current_value = nominal_t.expand(full_shape).clone()
            p._current_value[env_ids] = nominal_t[0]

    # ------------------------------------------------------------------
    # Extra obs hook (subclass override)
    # ------------------------------------------------------------------

    @property
    def privileged_obs_dim(self) -> int:
        """Total dimension of privileged observations (all observable perturbations)."""
        return sum(
            math.prod(p.dimension)
            for p in self._perturbation_cfg.all_perturbations()
            if p.observable
        )

    def _extra_obs_dim(self) -> int:
        """Return dimension of extra observations appended after sensor readings."""
        return 0

    def _extra_obs(self) -> Tensor | None:
        """Override in subclasses to append extra state to the observation."""
        return None

    # ------------------------------------------------------------------
    # Privileged obs + mode switch + curriculum
    # ------------------------------------------------------------------

    def get_privileged_obs(self) -> Tensor:
        """Concatenate privileged obs from all observable perturbations.

        Ordering: physics -> motor -> sensors -> actions, in list order.
        Global-scope values are broadcast to [n_envs, *dim].
        """
        parts: list[Tensor] = []
        for p in self._perturbation_cfg.all_perturbations():
            val = p.get_privileged_obs()
            if val is None:
                continue
            # Broadcast global [1, *dim] -> [n_envs, *dim]
            if val.shape[0] == 1 and self.n_envs > 1:
                val = val.expand(self.n_envs, *val.shape[1:])
            # Flatten to [n_envs, -1]
            parts.append(val.reshape(self.n_envs, -1))
        if not parts:
            return torch.zeros(self.n_envs, 0, device=self.device)
        return torch.cat(parts, dim=1)

    def set_mode(self, mode: PerturbationMode) -> None:
        """Switch all perturbations between DR and adversarial mode."""
        self._mode = mode
        for p in self._perturbation_cfg.all_perturbations():
            p.mode = mode

    def _check_adversarial(self, method: str) -> None:
        """Raise ValueError if not in adversarial mode."""
        if self._mode != PerturbationMode.ADVERSARIAL:
            raise ValueError(f"{method}() is only valid in ADVERSARIAL mode")

    def set_perturbation_values(self, values: dict[str, Tensor]) -> None:
        """Adversarial mode: set perturbation values by ID.

        Raises ValueError if called in DR mode. Raises KeyError if ID not found.
        """
        self._check_adversarial("set_perturbation_values")
        id_map = self._perturbation_cfg._id_map
        for pid, value in values.items():
            if pid not in id_map:
                raise KeyError(f"Unknown perturbation ID: {pid!r}")
            id_map[pid].set_value(value)

    def update_perturbation_params(self, params: dict[str, dict]) -> None:
        """Adversarial mode: update distribution params by ID.

        Raises ValueError if called in DR mode. Raises KeyError if ID not found.
        """
        self._check_adversarial("update_perturbation_params")
        id_map = self._perturbation_cfg._id_map
        for pid, new_params in params.items():
            if pid not in id_map:
                raise KeyError(f"Unknown perturbation ID: {pid!r}")
            id_map[pid].update_params(new_params)

    def set_curriculum_scale(self, scale: float | dict[str, float]) -> None:
        """Set curriculum_scale on perturbations (DR mode only)."""
        if isinstance(scale, (int, float)):
            for p in self._perturbation_cfg.all_perturbations():
                p.curriculum_scale = float(scale)
        else:
            id_map = self._perturbation_cfg._id_map
            for pid, s in scale.items():
                if pid not in id_map:
                    raise KeyError(f"Unknown perturbation ID: {pid!r}")
                id_map[pid].curriculum_scale = float(s)
