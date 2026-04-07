"""AdversarialEnv — thin wrapper for minimax adversarial training.

Wraps a RobustDroneEnv and exposes a second action space for the adversary.
The adversary controls a subset of perturbations via set_perturbation_values().
All step/reset logic is delegated to the wrapped env.

Reference: docs/03_api_design.md §5, docs/04_interactions.md §4.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from genesis_robust_rl.envs.robust_drone_env import RobustDroneEnv
from genesis_robust_rl.perturbations.base import EnvState, Perturbation, PerturbationMode


class AdversarialEnv:
    """Wrapper adding an adversary action space to RobustDroneEnv.

    The adversary outputs a flat Tensor[n_envs, adversary_dim] each step.
    The wrapper splits it into per-perturbation slices, calls
    set_perturbation_values(), then delegates to env.step().

    Constraints on adversary_targets (validated at construction):
      - Only scope="per_env" perturbations (global is not per-env controllable).
      - Only stateless perturbations (stateful internals override _current_value).
    """

    def __init__(
        self,
        env: RobustDroneEnv,
        adversary_targets: list[str],
        adversary_mode: Literal["value", "params"] = "value",
    ) -> None:
        if adversary_mode != "value":
            raise NotImplementedError(
                f"adversary_mode={adversary_mode!r} is not yet implemented; "
                f"only 'value' is supported"
            )

        self.env = env
        self.adversary_targets = list(adversary_targets)
        self.adversary_mode = adversary_mode

        # Resolve target perturbation objects and validate
        id_map = env._perturbation_cfg._id_map
        self._target_perturbations: list[Perturbation] = []
        for pid in self.adversary_targets:
            if pid not in id_map:
                raise KeyError(f"Unknown perturbation ID: {pid!r}")
            p = id_map[pid]
            if p.is_stateful:
                raise ValueError(f"Stateful perturbation {pid!r} cannot be an adversary target")
            if p.scope == "global":
                raise ValueError(f"Global-scope perturbation {pid!r} cannot be an adversary target")
            self._target_perturbations.append(p)

        # Compute per-target flat sizes and total adversary_dim
        self._slice_sizes: list[int] = [math.prod(p.dimension) for p in self._target_perturbations]
        adversary_dim = sum(self._slice_sizes)

        # Build adversary_action_space: flat 1D Box with concatenated bounds
        if adversary_dim > 0:
            lows: list[float] = []
            highs: list[float] = []
            for p in self._target_perturbations:
                flat_size = math.prod(p.dimension)
                lo, hi = p.bounds
                if isinstance(lo, (list, tuple)):
                    lows.extend(list(lo))
                    highs.extend(list(hi))
                else:
                    lows.extend([float(lo)] * flat_size)
                    highs.extend([float(hi)] * flat_size)
            self.adversary_action_space = gym.spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                shape=(adversary_dim,),
                dtype=np.float32,
            )
        else:
            self.adversary_action_space = gym.spaces.Box(
                low=np.zeros(0, dtype=np.float32),
                high=np.zeros(0, dtype=np.float32),
                shape=(0,),
                dtype=np.float32,
            )

        # Delegate spaces from wrapped env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Switch wrapped env to adversarial mode
        if env._mode != PerturbationMode.ADVERSARIAL:
            env.set_mode(PerturbationMode.ADVERSARIAL)

    # ------------------------------------------------------------------
    # Attribute forwarding (gym.Env compatibility)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward attribute lookups to the wrapped env for gym compatibility."""
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Gymnasium-like API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Tensor, dict]:
        """Delegate to env.reset(). Resets _current_value to nominal."""
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        drone_action: Tensor,
        adversary_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Execute one adversarial step following docs/04_interactions.md §4.

        Returns (obs, drone_reward, adv_reward, terminated, truncated, info).
        """
        # [A1] Validate adversary_action shape
        expected_shape = (self.env.n_envs, self.adversary_action_space.shape[0])
        if adversary_action.shape != expected_shape:
            raise ValueError(
                f"adversary_action shape {tuple(adversary_action.shape)} "
                f"does not match expected {expected_shape}"
            )

        # [A2] Split and apply perturbation values
        if self._slice_sizes:
            slices = torch.split(adversary_action, self._slice_sizes, dim=1)
            values: dict[str, Tensor] = {}
            for pid, p, s in zip(self.adversary_targets, self._target_perturbations, slices):
                values[pid] = s.reshape(self.env.n_envs, *p.dimension)
            self.env.set_perturbation_values(values)

        # [A3] Delegate to env.step()
        obs, drone_reward, terminated, truncated, info = self.env.step(drone_action)

        # [A4] Compute adversary reward
        adv_reward = self._adversary_reward(obs, drone_reward, self.env._last_env_state)

        return obs, drone_reward, adv_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Adversary reward (override point)
    # ------------------------------------------------------------------

    def _adversary_reward(
        self,
        obs: Tensor,
        drone_reward: Tensor,
        env_state: EnvState | None,
    ) -> Tensor:
        """Compute adversary reward. Default: zero-sum (-drone_reward).

        Override in subclasses for custom adversary objectives.
        obs is post-perturbation (what the policy sees).
        env_state is the true Genesis state (pre-observation noise).
        """
        return -drone_reward
