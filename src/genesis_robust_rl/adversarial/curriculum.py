"""Curriculum scheduler for progressive perturbation intensity."""

from __future__ import annotations

import math
from typing import Callable

from genesis_robust_rl.envs.robust_drone_env import RobustDroneEnv


class CurriculumScheduler:
    """Callback that adjusts curriculum_scale over training.

    Plugs into TrainConfig.callbacks. Calls env.set_curriculum_scale()
    with a schedule-derived scale at each invocation.
    DR mode only — no effect in adversarial mode (by design).
    """

    def __init__(
        self,
        env: RobustDroneEnv,
        schedule: str | Callable[[int, int], float] = "linear",
        start_scale: float = 0.0,
        end_scale: float = 1.0,
    ) -> None:
        self.env = env
        self.start_scale = start_scale
        self.end_scale = end_scale

        if callable(schedule):
            self._schedule_fn = schedule
        elif schedule == "linear":
            self._schedule_fn = self._linear
        elif schedule == "cosine":
            self._schedule_fn = self._cosine
        elif schedule == "step":
            self._schedule_fn = self._step
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

    def __call__(self, step: int, total_steps: int, **kwargs) -> None:
        """Compute scale and apply to env."""
        scale = self._schedule_fn(step, total_steps)
        self.env.set_curriculum_scale(scale)

    def _linear(self, step: int, total_steps: int) -> float:
        ratio = min(step / max(total_steps, 1), 1.0)
        return self.start_scale + (self.end_scale - self.start_scale) * ratio

    def _cosine(self, step: int, total_steps: int) -> float:
        ratio = min(step / max(total_steps, 1), 1.0)
        return (
            self.end_scale
            - (self.end_scale - self.start_scale) * (1.0 + math.cos(math.pi * ratio)) / 2.0
        )

    def _step(self, step: int, total_steps: int) -> float:
        return self.start_scale if step < total_steps / 2 else self.end_scale
