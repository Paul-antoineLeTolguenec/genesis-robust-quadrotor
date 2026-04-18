"""Cat 6 — Action perturbations: overhead (%) vs n_envs on Genesis CF2X.

Measures the overhead of the five action-space perturbations:

  * 6.1  action_noise
  * 6.2  action_deadzone
  * 6.3  action_saturation
  * 6.4  actuator_hysteresis
  * 6.5  esc_low_pass_filter

All five subclass ``ActionPerturbation``; their ``apply(action)`` is invoked on
a static action buffer shared by the measurement loop (no Genesis setter call).

Run:
    uv run python docs/impl/plot_perf_cat6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_6_action import (  # noqa: E402
    ActionDeadzone,
    ActionNoise,
    ActionSaturation,
    ActuatorHysteresis,
    ESCLowPassFilter,
)

CAT = 6
DT = 0.005


def _make_noise(scene, drone, n_envs: int) -> ActionNoise:
    return ActionNoise(n_envs=n_envs, dt=DT)


def _make_deadzone(scene, drone, n_envs: int) -> ActionDeadzone:
    return ActionDeadzone(n_envs=n_envs, dt=DT)


def _make_saturation(scene, drone, n_envs: int) -> ActionSaturation:
    return ActionSaturation(n_envs=n_envs, dt=DT)


def _make_hysteresis(scene, drone, n_envs: int) -> ActuatorHysteresis:
    return ActuatorHysteresis(n_envs=n_envs, dt=DT)


def _make_esc(scene, drone, n_envs: int) -> ESCLowPassFilter:
    return ESCLowPassFilter(n_envs=n_envs, dt=DT)


SPECS: list[PertSpec] = [
    PertSpec("cat6_action_noise", "action_noise", "action", _make_noise),
    PertSpec("cat6_action_deadzone", "action_deadzone", "action", _make_deadzone),
    PertSpec("cat6_action_saturation", "action_saturation", "action", _make_saturation),
    PertSpec("cat6_actuator_hysteresis", "actuator_hysteresis", "action", _make_hysteresis),
    PertSpec("cat6_esc_low_pass_filter", "esc_low_pass_filter", "action", _make_esc),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
