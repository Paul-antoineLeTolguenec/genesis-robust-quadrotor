"""Cat 3 — Temporal perturbations: overhead (%) vs n_envs on Genesis CF2X.

Measures the overhead of the six temporal perturbations:

  * 3.1  obs_fixed_delay            (obs kind)
  * 3.2  obs_variable_delay         (obs kind)
  * 3.3  action_fixed_delay         (action kind)
  * 3.4  action_variable_delay      (action kind)
  * 3.7  packet_loss                (action kind — Bernoulli hold)
  * 3.8  computation_overload       (action kind — burst skip)

Run:
    uv run python docs/impl/plot_perf_cat3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_3_temporal import (  # noqa: E402
    ActionFixedDelay,
    ActionVariableDelay,
    ComputationOverload,
    ObsFixedDelay,
    ObsVariableDelay,
    PacketLoss,
)

CAT = 3
DT = 0.005
OBS_SLICE = slice(0, 3)
OBS_DIM = 3
ACTION_DIM = 4


def _make_obs_fixed(scene, drone, n_envs: int) -> ObsFixedDelay:
    return ObsFixedDelay(obs_slice=OBS_SLICE, obs_dim=OBS_DIM, n_envs=n_envs, dt=DT)


def _make_obs_variable(scene, drone, n_envs: int) -> ObsVariableDelay:
    return ObsVariableDelay(obs_slice=OBS_SLICE, obs_dim=OBS_DIM, n_envs=n_envs, dt=DT)


def _make_action_fixed(scene, drone, n_envs: int) -> ActionFixedDelay:
    return ActionFixedDelay(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM)


def _make_action_variable(scene, drone, n_envs: int) -> ActionVariableDelay:
    return ActionVariableDelay(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM)


def _make_packet_loss(scene, drone, n_envs: int) -> PacketLoss:
    return PacketLoss(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM)


def _make_overload(scene, drone, n_envs: int) -> ComputationOverload:
    return ComputationOverload(n_envs=n_envs, dt=DT, action_dim=ACTION_DIM)


SPECS: list[PertSpec] = [
    PertSpec("cat3_obs_fixed_delay", "obs_fixed_delay", "obs", _make_obs_fixed),
    PertSpec("cat3_obs_variable_delay", "obs_variable_delay", "obs", _make_obs_variable),
    PertSpec("cat3_action_fixed_delay", "action_fixed_delay", "action", _make_action_fixed),
    PertSpec(
        "cat3_action_variable_delay",
        "action_variable_delay",
        "action",
        _make_action_variable,
    ),
    PertSpec("cat3_packet_loss", "packet_loss", "action", _make_packet_loss),
    PertSpec(
        "cat3_computation_overload",
        "computation_overload",
        "action",
        _make_overload,
    ),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
