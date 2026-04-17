"""Cat 8 — External disturbances: overhead (%) vs n_envs on Genesis CF2X.

Measures the overhead of the two external-wrench perturbations:

  * 8.1  body_force_disturbance   (constant uniform, 3-DoF force)
  * 8.2  body_torque_disturbance  (constant uniform, 3-DoF torque)

Run:
    uv run python docs/impl/plot_perf_cat8.py

Writes, per perturbation, one ``_perf.{csv,meta.json,png}`` under
``docs/impl/{data,assets}/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_8_external import (  # noqa: E402
    BodyForceDisturbance,
    BodyTorqueDisturbance,
)

CAT = 8
DT = 0.005


def _make_body_force(scene, drone, n_envs: int) -> BodyForceDisturbance:
    return BodyForceDisturbance(n_envs=n_envs, dt=DT)


def _make_body_torque(scene, drone, n_envs: int) -> BodyTorqueDisturbance:
    return BodyTorqueDisturbance(n_envs=n_envs, dt=DT)


SPECS: list[PertSpec] = [
    PertSpec(
        slug="cat8_body_force_disturbance",
        display_name="body_force_disturbance",
        kind="physics",
        factory=_make_body_force,
    ),
    PertSpec(
        slug="cat8_body_torque_disturbance",
        display_name="body_torque_disturbance",
        kind="physics",
        factory=_make_body_torque,
    ),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
