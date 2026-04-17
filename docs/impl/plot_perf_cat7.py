"""Cat 7 — Payload: overhead (%) vs n_envs on Genesis CF2X.

Measures the overhead of the three payload perturbations:

  * 7.1  payload_mass                (GenesisSetter, scalar Δm kg)
  * 7.2  payload_com_offset          (GenesisSetter, Δr m, 3-DoF)
  * 7.3  asymmetric_prop_guard_drag  (ExternalWrench, per-arm drag ratios)

Run:
    uv run python docs/impl/plot_perf_cat7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_7_payload import (  # noqa: E402
    AsymmetricPropGuardDrag,
    PayloadCOMOffset,
    PayloadMass,
)

CAT = 7
DT = 0.005


def _make_payload_mass(scene, drone, n_envs: int) -> PayloadMass:
    return PayloadMass(
        setter_fn=lambda v, idx: drone.set_mass_shift(v, idx),
        n_envs=n_envs,
        dt=DT,
    )


def _make_payload_com(scene, drone, n_envs: int) -> PayloadCOMOffset:
    return PayloadCOMOffset(
        setter_fn=lambda v, idx: drone.set_COM_shift(v, idx),
        n_envs=n_envs,
        dt=DT,
    )


def _make_asymmetric_drag(scene, drone, n_envs: int) -> AsymmetricPropGuardDrag:
    return AsymmetricPropGuardDrag(n_envs=n_envs, dt=DT)


SPECS: list[PertSpec] = [
    PertSpec(
        slug="cat7_payload_mass",
        display_name="payload_mass",
        kind="physics",
        factory=_make_payload_mass,
    ),
    PertSpec(
        slug="cat7_payload_com_offset",
        display_name="payload_com_offset",
        kind="physics",
        factory=_make_payload_com,
    ),
    PertSpec(
        slug="cat7_asymmetric_prop_guard_drag",
        display_name="asymmetric_prop_guard_drag",
        kind="physics",
        factory=_make_asymmetric_drag,
    ),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
