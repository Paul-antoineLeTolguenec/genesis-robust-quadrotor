"""Cat 5 — Wind perturbations: overhead (%) vs n_envs on Genesis CF2X.

All 9 wind perturbations subclass ``ExternalWrenchPerturbation``; their
``apply(scene, drone, env_state)`` applies a force or torque via
``scene.rigid_solver.apply_links_external_force``.

Run:
    uv run python docs/impl/plot_perf_cat5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_5_wind import (  # noqa: E402
    AdversarialWind,
    BladeVortexInteraction,
    ConstantWind,
    GroundEffectBoundary,
    PayloadSway,
    ProximityDisturbance,
    Turbulence,
    WindGust,
    WindShear,
)

CAT = 5
DT = 0.005


def _simple_factory(cls):
    def factory(scene, drone, n_envs: int):
        return cls(n_envs=n_envs, dt=DT)

    return factory


SPECS: list[PertSpec] = [
    PertSpec("cat5_constant_wind", "constant_wind", "physics", _simple_factory(ConstantWind)),
    PertSpec("cat5_turbulence", "turbulence", "physics", _simple_factory(Turbulence)),
    PertSpec("cat5_wind_gust", "wind_gust", "physics", _simple_factory(WindGust)),
    PertSpec("cat5_wind_shear", "wind_shear", "physics", _simple_factory(WindShear)),
    PertSpec(
        "cat5_adversarial_wind",
        "adversarial_wind",
        "physics",
        _simple_factory(AdversarialWind),
    ),
    PertSpec(
        "cat5_blade_vortex_interaction",
        "blade_vortex_interaction",
        "physics",
        _simple_factory(BladeVortexInteraction),
    ),
    PertSpec(
        "cat5_ground_effect_boundary",
        "ground_effect_boundary",
        "physics",
        _simple_factory(GroundEffectBoundary),
    ),
    PertSpec("cat5_payload_sway", "payload_sway", "physics", _simple_factory(PayloadSway)),
    PertSpec(
        "cat5_proximity_disturbance",
        "proximity_disturbance",
        "physics",
        _simple_factory(ProximityDisturbance),
    ),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
