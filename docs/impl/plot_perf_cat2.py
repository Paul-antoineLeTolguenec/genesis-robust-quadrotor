"""Cat 2 — Motor perturbations: overhead (%) vs n_envs on Genesis CF2X.

Cat 2 is split between two apply signatures:

  * ExternalWrench subclasses (6) → physics kind:
      thrust_coeff_kf, torque_coeff_km, propeller_thrust_asymmetry,
      motor_partial_failure, motor_back_emf, gyroscopic_effect
  * MotorCommand subclasses (7) → motor kind:
      motor_kill, motor_lag, motor_rpm_noise, motor_saturation,
      motor_wear, rotor_imbalance, motor_cold_start

Run:
    uv run python docs/impl/plot_perf_cat2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _perf_framework import PertSpec, run_perf_sweep  # noqa: E402

from genesis_robust_rl.perturbations.category_2_motor import (  # noqa: E402
    GyroscopicEffect,
    MotorBackEMF,
    MotorColdStart,
    MotorKill,
    MotorLag,
    MotorPartialFailure,
    MotorRPMNoise,
    MotorSaturation,
    MotorWear,
    PropellerThrustAsymmetry,
    RotorImbalance,
    ThrustCoefficientKF,
    TorqueCoefficientKM,
)

CAT = 2
DT = 0.005


def _simple(cls):
    def factory(scene, drone, n_envs: int):
        return cls(n_envs=n_envs, dt=DT)

    return factory


SPECS: list[PertSpec] = [
    # ExternalWrench → physics kind
    PertSpec("cat2_thrust_coeff_kf", "thrust_coeff_kf", "physics", _simple(ThrustCoefficientKF)),
    PertSpec("cat2_torque_coeff_km", "torque_coeff_km", "physics", _simple(TorqueCoefficientKM)),
    PertSpec(
        "cat2_propeller_thrust_asymmetry",
        "propeller_thrust_asymmetry",
        "physics",
        _simple(PropellerThrustAsymmetry),
    ),
    PertSpec(
        "cat2_motor_partial_failure",
        "motor_partial_failure",
        "physics",
        _simple(MotorPartialFailure),
    ),
    PertSpec("cat2_motor_back_emf", "motor_back_emf", "physics", _simple(MotorBackEMF)),
    PertSpec("cat2_gyroscopic_effect", "gyroscopic_effect", "physics", _simple(GyroscopicEffect)),
    # MotorCommand → motor kind
    PertSpec("cat2_motor_kill", "motor_kill", "motor", _simple(MotorKill)),
    PertSpec("cat2_motor_lag", "motor_lag", "motor", _simple(MotorLag)),
    PertSpec("cat2_motor_rpm_noise", "motor_rpm_noise", "motor", _simple(MotorRPMNoise)),
    PertSpec("cat2_motor_saturation", "motor_saturation", "motor", _simple(MotorSaturation)),
    PertSpec("cat2_motor_wear", "motor_wear", "motor", _simple(MotorWear)),
    PertSpec("cat2_rotor_imbalance", "rotor_imbalance", "motor", _simple(RotorImbalance)),
    PertSpec("cat2_motor_cold_start", "motor_cold_start", "motor", _simple(MotorColdStart)),
]


def main() -> None:
    run_perf_sweep(category=CAT, perturbations=SPECS)


if __name__ == "__main__":
    main()
