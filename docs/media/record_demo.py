"""Record a Genesis Crazyflie CF2X hover demo, with or without perturbations.

Usage:
    uv run python docs/media/record_demo.py --scenario baseline
    uv run python docs/media/record_demo.py --scenario perturbed

Outputs an MP4 to docs/media/<scenario>.mp4. Convert to GIF with
``docs/media/mp4_to_gif.py``.

Genesis can only be initialized once per process, so each scenario runs in its
own Python invocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import genesis as gs
import torch

from genesis_robust_rl.perturbations.base import EnvState, Perturbation
from genesis_robust_rl.perturbations.category_5_wind import WindGust

HOVER_RPM = 14475.9  # Calibrated CF2X hover RPM (mass 0.027 kg, KF 3.16e-10)
N_STEPS = 240
DT = 0.01
FPS = 30
RESOLUTION = (640, 480)
CAMERA_POS = (0.5, -0.5, 0.65)
CAMERA_LOOKAT = (0.0, 0.0, 0.5)
CAMERA_FOV = 45


def build_scene() -> tuple[gs.Scene, Any, Any]:
    """Initialize Genesis and build a 1-env scene with ground, drone, and camera."""
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=DT),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            batch_dofs_info=True,
            batch_links_info=True,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.5)),
    )
    camera = scene.add_camera(
        res=RESOLUTION,
        pos=CAMERA_POS,
        lookat=CAMERA_LOOKAT,
        fov=CAMERA_FOV,
        GUI=False,
    )
    scene.build(n_envs=1)
    return scene, drone, camera


def make_env_state(drone: Any) -> EnvState:
    """Read drone state into an EnvState (n_envs=1)."""
    pos = drone.get_pos()
    quat = drone.get_quat()
    vel = drone.get_vel()
    ang_vel = drone.get_ang()
    return EnvState(
        pos=pos,
        quat=quat,
        vel=vel,
        ang_vel=ang_vel,
        acc=torch.zeros_like(vel),
        rpm=torch.full((1, 4), HOVER_RPM, dtype=torch.float32),
        dt=DT,
        step=0,
    )


def build_perturbations(scenario: str, drone: Any) -> list[Perturbation]:
    """Instantiate perturbations for the requested scenario."""
    if scenario == "baseline":
        return []
    return [
        WindGust(
            n_envs=1,
            dt=DT,
            link_idx=drone.base_link_idx,
            distribution_params={
                "prob_low": 0.08,
                "prob_high": 0.08,
                "mag_low": 0.004,
                "mag_high": 0.009,
                "duration_low": 25,
                "duration_high": 60,
            },
            bounds=(-0.015, 0.015),
        ),
    ]


def run_scenario(scenario: str, output_path: Path) -> None:
    """Run N_STEPS of hover simulation and record an MP4."""
    scene, drone, camera = build_scene()
    perturbations = build_perturbations(scenario, drone)

    env_ids = torch.arange(1)
    for p in perturbations:
        p.tick(is_reset=True, env_ids=env_ids)
        if p.frequency == "per_step":
            p.tick(is_reset=False)

    rpm = torch.full((1, 4), HOVER_RPM, dtype=torch.float32)
    camera.start_recording()
    for _ in range(N_STEPS):
        env_state = make_env_state(drone)
        drone.set_propellels_rpm(rpm)
        for p in perturbations:
            p.tick(is_reset=False)
            p.apply(scene, drone, env_state)
        scene.step()
        camera.render()
    camera.stop_recording(save_to_filename=str(output_path), fps=FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["baseline", "perturbed"],
        required=True,
        help="baseline: no perturbations; perturbed: wind_gust + mass_shift active",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory where the MP4 will be saved",
    )
    args = parser.parse_args()

    output_path = args.output_dir / f"{args.scenario}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_scenario(args.scenario, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
