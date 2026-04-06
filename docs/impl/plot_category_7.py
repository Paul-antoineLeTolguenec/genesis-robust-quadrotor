"""Generate curriculum + per-env plots for category 7 — payload & configuration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import plotly.graph_objects as go
import plotly.io as pio
import torch

from genesis_robust_rl.perturbations.base import EnvState

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

N_ENVS = 16
DT = 0.01


def _save(fig: go.Figure, name: str) -> None:
    path = os.path.join(ASSETS, name)
    pio.write_image(fig, path, width=800, height=450)
    print(f"  saved {path}")


def _make_scene() -> MagicMock:
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    drone = MagicMock()
    drone.set_links_mass_shift = MagicMock()
    drone.set_links_COM_shift = MagicMock()
    scene.drone = drone
    return scene


def _make_env_state(n_envs: int) -> EnvState:
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 2.0,
        ang_vel=torch.randn(n_envs, 3) * 0.1,
        acc=torch.randn(n_envs, 3) * 0.2,
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=DT,
        step=0,
    )


# ---------------------------------------------------------------------------
# 7.1 PayloadMass
# ---------------------------------------------------------------------------


def plot_payload_mass():
    from genesis_robust_rl.perturbations.category_7_payload import PayloadMass

    print("7.1 PayloadMass")
    scene = _make_scene()

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = PayloadMass(setter_fn=scene.drone.set_links_mass_shift, n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(x=vals.flatten().numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40)
        )
    fig.update_layout(
        title="7.1 PayloadMass — Curriculum (mass distribution)",
        xaxis_title="Δm (kg)",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat7_payload_mass_curriculum.png")

    # Per-env
    p = PayloadMass(setter_fn=scene.drone.set_links_mass_shift, n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    vals = p._current_value  # [n_envs]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"env {i}" for i in range(N_ENVS)], y=vals.numpy()))
    fig.update_layout(
        title="7.1 PayloadMass — Per-env diversity",
        xaxis_title="Environment",
        yaxis_title="Δm (kg)",
    )
    _save(fig, "cat7_payload_mass_per_env.png")


# ---------------------------------------------------------------------------
# 7.2 PayloadCOMOffset
# ---------------------------------------------------------------------------


def plot_payload_com_offset():
    from genesis_robust_rl.perturbations.category_7_payload import PayloadCOMOffset

    print("7.2 PayloadCOMOffset")
    scene = _make_scene()

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = PayloadCOMOffset(setter_fn=scene.drone.set_links_COM_shift, n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        for ax, name in enumerate(["X", "Y", "Z"]):
            if scale in (0.0, 1.0):
                fig.add_trace(
                    go.Histogram(
                        x=vals[:, :, ax].flatten().numpy(),
                        name=f"{name} scale={scale}",
                        opacity=0.5,
                        nbinsx=40,
                    )
                )
    fig.update_layout(
        title="7.2 PayloadCOMOffset — Curriculum (X/Y/Z at scale 0 and 1)",
        xaxis_title="Δr (m)",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat7_payload_com_offset_curriculum.png")

    # Per-env
    p = PayloadCOMOffset(setter_fn=scene.drone.set_links_COM_shift, n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    vals = p._current_value  # [n_envs, 3]
    fig = go.Figure()
    for ax, name in enumerate(["X", "Y", "Z"]):
        fig.add_trace(
            go.Bar(
                x=[f"env {i}" for i in range(N_ENVS)],
                y=vals[:, ax].numpy(),
                name=name,
            )
        )
    fig.update_layout(
        title="7.2 PayloadCOMOffset — Per-env diversity",
        xaxis_title="Environment",
        yaxis_title="Δr (m)",
        barmode="group",
    )
    _save(fig, "cat7_payload_com_offset_per_env.png")


# ---------------------------------------------------------------------------
# 7.3 AsymmetricPropGuardDrag
# ---------------------------------------------------------------------------


def plot_asymmetric_prop_guard_drag():
    from genesis_robust_rl.perturbations.category_7_payload import (
        AsymmetricPropGuardDrag,
    )

    print("7.3 AsymmetricPropGuardDrag")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = AsymmetricPropGuardDrag(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(
                x=vals[:, :, 0].flatten().numpy(),
                name=f"scale={scale}",
                opacity=0.6,
                nbinsx=40,
            )
        )
    fig.update_layout(
        title="7.3 AsymmetricPropGuardDrag — Curriculum (arm 0 ratio)",
        xaxis_title="Drag ratio",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat7_asymmetric_drag_curriculum.png")

    # Per-env
    p = AsymmetricPropGuardDrag(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    vals = p._current_value  # [n_envs, 4]
    fig = go.Figure()
    for arm in range(4):
        fig.add_trace(
            go.Bar(
                x=[f"env {i}" for i in range(N_ENVS)],
                y=vals[:, arm].numpy(),
                name=f"arm {arm}",
            )
        )
    fig.update_layout(
        title="7.3 AsymmetricPropGuardDrag — Per-env diversity",
        xaxis_title="Environment",
        yaxis_title="Drag ratio",
        barmode="group",
    )
    _save(fig, "cat7_asymmetric_drag_per_env.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_payload_mass()
    plot_payload_com_offset()
    plot_asymmetric_prop_guard_drag()
    print("\nAll Cat 7 plots generated.")
