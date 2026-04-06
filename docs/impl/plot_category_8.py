"""Generate curriculum + per-env plots for category 8 — external disturbances."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import plotly.io as pio
import torch

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

N_ENVS = 16
DT = 0.01
N_STEPS = 200


def _save(fig: go.Figure, name: str) -> None:
    path = os.path.join(ASSETS, name)
    pio.write_image(fig, path, width=800, height=450)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 8.1 BodyForceDisturbance
# ---------------------------------------------------------------------------


def plot_body_force_curriculum():
    """Curriculum effect on constant-mode body force distribution."""
    from genesis_robust_rl.perturbations.category_8_external import (
        BodyForceDisturbance,
    )

    print("8.1 BodyForceDisturbance — Curriculum")
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = BodyForceDisturbance(n_envs=1000, dt=DT)
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
        title="8.1 BodyForceDisturbance — Curriculum (X-axis)",
        xaxis_title="Force (N)",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat8_body_force_curriculum.png")


def plot_body_force_per_env():
    """Per-env diversity: OU-mode force trajectories."""
    from genesis_robust_rl.perturbations.category_8_external import (
        BodyForceDisturbance,
    )

    print("8.1 BodyForceDisturbance — Per-env (OU)")
    p = BodyForceDisturbance(n_envs=N_ENVS, dt=DT, distribution="ou_process")
    p.tick(is_reset=True, env_ids=torch.arange(N_ENVS))

    # Collect trajectories
    traj = torch.zeros(N_STEPS, N_ENVS, 3)
    for t in range(N_STEPS):
        p.tick(is_reset=False)
        traj[t] = p._current_value.clone()

    fig = go.Figure()
    for i in range(min(N_ENVS, 8)):
        fig.add_trace(
            go.Scatter(
                y=traj[:, i, 0].numpy(),
                mode="lines",
                name=f"env {i} (X)",
                opacity=0.7,
            )
        )
    fig.update_layout(
        title="8.1 BodyForceDisturbance — Per-env OU trajectories (X-axis)",
        xaxis_title="Step",
        yaxis_title="Force (N)",
    )
    _save(fig, "cat8_body_force_per_env.png")


# ---------------------------------------------------------------------------
# 8.2 BodyTorqueDisturbance
# ---------------------------------------------------------------------------


def plot_body_torque_curriculum():
    """Curriculum effect on constant-mode body torque distribution."""
    from genesis_robust_rl.perturbations.category_8_external import (
        BodyTorqueDisturbance,
    )

    print("8.2 BodyTorqueDisturbance — Curriculum")
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = BodyTorqueDisturbance(n_envs=1000, dt=DT)
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
        title="8.2 BodyTorqueDisturbance — Curriculum (X-axis)",
        xaxis_title="Torque (N·m)",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat8_body_torque_curriculum.png")


def plot_body_torque_per_env():
    """Per-env diversity: OU-mode torque trajectories."""
    from genesis_robust_rl.perturbations.category_8_external import (
        BodyTorqueDisturbance,
    )

    print("8.2 BodyTorqueDisturbance — Per-env (OU)")
    p = BodyTorqueDisturbance(n_envs=N_ENVS, dt=DT, distribution="ou_process")
    p.tick(is_reset=True, env_ids=torch.arange(N_ENVS))

    traj = torch.zeros(N_STEPS, N_ENVS, 3)
    for t in range(N_STEPS):
        p.tick(is_reset=False)
        traj[t] = p._current_value.clone()

    fig = go.Figure()
    for i in range(min(N_ENVS, 8)):
        fig.add_trace(
            go.Scatter(
                y=traj[:, i, 0].numpy(),
                mode="lines",
                name=f"env {i} (X)",
                opacity=0.7,
            )
        )
    fig.update_layout(
        title="8.2 BodyTorqueDisturbance — Per-env OU trajectories (X-axis)",
        xaxis_title="Step",
        yaxis_title="Torque (N·m)",
    )
    _save(fig, "cat8_body_torque_per_env.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_body_force_curriculum()
    plot_body_force_per_env()
    plot_body_torque_curriculum()
    plot_body_torque_per_env()
    print("\nAll Cat 8 plots generated.")
