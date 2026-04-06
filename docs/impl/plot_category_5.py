"""Generate curriculum + per-env plots for category 5 — wind perturbations."""

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
# Helpers
# ---------------------------------------------------------------------------

from genesis_robust_rl.perturbations.base import EnvState  # noqa: E402


def _make_env_state(n_envs: int) -> EnvState:
    return EnvState(
        pos=torch.rand(n_envs, 3) * 2.0,
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 1.0,
        ang_vel=torch.randn(n_envs, 3) * 0.05,
        acc=torch.randn(n_envs, 3) * 0.2,
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=DT,
        step=0,
    )


# ---------------------------------------------------------------------------
# 5.1 ConstantWind
# ---------------------------------------------------------------------------


def plot_constant_wind():
    from genesis_robust_rl.perturbations.category_5_wind import ConstantWind

    print("5.1 ConstantWind")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = ConstantWind(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        mag = vals.norm(dim=-1).flatten()
        fig.add_trace(go.Histogram(x=mag.numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40))
    fig.update_layout(
        title="5.1 ConstantWind — Curriculum effect (wind magnitude)",
        xaxis_title="Wind magnitude [m/s]",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat5_constant_wind_curriculum.png")

    # Per-env
    p = ConstantWind(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    fig = go.Figure()
    v = p._current_value
    for axis, name in enumerate(["X", "Y", "Z"]):
        fig.add_trace(go.Bar(x=list(range(N_ENVS)), y=v[:, axis].numpy(), name=name))
    fig.update_layout(
        title="5.1 ConstantWind — Per-env wind velocity",
        xaxis_title="Env",
        yaxis_title="Wind [m/s]",
        barmode="group",
    )
    _save(fig, "cat5_constant_wind_per_env.png")


# ---------------------------------------------------------------------------
# 5.2 Turbulence
# ---------------------------------------------------------------------------


def plot_turbulence():
    from genesis_robust_rl.perturbations.category_5_wind import Turbulence

    print("5.2 Turbulence")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = Turbulence(n_envs=4, dt=DT)
        p.curriculum_scale = scale
        p.tick(is_reset=True)
        trace = []
        for _ in range(N_STEPS):
            p.tick(is_reset=False)
            trace.append(p._current_value[0, 0].item())
        fig.add_trace(go.Scatter(y=trace, name=f"scale={scale}", mode="lines"))
    fig.update_layout(
        title="5.2 Turbulence — Curriculum effect (env 0, X-axis)",
        xaxis_title="Step",
        yaxis_title="Wind velocity [m/s]",
    )
    _save(fig, "cat5_turbulence_curriculum.png")

    # Per-env
    p = Turbulence(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    traces = {i: [] for i in range(min(4, N_ENVS))}
    for _ in range(N_STEPS):
        p.tick(is_reset=False)
        for i in traces:
            traces[i].append(p._current_value[i, 0].item())
    fig = go.Figure()
    for i, t in traces.items():
        fig.add_trace(go.Scatter(y=t, name=f"env {i}", mode="lines"))
    fig.update_layout(
        title="5.2 Turbulence — Per-env diversity (X-axis)",
        xaxis_title="Step",
        yaxis_title="Wind velocity [m/s]",
    )
    _save(fig, "cat5_turbulence_per_env.png")


# ---------------------------------------------------------------------------
# 5.3 WindGust
# ---------------------------------------------------------------------------


def plot_wind_gust():
    from genesis_robust_rl.perturbations.category_5_wind import WindGust

    print("5.3 WindGust")

    # Curriculum
    fig = go.Figure()
    for scale in [0.25, 0.5, 1.0]:
        p = WindGust(
            n_envs=4,
            dt=DT,
            distribution_params={
                "prob_low": 0.05,
                "prob_high": 0.1,
                "mag_low": 2.0,
                "mag_high": 8.0,
                "duration_low": 3,
                "duration_high": 15,
            },
        )
        p.curriculum_scale = scale
        p.tick(is_reset=True)
        trace = []
        for _ in range(N_STEPS):
            p.tick(is_reset=False)
            trace.append(p._current_value[0].norm().item())
        fig.add_trace(go.Scatter(y=trace, name=f"scale={scale}", mode="lines"))
    fig.update_layout(
        title="5.3 WindGust — Curriculum effect (env 0, force magnitude)",
        xaxis_title="Step",
        yaxis_title="Force magnitude [N]",
    )
    _save(fig, "cat5_wind_gust_curriculum.png")

    # Per-env
    p = WindGust(
        n_envs=N_ENVS,
        dt=DT,
        distribution_params={
            "prob_low": 0.05,
            "prob_high": 0.15,
            "mag_low": 2.0,
            "mag_high": 8.0,
            "duration_low": 5,
            "duration_high": 15,
        },
    )
    p.tick(is_reset=True)
    traces = {i: [] for i in range(min(4, N_ENVS))}
    for _ in range(N_STEPS):
        p.tick(is_reset=False)
        for i in traces:
            traces[i].append(p._current_value[i].norm().item())
    fig = go.Figure()
    for i, t in traces.items():
        fig.add_trace(go.Scatter(y=t, name=f"env {i}", mode="lines"))
    fig.update_layout(
        title="5.3 WindGust — Per-env diversity (force magnitude)",
        xaxis_title="Step",
        yaxis_title="Force magnitude [N]",
    )
    _save(fig, "cat5_wind_gust_per_env.png")


# ---------------------------------------------------------------------------
# 5.4 WindShear
# ---------------------------------------------------------------------------


def plot_wind_shear():
    from genesis_robust_rl.perturbations.category_5_wind import WindShear

    print("5.4 WindShear")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = WindShear(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(x=vals.flatten().numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40)
        )
    fig.update_layout(
        title="5.4 WindShear — Curriculum effect (gradient distribution)",
        xaxis_title="Gradient [(m/s)/m]",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat5_wind_shear_curriculum.png")

    # Per-env
    p = WindShear(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    p.tick(is_reset=False)  # sample gradient
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(range(N_ENVS)), y=p._current_value.flatten().numpy(), name="gradient")
    )
    fig.update_layout(
        title="5.4 WindShear — Per-env gradient",
        xaxis_title="Env",
        yaxis_title="Gradient [(m/s)/m]",
    )
    _save(fig, "cat5_wind_shear_per_env.png")


# ---------------------------------------------------------------------------
# 5.5 AdversarialWind
# ---------------------------------------------------------------------------


def plot_adversarial_wind():
    from genesis_robust_rl.perturbations.category_5_wind import AdversarialWind

    print("5.5 AdversarialWind")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = AdversarialWind(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        mag = vals.norm(dim=-1).flatten()
        fig.add_trace(go.Histogram(x=mag.numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40))
    fig.update_layout(
        title="5.5 AdversarialWind — Curriculum effect (magnitude)",
        xaxis_title="Wind magnitude [m/s]",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat5_adversarial_wind_curriculum.png")

    # Lipschitz demo
    p = AdversarialWind(n_envs=1, dt=DT)
    p.tick(is_reset=True)
    trace = []
    for _ in range(N_STEPS):
        target = torch.randn(1, 3) * 10
        p.set_value(target)
        trace.append(p._current_value[0, 0].item())
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=trace, name="Lipschitz-constrained", mode="lines"))
    fig.update_layout(
        title="5.5 AdversarialWind — Lipschitz enforcement (X-axis)",
        xaxis_title="Step",
        yaxis_title="Wind velocity [m/s]",
    )
    _save(fig, "cat5_adversarial_wind_per_env.png")


# ---------------------------------------------------------------------------
# 5.6 BladeVortexInteraction
# ---------------------------------------------------------------------------


def plot_bvi():
    from genesis_robust_rl.perturbations.category_5_wind import BladeVortexInteraction

    print("5.6 BladeVortexInteraction")

    # Curriculum
    fig = go.Figure()
    for scale in [0.25, 0.5, 1.0]:
        p = BladeVortexInteraction(n_envs=4, dt=DT)
        p.curriculum_scale = scale
        p.tick(is_reset=True)
        trace = []
        for _ in range(N_STEPS):
            p.tick(is_reset=False)
            trace.append(p._current_value[0, 0].item())
        fig.add_trace(go.Scatter(y=trace, name=f"scale={scale}", mode="lines"))
    fig.update_layout(
        title="5.6 BVI — Curriculum effect (rotor 0, env 0)",
        xaxis_title="Step",
        yaxis_title="Thrust fraction",
    )
    _save(fig, "cat5_blade_vortex_curriculum.png")

    # Per-env
    p = BladeVortexInteraction(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    traces = {i: [] for i in range(min(4, N_ENVS))}
    for _ in range(N_STEPS):
        p.tick(is_reset=False)
        for i in traces:
            traces[i].append(p._current_value[i, 0].item())
    fig = go.Figure()
    for i, t in traces.items():
        fig.add_trace(go.Scatter(y=t, name=f"env {i}", mode="lines"))
    fig.update_layout(
        title="5.6 BVI — Per-env diversity (rotor 0)",
        xaxis_title="Step",
        yaxis_title="Thrust fraction",
    )
    _save(fig, "cat5_blade_vortex_per_env.png")


# ---------------------------------------------------------------------------
# 5.7 GroundEffectBoundary
# ---------------------------------------------------------------------------


def plot_ground_effect_boundary():
    from genesis_robust_rl.perturbations.category_5_wind import GroundEffectBoundary

    print("5.7 GroundEffectBoundary")

    # Altitude sweep
    fig = go.Figure()
    for diameter in [0.06, 0.092, 0.15]:
        p = GroundEffectBoundary(n_envs=100, dt=DT)
        p._current_value = torch.full((100,), diameter)
        altitudes = torch.linspace(0.01, 0.5, 100)
        env_state = _make_env_state(100)
        env_state.pos[:, 2] = altitudes
        wrench = p._compute_wrench(env_state)
        fig.add_trace(
            go.Scatter(
                x=altitudes.numpy(), y=wrench[:, 2].numpy(), name=f"D={diameter}m", mode="lines"
            )
        )
    fig.update_layout(
        title="5.7 GroundEffectBoundary — Force vs altitude",
        xaxis_title="Altitude [m]",
        yaxis_title="ΔF_z [N]",
    )
    _save(fig, "cat5_ground_effect_boundary_curriculum.png")

    # Per-env
    p = GroundEffectBoundary(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    p.tick(is_reset=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(range(N_ENVS)), y=p._current_value.flatten().numpy(), name="diameter")
    )
    fig.update_layout(
        title="5.7 GroundEffectBoundary — Per-env rotor diameter",
        xaxis_title="Env",
        yaxis_title="Diameter [m]",
    )
    _save(fig, "cat5_ground_effect_boundary_per_env.png")


# ---------------------------------------------------------------------------
# 5.8 PayloadSway
# ---------------------------------------------------------------------------


def plot_payload_sway():
    from genesis_robust_rl.perturbations.category_5_wind import PayloadSway

    print("5.8 PayloadSway")

    # Curriculum — pendulum angle over time
    fig = go.Figure()
    for scale in [0.25, 0.5, 1.0]:
        p = PayloadSway(n_envs=4, dt=DT)
        p.curriculum_scale = scale
        p.tick(is_reset=True)
        trace = []
        for _ in range(N_STEPS):
            p.tick(is_reset=False)
            trace.append(p._theta[0, 0].item())
        fig.add_trace(go.Scatter(y=trace, name=f"scale={scale}", mode="lines"))
    fig.update_layout(
        title="5.8 PayloadSway — Curriculum effect (θ_x, env 0)",
        xaxis_title="Step",
        yaxis_title="θ_x [rad]",
    )
    _save(fig, "cat5_payload_sway_curriculum.png")

    # Per-env
    p = PayloadSway(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    traces = {i: [] for i in range(min(4, N_ENVS))}
    for _ in range(N_STEPS):
        p.tick(is_reset=False)
        for i in traces:
            traces[i].append(p._theta[i, 0].item())
    fig = go.Figure()
    for i, t in traces.items():
        fig.add_trace(go.Scatter(y=t, name=f"env {i}", mode="lines"))
    fig.update_layout(
        title="5.8 PayloadSway — Per-env diversity (θ_x)",
        xaxis_title="Step",
        yaxis_title="θ_x [rad]",
    )
    _save(fig, "cat5_payload_sway_per_env.png")


# ---------------------------------------------------------------------------
# 5.9 ProximityDisturbance
# ---------------------------------------------------------------------------


def plot_proximity():
    from genesis_robust_rl.perturbations.category_5_wind import ProximityDisturbance

    print("5.9 ProximityDisturbance")

    # Force vs distance
    fig = go.Figure()
    for f_max in [0.1, 0.25, 0.5]:
        p = ProximityDisturbance(n_envs=100, dt=DT, surface_distance=2.0)
        p._current_value = torch.full((100,), f_max)
        env_state = _make_env_state(100)
        # Sweep X position from 1.7 to 2.0 (near +X wall at 2.0)
        env_state.pos[:, 0] = torch.linspace(1.5, 2.0, 100)
        env_state.pos[:, 1] = 0.0
        env_state.pos[:, 2] = 1.0  # far from ceiling
        wrench = p._compute_wrench(env_state)
        fig.add_trace(
            go.Scatter(
                x=(2.0 - env_state.pos[:, 0]).numpy(),
                y=wrench[:, 0].numpy(),
                name=f"F_max={f_max}N",
                mode="lines",
            )
        )
    fig.update_layout(
        title="5.9 ProximityDisturbance — Force vs distance to wall",
        xaxis_title="Distance to wall [m]",
        yaxis_title="F_x [N]",
    )
    _save(fig, "cat5_proximity_disturbance_curriculum.png")

    # Per-env
    p = ProximityDisturbance(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    p.tick(is_reset=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(N_ENVS)), y=p._current_value.flatten().numpy(), name="F_max"))
    fig.update_layout(
        title="5.9 ProximityDisturbance — Per-env F_max", xaxis_title="Env", yaxis_title="F_max [N]"
    )
    _save(fig, "cat5_proximity_disturbance_per_env.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_constant_wind()
    plot_turbulence()
    plot_wind_gust()
    plot_wind_shear()
    plot_adversarial_wind()
    plot_bvi()
    plot_ground_effect_boundary()
    plot_payload_sway()
    plot_proximity()
    print("\nDone — 18 plots generated.")
