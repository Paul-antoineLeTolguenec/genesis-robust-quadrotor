"""Generate curriculum + per-env plots for category 6 — action perturbations."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import plotly.io as pio
import torch

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

N_ENVS = 16
DT = 0.01
ACTION_DIM = 4


def _save(fig: go.Figure, name: str) -> None:
    path = os.path.join(ASSETS, name)
    pio.write_image(fig, path, width=800, height=450)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 6.1 ActionNoise
# ---------------------------------------------------------------------------


def plot_action_noise():
    from genesis_robust_rl.perturbations.category_6_action import ActionNoise

    print("6.1 ActionNoise")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = ActionNoise(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(x=vals.flatten().numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40)
        )
    fig.update_layout(
        title="6.1 ActionNoise — Curriculum (noise std distribution)",
        xaxis_title="σ",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat6_action_noise_curriculum.png")

    # Per-env effect
    p = ActionNoise(n_envs=N_ENVS, dt=DT)
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)
    p.tick(is_reset=False)
    action = torch.ones(N_ENVS, ACTION_DIM) * 0.5
    results = []
    for _ in range(100):
        p.tick(is_reset=False)
        results.append(p.apply(action)[:, 0].clone())
    data = torch.stack(results)
    fig = go.Figure()
    for i in range(min(4, N_ENVS)):
        fig.add_trace(go.Scatter(y=data[:, i].numpy(), name=f"env {i}", mode="lines"))
    fig.update_layout(
        title="6.1 ActionNoise — Per-env (action[0] over steps)",
        xaxis_title="Step",
        yaxis_title="Action value",
    )
    _save(fig, "cat6_action_noise_per_env.png")


# ---------------------------------------------------------------------------
# 6.2 ActionDeadzone
# ---------------------------------------------------------------------------


def plot_action_deadzone():
    from genesis_robust_rl.perturbations.category_6_action import ActionDeadzone

    print("6.2 ActionDeadzone")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = ActionDeadzone(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(x=vals.flatten().numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40)
        )
    fig.update_layout(
        title="6.2 ActionDeadzone — Curriculum (threshold distribution)",
        xaxis_title="Threshold",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat6_action_deadzone_curriculum.png")

    # Per-env
    p = ActionDeadzone(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(range(N_ENVS)), y=p._current_value.flatten().numpy(), name="threshold")
    )
    fig.update_layout(
        title="6.2 ActionDeadzone — Per-env threshold",
        xaxis_title="Env",
        yaxis_title="Threshold",
    )
    _save(fig, "cat6_action_deadzone_per_env.png")


# ---------------------------------------------------------------------------
# 6.3 ActionSaturation
# ---------------------------------------------------------------------------


def plot_action_saturation():
    from genesis_robust_rl.perturbations.category_6_action import ActionSaturation

    print("6.3 ActionSaturation")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = ActionSaturation(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(x=vals.flatten().numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40)
        )
    fig.update_layout(
        title="6.3 ActionSaturation — Curriculum (limit distribution)",
        xaxis_title="Limit",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat6_action_saturation_curriculum.png")

    # Per-env
    p = ActionSaturation(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(N_ENVS)), y=p._current_value.flatten().numpy(), name="limit"))
    fig.update_layout(
        title="6.3 ActionSaturation — Per-env limit",
        xaxis_title="Env",
        yaxis_title="Limit",
    )
    _save(fig, "cat6_action_saturation_per_env.png")


# ---------------------------------------------------------------------------
# 6.4 ActuatorHysteresis
# ---------------------------------------------------------------------------


def plot_hysteresis():
    from genesis_robust_rl.perturbations.category_6_action import ActuatorHysteresis

    print("6.4 ActuatorHysteresis")

    # Curriculum
    fig = go.Figure()
    for scale in [0.0, 0.25, 0.5, 1.0]:
        p = ActuatorHysteresis(n_envs=1000, dt=DT)
        p.curriculum_scale = scale
        vals = torch.stack([p.sample() for _ in range(100)])
        fig.add_trace(
            go.Histogram(x=vals.flatten().numpy(), name=f"scale={scale}", opacity=0.6, nbinsx=40)
        )
    fig.update_layout(
        title="6.4 Hysteresis — Curriculum (width distribution)",
        xaxis_title="Width",
        yaxis_title="Count",
        barmode="overlay",
    )
    _save(fig, "cat6_actuator_hysteresis_curriculum.png")

    # Per-env: sinusoidal input showing hysteresis effect
    import math

    p = ActuatorHysteresis(
        n_envs=1,
        dt=DT,
        distribution_params={"low": 0.02, "high": 0.02},
    )
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)
    steps = 200
    inputs, outputs = [], []
    for t in range(steps):
        a = torch.full((1, 4), math.sin(2 * math.pi * t / 50) * 0.5)
        out = p.apply(a)
        inputs.append(a[0, 0].item())
        outputs.append(out[0, 0].item())
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=inputs, name="input", mode="lines"))
    fig.add_trace(go.Scatter(y=outputs, name="output", mode="lines"))
    fig.update_layout(
        title="6.4 Hysteresis — Input vs output (sinusoidal)",
        xaxis_title="Step",
        yaxis_title="Action[0]",
    )
    _save(fig, "cat6_actuator_hysteresis_per_env.png")


# ---------------------------------------------------------------------------
# 6.5 ESCLowPassFilter
# ---------------------------------------------------------------------------


def plot_esc_filter():
    from genesis_robust_rl.perturbations.category_6_action import ESCLowPassFilter

    print("6.5 ESCLowPassFilter")

    # Curriculum: step response for different cutoff freqs
    fig = go.Figure()
    for fc in [5, 10, 20, 50]:
        p = ESCLowPassFilter(
            n_envs=1,
            dt=DT,
            distribution_params={"low": float(fc), "high": float(fc)},
        )
        p.curriculum_scale = 1.0
        p.tick(is_reset=True)
        action = torch.ones(1, 4)
        trace = []
        for _ in range(100):
            out = p.apply(action)
            trace.append(out[0, 0].item())
        fig.add_trace(go.Scatter(y=trace, name=f"fc={fc}Hz", mode="lines"))
    fig.update_layout(
        title="6.5 ESC Filter — Step response by cutoff frequency",
        xaxis_title="Step",
        yaxis_title="Filtered action[0]",
    )
    _save(fig, "cat6_esc_low_pass_filter_curriculum.png")

    # Per-env
    p = ESCLowPassFilter(n_envs=N_ENVS, dt=DT)
    p.tick(is_reset=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(range(N_ENVS)), y=p._current_value.flatten().numpy(), name="cutoff [Hz]")
    )
    fig.update_layout(
        title="6.5 ESC Filter — Per-env cutoff frequency",
        xaxis_title="Env",
        yaxis_title="Cutoff [Hz]",
    )
    _save(fig, "cat6_esc_low_pass_filter_per_env.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_action_noise()
    plot_action_deadzone()
    plot_action_saturation()
    plot_hysteresis()
    plot_esc_filter()
    print("\nDone — 10 plots generated.")
