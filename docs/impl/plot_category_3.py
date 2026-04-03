"""Generate technical plots for category 3 (temporal perturbations) documentation.

Outputs PNG files to docs/impl/assets/ via Plotly + kaleido.
Run: uv run python docs/impl/plot_category_3.py
"""

from pathlib import Path

import torch
import plotly.graph_objects as go
import plotly.io as pio

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]


# ---------------------------------------------------------------------------
# 3.1 — ObsFixedDelay: curriculum effect on delay distribution
# ---------------------------------------------------------------------------


def plot_obs_fixed_delay_curriculum() -> None:
    """Show how curriculum_scale compresses the delay distribution toward 0."""
    from genesis_robust_rl.perturbations.category_3_temporal import ObsFixedDelay

    n = 5000
    scales = [0.0, 0.5, 1.0]
    fig = go.Figure()
    torch.manual_seed(42)

    for scale, color in zip(scales, COLORS):
        p = ObsFixedDelay(obs_slice=slice(0, 3), obs_dim=3, n_envs=n, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample().squeeze(-1).numpy()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=12, name=f"scale={scale}",
            opacity=0.7, marker_color=color,
        ))

    fig.update_layout(
        title="ObsFixedDelay — curriculum_scale effect on delay distribution",
        xaxis_title="Delay (steps)", yaxis_title="Count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="nominal")
    pio.write_image(fig, ASSETS / "cat3_obs_fixed_delay_curriculum.png", scale=2)
    print("Saved: cat3_obs_fixed_delay_curriculum.png")


def plot_obs_fixed_delay_per_env() -> None:
    """Simulate episodes for 8 envs — each resamples delay independently."""
    from genesis_robust_rl.perturbations.category_3_temporal import ObsFixedDelay

    n_envs, n_episodes = 8, 30
    torch.manual_seed(42)
    p = ObsFixedDelay(obs_slice=slice(0, 3), obs_dim=3, n_envs=n_envs, dt=0.01)

    fig = go.Figure()
    for env_i in range(n_envs):
        delays = []
        for _ in range(n_episodes):
            p.tick(is_reset=True)
            delays.append(p._delay[env_i].item())
        fig.add_trace(go.Scatter(
            x=list(range(n_episodes)), y=delays,
            mode="lines+markers", name=f"env {env_i}",
            marker=dict(size=5), line=dict(width=1),
        ))

    fig.update_layout(
        title="ObsFixedDelay — per-env delay across episodes",
        xaxis_title="Episode", yaxis_title="Delay (steps)",
        template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_obs_fixed_delay_per_env.png", scale=2)
    print("Saved: cat3_obs_fixed_delay_per_env.png")


# ---------------------------------------------------------------------------
# 3.2 — ObsVariableDelay: per-step delay variation
# ---------------------------------------------------------------------------


def plot_obs_variable_delay_curriculum() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import ObsVariableDelay

    n = 5000
    scales = [0.0, 0.5, 1.0]
    fig = go.Figure()
    torch.manual_seed(42)

    for scale, color in zip(scales, COLORS):
        p = ObsVariableDelay(obs_slice=slice(0, 3), obs_dim=3, n_envs=n, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample().squeeze(-1).numpy()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=8, name=f"scale={scale}",
            opacity=0.7, marker_color=color,
        ))

    fig.update_layout(
        title="ObsVariableDelay — curriculum_scale effect",
        xaxis_title="Delay (steps)", yaxis_title="Count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_obs_variable_delay_curriculum.png", scale=2)
    print("Saved: cat3_obs_variable_delay_curriculum.png")


def plot_obs_variable_delay_per_env() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import ObsVariableDelay

    n_envs, n_steps = 4, 100
    torch.manual_seed(42)
    p = ObsVariableDelay(obs_slice=slice(0, 3), obs_dim=3, n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)

    fig = go.Figure()
    delays_per_env = {i: [] for i in range(n_envs)}
    for _ in range(n_steps):
        p.tick(is_reset=False)
        for i in range(n_envs):
            delays_per_env[i].append(p._delay[i].item())

    for i in range(n_envs):
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)), y=delays_per_env[i],
            mode="lines", name=f"env {i}", line=dict(width=1),
        ))

    fig.update_layout(
        title="ObsVariableDelay — per-step delay variation (4 envs)",
        xaxis_title="Step", yaxis_title="Delay (steps)",
        template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_obs_variable_delay_per_env.png", scale=2)
    print("Saved: cat3_obs_variable_delay_per_env.png")


# ---------------------------------------------------------------------------
# 3.3 — ActionFixedDelay: curriculum + per-env
# ---------------------------------------------------------------------------


def plot_action_fixed_delay_curriculum() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import ActionFixedDelay

    n = 5000
    scales = [0.0, 0.5, 1.0]
    fig = go.Figure()
    torch.manual_seed(42)

    for scale, color in zip(scales, COLORS):
        p = ActionFixedDelay(n_envs=n, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample().squeeze(-1).numpy()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=8, name=f"scale={scale}",
            opacity=0.7, marker_color=color,
        ))

    fig.update_layout(
        title="ActionFixedDelay — curriculum_scale effect",
        xaxis_title="Delay (steps)", yaxis_title="Count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_action_fixed_delay_curriculum.png", scale=2)
    print("Saved: cat3_action_fixed_delay_curriculum.png")


def plot_action_fixed_delay_per_env() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import ActionFixedDelay

    n_envs, n_episodes = 8, 30
    torch.manual_seed(42)
    p = ActionFixedDelay(n_envs=n_envs, dt=0.01)

    fig = go.Figure()
    for env_i in range(n_envs):
        delays = []
        for _ in range(n_episodes):
            p.tick(is_reset=True)
            delays.append(p._delay[env_i].item())
        fig.add_trace(go.Scatter(
            x=list(range(n_episodes)), y=delays,
            mode="lines+markers", name=f"env {env_i}",
            marker=dict(size=5), line=dict(width=1),
        ))

    fig.update_layout(
        title="ActionFixedDelay — per-env delay across episodes",
        xaxis_title="Episode", yaxis_title="Delay (steps)",
        template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_action_fixed_delay_per_env.png", scale=2)
    print("Saved: cat3_action_fixed_delay_per_env.png")


# ---------------------------------------------------------------------------
# 3.7 — PacketLoss: curriculum + per-env drop events
# ---------------------------------------------------------------------------


def plot_packet_loss_curriculum() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import PacketLoss

    n = 5000
    scales = [0.0, 0.5, 1.0]
    fig = go.Figure()
    torch.manual_seed(42)

    for scale, color in zip(scales, COLORS):
        p = PacketLoss(n_envs=n, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample().squeeze(-1).numpy()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=30, name=f"scale={scale}",
            opacity=0.7, marker_color=color,
        ))

    fig.update_layout(
        title="PacketLoss — curriculum_scale effect on drop probability",
        xaxis_title="Drop probability", yaxis_title="Count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_packet_loss_curriculum.png", scale=2)
    print("Saved: cat3_packet_loss_curriculum.png")


def plot_packet_loss_per_env() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import PacketLoss

    n_envs, n_steps = 4, 200
    torch.manual_seed(42)
    p = PacketLoss(n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)

    drops = {i: [] for i in range(n_envs)}
    for _ in range(n_steps):
        p.tick(is_reset=False)
        for i in range(n_envs):
            drops[i].append(p._drop_mask[i].item())

    fig = go.Figure()
    for i in range(n_envs):
        # Cumulative drop rate
        cumsum = [sum(drops[i][:t+1]) / (t+1) for t in range(n_steps)]
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)), y=cumsum,
            mode="lines", name=f"env {i}", line=dict(width=1.5),
        ))

    fig.update_layout(
        title="PacketLoss — cumulative drop rate per env over steps",
        xaxis_title="Step", yaxis_title="Cumulative drop rate",
        template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_packet_loss_per_env.png", scale=2)
    print("Saved: cat3_packet_loss_per_env.png")


# ---------------------------------------------------------------------------
# 3.8 — ComputationOverload: curriculum + stall events
# ---------------------------------------------------------------------------


def plot_computation_overload_curriculum() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import ComputationOverload

    n = 5000
    scales = [0.0, 0.5, 1.0]
    fig = go.Figure()
    torch.manual_seed(42)

    for scale, color in zip(scales, COLORS):
        p = ComputationOverload(n_envs=n, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample().squeeze(-1).numpy()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=30, name=f"scale={scale}",
            opacity=0.7, marker_color=color,
        ))

    fig.update_layout(
        title="ComputationOverload — curriculum_scale effect on skip probability",
        xaxis_title="Skip probability", yaxis_title="Count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_computation_overload_curriculum.png", scale=2)
    print("Saved: cat3_computation_overload_curriculum.png")


def plot_computation_overload_per_env() -> None:
    from genesis_robust_rl.perturbations.category_3_temporal import ComputationOverload

    n_envs, n_steps = 4, 300
    torch.manual_seed(42)
    p = ComputationOverload(n_envs=n_envs, dt=0.01)
    p.tick(is_reset=True)

    stalled = {i: [] for i in range(n_envs)}
    for _ in range(n_steps):
        p.tick(is_reset=False)
        for i in range(n_envs):
            stalled[i].append((p._skip_counter[i] > 0).item())

    fig = go.Figure()
    for i in range(n_envs):
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)), y=stalled[i],
            mode="lines", name=f"env {i}",
            line=dict(width=1), fill="tozeroy", opacity=0.5,
        ))

    fig.update_layout(
        title="ComputationOverload — stall events per env",
        xaxis_title="Step", yaxis_title="Stalled (1=yes, 0=no)",
        template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat3_computation_overload_per_env.png", scale=2)
    print("Saved: cat3_computation_overload_per_env.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    plot_obs_fixed_delay_curriculum()
    plot_obs_fixed_delay_per_env()
    plot_obs_variable_delay_curriculum()
    plot_obs_variable_delay_per_env()
    plot_action_fixed_delay_curriculum()
    plot_action_fixed_delay_per_env()
    plot_packet_loss_curriculum()
    plot_packet_loss_per_env()
    plot_computation_overload_curriculum()
    plot_computation_overload_per_env()
    print(f"\nAll plots saved to {ASSETS}/")
