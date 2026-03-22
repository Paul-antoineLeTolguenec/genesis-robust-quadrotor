"""Generate technical plots for category 1 (physics perturbations) documentation.

Outputs PNG files to docs/impl/assets/ via Plotly + kaleido.
Run: uv run python docs/impl/plot_category_1.py
"""
from pathlib import Path

import torch
import plotly.graph_objects as go
import plotly.io as pio

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1 — MassShift: curriculum_scale effect on sampled distribution
# ---------------------------------------------------------------------------


def plot_mass_shift_curriculum() -> None:
    """
    Show how curriculum_scale ∈ {0.0, 0.5, 1.0} compresses the sampling distribution
    of MassShift toward the nominal value (0.0 kg).

    Formula: value = nominal + (raw − nominal) × scale, clipped to bounds.
    With uniform(low=-0.05, high=0.1), nominal=0.0:
      scale=0.0 → all 0.0
      scale=0.5 → uniform(-0.025, 0.05)
      scale=1.0 → uniform(-0.05, 0.1)
    """
    n = 5000
    nominal = 0.0
    low, hi = -0.05, 0.1
    bounds = (-0.5, 1.0)
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    torch.manual_seed(0)
    raw = torch.empty(n).uniform_(low, hi).numpy()

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        values = (nominal + (raw - nominal) * scale)
        values = values.clip(bounds[0], bounds[1])
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=60,
                name=f"scale={scale}",
                opacity=0.7,
                marker_color=color,
            )
        )

    fig.update_layout(
        title="MassShift — curriculum_scale effect on sampled distribution",
        xaxis_title="Δm [kg]",
        yaxis_title="count",
        barmode="overlay",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="grey", annotation_text="nominal")

    pio.write_image(fig, ASSETS / "cat1_mass_shift_curriculum.png", scale=2)
    print("Saved: cat1_mass_shift_curriculum.png")


# ---------------------------------------------------------------------------
# Plot 2 — MassShift: per-env mass shift across episodes
# ---------------------------------------------------------------------------


def plot_mass_shift_per_env() -> None:
    """
    Simulate N episodes for 8 parallel envs — each episode resamples Δm independently.
    Illustrates the per-env diversity of domain randomization.
    """
    n_envs = 8
    n_episodes = 30
    low, hi = -0.05, 0.1

    torch.manual_seed(42)
    samples = torch.empty(n_envs, n_episodes).uniform_(low, hi).numpy()

    fig = go.Figure()
    for env_id in range(n_envs):
        fig.add_trace(
            go.Scatter(
                x=list(range(n_episodes)),
                y=samples[env_id],
                mode="lines+markers",
                name=f"env {env_id}",
                marker=dict(size=4),
                line=dict(width=1.5),
            )
        )

    fig.update_layout(
        title="MassShift — per-env sampled Δm across episodes",
        xaxis_title="episode",
        yaxis_title="Δm [kg]",
        legend_title="env",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="grey", annotation_text="nominal")

    pio.write_image(fig, ASSETS / "cat1_mass_shift_per_env.png", scale=2)
    print("Saved: cat1_mass_shift_per_env.png")


# ---------------------------------------------------------------------------
# Plot 3 — MassShift: tick() and apply() overhead vs n_envs (CPU)
# ---------------------------------------------------------------------------


def plot_mass_shift_perf() -> None:
    """Measure tick() and apply() latency across n_envs on CPU."""
    import time
    from unittest.mock import MagicMock
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from genesis_robust_rl.perturbations.category_1_physics import MassShift
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        setter = MagicMock()
        p = MassShift(setter_fn=setter, n_envs=n, dt=0.01)
        scene, drone = MagicMock(), MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title="MassShift — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )

    pio.write_image(fig, ASSETS / "cat1_mass_shift_perf.png", scale=2)
    print("Saved: cat1_mass_shift_perf.png")


# ---------------------------------------------------------------------------
# Plot 4 — COMShift: curriculum_scale effect on sampled distribution (x-axis)
# ---------------------------------------------------------------------------


def plot_com_shift_curriculum() -> None:
    """
    Show how curriculum_scale ∈ {0.0, 0.5, 1.0} compresses the per-axis
    sampling distribution of COMShift toward the nominal value (0.0 m).

    Formula: value = nominal + (raw − nominal) × scale, clipped to bounds.
    With uniform(low=-0.02, high=0.02), nominal=0.0 per axis:
      scale=0.0 → all 0.0
      scale=0.5 → uniform(-0.01, 0.01)
      scale=1.0 → uniform(-0.02, 0.02)
    """
    n = 5000
    nominal = 0.0
    low, hi = -0.02, 0.02
    bounds = (-0.05, 0.05)
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    torch.manual_seed(0)
    raw = torch.empty(n).uniform_(low, hi).numpy()

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        values = (nominal + (raw - nominal) * scale)
        values = values.clip(bounds[0], bounds[1])
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=60,
                name=f"scale={scale}",
                opacity=0.7,
                marker_color=color,
            )
        )

    fig.update_layout(
        title="COMShift — curriculum_scale effect on sampled distribution (per axis)",
        xaxis_title="Δr [m]",
        yaxis_title="count",
        barmode="overlay",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="grey", annotation_text="nominal")

    pio.write_image(fig, ASSETS / "cat1_com_shift_curriculum.png", scale=2)
    print("Saved: cat1_com_shift_curriculum.png")


# ---------------------------------------------------------------------------
# Plot 5 — COMShift: per-env CoM shift across episodes (3D trace)
# ---------------------------------------------------------------------------


def plot_com_shift_per_env() -> None:
    """
    Simulate N episodes for 6 parallel envs — each episode resamples Δr (3D) independently.
    Display Δx, Δy, Δz as stacked subplots to illustrate per-env diversity.
    """
    n_envs = 6
    n_episodes = 30
    low, hi = -0.02, 0.02
    axes = ["Δx", "Δy", "Δz"]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    torch.manual_seed(42)
    # [n_envs, n_episodes, 3]
    samples = torch.empty(n_envs, n_episodes, 3).uniform_(low, hi).numpy()

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=axes, vertical_spacing=0.08)

    for axis_idx in range(3):
        for env_id in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=samples[env_id, :, axis_idx],
                    mode="lines+markers",
                    name=f"env {env_id}" if axis_idx == 0 else None,
                    showlegend=(axis_idx == 0),
                    marker=dict(size=3, color=colors[env_id]),
                    line=dict(width=1.2, color=colors[env_id]),
                ),
                row=axis_idx + 1,
                col=1,
            )
        fig.add_hline(y=0.0, line_dash="dash", line_color="grey", row=axis_idx + 1, col=1)

    fig.update_layout(
        title="COMShift — per-env sampled Δr across episodes",
        xaxis3_title="episode",
        legend_title="env",
        template="plotly_white",
        width=700,
        height=600,
    )
    fig.update_yaxes(title_text="[m]")

    pio.write_image(fig, ASSETS / "cat1_com_shift_per_env.png", scale=2)
    print("Saved: cat1_com_shift_per_env.png")


# ---------------------------------------------------------------------------
# Plot 6 — COMShift: tick() and apply() overhead vs n_envs (CPU)
# ---------------------------------------------------------------------------


def plot_com_shift_perf() -> None:
    """Measure COMShift tick() and apply() latency across n_envs on CPU."""
    import time
    from unittest.mock import MagicMock
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from genesis_robust_rl.perturbations.category_1_physics import COMShift
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        setter = MagicMock()
        p = COMShift(setter_fn=setter, n_envs=n, dt=0.01)
        scene, drone = MagicMock(), MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title="COMShift — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )

    pio.write_image(fig, ASSETS / "cat1_com_shift_perf.png", scale=2)
    print("Saved: cat1_com_shift_perf.png")


# ---------------------------------------------------------------------------
# Plot 7 — InertiaTensor: curriculum_scale effect on sampled distribution (x-axis)
# ---------------------------------------------------------------------------


def plot_inertia_tensor_curriculum() -> None:
    """
    Show how curriculum_scale ∈ {0.0, 0.5, 1.0} compresses the per-axis
    sampling distribution of InertiaTensor scale toward the nominal value (1.0).

    Formula: value = nominal + (raw − nominal) × scale, clipped to bounds.
    With uniform(low=0.8, high=1.2), nominal=1.0 per axis:
      scale=0.0 → all 1.0 (identity — no perturbation)
      scale=0.5 → uniform(0.9, 1.1)
      scale=1.0 → uniform(0.8, 1.2)
    """
    n = 5000
    nominal = 1.0
    low, hi = 0.8, 1.2
    bounds = (0.5, 1.5)
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    torch.manual_seed(0)
    raw = torch.empty(n).uniform_(low, hi).numpy()

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        values = (nominal + (raw - nominal) * scale)
        values = values.clip(bounds[0], bounds[1])
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=60,
                name=f"scale={scale}",
                opacity=0.7,
                marker_color=color,
            )
        )

    fig.update_layout(
        title="InertiaTensor — curriculum_scale effect on inertia scale distribution (per axis)",
        xaxis_title="inertia scale factor s_j",
        yaxis_title="count",
        barmode="overlay",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")

    pio.write_image(fig, ASSETS / "cat1_inertia_tensor_curriculum.png", scale=2)
    print("Saved: cat1_inertia_tensor_curriculum.png")


# ---------------------------------------------------------------------------
# Plot 8 — InertiaTensor: per-env scale diversity across episodes (3 axes)
# ---------------------------------------------------------------------------


def plot_inertia_tensor_per_env() -> None:
    """
    Simulate N episodes for 6 parallel envs — each episode resamples the 3-axis
    inertia scale vector independently. Display s_x, s_y, s_z as stacked subplots.
    """
    n_envs = 6
    n_episodes = 30
    low, hi = 0.8, 1.2
    axes = ["s_x (roll)", "s_y (pitch)", "s_z (yaw)"]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    torch.manual_seed(42)
    # [n_envs, n_episodes, 3]
    samples = torch.empty(n_envs, n_episodes, 3).uniform_(low, hi).numpy()

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=axes, vertical_spacing=0.08)

    for axis_idx in range(3):
        for env_id in range(n_envs):
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=samples[env_id, :, axis_idx],
                    mode="lines+markers",
                    name=f"env {env_id}" if axis_idx == 0 else None,
                    showlegend=(axis_idx == 0),
                    marker=dict(size=3, color=colors[env_id]),
                    line=dict(width=1.2, color=colors[env_id]),
                ),
                row=axis_idx + 1,
                col=1,
            )
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=axis_idx + 1, col=1)

    fig.update_layout(
        title="InertiaTensor — per-env inertia scale across episodes",
        xaxis3_title="episode",
        legend_title="env",
        template="plotly_white",
        width=700,
        height=600,
    )
    fig.update_yaxes(title_text="scale")

    pio.write_image(fig, ASSETS / "cat1_inertia_tensor_per_env.png", scale=2)
    print("Saved: cat1_inertia_tensor_per_env.png")


# ---------------------------------------------------------------------------
# Plot 9 — InertiaTensor: tick() and apply() overhead vs n_envs (CPU)
# ---------------------------------------------------------------------------


def plot_inertia_tensor_perf() -> None:
    """Measure InertiaTensor tick() and apply() latency across n_envs on CPU."""
    import time
    from unittest.mock import MagicMock
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from genesis_robust_rl.perturbations.category_1_physics import InertiaTensor
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        p = InertiaTensor(
            mass_setter_fn=MagicMock(),
            com_setter_fn=MagicMock(),
            n_envs=n,
            dt=0.01,
        )
        scene, drone = MagicMock(), MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title="InertiaTensor — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )

    pio.write_image(fig, ASSETS / "cat1_inertia_tensor_perf.png", scale=2)
    print("Saved: cat1_inertia_tensor_perf.png")


# ---------------------------------------------------------------------------
# Plot 10 — MotorArmature: curriculum_scale effect on sampled distribution
# ---------------------------------------------------------------------------


def plot_motor_armature_curriculum() -> None:
    """
    Show how curriculum_scale ∈ {0.0, 0.5, 1.0} compresses the sampling distribution
    of MotorArmature toward the nominal value (1.0).

    Formula: value = nominal + (raw − nominal) × scale, clipped to bounds.
    With uniform(low=0.7, high=1.3), nominal=1.0:
      scale=0.0 → all 1.0 (no perturbation)
      scale=0.5 → uniform(0.85, 1.15)
      scale=1.0 → uniform(0.7, 1.3)
    """
    n = 5000
    nominal = 1.0
    low, hi = 0.7, 1.3
    bounds = (0.5, 1.5)
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    torch.manual_seed(0)
    raw = torch.empty(n).uniform_(low, hi).numpy()

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        values = (nominal + (raw - nominal) * scale)
        values = values.clip(bounds[0], bounds[1])
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=60,
                name=f"scale={scale}",
                opacity=0.7,
                marker_color=color,
            )
        )

    fig.update_layout(
        title="MotorArmature — curriculum_scale effect on armature ratio distribution",
        xaxis_title="armature ratio α",
        yaxis_title="count",
        barmode="overlay",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")

    pio.write_image(fig, ASSETS / "cat1_motor_armature_curriculum.png", scale=2)
    print("Saved: cat1_motor_armature_curriculum.png")


# ---------------------------------------------------------------------------
# Plot 11 — MotorArmature: per-env armature ratio across episodes
# ---------------------------------------------------------------------------


def plot_motor_armature_per_env() -> None:
    """
    Simulate N episodes for 8 parallel envs — each episode resamples α independently.
    Illustrates the per-env diversity of domain randomization.
    """
    n_envs = 8
    n_episodes = 30
    low, hi = 0.7, 1.3

    torch.manual_seed(42)
    samples = torch.empty(n_envs, n_episodes).uniform_(low, hi).numpy()

    fig = go.Figure()
    for env_id in range(n_envs):
        fig.add_trace(
            go.Scatter(
                x=list(range(n_episodes)),
                y=samples[env_id],
                mode="lines+markers",
                name=f"env {env_id}",
                marker=dict(size=4),
                line=dict(width=1.5),
            )
        )

    fig.update_layout(
        title="MotorArmature — per-env sampled armature ratio α across episodes",
        xaxis_title="episode",
        yaxis_title="α (armature ratio)",
        legend_title="env",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")

    pio.write_image(fig, ASSETS / "cat1_motor_armature_per_env.png", scale=2)
    print("Saved: cat1_motor_armature_per_env.png")


# ---------------------------------------------------------------------------
# Plot 12 — MotorArmature: tick() and apply() overhead vs n_envs (CPU)
# ---------------------------------------------------------------------------


def plot_motor_armature_perf() -> None:
    """Measure MotorArmature tick() and apply() latency across n_envs on CPU."""
    import time
    from unittest.mock import MagicMock
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from genesis_robust_rl.perturbations.category_1_physics import MotorArmature
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        setter = MagicMock()
        p = MotorArmature(setter_fn=setter, n_envs=n, dt=0.01)
        scene, drone = MagicMock(), MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title="MotorArmature — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )

    pio.write_image(fig, ASSETS / "cat1_motor_armature_perf.png", scale=2)
    print("Saved: cat1_motor_armature_perf.png")


# ---------------------------------------------------------------------------
# Plots 13–15 — FrictionRatio
# ---------------------------------------------------------------------------


def plot_friction_ratio_curriculum() -> None:
    """
    Show how curriculum_scale ∈ {0.0, 0.5, 1.0} compresses the sampling distribution
    of FrictionRatio toward the nominal value (1.0).

    Formula: value = nominal + (raw − nominal) × scale, clipped to bounds.
    With uniform(low=0.5, high=1.5), nominal=1.0:
      scale=0.0 → all 1.0 (no perturbation)
      scale=0.5 → uniform(0.75, 1.25)
      scale=1.0 → uniform(0.5, 1.5)
    """
    n = 5000
    nominal = 1.0
    low, hi = 0.5, 1.5
    bounds = (0.1, 3.0)
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    torch.manual_seed(0)
    raw = torch.empty(n).uniform_(low, hi).numpy()

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        values = (nominal + (raw - nominal) * scale)
        values = values.clip(bounds[0], bounds[1])
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=60,
                name=f"scale={scale}",
                opacity=0.7,
                marker_color=color,
            )
        )

    fig.update_layout(
        title="FrictionRatio — curriculum_scale effect on friction ratio distribution",
        xaxis_title="friction ratio r",
        yaxis_title="count",
        barmode="overlay",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")

    pio.write_image(fig, ASSETS / "cat1_friction_ratio_curriculum.png", scale=2)
    print("Saved: cat1_friction_ratio_curriculum.png")


def plot_friction_ratio_per_env() -> None:
    """
    Simulate N episodes for 8 parallel envs — each episode resamples r independently.
    Illustrates the per-env diversity of domain randomization.
    """
    n_envs = 8
    n_episodes = 30
    low, hi = 0.5, 1.5

    torch.manual_seed(42)
    samples = torch.empty(n_envs, n_episodes).uniform_(low, hi).numpy()

    fig = go.Figure()
    for env_id in range(n_envs):
        fig.add_trace(
            go.Scatter(
                x=list(range(n_episodes)),
                y=samples[env_id],
                mode="lines+markers",
                name=f"env {env_id}",
                marker=dict(size=4),
                line=dict(width=1.5),
            )
        )

    fig.update_layout(
        title="FrictionRatio — per-env sampled friction ratio across episodes",
        xaxis_title="episode",
        yaxis_title="friction ratio r",
        legend_title="env",
        template="plotly_white",
        width=700,
        height=420,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")

    pio.write_image(fig, ASSETS / "cat1_friction_ratio_per_env.png", scale=2)
    print("Saved: cat1_friction_ratio_per_env.png")


def plot_friction_ratio_perf() -> None:
    """Measure FrictionRatio tick() and apply() latency across n_envs on CPU."""
    import time
    from unittest.mock import MagicMock
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from genesis_robust_rl.perturbations.category_1_physics import FrictionRatio
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        setter = MagicMock()
        p = FrictionRatio(setter_fn=setter, n_envs=n, dt=0.01)
        env_state = EnvState(
            pos=torch.zeros(n, 3),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1),
            vel=torch.zeros(n, 3),
            ang_vel=torch.zeros(n, 3),
            acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000.0,
            dt=0.01,
            step=0,
        )
        scene = MagicMock()
        drone = MagicMock()
        drone.set_geoms_friction_ratio = setter

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.tick(is_reset=False)
            p.apply(scene, drone, env_state)

        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=False)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title="FrictionRatio — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )

    pio.write_image(fig, ASSETS / "cat1_friction_ratio_perf.png", scale=2)
    print("Saved: cat1_friction_ratio_perf.png")


def _plot_dof_gain_curriculum(name: str, slug: str, unit: str = "ratio") -> None:
    """Curriculum effect for DOF gain perturbations (1.6–1.9).

    Shows how curriculum_scale ∈ {0.0, 0.5, 1.0} compresses the multiplicative ratio
    distribution toward the nominal value (1.0).

    Formula: value = 1.0 + (raw − 1.0) × scale, clipped to bounds.
    With uniform(0.5, 2.0), nominal=1.0:
      scale=0.0 → all 1.0
      scale=0.5 → uniform(0.75, 1.5)
      scale=1.0 → uniform(0.5, 2.0)
    """
    n = 5000
    nominal = 1.0
    low, hi = 0.5, 2.0
    bounds = (0.5, 2.0)
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    torch.manual_seed(0)
    raw = torch.empty(n).uniform_(low, hi).numpy()

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        values = (nominal + (raw - nominal) * scale).clip(bounds[0], bounds[1])
        fig.add_trace(go.Histogram(
            x=values, nbinsx=60, name=f"scale={scale}",
            opacity=0.7, marker_color=color,
        ))

    fig.update_layout(
        title=f"{name} — curriculum_scale effect on sampled distribution",
        xaxis_title=f"{unit} (multiplicative ratio)",
        yaxis_title="count",
        barmode="overlay",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=700, height=420,
    )
    pio.write_image(fig, ASSETS / f"cat1_{slug}_curriculum.png", scale=2)
    print(f"Saved: cat1_{slug}_curriculum.png")


def _plot_dof_gain_per_env(name: str, slug: str) -> None:
    """Per-env DR diversity for DOF gain perturbations (1.6–1.9).

    Shows that each environment independently samples its ratio at curriculum_scale=1.0.
    """
    n_envs = 16
    low, hi = 0.5, 2.0
    nominal = 1.0

    torch.manual_seed(42)
    raw = torch.empty(n_envs).uniform_(low, hi)
    values = raw.numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(n_envs)),
        y=values,
        marker_color="#636EFA",
        name="sampled ratio",
    ))
    fig.add_hline(y=nominal, line_dash="dash", line_color="#EF553B",
                  annotation_text="nominal (1.0)", annotation_position="top right")

    fig.update_layout(
        title=f"{name} — per-env DR diversity (n_envs=16, curriculum_scale=1.0)",
        xaxis_title="env index",
        yaxis_title="multiplicative ratio",
        yaxis_range=[0.3, 2.2],
        template="plotly_white",
        width=700, height=420,
    )
    pio.write_image(fig, ASSETS / f"cat1_{slug}_per_env.png", scale=2)
    print(f"Saved: cat1_{slug}_per_env.png")


def _plot_dof_gain_perf(name: str, slug: str, setter_attr: str) -> None:
    """tick() and apply() CPU overhead vs n_envs for DOF gain perturbations (1.6–1.9)."""
    import time
    from unittest.mock import MagicMock
    from genesis_robust_rl.perturbations.category_1_physics import (
        PositionGainKp, VelocityGainKv, JointStiffness, JointDamping,
    )
    from tests.conftest import EnvState

    classes = {
        "position_gain_kp": PositionGainKp,
        "velocity_gain_kv": VelocityGainKv,
        "joint_stiffness": JointStiffness,
        "joint_damping": JointDamping,
    }
    cls = classes[slug]

    n_envs_range = [1, 4, 16, 64, 128, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        setter = MagicMock()
        p = cls(setter_fn=setter, n_envs=n, dt=0.01)
        env_state = EnvState(
            pos=torch.zeros(n, 3),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1),
            vel=torch.zeros(n, 3),
            ang_vel=torch.zeros(n, 3),
            acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000.0,
            dt=0.01,
            step=0,
        )
        scene = MagicMock()
        drone = MagicMock()

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title=f"{name} — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / f"cat1_{slug}_perf.png", scale=2)
    print(f"Saved: cat1_{slug}_perf.png")


# ---------------------------------------------------------------------------
# Plots for 1.10 AeroDragCoeff
# ---------------------------------------------------------------------------


def plot_aero_drag_curriculum() -> None:
    """curriculum_scale effect on sampled Cd multiplier distribution (axis 0)."""
    from genesis_robust_rl.perturbations.category_1_physics import AeroDragCoeff

    n_samples = 2000
    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    for scale, color in zip(scales, colors):
        p = AeroDragCoeff(n_envs=n_samples, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample()[:, 0].numpy()  # axis-0 multiplier
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=40, opacity=0.65,
            name=f"scale={scale}",
            marker_color=color,
        ))

    fig.update_layout(
        title="AeroDragCoeff — curriculum_scale effect on Cd multiplier (axis 0)",
        xaxis_title="Cd multiplier (dimensionless)",
        yaxis_title="count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_aero_drag_coeff_curriculum.png", scale=2)
    print("Saved: cat1_aero_drag_coeff_curriculum.png")


def plot_aero_drag_per_env() -> None:
    """Per-env DR diversity: sampled Cd multipliers across 32 envs, all 3 axes."""
    from genesis_robust_rl.perturbations.category_1_physics import AeroDragCoeff

    n_envs = 32
    p = AeroDragCoeff(n_envs=n_envs, dt=0.01)
    p.curriculum_scale = 1.0
    vals = p.sample().numpy()  # [n_envs, 3]

    fig = go.Figure()
    axis_colors = ["#636EFA", "#EF553B", "#00CC96"]
    axis_names = ["x", "y", "z"]
    for ax_idx, (color, ax_name) in enumerate(zip(axis_colors, axis_names)):
        fig.add_trace(go.Bar(
            x=list(range(n_envs)),
            y=vals[:, ax_idx].tolist(),
            name=f"Cd[{ax_name}]",
            marker_color=color,
            opacity=0.75,
        ))

    fig.update_layout(
        title="AeroDragCoeff — per-env Cd multiplier diversity (32 envs, 3 axes)",
        xaxis_title="env index",
        yaxis_title="Cd multiplier",
        barmode="group", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_aero_drag_coeff_per_env.png", scale=2)
    print("Saved: cat1_aero_drag_coeff_per_env.png")


def plot_aero_drag_perf() -> None:
    """tick() and _compute_wrench() CPU overhead vs n_envs for AeroDragCoeff."""
    import time
    from genesis_robust_rl.perturbations.category_1_physics import AeroDragCoeff
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 128, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        p = AeroDragCoeff(n_envs=n, dt=0.01)
        env_state = EnvState(
            pos=torch.zeros(n, 3),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1),
            vel=torch.randn(n, 3) * 2.0,
            ang_vel=torch.zeros(n, 3),
            acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000.0,
            dt=0.01,
            step=0,
        )
        from unittest.mock import MagicMock
        scene = MagicMock()
        scene.solver = MagicMock()
        scene.solver.apply_links_external_force = MagicMock()
        drone = MagicMock()

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
        marker=dict(size=6), line=dict(width=2, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
        marker=dict(size=6), line=dict(width=2, color="#EF553B"),
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(
        title="AeroDragCoeff — tick() and apply() overhead vs n_envs (CPU)",
        xaxis_title="n_envs", yaxis_title="latency [ms/step]",
        xaxis_type="log", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_aero_drag_coeff_perf.png", scale=2)
    print("Saved: cat1_aero_drag_coeff_perf.png")


# ---------------------------------------------------------------------------
# Plot — GroundEffect (1.11): curriculum_scale effect on k_ge vs altitude
# ---------------------------------------------------------------------------


def plot_ground_effect_curriculum() -> None:
    """Show how curriculum_scale compresses the Cheeseman-Bennett thrust augmentation.

    At scale=0.0: ΔF=0 everywhere (no ground effect applied).
    At scale=0.5: half the physical ground effect.
    At scale=1.0: full Cheeseman-Bennett correction.
    """
    import numpy as np

    R = 0.1  # rotor radius [m]
    nominal_thrust = 9.81 * 0.5  # N
    max_k_ge = 2.0
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    altitudes = np.linspace(R / 2.0 + 0.001, 4.0 * R, 200)

    fig = go.Figure()
    for scale, color in zip(scales, colors):
        delta_f = []
        for h in altitudes:
            ratio = R / (4.0 * h)
            k_ge = 1.0 / (1.0 - ratio ** 2)
            k_ge = min(k_ge, max_k_ge)
            df = (k_ge - 1.0) * nominal_thrust * scale
            delta_f.append(df)
        fig.add_trace(
            go.Scatter(
                x=altitudes,
                y=delta_f,
                mode="lines",
                name=f"scale={scale}",
                line=dict(color=color, width=2),
            )
        )

    fig.add_vline(x=R / 2.0, line_dash="dot", line_color="gray", annotation_text="h=R/2 (clamp)")
    fig.add_vline(x=4.0 * R, line_dash="dash", line_color="gray", annotation_text="h=4R (zone limit)")
    fig.update_layout(
        title="GroundEffect 1.11 — curriculum_scale effect on additional thrust",
        xaxis_title="altitude AGL [m]",
        yaxis_title="ΔF [N] (upward)",
        legend_title="curriculum_scale",
        template="plotly_white",
        width=800,
        height=450,
    )
    pio.write_image(fig, ASSETS / "cat1_ground_effect_curriculum.png", scale=2)
    print("Saved: cat1_ground_effect_curriculum.png")


# ---------------------------------------------------------------------------
# Plot — GroundEffect (1.11): per-env ΔF across a batch at varied altitudes
# ---------------------------------------------------------------------------


def plot_ground_effect_per_env() -> None:
    """Show ΔF per-env when each environment is at a different altitude.

    Simulates 16 environments with uniformly spaced altitudes in [0.05m, 0.50m],
    illustrating the natural per-env diversity arising from the physics model.
    """
    import numpy as np

    R = 0.1
    nominal_thrust = 9.81 * 0.5
    max_k_ge = 2.0
    n_envs = 16

    altitudes = np.linspace(R / 2.0 + 0.001, 0.50, n_envs)
    delta_f = []
    for h in altitudes:
        ratio = R / (4.0 * h)
        k_ge = 1.0 / (1.0 - ratio ** 2)
        k_ge = min(k_ge, max_k_ge)
        if h > 4.0 * R:
            k_ge = 1.0
        delta_f.append((k_ge - 1.0) * nominal_thrust)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"env {i:02d}\n(h={h:.2f}m)" for i, h in enumerate(altitudes)],
            y=delta_f,
            marker_color="#00CC96",
            name="ΔF [N]",
        )
    )
    fig.update_layout(
        title="GroundEffect 1.11 — per-env additional thrust ΔF at varied altitudes",
        xaxis_title="environment (altitude AGL)",
        yaxis_title="ΔF [N] (upward)",
        template="plotly_white",
        width=900,
        height=450,
    )
    pio.write_image(fig, ASSETS / "cat1_ground_effect_per_env.png", scale=2)
    print("Saved: cat1_ground_effect_per_env.png")


# ---------------------------------------------------------------------------
# Plot — GroundEffect (1.11): performance overhead vs n_envs
# ---------------------------------------------------------------------------


def plot_ground_effect_perf() -> None:
    """Measure _compute_wrench() overhead on CPU vs n_envs for GroundEffect.

    Uses torch-only vectorized ops; expected sub-millisecond at all sizes.
    """
    import time
    from unittest.mock import MagicMock
    from genesis_robust_rl.perturbations.category_1_physics import GroundEffect

    sizes = [1, 4, 16, 64, 128, 256, 512]
    tick_ms = []
    apply_ms = []
    n_warmup, n_steps = 50, 500

    for n in sizes:
        p = GroundEffect(n_envs=n, dt=0.01)
        p.tick(is_reset=True)

        pos = torch.zeros(n, 3)
        pos[:, 2] = 0.05  # close to ground
        from tests.conftest import EnvState as _EnvState
        env_state = _EnvState(
            pos=pos,
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1),
            vel=torch.zeros(n, 3),
            ang_vel=torch.zeros(n, 3),
            acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000.0,
            dt=0.01,
            step=0,
        )

        # Tick timing
        for _ in range(n_warmup):
            p.tick(is_reset=False)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            p.tick(is_reset=False)
        tick_ms.append((time.perf_counter() - t0) * 1000 / n_steps)

        # Apply timing (_compute_wrench only)
        for _ in range(n_warmup):
            p._compute_wrench(env_state)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            p._compute_wrench(env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / n_steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=tick_ms, mode="lines+markers", name="tick()",
        line=dict(color="#636EFA", width=2), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=apply_ms, mode="lines+markers", name="_compute_wrench()",
        line=dict(color="#EF553B", width=2), marker=dict(size=8),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="0.5 ms limit", annotation_position="top right")
    fig.update_layout(
        title="GroundEffect 1.11 — CPU overhead vs n_envs",
        xaxis_title="n_envs",
        yaxis_title="latency [ms/step]",
        xaxis_type="log",
        template="plotly_white",
        width=700,
        height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_ground_effect_perf.png", scale=2)
    print("Saved: cat1_ground_effect_perf.png")


# ---------------------------------------------------------------------------
# Plots for 1.12 ChassisGeometryAsymmetry
# ---------------------------------------------------------------------------


def plot_chassis_asymmetry_curriculum() -> None:
    """curriculum_scale effect on CoM offset distribution (x-axis)."""
    from genesis_robust_rl.perturbations.category_1_physics import ChassisGeometryAsymmetry
    from unittest.mock import MagicMock

    n_samples = 2000
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    fig = go.Figure()

    for scale, color in zip(scales, colors):
        p = ChassisGeometryAsymmetry(
            mass_setter_fn=MagicMock(),
            com_setter_fn=MagicMock(),
            n_envs=n_samples,
            dt=0.01,
        )
        p.curriculum_scale = scale
        vals = p.sample()[:, 0].numpy()  # x-offset
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=40, opacity=0.65,
            name=f"scale={scale}", marker_color=color,
        ))

    fig.update_layout(
        title="ChassisGeometryAsymmetry — curriculum_scale effect on CoM x-offset",
        xaxis_title="CoM x-offset [m]",
        yaxis_title="count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="grey", annotation_text="nominal")
    pio.write_image(fig, ASSETS / "cat1_chassis_asymmetry_curriculum.png", scale=2)
    print("Saved: cat1_chassis_asymmetry_curriculum.png")


def plot_chassis_asymmetry_per_env() -> None:
    """Per-env DR diversity: CoM offsets across 16 envs (x, y axes)."""
    from genesis_robust_rl.perturbations.category_1_physics import ChassisGeometryAsymmetry
    from unittest.mock import MagicMock

    n_envs = 16
    p = ChassisGeometryAsymmetry(
        mass_setter_fn=MagicMock(),
        com_setter_fn=MagicMock(),
        n_envs=n_envs,
        dt=0.01,
    )
    p.curriculum_scale = 1.0
    vals = p.sample().numpy()  # [n_envs, 4]: [dx, dy, dz, dm]

    fig = go.Figure()
    for ax_idx, (ax_name, color) in enumerate(zip(["dx", "dy", "dz", "dm"],
                                                   ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])):
        fig.add_trace(go.Bar(
            x=list(range(n_envs)), y=vals[:, ax_idx].tolist(),
            name=ax_name, marker_color=color, opacity=0.75,
        ))

    fig.update_layout(
        title="ChassisGeometryAsymmetry — per-env DR diversity (16 envs)",
        xaxis_title="env index",
        yaxis_title="value",
        barmode="group", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_chassis_asymmetry_per_env.png", scale=2)
    print("Saved: cat1_chassis_asymmetry_per_env.png")


def plot_chassis_asymmetry_perf() -> None:
    """tick() and apply() CPU overhead vs n_envs for ChassisGeometryAsymmetry."""
    import time
    from unittest.mock import MagicMock
    from genesis_robust_rl.perturbations.category_1_physics import ChassisGeometryAsymmetry
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 128, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        p = ChassisGeometryAsymmetry(
            mass_setter_fn=MagicMock(),
            com_setter_fn=MagicMock(),
            n_envs=n, dt=0.01,
        )
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )
        scene, drone = MagicMock(), MagicMock()

        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
                             line=dict(color="#636EFA", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
                             line=dict(color="#EF553B", width=2), marker=dict(size=6)))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(title="ChassisGeometryAsymmetry — overhead vs n_envs (CPU)",
                      xaxis_title="n_envs", yaxis_title="latency [ms/step]",
                      xaxis_type="log", template="plotly_white", width=700, height=420)
    pio.write_image(fig, ASSETS / "cat1_chassis_asymmetry_perf.png", scale=2)
    print("Saved: cat1_chassis_asymmetry_perf.png")


# ---------------------------------------------------------------------------
# Plots for 1.13 PropellerBladeDamage
# ---------------------------------------------------------------------------


def plot_blade_damage_curriculum() -> None:
    """curriculum_scale effect on propeller efficiency distribution (per blade)."""
    from genesis_robust_rl.perturbations.category_1_physics import PropellerBladeDamage

    n_samples = 2000
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    fig = go.Figure()

    for scale, color in zip(scales, colors):
        p = PropellerBladeDamage(n_envs=n_samples, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample()[:, 0].numpy()  # efficiency of blade 0
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=40, opacity=0.65,
            name=f"scale={scale}", marker_color=color,
        ))

    fig.update_layout(
        title="PropellerBladeDamage — curriculum_scale effect on blade efficiency",
        xaxis_title="efficiency η (dimensionless)",
        yaxis_title="count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")
    pio.write_image(fig, ASSETS / "cat1_blade_damage_curriculum.png", scale=2)
    print("Saved: cat1_blade_damage_curriculum.png")


def plot_blade_damage_per_env() -> None:
    """Per-env DR diversity: efficiency ratios across 8 envs, 4 blades."""
    from genesis_robust_rl.perturbations.category_1_physics import PropellerBladeDamage

    n_envs = 8
    p = PropellerBladeDamage(n_envs=n_envs, dt=0.01)
    p.curriculum_scale = 1.0
    vals = p.sample().numpy()  # [n_envs, 4]

    fig = go.Figure()
    blade_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for blade in range(4):
        fig.add_trace(go.Bar(
            x=list(range(n_envs)), y=vals[:, blade].tolist(),
            name=f"blade {blade}", marker_color=blade_colors[blade], opacity=0.8,
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")
    fig.update_layout(
        title="PropellerBladeDamage — per-env efficiency diversity (8 envs, 4 blades)",
        xaxis_title="env index",
        yaxis_title="efficiency η",
        barmode="group", template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_blade_damage_per_env.png", scale=2)
    print("Saved: cat1_blade_damage_per_env.png")


def plot_blade_damage_perf() -> None:
    """tick() and apply() CPU overhead vs n_envs for PropellerBladeDamage."""
    import time
    from unittest.mock import MagicMock
    from genesis_robust_rl.perturbations.category_1_physics import PropellerBladeDamage
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 128, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        p = PropellerBladeDamage(n_envs=n, dt=0.01)
        scene = MagicMock()
        scene.solver = MagicMock()
        scene.solver.apply_links_external_force = MagicMock()
        drone = MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
                             line=dict(color="#636EFA", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
                             line=dict(color="#EF553B", width=2), marker=dict(size=6)))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(title="PropellerBladeDamage — overhead vs n_envs (CPU)",
                      xaxis_title="n_envs", yaxis_title="latency [ms/step]",
                      xaxis_type="log", template="plotly_white", width=700, height=420)
    pio.write_image(fig, ASSETS / "cat1_blade_damage_perf.png", scale=2)
    print("Saved: cat1_blade_damage_perf.png")


# ---------------------------------------------------------------------------
# Plots for 1.14 StructuralFlexibility
# ---------------------------------------------------------------------------


def plot_structural_flexibility_curriculum() -> None:
    """curriculum_scale effect on spring stiffness k distribution."""
    from genesis_robust_rl.perturbations.category_1_physics import StructuralFlexibility

    n_samples = 2000
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    fig = go.Figure()

    for scale, color in zip(scales, colors):
        p = StructuralFlexibility(n_envs=n_samples, dt=0.01)
        p.curriculum_scale = scale
        vals = p.sample()[:, 0].numpy()  # k (spring stiffness)
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=40, opacity=0.65,
            name=f"scale={scale}", marker_color=color,
        ))

    fig.update_layout(
        title="StructuralFlexibility — curriculum_scale effect on spring stiffness k",
        xaxis_title="spring stiffness k [N·m/rad]",
        yaxis_title="count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="grey", annotation_text="nominal (0.0)")
    pio.write_image(fig, ASSETS / "cat1_structural_flexibility_curriculum.png", scale=2)
    print("Saved: cat1_structural_flexibility_curriculum.png")


def plot_structural_flexibility_per_env() -> None:
    """Per-env DR diversity: [k, b] pairs across 16 envs."""
    from genesis_robust_rl.perturbations.category_1_physics import StructuralFlexibility

    n_envs = 16
    p = StructuralFlexibility(n_envs=n_envs, dt=0.01)
    p.curriculum_scale = 1.0
    vals = p.sample().numpy()  # [n_envs, 2]

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["spring stiffness k", "damping coefficient b"],
                        vertical_spacing=0.1)

    fig.add_trace(go.Bar(x=list(range(n_envs)), y=vals[:, 0].tolist(),
                         name="k [N·m/rad]", marker_color="#636EFA"), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(n_envs)), y=vals[:, 1].tolist(),
                         name="b [N·m·s/rad]", marker_color="#EF553B"), row=2, col=1)

    fig.update_layout(
        title="StructuralFlexibility — per-env [k, b] diversity (16 envs)",
        template="plotly_white", width=700, height=500,
    )
    fig.update_xaxes(title_text="env index", row=2, col=1)
    pio.write_image(fig, ASSETS / "cat1_structural_flexibility_per_env.png", scale=2)
    print("Saved: cat1_structural_flexibility_per_env.png")


def plot_structural_flexibility_perf() -> None:
    """tick() and apply() CPU overhead vs n_envs for StructuralFlexibility."""
    import time
    from unittest.mock import MagicMock
    from genesis_robust_rl.perturbations.category_1_physics import StructuralFlexibility
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 128, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        p = StructuralFlexibility(n_envs=n, dt=0.01)
        scene = MagicMock()
        scene.solver = MagicMock()
        scene.solver.apply_links_external_torque = MagicMock()
        drone = MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.randn(n, 3) * 0.1, acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.tick(is_reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=True)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
                             line=dict(color="#636EFA", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
                             line=dict(color="#EF553B", width=2), marker=dict(size=6)))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(title="StructuralFlexibility — overhead vs n_envs (CPU)",
                      xaxis_title="n_envs", yaxis_title="latency [ms/step]",
                      xaxis_type="log", template="plotly_white", width=700, height=420)
    pio.write_image(fig, ASSETS / "cat1_structural_flexibility_perf.png", scale=2)
    print("Saved: cat1_structural_flexibility_perf.png")


# ---------------------------------------------------------------------------
# Plots for 1.15 BatteryVoltageSag
# ---------------------------------------------------------------------------


def plot_battery_voltage_sag_curriculum() -> None:
    """curriculum_scale effect on initial SoC distribution → voltage ratio."""
    from genesis_robust_rl.perturbations.category_1_physics import BatteryVoltageSag

    n_samples = 2000
    scales = [0.0, 0.5, 1.0]
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    fig = go.Figure()

    torch.manual_seed(0)
    for scale, color in zip(scales, colors):
        p = BatteryVoltageSag(n_envs=n_samples, dt=0.01)
        p.curriculum_scale = scale
        p.tick(is_reset=True)
        vals = p.sample().numpy()  # voltage_ratio in [0.7, 1.0]
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=40, opacity=0.65,
            name=f"scale={scale}", marker_color=color,
        ))

    fig.update_layout(
        title="BatteryVoltageSag — curriculum_scale effect on voltage ratio",
        xaxis_title="voltage ratio v_r (dimensionless)",
        yaxis_title="count",
        barmode="overlay", template="plotly_white", width=700, height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="nominal (1.0)")
    pio.write_image(fig, ASSETS / "cat1_battery_voltage_sag_curriculum.png", scale=2)
    print("Saved: cat1_battery_voltage_sag_curriculum.png")


def plot_battery_voltage_sag_per_env() -> None:
    """Per-env SoC depletion trajectories across 8 envs over 500 steps."""
    from genesis_robust_rl.perturbations.category_1_physics import BatteryVoltageSag

    n_envs = 8
    n_steps = 500
    p = BatteryVoltageSag(
        n_envs=n_envs, dt=0.01,
        distribution_params={
            "low": 0.5, "high": 1.0,
            "discharge_rate_low": 0.001, "discharge_rate_high": 0.01,
        },
    )
    p.curriculum_scale = 1.0
    p.tick(is_reset=True)

    trajectories = []
    for _ in range(n_steps):
        p.tick(is_reset=False)
        trajectories.append(p._soc.clone().numpy())

    import numpy as np
    traj = np.stack(trajectories, axis=1)  # [n_envs, n_steps]

    fig = go.Figure()
    for env_id in range(n_envs):
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)), y=traj[env_id].tolist(),
            mode="lines", name=f"env {env_id}", line=dict(width=1.5),
        ))

    fig.add_hline(y=0.0, line_dash="dash", line_color="grey", annotation_text="depleted")
    fig.update_layout(
        title="BatteryVoltageSag — per-env SoC depletion over 500 steps",
        xaxis_title="step",
        yaxis_title="state of charge (SoC)",
        yaxis_range=[-0.05, 1.05],
        legend_title="env",
        template="plotly_white", width=700, height=420,
    )
    pio.write_image(fig, ASSETS / "cat1_battery_voltage_sag_per_env.png", scale=2)
    print("Saved: cat1_battery_voltage_sag_per_env.png")


def plot_battery_voltage_sag_perf() -> None:
    """tick() and apply() CPU overhead vs n_envs for BatteryVoltageSag."""
    import time
    from unittest.mock import MagicMock
    from genesis_robust_rl.perturbations.category_1_physics import BatteryVoltageSag
    from tests.conftest import EnvState

    n_envs_range = [1, 4, 16, 64, 128, 256, 512]
    warmup, steps = 200, 2000
    tick_ms, apply_ms = [], []

    for n in n_envs_range:
        p = BatteryVoltageSag(n_envs=n, dt=0.01)
        scene = MagicMock()
        scene.solver = MagicMock()
        scene.solver.apply_links_external_force = MagicMock()
        drone = MagicMock()
        env_state = EnvState(
            pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0., 0., 0.]]).expand(n, -1),
            vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3), acc=torch.zeros(n, 3),
            rpm=torch.ones(n, 4) * 3000., dt=0.01, step=0,
        )

        p.tick(is_reset=True)
        for _ in range(warmup):
            p.tick(is_reset=False)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.tick(is_reset=False)
        tick_ms.append((time.perf_counter() - t0) * 1000 / steps)

        for _ in range(warmup):
            p.apply(scene, drone, env_state)
        t0 = time.perf_counter()
        for _ in range(steps):
            p.apply(scene, drone, env_state)
        apply_ms.append((time.perf_counter() - t0) * 1000 / steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_envs_range, y=tick_ms, mode="lines+markers", name="tick()",
                             line=dict(color="#636EFA", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=n_envs_range, y=apply_ms, mode="lines+markers", name="apply()",
                             line=dict(color="#EF553B", width=2), marker=dict(size=6)))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#636EFA",
                  annotation_text="tick() limit (0.1 ms)", annotation_position="top right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#EF553B",
                  annotation_text="apply() limit (0.05 ms)", annotation_position="bottom right")
    fig.update_layout(title="BatteryVoltageSag — overhead vs n_envs (CPU)",
                      xaxis_title="n_envs", yaxis_title="latency [ms/step]",
                      xaxis_type="log", template="plotly_white", width=700, height=420)
    pio.write_image(fig, ASSETS / "cat1_battery_voltage_sag_perf.png", scale=2)
    print("Saved: cat1_battery_voltage_sag_perf.png")


if __name__ == "__main__":
    plot_mass_shift_curriculum()
    plot_mass_shift_per_env()
    plot_mass_shift_perf()
    plot_com_shift_curriculum()
    plot_com_shift_per_env()
    plot_com_shift_perf()
    plot_inertia_tensor_curriculum()
    plot_inertia_tensor_per_env()
    plot_inertia_tensor_perf()
    plot_motor_armature_curriculum()
    plot_motor_armature_per_env()
    plot_motor_armature_perf()
    plot_friction_ratio_curriculum()
    plot_friction_ratio_per_env()
    plot_friction_ratio_perf()
    # 1.6 PositionGainKp
    _plot_dof_gain_curriculum("PositionGainKp", "position_gain_kp", unit="Kp ratio")
    _plot_dof_gain_per_env("PositionGainKp", "position_gain_kp")
    _plot_dof_gain_perf("PositionGainKp", "position_gain_kp", setter_attr="set_dofs_kp")
    # 1.7 VelocityGainKv
    _plot_dof_gain_curriculum("VelocityGainKv", "velocity_gain_kv", unit="Kv ratio")
    _plot_dof_gain_per_env("VelocityGainKv", "velocity_gain_kv")
    _plot_dof_gain_perf("VelocityGainKv", "velocity_gain_kv", setter_attr="set_dofs_kv")
    # 1.8 JointStiffness
    _plot_dof_gain_curriculum("JointStiffness", "joint_stiffness", unit="stiffness ratio")
    _plot_dof_gain_per_env("JointStiffness", "joint_stiffness")
    _plot_dof_gain_perf("JointStiffness", "joint_stiffness", setter_attr="set_dofs_stiffness")
    # 1.9 JointDamping
    _plot_dof_gain_curriculum("JointDamping", "joint_damping", unit="damping ratio")
    _plot_dof_gain_per_env("JointDamping", "joint_damping")
    _plot_dof_gain_perf("JointDamping", "joint_damping", setter_attr="set_dofs_damping")
    # 1.10 AeroDragCoeff
    plot_aero_drag_curriculum()
    plot_aero_drag_per_env()
    plot_aero_drag_perf()
    # 1.11 GroundEffect
    plot_ground_effect_curriculum()
    plot_ground_effect_per_env()
    plot_ground_effect_perf()
    # 1.12 ChassisGeometryAsymmetry
    plot_chassis_asymmetry_curriculum()
    plot_chassis_asymmetry_per_env()
    plot_chassis_asymmetry_perf()
    # 1.13 PropellerBladeDamage
    plot_blade_damage_curriculum()
    plot_blade_damage_per_env()
    plot_blade_damage_perf()
    # 1.14 StructuralFlexibility
    plot_structural_flexibility_curriculum()
    plot_structural_flexibility_per_env()
    plot_structural_flexibility_perf()
    # 1.15 BatteryVoltageSag
    plot_battery_voltage_sag_curriculum()
    plot_battery_voltage_sag_per_env()
    plot_battery_voltage_sag_perf()
    print("Done.")
