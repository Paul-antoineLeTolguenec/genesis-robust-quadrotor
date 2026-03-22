"""Shared fixtures for all test categories. Never modify this file in Phase 2+."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

torch.manual_seed(42)  # deterministic fixture data — prevents flaky tests

# ---------- EnvState ----------


@dataclass
class EnvState:
    pos: torch.Tensor  # [n_envs, 3]
    quat: torch.Tensor  # [n_envs, 4]  (w, x, y, z)
    vel: torch.Tensor  # [n_envs, 3]
    ang_vel: torch.Tensor  # [n_envs, 3]
    acc: torch.Tensor  # [n_envs, 3]
    rpm: torch.Tensor  # [n_envs, 4]
    dt: float
    step: int


@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param


@pytest.fixture
def mock_env_state(n_envs):
    """Realistic EnvState tensor batch for a hovering drone."""
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_envs, -1),
        vel=torch.randn(n_envs, 3) * 0.1,
        ang_vel=torch.randn(n_envs, 3) * 0.05,
        acc=torch.randn(n_envs, 3) * 0.2,
        rpm=torch.ones(n_envs, 4) * 3000.0,
        dt=0.01,
        step=0,
    )


# ---------- Scene / solver stub ----------


@pytest.fixture
def mock_scene():
    """Genesis scene stub with patched physics setters."""
    scene = MagicMock()
    scene.rigid_solver = MagicMock()
    scene.rigid_solver.apply_links_external_force = MagicMock()
    scene.rigid_solver.apply_links_external_torque = MagicMock()
    drone = MagicMock()
    drone.set_links_mass_shift = MagicMock()
    drone.set_links_COM_shift = MagicMock()
    drone.set_geoms_friction_ratio = MagicMock()
    drone.set_dofs_kp = MagicMock()
    drone.set_dofs_kv = MagicMock()
    drone.set_dofs_stiffness = MagicMock()
    drone.set_dofs_damping = MagicMock()
    scene.drone = drone
    return scene


# ---------- Lipschitz helper (not a fixture — import directly) ----------


def assert_lipschitz(perturbation, n_steps: int) -> None:
    """
    Verify that set_value() enforces the Lipschitz constraint over a trajectory.

    dt is read from perturbation.dt (set at construction).
    For n_steps consecutive calls with random values, the actual delta applied
    must never exceed lipschitz_k * dt per component.
    """
    if perturbation.lipschitz_k is None:
        return
    assert perturbation._current_value is not None, (
        "call tick(is_reset=True) before assert_lipschitz — _current_value is None"
    )
    k = perturbation.lipschitz_k
    dt = perturbation.dt
    prev = perturbation._current_value.clone()
    for _ in range(n_steps):
        candidate = prev + torch.randn_like(prev) * k * dt * 5
        perturbation.set_value(candidate)
        current = perturbation._current_value
        delta = (current - prev).abs().max().item()
        assert delta <= k * dt + 1e-6, f"Lipschitz violated: delta={delta:.6f} > k*dt={k * dt:.6f}"
        prev = current.clone()


# ---------- GPU fixtures ----------


@pytest.fixture
def mock_env_state_gpu(mock_env_state):
    """Move all EnvState tensors to CUDA. Skips if no CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return EnvState(
        pos=mock_env_state.pos.cuda(),
        quat=mock_env_state.quat.cuda(),
        vel=mock_env_state.vel.cuda(),
        ang_vel=mock_env_state.ang_vel.cuda(),
        acc=mock_env_state.acc.cuda(),
        rpm=mock_env_state.rpm.cuda(),
        dt=mock_env_state.dt,
        step=mock_env_state.step,
    )


@pytest.fixture
def perturbation_gpu(perturbation):
    """Move perturbation internal tensors to CUDA. Subagent implements .to(device).

    Requires: `perturbation` fixture defined in the category's conftest.py.
    This fixture is unusable at root scope — only works inside a category test directory.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return perturbation  # subagent adds device transfer


@pytest.fixture
def sensor_model_gpu(sensor_model):
    """Move sensor model internal tensors to CUDA. Subagent implements.

    Requires: `sensor_model` fixture defined in tests/category_4_sensor/conftest.py.
    This fixture is unusable at root scope — only works inside category_4_sensor/.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return sensor_model  # subagent adds device transfer
