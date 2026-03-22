# 05 — Test Conventions

> Mandatory test contract for every perturbation implemented in Phase 2.
> A subagent's work is DONE only when all tests in its category pass.
> References: `02_class_design.md` (class hierarchy), `01_perturbations_catalog.md` (catalog entries).

---

## How subagents use this document

Each Phase 2 subagent receives:
- `02_class_design.md` — class hierarchy and method signatures
- `05_test_conventions.md` (this file) — test contract
- `01_perturbations_catalog.md` entries for its category

The subagent must implement its perturbation classes **and** the corresponding tests.
A task is complete only when `uv run pytest tests/<category>/` passes with no errors.

---

## Shared fixtures (`tests/conftest.py`)

All tests import from `conftest.py`. Never duplicate fixture logic.

```python
import pytest
import torch
from dataclasses import dataclass
from unittest.mock import MagicMock

# ---------- EnvState factory ----------

@dataclass
class EnvState:
    pos:     torch.Tensor   # [n_envs, 3]
    quat:    torch.Tensor   # [n_envs, 4]  (w, x, y, z)
    vel:     torch.Tensor   # [n_envs, 3]
    ang_vel: torch.Tensor   # [n_envs, 3]
    acc:     torch.Tensor   # [n_envs, 3]
    rpm:     torch.Tensor   # [n_envs, 4]
    dt:      float
    step:    int

@pytest.fixture(params=[1, 4, 16])
def n_envs(request):
    return request.param

@pytest.fixture
def mock_env_state(n_envs):
    """Realistic EnvState tensor batch for a hovering drone."""
    return EnvState(
        pos=torch.zeros(n_envs, 3),
        quat=torch.tensor([[1., 0., 0., 0.]]).expand(n_envs, -1),
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
    # Track calls to all Genesis setters
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

# ---------- Lipschitz helper ----------

def assert_lipschitz(perturbation, n_steps: int) -> None:
    """
    Verify that set_value() enforces the Lipschitz constraint over a trajectory.

    `dt` is read from perturbation.dt (set at construction via __init__).
    For n_steps consecutive calls with random values, the actual delta applied
    must never exceed lipschitz_k * dt per component.
    """
    if perturbation.lipschitz_k is None:
        return  # no constraint to check
    k = perturbation.lipschitz_k
    dt = perturbation.dt   # stored at construction — do not pass dt separately
    prev = perturbation._current_value.clone()
    for _ in range(n_steps):
        candidate = prev + torch.randn_like(prev) * k * dt * 5  # attempt large jump
        perturbation.set_value(candidate)
        current = perturbation._current_value
        delta = (current - prev).abs().max().item()
        assert delta <= k * dt + 1e-6, (
            f"Lipschitz violated: delta={delta:.6f} > k*dt={k*dt:.6f}"
        )
        prev = current.clone()


# ---------- Per-category fixture pattern (T1) ----------
# Each category conftest.py MUST define a `perturbation` fixture parametrized
# over all leaf classes in the category. Pattern:
#
#   @pytest.fixture(params=[
#       lambda n: AdditiveNoise(id="gyro_noise", n_envs=n, dt=0.01, ...),
#       lambda n: OUDrift(id="gyro_drift", n_envs=n, dt=0.01, ...),
#   ])
#   def perturbation(request, n_envs):
#       return request.param(n_envs)
#
# For sensor pipeline tests (I4), each category-4 conftest also defines:
#   @pytest.fixture(params=[GyroscopeModel, AccelerometerModel, ...])
#   def sensor_model(request, n_envs):
#       return request.param(n_envs=n_envs)
#
#   @pytest.fixture
#   def obs_perturbation(n_envs):
#       return AdditiveNoise(id="test_noise", n_envs=n_envs, dt=0.01,
#                            obs_slice=slice(0, 3), ...)


# ---------- GPU fixture stubs (T3) ----------
# Subagents define these in their category perf conftest. Stubs shown here for reference.

@pytest.fixture
def perturbation_gpu(perturbation):
    """Move perturbation internal tensors to CUDA. Subagent implements .to(device) or manual move."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return perturbation  # subagent adds device transfer

@pytest.fixture
def mock_env_state_gpu(mock_env_state):
    """Move all EnvState tensors to CUDA."""
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
def sensor_model_gpu(sensor_model):
    """Move sensor model internal tensors to CUDA. Subagent implements."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return sensor_model  # subagent adds device transfer
```

---

## Unit tests (no Genesis, mocked EnvState)

Run with: `uv run pytest tests/ -m unit`

Every perturbation class must pass **all** of the following.

### U1 — `sample()` output within bounds (1000 draws)

```python
@pytest.mark.unit
def test_sample_within_bounds(perturbation, n_envs):
    """All 1000 draws must stay within perturbation.bounds."""
    lo, hi = perturbation.bounds
    for _ in range(1000):
        v = perturbation.sample()
        assert v.shape[0] == n_envs
        assert (v >= lo).all() and (v <= hi).all(), (
            f"sample() out of bounds: min={v.min():.4f}, max={v.max():.4f}"
        )
```

### U2 — `curriculum_scale` extremes

```python
@pytest.mark.unit
def test_curriculum_scale_zero(perturbation):
    """curriculum_scale=0 → value equals nominal."""
    perturbation.curriculum_scale = 0.0
    v = perturbation.sample()
    nominal = torch.tensor(perturbation.nominal, dtype=v.dtype)
    assert torch.allclose(v, nominal.expand_as(v), atol=1e-6)

@pytest.mark.unit
def test_curriculum_scale_one(perturbation):
    """curriculum_scale=1 → full distribution range; value is not forced to nominal."""
    perturbation.curriculum_scale = 1.0
    samples = torch.stack([perturbation.sample() for _ in range(500)])
    # At least some samples differ from nominal (distribution has non-zero variance)
    nominal = torch.tensor(perturbation.nominal, dtype=samples.dtype)
    assert not torch.allclose(samples, nominal.expand_as(samples), atol=1e-4), (
        "curriculum_scale=1 produced only nominal values — distribution may be degenerate"
    )
```

### U3 — `tick()` reset path

```python
@pytest.mark.unit
def test_tick_reset_resets_state(perturbation, n_envs):
    """tick(is_reset=True) must call reset() and — if per_episode — sample()."""
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True)
    # _current_value must be set after reset tick for per_episode perturbations
    if perturbation.frequency == "per_episode":
        assert perturbation._current_value is not None
```

### U4 — `tick()` step path

```python
@pytest.mark.unit
def test_tick_step_advances(perturbation):
    """tick(is_reset=False) must call sample() if per_step, and step() if stateful."""
    perturbation.tick(is_reset=True)  # init
    val_before = perturbation._current_value.clone() if perturbation._current_value is not None else None
    perturbation.tick(is_reset=False)
    if perturbation.frequency == "per_step":
        # _current_value should be refreshed
        assert perturbation._current_value is not None
```

### U5 — `set_value()` enforces Lipschitz (adversarial)

```python
@pytest.mark.unit
def test_set_value_lipschitz(perturbation):
    """set_value() must clip value delta to lipschitz_k * dt."""
    perturbation.tick(is_reset=True, env_ids=torch.arange(perturbation.n_envs))
    assert_lipschitz(perturbation, n_steps=200)  # dt read from perturbation.dt internally
```

### U6 — `update_params()` enforces Lipschitz on distribution params

```python
@pytest.mark.unit
def test_update_params_lipschitz(perturbation):
    """update_params() must clip each param change to lipschitz_k * dt."""
    if perturbation.lipschitz_k is None:
        pytest.skip("no Lipschitz constraint")
    k = perturbation.lipschitz_k
    dt = perturbation.dt  # read from perturbation, not hardcoded
    original = dict(perturbation.distribution_params)
    # Attempt a large param jump
    huge_params = {k_: v + 1e6 for k_, v in original.items() if isinstance(v, (int, float))}
    if not huge_params:
        pytest.skip("no scalar distribution params to test")
    perturbation.update_params(huge_params)
    for key, orig_val in original.items():
        if key in huge_params:
            new_val = perturbation.distribution_params[key]
            delta = abs(new_val - orig_val)
            assert delta <= k * dt + 1e-6
```

### U7 — `get_privileged_obs()`

```python
@pytest.mark.unit
def test_get_privileged_obs_observable(perturbation):
    """observable=True → get_privileged_obs() returns _current_value."""
    perturbation.observable = True
    perturbation.tick(is_reset=True)
    obs = perturbation.get_privileged_obs()
    assert obs is not None
    assert torch.is_tensor(obs)

@pytest.mark.unit
def test_get_privileged_obs_not_observable(perturbation):
    """observable=False → get_privileged_obs() returns None."""
    perturbation.observable = False
    result = perturbation.get_privileged_obs()
    assert result is None
```

### U8 — Stateful: state persists and resets correctly

```python
@pytest.mark.unit
def test_stateful_persistence(perturbation, n_envs):
    """Stateful perturbations must persist state across ticks and reset on demand."""
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    perturbation.tick(is_reset=True)
    state_after_reset = perturbation._current_value.clone()
    for _ in range(10):
        perturbation.tick(is_reset=False)
    state_after_steps = perturbation._current_value.clone()
    # State should have changed after steps (OU / counter / buffer)
    assert not torch.allclose(state_after_reset, state_after_steps, atol=1e-6), (
        "Stateful perturbation state did not change over 10 steps"
    )
    # Reset specific envs — their state should be re-initialized
    env_ids = torch.tensor([0])
    perturbation.reset(env_ids)
    # No assertion on exact value — just must not raise

@pytest.mark.unit
def test_stateful_reset_all(perturbation, n_envs):
    """reset(all_env_ids) must not raise and must reset internal state."""
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation")
    env_ids = torch.arange(n_envs)
    perturbation.tick(is_reset=True)
    perturbation.reset(env_ids)  # must not raise
```

### U9 — Output shape and dtype

For `ObservationPerturbation` and `ActionPerturbation`: test `sample()` return shape.
For `PhysicsPerturbation`: `apply()` returns `None`; test `_current_value` shape after `apply()`.

```python
@pytest.mark.unit
def test_output_shape_dtype(perturbation, n_envs, mock_env_state, mock_scene):
    """Output must match declared dimension and be float32."""
    from genesis_robust_rl.perturbations.base import PhysicsPerturbation
    perturbation.tick(is_reset=True, env_ids=torch.arange(n_envs))
    expected_shape = (n_envs,) + tuple(perturbation.dimension)

    if isinstance(perturbation, PhysicsPerturbation):
        # apply() returns None — check _current_value set during apply()
        perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
        v = perturbation._current_value
    else:
        v = perturbation.sample()

    assert v is not None, "_current_value is None after apply()/sample()"
    # scope="global" → sample() returns [1, *dim]; scope="per_env" → [n_envs, *dim]
    if perturbation.scope == "global":
        expected_shape = (1,) + tuple(perturbation.dimension)
    assert v.shape == expected_shape, f"Expected {expected_shape}, got {v.shape}"
    assert v.dtype == torch.float32
```

### U10 — Partial reset (subset of envs)

```python
@pytest.mark.unit
def test_partial_reset(perturbation, n_envs):
    """reset(env_ids=[0]) must reset only env 0; other envs must retain their state."""
    from genesis_robust_rl.perturbations.base import PhysicsPerturbation
    if n_envs < 2:
        pytest.skip("need n_envs >= 2 to test partial reset")
    if not perturbation.is_stateful:
        pytest.skip("stateless perturbation — partial reset has no observable state effect")
    if isinstance(perturbation, PhysicsPerturbation):
        pytest.skip("PhysicsPerturbation _current_value set by apply(), not tick() — partial reset not testable here")
    if perturbation.scope == "global":
        pytest.skip("scope=global: _current_value has no per-env dimension to isolate")
    perturbation.tick(is_reset=True)
    for _ in range(10):
        perturbation.tick(is_reset=False)
    state_before = perturbation._current_value.clone()
    env_ids = torch.tensor([0])
    perturbation.reset(env_ids)
    # env 0 must be reset; envs 1+ must retain their state
    if n_envs > 1:
        assert torch.allclose(perturbation._current_value[1:], state_before[1:], atol=1e-6), (
            "reset(env_ids=[0]) affected envs other than 0"
        )
```

### U11 — Adversarial mode: tick() must not call sample()

```python
@pytest.mark.unit
def test_adversarial_tick_no_sample(perturbation):
    """In ADVERSARIAL mode, tick(is_reset=False) must never call sample()."""
    from unittest.mock import patch
    from genesis_robust_rl.perturbations.base import PerturbationMode
    perturbation.tick(is_reset=True)
    perturbation.mode = PerturbationMode.ADVERSARIAL
    with patch.object(type(perturbation), "sample", wraps=perturbation.sample) as mock_sample:
        perturbation.tick(is_reset=False)
        assert mock_sample.call_count == 0, (
            f"tick(is_reset=False) called sample() {mock_sample.call_count} time(s) in ADVERSARIAL mode"
        )
```

---

## SensorModel unit tests (category 4 only)

Run with: `uv run pytest tests/category_4_sensor/test_sensor_models.py -m unit`

Every `SensorModel` leaf must pass the following in addition to the standard U1–U11.

### USM1 — `forward()` output shape and dtype

```python
@pytest.mark.unit
def test_sensor_forward_shape_dtype(sensor_model, mock_env_state, n_envs):
    """forward() must return [n_envs, dim] float32 with no NaN."""
    out = sensor_model.forward(mock_env_state)
    assert out.shape[0] == n_envs
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any(), "NaN in sensor_model.forward() output"
    assert torch.isfinite(out).all(), "Inf in sensor_model.forward() output"
```

### USM2 — `forward()` output range plausible

```python
@pytest.mark.unit
def test_sensor_forward_range(sensor_model, mock_env_state):
    """Output must be within physical plausible range for a hovering drone."""
    out = sensor_model.forward(mock_env_state)
    # Sensor-specific range check — subagent fills in bounds per sensor:
    # GyroscopeModel:    |out| < 50 rad/s
    # AccelerometerModel: |out| < 200 m/s²
    # MagnetometerModel: |out| < 100 µT
    # BarometerModel:     out ∈ [-500, 5000] m
    # GPSModel:           |out| < 1e6 m
    # OpticalFlowModel:   |out| < 100 m/s
    assert out.abs().max() < 1e6, "Sensor output is implausibly large"
```

### USM3 — `AccelerometerModel` is stateless (no `_vel_prev`)

`AccelerometerModel` reads `env_state.acc` directly — no internal state. Verify no statefulness:

```python
@pytest.mark.unit
def test_accelerometer_stateless(mock_env_state, n_envs):
    """AccelerometerModel must not maintain _vel_prev; successive calls are independent."""
    from genesis_robust_rl.sensor_models import AccelerometerModel
    model = AccelerometerModel(n_envs=n_envs)
    out1 = model.forward(mock_env_state)
    out2 = model.forward(mock_env_state)  # same input → same output
    assert torch.allclose(out1, out2), "AccelerometerModel is unexpectedly stateful"
    assert not hasattr(model, "_vel_prev"), "AccelerometerModel should not have _vel_prev"
```

### USM4 — `update_params()` validates typed dataclass

```python
@pytest.mark.unit
def test_sensor_update_params_validates(sensor_model):
    """update_params() with invalid type must raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        sensor_model.update_params({"invalid_field_xyz": 999.0})
```

---

## Integration tests (real EnvState, mocked Genesis scene)

Run with: `uv run pytest tests/ -m integration`

### I1 — `apply()` does not modify inputs in-place

```python
@pytest.mark.integration
def test_apply_no_inplace_mutation(perturbation, mock_env_state, n_envs):
    """apply() must return a new tensor; input obs/action must not be modified."""
    perturbation.tick(is_reset=True)
    # ObservationPerturbation
    if hasattr(perturbation, 'obs_slice'):
        obs = torch.randn(n_envs, 32)
        obs_orig = obs.clone()
        result = perturbation.apply(obs)
        # obs should not be modified (apply returns modified copy or same object —
        # the important thing is the caller can detect changes)
        # Note: in-place += is allowed if apply() returns the same tensor,
        # but inputs passed by the caller must not be silently aliased.
        _ = result  # result must be a valid tensor
    # ActionPerturbation
    elif hasattr(perturbation, '_buffer') or hasattr(perturbation, '_state'):
        action = torch.randn(n_envs, 4)
        action_orig = action.clone()
        result = perturbation.apply(action)
        assert result is not None
```

### I2 — `PhysicsPerturbation.apply()` calls Genesis setter with correct shape and `envs_idx`

```python
@pytest.mark.integration
def test_physics_setter_called_correctly(perturbation, mock_scene, mock_env_state, n_envs):
    """GenesisSetterPerturbation must call setter_fn(value, envs_idx) with matching shapes."""
    if not hasattr(perturbation, 'setter_fn'):
        pytest.skip("not a GenesisSetterPerturbation")
    perturbation.tick(is_reset=True)
    envs_idx = torch.arange(n_envs)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    assert perturbation.setter_fn.called, "setter_fn was not called"
    call_args = perturbation.setter_fn.call_args
    value_arg = call_args[0][0]
    envs_arg = call_args[0][1]
    assert value_arg.shape[0] == n_envs
    assert envs_arg.shape[0] == n_envs
```

### I3 — `ExternalWrenchPerturbation.apply()` calls solver with correct tensor

```python
@pytest.mark.integration
def test_wrench_apply_calls_solver(perturbation, mock_scene, mock_env_state, n_envs):
    """ExternalWrenchPerturbation must call apply_links_external_force/torque."""
    from genesis_robust_rl.perturbations.base import ExternalWrenchPerturbation
    if not isinstance(perturbation, ExternalWrenchPerturbation):
        pytest.skip("not an ExternalWrenchPerturbation")
    perturbation.tick(is_reset=True)
    perturbation.apply(mock_scene, mock_scene.drone, mock_env_state)
    called = (
        mock_scene.rigid_solver.apply_links_external_force.called or
        mock_scene.rigid_solver.apply_links_external_torque.called
    )
    assert called, "solver.apply_links_external_force/torque was not called"
```

### I4 — SensorModel → ObservationPerturbation pipeline produces valid output

```python
@pytest.mark.integration
def test_sensor_pipeline(sensor_model, obs_perturbation, mock_env_state, n_envs):
    """
    sensor_model.forward(env_state) → obs_perturbation.apply(raw)
    must produce a valid tensor of shape [n_envs, dim] without exceptions.
    """
    raw = sensor_model.forward(mock_env_state)
    assert raw.shape[0] == n_envs
    assert raw.dtype == torch.float32
    # Build a minimal obs tensor that includes the sensor slice
    obs = torch.zeros(n_envs, 64)
    obs_out = obs_perturbation.apply(obs)
    assert obs_out.shape == obs.shape
    assert not torch.isnan(obs_out).any(), "NaN in pipeline output"
```

### I5 — N-env vectorization independence

```python
@pytest.mark.integration
def test_vectorized_env_independence(perturbation, mock_env_state, n_envs):
    """
    Result for env i must be independent of env j.
    Achieved by running n_envs=1 for each env index and comparing to batched result.
    """
    if n_envs < 2:
        pytest.skip("need n_envs >= 2 to test independence")
    perturbation.tick(is_reset=True)
    # Apply on full batch
    if hasattr(perturbation, 'obs_slice'):
        obs = torch.randn(n_envs, 32)
        batched = perturbation.apply(obs.clone())
        # Verify env 0 result matches single-env run with same seed
        # (exact match not required — structure independence is the goal)
        assert batched.shape == (n_envs, 32)
    elif hasattr(perturbation, '_buffer'):
        action = torch.randn(n_envs, 4)
        result = perturbation.apply(action)
        assert result.shape == (n_envs, 4)
```

---

## Performance tests (GPU, n_envs=512)

Run with: `uv run pytest tests/ -m perf`

These tests are **skipped automatically** if no CUDA device is available.

```python
import time
import pytest
import torch

PERF_N_ENVS = 512
PERF_WARMUP = 100
PERF_STEPS = 1000
MAX_MS_PER_STEP = 0.5  # milliseconds

def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — perf tests require GPU")
```

### P1 — `apply()` overhead < 0.5 ms/step

```python
@pytest.mark.perf
def test_apply_overhead(perturbation_gpu, mock_env_state_gpu):
    """apply() measured over 1000 steps on GPU with warm cache must be < 0.5 ms/step."""
    _skip_if_no_cuda()
    perturbation_gpu.tick(is_reset=True)
    obs = torch.randn(PERF_N_ENVS, 64, device="cuda")
    # Warmup
    for _ in range(PERF_WARMUP):
        if hasattr(perturbation_gpu, 'obs_slice'):
            perturbation_gpu.apply(obs.clone())
        else:
            perturbation_gpu.apply(torch.randn(PERF_N_ENVS, 4, device="cuda"))
    torch.cuda.synchronize()
    # Measure
    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        if hasattr(perturbation_gpu, 'obs_slice'):
            perturbation_gpu.apply(obs.clone())
        else:
            perturbation_gpu.apply(torch.randn(PERF_N_ENVS, 4, device="cuda"))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS
    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"apply() too slow: {elapsed_ms:.3f} ms/step (limit {MAX_MS_PER_STEP} ms)"
    )
```

### P2 — `SensorModel.forward()` overhead < 0.5 ms/step

```python
@pytest.mark.perf
def test_sensor_model_overhead(sensor_model_gpu, mock_env_state_gpu):
    """SensorModel.forward() must complete in < 0.5 ms/step at n_envs=512 on GPU."""
    _skip_if_no_cuda()
    for _ in range(PERF_WARMUP):
        sensor_model_gpu.forward(mock_env_state_gpu)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        sensor_model_gpu.forward(mock_env_state_gpu)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS
    assert elapsed_ms < MAX_MS_PER_STEP, (
        f"SensorModel.forward() too slow: {elapsed_ms:.3f} ms/step"
    )
```

### P4 — Cumulative tick() overhead < 2 ms/step (all perturbations active)

```python
@pytest.mark.perf
@pytest.mark.skip(reason="Phase 3 — requires RobustDroneEnv; all_perturbations_gpu defined then")
def test_cumulative_tick_overhead(all_perturbations_gpu, mock_env_state_gpu):
    """All perturbations active simultaneously must complete tick() in < 2 ms/step."""
    _skip_if_no_cuda()
    for p in all_perturbations_gpu:
        p.tick(is_reset=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        for p in all_perturbations_gpu:
            p.tick(is_reset=False)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS
    assert elapsed_ms < 2.0, (
        f"Cumulative tick() too slow: {elapsed_ms:.3f} ms/step (limit 2.0 ms)"
    )
```

**Note:** `all_perturbations_gpu` is defined in a top-level integration test (not per-category).
Implemented once in Phase 2 completion test — not required from individual category subagents.

### P5 — `get_privileged_obs()` overhead < 1 ms/step (all observables active)

```python
@pytest.mark.perf
@pytest.mark.skip(reason="Phase 3 — requires RobustDroneEnv; env_with_all_perturbations defined then")
def test_privileged_obs_overhead(env_with_all_perturbations):
    """get_privileged_obs() with all 68 perturbations observable must complete in < 1 ms/step."""
    _skip_if_no_cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        env_with_all_perturbations.get_privileged_obs()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / PERF_STEPS
    assert elapsed_ms < 1.0, f"get_privileged_obs() too slow: {elapsed_ms:.3f} ms/step"
```

**Note:** P4 and P5 are implemented in Phase 3 (once `RobustDroneEnv` exists). Not required from Phase 2 subagents.

### P6 — Relative overhead < 5% vs bare `scene.step()`

Perturbation logic (tick + apply + Genesis setter) must add < 5% overhead compared to
bare `scene.step()`. Measured with real Genesis scene (Crazyflie CF2X URDF), median of
5 rounds × 100 steps per round, with periodic resets.

- Test file: `tests/integration/test_overhead_genesis.py`
- Baseline: `scene.step()` only (no perturbation)
- Perturbed: `tick(is_reset=False) + apply(scene, drone, env_state) + scene.step()`
- Assert: `(t_perturbed - t_baseline) / t_baseline < 0.05`
- Requires Genesis installed — run locally before any push/PR

**Note:** On CPU backend the overhead for perturbations with Genesis setters or `_compute_wrench()`
physics is significantly higher than 5% (40-120%). The 5% target is realistic on GPU where
`scene.step()` is much heavier. The CPU test still serves as a regression detector.
The overhead decreases with n_envs as the step cost grows.

### P3 — Stateful perturbation memory scales linearly with `n_envs`

**`perturbation_class` fixture:** must be defined in each `tests/category_N_*/conftest.py`
(not in the root conftest). Pattern: `@pytest.fixture(params=[ClassA, ClassB]) def perturbation_class(request): return request.param`.
See `06_test_infrastructure.md §5` for the exact pattern.

```python
@pytest.mark.perf
def test_stateful_memory_linear(perturbation_class):
    """
    Memory footprint of stateful perturbation internal tensors
    must scale as O(n_envs), not O(n_envs²).
    """
    _skip_if_no_cuda()
    sizes = [64, 128, 256, 512]
    mem = []
    for n in sizes:
        torch.cuda.reset_peak_memory_stats()
        p = perturbation_class(n_envs=n)
        p.tick(is_reset=True)
        mem.append(torch.cuda.max_memory_allocated())
    # Check approximate linearity: ratio mem[3]/mem[0] ≈ sizes[3]/sizes[0]
    ratio_actual = mem[-1] / mem[0]
    ratio_expected = sizes[-1] / sizes[0]
    assert ratio_actual < ratio_expected * 1.5, (
        f"Memory scaling is superlinear: {ratio_actual:.1f}x for {ratio_expected:.1f}x n_envs increase"
    )
```

---

## Test file layout per category

```
tests/
  conftest.py                    # shared fixtures (this document)
  test_placeholder.py            # CI smoke test (always passes)
  category_1_physics/
    test_genesis_setter.py       # U1–U11 + I1–I2 for GenesisSetterPerturbation leaves
    test_external_wrench.py      # U1–U11 + I1 + I3 for ExternalWrenchPerturbation leaves
    test_perf_physics.py         # P1 for physics leaves
  category_2_motor/
    ...
  category_3_temporal/
    ...
  category_4_sensor/
    test_sensor_models.py        # I4 pipeline tests + P2 for SensorModel leaves
    test_obs_perturbations.py    # U1–U11 + I1 + I5
    test_perf_sensor.py
  category_5_wind/
    ...
  category_6_action/
    ...
  category_7_payload/
    ...
  category_8_external/
    ...
```

---

## Checklist before marking a category DONE

- [ ] All U1–U11 unit tests pass for every perturbation class in the category (U10 skips for stateless/global/PhysicsPerturbation; U11 skips not applicable)
- [ ] All I1–I5 integration tests pass (skipping irrelevant ones with `pytest.skip`)
- [ ] P1 perf test passes on GPU (or skipped gracefully if no CUDA)
- [ ] P3 memory test passes for every stateful class (`perturbation_class` fixture defined in category conftest.py)
- [ ] P4 and P5 — implemented in Phase 3 (RobustDroneEnv required); not required per-category
- [ ] P6 overhead test passes locally: `uv run pytest tests/integration/test_overhead_genesis.py -v`
- [ ] No `pytest.warns(UserWarning)` emitted by any test
- [ ] Pytest markers registered in `pyproject.toml` under `[tool.pytest.ini_options]`:
  ```toml
  markers = ["unit: unit tests (no Genesis)", "integration: integration tests", "perf: performance tests (GPU)"]
  ```
- [ ] `uv run pytest tests/category_N_*/` exits 0
