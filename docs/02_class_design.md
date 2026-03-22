# 02 — Class Design

> Validated class hierarchy for `genesis-robust-rl`.
> No code is written before this document is approved.
> References: `00_feasibility.md`, `00b_sensor_models.md`, `01_perturbations_catalog.md`.

---

## Design principles

- **One intermediate node = one shared implementation pattern** (N≥3 classes share identical code)
- **No stateful intermediate class** — `step()` is a no-op in the base; stateful leaves override it
- **Two independent hierarchies** — `SensorModel` and `Perturbation` are decoupled; the env composes them
- **No fusion assumption** — `SensorModel` outputs raw readings; the user's pipeline decides what to do with them
- **DR vs adversarial separation** — `sample()` for domain randomization, `set_value()` for adversarial mode
- **Lipschitz enforced at write time** — `update_params()` clips param changes; `set_value()` clips value changes

---

## Utility components (composition, not inheritance)

Used by multiple leaves. Not part of either hierarchy.

### `OUProcess`

Ornstein-Uhlenbeck process state for a batch of environments.

```
OUProcess
  state: Tensor[n_envs, dim]
  reset(env_ids: Tensor)
  step(theta: float, sigma: float, mu: float, dt: float) → None   # dt passed per call

**Implementation constraint:** all ops inside `step()` must use `torch` exclusively (no `numpy`,
no Python loops over envs). Any numpy call forces a CPU/GPU sync at every RL step, destroying
throughput at n_envs=512. Use `torch.randn_like(self.state)` for the Wiener increment.
```

`dt` is passed per call to `step()` — not stored at construction. Callers (e.g. `OUDrift.step()`)
read `self.dt` from the owning `Perturbation` and pass it through.

Used by: `OUDrift`, stateful `ExternalWrenchPerturbation` leaves (turbulence, blade vortex…).

### `DelayBuffer`

Circular buffer storing the last `max_delay` steps for a batch of environments.
`delay` is per-env and integer-valued; clamped to `[0, max_delay - 1]` at read time.

```
DelayBuffer
  buffer: Tensor[n_envs, max_delay, dim]
  write_ptr: Tensor[n_envs]   # current write position, incremented modulo max_delay
  reset(env_ids: Tensor)
  push(x: Tensor[n_envs, dim]) → None
  read(delay: Tensor[n_envs]) → Tensor[n_envs, dim]   # delay in steps, integer, clamped to [0, max_delay-1]

**Circular index formula (canonical):**
```
push(): buffer[env, write_ptr[env], :] = x[env]; write_ptr[env] = (write_ptr[env] + 1) % max_delay
read(): idx = (write_ptr[env] - delay[env] - 1) % max_delay; return buffer[env, idx, :]
```
At delay=0: returns the just-pushed value (pass-through). At delay=D: returns obs from D steps ago.
Implemented as fully vectorized torch ops — no Python loops over env dimension.
```

**Cold start:** buffer is zero-initialized. Reads before the first full fill (fewer than `max_delay`
pushes) return zeros — equivalent to zero-order hold from initialization. This is acceptable for
training; policies learn to handle initial transients.

Used by: `ObsDelayBuffer`, `ActionDelayBuffer`.

---

## Hierarchy 1 — SensorModel

Converts the true Genesis state into a raw sensor reading.
The env calls `forward()` after each `scene.step()` and exposes the result as part of the observation.

**Sensor perturbation pipeline (mandatory order):**
```
env_state (Genesis true state)
  → sensor_model.forward(env_state)     # raw reading, shape [n_envs, dim]
  → obs_perturbation.apply(raw)         # corrupts the raw output
  → policy observation
```

Perturbations that modify sensor *characteristics* (e.g. M_soft, b_hard in magnetometer)
do so by calling `sensor_model.update_params()` — not by subclassing `SensorModel`.

```
SensorModel (ABC)
  __init__(n_envs: int)
  forward(env_state: EnvState) → Tensor[n_envs, dim]
  update_params(new_params: dict) → None   # validates and updates internal params
  reset(env_ids: Tensor) → None            # no-op by default; stateful leaves override
```

Each leaf defines its own typed params as a dataclass (not a raw dict). `update_params()`
validates shapes and types before applying.

**`update_params()` polymorphism:** the base signature accepts `dict`; each leaf casts
to its typed dataclass and validates before applying. Type errors raise `ValueError`.

**Lipschitz clip norm for `update_params()`:** element-wise L∞ — each scalar and each
element of a vector/matrix is clipped independently by `±lipschitz_k × dt`.
This applies uniformly to scalars, `vector(n)`, and `matrix(n,m)` params (e.g. `C_misalign`, `M_soft`).

### Leaves

| Class | Output shape | Params dataclass | Catalog refs |
|---|---|---|---|
| `GyroscopeModel` | `[n_envs, 3]` rad/s | `GyroParams` | 4.1, 4.2, 4.3, 4.5 |
| `AccelerometerModel` | `[n_envs, 3]` m/s² | `AccelParams` | 4.4, 4.5, 4.15 |
| `MagnetometerModel` | `[n_envs, 3]` µT | `MagParams` | 4.12 |
| `BarometerModel` | `[n_envs, 1]` m | `BaroParams` | 4.13 |
| `GPSModel` | `[n_envs, 3]` m | `GPSParams` | 4.6, 4.7, 4.8 |
| `OpticalFlowModel` | `[n_envs, 2]` m/s | `FlowParams` | 4.14 |

`AccelerometerModel` is **stateless** — it reads `env_state.acc` (computed by the env as
`(vel_t - vel_{t-1}) / dt`) and projects gravity into the body frame:
`accel_true = env_state.acc - R_body^T × [0, 0, -9.81]`. No internal state needed; no `reset()` override.

---

## Hierarchy 2 — Perturbation

### Base class

```
Perturbation (ABC)
  # Constructor — all catalog fields are constructor arguments
  __init__(
      id: str,
      n_envs: int,
      dt: float,                           # simulation timestep — stored for Lipschitz + OUProcess
      mode: Literal["fixed", "dynamic"],
      frequency: Literal["per_episode", "per_step"],
      scope: Literal["per_env", "global"],
      distribution: str,
      distribution_params: dict,
      bounds: tuple,
      nominal: Any,
      dimension: tuple[int, ...],
      lipschitz_k: float | None = None,
      curriculum_scale: float = 1.0,
      observable: bool = True,
      risk: Literal["low", "medium", "high"] = "low",
      priority: int = 1,
  )

  # Config — matches catalog fields
  id: str
  n_envs: int
  dt: float
  mode: Literal["fixed", "dynamic"]
  frequency: Literal["per_episode", "per_step"]
  scope: Literal["per_env", "global"]
  distribution: str
  distribution_params: dict
  bounds: tuple
  nominal: Any
  dimension: tuple[int, ...]
  lipschitz_k: float | None
  curriculum_scale: float              # [0, 1] — 0 → nominal, 1 → full range
  observable: bool
  risk: Literal["low", "medium", "high"]
  priority: int

  # Internal state
  _current_value: Tensor | None        # last value applied — used for privileged obs + Lipschitz
  _params_prev: dict | None            # previous distribution_params — for Lipschitz enforcement
  is_stateful: bool = False            # set to True in stateful leaves to avoid unnecessary step() calls

  # Lifecycle — called by env
  reset(env_ids: Tensor) → None
  tick(is_reset: bool, env_ids: Tensor | None = None) → None  # orchestrates sample() and step(); env_ids=None means all envs
  step() → None                                  # no-op by default; override in stateful leaves

  # Value interface
  sample() → Tensor                    # DR mode: draw from distribution, apply curriculum_scale
  set_value(value: Tensor) → None      # adversarial mode: Lipschitz clip on value; uses self.dt
  update_params(new_params: dict) → None  # adversarial mode: Lipschitz clip on params; uses self.dt

  # Privileged observation
  get_privileged_obs() → Tensor | None  # returns _current_value if observable=True, else None
```

**`tick()` orchestration logic:**
```
tick(is_reset, env_ids):
  if is_reset:
    reset(env_ids)
    if frequency == "per_episode": sample()
  else:
    if frequency == "per_step": sample()
    if is_stateful: step()
```

**`scope` and output shape:**
- `scope="per_env"` → `sample()` returns `Tensor[n_envs, *dimension]`
- `scope="global"` → `sample()` returns `Tensor[1, *dimension]`; `apply()` relies on PyTorch broadcasting;
  `get_privileged_obs()` returns `Tensor[1, *dimension]`

**`sample()` formula:**
```
raw = draw_from(distribution, distribution_params)
value = nominal + (raw − nominal) × curriculum_scale   # clipped to bounds
_current_value = value
```

**`set_value()` Lipschitz enforcement (adversarial):**
```
if lipschitz_k is not None and _current_value is not None:
    delta = value − _current_value
    value = _current_value + clip(delta, −lipschitz_k × dt, +lipschitz_k × dt)
_current_value = value
```

**`update_params()` Lipschitz enforcement (adversarial on distribution params):**
```
if lipschitz_k is not None and _params_prev is not None:
    for each param: new_p = clip(new_p, prev_p ± lipschitz_k × dt)
_params_prev = new_params
distribution_params = new_params
```

---

### Branch: PhysicsPerturbation

```
PhysicsPerturbation (ABC) — Perturbation
  apply(scene, drone, env_state: EnvState) → None
```

#### `GenesisSetterPerturbation`

Wraps a single Genesis API setter. `setter_fn` must accept `(value: Tensor, envs_idx: Tensor)`.

```
GenesisSetterPerturbation — PhysicsPerturbation
  setter_fn: Callable[[Tensor, Tensor], None]
  apply(scene, drone, env_state) → None
    self.setter_fn(self._current_value, envs_idx)
```

Covers: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 7.1, 7.2

#### `ExternalWrenchPerturbation`

Applies a force or torque via `solver.apply_links_external_force/torque`.
Stateless leaves override only `_compute_wrench()`.
Stateful leaves also set `is_stateful = True` and override `step()`.

```
ExternalWrenchPerturbation (ABC) — PhysicsPerturbation
  frame: Literal["local", "world"]
  link_idx: int
  duration_mode: Literal["continuous", "pulse"]

  _compute_wrench(env_state: EnvState) → Tensor[n_envs, 3]   # abstract
  apply(scene, drone, env_state) → None
    wrench = self._compute_wrench(env_state)
    self._current_value = wrench
    solver.apply_links_external_force(wrench, link_idx, envs_idx, local=(frame=="local"))
```

Stateless leaves: 1.10, 1.11, 1.12, 1.13, 1.14, 2.1–2.5, 2.11, 2.13,
                  5.1, 5.4, 5.5, 5.7, 5.9, 7.3, 8.1 (constant mode), 8.2 (constant mode)

Stateful leaves (`is_stateful=True`, override `step()`):
  1.15, 5.2, 5.3, 5.6, 5.8, 8.1 (OU mode), 8.2 (OU mode)

#### `MotorCommandPerturbation`

Transforms the RPM command tensor before thrust physics. Used for perturbations that modify
motor response without injecting external force/torque — i.e. they intercept the RPM signal.

```
MotorCommandPerturbation (ABC) — Perturbation
  apply(rpm_cmd: Tensor[n_envs, 4]) → Tensor[n_envs, 4]
```

Note: the env calls `MotorCommandPerturbation.apply(rpm_cmd)` instead of passing rpm directly
to Genesis. The returned `rpm_actual` is then used for thrust computation.

Stateless leaves: 2.5 (motor kill — binary mask per episode, no step() needed), 2.7 (RPM noise), 2.8 (RPM saturation)

Stateful leaves (`is_stateful=True`):
  2.6 (first-order IIR lag on rpm_actual),
  2.9 (motor wear — efficiency decay over episode),
  2.10 (rotor imbalance — sinusoidal RPM modulation; also injects correlated IMU noise via `AdditiveNoise`),
  2.12 (cold-start asymmetry — exponential KF warmup)

> **2.10 dual-hook note:** `RotorImbalancePerturbation` is a `MotorCommandPerturbation` for the
> RPM modulation side. It additionally holds a reference to an `AdditiveNoise` instance that it
> updates each step to inject correlated vibration noise into the IMU channels. The env registers
> both hooks. This is the only dual-hook perturbation in the catalog.

---

### Branch: ObservationPerturbation

```
ObservationPerturbation (ABC) — Perturbation
  obs_slice: slice                     # which channels this perturbation targets
  apply(obs: Tensor[n_envs, obs_dim]) → Tensor[n_envs, obs_dim]
```

All leaves update `_current_value` inside `apply()`.

#### `AdditiveNoise`

Stateless. Adds `_current_value` to the targeted obs slice.

```
AdditiveNoise — ObservationPerturbation
  apply(obs) → obs
    obs[:, self.obs_slice] += self._current_value
    return obs
```

Covers: 4.1, 4.2, 4.4 (noise+bias), 4.5, 4.6, 4.8, 4.9, 4.10, 4.11, 4.12, 4.14, 4.15

#### `OUDrift`

Stateful (`is_stateful=True`). OU process state added to obs slice each step.

```
OUDrift — ObservationPerturbation
  _ou: OUProcess
  reset(env_ids) → None       # delegates to _ou.reset(env_ids)
  step() → None               # _ou.step(theta, sigma, mu, dt)
  apply(obs) → obs
    self._current_value = self._ou.state
    obs[:, self.obs_slice] += self._current_value
    return obs
```

Covers: 4.3, 4.4 (drift part), 4.13, 4.16

#### `ObsDropout`

Stateful (`is_stateful=True`). Drops obs slice for a sampled duration, holding last valid value.

```
ObsDropout — ObservationPerturbation
  _counter: Tensor[n_envs]
  _last_valid: Tensor[n_envs, slice_dim]
  reset(env_ids) → None
  step() → None               # decrement counter, sample new dropout events when counter == 0
  apply(obs) → obs
    # where counter > 0: replace obs slice with _last_valid (hold)
    # where counter == 0: update _last_valid, pass through
    self._current_value = (counter > 0).float()   # 1 = active dropout
    return obs
```

Covers: 4.7

#### `ObsDelayBuffer`

Stateful (`is_stateful=True`). Delays obs slice by a per-env sampled number of steps.

```
ObsDelayBuffer — ObservationPerturbation
  _buffer: DelayBuffer
  _delay: Tensor[n_envs]      # sampled once per episode (integer steps)
  reset(env_ids) → None
  step() → None
  apply(obs) → obs
    # push→read order (canonical): push current obs first, then read delayed.
    # delay=0 → read returns just-pushed value (pass-through).
    # delay=D → returns obs from D steps ago; zeros on cold start (buffer uninitialized).
    self._buffer.push(obs[:, self.obs_slice])
    delayed = self._buffer.read(self._delay)
    obs[:, self.obs_slice] = delayed
    self._current_value = self._delay.float()
    return obs
```

Covers: 3.1, 3.2

---

### Branch: ActionPerturbation

```
ActionPerturbation (ABC) — Perturbation
  apply(action: Tensor[n_envs, action_dim]) → Tensor[n_envs, action_dim]
```

All leaves update `_current_value` inside `apply()`.

#### Stateless action leaves

Three distinct stateless leaves — no shared implementation, so no intermediate class.

```
ActionNoise — ActionPerturbation          # 6.1
  apply(action) → action
    noise = sample_from(distribution, _current_value)   # _current_value = noise std
    return action + noise

ActionDeadzone — ActionPerturbation       # 6.2
  apply(action) → action
    # zero out channels where |action| < _current_value (threshold)
    self._current_value = threshold
    return torch.where(action.abs() < threshold, torch.zeros_like(action), action)

ActionSaturation — ActionPerturbation     # 6.3
  apply(action) → action
    # clip action to [-_current_value, +_current_value] (per-axis limit)
    self._current_value = limit
    return action.clamp(-limit, limit)
```

Covers: 6.1 (`ActionNoise`), 6.2 (`ActionDeadzone`), 6.3 (`ActionSaturation`)

#### `ActionDelayBuffer`

Stateful (`is_stateful=True`). Delays action by a per-env sampled number of steps (ZOH).

```
ActionDelayBuffer — ActionPerturbation
  _buffer: DelayBuffer
  _delay: Tensor[n_envs]
  reset(env_ids) → None
  step() → None
  apply(action) → action
```

Covers: 3.3, 3.4, 3.7, 3.8

#### `StatefulActionFilter`

Stateful (`is_stateful=True`). Applies a first-order IIR filter or hysteresis to the action stream.
Both share the same `_state: Tensor[n_envs, action_dim]` update pattern.

```
StatefulActionFilter — ActionPerturbation
  _state: Tensor[n_envs, action_dim]
  reset(env_ids) → None
  step() → None
  apply(action) → action
```

Covers: 6.4 (hysteresis), 6.5 (ESC IIR filter)

---

## Env-level wrappers (not Perturbation subclasses)

These two catalog entries operate at the env loop level. They are **not** `Perturbation` subclasses
and are not registered in the perturbation registry. They are implemented directly in the env wrapper.
Their catalog fields (mode, distribution, etc.) serve as documentation only.

### 3.5 Control loop jitter

Implemented via the substeps trick. The env calls `scene.step()` N times per RL step, where
`N = round(dt_effective / dt_genesis)` and `dt_effective` is sampled from the perturbation distribution
each step (scope=`global` — same N for all envs in a batch). No `Perturbation` subclass needed.

### 3.6 Obs/action desynchronization

Combines one `ObsDelayBuffer` and one `ActionDelayBuffer` with correlated delays.
Implemented as a composite wrapper in the env layer — not a `Perturbation` subclass.

---

## EnvState struct (shared across all hooks)

```python
@dataclass
class EnvState:
    pos:      Tensor  # [n_envs, 3]  world position (m)
    quat:     Tensor  # [n_envs, 4]  orientation (w, x, y, z)
    vel:      Tensor  # [n_envs, 3]  body linear velocity (m/s)
    ang_vel:  Tensor  # [n_envs, 3]  body angular velocity (rad/s)
    acc:      Tensor  # [n_envs, 3]  body linear acceleration (m/s²) — required by AccelerometerModel
    rpm:      Tensor  # [n_envs, 4]  propeller RPM
    dt:       float
    step:     int
```

`acc` is computed by the env as `(vel_t - vel_{t-1}) / dt` before building `EnvState`.

---

## Summary

| Component | Type | Purpose |
|---|---|---|
| `OUProcess` | Utility | OU state for stateful perturbations |
| `DelayBuffer` | Utility | Circular buffer for delay perturbations |
| `SensorModel` | ABC | Genesis state → raw sensor reading |
| `Perturbation` | ABC | Base: config, Lipschitz, curriculum, privileged obs |
| `PhysicsPerturbation` | ABC | Side-effects on Genesis scene |
| `GenesisSetterPerturbation` | Leaf group | Wraps Genesis API setters (11 perturbations) |
| `ExternalWrenchPerturbation` | ABC | Force/torque via external wrench API |
| `ObservationPerturbation` | ABC | Transforms obs tensor; targets `obs_slice` |
| `AdditiveNoise` | Leaf group | Stateless additive noise on obs slice |
| `OUDrift` | Leaf group | Stateful OU drift on obs slice |
| `ObsDropout` | Leaf | Stateful dropout with hold |
| `ObsDelayBuffer` | Leaf | Stateful obs delay |
| `ActionPerturbation` | ABC | Transforms action tensor |
| `ActionNoise` / `ActionDeadzone` / `ActionSaturation` | Leaves | Stateless action transforms (no shared impl) |
| `ActionDelayBuffer` | Leaf | Stateful action delay + ZOH |
| `StatefulActionFilter` | Leaf group | Stateful IIR / hysteresis filter |
