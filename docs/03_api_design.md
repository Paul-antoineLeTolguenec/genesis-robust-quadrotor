# 03 — API Design

> Gymnasium + adversarial API for `genesis-robust-rl`.
> References: `00_feasibility.md`, `01_perturbations_catalog.md`, `02_class_design.md`.
> No code is written before this document is approved.

---

## Design principles

- **Gymnasium-compatible** — `RobustDroneEnv` implements the standard `gym.Env` interface so it
  works with any RL library (SB3, CleanRL, rllib…) without modification.
- **Mode switch, not two classes** — DR and adversarial share the same `RobustDroneEnv`;
  the `PerturbationMode` flag controls how perturbations receive their values.
- **tick() is frozen in adversarial mode** — `sample()` is never called when mode is ADVERSARIAL;
  the adversary is the sole source of perturbation values.
- **Privileged obs opt-in** — `get_privileged_obs()` is a separate method, not part of the standard
  Gymnasium obs. The user decides whether to pass it to the policy.
- **Perturbation registry decoupled** — the registry is a standalone utility; the env takes
  instantiated `Perturbation` objects, not registry keys. The registry is for config-driven workflows.
- **Variable delta_t via substeps** — no `SimOptions.dt` mutation at runtime; the env calls
  `scene.step()` N times per RL step. N is either fixed or sampled by perturbation 3.5.
- **All tensors on GPU** — Genesis runs on GPU; all perturbation tensors live on the same device.
  The user is responsible for `.cpu()` transfers if an RL library requires it.

---

## 1. `PerturbationMode`

```python
class PerturbationMode(str, Enum):
    DOMAIN_RANDOMIZATION = "domain_randomization"
    ADVERSARIAL = "adversarial"
```

Controls how perturbation values are set each step:

| Mode | Value source | `tick()` calls `sample()` | Lipschitz enforced |
|---|---|---|---|
| `DOMAIN_RANDOMIZATION` | `tick()` → `sample()` | Yes | No |
| `ADVERSARIAL` | `set_value()` or `update_params()` | **No** — frozen | Yes (via `lipschitz_k`) |

**Adversarial mode — tick() contract:** in ADVERSARIAL mode, `tick(is_reset=False)` still advances
stateful perturbations via `step()` (OUDrift, DelayBuffers…), but **never calls `sample()`**,
regardless of `frequency`. The `_current_value` is only updated by `set_value()`.
This holds in **both modes**: `step()` is always called on stateful perturbations each non-reset
tick, regardless of mode — it maintains internal dynamics (OU state, delay buffer) independently
of how `_current_value` is set.

**Adversarial mode — stateful constraint:** stateful perturbations (`is_stateful=True`) must not
be listed in `adversary_targets`, because their `step()` overrides `_current_value` internally
(e.g. OUDrift sets `_current_value = _ou.state` on each step). Only stateless perturbations are
adversarially controllable. `AdversarialEnv.__init__()` raises `ValueError` if a stateful
perturbation is listed in `adversary_targets`.

**Adversarial mode — scope constraint:** only `scope="per_env"` perturbations may appear in
`adversary_targets`. Global-scope perturbations are not adversarially controllable (a global value
controlled per-env would be contradictory). `AdversarialEnv.__init__()` raises `ValueError` if
a global-scope perturbation is listed.

---

## 2. `PerturbationConfig`

Flat configuration struct passed to `RobustDroneEnv.__init__`. Groups perturbations by hook.

```python
@dataclass
class PerturbationConfig:
    physics:    list[PhysicsPerturbation]      = field(default_factory=list)
    motor:      list[MotorCommandPerturbation]  = field(default_factory=list)
    sensors:    list[ObservationPerturbation]   = field(default_factory=list)
    actions:    list[ActionPerturbation]        = field(default_factory=list)
    mode:       PerturbationMode               = PerturbationMode.DOMAIN_RANDOMIZATION
```

**Hook-to-list mapping** (mirrors catalog `application_hook` field):

| Catalog hook | List | When called |
|---|---|---|
| `reset` | all lists | inside `env.reset()` via `tick(is_reset=True)` |
| `pre_physics` | `motor` | before `scene.step()`, intercepts RPM command |
| `post_physics` | `physics` | after `scene.step()`, side-effects on scene state |
| `on_observation` | `sensors` | after sensor models run, corrupts raw obs |
| `on_action` | `actions` | before motor perturbations, corrupts policy action |

> **Catalog note:** motor perturbations are cataloged under `application_hook: pre_physics` because
> they run before the physics step. They form a distinct branch (`MotorCommandPerturbation`) that
> intercepts the RPM command tensor — this is a sub-type of `pre_physics`, not a separate hook.

> **Physics note:** `PhysicsPerturbation.apply()` is called after `scene.step()` (post_physics).
> Genesis setters take effect on the next `scene.step()` call, so the ordering is equivalent to
> "set parameters before the next step's physics computation."

**Perturbation ID lookup:** when `set_perturbation_values({"id": value})` is called, the env does
a linear search across all four lists in declaration order (physics → motor → sensors → actions).
IDs must be unique across all lists; `PerturbationConfig` validates uniqueness at construction time.

**Dual-hook perturbations (2.10):** `RotorImbalancePerturbation` spans both the `motor` hook
(RPM modulation) and the `sensors` hook (correlated IMU noise via an `AdditiveNoise` instance).
The user must instantiate both objects and register them in their respective lists:

```python
rotor_imbalance = RotorImbalancePerturbation(id="rotor_imbalance", ...)
imu_noise = rotor_imbalance.imu_noise   # AdditiveNoise managed internally, exposed as attribute

cfg = PerturbationConfig(
    motor=   [rotor_imbalance],
    sensors= [imu_noise],          # env registers it in the on_observation chain
)
```

`rotor_imbalance.step()` updates `imu_noise._current_value` each step — the env does not need to
know about this internal coupling.

**Env-level wrappers (3.5, 3.6):** these are not `Perturbation` subclasses and are not listed in
`PerturbationConfig`. They are configured via dedicated constructor arguments:
- 3.5 (control loop jitter) → `substeps_range: tuple[int, int] | None` in `RobustDroneEnv.__init__`
- 3.6 (obs/action desync) → `desync_cfg: DesyncConfig | None` in `RobustDroneEnv.__init__`

```python
@dataclass
class DesyncConfig:
    obs_delay_range:    tuple[int, int]  # [min, max] obs delay in steps (integer)
    action_delay_range: tuple[int, int]  # [min, max] action delay in steps (integer)
    correlation:        float = 1.0      # Pearson correlation in [0, 1] between obs and action
                                         # delay samples; 1.0 = fully correlated (same ratio)
```

Both delay buffers are zero-initialized (ZOH cold-start, consistent with `DelayBuffer` in `02_class_design.md`).
Delays are sampled once per episode at `reset()`, not per step.

---

## 3. `SensorConfig`

Maps sensor names to instantiated `SensorModel` objects.

```python
@dataclass
class SensorConfig:
    gyroscope:     GyroscopeModel     | None = None
    accelerometer: AccelerometerModel | None = None
    magnetometer:  MagnetometerModel  | None = None
    barometer:     BarometerModel     | None = None
    gps:           GPSModel           | None = None
    optical_flow:  OpticalFlowModel   | None = None
```

The env calls `forward()` on each non-None model after every `scene.step()`. The resulting raw
readings are concatenated in declaration order to form the sensor part of the observation.
`EnvState.acc` passed to sensor models is the **true acceleration** (before any noise is applied).

---

## 4. `RobustDroneEnv`

### 4.1 Signature

```python
class RobustDroneEnv(gym.Env):
    def __init__(
        self,
        n_envs:            int,
        perturbation_cfg:  PerturbationConfig,
        sensor_cfg:        SensorConfig,
        dt:                float = 0.01,           # Genesis sim timestep (seconds)
        substeps:          int   = 1,              # fixed substeps per RL step (overridden by 3.5)
        substeps_range:    tuple[int, int] | None = None,  # activates 3.5 jitter if set
        desync_cfg:        DesyncConfig | None = None,     # activates 3.6 desync if set
        max_steps:         int   = 1000,
        headless:          bool  = True,
    ) -> None: ...
```

**`observation_space` construction:** computed statically in `__init__` from:
1. Active sensor dims (from `SensorConfig` — fixed per sensor type, listed in §7).
2. `_extra_obs_dim() → int` — override in subclasses to declare extra state dimension (default 0).

No call to `reset()` is needed to build `observation_space`. The env relies on the fact that
sensor output dims are fully determined by sensor type, not by runtime data.

```python
def _extra_obs_dim(self) -> int:
    """Return dimension of extra observations appended after sensor readings. Default: 0."""
    return 0
```

### 4.2 Core Gymnasium methods

```python
def reset(
    self,
    *,
    env_ids: Tensor | None = None,   # None → reset all envs; subset reset for auto-reset wrappers
    seed:    int    | None = None,
    options: dict   | None = None,
) -> tuple[Tensor, dict]:
    """
    Reset environments. Sequence:
      1. scene.reset(env_ids)
      2. self._vel_prev[env_ids] = zeros  (so acc = vel_1 / dt on first step; assumes rest start)
      3. For each perturbation: _current_value = nominal  (ensures clean adversarial state)
      4. tick(is_reset=True, env_ids) on all perturbations
         - DR mode: triggers sample() for per_episode perturbations
         - Adversarial mode: triggers reset() on stateful internals only; sample() is skipped
      5. Build obs: sensor models → sensor perturbations → concat _extra_obs() if any
    Returns (obs, info). info["privileged_obs"] contains get_privileged_obs() output.
    """

def step(
    self,
    action: Tensor,   # [n_envs, action_dim] — on GPU
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
    """
    Standard Gymnasium step → (obs, reward, terminated, truncated, info).
    terminated, truncated: Tensor[n_envs, bool] — per-env flags (no auto-reset).
    info["privileged_obs"] contains get_privileged_obs() output.
    """
```

### 4.3 Internal step loop

```
step(action):
  1. action_perturbed = _apply_action_perturbations(action)
     # hook: on_action — ActionPerturbation.apply() chain, in cfg.actions list order

  2. rpm_cmd = policy_to_rpm(action_perturbed)
     # env-specific: converts action to RPM command tensor [n_envs, 4]

  3. rpm_actual = _apply_motor_perturbations(rpm_cmd)
     # hook: pre_physics — MotorCommandPerturbation.apply() chain, in cfg.motor list order

  4. for _ in range(_get_substeps()):
       scene.step(rpm_actual)
     # substeps: fixed or sampled by 3.5 jitter

  5. env_state = _build_env_state()
     # reads Genesis state; computes acc = (vel_t - vel_{t-1}) / dt; stores vel_t as _vel_prev

  6. _apply_physics_perturbations(env_state)
     # hook: post_physics — PhysicsPerturbation.apply() chain, in cfg.physics list order
     # side-effects on scene; take effect on next scene.step()

  7. raw_obs = _run_sensor_models(env_state)
     # SensorModel.forward(env_state) for each active sensor; concat in SensorConfig order

  8. obs = _apply_sensor_perturbations(raw_obs)
     # hook: on_observation — ObservationPerturbation.apply() chain, in cfg.sensors list order

  9. tick(is_reset=False, env_ids=all) on all perturbations
     # DR mode:         advances stateful (step()), resamples per_step perturbations (sample())
     # Adversarial mode: advances stateful only (step()); sample() is never called

  10. reward, terminated, truncated = _compute_reward(env_state)

  11. info["privileged_obs"] = get_privileged_obs()

  return obs, reward, terminated, truncated, info
```

### 4.4 Internal methods

```python
def _build_env_state(self) -> EnvState:
    """
    Read Genesis state tensors and build EnvState.
    acc is computed as (vel_t - self._vel_prev) / dt; self._vel_prev is updated to vel_t.
    self._vel_prev is initialized to zeros at first reset.
    Stores result as self._last_env_state (used by AdversarialEnv._adversary_reward).
    All tensors on GPU (same device as Genesis).
    """

def _apply_physics_perturbations(self, env_state: EnvState) -> None:
    """
    Call apply(scene, drone, env_state) on each PhysicsPerturbation, in cfg.physics order.
    global-scope perturbations return Tensor[1, *dim]; PyTorch broadcasting is implicit in
    the Genesis setter calls (all setters accept [1, *dim] and broadcast over envs_idx).
    """

def _apply_motor_perturbations(self, rpm_cmd: Tensor) -> Tensor:
    """
    Chain apply(rpm_cmd) through each MotorCommandPerturbation in cfg.motor order.
    Each apply() receives the output of the previous. Returns final rpm_actual [n_envs, 4].
    global-scope motor perturbations broadcast their [1, *dim] value via PyTorch broadcasting.
    """

def _run_sensor_models(self, env_state: EnvState) -> Tensor:
    """
    Call forward(env_state) on each active SensorModel in SensorConfig declaration order.
    Concatenate results along dim=1. Shape: [n_envs, total_sensor_dim].
    """

def _apply_sensor_perturbations(self, raw_obs: Tensor) -> Tensor:
    """
    Chain apply(obs) through each ObservationPerturbation in cfg.sensors order.
    Returns corrupted obs [n_envs, total_sensor_dim].
    global-scope perturbations broadcast [1, *dim] over n_envs via PyTorch.
    """

def _apply_action_perturbations(self, action: Tensor) -> Tensor:
    """
    Chain apply(action) through each ActionPerturbation in cfg.actions order.
    Returns perturbed action [n_envs, action_dim].
    global-scope perturbations broadcast [1, *dim] over n_envs via PyTorch.
    """

def _extra_obs(self) -> Tensor | None:
    """
    Override in subclasses to append extra state to the observation.
    Return shape: [n_envs, _extra_obs_dim()]. Default: None (nothing appended).
    Called after _apply_sensor_perturbations(); concatenated along dim=1.
    """
    return None

def _get_substeps(self) -> int:
    """
    Return the number of scene.step() calls per RL step.
    Always returns a single integer shared across all n_envs (scope=global — all envs
    execute the same N substeps per RL step).
    If substeps_range is set (3.5 active): sample uniformly from substeps_range each step.
    Otherwise: return the fixed substeps constructor argument.
    """
```

### 4.5 Privileged obs

```python
def get_privileged_obs(self) -> Tensor:
    """
    Concatenate get_privileged_obs() from all perturbations with observable=True.
    Ordering: lists in declaration order (physics → motor → sensors → actions);
    within each list, items are concatenated in list order.
    global-scope perturbations return Tensor[1, *dim]; broadcast to [n_envs, *dim] before concat.
    Perturbations with observable=False are skipped (contribute 0 dims).
    Returns Tensor[n_envs, n_observable_params] on GPU.
    If no perturbation is observable, returns empty Tensor[n_envs, 0].
    """
```

### 4.6 Mode switch

```python
def set_mode(self, mode: PerturbationMode) -> None:
    """
    Switch all perturbations between DR and adversarial mode at runtime.
    Does not reset _current_value — perturbations keep their last value until the next
    reset() or set_perturbation_values() call.
    """

def set_perturbation_values(
    self,
    values: dict[str, Tensor],   # {perturbation_id: value_tensor}
) -> None:
    """
    Adversarial mode only. For each id: linear search across physics→motor→sensors→actions;
    calls perturbation.set_value(tensor), which applies Lipschitz clipping internally.
    Lipschitz clipping is relative to _current_value (the last value set, or nominal after reset).
    On the first call after reset(), _current_value = nominal, so clipping starts from nominal.
    If lipschitz_k is None, no clipping is applied.
    Raises ValueError if called in DOMAIN_RANDOMIZATION mode.
    Raises KeyError if perturbation_id not found.
    """

def update_perturbation_params(
    self,
    params: dict[str, dict],     # {perturbation_id: new_distribution_params}
) -> None:
    """
    Adversarial mode only. Calls perturbation.update_params(dict) on each named perturbation.
    Lipschitz enforcement applied inside update_params().
    Raises ValueError if called in DOMAIN_RANDOMIZATION mode.
    """
```

### 4.7 Variable delta_t (substeps)

`substeps_range` activates perturbation 3.5 (control loop jitter). When set, `_get_substeps()`
samples an integer uniformly from `[substeps_range[0], substeps_range[1]]` each RL step (scope=global:
same N for all envs). The RL-level dt seen by perturbation timescales is:

```
rl_dt = N × genesis_dt   (where N is the sampled substep count)
```

Lipschitz bounds and OU parameters in all perturbations reference `rl_dt`, not `genesis_dt`.
When `substeps_range` is active, the env updates `p.dt = rl_dt` on each perturbation **before**
calling `tick()`. The `tick(is_reset, env_ids)` signature is not modified — `rl_dt` is propagated
via the `dt` attribute so stateful `step()` implementations read `self.dt` directly.

---

## 5. `AdversarialEnv`

Wraps `RobustDroneEnv` and exposes a second action space for the adversary.

**Constraints on `adversary_targets`** (validated at construction):
- Only `scope="per_env"` perturbations — global-scope is not adversarially controllable.
- Only stateless perturbations (`is_stateful=False`) — stateful internals override `_current_value`.

### 5.1 Signature

```python
class AdversarialEnv:
    def __init__(
        self,
        env:               RobustDroneEnv,
        adversary_targets: list[str],          # perturbation IDs the adversary controls
        adversary_mode:    Literal["value", "params"] = "value",
    ) -> None:
        """
        Validate adversary_targets: raises ValueError for stateful or global-scope perturbations.
        Build adversary_action_space as gym.spaces.Box:
          - low/high: concatenated bounds of each target perturbation, in adversary_targets order
          - shape: (sum of prod(dim(p)) for p in adversary_targets,)  — flat 1D
        """

    adversary_action_space: gym.spaces.Box    # flat 1D Box, bounds from target perturbations
    observation_space:      gym.spaces.Space  # same as wrapped env
    action_space:           gym.spaces.Space  # same as wrapped env (drone action)
```

**`adversary_action_space` shape:** flat 1D, length = sum of `prod(dimension)` for each perturbation
in `adversary_targets`. Example: two perturbations with `dimension=(3,)` and `dimension=(1,)` →
shape `(4,)`. Bounds are the concatenated `[bounds[0], bounds[1]]` from each target.

The adversary outputs `Tensor[n_envs, adversary_dim]` — per-env values for all envs simultaneously.

### 5.2 Step

```python
def step(
    self,
    drone_action:     Tensor,   # [n_envs, action_dim]
    adversary_action: Tensor,   # [n_envs, adversary_dim] — must match adversary_action_space
) -> tuple[
    Tensor,  # obs          [n_envs, obs_dim]
    Tensor,  # drone_reward [n_envs]
    Tensor,  # adv_reward   [n_envs]
    Tensor,  # terminated   [n_envs, bool]
    Tensor,  # truncated    [n_envs, bool]
    dict,    # info — includes "privileged_obs"
]:
    """
    1. Validate adversary_action shape; raises ValueError on mismatch
       (expected shape: [n_envs, adversary_dim]).
    2. Split adversary_action into per-perturbation slices along dim=1,
       in adversary_targets list order (slice sizes = prod(dimension) per target).
    3. Reshape each slice to [n_envs, *dimension] and call set_perturbation_values().
    4. Call env.step(drone_action) → obs, drone_reward, terminated, truncated, info.
    5. adv_reward = _adversary_reward(obs, drone_reward, env_state).
    info["privileged_obs"] contains env.get_privileged_obs() — the full output for ALL
    observable perturbations, not filtered to adversary_targets. The adversary agent
    may use non-controlled perturbations' values to infer environment dynamics.
    Returns all outputs on GPU.
    """

def reset(self, *, seed=None, options=None) -> tuple[Tensor, dict]:
    """Delegates to env.reset(). Resets _current_value to nominal on all perturbations."""
```

### 5.3 Example

```python
# Two stateless, per-env perturbations the adversary controls
mass_shift = MassShiftPerturbation(
    id="mass_shift", n_envs=16, dt=0.01,
    scope="per_env", bounds=(-0.05, 0.05), ...
)
wind_gust = ConstantWindPerturbation(
    id="wind_gust", n_envs=16, dt=0.01,
    scope="per_env", bounds=((-5, -5, -5), (5, 5, 5)), dimension=(3,), ...
)

env = RobustDroneEnv(
    n_envs=16,
    perturbation_cfg=PerturbationConfig(
        physics=[mass_shift, wind_gust],
        mode=PerturbationMode.ADVERSARIAL,
    ),
    sensor_cfg=SensorConfig(gyroscope=GyroscopeModel(n_envs=16)),
    dt=0.01,
)

adv_env = AdversarialEnv(
    env=env,
    adversary_targets=["mass_shift", "wind_gust"],  # dim 1 + dim 3 → adversary_dim = 4
)
# adv_env.adversary_action_space = Box(low=[−0.05, −5, −5, −5], high=[0.05, 5, 5, 5], shape=(4,))

obs, info = adv_env.reset()
drone_action    = Tensor([n_envs, 4])   # from drone policy
adversary_action = Tensor([n_envs, 4])  # from adversary policy
obs, r_drone, r_adv, term, trunc, info = adv_env.step(drone_action, adversary_action)
```

### 5.5 Adversary reward

```python
def _adversary_reward(
    self,
    obs:          Tensor,      # [n_envs, obs_dim] — post-perturbation observation
    drone_reward: Tensor,      # [n_envs] — reward computed by env._compute_reward()
    env_state:    EnvState,    # true Genesis state at current step (pre-observation noise)
) -> Tensor:                   # [n_envs]
    """
    Override to customize adversary reward. Default: zero-sum (−drone_reward).
    obs is post-perturbation (what the policy sees). env_state is the true state.
    """
    return -drone_reward
```

---

## 6. `PerturbationRegistry`

Standalone utility for config-driven instantiation. Not required by `RobustDroneEnv` — the env
accepts instantiated objects directly.

```python
class PerturbationRegistry:
    def register(self, id: str, cls: type[Perturbation]) -> None:
        """Register a perturbation class. Raises ValueError on duplicate id."""

    def get(self, id: str) -> type[Perturbation]:
        """Retrieve a class by key. Raises KeyError if not found."""

    def list(self) -> list[str]:
        """Return all registered perturbation IDs, sorted alphabetically."""

    def build(self, id: str, n_envs: int, dt: float, **kwargs) -> Perturbation:
        """
        Instantiate a perturbation from its registered key + kwargs.
        Raises KeyError if id not found. Raises TypeError if required kwargs are missing.
        """

    def build_from_config(self, config: dict) -> Perturbation:
        """
        Build from a plain dict: {"id": ..., "n_envs": ..., "dt": ..., ...kwargs}.
        Raises KeyError if "id" missing or unknown. Raises TypeError on missing required fields.
        """
```

**Auto-registration:** each `Perturbation` leaf is decorated with `@perturbation_registry.register("id")`
at class definition time. The decorator sets `cls._registry_id = id` and registers the class.

**Global singleton:**
```python
from genesis_robust_rl.perturbations import perturbation_registry
p = perturbation_registry.build("mass_shift", n_envs=16, dt=0.01, ...)
```

**Seed propagation:** `build()` and `build_from_config()` accept an optional `seed: int` kwarg.
When provided, it is forwarded to `Perturbation.__init__()` which seeds the internal RNG
(`torch.Generator`) used by `sample()`. Perturbations without stochastic sampling ignore it.

---

## 7. Observation space layout

The full observation vector returned by `step()`:

```
obs = concat([sensor_readings, extra_state], dim=1)   # [n_envs, obs_dim]
```

- **`sensor_readings`** — outputs of active `SensorModel.forward()` after `ObservationPerturbation.apply()`,
  concatenated in `SensorConfig` declaration order.
- **`extra_state`** — output of `_extra_obs()` if overridden in a subclass (default: nothing).

Privileged obs is **never** part of `obs` — it is always accessed separately via `get_privileged_obs()`.

**`observation_space` construction (at `__init__`):**
```
obs_dim = sum(sensor_dim for active sensors) + _extra_obs_dim()
observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(obs_dim,), dtype=np.float32)
```

Sensor output dims (fixed per sensor type):

| Sensor | Dim | Unit |
|---|---|---|
| `GyroscopeModel` | 3 | rad/s |
| `AccelerometerModel` | 3 | m/s² |
| `MagnetometerModel` | 3 | µT |
| `BarometerModel` | 1 | m |
| `GPSModel` | 3 | m |
| `OpticalFlowModel` | 2 | m/s |

**Device:** all tensors in `obs`, `reward`, `terminated`, `truncated` are on GPU (same device as
Genesis). If an RL library requires CPU tensors, the user wraps with a standard `ToTensorWrapper`
or calls `.cpu()` explicitly.

**n_envs=1:** the API degrades gracefully — all shapes and logic work identically for `n_envs=1`.
No special-casing is needed; `scope="global"` tensors `[1, *dim]` broadcast to `[1, *dim]`.

---

## 8. Curriculum integration

`curriculum_scale` is a per-perturbation `float` in `[0, 1]`. It applies **only in DR mode** —
in adversarial mode, `set_value()` sets `_current_value` directly and `curriculum_scale` is ignored.

```python
def set_curriculum_scale(
    self,
    scale: float | dict[str, float],   # global scalar or {perturbation_id: scale}
) -> None:
    """
    Set curriculum_scale on perturbations (DR mode only; no-op in adversarial mode).
    float → all perturbations. dict → only named perturbations.
    curriculum_scale is mutable after construction; the sample() formula reads it each call.
    """
```

---

## 9. Error handling conventions

| Situation | Exception |
|---|---|
| `set_perturbation_values()` in DR mode | `ValueError` |
| `update_perturbation_params()` in DR mode | `ValueError` |
| Unknown perturbation ID in `set_perturbation_values()` | `KeyError` |
| Duplicate perturbation ID in `PerturbationConfig` | `ValueError` (at construction) |
| `adversary_targets` contains stateful perturbation | `ValueError` (at `AdversarialEnv.__init__`) |
| `adversary_targets` contains global-scope perturbation | `ValueError` (at `AdversarialEnv.__init__`) |
| `adversary_action` wrong shape in `AdversarialEnv.step()` | `ValueError` |
| `SensorModel.update_params()` type mismatch | `ValueError` |
| Registry key collision in `register()` | `ValueError` |
| `registry.build()` with missing required kwarg | `TypeError` |

---

## Summary

| Component | Role |
|---|---|
| `PerturbationMode` | Enum — DR vs adversarial; governs tick() behavior |
| `PerturbationConfig` | Groups instantiated perturbations by hook; validates unique IDs |
| `DesyncConfig` | Config for 3.6 obs/action desync (not a Perturbation subclass) |
| `SensorConfig` | Maps sensor names to `SensorModel` instances |
| `RobustDroneEnv` | Gymnasium-compatible env; composes perturbations + sensors |
| `AdversarialEnv` | Wrapper adding adversary action space + dual reward |
| `PerturbationRegistry` | Config-driven instantiation utility |
