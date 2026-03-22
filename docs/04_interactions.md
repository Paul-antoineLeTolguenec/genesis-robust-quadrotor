# 04 — Component Interactions

> Interaction sequences between `RobustDroneEnv`, `Perturbation`, `SensorModel`, and `AdversarialEnv`.
> References: `02_class_design.md`, `03_api_design.md`.
> No code is written before this document is approved.

---

## Design principles

- **Linear call chains, no callbacks** — the env drives all component calls in a fixed, deterministic
  order. No component calls back into the env.
- **Separation of concerns** — `tick()` is the sole orchestration point for each `Perturbation`;
  the env never calls `sample()`, `step()`, or `reset()` directly on a perturbation.
- **SensorModel and Perturbation are fully decoupled** — sensor models produce raw readings;
  observation perturbations corrupt them. Neither knows about the other.
- **AdversarialEnv is a thin wrapper** — it sets perturbation values, then delegates entirely
  to `env.step()`. No step logic is duplicated.

---

## 1. Component dependency graph

Static relationships (arrows = "depends on" / "calls"):

```
PerturbationRegistry
  └─ builds → Perturbation instances (optional; env accepts instances directly)

RobustDroneEnv
  ├─ owns → Genesis scene + drone
  ├─ reads → PerturbationConfig
  │    ├─ physics:  list[PhysicsPerturbation]
  │    ├─ motor:    list[MotorCommandPerturbation]
  │    ├─ sensors:  list[ObservationPerturbation]
  │    └─ actions:  list[ActionPerturbation]
  ├─ reads → SensorConfig
  │    └─ {name: SensorModel | None}
  ├─ reads → DesyncConfig (optional, for 3.6)
  ├─ exposes → get_privileged_obs(), set_perturbation_values(), set_curriculum_scale()
  └─ stores  → _last_env_state (initialized to None at __init__; updated each step [5]; read by AdversarialEnv._adversary_reward)

AdversarialEnv
  ├─ wraps → RobustDroneEnv (delegates reset + step)
  ├─ validates → adversary_targets ⊆ per_env + stateless perturbations
  └─ exposes → adversary_action_space, step(drone_action, adversary_action)

Perturbation (each instance)
  ├─ owns → _current_value, _params_prev
  ├─ may own → OUProcess, DelayBuffer (stateful leaves only)
  └─ exposes → tick(), get_privileged_obs(), set_value(), update_params()

SensorModel (each instance)
  ├─ reads → EnvState (provided by env each step)
  └─ exposes → forward(env_state), update_params(new_params), reset(env_ids)

EnvState (struct, not a class)
  └─ produced by → RobustDroneEnv._build_env_state()
     consumed by → SensorModel.forward(), PhysicsPerturbation.apply()
```

**Key decouplings:**
- `SensorModel` never calls `Perturbation` — the env chains them.
- `Perturbation` never reads from `SensorModel` — `EnvState` is the sole state carrier.
- `PerturbationConfig` holds only references; it performs no logic beyond uniqueness validation.

---

## 2. Reset sequence

Caller: `env.reset(env_ids=None | Tensor, seed=None, options=None)`

```
reset(env_ids):

  [1] scene.reset(env_ids)
      Genesis resets physics state for the requested envs.
      If env_ids is None: all envs are reset.

  [2] Reset _vel_prev for the affected envs:
        if env_ids is None: self._vel_prev.zero_()     # full reset: all envs
        else:               self._vel_prev[env_ids] = 0  # partial reset: env_ids is a 1D integer Tensor
      Clears stored velocity so acc = vel_1 / dt on the first step
      (rest-start assumption: drone is stationary at reset).
      Note: _vel_prev is a Tensor[n_envs, 3] allocated to zeros in __init__(), not lazily
      at first reset. This ensures step() is safe to call even before reset().

  [3] For each perturbation p in (physics + motor + sensors + actions):
        p._current_value = p.nominal
      Ensures a clean adversarial baseline: next set_value() clips relative to nominal.
      In DR mode this is overwritten by tick() immediately after.

  [4] For each SensorModel m in sensor_cfg (non-None):
        m.reset(env_ids)
      No-op for stateless sensors (AccelerometerModel, GyroscopeModel in most configs).
      Override only in custom stateful sensor leaves.

  [5] For each perturbation p in (physics + motor + sensors + actions):
        p.tick(is_reset=True, env_ids=env_ids)

      tick(is_reset=True) contract per perturbation:
        [5a] p.reset(env_ids)
             - Stateful leaves: reset internal state (OUProcess, DelayBuffer, counters).
             - Stateless leaves: no-op.
        [5b] DR mode:
               if p.frequency == "per_episode": p.sample()  → p._current_value updated
               if p.frequency == "per_step":    nothing (sampled at step time)
             ADVERSARIAL mode:
               sample() is never called; _current_value remains at nominal (set in step [3]).

  [6] Build initial obs:
        env_state = _build_env_state()          # reads Genesis state post-reset
        raw_obs   = _run_sensor_models(env_state)
        obs       = _apply_sensor_perturbations(raw_obs)
        extra     = _extra_obs()
        obs       = concat([obs, extra], dim=1) if extra is not None else obs

  [7] Build info:
        info["privileged_obs"] = get_privileged_obs()

  return (obs, info)
```

**Mode-dependent behavior at reset:**

| Step | DR mode | ADVERSARIAL mode |
|---|---|---|
| Step [3] | _current_value = nominal (overwritten by tick in [5b]) | _current_value = nominal (adversary sets values next step) |
| Step [5b] | per_episode perturbations call sample() | sample() never called |
| Post-reset state | _current_value = sampled value | _current_value = nominal |

---

## 3. Step sequence — DR mode

Caller: `env.step(action: Tensor[n_envs, action_dim])`

```
step(action):

  [0] (3.6 desync action delay — only when desync_cfg is not None)
      action_to_use = action_delay_buffer.read(_action_delay)
      action_delay_buffer.push(action)
      # If desync_cfg is None: action_to_use = action (pass-through)

  [1] action_perturbed = _apply_action_perturbations(action_to_use)
      For each p in cfg.actions (in list order):
        action_perturbed = p.apply(action_perturbed)
        # p._current_value updated inside apply()
      Hook: on_action.

  [2] rpm_cmd = policy_to_rpm(action_perturbed)
      Env-specific. Converts policy action to RPM command Tensor[n_envs, 4].

  [3] rpm_actual = _apply_motor_perturbations(rpm_cmd)
      For each p in cfg.motor (in list order):
        rpm_cmd = p.apply(rpm_cmd)
        # p._current_value updated inside apply(); chained result passed to next
      Hook: pre_physics.

  [4] N = _get_substeps()
      If substeps_range is set (3.5 active): N ~ Uniform(substeps_range) (integer, global)
      Else: N = self.substeps (fixed constructor arg)

      For _ in range(N):
        scene.step(rpm_actual)
      # Genesis advances physics N × genesis_dt seconds.

  [5] env_state = _build_env_state()
      pos, quat, vel, ang_vel = read from Genesis state tensors
      acc = (vel - self._vel_prev) / self.dt
      self._vel_prev = vel
      rpm = read propeller RPM from Genesis
      env_state = EnvState(pos, quat, vel, ang_vel, acc, rpm, dt=self.dt, step=self._step_count)
      self._last_env_state = env_state   # stored for AdversarialEnv._adversary_reward()

  [6] _apply_physics_perturbations(env_state)
      For each p in cfg.physics (in list order):
        p.apply(scene, drone, env_state)
        # PhysicsPerturbation calls Genesis setters (GenesisSetterPerturbation)
        # or apply_links_external_force/torque (ExternalWrenchPerturbation)
        # Side-effects take effect on the next scene.step() call.
      Hook: post_physics (temporal position: after current scene.step()).
      Catalog label: "pre_physics" (semantic: setter values are active at the NEXT scene.step()).
      Both labels are correct; see 01_perturbations_catalog.md §Hook signatures for the canonical note.

  [7] raw_obs = _run_sensor_models(env_state)
      For each SensorModel m in sensor_cfg (non-None, in declaration order):
        reading = m.forward(env_state)     # raw reading, shape [n_envs, dim]
      raw_obs = concat(readings, dim=1)    # [n_envs, total_sensor_dim]

  [8] obs = _apply_sensor_perturbations(raw_obs)
      For each p in cfg.sensors (in list order):
        raw_obs = p.apply(raw_obs)
        # p._current_value updated inside apply(); chained result passed to next
      Hook: on_observation.
      obs = concat([raw_obs, _extra_obs()], dim=1) if _extra_obs() is not None else raw_obs

  [8b] (3.6 desync obs delay — only when desync_cfg is not None)
       obs_delayed = obs_delay_buffer.read(_obs_delay)
       obs_delay_buffer.push(obs)
       obs = obs_delayed
       # If desync_cfg is None: obs unchanged (pass-through)

  [9] tick(is_reset=False, env_ids=torch.arange(n_envs)) on all perturbations
      # env_ids=torch.arange(n_envs) means "all envs" at non-reset ticks; never None here.
      # If substeps_range is active: update p.dt = rl_dt on each perturbation before tick():
      if substeps_range is not None:
        for p in all_perturbations:
          p.dt = rl_dt   # propagate variable rl_dt before stateful step() reads self.dt
      # Note: at 68 perturbations and 1000 steps/s, this is ~68k Python attr mutations/s.
      # Acceptable in isolation; covered by cumulative tick() perf test (see 05_test_conventions.md §P4).
      For each p in (physics + motor + sensors + actions):
        p.tick(is_reset=False, env_ids=torch.arange(n_envs))

      tick(is_reset=False) contract per perturbation (DR mode):
        if p.frequency == "per_step":  p.sample()   # resamples _current_value
        if p.is_stateful:              p.step()      # advances internal dynamics (OU, buffer)
        # per_episode perturbations: neither sample() nor step() called here

  [10] reward, terminated, truncated = _compute_reward(env_state)
       Env-specific. Accesses env_state and current obs.

  [11] info["privileged_obs"] = get_privileged_obs()
       Concatenate get_privileged_obs() from all observable perturbations
       (physics → motor → sensors → actions, in list order).

  return (obs, reward, terminated, truncated, info)
```

**Call ordering rationale:**

- Actions are perturbed **before** motor perturbations: the policy's output is corrupted, then
  the corrupted signal is intercepted at the motor level.
- Physics perturbations run **after** `scene.step()`: Genesis setters take effect on the next
  physics step, so functionally "they set parameters before the next step's physics."
- Sensor models run **after** physics perturbations: the env reads the Genesis state that results
  from the perturbed physics.
- `tick()` runs **after** all apply() calls: per_step perturbations are resampled for the *next*
  step. This means `_current_value` at step T was set by tick() at end of step T−1 (or by reset
  for step 0).

---

## 4. Step sequence — ADVERSARIAL mode

Caller: `adv_env.step(drone_action, adversary_action)`

```
AdversarialEnv.step(drone_action, adversary_action):

  [A1] Validate adversary_action shape:
       Expected: [n_envs, adversary_dim]  where adversary_dim = sum(prod(dim(p)) for p in targets)
       Raises ValueError on mismatch.

  [A2] Split adversary_action into per-perturbation slices along dim=1:
       slices = split(adversary_action, [prod(dim(p)) for p in adversary_targets], dim=1)
       For each (perturbation_id, slice) in zip(adversary_targets, slices):
         value = slice.reshape(n_envs, *perturbation.dimension)
         env.set_perturbation_values({perturbation_id: value})
           → perturbation.set_value(value)
               if lipschitz_k is not None and _current_value is not None:
                 delta = value − _current_value
                 value = _current_value + clip(delta, ±lipschitz_k × p.dt)
                 # p.dt here is rl_dt_{t-1} (set at step [9] of the previous step).
                 # This 1-step lag is intentional and acceptable: rl_dt varies slowly
                 # and the Lipschitz bound is approximate by design.
               _current_value = value

  [A3] obs, drone_reward, terminated, truncated, info = env.step(drone_action)
       Delegates entirely to RobustDroneEnv.step(), which runs the full step sequence [1–11].

       Difference from DR mode within env.step():
       - Step [9] tick(is_reset=False):
           ADVERSARIAL mode → stateful perturbations call step() (advances dynamics)
                           → sample() is NEVER called (frozen)
           _current_value of non-stateful perturbations = last set by set_value() (step A2)

  [A4] adv_reward = _adversary_reward(obs, drone_reward, self.env._last_env_state)
       Default: −drone_reward (zero-sum).
       Override in subclass for custom adversary objectives.

       env_state is accessed via env._last_env_state — RobustDroneEnv stores the most recent
       EnvState as an attribute after _build_env_state() in step [5]:
         self._last_env_state = env_state   # updated every step
       _last_env_state is initialized to None at __init__(); it is guaranteed non-None at [A4]
       because env.step() (step [A3]) always runs step [5] before returning.
       AdversarialEnv accesses it as self.env._last_env_state.
       This avoids modifying the env.step() return signature.

  return (obs, drone_reward, adv_reward, terminated, truncated, info)
```

**Adversarial vs DR tick() comparison:**

| Event | DR mode | ADVERSARIAL mode |
|---|---|---|
| per_step perturbation | `sample()` called → new _current_value | `sample()` skipped; _current_value from adversary |
| per_episode perturbation | no tick action mid-episode | no tick action mid-episode |
| stateful perturbation | `step()` advances dynamics | `step()` advances dynamics (same) |
| _current_value update source | `sample()` | `set_value()` via adversary_action |

---

## 5. Special cases

### 5.1 per_episode vs per_step perturbations

`frequency` determines when `_current_value` is updated during a DR episode:

```
per_episode perturbation:
  reset():   tick(is_reset=True) → sample()   ← value set once
  step() ×N: tick(is_reset=False) → nothing   ← value unchanged across all steps

per_step perturbation:
  reset():   tick(is_reset=True) → NO sample() ← _current_value = nominal (from reset step [3])
  step() ×N: tick(is_reset=False) → sample()  ← new value every step
```

> **Note:** `per_step` perturbations do NOT receive an initial `sample()` at reset.
> Per `02_class_design.md`, `tick(is_reset=True)` only calls `sample()` when `frequency == "per_episode"`.
> A `per_step` perturbation starts each episode at `_current_value = nominal` and is first sampled
> at the end of step 0 (tick at step [9]) — effective from step 1 onwards.

In ADVERSARIAL mode: both behave the same — `sample()` never called; `_current_value` is only
updated by `set_value()`.

### 5.2 Stateful vs stateless in both modes

```
stateless (is_stateful=False):
  tick(is_reset=False): step() skipped
  tick(is_reset=True):  reset() is a no-op

stateful (is_stateful=True):
  tick(is_reset=False): step() called every step in BOTH DR and ADVERSARIAL mode
  tick(is_reset=True):  reset() resets internal state (OUProcess, DelayBuffer, counters)
```

Stateful perturbations maintain independent internal dynamics regardless of mode. In ADVERSARIAL
mode, `step()` is still called — it advances `OUProcess`, ticks `DelayBuffer`, decrements counters.
The adversary cannot directly control stateful perturbations' `_current_value` because `step()`
would override it. This is why stateful perturbations are excluded from `adversary_targets`.

### 5.3 scope=global broadcasting

`scope="global"` perturbations sample a single value shared by all envs:

```
sample() returns Tensor[1, *dimension]
apply():
  - ObservationPerturbation: obs[:, slice] += value   # broadcasts [1, dim] → [n_envs, dim]
  - PhysicsPerturbation:     genesis_setter(value, envs_idx)  # setter broadcasts internally
  - MotorCommandPerturbation: rpm + value  # broadcasts [1, 4] → [n_envs, 4]

get_privileged_obs():
  value.expand(n_envs, *dimension)   # broadcast to [n_envs, *dim] before concat
```

Global perturbations are never in `adversary_targets` (validated at `AdversarialEnv.__init__()`).

### 5.4 2.10 — RotorImbalancePerturbation dual-hook

`RotorImbalancePerturbation` spans two hooks. The user registers both objects explicitly:

```
# Construction (user side):
rotor_imbalance = RotorImbalancePerturbation(id="rotor_imbalance", ...)
imu_noise = rotor_imbalance.imu_noise  # AdditiveNoise managed internally, exposed as attribute

cfg = PerturbationConfig(
    motor=   [rotor_imbalance],   # hook: pre_physics
    sensors= [imu_noise],         # hook: on_observation
)
```

**Frequency and tick() invariant for `imu_noise`:**

`imu_noise` must have `frequency = "per_episode"`. This avoids the conflict where `tick()`'s
`sample()` call (in DR per_step mode) would overwrite the noise amplitude set by
`rotor_imbalance.step()`. With `per_episode`, `sample()` is only called at reset.

`rotor_imbalance.step()` calls `imu_noise.update_params({"std": amplitude})` to set the noise
distribution width, then immediately calls `imu_noise.sample()` to draw a new `_current_value`
from the updated params. This explicit sample() call is necessary because `tick()` never calls
`sample()` mid-episode for `per_episode` perturbations, and `apply()` uses `_current_value`
as-is without drawing.

**Interaction sequence within a step:**

```
step():
  [3] _apply_motor_perturbations(rpm_cmd):
        rotor_imbalance.apply(rpm_cmd)
          → modulates RPM sinusoidally; returns rpm_actual
          → rotor_imbalance._current_value = rpm_modulation  [n_envs, 4]

  [9] tick(is_reset=False):
        rotor_imbalance.tick():
          → rotor_imbalance.step():
               updates sinusoid phase
               amplitude = vibration_amplitude(rotor_imbalance._current_value)
               imu_noise.update_params({"std": amplitude})  # updates distribution_params
               imu_noise.sample()  # explicit redraw from updated params → imu_noise._current_value
               # rotor_imbalance.step() drives both the param update AND the redraw.
               # tick() never calls imu_noise.sample() mid-episode (per_episode frequency).

        imu_noise.tick():
          → is_stateful=False → step() skipped
          → frequency="per_episode" → sample() NOT called (consistent with 02_class_design.md)
          # _current_value already updated by rotor_imbalance.step() above

  [8] (next step) imu_noise.apply(obs):
        obs[:, slice] += imu_noise._current_value   # uses value set by rotor_imbalance.step()
```

**Reset behavior:**

At `tick(is_reset=True)`: `imu_noise.sample()` is called (per_episode) using the **nominal**
`distribution_params` (no imbalance at episode start). From step 0 tick onwards,
`rotor_imbalance.step()` begins updating `imu_noise.distribution_params` to reflect the
growing sinusoidal imbalance. The noise level increases from nominal over the episode.

> **Design note:** the env has no knowledge of the internal coupling between `rotor_imbalance` and
> `imu_noise`. The coupling is entirely encapsulated in `rotor_imbalance.step()`. The env simply
> calls `tick()` on both objects in list order.

### 5.5 Substeps jitter (3.5) and desync (3.6)

**3.5 — Control loop jitter:**

```
_get_substeps() called at each step():
  if substeps_range is not None:
    N = int(Uniform(substeps_range[0], substeps_range[1]))   # same N for all n_envs
  else:
    N = self.substeps   # fixed

for _ in range(N):
  scene.step(rpm_actual)

rl_dt = N × genesis_dt   # effective RL timestep varies each step
```

Perturbation `dt` stored at construction references `genesis_dt`. When `rl_dt` varies (substeps
changes), the env updates `p.dt = rl_dt` on **each perturbation** before calling `tick()`:

```
rl_dt = N × genesis_dt
if substeps_range is not None:
  for p in all_perturbations:
    p.dt = rl_dt   # mutate dt in-place before tick()
```

Stateful `step()` implementations read `self.dt` when calling `OUProcess.step(dt=self.dt)`.
Lipschitz enforcement in `set_value()` also reads `self.dt`. This pattern is consistent with
`02_class_design.md`: "Callers read `self.dt` from the owning Perturbation and pass it through."
The `tick()` signature `(is_reset, env_ids)` is not modified — `rl_dt` is propagated via the
attribute, not as a parameter.

**3.6 — Obs/action desync:**

```
# Configured via DesyncConfig, not a Perturbation subclass.
# Instantiated internally by RobustDroneEnv as two DelayBuffers.

At reset():
  sample correlated delay pair (obs_delay, action_delay) from DesyncConfig ranges
  correlation controls the Pearson correlation between the two delay samples
```

**Positioning in the step chain:**

3.6 delays are applied as explicit sub-steps, distinct from `Perturbation.apply()` chains:

```
step(action):
  [0]  (3.6 action delay — BEFORE action perturbations)
       action_to_use = action_delay_buffer.read(_action_delay)
       action_delay_buffer.push(action)
       # action_to_use is the raw policy action issued _action_delay steps ago

  [1]  action_perturbed = _apply_action_perturbations(action_to_use)
       # ActionNoise, ActionSaturation etc. corrupt the (already-delayed) action
       Hook: on_action.

  ...

  [8]  obs = _apply_sensor_perturbations(raw_obs)
       # ObservationPerturbation chain corrupts the current raw obs
       Hook: on_observation.

  [8b] (3.6 obs delay — AFTER sensor perturbations)
       obs_delayed = obs_delay_buffer.read(_obs_delay)
       obs_delay_buffer.push(obs)
       obs = obs_delayed
       # policy receives the corrupted obs from _obs_delay steps ago
```

**Ordering rationale:**
- Action delay runs FIRST (step [0]): it models the communication latency between policy and
  motors; the delayed action arrives, then hardware effects (ActionNoise, saturation) are applied.
- Obs delay runs LAST (step [8b]): sensor perturbations corrupt the current reading, then the
  corrupted reading is delayed — the policy observes a stale, already-corrupted obs.

**Cold start (ZOH):** both buffers are zero-initialized. On the first `_obs_delay` steps, `read()`
returns zeros; on the first `_action_delay` steps, `read()` returns zeros. This is consistent with
`DelayBuffer` in `02_class_design.md`.

**Obs at reset (step [6]):** the obs built at reset does NOT pass through the delay buffers.
The buffers are zero-initialized immediately before, so the initial obs is the current
(non-delayed) sensor reading. The delay effect activates only from the first `step()` call.

3.6 delay buffers are **not** `Perturbation` instances and are not in `cfg.sensors` or `cfg.actions`.
They are internal to `RobustDroneEnv` and activated only when `desync_cfg` is not None.

---

## 6. Mode switch at runtime

```
env.set_mode(PerturbationMode.ADVERSARIAL)
  For each perturbation p:
    p.mode = ADVERSARIAL
  # _current_value NOT reset — perturbations keep their last sampled value
  # until the next reset() or set_perturbation_values() call

Recommended pattern (switch between episodes):
  obs, info = env.reset()        # step [3]: _current_value = nominal; step [5b]: tick skips sample()
  env.set_mode(ADVERSARIAL)      # or set mode before reset(); same effect after reset()
  # First adversary action sets values from nominal baseline
```

`set_mode()` affects only the `tick()` behavior at the **next** step. The current `_current_value`
is preserved. This allows switching mode mid-training without corrupting perturbation state.

---

## 7. Curriculum update

`set_curriculum_scale()` modifies the `curriculum_scale` attribute on each perturbation:

```
env.set_curriculum_scale(0.5)
  For each perturbation p:
    p.curriculum_scale = 0.5

Effect on sample() (DR mode only):
  raw = draw_from(distribution, distribution_params)
  value = nominal + (raw − nominal) × curriculum_scale   # = 0.5 → half the range
  _current_value = value (clipped to bounds)
```

`curriculum_scale` is a **live attribute** — `sample()` reads it on every call. No re-instantiation needed.
In ADVERSARIAL mode, `set_curriculum_scale()` has no effect: `sample()` is never called, and
`set_value()` takes values directly from the adversary regardless of `curriculum_scale`.

---

## 8. Perturbation tick() summary across all cases

Condensed reference for all combinations of mode × frequency × is_stateful × is_reset:

| Mode | is_reset | frequency | is_stateful | sample() | step() | reset() |
|---|---|---|---|---|---|---|
| DR | True | per_episode | False | ✓ | — | ✓ (no-op) |
| DR | True | per_episode | True | ✓ | — | ✓ (resets state) |
| DR | True | per_step | False | — | — | ✓ (no-op) |
| DR | True | per_step | True | — | — | ✓ (resets state) |
| DR | False | per_episode | False | — | — | — |
| DR | False | per_episode | True | — | ✓ | — |
| DR | False | per_step | False | ✓ | — | — |
| DR | False | per_step | True | ✓ | ✓ | — |
| ADV | True | any | False | — | — | ✓ (no-op) |
| ADV | True | any | True | — | — | ✓ (resets state) |
| ADV | False | any | False | — | — | — |
| ADV | False | any | True | — | ✓ | — |

> `reset()` in the table refers to `p.reset(env_ids)` called inside `tick()`.
> In ADVERSARIAL mode, `_current_value` is always set by `set_value()`, never by `sample()`.

---

## Summary

| Sequence | Trigger | Key actors |
|---|---|---|
| reset() | user or auto-reset wrapper | scene, perturbations (all lists), sensor models |
| step() DR | RL training loop | env, Genesis scene, sensor models, perturbations (all hooks) |
| step() ADV | AdversarialEnv | adversary_action → set_value() → env.step() |
| tick() per_episode | is_reset=True only | sample() at episode start; nothing mid-episode |
| tick() per_step | is_reset=False | sample() every step (DR) or skipped (ADV) |
| tick() stateful | is_reset=False | step() always called in both modes |
| 2.10 dual-hook | rotor_imbalance.step() | updates imu_noise distribution params internally |
| 3.5 jitter | _get_substeps() | integer N sampled globally each RL step |
| 3.6 desync | action delay [0] + obs delay [8b] | action delayed before perturbations; obs delayed after; ZOH cold start |
| mode switch | set_mode() | changes tick() branch; _current_value preserved |
| curriculum | set_curriculum_scale() | live attribute; affects sample() formula in DR mode only |
