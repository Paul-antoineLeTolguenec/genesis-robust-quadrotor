# 00 — Genesis Feasibility Study

> Verified against Genesis 0.3.14 (genesis-world).
> Sources: readthedocs, rigid_solver.py source, rigid_entity.py source, GitHub issues.

## Purpose

This document answers the feasibility questions for each perturbation category
before the perturbation catalog is written. Every entry in `01_perturbations_catalog.md`
must reference a finding here.

---

## 1. Variable timestep (delta_t perturbation)

**Status: NOT FEASIBLE via Genesis API — workaround required.**

`SimOptions.dt` and `SimOptions.substeps` are **read-only properties** set at
`scene.build()`. There is no `set_dt()` or equivalent method at runtime.

```python
# simulator.py — read-only properties
@property
def dt(self) -> float: return self._dt
@property
def substeps(self): return self._substeps
```

**Workaround options:**
- **Substeps trick**: build the scene with a small base `dt` (e.g. 2ms) and a
  large `substeps`. To simulate a longer effective step, call `scene.step()`
  multiple times before returning an observation. This emulates variable delta_t
  at the RL loop level without touching Genesis internals.
- **Multi-scene**: create several scenes with different `dt` values and select
  one per episode. Expensive but exact.

**Recommended approach for this project:** substeps trick — expose effective
`delta_t` as a perturbation parameter that controls how many Genesis steps are
consumed per RL step.

**Implication for design:** delta_t perturbation operates at the *env wrapper*
level, not at the physics engine level.

---

## 2. Runtime physics parameters

**Status: FEASIBLE including per-step — source code confirmed.**

Genesis exposes per-environment, per-link/dof setters after `scene.build()`.
Requires enabling `batch_dofs_info=True` and `batch_links_info=True` in
`RigidOptions`.

### Source code analysis (rigid_solver.py + rigid_entity.py)

All setters are **direct Taichi kernel writes to GPU tensors**. No Python-side
rebuild, no JIT recompilation, no scene reconstruction is triggered.

| Parameter        | API method                   | Writes to      | Per-step safe | Notes                              |
|------------------|------------------------------|----------------|---------------|------------------------------------|
| Mass shift       | `set_links_mass_shift()`     | `links_state`  | YES           | Dynamic state tensor               |
| Center of mass   | `set_links_COM_shift()`      | `links_state`  | YES           | Dynamic state tensor               |
| Inertial mass    | `set_links_inertial_mass()`  | `links_info`   | LIKELY YES    | Static info struct, direct write   |
| Friction ratio   | `set_geoms_friction_ratio()` | `geoms_state`  | YES           | Dynamic state tensor               |
| Position gain Kp | `set_dofs_kp()`              | `dofs_info`    | YES           | Direct kernel write                |
| Velocity gain Kv | `set_dofs_kv()`              | `dofs_info`    | YES           | Direct kernel write                |
| Stiffness        | `set_dofs_stiffness()`       | `dofs_info`    | YES           | Direct kernel write                |
| Damping          | `set_dofs_damping()`         | `dofs_info`    | YES           | Direct kernel write                |
| Motor inertia    | `set_dofs_armature()`        | `dofs_info`    | COSTLY        | Triggers `_init_invweight_and_meaninertia` recompute |
| Inertia tensor   | no direct setter found       | —              | UNKNOWN       | `kernel_adjust_link_inertia` exists but not exposed |
| Motor thrust map | `drone._KF` mutation OR `apply_links_external_force` | Python scalar / per-env tensor | YES | `_KF` mutation: same KF all envs, zero overhead. External forces: full per-env support via `envs_idx`. |

### Key finding: "once after build" is a performance recommendation, NOT a constraint

The Genesis documentation recommends applying physics DR once per episode for
performance reasons. Technically, these setters are all idempotent GPU tensor
writes — there is no mechanism that prevents calling them at every step.

**Per-step calls are expected to be performant** (Taichi kernel launches are
cheap; these are single tensor assignments on GPU). The Huygens-Steiner
recompute is already inside the `scene.step()` loop regardless of whether
values change — so calling these setters adds no extra computation.

> **To validate empirically (Phase 2):** confirm that per-step calls to
> `set_mass_shift` / `set_COM_shift` produce no measurable overhead in the
> training loop, and that intra-episode physical consistency holds in practice.

**`set_dofs_armature` exception:** triggers `_init_invweight_and_meaninertia()`
which recomputes mass matrix properties. Likely too expensive for per-step
adversarial use. Use only for per-episode DR.

### Implication for design

- **Domain randomization mode**: all parameters, sampled once per episode reset.
- **Adversarial mode (per-step)**: mass shift, CoM shift, friction, Kp/Kv,
  stiffness, damping are all safe. Armature excluded.
- Full inertia tensor perturbation is not directly supported — approximate via
  mass + CoM shift, or investigate `kernel_adjust_link_inertia` in Phase 2.

---

## 3. Robot state access

**Status: FEASIBLE — first-class API.**

`DroneEntity` exposes the following state at each step:
- Position (3D)
- Attitude / orientation (quaternion)
- Body linear velocity (3D)
- Body angular velocity (3D)
- Previous actions

These are sufficient to build any observation perturbation (noise injection,
delay buffer, partial masking).

**Implication for design:** observation perturbations (noise, delay, masking)
are fully implementable at the env wrapper level, independently of Genesis
internals.

---

## 4. Headless mode

**Status: FEASIBLE — standard pattern, some server setup required.**

```python
scene = gs.Scene(show_viewer=False)  # headless
```

- Works out of the box locally without a display.
- On GPU servers: requires EGL + NVIDIA driver configuration. Without it,
  rendering silently falls back to MESA (CPU), which degrades performance.
- For RL training (no rendering needed): `show_viewer=False` is sufficient and
  has no performance impact.
- For video recording: use `gs.renderers.BatchRenderer` (GPU-accelerated).

**Implication for design:** no constraint on environment design. Training
scripts should always pass `show_viewer=False`.

---

## 5. Parallel environments (vectorized training)

**Status: FEASIBLE — native support.**

Genesis natively supports `n_envs` parallel environments in a single scene
(GPU-batched). All DR setters above accept `(n_envs, ...)` tensors.

**Implication for design:** our `RobustDroneEnv` should expose `n_envs` as a
constructor parameter and pass it through to Genesis.

---

## 6. Motor thrust map (KF / KM perturbation)

**Status: FULLY FEASIBLE — per-step and per-env (via external forces workaround).**

### Source code analysis (drone_entity.py + abd/accessor.py)

`DroneEntity` exposes `KF` (thrust coefficient) and `KM` (torque coefficient) as
**read-only properties** initialized from XML at build time:

```python
@property
def KF(self): return self._KF  # no setter
@property
def KM(self): return self._KM  # no setter
```

At each call to `set_propellels_rpm()`, `KF` and `KM` are passed as `qd.float32`
scalars to the Taichi kernel:

```python
# kernel_set_drone_rpm (abd/accessor.py)
force  = qd.Vector([0.0, 0.0, rpm[i_b, i_prop] ** 2 * KF])
torque = qd.Vector([0.0, 0.0, rpm[i_b, i_prop] ** 2 * KM * spin[i_prop]])
```

**Key finding:** `KF`/`KM` are **not stored in a persistent Taichi tensor** — they
are passed fresh on every call. Directly mutating the Python attribute suffices:

```python
drone._KF = new_kf_value  # per-step or per-episode — no overhead
drone.set_propellels_rpm(rpm)
```

### Limitation: no per-env variation via built-in API

Unlike mass_shift / friction (which write to `(n_envs, ...)` tensors), `KF`/`KM`
are global scalars. All parallel environments share the same thrust coefficient
when using `set_propellels_rpm()`.

### Workaround A — RPM scaling

Since `force ∝ rpm² × KF`, scale rpm by `√(KF_env / KF_nominal)` before passing
to Genesis. Emulates per-env KF but modifies the control signal — adds wrapper
complexity and requires care at observation level (policy sees unscaled actions).

### Workaround B — External forces (RECOMMENDED, confirmed feasible)

Genesis exposes on `RigidSolver`:
```python
solver.apply_links_external_force(force, links_idx=propeller_links, envs_idx=..., local=True)
solver.apply_links_external_torque(torque, links_idx=propeller_links, envs_idx=..., local=True)
```
Both accept `envs_idx` and `(n_envs, n_links, 3)` force tensors — **natively per-env**.

**Implementation:** bypass `set_propellels_rpm()` entirely. The wrapper computes:
```python
force_z[i] = rpm[i]^2 * KF_i   # per-env KF
torque_z[i] = rpm[i]^2 * KM_i * spin
```
and applies them via `apply_links_external_force`. Policy still outputs RPM commands;
wrapper handles the thrust physics with per-env coefficients.

### Implication for design

- **Per-step adversarial, same KF across envs**: use `drone._KF` mutation — zero overhead.
- **Per-env different KF** (DR or adversarial): use Workaround B — fully native, per-step safe.
- Flag in catalog: `KF`/`KM` perturbation is **FULLY FEASIBLE** via Workaround B.

---

## Summary table

| Feature                          | Status       | Risk level | Notes                              |
|----------------------------------|--------------|------------|------------------------------------|
| Variable delta_t (runtime)       | Workaround   | Medium     | Substeps trick at wrapper level    |
| Mass / CoM perturbation          | Feasible     | Low        | Per-step + per-env native (dynamic tensor) |
| Friction perturbation            | Feasible     | Low        | Per-step + per-env native (dynamic tensor) |
| Motor gain (Kp/Kv) perturbation  | Feasible     | Low        | Per-step + per-env native (dofs_info tensor) |
| Per-step physics perturbation    | Feasible     | Low-Med    | Direct GPU tensor writes (confirmed in source) |
| Per-step armature perturbation   | Costly       | Medium     | Triggers mass matrix recompute     |
| Inertia tensor perturbation      | No direct API| Medium     | Approx via mass+CoM; kernel exists |
| Motor thrust map (KF/KM)         | Feasible     | Low        | Per-step via `_KF` mutation; per-env via `apply_links_external_force` (native) |
| Observation noise / delay        | Feasible     | Low        | Pure wrapper logic                 |
| Action perturbation              | Feasible     | Low        | Pure wrapper logic                 |
| Headless training                | Feasible     | Low        | show_viewer=False                  |
| Parallel envs (n_envs)           | Feasible     | Low        | Native Genesis support             |
