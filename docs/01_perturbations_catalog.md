# 01 — Perturbations Catalog

> Scope: quadrotor drone (Genesis 0.3.14).
> Every entry references `docs/00_feasibility.md`.
> This document must be fully validated before any class is written (`docs/02_class_design.md`).

---

## Perturbation field definitions

Each perturbation is fully specified by the following fields:

| Field | Description | Possible values |
|---|---|---|
| `mode` | Whether the value is fixed for the episode or changes over time | `fixed` / `dynamic` |
| `frequency` | When the value is resampled | `per_episode` / `per_step` |
| `scope` | Whether each parallel env gets its own value | `per_env` / `global` |
| `stateful` | Whether the perturbation has internal dynamics requiring `reset()` + `step()` | `yes` / `no` — maps to `is_stateful: bool` in the `Perturbation` base class |
| `distribution` | Sampling distribution(s) supported | `uniform`, `gaussian`, `beta`, `ou_process`, `constant` |
| `distribution_params` | Configurable parameters of the chosen distribution — see reference table below | typed parameter list |
| `bounds` | Hard physical bounds `[min, max]` — mandatory for all; sampled values are always clipped | physical values + unit |
| `nominal` | Default unperturbed value | value + unit |
| `dimension` | Shape of the perturbation tensor | `scalar`, `vector(n)`, `matrix(n,m)` |
| `application_hook` | Where in the env loop the perturbation is applied | `reset`, `pre_physics`, `post_physics`, `on_observation`, `on_action` |
| `lipschitz_k` | Max allowed variation `‖p_t - p_{t-1}‖` per step (None = per_episode only) | float or `None` |
| `curriculum_scale` | Global intensity multiplier — **universal, applies to all perturbations** — formula: `value = nominal + (sampled − nominal) × curriculum_scale`, then clipped to `bounds`. At `0.0`: no perturbation. At `1.0`: full perturbation. | `float ∈ [0.0, 1.0]`, default `1.0` |
| `observable` | Whether the current perturbation value can be included in the privileged observation vector (teacher-student, DR conditioning). **Default: `yes` for all entries unless explicitly noted.** Does not mean it is always exposed — controlled by env config. | `yes` / `no` |
| `state_variables` | Internal variables initialized at `reset()` and updated at `step()` — only for stateful perturbations | list of `name: Type[shape]` |
| `implementation` | Where it is implemented | `genesis_api` / `wrapper` |
| `feasibility_ref` | Section in `docs/00_feasibility.md` | section number |
| `risk` | Implementation risk level | `low` / `medium` / `high` |
| `priority` | Implementation priority tier | `1` (high) / `2` (medium) / `3` (low) |

### Distribution parameters reference

| Distribution | Parameters | Description |
|---|---|---|
| `uniform` | `low: float`, `high: float` | Sample ∈ [low, high] uniformly |
| `gaussian` | `mean: float`, `std: float` | Normal distribution, clipped to `bounds` |
| `beta` | `alpha: float`, `beta: float` | Beta distribution, scaled to `bounds` |
| `ou_process` | `mu: float`, `theta: float`, `sigma: float` | Ornstein-Uhlenbeck: `dx = θ(μ−x)dt + σ√dt·ε` — requires `dt` from env |
| `constant` | `value: float` | Fixed value, no sampling |

> `curriculum_scale` is not listed per entry — it applies universally.
> `observable` defaults to `yes` for all entries — not repeated per entry unless `no`.

---

## Hook signatures

Each `application_hook` defines the interface between the perturbation and the env loop.
The base class `apply()` method is dispatched to the correct hook at runtime.

| Hook | Called at | Receives | Returns | Notes |
|---|---|---|---|---|
| `reset` | `env.reset(env_ids)` | `env_ids: Tensor[k]` | `None` | Reinitializes perturbation params + state for selected envs |
| `pre_physics` | Motor perturbations — called at step [3], before `scene.step()`. Genesis setter perturbations (1.x, 8.x) are called at step [6], **after** `scene.step()` — but since Genesis setters take effect at the *next* `scene.step()`, they are semantically pre-physics for the next step. The temporal position is post-physics; the effect is pre-physics. See `04_interactions.md §3 step [6]`. | `scene: gs.Scene`, `drone: DroneEntity`, `env_state: EnvState` | `None` | Side-effect: modifies Genesis tensors or applies external forces |
| `post_physics` | After `scene.step()` | `scene: gs.Scene`, `drone: DroneEntity`, `env_state: EnvState` | `None` | Side-effect: rarely used, reserved for post-step corrections |
| `on_observation` | After physics, before returning obs | `obs: Tensor[n_envs, obs_dim]` | `obs: Tensor[n_envs, obs_dim]` | Pure function on obs tensor — must not modify in-place |
| `on_action` | Before action is forwarded to physics | `action: Tensor[n_envs, act_dim]` | `action: Tensor[n_envs, act_dim]` | Pure function on action tensor — must not modify in-place |

### EnvState

`EnvState` is a lightweight named struct passed to physics hooks, containing:

```
pos:       Tensor[n_envs, 3]   — drone position (world frame)
quat:      Tensor[n_envs, 4]   — attitude quaternion
vel:       Tensor[n_envs, 3]   — linear velocity (body frame)
ang_vel:   Tensor[n_envs, 3]   — angular velocity (body frame)
acc:       Tensor[n_envs, 3]   — body linear acceleration (m/s²) — computed by env as (vel_t - vel_{t-1}) / dt
rpm:       Tensor[n_envs, 4]   — current motor RPM
dt:        float                — effective timestep
step:      int                  — current step count within episode
```

---

## Category 1 — Robot physical dynamics

> **Hook timing note (categories 1, 5, 7, 8 — GenesisSetterPerturbation and ExternalWrenchPerturbation):**
> These perturbations are called at env step [6], **after** `scene.step()` (temporal position = post_physics).
> Genesis setters take effect at the **next** `scene.step()` call — so their *effect* is pre-physics for the next step.
> In the summary table these are marked `post_physics¹`; the `¹` footnote references this note.
> `MotorCommandPerturbation` (category 2) is genuinely called before `scene.step()` at step [3].
> See `04_interactions.md §3 steps [3] and [6]` for the canonical call sequence.

### 1.1 Mass shift

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[-0.5×m_nom, +1.0×m_nom]` (kg) |
| `nominal` | `0.0 kg` (no shift) |
| `dimension` | `scalar` per link |
| `application_hook` | `reset` (DR) or `pre_physics` (adversarial) |
| `lipschitz_k` | small (e.g. 0.01 kg/step) if dynamic |
| `implementation` | `genesis_api` — `set_links_mass_shift()` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 1.2 Center of mass shift

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[-0.05, +0.05]` m per axis |
| `nominal` | `[0, 0, 0]` m |
| `dimension` | `vector(3)` per link |
| `application_hook` | `reset` or `pre_physics` |
| `lipschitz_k` | small (e.g. 0.001 m/step) if dynamic |
| `implementation` | `genesis_api` — `set_links_COM_shift()` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 1.3 Inertia tensor (approximate)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.5×I_nom, 1.5×I_nom]` (kg·m²) |
| `nominal` | nominal inertia tensor |
| `dimension` | `scalar` (isotropic scaling) or `vector(3)` (per-axis) |
| `application_hook` | `reset` |
| `lipschitz_k` | None |
| `implementation` | `genesis_api` — approximated via mass + CoM shift (Huygens-Steiner) |
| `feasibility_ref` | §2 |
| `risk` | medium — approximation only, no direct tensor setter |
| `priority` | 2 |

### 1.4 Motor armature (rotor inertia)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.5×a_nom, 1.5×a_nom]` (kg·m²) |
| `nominal` | nominal armature value |
| `dimension` | `scalar` per DOF |
| `application_hook` | `reset` |
| `lipschitz_k` | None |
| `implementation` | `genesis_api` — `set_dofs_armature()` |
| `feasibility_ref` | §2 |
| `risk` | medium — expensive recompute, per-episode only |
| `priority` | 2 |

### 1.5 Friction ratio

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.1, 3.0]` (ratio relative to nominal) |
| `nominal` | `1.0` |
| `dimension` | `scalar` per geometry |
| `application_hook` | `reset` or `pre_physics` |
| `lipschitz_k` | 0.05/step if dynamic |
| `implementation` | `genesis_api` — `set_geoms_friction_ratio()` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 2 |

### 1.6 Position gain Kp

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.5×Kp_nom, 2.0×Kp_nom]` |
| `nominal` | nominal Kp |
| `dimension` | `scalar` per DOF |
| `application_hook` | `reset` or `pre_physics` |
| `lipschitz_k` | small fraction of range if dynamic |
| `implementation` | `genesis_api` — `set_dofs_kp()` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 2 |

### 1.7 Velocity gain Kv

Identical structure to 1.6 — `set_dofs_kv()`.

### 1.8 Joint stiffness

Identical structure to 1.6 — `set_dofs_stiffness()`.

### 1.9 Joint damping

Identical structure to 1.6 — `set_dofs_damping()`.

### 1.10 Aerodynamic drag coefficient

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.5×Cd_nom, 2.0×Cd_nom]` |
| `nominal` | nominal drag coefficient |
| `dimension` | `vector(3)` (per axis) |
| `application_hook` | `pre_physics` (applied as external force `F = -Cd × v²`) |
| `lipschitz_k` | small if dynamic |
| `implementation` | `wrapper` — `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 2 |

### 1.11 Ground effect

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `constant` (physics model, not sampled) |
| `bounds` | altitude-dependent: significant below `2×rotor_radius` |
| `nominal` | no ground effect at high altitude |
| `dimension` | `scalar` (thrust multiplier) |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | implicit (continuous w.r.t. altitude) |
| `implementation` | `wrapper` — thrust correction model |
| `feasibility_ref` | §2, §6 |
| `risk` | medium |
| `priority` | 2 |

### 1.12 Chassis geometry asymmetry (structural)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `±5%` arm length deviation per arm |
| `nominal` | symmetric geometry |
| `dimension` | `vector(4)` (one scalar per arm) |
| `application_hook` | `reset` (propagated as CoM + inertia perturbation) |
| `lipschitz_k` | None |
| `implementation` | `wrapper` — propagated as CoM + inertia perturbation |
| `feasibility_ref` | §2 |
| `risk` | medium |
| `priority` | 3 |

### 1.13 Propeller blade damage

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `beta` |
| `bounds` | efficiency ratio `[0.5, 1.0]` per blade, affects lift curve slope |
| `nominal` | `[1.0, 1.0, 1.0, 1.0]` |
| `dimension` | `vector(4)` |
| `application_hook` | `reset` — propagated as per-propeller KF scaling |
| `lipschitz_k` | None |
| `implementation` | `genesis_api` — `apply_links_external_force` with modified KF |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 3 |

### 1.14 Structural flexibility (payload / landing gear)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` |
| `bounds` | stiffness `[50, 500]` N·m/rad, damping `[0.1, 5.0]` N·m·s/rad |
| `nominal` | rigid (infinite stiffness) |
| `dimension` | `scalar` (isotropic) |
| `application_hook` | `reset` — modeled as spring-damper torque added at `pre_physics` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` — residual torque via `apply_links_external_torque` |
| `feasibility_ref` | §2 |
| `risk` | medium |
| `priority` | 3 |

### 1.15 Battery voltage sag

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float` (min initial SoC), `high: float` (max initial SoC), `discharge_rate_low: float`, `discharge_rate_high: float` |
| `bounds` | voltage ratio `[0.7, 1.0]` (× nominal voltage) |
| `nominal` | `1.0` (full charge, no sag) |
| `dimension` | `scalar` |
| `application_hook` | `pre_physics` — scales KF proportionally to voltage ratio |
| `lipschitz_k` | implicit (slow monotonic depletion) |
| `state_variables` | `soc: Tensor[n_envs]`, `discharge_rate: Tensor[n_envs]` |
| `implementation` | `wrapper` — KF scaling via `apply_links_external_force` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 1 |

---

## Category 2 — Motors

### 2.1 Thrust coefficient KF

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.5×KF_nom, 1.5×KF_nom]` |
| `nominal` | `3.16e-10` N/(RPM²) |
| `dimension` | `scalar` (shared) or `vector(4)` (per propeller) |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | small if dynamic |
| `implementation` | `genesis_api` — `drone._KF` mutation (shared) or `apply_links_external_force` (per-env) |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 1 |

### 2.2 Torque coefficient KM

Identical structure to 2.1 — `drone._KM` or external torque.

### 2.3 Per-propeller thrust asymmetry

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.7, 1.3]` ratio per propeller (relative to mean KF) |
| `nominal` | `[1.0, 1.0, 1.0, 1.0]` |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | small if dynamic |
| `implementation` | `genesis_api` — `apply_links_external_force` per propeller |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 1 |

### 2.4 Motor partial failure

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` (failure fraction in [0,1]) |
| `bounds` | `[0.0, 1.0]` (0 = full failure, 1 = nominal) per motor |
| `nominal` | `[1.0, 1.0, 1.0, 1.0]` |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | None (per_episode) or large jump allowed (failure event) |
| `implementation` | `genesis_api` — `apply_links_external_force` with scaled thrust |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 1 |

### 2.5 Motor total failure (kill)

Subset of 2.4 — one or more motors locked to `0.0`. Kept separate for discrete event handling.

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` (number of killed motors) |
| `distribution_params` | `min_killed: int = 0`, `max_killed: int = 1` |
| `bounds` | `[0, 4]` motors killed |
| `nominal` | `0` (no motor killed) |
| `dimension` | `vector(4)` — binary mask, 1 = killed |
| `application_hook` | `pre_physics` — zeroes force/torque for selected propeller links |
| `lipschitz_k` | None |
| `state_variables` | none — binary mask stored in `_current_value`, sampled at reset, constant across episode |
| `implementation` | `MotorCommandPerturbation` leaf — sets RPM to 0 for masked motors before thrust computation |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 1 |

### 2.6 Motor first-order lag (response dynamics)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float` (min τ), `high: float` (max τ) |
| `bounds` | `τ ∈ [0.01, 0.1]` s |
| `nominal` | `τ = 0.033` s (typical brushless) |
| `dimension` | `scalar` (shared τ) or `vector(4)` (per motor) |
| `application_hook` | `pre_physics` — filters commanded RPM: `rpm_actual += (rpm_cmd - rpm_actual) / τ × dt` |
| `lipschitz_k` | implicit (continuity enforced by lag model) |
| `state_variables` | `rpm_actual: Tensor[n_envs, 4]`, `tau: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 1 |

### 2.7 Motor RPM noise (commanded vs actual)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian`, `uniform` |
| `bounds` | `±5%` of commanded RPM |
| `nominal` | `0.0` noise |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | N/A (i.i.d. noise) |
| `implementation` | `wrapper` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 2 |

### 2.8 Motor saturation (RPM clipping)

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` |
| `bounds` | `[0.7×RPM_max, 1.0×RPM_max]` |
| `nominal` | `RPM_max` (nominal max from URDF) |
| `dimension` | `scalar` or `vector(4)` |
| `application_hook` | `pre_physics` — hard clip on commanded RPM |
| `lipschitz_k` | None |
| `implementation` | `wrapper` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 2 |

### 2.9 Motor wear / time-varying degradation

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `rate_low: float`, `rate_high: float` (degradation per step) |
| `bounds` | `[0.8, 1.0]` efficiency at end of episode |
| `nominal` | `1.0` (no wear) |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | implicit (slow monotonic drift) |
| `state_variables` | `efficiency: Tensor[n_envs, 4]`, `rate: Tensor[n_envs, 4]` |
| `implementation` | `wrapper` — KF scaling |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 2 |

### 2.10 Rotor imbalance (vibration)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float`, `high: float` (imbalance magnitude ratio) |
| `bounds` | imbalance ratio `[0.0, 0.05]` |
| `nominal` | `0.0` |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` (structural force) + propagates to `on_observation` (IMU noise) |
| `lipschitz_k` | implicit (sinusoidal) |
| `state_variables` | `phase: Tensor[n_envs, 4]`, `magnitude: Tensor[n_envs, 4]` |
| `implementation` | `wrapper` — sinusoidal force + correlated IMU noise |
| `feasibility_ref` | §2, §6 |
| `risk` | medium — coupling between physics and sensor layers |
| `priority` | 3 |

### 2.11 Motor back-EMF coupling

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` (back-EMF constant Ke) |
| `bounds` | `Ke ∈ [0.5×Ke_nom, 1.5×Ke_nom]` |
| `nominal` | nominal Ke (V·s/rad) |
| `dimension` | `scalar` or `vector(4)` (per motor) |
| `application_hook` | `pre_physics` — torque penalty: `τ_brake = Ke² × ω / R` |
| `lipschitz_k` | implicit (depends on ω) |
| `implementation` | `wrapper` — residual torque via `apply_links_external_torque` |
| `feasibility_ref` | §6 |
| `risk` | medium — requires BLDC model parameters |
| `priority` | 3 |

### 2.12 Motor cold-start asymmetry

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float`, `high: float` (initial KF overhead multiplier) |
| `bounds` | startup torque overhead `[1.0, 1.5]×` nominal |
| `nominal` | no overhead (warm motor) |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` — modifies effective KF during warmup phase |
| `lipschitz_k` | implicit (exponential warmup) |
| `state_variables` | `temperature: Tensor[n_envs, 4]`, `warmup_tau: float` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 3 |

### 2.13 Gyroscopic effect magnitude variation

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | rotor inertia ratio `[0.5, 1.5]×` nominal |
| `nominal` | nominal rotor moment of inertia |
| `dimension` | `scalar` |
| `application_hook` | `pre_physics` — gyroscopic torque: `τ = I_rotor × ω_rotor × ω_body` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` — external torque |
| `feasibility_ref` | §2 |
| `risk` | medium |
| `priority` | 3 |

---

## Category 3 — Temporal / Latency

### 3.1 Observation delay (fixed)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: int`, `high: int` (delay in steps) |
| `bounds` | `k ∈ [0, 10]` steps |
| `nominal` | `k = 0` (no delay) |
| `dimension` | `scalar` (integer) |
| `application_hook` | `on_observation` — returns `obs[t - k]` instead of `obs[t]` |
| `lipschitz_k` | None |
| `state_variables` | `buffer: Tensor[n_envs, max_delay, obs_dim]` (circular, max_delay = `bounds.high`), `k: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 3.2 Observation delay (variable per step)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: int`, `high: int` (delay in steps, resampled each step) |
| `bounds` | `k ∈ [0, 5]` steps |
| `nominal` | `k = 0` |
| `dimension` | `scalar` (integer) |
| `application_hook` | `on_observation` |
| `lipschitz_k` | N/A (discrete, stochastic) |
| `state_variables` | `buffer: Tensor[n_envs, max_delay, obs_dim]` (circular, max_delay = `bounds.high`) |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 3.3 Action delay (fixed)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: int`, `high: int` (delay in steps) |
| `bounds` | `d ∈ [0, 5]` steps |
| `nominal` | `d = 0` (no delay) |
| `dimension` | `scalar` (integer) |
| `application_hook` | `on_action` — applies `action[t - d]` instead of `action[t]` |
| `lipschitz_k` | None |
| `state_variables` | `buffer: Tensor[n_envs, max_delay, action_dim]` (circular, max_delay = `bounds.high`), `d: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 3.4 Action delay (variable per step)

Identical structure to 3.3 but `frequency = per_step`, `mode = dynamic`.

### 3.5 Control loop jitter (variable delta_t)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `global` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `dt ∈ [0.5×dt_nom, 2.0×dt_nom]` |
| `nominal` | `dt_nom` (base Genesis dt × substeps) |
| `dimension` | `scalar` |
| `application_hook` | `pre_physics` — controls number of Genesis substeps called |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` — substeps trick |
| `feasibility_ref` | §1 |
| `risk` | medium |
| `priority` | 2 |

### 3.6 Obs/action desynchronization

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `obs_low: int`, `obs_high: int`, `act_low: int`, `act_high: int` |
| `bounds` | obs delay `[0,5]` steps, action delay `[0,5]` steps |
| `nominal` | both `0` |
| `dimension` | `vector(2)` (integer pair) |
| `application_hook` | `on_observation` + `on_action` |
| `lipschitz_k` | N/A |
| `state_variables` | `obs_buffer: Tensor[n_envs, max_delay, obs_dim]` (circular), `act_buffer: Tensor[n_envs, max_delay, action_dim]` (circular) — owned by `RobustDroneEnv` (env-level wrapper), not a Perturbation instance; managed via `DesyncConfig` |
| `implementation` | `wrapper` — combines 3.1 + 3.3 with independent buffers |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 3.7 Packet loss (control command drop)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float`, `high: float` (drop probability range) |
| `bounds` | drop prob `[0.0, 0.3]` |
| `nominal` | prob `= 0.0` |
| `dimension` | `scalar` |
| `application_hook` | `on_action` — holds last valid action (zero-order hold) on drop |
| `lipschitz_k` | N/A |
| `state_variables` | `last_action: Tensor[n_envs, action_dim]`, `drop_prob: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 3.8 Computation overload (skipped control cycles)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `prob_low: float`, `prob_high: float`, `duration_low: int`, `duration_high: int` |
| `bounds` | skip prob `[0.0, 0.1]`, duration `[1, 5]` steps |
| `nominal` | prob `= 0.0` |
| `dimension` | `scalar` |
| `application_hook` | `on_action` — repeats last action for skipped steps |
| `lipschitz_k` | N/A |
| `state_variables` | `skip_counter: Tensor[n_envs]`, `last_action: Tensor[n_envs, action_dim]`, `skip_prob: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

---

## Category 4 — Sensors

> **Feasibility note:** all perturbations in this category reference `§3` of `00_feasibility.md` for
> Genesis state access feasibility (position, velocity, angular velocity, RPM are all readable at each step).
> For the sensor forward model (noise structure, parameter names, output shape), each entry additionally
> references the corresponding section of `docs/00b_sensor_models.md`.
> The pipeline is always: `SensorModel.forward(env_state)` → `ObservationPerturbation.apply(raw)`.

### 4.1 Gyroscope white noise

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian` |
| `bounds` | std `[0.0, 0.05]` rad/s |
| `nominal` | std `= 0.0` |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` |
| `lipschitz_k` | N/A (i.i.d.) |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 4.2 Gyroscope bias

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian`, `uniform` |
| `bounds` | `[-0.1, 0.1]` rad/s per axis |
| `nominal` | `[0, 0, 0]` rad/s |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 4.3 Gyroscope drift (random walk)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `ou_process` |
| `distribution_params` | `mu: float` (0.0), `theta: float` (mean reversion rate), `sigma: float` (noise scale) |
| `bounds` | drift magnitude `[-0.2, 0.2]` rad/s |
| `nominal` | `0.0` |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` |
| `lipschitz_k` | drift rate (e.g. 0.001 rad/s per step) |
| `state_variables` | `x_t: Tensor[n_envs, 3]` (OU process state) |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 4.4 Accelerometer noise + bias + drift

Identical structure to 4.1 / 4.2 / 4.3 but applied to linear acceleration (m/s²).
Bounds: noise std `[0.0, 0.5]` m/s², bias `[-1.0, 1.0]` m/s².

### 4.5 Sensor cross-axis sensitivity (misalignment)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian` (small rotation angles) |
| `bounds` | misalignment angle `[-5°, 5°]` per axis |
| `nominal` | identity rotation |
| `dimension` | `matrix(3,3)` (rotation matrix close to I) |
| `application_hook` | `on_observation` — multiply obs vector by rotation matrix |
| `lipschitz_k` | None |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.6 Position sensor noise (GPS / mocap)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian` |
| `bounds` | std `[0.0, 0.5]` m per axis |
| `nominal` | std `= 0.0` |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 4.7 Position sensor dropout

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `prob_low: float`, `prob_high: float`, `duration_low: int`, `duration_high: int` |
| `bounds` | dropout prob `[0.0, 0.3]`, duration `[1, 20]` steps |
| `nominal` | prob `= 0.0` |
| `dimension` | `scalar` |
| `application_hook` | `on_observation` — zeroes or holds last valid position |
| `lipschitz_k` | N/A |
| `state_variables` | `dropout_counter: Tensor[n_envs]`, `last_pos: Tensor[n_envs, 3]`, `drop_prob: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.8 Position outliers (GPS multipath)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` (spike probability + magnitude) |
| `bounds` | spike prob `[0.0, 0.05]`, magnitude `[0.5, 5.0]` m |
| `nominal` | prob `= 0.0` |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.9 Velocity estimation noise

Identical structure to 4.6 but for velocity (m/s). Bounds: std `[0.0, 0.3]` m/s.

### 4.10 Sensor quantization (discretization)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `constant` |
| `bounds` | resolution `[1e-4, 1e-2]` per unit |
| `nominal` | continuous (no quantization) |
| `dimension` | `scalar` per channel |
| `application_hook` | `on_observation` — `round(obs / res) * res` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.11 Partial observation masking (sensor channel dropout)

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` (Bernoulli per channel) |
| `bounds` | masking prob per channel `[0.0, 1.0]` |
| `nominal` | all channels visible |
| `dimension` | `vector(obs_dim)` binary mask |
| `application_hook` | `on_observation` |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.12 Magnetometer interference

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian`, `uniform` |
| `bounds` | interference magnitude `[-50, +50]` µT per axis |
| `nominal` | `[0, 0, 0]` µT |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` — corrupts raw magnetometer reading `B_raw ∈ R^3` (µT); requires `MagnetometerModel` forward model (`B_raw = R_body^T × B_earth + hard_iron + soft_iron × field + motor_interference + noise`) |
| `lipschitz_k` | None (fixed interference field) or small if dynamic |
| `implementation` | `wrapper` — additive offset on `B_raw` output of `MagnetometerModel` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 4.13 Barometer noise / pressure wash drift

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `ou_process` (drift) + `gaussian` (noise) |
| `distribution_params` | `noise_std_low: float`, `noise_std_high: float`, `mu: float` (0.0), `theta: float`, `sigma: float` |
| `bounds` | noise std `[0.0, 0.5]` m, drift `[-2.0, 2.0]` m |
| `nominal` | std `= 0.0`, drift `= 0.0` |
| `dimension` | `scalar` |
| `application_hook` | `on_observation` — corrupts altitude channel |
| `lipschitz_k` | drift rate `~0.01` m/step |
| `state_variables` | `drift: Tensor[n_envs]` (OU state), `noise_std: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 4.14 Optical flow sensor noise / divergence

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian`, `uniform` |
| `bounds` | noise std `[0.0, 0.5]` m/s, dropout prob `[0.0, 0.2]` |
| `nominal` | std `= 0.0` |
| `dimension` | `vector(2)` (horizontal velocity) |
| `application_hook` | `on_observation` |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.15 IMU vibration saturation (RPM-correlated noise)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian` (amplitude proportional to mean RPM²) |
| `bounds` | vibration gain `[0.0, 0.1]` (rad/s) / (RPM²) |
| `nominal` | gain `= 0.0` |
| `dimension` | `vector(3)` |
| `application_hook` | `on_observation` — noise std scales with current RPM |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 4.16 Clock drift between onboard modules

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float`, `high: float` (drift rate in ppm) |
| `bounds` | drift rate `[-100, +100]` ppm |
| `nominal` | `0` ppm |
| `dimension` | `scalar` |
| `application_hook` | `on_observation` — introduces sub-step timing offset, equivalent to fractional obs delay |
| `lipschitz_k` | implicit (linear accumulation) |
| `state_variables` | `phase_offset: Tensor[n_envs]` (accumulated time offset in seconds), `rate: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | medium — sub-step interpolation required |
| `priority` | 3 |

---

## Category 5 — Wind and external forces

### 5.1 Constant wind

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | magnitude `[0.0, 10.0]` m/s, direction uniform on S² |
| `nominal` | `[0, 0, 0]` m/s |
| `dimension` | `vector(3)` |
| `application_hook` | `pre_physics` — applied as external force `F = Cd × A × ρ × v²` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` — `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 5.2 Turbulence (stochastic wind)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `ou_process` |
| `distribution_params` | `mu: float` (0.0), `theta: float` (mean reversion), `sigma_low: float`, `sigma_high: float` (noise scale range) |
| `bounds` | magnitude `[0.0, 5.0]` m/s per axis |
| `nominal` | `[0, 0, 0]` m/s |
| `dimension` | `vector(3)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | implicit (OU mean-reversion limits per-step variation) |
| `state_variables` | `x_t: Tensor[n_envs, 3]` (OU state per axis), `sigma: Tensor[n_envs]` |
| `implementation` | `wrapper` — `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 5.3 Wind gust (impulse)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `prob_low: float`, `prob_high: float`, `mag_low: float`, `mag_high: float`, `duration_low: int`, `duration_high: int` |
| `bounds` | magnitude `[0.0, 15.0]` m/s, duration `[1, 20]` steps |
| `nominal` | prob `= 0.0` |
| `dimension` | `vector(3)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | N/A (event-based) |
| `state_variables` | `gust_counter: Tensor[n_envs]`, `gust_force: Tensor[n_envs, 3]`, `gust_prob: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 2 |

### 5.4 Wind shear (altitude-dependent)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` (shear gradient) |
| `bounds` | gradient `[0.0, 2.0]` (m/s)/m |
| `nominal` | `0.0` gradient |
| `dimension` | `scalar` (gradient) + `vector(3)` (direction) |
| `application_hook` | `pre_physics` — wind magnitude = f(altitude) |
| `lipschitz_k` | implicit (continuous w.r.t. altitude) |
| `implementation` | `wrapper` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 2 |

### 5.5 Adversarial wind

Same physical model as 5.1–5.4 but `mode = dynamic`, magnitude and direction controlled by an external adversarial agent each step. Bounds enforced by `lipschitz_k`.

### 5.6 Blade-vortex interaction (rotor wake ingestion)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `ou_process` |
| `distribution_params` | `mu: float` (0.0), `theta: float`, `sigma_low: float`, `sigma_high: float` |
| `bounds` | thrust perturbation `[-10%, +10%]` per rotor |
| `nominal` | no interaction |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | implicit (OU process) |
| `state_variables` | `x_t: Tensor[n_envs, 4]` (OU state per rotor), `sigma: Tensor[n_envs]` |
| `implementation` | `wrapper` — stochastic thrust perturbation per propeller |
| `feasibility_ref` | §2, §6 |
| `risk` | medium — wake model is approximate |
| `priority` | 3 |

### 5.7 Ground effect boundary transition

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `constant` (physics model) |
| `bounds` | transition zone `[1×D_rotor, 3×D_rotor]` altitude |
| `nominal` | above transition zone |
| `dimension` | `scalar` (thrust multiplier gradient) |
| `application_hook` | `pre_physics` — distinct from 1.11: captures the dynamic in/out transition |
| `lipschitz_k` | implicit (continuous w.r.t. altitude rate) |
| `implementation` | `wrapper` |
| `feasibility_ref` | §2 |
| `risk` | medium |
| `priority` | 2 |

### 5.8 Payload sway (suspended load pendulum dynamics)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `length_low: float`, `length_high: float` (cable m), `mass_low: float`, `mass_high: float` (kg) |
| `bounds` | cable length `[0.1, 1.0]` m, mass `[0.0, 0.5]` kg |
| `nominal` | no payload |
| `dimension` | `vector(4)` (pendulum state: θ_x, θ_y, θ̇_x, θ̇_y) |
| `application_hook` | `pre_physics` — reaction force on drone body |
| `lipschitz_k` | implicit (pendulum dynamics) |
| `state_variables` | `theta: Tensor[n_envs, 2]`, `theta_dot: Tensor[n_envs, 2]`, `length: Tensor[n_envs]`, `mass: Tensor[n_envs]` |
| `implementation` | `wrapper` — pendulum ODE integrated in wrapper, reaction via `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | medium |
| `priority` | 2 |

### 5.9 Proximity aerodynamic disturbance (wall / ceiling effect)

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `constant` (physics model — function of distance to surface) |
| `bounds` | lateral force up to `±0.5 N` within `0.3 m` of surface |
| `nominal` | no disturbance (open space) |
| `dimension` | `vector(3)` |
| `application_hook` | `pre_physics` — requires scene geometry query for proximity |
| `lipschitz_k` | implicit (continuous w.r.t. distance) |
| `implementation` | `wrapper` — external force via `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | medium — requires proximity sensing at wrapper level |
| `priority` | 2 |

---

## Category 6 — Actions

### 6.1 Action noise

| Field | Value |
|---|---|
| `mode` | `dynamic` |
| `frequency` | `per_step` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `gaussian`, `uniform` |
| `bounds` | std `[0.0, 0.1]` (normalized action space) |
| `nominal` | std `= 0.0` |
| `dimension` | `vector(action_dim)` |
| `application_hook` | `on_action` |
| `lipschitz_k` | N/A |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 1 |

### 6.2 Action deadzone

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` |
| `bounds` | deadzone width `[0.0, 0.1]` (normalized) |
| `nominal` | `0.0` |
| `dimension` | `scalar` or `vector(action_dim)` |
| `application_hook` | `on_action` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 6.3 Action saturation (reduced range)

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` |
| `bounds` | effective range `[0.5, 1.0]` of nominal |
| `nominal` | full range |
| `dimension` | `scalar` or `vector(action_dim)` |
| `application_hook` | `on_action` |
| `lipschitz_k` | None |
| `implementation` | `wrapper` |
| `feasibility_ref` | §3 |
| `risk` | low |
| `priority` | 2 |

### 6.4 Actuator hysteresis

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float`, `high: float` (hysteresis width, normalized RPM) |
| `bounds` | hysteresis width `[0.0, 0.05]` (normalized RPM) |
| `nominal` | `0.0` |
| `dimension` | `scalar` or `vector(4)` |
| `application_hook` | `on_action` — output differs based on command direction |
| `lipschitz_k` | None |
| `state_variables` | `last_direction: Tensor[n_envs, 4]` (sign of last RPM change per motor), `width: Tensor[n_envs, 4]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 2 |

### 6.5 ESC low-pass filtering

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | **yes** |
| `distribution` | `uniform` |
| `distribution_params` | `low: float`, `high: float` (cutoff frequency range, Hz) |
| `bounds` | cutoff freq `[5.0, 50.0]` Hz |
| `nominal` | no filtering (infinite cutoff) |
| `dimension` | `scalar` or `vector(4)` |
| `application_hook` | `on_action` — first-order IIR filter: `y += (x - y) × (2πf_c × dt)` |
| `lipschitz_k` | implicit (filter enforces continuity) |
| `state_variables` | `filtered_cmd: Tensor[n_envs, 4]`, `cutoff_freq: Tensor[n_envs]` |
| `implementation` | `wrapper` |
| `feasibility_ref` | §6 |
| `risk` | low |
| `priority` | 2 |

---

## Category 7 — Payload & Configuration

### 7.1 Payload mass uncertainty

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` (or mid-episode change event) |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[0.0, 0.5]` kg additional mass |
| `nominal` | `0.0 kg` (no payload) |
| `dimension` | `scalar` |
| `application_hook` | `reset` or event-triggered `pre_physics` |
| `lipschitz_k` | None (episodic) or step-change on delivery event |
| `implementation` | `genesis_api` — `set_links_mass_shift()` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 7.2 Payload CoM offset

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform`, `gaussian` |
| `bounds` | `[-0.1, +0.1]` m per axis (offset from body frame center) |
| `nominal` | `[0, 0, 0]` m |
| `dimension` | `vector(3)` |
| `application_hook` | `reset` |
| `lipschitz_k` | None |
| `implementation` | `genesis_api` — `set_links_COM_shift()` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 7.3 Asymmetric prop guard drag

| Field | Value |
|---|---|
| `mode` | `fixed` |
| `frequency` | `per_episode` |
| `scope` | `per_env` |
| `stateful` | no |
| `distribution` | `uniform` |
| `bounds` | drag asymmetry ratio `[0.8, 1.2]` per arm |
| `nominal` | `[1.0, 1.0, 1.0, 1.0]` |
| `dimension` | `vector(4)` |
| `application_hook` | `pre_physics` — per-arm drag scaling |
| `lipschitz_k` | None |
| `implementation` | `wrapper` — `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 3 |

---

## Category 8 — External disturbances

### 8.1 Body force disturbance

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | conditional — `no` when distribution ∈ {uniform, gaussian}; `yes` when distribution = `ou_process` (requires `OUProcess` state + `reset()`/`step()`) |
| `distribution` | `uniform`, `gaussian`, `ou_process` |
| `distribution_params` | `low: float`, `high: float` (force magnitude per axis, N) — or OU params if stochastic |
| `bounds` | `[-5.0, +5.0]` N per axis |
| `nominal` | `[0, 0, 0]` N |
| `dimension` | `vector(3)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | small if dynamic (e.g. 0.5 N/step) |
| `frame` | `world` or `local` (body frame) |
| `link_idx` | body link index (default: main body) |
| `duration_mode` | `continuous` (applied every step) or `pulse` (single step only) |
| `implementation` | `wrapper` — `apply_links_external_force` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

### 8.2 Body torque disturbance

| Field | Value |
|---|---|
| `mode` | `fixed` or `dynamic` |
| `frequency` | `per_episode` or `per_step` |
| `scope` | `per_env` |
| `stateful` | conditional — `no` when distribution ∈ {uniform, gaussian}; `yes` when distribution = `ou_process` |
| `distribution` | `uniform`, `gaussian`, `ou_process` |
| `distribution_params` | `low: float`, `high: float` (torque magnitude per axis, N·m) — or OU params if stochastic |
| `bounds` | `[-1.0, +1.0]` N·m per axis |
| `nominal` | `[0, 0, 0]` N·m |
| `dimension` | `vector(3)` |
| `application_hook` | `pre_physics` |
| `lipschitz_k` | small if dynamic (e.g. 0.05 N·m/step) |
| `frame` | `world` or `local` (body frame) |
| `link_idx` | body link index (default: main body) |
| `duration_mode` | `continuous` or `pulse` |
| `implementation` | `wrapper` — `apply_links_external_torque` |
| `feasibility_ref` | §2 |
| `risk` | low |
| `priority` | 1 |

---

## Summary table

| ID | Name | Cat. | Stateful | Hook | Impl. | Risk | Priority |
|---|---|---|---|---|---|---|---|
| 1.1 | Mass shift | Physics | no | post_physics¹ | genesis_api | low | 1 |
| 1.2 | CoM shift | Physics | no | post_physics¹ | genesis_api | low | 1 |
| 1.3 | Inertia tensor | Physics | no | reset | genesis_api | medium | 2 |
| 1.4 | Armature | Physics | no | reset | genesis_api | medium | 2 |
| 1.5 | Friction | Physics | no | post_physics¹ | genesis_api | low | 2 |
| 1.6 | Kp gain | Physics | no | post_physics¹ | genesis_api | low | 2 |
| 1.7 | Kv gain | Physics | no | post_physics¹ | genesis_api | low | 2 |
| 1.8 | Stiffness | Physics | no | post_physics¹ | genesis_api | low | 2 |
| 1.9 | Damping | Physics | no | post_physics¹ | genesis_api | low | 2 |
| 1.10 | Drag coefficient | Physics | no | post_physics¹ | wrapper | low | 2 |
| 1.11 | Ground effect | Physics | no | post_physics¹ | wrapper | medium | 2 |
| 1.12 | Chassis asymmetry | Physics | no | reset | wrapper | medium | 3 |
| 1.13 | Blade damage | Physics | no | reset | genesis_api | low | 3 |
| 1.14 | Structural flexibility | Physics | no | reset | wrapper | medium | 3 |
| 1.15 | Battery voltage sag | Physics | **yes** | post_physics¹ | wrapper | low | 1 |
| 2.1 | KF thrust coeff | Motor | no | pre_physics | genesis_api | low | 1 |
| 2.2 | KM torque coeff | Motor | no | pre_physics | genesis_api | low | 1 |
| 2.3 | Propeller asymmetry | Motor | no | pre_physics | genesis_api | low | 1 |
| 2.4 | Motor partial failure | Motor | no | pre_physics | genesis_api | low | 1 |
| 2.5 | Motor kill | Motor | no | pre_physics | wrapper | low | 1 |
| 2.6 | Motor lag (1st order) | Motor | **yes** | pre_physics | wrapper | low | 1 |
| 2.7 | RPM noise | Motor | no | pre_physics | wrapper | low | 2 |
| 2.8 | RPM saturation | Motor | no | pre_physics | wrapper | low | 2 |
| 2.9 | Motor wear | Motor | **yes** | pre_physics | wrapper | low | 2 |
| 2.10 | Rotor imbalance | Motor | **yes** | pre_physics+obs | wrapper | medium | 3 |
| 2.11 | Back-EMF coupling | Motor | no | pre_physics | wrapper | medium | 3 |
| 2.12 | Cold-start asymmetry | Motor | **yes** | pre_physics | wrapper | low | 3 |
| 2.13 | Gyroscopic variation | Motor | no | pre_physics | wrapper | medium | 3 |
| 3.1 | Obs delay (fixed) | Temporal | **yes** | on_observation | wrapper | low | 1 |
| 3.2 | Obs delay (variable) | Temporal | **yes** | on_observation | wrapper | low | 1 |
| 3.3 | Action delay (fixed) | Temporal | **yes** | on_action | wrapper | low | 1 |
| 3.4 | Action delay (variable) | Temporal | **yes** | on_action | wrapper | low | 1 |
| 3.5 | Control loop jitter | Temporal | no | pre_physics | wrapper | medium | 2 |
| 3.6 | Obs/action desync | Temporal | **yes** | on_obs+on_action | wrapper | low | 2 |
| 3.7 | Packet loss | Temporal | **yes** | on_action | wrapper | low | 1 |
| 3.8 | Skipped cycles | Temporal | **yes** | on_action | wrapper | low | 1 |
| 4.1 | Gyro white noise | Sensor | no | on_observation | wrapper | low | 1 |
| 4.2 | Gyro bias | Sensor | no | on_observation | wrapper | low | 1 |
| 4.3 | Gyro drift | Sensor | **yes** | on_observation | wrapper | low | 1 |
| 4.4 | Accel noise/bias/drift | Sensor | **yes** | on_observation | wrapper | low | 1 |
| 4.5 | Sensor misalignment | Sensor | no | on_observation | wrapper | low | 2 |
| 4.6 | Position noise | Sensor | no | on_observation | wrapper | low | 1 |
| 4.7 | Position dropout | Sensor | **yes** | on_observation | wrapper | low | 2 |
| 4.8 | Position outliers | Sensor | no | on_observation | wrapper | low | 2 |
| 4.9 | Velocity noise | Sensor | no | on_observation | wrapper | low | 1 |
| 4.10 | Quantization | Sensor | no | on_observation | wrapper | low | 2 |
| 4.11 | Channel masking | Sensor | no | on_observation | wrapper | low | 2 |
| 4.12 | Magnetometer interference | Sensor | no | on_observation | wrapper | low | 1 |
| 4.13 | Barometer drift | Sensor | **yes** | on_observation | wrapper | low | 1 |
| 4.14 | Optical flow noise | Sensor | no | on_observation | wrapper | low | 2 |
| 4.15 | IMU vibration saturation | Sensor | no | on_observation | wrapper | low | 2 |
| 4.16 | Clock drift | Sensor | **yes** | on_observation | wrapper | medium | 3 |
| 5.1 | Constant wind | Wind | no | pre_physics | wrapper | low | 1 |
| 5.2 | Turbulence (OU) | Wind | **yes** | pre_physics | wrapper | low | 1 |
| 5.3 | Wind gust | Wind | **yes** | pre_physics | wrapper | low | 2 |
| 5.4 | Wind shear | Wind | no | pre_physics | wrapper | low | 2 |
| 5.5 | Adversarial wind | Wind | **yes** | pre_physics | wrapper | low | 1 |
| 5.6 | Blade-vortex interaction | Wind | **yes** | pre_physics | wrapper | medium | 3 |
| 5.7 | Ground effect transition | Wind | no | pre_physics | wrapper | medium | 2 |
| 5.8 | Payload sway (pendulum) | Wind | **yes** | pre_physics | wrapper | medium | 2 |
| 5.9 | Wall/ceiling proximity | Wind | no | pre_physics | wrapper | medium | 2 |
| 6.1 | Action noise | Action | no | on_action | wrapper | low | 1 |
| 6.2 | Action deadzone | Action | no | on_action | wrapper | low | 2 |
| 6.3 | Action saturation | Action | no | on_action | wrapper | low | 2 |
| 6.4 | Actuator hysteresis | Action | **yes** | on_action | wrapper | low | 2 |
| 6.5 | ESC low-pass filter | Action | **yes** | on_action | wrapper | low | 2 |
| 7.1 | Payload mass | Payload | no | reset | genesis_api | low | 1 |
| 7.2 | Payload CoM offset | Payload | no | reset | genesis_api | low | 1 |
| 7.3 | Prop guard drag | Payload | no | pre_physics | wrapper | low | 3 |
| 8.1 | Body force disturbance | External | conditional | post_physics¹ | wrapper | low | 1 |
| 8.2 | Body torque disturbance | External | conditional | post_physics¹ | wrapper | low | 1 |

> ¹ **post_physics¹**: called at env step [6], after `scene.step()`. Genesis setters take effect at the *next* physics step. See Category 1 hook timing note.
