# Gymnasium Environment

**Read when**: touching files in `src/genesis_robust_rl/envs/`.

---

## Components

- `RobustDroneEnv` — base Gymnasium env wrapping Genesis Crazyflie CF2X simulation.
- `AdversarialEnv` — wrapper adding adversary action space (does NOT inherit `gym.Env`).
- `config.py` — `EnvConfig` dataclass for perturbation list, mode, curriculum settings.

## RobustDroneEnv Key Details

- Integrates all perturbation categories via hooks: physics (post-step), obs (post-obs), action (pre-step).
- Variable delta_t via substeps (perturbation 3.5) and desync (perturbation 3.6).
- `privileged_obs_dim` property — exposes perturbation state for adversary observation.
- Supports both DR and adversarial modes simultaneously.

## AdversarialEnv Key Details

- Does NOT inherit `gym.Env` — step returns 6-tuple (non-standard).
- `__getattr__` forwards to wrapped env for attribute compatibility.
- `adversary_mode="params"` raises `NotImplementedError` (future work).
- `set_value()` does NOT clip to bounds — only Lipschitz clip (by design).
- `set_value()` uses previous step's `dt` when `substeps_range` is active (documented as intentional in `docs/04_interactions.md` section 4).
- `reset(env_ids=...)` supports partial resets.

## Genesis API (v0.4.0)

- `drone.set_mass_shift()` / `drone.set_COM_shift()` / `drone.set_friction_ratio()` — entity-level.
- `scene.rigid_solver.apply_links_external_force/torque()` — solver-level.
- `scene.rigid_solver.set_drone_rpm(prop_link_idx, rpm, spin, KF, KM, invert)`.
- `gs.init()` can only be called once — no re-init.
- CF2X: 1 link (base_link), 6 DOFs (free-floating), mass=0.027kg.
- Backend CPU works on Mac M4 Pro; no MPS backend; CUDA for GPU.

## Cross-refs

- UP: `constraints.md` #2 (design before code), #15 (unit tests required)
- UP: `architecture.md` "Data Flow: Training Loop"
- PEER: `topics/perturbation_engine.md`, `topics/adversarial_training.md`
