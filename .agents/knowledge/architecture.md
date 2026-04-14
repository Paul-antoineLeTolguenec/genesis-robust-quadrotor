# genesis-robust-rl Architecture Overview

## Module Dependency Graph

```
                    +-----------------+
                    |  adversarial/   |
                    | (training loop, |
                    |  PPO, curriculum)|
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    |     envs/       |
                    | (RobustDroneEnv,|
                    |  AdversarialEnv)|
                    +--------+--------+
                             |
                    +--------+--------+
                    |                 |
                    v                 v
          +-----------------+  +-----------------+
          | perturbations/  |  | sensor_models.py|
          | (base + 8 cats) |  | (6 fwd models)  |
          +-----------------+  +-----------------+
                    |
                    v
          +-----------------+
          |    Genesis API  |
          | (scene, solver) |
          +-----------------+
```

### Key Dependency Rules

| Module | Depends On | Depended By |
|--------|-----------|-------------|
| `perturbations/` | torch, numpy | `envs/` |
| `sensor_models.py` | torch, numpy | `envs/` (via cat 4) |
| `envs/` | perturbations, sensor_models, Genesis, Gymnasium | `adversarial/` |
| `adversarial/` | envs, torch | (user code) |

---

## Data Flow: Training Loop

```
1. train() loop
   |
   2. env.reset() / env.step(action)
      |
      3. RobustDroneEnv
         |-- tick(is_reset) on all perturbations --> sample/advance stateful
         |-- apply() physics perturbations --> Genesis setters / external wrench
         |-- scene.step()
         |-- apply() sensor perturbations --> corrupt observations
         |-- apply() action perturbations --> corrupt motor commands
         |
      4. AdversarialEnv (wrapper)
         |-- adversary observes privileged_obs
         |-- adversary sets perturbation values (set_value)
         |-- Lipschitz clipping enforced
         |
   5. PPOAgent.collect_rollouts() / update()
      |-- protagonist optimizes reward
      |-- adversary minimizes reward (RARL/RAP modes)
```

---

## Key Design Patterns

### Perturbation Hierarchy
`Perturbation` (abstract) -> `PhysicsPerturbation` -> `GenesisSetterPerturbation` | `ExternalWrenchPerturbation`.
Separate leaves: `MotorCommandPerturbation`, `ObservationPerturbation`, `ActionPerturbation`.
Each perturbation has `tick()` (sample) and `apply()` (effect).

### Two Modes: DR vs Adversarial
`PerturbationMode.DOMAIN_RANDOMIZATION`: tick() samples from distributions.
`PerturbationMode.ADVERSARIAL`: tick() never samples; values set externally via `set_value()`.

### Registry Pattern
`PerturbationRegistry` singleton — `perturbation_registry.build("id", n_envs, dt, ...)`.
All 69 perturbations auto-registered via class decorator.

### Sensor Forward Models
`SensorModel` is a SEPARATE hierarchy (not a Perturbation). Outputs raw readings only (no fusion assumption). Category 4 perturbations corrupt sensor model outputs.

---

## Configuration

- `envs/config.py` — `EnvConfig` dataclass: perturbation list, mode, curriculum settings
- `pyproject.toml` — project deps, ruff/pytest config
- `.github/workflows/ci.yml` — lint + test on push/PR to main

---

## Project Structure

```
genesis-robust-rl/
  src/genesis_robust_rl/
    perturbations/     # perturbation catalog + dynamics (base + 8 categories)
    envs/              # gym environments (RobustDroneEnv, AdversarialEnv)
    adversarial/       # adversarial agent API + training loop (DR/RARL/RAP)
    sensor_models.py   # phenomenological sensor forward models (6 models)
    utils/
  tests/
    category_1_physics/ ... category_8_external/  # per-category unit tests
    adversarial/       # adversarial module tests
    envs/              # env tests
    integration/       # P6 overhead tests (Genesis required)
    conftest.py        # shared fixtures (mock_env_state, mock_scene, assert_lipschitz)
  docs/
    00_feasibility.md          # Genesis API feasibility reference
    00b_sensor_models.md       # sensor forward model reference
    01_perturbations_catalog.md
    02_class_design.md
    03_api_design.md
    04_interactions.md
    05_test_conventions.md
    06_test_infrastructure.md
    07_adversarial_training.md
    impl/              # per-perturbation docs + plots
      assets/          # generated PNG plots
  .claude/
    commands/          # slash-command skills (implement-*, judge-*)
  .agents/
    knowledge/         # agent docs (this file, constraints, topics, routing)
    skills/            # skills index
  AGENTS.md            # agent entry point
  CLAUDE.md            # pointer to AGENTS.md
  ROADMAP.md           # milestone tracker (always read at session start)
  pyproject.toml
  .github/workflows/ci.yml
```

---

## Extension Points

- **New perturbation**: see skill `/implement-perturbation` or topic `topics/perturbation_engine.md`
- **New sensor model**: add to `sensor_models.py`, follow `00b_sensor_models.md` spec
- **New training mode**: extend `adversarial/training_loop.py`, add to `TrainingMode` enum
