# ROADMAP — genesis-robust-rl

## How to use this file
At the start of each session, read this file to locate the current milestone.
Update status when starting (`in_progress`) and when done (`completed`).

**Design review rule:** before marking any design doc `[x]`, launch 2–3 parallel review subagents
(see `CLAUDE.md` → "Design Review Protocol"). Resolve all blocking issues first.

## Status legend
- `[ ]` not started
- `[~]` in progress
- `[x]` completed

---

## Phase 0 — Project Setup
- [x] Create GitHub repo
- [x] Clone locally, init CLAUDE.md + ROADMAP.md
- [x] Init Python project (pyproject.toml, UV, src layout)
- [x] Setup CI (GitHub Actions: lint + tests)

## Phase 0.5 — Feasibility Study
- [x] `docs/00_feasibility.md` — Genesis API feasibility verified and documented

## Phase 1 — Design (no code)
- [x] `docs/01_perturbations_catalog.md` — complete perturbation catalog validated (must ref 00_feasibility.md)
- [x] `docs/00b_sensor_models.md` — phenomenological sensor forward models (gyro, accel, mag, baro, GPS, optical flow)
- [x] `docs/02_class_design.md` — class hierarchy validated (includes SensorModel layer)
- [x] `docs/05_test_conventions.md` — mandatory test contract for subagents (unit + integration + perf)
- [x] `docs/03_api_design.md` — Gym + adversarial API validated
- [x] `docs/04_interactions.md` — component interactions validated

## Phase 1.5 — Test Infrastructure (prerequisite for parallel implementation)
> Goal: give each subagent a self-contained test harness so it can autonomously validate its work.
> Each subagent receives: `02_class_design.md` + `05_test_conventions.md` + `06_test_infrastructure.md` + catalog entries for its category.
> A subagent's work is DONE only when all its tests pass (unit + integration + perf).

- [x] `docs/06_test_infrastructure.md` — test infrastructure design validated
- [x] `pytest` + `pytest-cov` configured in `pyproject.toml`
- [x] `tests/conftest.py` — shared fixtures:
  - `mock_env_state(n_envs)` — realistic `EnvState` tensor batch
  - `mock_scene()` — Genesis scene stub with patched setters
  - `assert_lipschitz(perturbation, n_steps)` — helper to verify Lipschitz over a trajectory
- [x] Placeholder test passing in CI (`tests/test_placeholder.py`)

## Phase 2 — Perturbation Engine (parallelizable by category)
> **Prerequisite (sequential):** implement `base.py` first — all parallel categories depend on it.
>
> **Pipeline per category (5 phases):**
> 1. **Implement** (`/implement-category N`) — code + tests + doc + plots (sequential per perturbation)
> 2. **Review** (3 agents //) — correctness, code quality, perf patterns → fix BLOCKING issues
> 3. **Measure** — P6 overhead test with real Genesis CF2X → overhead report
> 4. **Optimize** (`/judge-category N`) — propose optimizations for perturbations >100% overhead
> 5. **Judge** — synthesis report → user decides which optimizations to apply
>
> Phases 1–3 are autonomous. Phases 4–5 require user decisions.
> A category is DONE when: all tests pass, all BLOCKING fixed, all ❌FAIL optimized under 200%.

- [x] Base `Perturbation` class + `OUProcess` + `DelayBuffer` + registry (`src/genesis_robust_rl/perturbations/base.py`)
- [x] **[parallel]** Category 1 — Physics (GenesisSetterPerturbation, ExternalWrenchPerturbation) — 15 perturbations (15/15 done: 1.1–1.15 ✓)
- [ ] **[parallel]** Category 2 — Motor (MotorCommandPerturbation leaves) — 13 perturbations
- [ ] **[parallel]** Category 3 — Temporal (ObsDelayBuffer, ActionDelayBuffer) — 8 perturbations
- [ ] **[parallel]** Category 4 — Sensor (SensorModel + ObservationPerturbation leaves) — 16 perturbations
- [ ] **[parallel]** Category 5 — Wind (ExternalWrenchPerturbation leaves) — 9 perturbations
- [ ] **[parallel]** Category 6 — Action (ActionPerturbation leaves) — 5 perturbations
- [ ] **[parallel]** Category 7 — Payload (GenesisSetterPerturbation leaves) — 3 perturbations
- [ ] **[parallel]** Category 8 — External force/torque (ExternalWrenchPerturbation leaves) — 2 perturbations
- [ ] Perturbation registry + auto-doc API

## Phase 3 — Base Gym Environment
- [ ] Genesis quadrotor integration
- [ ] `RobustDroneEnv` base class (Gymnasium-compatible)
- [ ] Perturbation application hooks (physics, obs, action)
- [ ] Variable delta_t integration
- [ ] Domain randomization mode (sample from distributions)
- [ ] Unit tests

## Phase 4 — Adversarial API
- [ ] `AdversarialEnv` wrapper — adversary sets perturbations each step
- [ ] Action space for adversary (bounded, Lipschitz-constrained)
- [ ] Minimax training loop utility
- [ ] Adversarial agent base class
- [ ] Unit tests

## Phase 5 — Robust RL Contribution
- [ ] Curriculum over perturbation intensity
- [ ] Algorithm integration (PPO + adversarial)
- [ ] Benchmarks vs DR classical / minimax / DRRL
- [ ] Sim-to-real transfer experiments

## Phase 6 — Documentation & Release
- [ ] README with quickstart
- [ ] Full API reference
- [ ] Example notebooks
- [ ] PyPI release

## Phase 7 — GitHub Pages Site
- [ ] MkDocs + Material theme setup
- [ ] `mkdocs.yml` config — nav, theme, plugins
- [ ] Migrate `docs/impl/` technical docs into MkDocs structure
- [ ] GitHub Actions workflow: push to `main` → build → deploy to GitHub Pages
- [ ] Live site at `paul-antoineletolguenec.github.io/genesis-robust-rl`

---

## Current milestone
**Phase 2 — Perturbation Engine** — in progress

**base.py done + audited.** 5 blocking bugs fixed (tick DR guard, update_params Lipschitz, wrench_type dispatch, PerturbationRegistry). Genesis v0.4.0 migration: `scene.solver` → `scene.rigid_solver` in all code + tests.
Cat 1 : 15/15 done (1.1–1.15 ✓, 741 passed, 234 skipped). 45 PNG generated.
P6 overhead test added and passing (combined overhead +3.9% < 5% limit, Crazyflie CF2X, n_envs=16).

**Immediate next action:** push → CI check → implement Cat 2–8 in parallel.

**Key design constraints to respect:**
- `MotorCommandPerturbation` inherits from `Perturbation` (not `PhysicsPerturbation`)
- Physics perturbations called at step [6] post_physics¹ — see `01_perturbations_catalog.md` §Category 1 note
- `OUProcess.step()` must use torch ops only (no numpy)
- `DelayBuffer` uses circular `Tensor[n_envs, max_delay, dim]` — no Python deque
- `tick(is_reset, env_ids=None)` — env_ids is optional (None = all envs)
