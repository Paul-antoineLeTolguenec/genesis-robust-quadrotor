# ROADMAP — genesis-robust-rl

## How to use this file
At the start of each session, read this file to locate the current milestone.
Update status when starting (`in_progress`) and when done (`completed`).

**Design review rule:** before marking any design doc `[x]`, launch 2–3 parallel review subagents
(see `.agents/knowledge/topics/design_review.md`). Resolve all blocking issues first.

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
- [x] **[parallel]** Category 2 — Motor (MotorCommandPerturbation + ExternalWrenchPerturbation leaves) — 13 perturbations (13/13 done: 2.1–2.13 ✓)
- [x] **[parallel]** Category 3 — Temporal (ObsDelayBuffer, ActionDelayBuffer) — 6/6 perturbations done (3.1–3.4, 3.7, 3.8 ✓; 3.5, 3.6 are env-level wrappers → Phase 3)
- [x] **[parallel]** Category 4 — Sensor (SensorModel + ObservationPerturbation leaves) — 16/16 perturbations done (4.1–4.16 ✓) + 6 SensorModel forward models
- [x] **[parallel]** Category 5 — Wind (ExternalWrenchPerturbation leaves) — 9/9 perturbations done (5.1–5.9 ✓)
- [x] **[parallel]** Category 6 — Action (ActionPerturbation leaves) — 5/5 perturbations done (6.1–6.5 ✓)
- [x] **[parallel]** Category 7 — Payload (GenesisSetterPerturbation + ExternalWrenchPerturbation leaves) — 3/3 perturbations done (7.1–7.3 ✓)
- [x] **[parallel]** Category 8 — External force/torque (ExternalWrenchPerturbation leaves) — 2/2 perturbations done (8.1–8.2 ✓)
## Phase 3 — Base Gym Environment
- [x] Genesis quadrotor integration
- [x] `RobustDroneEnv` base class (Gymnasium-compatible)
- [x] Perturbation application hooks (physics, obs, action)
- [x] Variable delta_t integration (substeps 3.5 + desync 3.6)
- [x] Domain randomization mode (sample from distributions)
- [x] Adversarial mode API (set_perturbation_values, update_params, curriculum)
- [x] Unit tests (101 passed, 0 skipped)

## Phase 4 — Adversarial API
- [x] `AdversarialEnv` wrapper — adversary sets perturbations each step
- [x] Action space for adversary (bounded, Lipschitz-constrained)
- [x] Minimax training loop utility (absorbed into Phase 5)
- [x] Adversarial agent base class (absorbed into Phase 5)
- [x] Unit tests (37 passed)

## Phase 5 — Robust RL Contribution
- [x] `AdversarialAgent` Protocol + `RolloutData` (`adversarial/protocol.py`)
- [x] `ActorCritic` shared MLP network (`adversarial/networks.py`)
- [x] `PPOAgent` reference implementation (`adversarial/ppo_agent.py`)
- [x] `CurriculumScheduler` callback (`adversarial/curriculum.py`)
- [x] `train()` loop — DR / RARL / RAP modes (`adversarial/training_loop.py`)
- [x] `privileged_obs_dim` property on `RobustDroneEnv`
- [x] `env_ids` support on `AdversarialEnv.reset()`
- [x] Design doc `docs/07_adversarial_training.md` (reviewed, 4 BLOCKING fixed)
- [x] Unit tests (32 passed, 3144 total suite)
- [ ] Benchmarks vs DR classical / minimax / DRRL
- [ ] Sim-to-real transfer experiments

## Phase 6 — Documentation & Release
- [x] README showcase (badges, plots, quickstart) — PR #13
- [x] Genesis hover demo GIFs (baseline vs wind_gust) — PR #14
- [x] Scientific plot refactor — PR #15 merged (shared framework + 123 viz PNG across 8 cats)
- [x] Genesis perf regen — PR #16 merged (69/69 perturbations, 207 artifacts)
  - [x] Shared `docs/impl/_perf_framework.py` — Genesis init + build_scene + measure_overhead
  - [x] `plot_perf_cat{1..8}.py` — one script per category, unified methodology
  - [x] 69 × (`_perf.csv` + `_perf.meta.json` + `_perf.png`) under `docs/impl/{data,assets}/`
- [~] README rewrite with benchmark methodology + hardware-annotated plots (feat/phase6-readme-v2)
  - [x] `docs/impl/plot_hero_overhead.py` + `hero_overhead.{csv,meta.json,png}` (69 perturbations aggregate)
  - [x] Honest per-class overhead table replacing the "< 5 %" claim
  - [x] Benchmark methodology section (baseline/perturbed, warmup, rounds, reproducibility)
- [ ] Perturbation registry + auto-doc API
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
**Phase 6 — Documentation & Release** — IN PROGRESS

**PR #15 merged**: scientific plot refactor — shared framework + 123 visualisation PNG/CSV/meta
across 69 perturbations (Apple M4 Pro, CPU). Identity-cache fix on
`GenesisSetterPerturbation.apply()` brought `mass_shift` from +60% → +1% overhead.

**Genesis perf regen (feat/phase6-perf-regen) — 69/69 perturbations measured:**
- Shared `_perf_framework.py` factors the canonical methodology
  (warmup=30, rounds=5, steps=100, reset_every=20, n_envs ∈ {1, 4, 16, 64, 128})
  across four apply kinds (physics / motor / obs / action).
- One `plot_perf_cat<N>.py` per category → 69 × (`_perf.csv` + `_perf.meta.json` + `_perf.png`).
- Key observations:
  - GenesisSetter with identity cache: +1–8% (mass_shift, com_shift, payload_*, *_gain_*).
  - ExternalWrench (no identity cache): +35–120% (all wind, all payload_drag, Cat 2 back-emf).
  - InertiaTensor / ChassisGeometryAsymmetry trigger two setters per step → +100–130%.
  - MotorCommand / ObservationPerturbation / ActionPerturbation: +0–30% typically.

**README rewrite (feat/phase6-readme-v2):**
- New hero plot `docs/impl/assets/hero_overhead.png` aggregates the 69 perturbations
  (median + IQR @ n_envs=16), colored by category, with the 5 % budget line.
- The "< 5 % overhead" marketing claim was dropped in favour of a per-class table
  that reports the real cost of each apply signature.
- New **Benchmark methodology** section documents the harness
  (baseline = `scene.step()` only, perturbed = `tick + apply + step`, warmup=30,
  rounds=5, steps=100, reset_every=20, n_envs ∈ {1, 4, 16, 64, 128}) and how to
  reproduce any number from the CSV + meta JSON pair that ships with each plot.

**Remaining Phase 6 items:**
- Perturbation registry + auto-doc API.
- Full API reference.
- Example notebooks.
- PyPI release.

**Phase 5 deferred items:**
- Benchmarks vs DR classical / minimax / DRRL.
- Sim-to-real transfer experiments (requires hardware).
