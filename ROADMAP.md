# ROADMAP — genesis-robust-rl

## How to use this file
At the start of each session, read this file to locate the current milestone.
Update status when starting (`in_progress`) and when done (`completed`).

## Status legend
- `[ ]` not started
- `[~]` in progress
- `[x]` completed

---

## Phase 0 — Project Setup
- [x] Create GitHub repo
- [x] Clone locally, init CLAUDE.md + ROADMAP.md
- [ ] Init Python project (pyproject.toml, UV, src layout)
- [ ] Setup CI (GitHub Actions: lint + tests)

## Phase 1 — Design (no code)
- [ ] `docs/01_perturbations_catalog.md` — complete perturbation catalog validated
- [ ] `docs/02_class_design.md` — class hierarchy validated
- [ ] `docs/03_api_design.md` — Gym + adversarial API validated
- [ ] `docs/04_interactions.md` — component interactions validated

## Phase 2 — Perturbation Engine
- [ ] Base `Perturbation` class (bounds, modes, distributions)
- [ ] Rectangular dynamics
- [ ] Lipschitz dynamics (with delta_t dependency)
- [ ] Observation delay buffer (delta_k)
- [ ] Distribution sampling (gaussian, beta, uniform)
- [ ] Perturbation registry + auto-doc API
- [ ] Unit tests for all perturbation types

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

---

## Current milestone
**Phase 0 — Project Setup** (in progress)
