# Hard Constraints

Quick index: **#1-5** Workflow | **#6-8** Design & Feasibility | **#9-11** Sensor Rules | **#12-14** Code Standards | **#15-17** Testing | **#18-19** Performance | **#20-21** Documentation | **#22-24** Git

These constraints MUST NOT be violated. Consult this file before making any code changes.

---

## Workflow (1-5)

### 1. Read ROADMAP.md at Session Start
Always read `ROADMAP.md` to locate the current milestone before doing any work.

### 2. Design Before Code
All structural decisions are documented in `docs/` before implementation. Never write code before the relevant design doc is validated. Specific gates:
- `docs/01_perturbations_catalog.md` must be complete and validated before any env code.
- `docs/02_class_design.md` must be validated before any class is written.
- If a design decision is uncertain, open it as a question to the user — do not guess.

### 3. Plan Before Implement
Propose a minimal textual plan and wait for explicit user approval before editing code.

### 4. One Milestone at a Time
Follow the phase order in `ROADMAP.md`. Do not skip ahead or work on future phases.

### 5. Update ROADMAP.md and MEMORY.md After Every Change
After completing a perturbation or milestone:
- Update `ROADMAP.md`: item status (`[~]`/`[x]`), counts, and "Current milestone" section.
- Update `MEMORY.md`: current state block.

---

## Design & Feasibility (6-8)

### 6. Feasibility Reference
`docs/00_feasibility.md` is the Genesis API feasibility reference. Read it before any design decision. Every perturbation in the catalog MUST reference a finding in this doc.

### 7. Sensor Model Reference
`docs/00b_sensor_models.md` is the sensor forward model reference. Read it before designing any sensor perturbation (category 4). Every sensor perturbation must reference a model in this doc.

### 8. Flag Unknown/High Risk
Any perturbation marked `Unknown` or `High` risk in `00_feasibility.md` must be explicitly flagged in the catalog. Do not design perturbations with no feasibility entry — add one first.

---

## Sensor Rules (9-11)

### 9. No Fusion Assumption
Sensor models output raw readings only (uT, rad/s, m/s^2, Pa...). Never assume Kalman filter, transformer, or any estimator downstream.

### 10. No User Pipeline Assumption
Sensor perturbations corrupt the output of a `SensorModel` forward model, not a fused state estimate.

### 11. Separate Sensor Hierarchy
`SensorModel` is a SEPARATE class hierarchy from `Perturbation`. Do not merge them.

---

## Code Standards (12-14)

### 12. Python with UV
Use `uv add` for dependencies, `uv run` for execution. No pip.

### 13. Type Hints Everywhere
Type hints in all function signatures. No type in variable names — name represents content.

### 14. Docstrings on Public API
All public classes and functions must have a docstring. Code comments in English.

### 14b. Minimal, Production-Quality Code
Write minimal, production-quality code — no over-engineering. Do not add abstractions, helpers, or validation beyond what the task requires.

---

## Testing (15-17)

### 15. Unit Tests Required
Every new class or function must have a corresponding unit test in `tests/`. Run with `uv run pytest`.

### 16. Perturbation Perf Tests
Every perturbation must have CPU perf tests (tick + apply, n_envs=1 and 512). Fails if thresholds exceeded.

### 17. CI Must Pass
CI (lint + tests) must pass before merging any PR. No exceptions.

---

## Performance (18-19)

### 18. P6 Overhead < 5%
Perturbation logic (sample + curriculum + clamp) must add < 5% overhead vs passing fixed tensors to the same Genesis setters. Test: `tests/integration/test_overhead_genesis.py`.

### 19. P6 Methodology
Uses real Crazyflie CF2X URDF, n_envs=16, median of 5 rounds x 100 steps. Baseline = fixed tensors -> Genesis setter -> step. Perturbed = sample+clip -> Genesis setter -> step. Delta measures ONLY our code, not Genesis API cost. Must pass locally before any push/PR.

---

## Documentation (20-21)

### 20. Doc Per Perturbation
After implementing each perturbation: add a section in `docs/impl/category_N_*.md` with formal definition (equation), parameter table, catalog reference.

### 21. Three Plots Per Perturbation
Add 3 Plotly graphs in `docs/impl/plot_category_N.py`: curriculum effect, per-env DR, perf overhead vs n_envs. Export as PNG to `docs/impl/assets/` via `pio.write_image()`. Run immediately after writing the script.

---

## Git (22-24)

### 22. Branch Per Feature
Branch naming: `feat/<description>`. One branch per feature or milestone.

### 23. PRs Only
Never push directly to `main`. Always use pull requests.

### 24. No Force Push
Never force push. Commit messages: concise, imperative, English.
