# CLAUDE.md — Project Rules for genesis-robust-rl

## Language
- Chat in **French**, code comments and docs in **English**

## Workflow Rules (MANDATORY)
1. **Always read `ROADMAP.md` at the start of each session** to locate current milestone
2. **Never write code before design is validated** — design lives in `docs/`
3. **Update `ROADMAP.md` after every perturbation** — update the item status (`[~]`/`[x]`), the count (N/15 done, N passed, M skipped), and the "Current milestone" section. Also update `MEMORY.md` (current state block).
4. **Propose a minimal textual plan** and wait for explicit user approval before editing code
5. **One milestone at a time** — do not skip ahead

## Design Review Protocol (MANDATORY)

After completing any design document in `docs/`, **before marking it `[x]` in ROADMAP.md**:

1. Launch **2 or 3 review subagents in parallel**, each with the following context:
   - The document just written (full content)
   - All prerequisite documents it references (e.g. `00_feasibility.md`, `01_perturbations_catalog.md`, `02_class_design.md`)
   - A precise review prompt — see template below

2. **Synthesis**: collect all blocking issues across reviewers, present them to the user, and iterate until resolved before moving forward.

### Review subagent prompt template

```
You are a senior robotics/ML engineer reviewing a design document for a robustness RL library for quadrotors.

## Project context
- Simulator: Genesis (Python, GPU-batched, n_envs parallel envs)
- Goal: perturbation engine for domain randomization and adversarial training of drone policies
- Design-first: no code is written before all design docs are validated
- Key constraint: no fusion assumption — sensor models output raw readings only

## Documents to read (provided below)
[list each document with full content]

## Review instructions
Check for:
1. **Internal consistency** — does the document contradict itself?
2. **Cross-document consistency** — does it contradict any of the prerequisite documents?
3. **Completeness** — are there gaps, undefined terms, or missing cases?
4. **Implementability** — is every interface precisely enough defined to be implemented without ambiguity?
5. **Edge cases** — are reset, per-env vs global scope, stateful vs stateless, adversarial vs DR all handled?

Return: a numbered list of **blocking issues** (must fix before proceeding) and **non-blocking suggestions** (nice to have). Be concise.
```

## Design-First Protocol
- All structural decisions are documented in `docs/` before implementation
- If a design decision is uncertain, open it as a question to the user
- `docs/01_perturbations_catalog.md` must be complete and validated before any env code
- `docs/02_class_design.md` must be validated before any class is written

## Feasibility Rule (MANDATORY)
- `docs/00_feasibility.md` is the Genesis API feasibility reference — read it before any design decision
- `docs/00b_sensor_models.md` is the sensor forward model reference — read it before designing any sensor perturbation (cat. 4)
- **Every perturbation** in `docs/01_perturbations_catalog.md` MUST reference a finding in `00_feasibility.md`
- **Every sensor perturbation** must reference a model in `00b_sensor_models.md`
- Any perturbation marked `Unknown` or `High` risk in `00_feasibility.md` must be explicitly flagged in the catalog
- Do not design perturbations that have no feasibility entry — add one first

## Sensor Design Rule (MANDATORY)
- **No fusion assumption** — sensor models output raw readings only (µT, rad/s, m/s², Pa...)
- **No assumption on user pipeline** — never assume Kalman filter, transformer, or any estimator
- Sensor perturbations corrupt the output of a `SensorModel` forward model, not a fused state estimate

## Code Standards
- **Python with UV** (`uv add`, `uv run`)
- **Type hints** in all function signatures
- **No type in variable names** — name represents content
- Minimal, production-quality code — no over-engineering
- All public classes and functions must have a docstring

## Testing
- Every new class or function must have a corresponding unit test in `tests/`
- Every perturbation must have CPU perf tests (tick + apply, n_envs=1 and 512) — fails if thresholds exceeded
- Run tests with `uv run pytest`
- CI must pass before merging any PR

## Performance Rule (MANDATORY)
- **P6 — Relative overhead < 5%**: perturbation logic (sample + curriculum + clamp) must add < 5% overhead vs passing fixed tensors to the same Genesis setters
- Test: `tests/integration/test_overhead_genesis.py` — uses real Crazyflie CF2X URDF, n_envs=16, median of 5 rounds
- Methodology: baseline = fixed tensors → Genesis setter → step; perturbed = sample+clip → Genesis setter → step. Delta measures ONLY our code, not Genesis API cost.
- **Must pass locally before any push/PR** (requires Genesis installed)
- Run: `uv run pytest tests/integration/test_overhead_genesis.py -v -s`

## Documentation Rule (MANDATORY per perturbation)
After implementing each perturbation:
1. Add a section in `docs/impl/category_N_*.md` with: formal definition (equation), parameter table, catalog reference
2. Add 3 Plotly graphs in `docs/impl/plot_category_N.py`: curriculum effect, per-env DR, perf overhead vs n_envs
3. Graphs exported as PNG to `docs/impl/assets/` via `pio.write_image()` — **run immediately** with `uv run python docs/impl/plot_category_N.py` (kaleido works directly; if Chromium missing, call `uv run python -c "from kaleido import get_chrome_sync; get_chrome_sync()"` first)
4. Doc title must be descriptive — never "Category N — ..."

## Git
- Branch per feature/milestone: `feat/milestone-N-description`
- Commit messages: concise, imperative, English
- Never push directly to `main` — use PRs
- Never force push

## Project Structure
```
genesis-robust-rl/
  src/genesis_robust_rl/
    perturbations/     # perturbation catalog + dynamics
    envs/              # gym environments
    adversarial/       # adversarial agent API
    utils/
  tests/
    integration/       # P6 overhead tests (Genesis required)
  docs/
    00_feasibility.md
    00b_sensor_models.md
    01_perturbations_catalog.md
    02_class_design.md
    03_api_design.md
    04_interactions.md
    05_test_conventions.md
    06_test_infrastructure.md
    impl/              # per-perturbation docs + plots
      assets/          # generated PNG plots
  .claude/
    commands/          # implement-perturbation skill
  ROADMAP.md
  pyproject.toml
  .github/workflows/ci.yml
```
