# CLAUDE.md — Project Rules for genesis-robust-rl

## Language
- Chat in **French**, code comments and docs in **English**

## Workflow Rules (MANDATORY)
1. **Always read `ROADMAP.md` at the start of each session** to locate current milestone
2. **Never write code before design is validated** — design lives in `docs/`
3. **Update milestone status in `ROADMAP.md`** before moving to the next one
4. **Propose a minimal textual plan** and wait for explicit user approval before editing code
5. **One milestone at a time** — do not skip ahead

## Design-First Protocol
- All structural decisions are documented in `docs/` before implementation
- If a design decision is uncertain, open it as a question to the user
- `docs/01_perturbations_catalog.md` must be complete and validated before any env code
- `docs/02_class_design.md` must be validated before any class is written

## Feasibility Rule (MANDATORY)
- `docs/00_feasibility.md` is the Genesis API feasibility reference — read it before any design decision
- **Every perturbation** in `docs/01_perturbations_catalog.md` MUST reference a finding in `00_feasibility.md`
- Any perturbation marked `Unknown` or `High` risk in `00_feasibility.md` must be explicitly flagged in the catalog
- Do not design perturbations that have no feasibility entry — add one first

## Code Standards
- **Python with UV** (`uv add`, `uv run`)
- **Type hints** in all function signatures
- **No type in variable names** — name represents content
- Minimal, production-quality code — no over-engineering
- All public classes and functions must have a docstring

## Testing
- Every new class or function must have a corresponding unit test in `tests/`
- Run tests with `uv run pytest`
- CI must pass before merging any PR

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
  docs/
    01_perturbations_catalog.md
    02_class_design.md
    03_api_design.md
    04_interactions.md
  ROADMAP.md
  pyproject.toml
  .github/workflows/ci.yml
```
