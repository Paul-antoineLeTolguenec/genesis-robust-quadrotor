# genesis-robust-rl Development Guide

## Project Overview

Robust RL library for quadrotor drones (Crazyflie CF2X) using the Genesis simulator. Provides a perturbation engine (69 perturbations, 8 categories) for domain randomization and adversarial training of drone policies, exposed through a Gymnasium-compatible environment.

- **Stack**: Python >=3.10 | Genesis >=0.4.0 | Gymnasium | PyTorch | NumPy/SciPy
- **License**: unlicensed (private) | **Package manager**: UV

**Language**: Chat in French, code comments and docs in English.

## Context Loading

On session start, read **Tier 1**:
- `ROADMAP.md` — current milestone and phase status
- `.agents/knowledge/architecture.md` — module graph, data flow, key patterns
- `.agents/knowledge/constraints.md` — hard rules, indexed by category

**Tier 2**: topic docs triggered by change area. See `.agents/knowledge/README.md`.

## Core Operating Principles

1. **Constraints first** — read `constraints.md` before any change; search codebase before guessing.
2. **Design before code** — structural decisions live in `docs/`; never implement without validated design.
3. **One milestone at a time** — check `ROADMAP.md`; do not skip ahead or work on future phases.
4. **Plan before implement** — propose a minimal textual plan and wait for explicit user approval.
5. **Challenge first, execute second** — spot a simpler alternative? Raise before executing.

## Development Commands

```bash
uv sync --extra dev          # Install deps
uv run pytest                # Run tests (unit + integration, no Genesis)
uv run pytest -m genesis     # Run Genesis-dependent tests (local only)
uv run pytest tests/integration/test_overhead_genesis.py -v -s  # P6 overhead
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
```

## Available Skills (`.claude/commands/`)

| Skill | Purpose | Use When |
|-------|---------|----------|
| `/implement-perturbation` | Implement a single perturbation | Adding a new perturbation to an existing category |
| `/implement-category` | Implement an entire category | Starting a new perturbation category (code + tests + doc + plots) |
| `/judge-category` | Optimize a category | P6 overhead > 100% on a category |
| `/implement-phase3` | Implement base Gym env | Phase 3 work |
| `/implement-adversarial-env` | Implement AdversarialEnv | Phase 4 work |
| `/implement-phase6` | Documentation & release | Phase 6 work |

## Commit & PR Conventions

- Branch per feature/milestone: `feat/milestone-N-description` (e.g., `feat/phase3-gym-env`)
- Commit messages: concise, imperative, English
- Never push directly to `main` — always use PRs
- Never force push
- CI must pass before merging
