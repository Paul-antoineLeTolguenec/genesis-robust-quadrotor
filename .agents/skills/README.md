# Agent Skills

## Available Skills

Skills live in `.claude/commands/` as Claude Code slash commands.

| Skill | Purpose | Use When |
|-------|---------|----------|
| `/implement-perturbation` | Implement a single perturbation (code + tests + doc + plots) | Adding a new perturbation |
| `/implement-category` | Implement an entire category sequentially | Starting a new perturbation category |
| `/judge-category` | Generate optimization proposals for a category | P6 overhead > 100% |
| `/implement-phase3` | Implement base Gymnasium environment | Phase 3 work |
| `/implement-adversarial-env` | Implement AdversarialEnv wrapper | Phase 4 work |
| `/implement-phase6` | Documentation and release | Phase 6 work |

## Adding New Skills

1. Create command file: `.claude/commands/<name>.md`
2. Update this README
3. Register in `AGENTS.md` skills table
