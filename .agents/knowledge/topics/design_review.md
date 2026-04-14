# Design Review Protocol

**Read when**: creating or editing design documents in `docs/`.

---

## When to Apply

After completing any design document in `docs/`, BEFORE marking it `[x]` in ROADMAP.md.

## Process

1. Launch **2-3 review subagents in parallel**, each receiving:
   - The document just written (full content)
   - All prerequisite documents it references
   - The review prompt (see below)

2. **Synthesis**: collect all blocking issues, present to user, iterate until resolved.

## Review Prompt Template

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
1. Internal consistency — does the document contradict itself?
2. Cross-document consistency — does it contradict any prerequisite documents?
3. Completeness — are there gaps, undefined terms, or missing cases?
4. Implementability — is every interface precisely enough defined to be implemented without ambiguity?
5. Edge cases — are reset, per-env vs global scope, stateful vs stateless, adversarial vs DR all handled?

Return: a numbered list of blocking issues (must fix) and non-blocking suggestions (nice to have). Be concise.
```

## Cross-refs

- UP: `constraints.md` #2 (design before code), #6-8 (feasibility/sensor references)
- PEER: `topics/perturbation_engine.md`
