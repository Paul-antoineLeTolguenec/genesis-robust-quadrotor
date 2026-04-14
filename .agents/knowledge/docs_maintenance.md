# Knowledge Documentation Maintenance

**Read when**: adding or editing `.agents/` documentation.

---

## Architecture

The knowledge system uses a 3-layer design:

```
Root:    AGENTS.md                     -- project identity, principles (always read)
Tier 1:  constraints.md, architecture.md -- core facts (always read)
Routing: README.md                     -- trigger table from Tier 1 to Tier 2
Leaves:  topics/*.md                   -- self-contained detail (read when triggered)
Skills:  .claude/commands/             -- executable slash commands (Claude Code)
```

## Node Roles

**Non-leaf** (root + Tier 1): state facts concisely, then index to leaf docs. No inline explanations or code examples — those belong in leaves.

**Leaf** (`topics/*.md`): self-contained. Include code refs, checklists, gotchas. No filler prose, no restating parent content.

**Routing** (`README.md`): trigger table only. No prose.

## Cross-Reference Rules

1. Every leaf links UP to constraints/architecture via `## Cross-refs`.
2. Reference constraint numbers (`constraints.md #7`) — never re-explain the rule.
3. No duplication across layers — parent points, child details.

## Maintenance Checklist

| Change | Required updates |
|--------|-----------------|
| New topic doc | Add row to `README.md` routing table |
| New constraint | Update quick index range in `constraints.md` header |
| Append-only list | Only append, never reorder or remove |
| Moved detail | Replace inline content with pointer to the leaf |
