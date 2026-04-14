# Perturbation Documentation

**Read when**: adding documentation or plots for a perturbation.

---

## Required Per Perturbation

### 1. Implementation Doc

Add a section in `docs/impl/category_N_*.md` with:
- Formal definition (equation)
- Parameter table
- Catalog reference (link to `docs/01_perturbations_catalog.md`)

Doc title must be descriptive — never "Category N — ...".

### 2. Three Plotly Graphs

Add in `docs/impl/plot_category_N.py`:
1. **Curriculum effect** — how perturbation evolves with curriculum_scale
2. **Per-env DR** — distribution across n_envs
3. **Perf overhead vs n_envs** — scaling behavior

### 3. Export

- Export as PNG to `docs/impl/assets/` via `pio.write_image()`
- Run immediately: `uv run python docs/impl/plot_category_N.py`
- If Chromium error: run `uv run python -c "from kaleido import get_chrome_sync; get_chrome_sync()"` first, then retry

## Cross-refs

- UP: `constraints.md` #20-21 (doc and plots per perturbation)
- PEER: `topics/perturbation_engine.md`, `topics/performance.md`
