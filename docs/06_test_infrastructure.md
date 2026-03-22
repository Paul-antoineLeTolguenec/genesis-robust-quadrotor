# 06 — Test Infrastructure

> Setup guide for the test harness used across all implementation phases.
> Complements `05_test_conventions.md` (which defines *what* to test).
> This document defines *how* the harness is configured and bootstrapped.
> References: `02_class_design.md`, `05_test_conventions.md`.

---

## Design principles

- **Single harness, all phases** — pytest config is set up once in Phase 1.5; all Phase 2+ subagents reuse it unchanged.
- **Markers isolate test tiers** — `unit`, `integration`, `perf` are registered markers; `perf` is excluded from the default run via `addopts`.
- **No Genesis required for unit tests** — `mock_scene()` stubs all Genesis API calls; unit tests run on any machine with CPU.
- **GPU tests are optional** — `perf` tests skip gracefully when CUDA is unavailable via an explicit `pytest.skip()` call.

---

## 1. pyproject.toml configuration

### Changes to apply (diff on top of existing `pyproject.toml`)

**Add `torch` to dev dependencies** (required by `conftest.py` and all tests):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",       # align with existing constraint
    "pytest-cov>=4.0",   # align with existing constraint
    "ruff>=0.4",
    "torch>=2.0",        # required by conftest.py fixtures
]
genesis = [
    "genesis-world>=0.2",
]
```

**Add markers, coverage config, and `addopts` to `[tool.pytest.ini_options]`**:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -m 'not perf'"   # perf excluded by default; run explicitly with -m perf
markers = [
    "unit: unit tests — no Genesis, mocked EnvState (fast, CPU-only)",
    "integration: integration tests — real EnvState, mocked Genesis scene",
    "perf: performance tests — GPU required, n_envs=512",
]

[tool.coverage.run]
source = ["src/genesis_robust_rl"]
omit = ["tests/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

### Install

```bash
uv sync --extra dev          # installs pytest, pytest-cov, ruff, torch
uv pip install -e .          # editable install of genesis_robust_rl (required before running tests)
```

### Run commands

```bash
uv run pytest                              # unit + integration (perf excluded by addopts)
uv run pytest -m unit                     # unit tests only
uv run pytest -m "unit or integration"    # explicit CI equivalent
uv run pytest -m perf                     # perf tests only (requires GPU)
uv run pytest tests/category_1_physics/   # single category (all tiers)
uv run pytest --cov=src/genesis_robust_rl # with coverage report
```

---

## 2. tests/ directory structure

```
tests/
  conftest.py                    # shared fixtures (from 05_test_conventions.md)
  test_placeholder.py            # CI smoke test — Phase 1.5 only
  category_1_physics/
    conftest.py                  # perturbation fixture parametrized over category 1 leaves
    test_genesis_setter.py
    test_external_wrench.py
    test_perf_physics.py
  category_2_motor/
    conftest.py
    test_motor_perturbations.py
    test_perf_motor.py
  category_3_temporal/
    conftest.py
    test_temporal_perturbations.py
    # no perf file — temporal perturbations operate at RL-step level, not inner loop
  category_4_sensor/
    conftest.py
    test_sensor_models.py
    test_obs_perturbations.py
    test_perf_sensor.py
  category_5_wind/
    conftest.py
    test_wind_perturbations.py
    test_perf_wind.py
  category_6_action/
    conftest.py
    test_action_perturbations.py
    test_perf_action.py
  category_7_payload/
    conftest.py
    test_payload_perturbations.py
  category_8_external/
    conftest.py
    test_external_perturbations.py
    test_perf_external.py
```

---

## 3. Placeholder test

`tests/test_placeholder.py` — used only during Phase 1.5 to validate the harness before Phase 2 tests exist.
The `-m 'not perf'` `addopts` filter would skip this file if it had no `unit` marker, so all three tests
are marked `unit` to ensure they are collected.

```python
"""Placeholder test — validates CI test harness setup. Remove after Phase 2 is complete."""
import importlib
import pytest


@pytest.mark.unit
def test_harness_ok():
    """CI smoke test: pytest runs, markers are registered."""
    assert True


@pytest.mark.unit
def test_torch_importable():
    """torch must be importable (listed in dev extras)."""
    import torch  # noqa: F401
    assert torch.__version__, "torch.__version__ is empty"


@pytest.mark.unit
def test_genesis_robust_rl_importable():
    """Package must be importable after `uv pip install -e .`."""
    spec = importlib.util.find_spec("genesis_robust_rl")
    assert spec is not None, (
        "genesis_robust_rl not found — run `uv pip install -e .` before testing"
    )
```

---

## 4. Shared conftest.py

`tests/conftest.py` contains all fixtures from `05_test_conventions.md`. Minimal reference:

| Fixture | Type | Scope | Description |
|---|---|---|---|
| `n_envs` | fixture (params=[1,4,16]) | function | env batch size |
| `mock_env_state(n_envs)` | fixture | function | hovering drone `EnvState`, CPU tensors |
| `mock_scene()` | fixture | function | Genesis scene stub — all setters are `MagicMock` |
| `assert_lipschitz(p, n)` | helper function | — | verifies Lipschitz over n steps; reads `p.dt` internally |
| `mock_env_state_gpu` | fixture | function | same as `mock_env_state` with all tensors on CUDA; skips if no CUDA |
| `perturbation_gpu` | fixture | function | moves perturbation internal tensors to CUDA; skips if no CUDA |
| `sensor_model_gpu` | fixture | function | moves sensor model internal tensors to CUDA; skips if no CUDA |

**GPU fixture skip pattern** (used in all three GPU fixtures):

```python
@pytest.fixture
def mock_env_state_gpu(mock_env_state):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # move all tensors to cuda and return
    ...
```

**`assert_lipschitz` is a helper function, not a pytest fixture.** Import directly:

```python
from tests.conftest import assert_lipschitz
```

**Rule:** never duplicate fixture logic in a category `conftest.py`. Always reuse root fixtures.

---

## 5. Category conftest.py pattern

Each `tests/category_N_*/conftest.py` must define exactly one fixture: `perturbation`, parametrized
over all leaf classes in the category. Minimal pattern:

```python
import pytest
from genesis_robust_rl.perturbations.category_N_name import ClassA, ClassB

@pytest.fixture(params=[
    lambda n: ClassA(id="a", n_envs=n, dt=0.01, ...),
    lambda n: ClassB(id="b", n_envs=n, dt=0.01, ...),
])
def perturbation(request, n_envs):
    return request.param(n_envs)
```

For category 4 (sensor), also define `sensor_model` and `obs_perturbation` fixtures — see
`05_test_conventions.md` §Shared fixtures for the exact pattern.

**`perturbation_class` fixture (for P3 memory test):**

```python
@pytest.fixture(params=[ClassA, ClassB])
def perturbation_class(request):
    """Returns the class itself (not an instance) for memory scaling tests."""
    return request.param
```

---

## 6. Phase 2 subagent workflow

Each subagent receives the following documents:

```
docs/02_class_design.md              # class hierarchy + method signatures
docs/05_test_conventions.md          # test contract (U1–U11, I1–I5, P1–P6)
docs/06_test_infrastructure.md       # this file
docs/01_perturbations_catalog.md     # entries for its category only
docs/00_feasibility.md               # Genesis API feasibility (all categories)
docs/00b_sensor_models.md            # sensor forward models (category 4 only)
```

**Steps:**

```
1. Implement perturbation classes in src/genesis_robust_rl/perturbations/category_N_*.py
2. Create tests/category_N_*/conftest.py with perturbation fixture (§5 pattern above)
3. Copy relevant test templates from 05_test_conventions.md; fill in category-specific values
4. uv run pytest tests/category_N_*/ -m "unit or integration"
5. Fix until all tests pass (0 failures, 0 errors)
6. uv run pytest tests/category_N_*/ -m perf  (if GPU available; else skip)
7. DONE only when: uv run pytest tests/category_N_*/ exits 0
```

**Must NOT:**
- Modify `tests/conftest.py` — shared across all categories
- Add fixtures to root `conftest.py` without user approval
- Use `pytest.skip()` except for GPU-absent perf tests

---

## 7. CI integration

### Phase 1.5 CI (placeholder only)

While only `test_placeholder.py` exists, the CI command must not filter by marker — the filter
`-m 'not perf'` in `addopts` is sufficient:

```yaml
- name: Run tests
  run: uv run pytest --cov=src/genesis_robust_rl --cov-report=xml
```

### Phase 2+ CI (categories present)

Once category tests exist, coverage threshold enforcement activates:

```yaml
- name: Install
  run: |
    uv sync --extra dev
    uv pip install -e .

- name: Run tests
  run: uv run pytest --cov=src/genesis_robust_rl --cov-report=xml

- name: Upload coverage
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report
    path: coverage.xml
```

`addopts = "-m 'not perf'"` ensures perf tests are always excluded from CI.
`fail_under = 80` is enforced by pytest-cov at the end of the run.

---

## Summary

| Component | Purpose |
|---|---|
| `pyproject.toml` additions | markers, addopts, coverage config, torch dev dep |
| `tests/conftest.py` | shared fixtures — never modified by subagents |
| `tests/test_placeholder.py` | Phase 1.5 smoke test; removed after Phase 2 |
| `tests/category_N_*/conftest.py` | per-category `perturbation` fixture (parametrized) |
| `tests/category_N_*/test_*.py` | test files per 05_test_conventions.md templates |
