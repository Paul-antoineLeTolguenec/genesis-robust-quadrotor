# External Disturbances — Category 8

## 8.1 BodyForceDisturbance

### Description

Per-env external body force disturbance applied via `apply_links_external_force`.
Supports two regimes: **constant** (uniform/gaussian, stateless) and **OU** (ou_process, stateful).

### Formal definition

**Constant mode:**
$$F = \text{sample}(\mathcal{D}) \cdot c_s + F_{\text{nom}} \cdot (1 - c_s), \quad F \in [-5, +5]\ \text{N per axis}$$

**OU mode:**
$$dF = \theta(\mu - F)\,dt + \sigma\sqrt{dt}\,\varepsilon, \quad \sigma_i \sim \mathcal{U}[\sigma_{\text{low}}, \sigma_{\text{high}}]$$

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `distribution` | `"uniform"` | `"uniform"`, `"gaussian"`, or `"ou_process"` |
| `bounds` | `[-5.0, +5.0]` | Hard clamp per axis (N) |
| `nominal` | `[0, 0, 0]` | Unperturbed force (N) |
| `lipschitz_k` | `0.5` | Max ΔF per step in adversarial mode (N/step) |
| `frame` | `"world"` | `"world"` or `"local"` |
| `link_idx` | `0` | Body link index |
| `duration_mode` | `"continuous"` | `"continuous"` or `"pulse"` |

**OU-specific params:** `theta` (mean reversion), `sigma_low`/`sigma_high` (per-env volatility range), `mu` (long-term mean).

### Catalog reference

Category 8, entry 8.1 — `docs/01_perturbations_catalog.md`

### Performance overhead vs n_envs

*Measured in Phase 3 — see P6 overhead table below.*

### Usage example

```python
from genesis_robust_rl.perturbations.category_8_external import BodyForceDisturbance

# Constant mode (stateless)
force = BodyForceDisturbance(n_envs=16, dt=0.01, distribution="uniform")

# OU mode (stateful)
force_ou = BodyForceDisturbance(n_envs=16, dt=0.01, distribution="ou_process")
```

### Genesis API note

Uses `scene.rigid_solver.apply_links_external_force(wrench, link_idx, envs_idx, local=...)`.

---

## 8.2 BodyTorqueDisturbance

### Description

Per-env external body torque disturbance applied via `apply_links_external_torque`.
Same dual-regime design as 8.1 but with torque-appropriate bounds.

### Formal definition

**Constant mode:**
$$\tau = \text{sample}(\mathcal{D}) \cdot c_s + \tau_{\text{nom}} \cdot (1 - c_s), \quad \tau \in [-1, +1]\ \text{N·m per axis}$$

**OU mode:**
$$d\tau = \theta(\mu - \tau)\,dt + \sigma\sqrt{dt}\,\varepsilon, \quad \sigma_i \sim \mathcal{U}[\sigma_{\text{low}}, \sigma_{\text{high}}]$$

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `distribution` | `"uniform"` | `"uniform"`, `"gaussian"`, or `"ou_process"` |
| `bounds` | `[-1.0, +1.0]` | Hard clamp per axis (N·m) |
| `nominal` | `[0, 0, 0]` | Unperturbed torque (N·m) |
| `lipschitz_k` | `0.05` | Max Δτ per step in adversarial mode (N·m/step) |
| `frame` | `"world"` | `"world"` or `"local"` |
| `link_idx` | `0` | Body link index |
| `duration_mode` | `"continuous"` | `"continuous"` or `"pulse"` |

**OU-specific params:** `theta` (default 1.0), `sigma_low`/`sigma_high` (default 0.05/0.3), `mu` (default 0.0).

### Catalog reference

Category 8, entry 8.2 — `docs/01_perturbations_catalog.md`

### Performance overhead vs n_envs

*Measured in Phase 3 — see P6 overhead table below.*

### Usage example

```python
from genesis_robust_rl.perturbations.category_8_external import BodyTorqueDisturbance

# Constant mode (stateless)
torque = BodyTorqueDisturbance(n_envs=16, dt=0.01, distribution="uniform")

# OU mode (stateful)
torque_ou = BodyTorqueDisturbance(n_envs=16, dt=0.01, distribution="ou_process")
```

### Genesis API note

Uses `scene.rigid_solver.apply_links_external_torque(wrench, link_idx, envs_idx, local=...)`.
