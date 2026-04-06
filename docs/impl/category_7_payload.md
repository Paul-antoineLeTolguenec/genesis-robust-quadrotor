# Payload & Configuration Perturbations

Implementation documentation for Category 7 perturbations.

---

## 7.1 PayloadMass

**Description:** Per-env payload mass uncertainty applied via `set_links_mass_shift()`. Models unknown payload mass (always non-negative, unlike 1.1 MassShift which allows negative shifts).

**Formal definition:**
$$\Delta m \sim \mathcal{U}(0, m_{max}), \quad \Delta m \in [0, 0.5] \text{ kg}$$

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (0.0, 0.5) | Payload mass range (kg) |
| `nominal` | float | 0.0 | No payload |
| `distribution` | str | "uniform" | "uniform" or "gaussian" |

**Catalog reference:** 7.1

**Genesis API:** `drone.set_links_mass_shift(value, envs_idx)`

---

## 7.2 PayloadCOMOffset

**Description:** Per-env payload center-of-mass offset via `set_links_COM_shift()`. Models asymmetric payload attachment displacing the drone CoM.

**Formal definition:**
$$\Delta \mathbf{r} \sim \mathcal{U}(-0.1, 0.1)^3, \quad \Delta \mathbf{r} \in [-0.1, +0.1]^3 \text{ m}$$

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (-0.1, 0.1) | Per-axis CoM offset range (m) |
| `nominal` | list | [0, 0, 0] | No offset |
| `dimension` | tuple | (3,) | 3D vector |

**Catalog reference:** 7.2

**Genesis API:** `drone.set_links_COM_shift(value, envs_idx)`

---

## 7.3 AsymmetricPropGuardDrag

**Description:** Per-env asymmetric propeller guard drag via external force. Models manufacturing tolerances in prop guards causing uneven drag across the 4 arms.

**Formal definition:**
$$\mathbf{F} = -\sum_{j=1}^{4} r_j \cdot C_{d,nom} \cdot \text{sign}(\mathbf{v}) \cdot \mathbf{v}^2$$

where $r_j \in [0.8, 1.2]$ is the per-arm drag ratio and $C_{d,nom} = 0.025$ N·s²/m².

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (0.8, 1.2) | Per-arm drag ratio range |
| `nominal` | list | [1,1,1,1] | Symmetric drag |
| `Cd_nom` | float | 0.025 | Nominal drag coefficient |
| `dimension` | tuple | (4,) | One ratio per arm |

**Catalog reference:** 7.3

**Genesis API:** `solver.apply_links_external_force(wrench, link_idx, envs_idx)`

**Note:** `preserve_current_value=True` — privileged obs returns the 4 drag ratios (more informative than the derived force).
