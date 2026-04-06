# Wind and External Forces Perturbations

Implementation documentation for Category 5 perturbations.

---

## 5.1 ConstantWind

**Description:** Per-env constant wind velocity sampled per episode, applied as aerodynamic drag force.

**Formal definition:**
$$F = C_d \cdot \text{sign}(v_{wind} - v_{drone}) \cdot (v_{wind} - v_{drone})^2$$

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drag_coeff` | float | 0.1 | Effective drag coefficient Cd×A×ρ/2 [N·s²/m²] |
| `distribution` | str | "uniform" | Sampling distribution |
| `bounds` | tuple | (-10, 10) | Per-axis wind velocity [m/s] |
| `nominal` | list | [0, 0, 0] | No wind |

**Catalog reference:** 5.1
**Performance overhead:** (measured in Phase 3)

---

## 5.2 Turbulence

**Description:** Stochastic wind via 3D Ornstein-Uhlenbeck process per environment.

**Formal definition:**
$$dv = \theta(\mu - v)dt + \sigma\sqrt{dt}\,\varepsilon$$
$$F = C_d \cdot \text{sign}(v_{OU} - v_{drone}) \cdot (v_{OU} - v_{drone})^2$$

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drag_coeff` | float | 0.1 | Effective drag coefficient |
| `sigma_low/high` | float | 0.5 / 2.0 | OU noise amplitude range |
| `theta` | float | 1.0 | Mean-reversion rate |
| `bounds` | tuple | (-5, 5) | Per-axis velocity clamp [m/s] |

**Catalog reference:** 5.2

---

## 5.3 WindGust

**Description:** Random impulse wind events with stochastic triggering, duration, and direction.

**Formal definition:**
- Trigger: Bernoulli(p) per step per env (inactive envs only)
- Active: F = magnitude × direction (uniform on S²) for sampled duration
- Expired: F = 0

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob_low/high` | float | 0.0 / 0.05 | Gust trigger probability range |
| `mag_low/high` | float | 1.0 / 10.0 | Gust magnitude range [N] |
| `duration_low/high` | int | 1 / 20 | Gust duration range [steps] |
| `bounds` | tuple | (-15, 15) | Per-axis force clamp [N] |

**Catalog reference:** 5.3

---

## 5.4 WindShear

**Description:** Altitude-dependent wind gradient producing horizontal drag force.

**Formal definition:**
$$v_{wind} = \text{gradient} \times h \times \hat{d}$$
$$F = C_d \cdot \text{sign}(v_{wind} - v_{drone}) \cdot (v_{wind} - v_{drone})^2$$

where $h$ = altitude, $\hat{d}$ = horizontal wind direction (sampled per episode).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drag_coeff` | float | 0.1 | Effective drag coefficient |
| `bounds` | tuple | (0, 2) | Gradient range [(m/s)/m] |

**Catalog reference:** 5.4

---

## 5.5 AdversarialWind

**Description:** Same model as ConstantWind but adversary-controlled with Lipschitz enforcement.

**Formal definition:** Same as 5.1, with per-step Lipschitz constraint on wind velocity:
$$\|v_{t+1} - v_t\|_\infty \leq k \cdot dt$$

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lipschitz_k` | float | 5.0 | Max velocity change rate [m/s²] |
| `bounds` | tuple | (-10, 10) | Per-axis velocity [m/s] |

**Catalog reference:** 5.5

---

## 5.6 BladeVortexInteraction

**Description:** Stochastic rotor wake ingestion modeled as per-rotor OU thrust perturbation.

**Formal definition:**
$$\Delta T_j = x_j \cdot K_F \cdot \omega_j^2, \quad x_j \sim \text{OU}(\theta, \sigma, \mu)$$
$$F_z = \sum_{j=1}^{4} \Delta T_j$$

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `KF` | float | 3.16e-10 | Thrust coefficient [N/RPM²] |
| `sigma_low/high` | float | 0.01 / 0.05 | OU noise range |
| `theta` | float | 2.0 | Mean-reversion rate |
| `bounds` | tuple | (-0.1, 0.1) | Fractional thrust perturbation |

**Catalog reference:** 5.6

---

## 5.7 GroundEffectBoundary

**Description:** Dynamic ground-effect transition using Cheeseman-Bennett model with perturbable rotor diameter.

**Formal definition:**
$$k_{ge} = \frac{1}{1 - (R/(4h))^2}, \quad R = D/2$$
$$\Delta F_z = (k_{ge} - 1) \cdot T_{nominal}$$

Transition zone: $h \in [D, 3D]$. Above: no effect.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nominal_thrust` | float | 0.265 | Hover thrust [N] (CF2X) |
| `max_k_ge` | float | 2.0 | Max ground-effect multiplier |
| `bounds` | tuple | (0.05, 0.2) | Rotor diameter range [m] |

**Catalog reference:** 5.7

---

## 5.8 PayloadSway

**Description:** Suspended load pendulum dynamics producing reaction force on drone.

**Formal definition:**
$$\ddot{\theta} = -\frac{g}{L}\sin(\theta)$$
$$F_{xy} = -m \cdot g \cdot \sin(\theta), \quad F_z = -m \cdot g \cdot (\cos(\theta) - 1)$$

Integrated via symplectic Euler. Initial angles perturbed to break symmetry.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `length_low/high` | float | 0.1 / 1.0 | Cable length range [m] |
| `mass_low/high` | float | 0.01 / 0.5 | Payload mass range [kg] |
| `bounds` | tuple | (-5, 5) | Per-axis reaction force clamp [N] |

**Catalog reference:** 5.8

---

## 5.9 ProximityDisturbance

**Description:** Aerodynamic disturbance near walls/ceiling, decaying quadratically with distance.

**Formal definition:**
$$F = F_{max} \cdot \left(1 - \frac{d}{d_{max}}\right)^2 \cdot \hat{n}$$

where $d$ = distance to nearest surface, $\hat{n}$ = repulsive direction.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d_max` | float | 0.3 | Max influence distance [m] |
| `surface_distance` | float | 2.0 | Wall distance from origin [m] |
| `bounds` | tuple | (0, 0.5) | F_max range [N] |

**Catalog reference:** 5.9
