# Action Perturbations

Implementation documentation for Category 6 perturbations.

---

## 6.1 ActionNoise

**Description:** Additive noise on the action vector at each step.

**Formal definition:**
$$a_{out} = a_{in} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

| Parameter | Type | Default | Description |
|---|---|---|---|
| `noise_type` | str | "gaussian" | "gaussian" or "uniform" |
| `bounds` | tuple | (0, 0.1) | Noise std range |
| `nominal` | float | 0.0 | No noise |

**Catalog reference:** 6.1

---

## 6.2 ActionDeadzone

**Description:** Zeros out action channels below a threshold, modeling actuator dead-band.

**Formal definition:**
$$a_{out,i} = \begin{cases} 0 & \text{if } |a_{in,i}| < \tau \\ a_{in,i} & \text{otherwise} \end{cases}$$

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (0, 0.1) | Threshold range (normalized) |
| `nominal` | float | 0.0 | No deadzone |

**Catalog reference:** 6.2

---

## 6.3 ActionSaturation

**Description:** Clips action to a reduced symmetric range.

**Formal definition:**
$$a_{out} = \text{clamp}(a_{in}, -L, +L), \quad L \in [0.5, 1.0]$$

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (0.5, 1.0) | Saturation limit range |
| `nominal` | float | 1.0 | Full range |

**Catalog reference:** 6.3

---

## 6.4 ActuatorHysteresis

**Description:** Direction-dependent offset modeling mechanical hysteresis.

**Formal definition:**
$$a_{out} = a_{in} + \text{sign}(a_{in} - a_{prev}) \cdot \frac{w}{2}$$

where $w$ is the hysteresis width sampled per episode.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (0, 0.05) | Width range (normalized RPM) |
| `nominal` | float | 0.0 | No hysteresis |

**Catalog reference:** 6.4

---

## 6.5 ESCLowPassFilter

**Description:** First-order IIR low-pass filter modeling ESC bandwidth.

**Formal definition:**
$$y_t = y_{t-1} + (x_t - y_{t-1}) \cdot \alpha, \quad \alpha = \text{clamp}(2\pi f_c \cdot dt, 0, 1)$$

At infinite $f_c$: pass-through. At low $f_c$: heavy smoothing.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bounds` | tuple | (5, 50) | Cutoff frequency range [Hz] |
| `nominal` | float | 50.0 | Minimal filtering |

**Catalog reference:** 6.5
