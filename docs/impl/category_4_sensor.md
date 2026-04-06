# Sensor Observation Perturbations — Implementation Notes

> Category 4 of the perturbation catalog.
> All perturbations inherit from `ObservationPerturbation` and operate on the obs tensor.
> SensorModel forward models produce clean readings; perturbations corrupt them.

---

## SensorModel Forward Models

Six phenomenological models convert Genesis state to raw sensor readings:

| Model | Output dim | Forward model |
|---|---|---|
| `GyroscopeModel` | 3 (rad/s) | `C_misalign @ ang_vel` |
| `AccelerometerModel` | 3 (m/s²) | `C_misalign @ (acc - R^T @ g)` |
| `MagnetometerModel` | 3 (µT) | `M_soft @ (R^T @ B_earth) + b_hard` |
| `BarometerModel` | 1 (m) | `pos_z - ref_alt` |
| `GPSModel` | 3 (m) | `pos + offset` |
| `OpticalFlowModel` | 2 (m/s) | `vel_xy / max(alt, ε)` |

All models are stateless (except potential drift in barometer) and use torch-only operations.

---

## 4.1 GyroNoise — Gyroscope white noise

**Formal definition:**
```
obs_gyro += N(μ, σ²) × curriculum_scale
```

| Parameter | Default | Bounds |
|---|---|---|
| `distribution` | gaussian | — |
| `std` | 0.02 rad/s | [0.0, 0.05] |
| `frequency` | per_step | — |

**Catalog ref:** 4.1 | **Class:** `GyroNoise(AdditiveNoise)` | **Stateful:** no

---

## 4.2 GyroBias — Gyroscope constant bias

**Formal definition:**
```
obs_gyro += bias,  bias ~ U[-0.1, 0.1] per axis, sampled per episode
```

| Parameter | Default | Bounds |
|---|---|---|
| `distribution` | uniform | — |
| `range` | ±0.1 rad/s | [-0.1, 0.1] |
| `frequency` | per_episode | — |

**Catalog ref:** 4.2 | **Class:** `GyroBias(AdditiveNoise)` | **Stateful:** no

---

## 4.3 GyroDrift — Gyroscope OU drift

**Formal definition:**
```
drift_{t+1} = drift_t + θ(μ - drift_t)dt + σ√dt × ε
obs_gyro += drift_t
```

| Parameter | Default | Bounds |
|---|---|---|
| `ou_theta` | 0.15 | — |
| `ou_sigma` | 0.01 | — |
| `drift` | — | [-0.2, 0.2] rad/s |
| `lipschitz_k` | 0.001 | — |

**Catalog ref:** 4.3 | **Class:** `GyroDrift(OUDrift)` | **Stateful:** yes

---

## 4.4 AccelNoiseBiasDrift — Accelerometer composite

**Formal definition:**
```
obs_accel += noise + bias + drift
  noise ~ N(0, noise_std²) per step
  bias  ~ U[-bias_range, bias_range] per episode
  drift = OU process (θ, σ, μ)
```

| Parameter | Default | Bounds |
|---|---|---|
| `noise_std` | 0.1 m/s² | — |
| `bias_range` | 1.0 m/s² | — |
| `ou_theta` | 0.15 | — |
| `ou_sigma` | 0.02 | — |
| `total` | — | [-2.0, 2.0] m/s² |

**Catalog ref:** 4.4 | **Class:** `AccelNoiseBiasDrift(ObservationPerturbation)` | **Stateful:** yes

---

## 4.5 SensorCrossAxis — Misalignment rotation

**Formal definition:**
```
obs_3d = R_misalign × obs_3d
R_misalign ≈ I + [θ]×,  θ ~ N(0, σ²), |θ| ≤ 5°
```

| Parameter | Default | Bounds |
|---|---|---|
| `angle_std` | 0.02 rad | — |
| `angle_max` | 0.0873 rad (5°) | — |
| `frequency` | per_episode | — |

**Catalog ref:** 4.5 | **Class:** `SensorCrossAxis(ObservationPerturbation)` | **Stateful:** no

---

## 4.6 PositionNoise — GPS/mocap noise

**Formal definition:**
```
obs_pos += N(0, σ²) per axis
```

| Parameter | Default | Bounds |
|---|---|---|
| `std` | 0.1 m | [-0.5, 0.5] m |
| `frequency` | per_step | — |

**Catalog ref:** 4.6 | **Class:** `PositionNoise(AdditiveNoise)` | **Stateful:** no

---

## 4.7 PositionDropout — GPS dropout with hold

**Formal definition:**
```
if dropout_active:
    obs_pos = last_valid_pos
trigger: Bernoulli(p), duration ~ U[d_lo, d_hi]
```

| Parameter | Default | Bounds |
|---|---|---|
| `drop_prob` | — | [0.0, 0.3] |
| `duration` | [1, 20] steps | — |
| `frequency` | per_step | — |

**Catalog ref:** 4.7 | **Class:** `PositionDropout(ObsDropout)` | **Stateful:** yes

---

## 4.8 PositionOutlier — GPS multipath spikes

**Formal definition:**
```
spike = Bernoulli(p_spike) × U[mag_lo, mag_hi] × random_direction
obs_pos += spike
```

| Parameter | Default | Bounds |
|---|---|---|
| `spike_prob` | 0.01 | [0.0, 0.05] |
| `magnitude` | [0.5, 5.0] m | [-5.0, 5.0] |

**Catalog ref:** 4.8 | **Class:** `PositionOutlier(ObservationPerturbation)` | **Stateful:** no

---

## 4.9 VelocityNoise — Velocity estimation noise

**Formal definition:**
```
obs_vel += N(0, σ²) per axis
```

| Parameter | Default | Bounds |
|---|---|---|
| `std` | 0.1 m/s | [-0.3, 0.3] m/s |

**Catalog ref:** 4.9 | **Class:** `VelocityNoise(AdditiveNoise)` | **Stateful:** no

---

## 4.10 SensorQuantization — Discretization

**Formal definition:**
```
obs_quantized = round(obs / resolution) × resolution
```

| Parameter | Default | Bounds |
|---|---|---|
| `resolution` | — | [1e-4, 1e-2] |
| `frequency` | per_episode | — |

**Catalog ref:** 4.10 | **Class:** `SensorQuantization(ObservationPerturbation)` | **Stateful:** no

---

## 4.11 ObsChannelMasking — Channel dropout

**Formal definition:**
```
mask = Bernoulli(1 - p) per channel
obs_masked = obs × mask
```

| Parameter | Default | Bounds |
|---|---|---|
| `mask_prob` | — | [0.0, 1.0] |
| `frequency` | per_episode | — |

**Catalog ref:** 4.11 | **Class:** `ObsChannelMasking(ObservationPerturbation)` | **Stateful:** no

---

## 4.12 MagnetometerInterference — Magnetic offset

**Formal definition:**
```
obs_mag += interference_offset
```

| Parameter | Default | Bounds |
|---|---|---|
| `offset` | — | [-50, +50] µT |
| `frequency` | per_episode | — |

**Catalog ref:** 4.12 | **Class:** `MagnetometerInterference(AdditiveNoise)` | **Stateful:** no

---

## 4.13 BarometerDrift — Altitude OU drift + noise

**Formal definition:**
```
drift_{t+1} = drift_t + θ(μ - drift_t)dt + σ√dt × ε
obs_alt += drift_t + N(0, noise_std²)
```

| Parameter | Default | Bounds |
|---|---|---|
| `noise_std` | 0.1 m | — |
| `ou_theta` | 0.05 | — |
| `ou_sigma` | 0.02 | — |
| `drift` | — | [-2.0, 2.0] m |

**Catalog ref:** 4.13 | **Class:** `BarometerDrift(ObservationPerturbation)` | **Stateful:** yes

---

## 4.14 OpticalFlowNoise — Flow sensor noise

**Formal definition:**
```
obs_flow += N(0, σ²) per axis
```

| Parameter | Default | Bounds |
|---|---|---|
| `std` | 0.1 m/s | [-0.5, 0.5] m/s |

**Catalog ref:** 4.14 | **Class:** `OpticalFlowNoise(AdditiveNoise)` | **Stateful:** no

---

## 4.15 IMUVibration — RPM-correlated noise

**Formal definition:**
```
σ_vib = gain × mean(rpm²)
obs_accel += N(0, σ_vib²)
```

| Parameter | Default | Bounds |
|---|---|---|
| `gain` | — | [0.0, 0.1] |
| `frequency` | per_step | — |

**Catalog ref:** 4.15 | **Class:** `IMUVibration(ObservationPerturbation)` | **Stateful:** no

---

## 4.16 ClockDrift — Module timing desynchronization

**Formal definition:**
```
phase_{t+1} = phase_t + rate_ppm × 1e-6 × dt
α = frac(|phase| / dt)
obs_out = (1 - α) × obs_current + α × obs_previous
```

| Parameter | Default | Bounds |
|---|---|---|
| `rate_ppm` | — | [-100, +100] ppm |
| `frequency` | per_step | — |

**Catalog ref:** 4.16 | **Class:** `ClockDrift(ObservationPerturbation)` | **Stateful:** yes

---

## Performance Overhead

*Measured with Genesis CF2X, n_envs=16 — see Phase 3 results.*

| Group | Overhead | Status |
|---|---|---|
| obs perturbations | TBD | — |
