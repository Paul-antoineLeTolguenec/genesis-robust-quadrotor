# 00b — Sensor Forward Models

> Reference document for phenomenological sensor models used in perturbation design.
> Each model converts the true Genesis state into a raw sensor reading, without
> assuming any downstream fusion (Kalman, transformer, or otherwise).
> The user's policy receives raw sensor outputs and decides how to process them.

---

## Design principles

- **No fusion assumption** — models output raw sensor readings only.
- **Phenomenological, not first-principles** — parameters are calibrated against
  real hardware, not derived from physics PDEs (no Maxwell, no Navier-Stokes).
- **Perturbable parameters** — every model constant is a perturbation target.
  The `SensorModel` layer and `SensorPerturbation` classes operate on these parameters.
- **Industry standard** — this approach is used by AirSim, Gazebo, PX4 SITL.

---

## Inputs from Genesis (true state, per env per step)

```
pos        : Tensor[n_envs, 3]   # world position (m)
quat       : Tensor[n_envs, 4]   # body orientation (w, x, y, z)
vel        : Tensor[n_envs, 3]   # body linear velocity (m/s)
ang_vel    : Tensor[n_envs, 3]   # body angular velocity (rad/s)
rpm        : Tensor[n_envs, 4]   # propeller RPM
dt         : float               # simulation timestep (s)
```

---

## 1. IMU — Gyroscope

**Physical quantity:** body angular velocity (rad/s)

```
gyro_raw = C_misalign × ang_vel_true + bias + drift_t + white_noise
```

| Parameter | Description | Perturbation |
|---|---|---|
| `C_misalign` | 3×3 rotation (cross-axis sensitivity) | 4.5 |
| `bias` | constant per-episode offset (rad/s) | 4.2 |
| `drift_t` | OU process state (rad/s) | 4.3 |
| `white_noise` | i.i.d. Gaussian, std σ | 4.1 |

`drift_t` evolves as:
```
drift_{t+1} = drift_t + theta × (mu - drift_t) × dt + sigma × sqrt(dt) × N(0,1)
```

---

## 2. IMU — Accelerometer

**Physical quantity:** specific force = body linear acceleration − gravity projected in body frame (m/s²)

```
g_body     = R_body^T × [0, 0, -9.81]
accel_true = env_state.acc - g_body  # env_state.acc = (vel_t - vel_{t-1}) / dt, provided by env
accel_raw  = C_misalign × accel_true + bias + drift_t + white_noise
             + vibration_gain × mean(rpm²) × N(0,1)
```

`AccelerometerModel` is **stateless** — it reads `acc` directly from `EnvState` and does not
maintain `_vel_prev` internally. No `reset()` override needed.

| Parameter | Description | Perturbation |
|---|---|---|
| `C_misalign` | cross-axis sensitivity | 4.5 |
| `bias` | constant per-episode offset (m/s²) | 4.4 |
| `drift_t` | OU process state (m/s²) | 4.4 |
| `white_noise` | i.i.d. Gaussian | 4.4 |
| `vibration_gain` | RPM-correlated noise gain | 4.15 |

---

## 3. Magnetometer

**Physical quantity:** magnetic field vector in body frame (µT)

```
B_earth    = WMM2020_lookup(lat, lon, alt)   # world frame reference field
B_body     = R_body^T × B_earth              # projected into body frame
B_raw      = M_soft × B_body + b_hard + b_motor(rpm) + white_noise
```

| Parameter | Description | Perturbation |
|---|---|---|
| `M_soft` | 3×3 soft-iron distortion matrix (≈ I at nominal) | 4.12 |
| `b_hard` | hard-iron offset vector (µT), constant per-episode | 4.12 |
| `b_motor` | motor interference: `f(mean(rpm))` OU process | 4.12 |
| `white_noise` | i.i.d. Gaussian | 4.12 |

> `B_earth` is a constant for a fixed flight arena. For curriculum purposes it can
> be treated as a perturbable parameter (heading reference shift).

---

## 4. Barometer

**Physical quantity:** atmospheric pressure → altitude estimate (m)

```
P_true     = P0 × (1 - 2.2558e-5 × alt)^5.2559    # ISA model
alt_raw    = barometric_formula_inverse(P_true)
           + drift_t + white_noise + pressure_wash(vel_z)
```

| Parameter | Description | Perturbation |
|---|---|---|
| `drift_t` | OU process state (m) — pressure wash / thermal drift | 4.13 |
| `white_noise` | i.i.d. Gaussian, std σ (m) | 4.13 |
| `pressure_wash` | empirical term proportional to vertical velocity | 4.13 |

Output: scalar altitude estimate (m).

---

## 5. GPS / Position sensor

**Physical quantity:** position in world frame (m)

```
pos_raw    = pos_true + multipath_spike × Bernoulli(p_spike) + white_noise
```

| Parameter | Description | Perturbation |
|---|---|---|
| `white_noise` | i.i.d. Gaussian, std σ (m) | 4.6 |
| `multipath_spike` | random large outlier, magnitude uniform | 4.8 |
| `dropout` | zero / hold last valid, event-driven | 4.7 |

Output: `vector(3)` position (m).

---

## 6. Optical Flow sensor

**Physical quantity:** apparent horizontal velocity in sensor frame (m/s or px/s)

```
flow_raw   = vel_horizontal_true / alt_true + white_noise
```

This is a simplified divergence model. No image rendering required.

| Parameter | Description | Perturbation |
|---|---|---|
| `white_noise` | i.i.d. Gaussian | 4.14 |
| `dropout` | zero output (surface lost) | 4.14 |

Output: `vector(2)` horizontal flow (m/s).

---

## Summary

| Sensor | Output dim | Forward model complexity | Requires `SensorModel` class |
|---|---|---|---|
| Gyroscope | 3 | Low — linear + OU | Yes |
| Accelerometer | 3 | Low — linear + OU | Yes |
| Magnetometer | 3 | Medium — projection + distortion | **Yes** |
| Barometer | 1 | Low — ISA + OU | Yes |
| GPS / Position | 3 | Low — noise + spike | Yes |
| Optical Flow | 2 | Low — kinematic ratio | Yes |

All models are pure Python/Torch wrappers on Genesis state — no Genesis API calls required.
