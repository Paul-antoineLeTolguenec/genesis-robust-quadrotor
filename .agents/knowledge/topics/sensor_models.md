# Sensor Models

**Read when**: touching `src/genesis_robust_rl/sensor_models.py` or category 4 perturbations.

---

## Forward Models

Six phenomenological sensor models outputting raw readings (no fusion):

| Model | Output Unit | Key Parameters |
|-------|-------------|----------------|
| Gyroscope | rad/s | bias instability, ARW, temperature sensitivity |
| Accelerometer | m/s^2 | bias, scale factor, cross-axis sensitivity |
| Magnetometer | uT | hard/soft iron distortion, inclination |
| Barometer | Pa | temperature drift, altitude noise |
| GPS | lat/lon/alt | HDOP, multipath, outage probability |
| Optical Flow | px/s | texture dependency, range scaling |

## Critical Rule

**No fusion assumption** (constraint #9-10): these models output raw sensor readings. Never assume a Kalman filter, EKF, or any estimator downstream. Category 4 perturbations corrupt the OUTPUT of these models.

## Reference Docs

- `docs/00b_sensor_models.md` — phenomenological model equations and parameters.
- `docs/01_perturbations_catalog.md` — category 4 entries reference sensor models.

## Cross-refs

- UP: `constraints.md` #7 (sensor model reference), #9-10 (no fusion assumption)
- UP: `architecture.md` "Sensor Forward Models"
- PEER: `topics/perturbation_engine.md`
