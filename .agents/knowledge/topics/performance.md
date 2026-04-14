# Performance & P6 Overhead

**Read when**: running P6 overhead tests or optimizing perturbation performance.

---

## P6 Rule

Perturbation logic (sample + curriculum + clamp) must add **< 5% overhead** vs passing fixed tensors to the same Genesis setters (constraint #18).

## Test Setup

- File: `tests/integration/test_overhead_genesis.py`
- URDF: real Crazyflie CF2X
- n_envs: 16
- Methodology: median of 5 rounds x 100 steps
- Baseline: fixed tensors -> Genesis setter -> step
- Perturbed: sample+clip -> Genesis setter -> step
- Measures ONLY perturbation logic overhead, NOT Genesis API cost

## Historical Results (2025-03-21)

| Perturbation | Overhead |
|-------------|----------|
| mass_shift | +2.6% |
| com_shift | +1.7% |
| ext_force | +2.7% |
| ext_torque | +2.8% |
| COMBINED | +3.9% |

## Optimization Pipeline

When overhead > 100% (CPU perf tests):
1. Run `/judge-category N` — generates optimization proposals
2. User reviews and decides which optimizations to apply
3. Re-measure after applying

## Requirements

- Genesis must be installed locally (not available in CI)
- Must pass before any push/PR
- CPU backend works on Mac M4 Pro; CUDA for GPU

## Cross-refs

- UP: `constraints.md` #18-19 (P6 overhead rule and methodology)
- UP: `architecture.md` "Configuration"
- PEER: `topics/perturbation_engine.md`
