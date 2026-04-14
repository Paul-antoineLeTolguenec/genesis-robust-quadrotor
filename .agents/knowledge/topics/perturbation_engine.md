# Perturbation Engine

**Read when**: touching files in `src/genesis_robust_rl/perturbations/`.

---

## Class Hierarchy

```
Perturbation (abstract)
├── PhysicsPerturbation
│   ├── GenesisSetterPerturbation  (mass, COM, friction, DOF gains...)
│   └── ExternalWrenchPerturbation (forces, torques, wind, payload)
├── MotorCommandPerturbation       (RPM corruption)
├── ObservationPerturbation        (sensor noise, bias, delay)
└── ActionPerturbation             (action noise, saturation, dead zone)
```

## Key Interfaces

- `tick(is_reset: bool, env_ids: Tensor | None = None)` — sample new value (DR) or advance stateful process. `env_ids=None` means all envs.
- `apply(...)` — apply perturbation effect. Signature varies by leaf type.
- `set_value(value: Tensor)` — adversarial mode: external value injection with Lipschitz clip.
- `update_params(new_params: dict)` — curriculum: update distribution bounds at runtime.

## Modes

- `PerturbationMode.DOMAIN_RANDOMIZATION` — tick() samples from distribution.
- `PerturbationMode.ADVERSARIAL` — tick() does NOT sample; values come from `set_value()`.

## Stateful Processes

- `OUProcess` — Ornstein-Uhlenbeck for correlated noise (torch-only, no numpy).
- `DelayBuffer` — circular tensor buffer `[n_envs, max_delay, dim]`.

## Registry

`PerturbationRegistry` singleton with `get/list/build/build_from_config`. All perturbations auto-registered.

## Batch Shape Convention

`_batch_shape()` returns `(1, *dim)` if scope="global", `(n_envs, *dim)` if per-env.

## Implementation Cycle

Per perturbation: code -> tests (U1-U11, I1-I3, P1) -> doc + 3 plots -> review -> P6 measure.
See skill `/implement-perturbation` for the full workflow.

## Cross-refs

- UP: `constraints.md` #6 (feasibility reference), #15-16 (testing), #18-19 (P6 overhead)
- UP: `architecture.md` "Perturbation Hierarchy"
- PEER: `topics/performance.md`, `topics/perturbation_docs.md`
