# Adversarial Training

**Read when**: touching files in `src/genesis_robust_rl/adversarial/`.

---

## Components

| File | Role |
|------|------|
| `protocol.py` | `AdversarialAgent` Protocol + `RolloutData` dataclass |
| `networks.py` | `ActorCritic` shared MLP (Gaussian policy + value head) |
| `ppo_agent.py` | `PPOAgent` — reference PPO with GAE |
| `curriculum.py` | `CurriculumScheduler` — linear/cosine/step schedules, callback |
| `training_loop.py` | `train()` — DR / RARL (alternating) / RAP (joint) modes |

## Training Modes

- **DR** (Domain Randomization): protagonist only, perturbations sampled from distributions.
- **RARL** (Robust Adversarial RL): alternating updates — protagonist then adversary.
- **RAP** (Robust Adversarial Play): joint updates — both agents updated each iteration.

## Key Design Decisions

- `AdversarialAgent` is a `Protocol` (duck typing), not an ABC.
- Adversary observes `privileged_obs` (perturbation state) from `RobustDroneEnv`.
- Lipschitz clipping on adversary actions is enforced in `AdversarialEnv.set_value()`.
- Curriculum scheduler adjusts perturbation bounds during training.

## Design Doc

`docs/07_adversarial_training.md` — validated design reference. Read before structural changes.

## Cross-refs

- UP: `constraints.md` #2 (design before code), #3 (plan before implement)
- UP: `architecture.md` "Data Flow: Training Loop"
- PEER: `topics/gym_environment.md`, `topics/perturbation_engine.md`
