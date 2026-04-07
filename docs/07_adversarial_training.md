# 07 — Adversarial Training Module

> Training utilities for robust RL: DR baseline, RARL (alternating), RAP (joint).
> References: `03_api_design.md` (§5 AdversarialEnv), `04_interactions.md` (§4 adversarial step).
> No code is written before this document is approved.

---

## Design principles

- **Algo-agnostic loop** — the training loop takes two opaque agents satisfying a Protocol.
  Users can bring their own PPO (SB3, CleanRL, custom) wrapped in ~10 lines.
- **Protocol over ABC** — `AdversarialAgent` is a `typing.Protocol`, not an abstract base class.
  This aligns with duck typing and avoids forcing inheritance.
- **No new dependencies** — only `torch` and `gymnasium` (already required). No SB3, no CleanRL.
- **Callbacks, not hooks** — curriculum, logging, checkpointing are callbacks passed to the loop.
  The loop itself has no knowledge of these concerns.
- **DR and adversarial in one loop** — the same `train()` function handles DR (5-tuple `env.step()`)
  and adversarial (6-tuple `AdversarialEnv.step()`) modes, so baselines share the exact same code path.

---

## 1. Agent Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class AdversarialAgent(Protocol):
    """Minimal interface for any agent in the minimax loop."""

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Select action from observation.

        Args:
            obs: Tensor[n_envs, obs_dim]

        Returns:
            action:   Tensor[n_envs, action_dim] — sampled action
            log_prob: Tensor[n_envs]             — log-probability of sampled action
            value:    Tensor[n_envs]             — critic value estimate
        """
        ...

    def update(self, rollout: "RolloutData") -> dict[str, float]:
        """Run one gradient update on collected rollout data.

        Returns dict of scalar metrics (e.g. {"policy_loss": ..., "value_loss": ...}).
        """
        ...
```

`DroneAgent` is a type alias for `AdversarialAgent` — same interface, distinct name for clarity.

**Why Protocol?** An SB3 PPO wrapper satisfying this interface is ~15 lines. An ABC would force
inheritance and prevent drop-in replacement.

---

## 2. RolloutData

```python
@dataclass
class RolloutData:
    """Flat batch of rollout experience. Not a replay buffer."""

    obs: Tensor         # [T, n_envs, obs_dim]
    actions: Tensor     # [T, n_envs, action_dim]
    rewards: Tensor     # [T, n_envs]
    dones: Tensor       # [T, n_envs]       — terminated | truncated
    log_probs: Tensor   # [T, n_envs]
    values: Tensor      # [T, n_envs]
    last_value: Tensor  # [n_envs]           — bootstrap value for GAE at end of rollout
```

Produced by the training loop's rollout phase. Consumed by `agent.update()`.
All tensors on the same device as the env (GPU in production, CPU in tests).

**`last_value`:** the critic's value estimate at `obs[T]` (the observation after the last stored
transition). Required by GAE to bootstrap the advantage computation for incomplete trajectories.
Computed by calling `agent.act(obs)` one last time after the rollout collection loop and extracting
the value estimate.

**`RolloutBuffer`** is an internal helper class in `training_loop.py` that pre-allocates the tensors,
provides `store(obs, action, reward, done, log_prob, value)` per step, and returns a `RolloutData`
via `get(last_value)`. It is not part of the public API.

---

## 3. ActorCritic Network

```python
class ActorCritic(nn.Module):
    """Shared-trunk MLP with Gaussian policy head and value head."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None: ...

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for rollout collection (no grad needed by caller).
        Returns (action, log_prob, value).
        Action is sampled from N(mean, exp(log_std)).
        """

    def evaluate(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for PPO update.
        Returns (log_prob, entropy, value) for the given (obs, action) pair.
        """
```

**Architecture:**
- Shared MLP trunk: `obs_dim → hidden[0] → ... → hidden[-1]`, with `tanh` activations.
- Policy head: `hidden[-1] → action_dim` (mean) + learnable `log_std` parameter (per-action-dim).
- Value head: `hidden[-1] → 1`.
- Action sampling: `Normal(mean, exp(log_std))`, no squashing. Actions are **clamped to the
  action space bounds** inside `act()` before returning. For the adversary, this means clamping
  to `adversary_action_space.low/high`. For the drone, clamping to `action_space.low/high`.
  This is simpler than tanh squashing and avoids log-prob correction.

Used for both drone and adversary agents (separate instances, different obs_dim/action_dim).

**Action bounds enforcement:** the training loop clamps sampled actions to the agent's action
space bounds before passing them to `env.step()`. This guarantees that `set_perturbation_values()`
never receives out-of-bounds values, even though `set_value()` does not clip to bounds internally.

---

## 4. PPOAgent

```python
class PPOAgent:
    """Reference PPO implementation satisfying AdversarialAgent protocol."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        mini_batch_size: int = 64,
        hidden_sizes: tuple[int, ...] = (64, 64),
        device: str | torch.device = "cpu",
    ) -> None: ...

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Select action (eval mode, no grad). Returns (action, log_prob, value)."""

    def update(self, rollout: RolloutData) -> dict[str, float]:
        """PPO update with GAE. Returns {"policy_loss", "value_loss", "entropy"}."""
```

**PPO update internals:**
1. Compute GAE advantages from `rollout.rewards`, `rollout.values`, `rollout.dones`, `gamma`, `gae_lambda`.
2. Normalize advantages (mean=0, std=1).
3. For `n_epochs` passes over the data:
   a. Shuffle and split into mini-batches of size `mini_batch_size`.
   b. For each mini-batch:
      - `log_prob, entropy, value = network.evaluate(obs, action)`
      - Ratio `r = exp(log_prob - old_log_prob)`
      - Clipped surrogate: `min(r * A, clip(r, 1-eps, 1+eps) * A)`
      - Value loss: `MSE(value, returns)` (returns = advantages + values)
      - Total loss: `-policy_loss + value_coef * value_loss - entropy_coef * entropy`
      - Gradient step with `max_grad_norm` clipping.
4. Return mean losses over all mini-batches.

**Seed:** `torch.manual_seed(seed)` called in `__init__` if `seed` is provided.

---

## 5. CurriculumScheduler

```python
class CurriculumScheduler:
    """Callback that increases curriculum_scale over training."""

    def __init__(
        self,
        env: RobustDroneEnv,
        schedule: str | Callable[[int, int], float] = "linear",
        start_scale: float = 0.0,
        end_scale: float = 1.0,
    ) -> None: ...

    def __call__(self, step: int, total_steps: int, **kwargs) -> None:
        """Called as a training callback. Computes scale and calls env.set_curriculum_scale()."""
```

**Built-in schedules:**

| Name | Formula |
|---|---|
| `"linear"` | `start + (end - start) * (step / total)` |
| `"cosine"` | `end - (end - start) * (1 + cos(pi * step / total)) / 2` |
| `"step"` | `start` if `step < total/2` else `end` |

**Custom:** pass any `Callable[[current_step, total_steps], float]`.

The scheduler is a callback plugged into `TrainConfig.callbacks`. It is fully decoupled from the
training loop — the loop calls `callback(step, total_steps)` after each rollout update.

**DR mode only:** `set_curriculum_scale()` has no effect in adversarial mode (by design in
`RobustDroneEnv`). The scheduler is useful for DR baselines or mixed DR→ADV curricula.

---

## 6. Training Loop

### 6.1 Modes

```python
class TrainingMode(str, Enum):
    DR = "dr"        # drone only, DR perturbations (5-tuple env.step)
    RARL = "rarl"    # alternating updates (Pinto et al. 2017)
    RAP = "rap"      # joint updates (Vinitsky et al. 2020)
```

### 6.2 Config

```python
@dataclass
class TrainConfig:
    mode: TrainingMode = TrainingMode.RARL
    total_timesteps: int = 1_000_000
    rollout_steps: int = 2048
    adversary_warmup_steps: int = 0
    adversary_obs_key: str = "privileged_obs"  # key in info dict
    callbacks: list[Callable] = field(default_factory=list)
    log_interval: int = 1
    seed: int | None = None
```

### 6.3 Entry point

```python
def train(
    env: AdversarialEnv | RobustDroneEnv,
    drone_agent: AdversarialAgent,
    adversary_agent: AdversarialAgent | None = None,
    config: TrainConfig = TrainConfig(),
) -> dict[str, list[float]]:
    """Main training entry point.

    Args:
        env: AdversarialEnv for RARL/RAP, RobustDroneEnv for DR.
        drone_agent: Protagonist agent (controls drone action).
        adversary_agent: Adversary agent (controls perturbation values). None for DR mode.
        config: Training configuration.

    Returns:
        Dictionary of metric lists: {"drone_reward": [...], "adv_reward": [...], ...}
    """
```

### 6.4 Loop pseudocode

```
seed(config.seed)
obs, info = env.reset()
global_step = 0
rollout_count = 0
metrics = defaultdict(list)

while global_step < config.total_timesteps:
    # --- Collect rollout ---
    drone_rollout = RolloutBuffer(rollout_steps, n_envs)
    adv_rollout = RolloutBuffer(rollout_steps, n_envs)

    for t in range(config.rollout_steps):
        drone_action, d_logp, d_val = drone_agent.act(obs)

        adv_obs = info.get(config.adversary_obs_key, zeros(n_envs, 0))
        if adversary_agent and global_step >= config.adversary_warmup_steps:
            adv_action, a_logp, a_val = adversary_agent.act(adv_obs)
        else:
            adv_action = zeros(n_envs, adversary_dim)
            a_logp = zeros(n_envs)
            a_val = zeros(n_envs)

        # Clamp actions to space bounds
        drone_action = drone_action.clamp(action_space.low, action_space.high)
        if adversary_agent:
            adv_action = adv_action.clamp(adv_action_space.low, adv_action_space.high)

        if isinstance(env, AdversarialEnv):
            obs_next, d_rew, a_rew, term, trunc, info = env.step(drone_action, adv_action)
        else:
            obs_next, d_rew, term, trunc, info = env.step(drone_action)
            a_rew = zeros(n_envs)

        # Store in rollout buffers
        drone_rollout.store(obs, drone_action, d_rew, term | trunc, d_logp, d_val)
        adv_rollout.store(adv_obs, adv_action, a_rew, term | trunc, a_logp, a_val)

        # Auto-reset done envs
        done_mask = term | trunc
        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs_next_reset, info = env.reset(env_ids=done_ids)
            obs_next[done_ids] = obs_next_reset[done_ids]

        obs = obs_next
        global_step += n_envs

    # Bootstrap value for GAE
    _, _, d_last_val = drone_agent.act(obs)
    _, _, a_last_val = adversary_agent.act(adv_obs) if adversary_agent else (None, None, zeros(n_envs))

    # --- Update agents (mode-dependent) ---
    drone_metrics = {}
    adv_metrics = {}

    if config.mode == DR or config.mode == RAP:
        # DR: always update drone. RAP: always update both.
        drone_metrics = drone_agent.update(drone_rollout.get(d_last_val))
        if adversary_agent and global_step >= config.adversary_warmup_steps:
            adv_metrics = adversary_agent.update(adv_rollout.get(a_last_val))

    elif config.mode == RARL:
        # RARL: alternate — even rollouts update drone, odd update adversary
        if rollout_count % 2 == 0:
            drone_metrics = drone_agent.update(drone_rollout.get(d_last_val))
        else:
            if adversary_agent and global_step >= config.adversary_warmup_steps:
                adv_metrics = adversary_agent.update(adv_rollout.get(a_last_val))

    rollout_count += 1

    # --- Callbacks ---
    for cb in config.callbacks:
        cb(step=global_step, total_steps=config.total_timesteps,
           drone_metrics=drone_metrics, adv_metrics=adv_metrics)

    # --- Log ---
    metrics["drone_reward"].append(drone_rollout.mean_reward())
    metrics["adv_reward"].append(adv_rollout.mean_reward())

return dict(metrics)
```

### 6.5 Mode differences

| Aspect | DR | RARL | RAP |
|---|---|---|---|
| Env type | `RobustDroneEnv` | `AdversarialEnv` | `AdversarialEnv` |
| adversary_agent | `None` | Required | Required |
| step() return | 5-tuple | 6-tuple | 6-tuple |
| Drone update | Every rollout | Every rollout | Every rollout |
| Adversary update | N/A | Alternating with drone | Every rollout (joint) |
| Perturbation values | `sample()` via DR | `set_value()` via adversary | `set_value()` via adversary |

**RARL alternation:** in RARL mode, the drone is updated on even rollouts (`rollout_count % 2 == 0`)
and the adversary on odd rollouts. Both agents still **collect experience** every rollout (both
call `act()` each step); only the gradient update is alternated. This is consistent with Pinto
et al. 2017 where the protagonist and adversary are trained in alternation, not simultaneously.
"Frozen" means "no `update()` call", not "no `act()` call".

### 6.6 Auto-reset

The training loop handles per-env resets explicitly:

```python
done_mask = terminated | truncated
if done_mask.any():
    done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
    obs_reset, info_reset = env.reset(env_ids=done_ids)
    obs[done_ids] = obs_reset[done_ids]
```

**Prerequisite:** `AdversarialEnv.reset()` must be extended to accept `env_ids` and delegate
to `env.reset(env_ids=env_ids)`. This is a small change to the existing `AdversarialEnv`:

```python
def reset(self, *, env_ids=None, seed=None, options=None):
    return self.env.reset(env_ids=env_ids, seed=seed, options=options)
```

This avoids accessing `env.env` (internal attribute bypass) and maintains proper encapsulation.

---

## 7. `privileged_obs_dim` property

Added to `RobustDroneEnv` so the adversary can construct its network at init time:

```python
@property
def privileged_obs_dim(self) -> int:
    """Total dimension of privileged observations (all observable perturbations)."""
    return sum(
        math.prod(p.dimension)
        for p in self._perturbation_cfg.all_perturbations()
        if p.observable
    )
```

This is a static computation (no `reset()` needed), consistent with how `observation_space`
is built in `__init__`.

---

## 8. Error handling

| Situation | Exception |
|---|---|
| RARL/RAP mode without adversary_agent | `ValueError` |
| DR mode with adversary_agent not None | `ValueError` |
| env is not `AdversarialEnv` in RARL/RAP mode | `TypeError` |
| env is `AdversarialEnv` in DR mode | `TypeError` |
| agent does not satisfy Protocol | `TypeError` (runtime_checkable) |

---

## 9. Usage example

```python
from genesis_robust_rl.envs import AdversarialEnv, PerturbationConfig, SensorConfig
from genesis_robust_rl.adversarial import (
    PPOAgent, train, TrainConfig, TrainingMode, CurriculumScheduler,
)

# Build env
env = MyDroneEnv(scene=scene, drone=drone, n_envs=64, perturbation_cfg=cfg, ...)
adv_env = AdversarialEnv(env, adversary_targets=["mass_shift", "wind_gust"])

# Create agents
drone = PPOAgent(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    device="cuda",
)
adversary = PPOAgent(
    obs_dim=env.privileged_obs_dim,
    action_dim=adv_env.adversary_action_space.shape[0],
    device="cuda",
)

# Curriculum (DR mode only, ignored in adversarial)
curriculum = CurriculumScheduler(env, schedule="linear")

# Train RARL
metrics = train(
    env=adv_env,
    drone_agent=drone,
    adversary_agent=adversary,
    config=TrainConfig(
        mode=TrainingMode.RARL,
        total_timesteps=1_000_000,
        adversary_warmup_steps=50_000,
        callbacks=[curriculum],
    ),
)
```

---

## Summary

| Component | File | Role |
|---|---|---|
| `AdversarialAgent` | `protocol.py` | Duck-typed agent interface |
| `RolloutData` | `protocol.py` | Flat rollout batch container |
| `ActorCritic` | `networks.py` | Shared MLP actor-critic |
| `PPOAgent` | `ppo_agent.py` | Reference PPO satisfying the Protocol |
| `CurriculumScheduler` | `curriculum.py` | Callback for DR curriculum |
| `train()` | `training_loop.py` | DR / RARL / RAP training entry point |
| `TrainConfig` | `training_loop.py` | Training hyperparameters |
| `TrainingMode` | `training_loop.py` | Enum: DR / RARL / RAP |
