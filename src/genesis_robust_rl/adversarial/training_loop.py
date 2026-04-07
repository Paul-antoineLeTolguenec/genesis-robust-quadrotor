"""Training loop for DR, RARL (alternating), and RAP (joint) modes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from torch import Tensor

from genesis_robust_rl.adversarial.protocol import AdversarialAgent, RolloutData
from genesis_robust_rl.envs.adversarial_env import AdversarialEnv
from genesis_robust_rl.envs.robust_drone_env import RobustDroneEnv


class TrainingMode(str, Enum):
    """Training paradigm selector."""

    DR = "dr"
    RARL = "rarl"
    RAP = "rap"


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    mode: TrainingMode = TrainingMode.RARL
    total_timesteps: int = 1_000_000
    rollout_steps: int = 2048
    adversary_warmup_steps: int = 0
    adversary_obs_key: str = "privileged_obs"
    callbacks: list[Callable[..., Any]] = field(default_factory=list)
    log_interval: int = 1
    seed: int | None = None


class _RolloutBuffer:
    """Internal buffer that accumulates transitions during rollout collection."""

    def __init__(
        self,
        rollout_steps: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.obs = torch.zeros(rollout_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(rollout_steps, n_envs, action_dim, device=device)
        self.rewards = torch.zeros(rollout_steps, n_envs, device=device)
        self.dones = torch.zeros(rollout_steps, n_envs, device=device)
        self.log_probs = torch.zeros(rollout_steps, n_envs, device=device)
        self.values = torch.zeros(rollout_steps, n_envs, device=device)
        self._ptr = 0

    def store(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        log_prob: Tensor,
        value: Tensor,
    ) -> None:
        """Store one transition."""
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.rewards[t] = reward
        self.dones[t] = done.float()
        self.log_probs[t] = log_prob
        self.values[t] = value
        self._ptr += 1

    def get(self, last_value: Tensor) -> RolloutData:
        """Return collected data as RolloutData."""
        return RolloutData(
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            log_probs=self.log_probs,
            values=self.values,
            last_value=last_value,
        )

    def mean_reward(self) -> float:
        """Mean reward across all steps and envs."""
        return self.rewards.mean().item()


def train(
    env: AdversarialEnv | RobustDroneEnv,
    drone_agent: AdversarialAgent,
    adversary_agent: AdversarialAgent | None = None,
    config: TrainConfig = TrainConfig(),
) -> dict[str, list[float]]:
    """Main training entry point for DR / RARL / RAP.

    Args:
        env: AdversarialEnv for RARL/RAP, RobustDroneEnv for DR.
        drone_agent: Protagonist agent.
        adversary_agent: Adversary agent. None for DR mode.
        config: Training configuration.

    Returns:
        Dictionary of metric time series.
    """
    # Validate mode/env/agent consistency
    is_adv = isinstance(env, AdversarialEnv)
    if config.mode in (TrainingMode.RARL, TrainingMode.RAP):
        if not is_adv:
            raise TypeError(f"RARL/RAP mode requires AdversarialEnv, got {type(env).__name__}")
        if adversary_agent is None:
            raise ValueError("RARL/RAP mode requires adversary_agent")
    if config.mode == TrainingMode.DR:
        if is_adv:
            raise TypeError("DR mode requires RobustDroneEnv, not AdversarialEnv")
        if adversary_agent is not None:
            raise ValueError("DR mode does not accept adversary_agent")

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Resolve dimensions
    n_envs = env.n_envs
    device = env.device
    obs_dim = env.observation_space.shape[0]
    drone_action_dim = env.action_space.shape[0]
    adv_action_dim = env.adversary_action_space.shape[0] if is_adv else 0
    adv_obs_dim = env.privileged_obs_dim if hasattr(env, "privileged_obs_dim") else 0

    # Action space bounds (as tensors for clamping)
    drone_low = torch.as_tensor(env.action_space.low, device=device)
    drone_high = torch.as_tensor(env.action_space.high, device=device)
    if is_adv:
        adv_low = torch.as_tensor(env.adversary_action_space.low, device=device)
        adv_high = torch.as_tensor(env.adversary_action_space.high, device=device)

    # Init
    obs, info = env.reset()
    global_step = 0
    rollout_count = 0
    metrics: dict[str, list[float]] = defaultdict(list)

    while global_step < config.total_timesteps:
        # Allocate rollout buffers
        drone_buf = _RolloutBuffer(config.rollout_steps, n_envs, obs_dim, drone_action_dim, device)
        adv_buf = (
            _RolloutBuffer(
                config.rollout_steps, n_envs, max(adv_obs_dim, 1), adv_action_dim, device
            )
            if is_adv
            else None
        )

        for _t in range(config.rollout_steps):
            drone_action, d_logp, d_val = drone_agent.act(obs)
            drone_action = drone_action.clamp(drone_low, drone_high)

            # Adversary observation
            default_adv_obs = torch.zeros(n_envs, max(adv_obs_dim, 1), device=device)
            adv_obs = info.get(config.adversary_obs_key, default_adv_obs)

            if adversary_agent is not None and global_step >= config.adversary_warmup_steps:
                adv_action, a_logp, a_val = adversary_agent.act(adv_obs)
                adv_action = adv_action.clamp(adv_low, adv_high)
            else:
                adv_action = torch.zeros(n_envs, adv_action_dim, device=device)
                a_logp = torch.zeros(n_envs, device=device)
                a_val = torch.zeros(n_envs, device=device)

            # Step
            if is_adv:
                obs_next, d_rew, a_rew, term, trunc, info = env.step(drone_action, adv_action)
            else:
                obs_next, d_rew, term, trunc, info = env.step(drone_action)
                a_rew = torch.zeros(n_envs, device=device)

            done_mask = term | trunc

            # Store
            drone_buf.store(obs, drone_action, d_rew, done_mask, d_logp, d_val)
            if adv_buf is not None:
                adv_buf.store(adv_obs, adv_action, a_rew, done_mask, a_logp, a_val)

            # Auto-reset done envs
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.dim() == 0:
                    done_ids = done_ids.unsqueeze(0)
                obs_reset, info_reset = env.reset(env_ids=done_ids)
                obs_next[done_ids] = obs_reset[done_ids]
                # Merge info from reset
                for k, v in info_reset.items():
                    if isinstance(v, Tensor) and k in info:
                        info[k][done_ids] = v[done_ids]

            obs = obs_next
            global_step += n_envs

        # Bootstrap values for GAE
        _, _, d_last_val = drone_agent.act(obs)
        if adversary_agent is not None:
            _, _, a_last_val = adversary_agent.act(adv_obs)
        else:
            a_last_val = torch.zeros(n_envs, device=device)

        # Update agents (mode-dependent)
        drone_metrics: dict[str, float] = {}
        adv_metrics: dict[str, float] = {}

        adv_ready = (
            adversary_agent is not None
            and adv_buf is not None
            and global_step >= config.adversary_warmup_steps
        )

        if config.mode in (TrainingMode.DR, TrainingMode.RAP):
            drone_metrics = drone_agent.update(drone_buf.get(d_last_val))
            if adv_ready:
                adv_metrics = adversary_agent.update(adv_buf.get(a_last_val))
        elif config.mode == TrainingMode.RARL:
            if rollout_count % 2 == 0:
                drone_metrics = drone_agent.update(drone_buf.get(d_last_val))
            elif adv_ready:
                adv_metrics = adversary_agent.update(adv_buf.get(a_last_val))

        rollout_count += 1

        # Callbacks
        for cb in config.callbacks:
            cb(
                step=global_step,
                total_steps=config.total_timesteps,
                drone_metrics=drone_metrics,
                adv_metrics=adv_metrics,
            )

        # Metrics
        metrics["drone_reward"].append(drone_buf.mean_reward())
        if is_adv and adv_buf is not None:
            metrics["adv_reward"].append(adv_buf.mean_reward())

    return dict(metrics)
