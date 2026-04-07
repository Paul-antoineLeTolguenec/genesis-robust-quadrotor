"""Reference PPO implementation satisfying the AdversarialAgent protocol."""

from __future__ import annotations

import torch
from torch import Tensor

from genesis_robust_rl.adversarial.networks import ActorCritic
from genesis_robust_rl.adversarial.protocol import RolloutData


class PPOAgent:
    """Proximal Policy Optimization agent with GAE.

    Satisfies the AdversarialAgent protocol. Can be used as either
    the drone agent or the adversary agent.
    """

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
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(device)

        self.network = ActorCritic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Select action in inference mode. Returns (action, log_prob, value)."""
        self.network.eval()
        obs = obs.to(self.device)
        action, log_prob, value = self.network.act(obs)
        return action, log_prob, value

    def update(self, rollout: RolloutData) -> dict[str, float]:
        """PPO update with GAE. Returns loss metrics."""
        self.network.train()

        # Move rollout to device
        obs = rollout.obs.to(self.device)
        actions = rollout.actions.to(self.device)
        old_log_probs = rollout.log_probs.to(self.device)
        rewards = rollout.rewards.to(self.device)
        dones = rollout.dones.to(self.device)
        values = rollout.values.to(self.device)
        last_value = rollout.last_value.to(self.device)

        # Compute GAE advantages
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Flatten [T, n_envs, ...] -> [T*n_envs, ...]
        t, n = obs.shape[:2]
        flat_obs = obs.reshape(t * n, -1)
        flat_actions = actions.reshape(t * n, -1)
        flat_old_log_probs = old_log_probs.reshape(t * n)
        flat_advantages = advantages.reshape(t * n)
        flat_returns = returns.reshape(t * n)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        total_samples = t * n
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_samples)
                idx = indices[start:end]

                log_prob, entropy, value = self.network.evaluate(flat_obs[idx], flat_actions[idx])

                # PPO clipped surrogate
                ratio = (log_prob - flat_old_log_probs[idx]).exp()
                adv = flat_advantages[idx]
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (value - flat_returns[idx]).pow(2).mean()

                # Total loss
                loss = (
                    policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def _compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        last_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: [T, n_envs]
            values: [T, n_envs]
            dones: [T, n_envs]
            last_value: [n_envs] — bootstrap value at t=T

        Returns:
            advantages: [T, n_envs]
            returns: [T, n_envs]
        """
        t_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(last_value)

        for t in reversed(range(t_steps)):
            next_value = last_value if t == t_steps - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns
