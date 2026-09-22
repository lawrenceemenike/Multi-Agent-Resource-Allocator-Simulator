"""MAPPO (Multi-Agent PPO) with Centralized Critic and Persona Conditioning."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Dict, List, Tuple, Optional, Any

from maras.environment.economy_env import EconomyEnv


class Actor(nn.Module):
    """Decentralized Actor Network conditioned on agent observation and persona."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        feat = self.net(obs)
        mu = self.mu_head(feat)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0))
        dist = Normal(mu, std)
        return dist, mu

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self.forward(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """Centralized Critic Network taking the global multi-agent state."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class MAPPOTrainer:
    """Multi-Agent PPO Trainer with Centralized Critic."""

    def __init__(
        self,
        env: EconomyEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        agent_sample = env.possible_agents[0]
        self.obs_dim = env.observation_spaces[agent_sample].shape[0]
        self.action_dim = env.action_spaces[agent_sample].shape[0]
        self.state_dim = self.obs_dim * len(env.possible_agents)

        # Networks
        self.actor = Actor(self.obs_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = CentralizedCritic(self.state_dim, hidden_dim=hidden_dim * 2).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.training_history: List[Dict[str, float]] = []

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Generalized Advantage Estimation (GAE) and target returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def train(
        self,
        total_steps: int = 20000,
        rollout_length: int = 200,
        ppo_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Runs the complete MAPPO training loop over environment steps."""
        obs, _ = self.env.reset()
        global_step = 0
        update_count = 0

        while global_step < total_steps:
            # Storage for rollout
            b_obs: List[Dict[str, np.ndarray]] = []
            b_states: List[np.ndarray] = []
            b_actions: List[Dict[str, np.ndarray]] = []
            b_logprobs: List[Dict[str, float]] = []
            b_values: List[float] = []
            b_rewards: List[Dict[str, float]] = []
            b_dones: List[bool] = []

            # 1. Collect rollout
            for _ in range(rollout_length):
                state = self.env.get_global_state()
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    val = self.critic(state_t).item()

                actions_dict: Dict[str, np.ndarray] = {}
                logprobs_dict: Dict[str, float] = {}

                for agent_id in self.env.agents:
                    agent_obs = obs[agent_id]
                    obs_t = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        act_t, lp_t = self.actor.get_action(obs_t)
                    actions_dict[agent_id] = act_t.squeeze(0).cpu().numpy()
                    logprobs_dict[agent_id] = lp_t.item()

                next_obs, rewards, terminations, truncations, _ = self.env.step(actions_dict)
                done = any(terminations.values()) or any(truncations.values()) or not self.env.agents

                b_obs.append(obs)
                b_states.append(state)
                b_actions.append(actions_dict)
                b_logprobs.append(logprobs_dict)
                b_values.append(val)
                b_rewards.append(rewards)
                b_dones.append(done)

                global_step += len(self.env.possible_agents)

                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

            # Final next value for GAE
            last_state = self.env.get_global_state()
            last_state_t = torch.as_tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                next_val = self.critic(last_state_t).item()

            # 2. Compute GAE and Flatten Data
            # Mean agent reward per step for centralized critic targets
            step_rewards = np.array(
                [np.mean(list(r.values())) if r else 0.0 for r in b_rewards], dtype=np.float32
            )
            step_values = np.array(b_values, dtype=np.float32)
            step_dones = np.array(b_dones, dtype=np.float32)

            advantages, returns = self.compute_gae(step_rewards, step_values, step_dones, next_val)

            # Flatten agent transitions for Actor and Critic updates
            flat_obs = []
            flat_actions = []
            flat_old_logprobs = []
            flat_advantages = []

            for t in range(rollout_length):
                t_adv = advantages[t]
                for agent_id in self.env.possible_agents:
                    if agent_id in b_actions[t]:
                        flat_obs.append(b_obs[t][agent_id])
                        flat_actions.append(b_actions[t][agent_id])
                        flat_old_logprobs.append(b_logprobs[t][agent_id])
                        flat_advantages.append(t_adv)

            flat_obs_t = torch.as_tensor(np.array(flat_obs), dtype=torch.float32, device=self.device)
            flat_act_t = torch.as_tensor(np.array(flat_actions), dtype=torch.float32, device=self.device)
            flat_old_lp_t = torch.as_tensor(np.array(flat_old_logprobs), dtype=torch.float32, device=self.device)
            flat_adv_t = torch.as_tensor(np.array(flat_advantages), dtype=torch.float32, device=self.device)
            # Normalize advantages
            flat_adv_t = (flat_adv_t - flat_adv_t.mean()) / (flat_adv_t.std() + 1e-8)

            states_t = torch.as_tensor(np.array(b_states), dtype=torch.float32, device=self.device)
            returns_t = torch.as_tensor(np.array(returns), dtype=torch.float32, device=self.device)

            # 3. PPO Optimization Epochs
            total_samples = len(flat_obs)
            last_p_loss = 0.0
            last_v_loss = 0.0
            last_ent = 0.0

            for _ in range(ppo_epochs):
                indices = np.random.permutation(total_samples)
                for start in range(0, total_samples, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    b_o = flat_obs_t[batch_idx]
                    b_a = flat_act_t[batch_idx]
                    b_olp = flat_old_lp_t[batch_idx]
                    b_adv = flat_adv_t[batch_idx]

                    # Actor loss
                    new_lp, entropy = self.actor.evaluate_action(b_o, b_a)
                    ratio = torch.exp(new_lp - b_olp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss on state transitions
                    c_idx = np.random.randint(0, len(states_t), size=min(len(batch_idx), len(states_t)))
                    v_pred = self.critic(states_t[c_idx])
                    value_loss = nn.functional.mse_loss(v_pred, returns_t[c_idx])

                    # Total Loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                    last_p_loss = policy_loss.item()
                    last_v_loss = value_loss.item()
                    last_ent = entropy.mean().item()

            update_count += 1
            avg_reward = float(np.mean(step_rewards))
            self.training_history.append(
                {
                    "step": global_step,
                    "policy_loss": last_p_loss,
                    "value_loss": last_v_loss,
                    "entropy": last_ent,
                    "avg_reward": avg_reward,
                }
            )

        return {
            "total_steps": global_step,
            "updates": update_count,
            "final_policy_loss": last_p_loss,
            "final_value_loss": last_v_loss,
            "history": self.training_history,
        }
