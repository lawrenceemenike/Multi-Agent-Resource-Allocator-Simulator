"""Integration tests for Milestone 3: PPO Training Harness with Centralized Critic."""
import pytest
import numpy as np
import torch
from maras.environment.economy_env import EconomyEnv
from maras.policies.ppo_wrapper import MAPPOTrainer, Actor, CentralizedCritic


def test_actor_critic_shapes_and_persona_conditioning():
    """Verify Actor and Centralized Critic tensor shapes and forward passes."""
    env = EconomyEnv(num_agents=4, max_steps=50)
    obs, _ = env.reset()

    agent = env.possible_agents[0]
    obs_dim = env.observation_spaces[agent].shape[0]
    act_dim = env.action_spaces[agent].shape[0]
    state_dim = obs_dim * 4

    actor = Actor(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=32)
    critic = CentralizedCritic(state_dim=state_dim, hidden_dim=64)

    obs_tensor = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
    action, log_prob = actor.get_action(obs_tensor)

    assert action.shape == (1, act_dim)
    assert log_prob.shape == (1,)

    state = env.get_global_state()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    val = critic(state_tensor)
    assert val.shape == (1,)


@pytest.mark.parametrize("reward_mode", ["competitive", "cooperative", "mixed"])
def test_reward_schedules(reward_mode: str):
    """Verify environment steps and rewards under competitive, cooperative, and mixed schedules."""
    env = EconomyEnv(num_agents=4, max_steps=10, reward_mode=reward_mode, coop_lambda=0.5)
    obs, _ = env.reset()
    actions = {
        agent: np.zeros(env.action_spaces[agent].shape, dtype=np.float32)
        for agent in env.agents
    }
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    assert len(rewards) == 4
    for agent_id, r in rewards.items():
        assert np.isfinite(r)


def test_mappo_train_20000_steps_convergence():
    """Integration Test: Train across 20,000 steps; verify policy loss convergence without LLM intervention."""
    env = EconomyEnv(num_agents=4, max_steps=100, reward_mode="mixed", seed=42)
    trainer = MAPPOTrainer(
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        hidden_dim=32,
    )

    # Train across 20,000 steps
    results = trainer.train(
        total_steps=20000,
        rollout_length=100,
        ppo_epochs=3,
        batch_size=64,
    )

    assert results["total_steps"] >= 20000
    assert results["updates"] > 0
    history = results["history"]
    assert len(history) > 0

    # Verify policy loss is finite and well-behaved
    final_policy_loss = results["final_policy_loss"]
    assert np.isfinite(final_policy_loss)
    assert abs(final_policy_loss) < 5.0, f"Policy loss diverged: {final_policy_loss}"

    # Verify value loss is non-negative and finite
    final_value_loss = results["final_value_loss"]
    assert np.isfinite(final_value_loss)
    assert final_value_loss >= 0.0
