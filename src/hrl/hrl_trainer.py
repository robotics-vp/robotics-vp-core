"""
HRL Training Infrastructure.

Provides training loops for:
- Low-level skill policies (π_L)
- High-level controller (π_H)
- End-to-end hierarchical training
"""

import os
import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .skills import SkillID, SkillParams, SkillTrajectory
from .low_level_policy import LowLevelSkillPolicy, ScriptedSkillPolicy
from .high_level_controller import HighLevelController, ScriptedHighLevelController
from .skill_termination import SkillTerminationDetector, SkillRewardShaper


class SkillTrainer:
    """
    Trains low-level skill policies using PPO.
    """

    def __init__(
        self,
        env,
        skill_id,
        policy=None,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.env = env
        self.skill_id = skill_id
        self.device = device

        if policy is None:
            policy = LowLevelSkillPolicy()
        self.policy = policy.to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Helpers
        self.termination_detector = SkillTerminationDetector()
        self.reward_shaper = SkillRewardShaper()

        # Statistics
        self.episode_rewards = []
        self.episode_successes = []

    def collect_trajectories(self, n_steps=2048, skill_params=None):
        """
        Collect trajectories for current skill.

        Args:
            n_steps: Number of steps to collect
            skill_params: SkillParams (if None, uses default)

        Returns:
            buffer: dict with collected data
        """
        if skill_params is None:
            skill_params = SkillParams.default_for_skill(self.skill_id)

        # Convert params to tensor
        skill_params_arr = skill_params.to_array()
        skill_params_t = torch.FloatTensor(skill_params_arr).to(self.device)
        skill_id_t = torch.tensor(self.skill_id, dtype=torch.long, device=self.device)

        # Initialize buffers
        obs_buf = []
        action_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        log_prob_buf = []
        skill_id_buf = []
        skill_params_buf = []

        obs, info = self.env.reset()
        episode_reward = 0
        step_in_skill = 0

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).to(self.device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.act(
                    obs_t, skill_id_t, skill_params_t
                )

            # Step environment
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            next_obs, env_reward, env_done, truncated, info = self.env.step(action_np)

            # Compute shaped reward for skill
            shaped_reward, reward_components = self.reward_shaper.compute_reward(
                self.skill_id,
                obs,
                next_obs,
                action_np,
                info,
                skill_params,
                step_in_skill
            )

            # Check skill termination
            skill_done, skill_success, reason = self.termination_detector.is_done(
                self.skill_id,
                next_obs,
                info,
                step_in_skill,
                skill_params.timeout_steps
            )

            # Store transition
            obs_buf.append(obs)
            action_buf.append(action_np)
            reward_buf.append(shaped_reward)
            done_buf.append(skill_done or env_done)
            value_buf.append(value.item() if isinstance(value, torch.Tensor) else value)
            log_prob_buf.append(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob)
            skill_id_buf.append(self.skill_id)
            skill_params_buf.append(skill_params_arr)

            episode_reward += shaped_reward
            step_in_skill += 1

            # Handle episode/skill end
            if skill_done or env_done:
                self.episode_rewards.append(episode_reward)
                self.episode_successes.append(skill_success)

                # Reset
                obs, info = self.env.reset()
                episode_reward = 0
                step_in_skill = 0

                # Randomize skill params slightly
                skill_params = self._randomize_params(skill_params)
                skill_params_arr = skill_params.to_array()
                skill_params_t = torch.FloatTensor(skill_params_arr).to(self.device)
            else:
                obs = next_obs

        # Convert to tensors
        buffer = {
            'observations': torch.FloatTensor(np.array(obs_buf)).to(self.device),
            'actions': torch.FloatTensor(np.array(action_buf)).to(self.device),
            'rewards': torch.FloatTensor(np.array(reward_buf)).to(self.device),
            'dones': torch.FloatTensor(np.array(done_buf)).to(self.device),
            'values': torch.FloatTensor(np.array(value_buf)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(log_prob_buf)).to(self.device),
            'skill_ids': torch.LongTensor(np.array(skill_id_buf)).to(self.device),
            'skill_params': torch.FloatTensor(np.array(skill_params_buf)).to(self.device),
        }

        return buffer

    def _randomize_params(self, params):
        """Add small random perturbations to skill parameters."""
        new_params = SkillParams(
            target_clearance=params.target_clearance + np.random.normal(0, 0.01),
            approach_speed=np.clip(params.approach_speed + np.random.normal(0, 0.05), 0.1, 1.0),
            grasp_force=np.clip(params.grasp_force + np.random.normal(0, 0.05), 0.1, 1.0),
            retract_speed=np.clip(params.retract_speed + np.random.normal(0, 0.05), 0.1, 1.0),
            timeout_steps=params.timeout_steps
        )
        return new_params

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            if dones[t]:
                next_value = 0

            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update_policy(self, buffer, n_epochs=10, batch_size=64):
        """
        Update policy using PPO.

        Args:
            buffer: Collected trajectory data
            n_epochs: Number of PPO epochs
            batch_size: Mini-batch size

        Returns:
            metrics: Training metrics
        """
        # Compute advantages
        advantages, returns = self.compute_gae(
            buffer['rewards'],
            buffer['values'],
            buffer['dones']
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # PPO epochs
        for _ in range(n_epochs):
            # Random indices
            indices = torch.randperm(len(buffer['observations']))

            # Mini-batch updates
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch
                obs_batch = buffer['observations'][batch_indices]
                action_batch = buffer['actions'][batch_indices]
                old_log_prob_batch = buffer['log_probs'][batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]
                skill_id_batch = buffer['skill_ids'][batch_indices]
                skill_params_batch = buffer['skill_params'][batch_indices]

                # Evaluate actions
                new_log_prob, entropy, value = self.policy.evaluate_actions(
                    obs_batch, skill_id_batch, skill_params_batch, action_batch
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value, return_batch)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'mean_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'success_rate': np.mean(self.episode_successes[-100:]) if self.episode_successes else 0,
        }

        return metrics

    def train(self, total_steps=100000, steps_per_update=2048, log_interval=10):
        """
        Train skill policy.

        Args:
            total_steps: Total training steps
            steps_per_update: Steps to collect before each update
            log_interval: Log every N updates
        """
        print(f"Training {SkillID.name(self.skill_id)} policy...")

        updates = 0
        steps_collected = 0

        while steps_collected < total_steps:
            # Collect trajectories
            buffer = self.collect_trajectories(steps_per_update)
            steps_collected += steps_per_update

            # Update policy
            metrics = self.update_policy(buffer)
            updates += 1

            # Log
            if updates % log_interval == 0:
                print(f"Update {updates}, Steps: {steps_collected}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Mean Reward: {metrics['mean_episode_reward']:.4f}")
                print(f"  Success Rate: {metrics['success_rate']:.2%}")

        return metrics

    def save(self, path):
        """Save trained policy."""
        self.policy.save(path)
        print(f"Saved {SkillID.name(self.skill_id)} policy to {path}")


class HighLevelTrainer:
    """
    Trains high-level controller using PPO.

    The high-level controller selects skills to execute.
    """

    def __init__(
        self,
        env,
        pi_h,
        pi_l,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        device='cpu'
    ):
        self.env = env
        self.pi_h = pi_h.to(device)
        self.pi_l = pi_l  # Can be scripted or learned
        self.device = device

        self.optimizer = optim.Adam(self.pi_h.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.termination_detector = SkillTerminationDetector()

        # Statistics
        self.task_successes = []
        self.episode_lengths = []

    def collect_episode(self, max_skills=10):
        """
        Collect single episode with hierarchical execution.

        Returns:
            episode_data: dict with high-level decisions
            task_success: bool
        """
        obs, info = self.env.reset()

        # High-level data
        hl_obs = []
        hl_skill_ids = []
        hl_skill_params = []
        hl_rewards = []
        hl_dones = []
        hl_values = []
        hl_log_probs = []

        skill_count = 0
        total_reward = 0
        task_success = False

        while skill_count < max_skills:
            # High-level decision
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                skill_id, skill_params_t, log_prob, value = self.pi_h.select_skill(obs_t)

            skill_params = SkillParams.from_array(skill_params_t.squeeze(0).cpu().numpy())

            # Execute skill
            skill_reward = 0
            step_in_skill = 0

            while True:
                # Low-level action
                if isinstance(self.pi_l, ScriptedSkillPolicy):
                    action = self.pi_l.act(obs, skill_id, skill_params)
                else:
                    obs_t = torch.FloatTensor(obs).to(self.device)
                    skill_id_t = torch.tensor(skill_id, dtype=torch.long, device=self.device)
                    skill_params_t = torch.FloatTensor(skill_params.to_array()).to(self.device)
                    with torch.no_grad():
                        action, _, _ = self.pi_l.act(obs_t, skill_id_t, skill_params_t)
                    action = action.cpu().numpy()

                # Step environment
                next_obs, env_reward, env_done, truncated, info = self.env.step(action)
                skill_reward += env_reward
                step_in_skill += 1

                # Check skill termination
                skill_done, skill_success, reason = self.termination_detector.is_done(
                    skill_id, next_obs, info, step_in_skill, skill_params.timeout_steps
                )

                obs = next_obs

                if skill_done or env_done:
                    break

            # Record high-level decision
            hl_obs.append(obs)
            hl_skill_ids.append(skill_id)
            hl_skill_params.append(skill_params.to_array())
            hl_rewards.append(skill_reward)
            hl_dones.append(env_done or info.get('success', False))
            hl_values.append(value.item())
            hl_log_probs.append(log_prob.item())

            total_reward += skill_reward
            skill_count += 1

            # Check task completion
            if info.get('success', False):
                task_success = True
                break

            if env_done:
                break

        episode_data = {
            'observations': np.array(hl_obs),
            'skill_ids': np.array(hl_skill_ids),
            'skill_params': np.array(hl_skill_params),
            'rewards': np.array(hl_rewards),
            'dones': np.array(hl_dones),
            'values': np.array(hl_values),
            'log_probs': np.array(hl_log_probs),
        }

        self.task_successes.append(task_success)
        self.episode_lengths.append(skill_count)

        return episode_data, task_success

    def train(self, n_episodes=1000, batch_size=32, log_interval=50):
        """Train high-level controller."""
        print("Training High-Level Controller...")

        for episode in range(n_episodes):
            episode_data, success = self.collect_episode()

            # Simple policy gradient update (for demonstration)
            # In full implementation, use PPO with multiple episodes

            if len(episode_data['rewards']) > 0:
                # Compute discounted returns
                returns = []
                G = 0
                for r in reversed(episode_data['rewards']):
                    G = r + self.gamma * G
                    returns.insert(0, G)

                returns = torch.FloatTensor(returns).to(self.device)
                log_probs = torch.FloatTensor(episode_data['log_probs']).to(self.device)
                values = torch.FloatTensor(episode_data['values']).to(self.device)

                # Advantage
                advantages = returns - values

                # Policy gradient loss
                policy_loss = -(log_probs * advantages.detach()).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (episode + 1) % log_interval == 0:
                success_rate = np.mean(self.task_successes[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Mean Skills: {mean_length:.1f}")

        return {
            'success_rate': np.mean(self.task_successes),
            'mean_episode_length': np.mean(self.episode_lengths),
        }


# Add F import for functional operations
if TORCH_AVAILABLE:
    import torch.nn.functional as F
