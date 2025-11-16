"""
Minimal PPO implementation with novelty-weighted loss.

Integrates diffusion-based novelty signals to weight training samples
by their informational value for economic data valuation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class ActorCritic(nn.Module):
    """
    Simple actor-critic network for continuous control.

    Actor outputs mean of Gaussian policy.
    Critic outputs state value estimate.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        Args:
            obs: [B, obs_dim] tensor

        Returns:
            action_mean: [B, action_dim]
            action_std: [B, action_dim]
            value: [B, 1]
        """
        features = self.shared(obs)

        # Actor output
        action_mean = torch.sigmoid(self.actor_mean(features))  # [0, 1] for speed
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)

        # Critic output
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        action_mean, action_std, value = self.forward(obs)

        if deterministic:
            return action_mean, value

        # Sample from Gaussian
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)

        # Clip to valid range [0, 1]
        action = torch.clamp(action, 0.0, 1.0)

        return action, action_logprob, value

    def evaluate_actions(self, obs, actions):
        """Evaluate log prob and entropy of actions under current policy."""
        action_mean, action_std, value = self.forward(obs)

        dist = torch.distributions.Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action_logprobs, value, entropy


class PPOAgent:
    """
    PPO agent with novelty-weighted loss for data valuation.

    Key features:
    - Novelty weighting: samples with higher novelty get higher weight
    - Economic valuation: tracks ΔMPL per batch for data pricing
    - Modular: ready for video diffusion latents
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        novelty_alpha=1.0,  # Weight novelty in sample weighting
        novelty_beta=0.0,   # Bias term for weighting
        device='cpu'
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.novelty_alpha = novelty_alpha
        self.novelty_beta = novelty_beta
        self.device = torch.device(device)

        # Networks
        self.ac = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        # Trajectory buffer
        self.clear_buffer()

        # Statistics
        self.update_count = 0

    def clear_buffer(self):
        """Clear trajectory buffer."""
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'logprobs': [],
            'dones': [],
            'novelty': []  # For novelty weighting
        }

    def select_action(self, obs, novelty=None):
        """
        Select action from policy.

        Args:
            obs: dict or array observation
            novelty: Optional novelty score for this observation

        Returns:
            action: numpy array
            logprob: float
            value: float
        """
        # Convert obs dict to tensor
        if isinstance(obs, dict):
            obs_tensor = self._obs_to_tensor(obs)
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, logprob, value = self.ac.get_action(obs_tensor)

        # Store in buffer
        self.buffer['obs'].append(obs_tensor.cpu())
        self.buffer['actions'].append(action.cpu())
        self.buffer['logprobs'].append(logprob.cpu())
        self.buffer['values'].append(value.cpu())
        self.buffer['novelty'].append(novelty if novelty is not None else 1.0)

        return action.cpu().numpy()[0], logprob.item(), value.item()

    def store_transition(self, reward, done):
        """Store reward and done flag."""
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)

    def _obs_to_tensor(self, obs_dict):
        """Convert observation dict to tensor."""
        # Simple feature extraction from dishwashing env
        features = np.array([
            obs_dict['t'] / 3600.0,  # Normalize time
            obs_dict['completed'] / 200.0,  # Normalize completed
            obs_dict['attempts'] / 300.0,  # Normalize attempts
            obs_dict['errors'] / 50.0  # Normalize errors
        ], dtype=np.float32)
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def compute_advantages(self, last_value=0.0):
        """
        Compute GAE advantages.

        Returns:
            advantages: [T] tensor
            returns: [T] tensor
        """
        rewards = torch.FloatTensor(self.buffer['rewards']).to(self.device)
        values = torch.cat(self.buffer['values']).to(self.device)
        dones = torch.FloatTensor(self.buffer['dones']).to(self.device)

        # Append last value for bootstrapping
        last_value_tensor = torch.FloatTensor([[last_value]]).to(self.device)
        values = torch.cat([values, last_value_tensor])

        advantages = []
        gae = 0

        # GAE computation (backward pass)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values[:-1]

        return advantages, returns

    def compute_sample_weights(self, advantages, novelty_scores, clip_range=(0.5, 2.0)):
        """
        Compute sample weights from advantages and novelty.

        Valuation proxy: v_i = |A_i| × Novelty_i
        Weights: w_i = σ(α * v_i + β)
        Then clipped to [0.5, 2.0] and normalized to mean≈1

        Args:
            advantages: [T] tensor
            novelty_scores: [T] tensor or list
            clip_range: (min, max) for weight clipping

        Returns:
            weights: [T] tensor, clipped and normalized
            valuations: [T] tensor (raw valuation proxy)
            weight_stats: dict with mean, p90, etc.
        """
        if not isinstance(novelty_scores, torch.Tensor):
            novelty_scores = torch.FloatTensor(novelty_scores).to(self.device)

        # Valuation proxy: |A| * novelty
        valuations = torch.abs(advantages) * novelty_scores

        # Convert to weights via sigmoid
        weights_raw = torch.sigmoid(
            self.novelty_alpha * valuations + self.novelty_beta
        )

        # Clip to prevent extreme values
        weights = torch.clamp(weights_raw, clip_range[0], clip_range[1])

        # Normalize to mean≈1 (prevents hijacking PPO)
        weights = weights / (weights.mean().detach() + 1e-6)

        # Compute statistics
        weight_stats = {
            'mean': weights.mean().item(),
            'p90': weights.quantile(0.9).item(),
            'min': weights.min().item(),
            'max': weights.max().item()
        }

        return weights, valuations, weight_stats

    def update(self, last_value=0.0):
        """
        PPO update with novelty-weighted loss.

        Returns:
            dict with training metrics
        """
        # Compute advantages
        advantages, returns = self.compute_advantages(last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare batch data
        obs = torch.cat(self.buffer['obs']).to(self.device)
        actions = torch.cat(self.buffer['actions']).to(self.device)
        old_logprobs = torch.cat(self.buffer['logprobs']).to(self.device)
        novelty_scores = self.buffer['novelty']

        # Compute sample weights (with clipping & normalization)
        weights, valuations, weight_stats = self.compute_sample_weights(advantages, novelty_scores)

        # Training metrics
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],  # Track KL for stability
            'mean_novelty': torch.tensor(novelty_scores).mean().item(),
            'mean_weight': weight_stats['mean'],
            'p90_weight': weight_stats['p90'],
            'mean_valuation': valuations.mean().item()
        }

        # PPO epochs
        for epoch in range(self.epochs):
            # Shuffle indices for mini-batches
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # Mini-batch data
                obs_batch = obs[batch_idx]
                actions_batch = actions[batch_idx]
                old_logprobs_batch = old_logprobs[batch_idx]
                advantages_batch = advantages[batch_idx]
                returns_batch = returns[batch_idx]
                weights_batch = weights[batch_idx]

                # Evaluate actions under current policy
                logprobs, values, entropy = self.ac.evaluate_actions(
                    obs_batch, actions_batch
                )

                # KL divergence (for monitoring stability)
                kl = (old_logprobs_batch - logprobs).mean()

                # PPO clipped objective
                ratio = torch.exp(logprobs - old_logprobs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_batch

                # Policy loss (weighted by novelty)
                policy_loss = -(torch.min(surr1, surr2) * weights_batch).mean()

                # Value loss (weighted by novelty)
                value_loss = ((returns_batch - values.squeeze()) ** 2 * weights_batch).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Log metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.mean().item())
                metrics['total_loss'].append(loss.item())
                metrics['kl_divergence'].append(kl.item())

        # Average metrics
        for key in ['policy_loss', 'value_loss', 'entropy', 'total_loss', 'kl_divergence']:
            metrics[key] = np.mean(metrics[key])

        self.update_count += 1
        self.clear_buffer()

        return metrics

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
