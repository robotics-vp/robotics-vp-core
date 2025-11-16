"""
Soft Actor-Critic (SAC) with novelty-weighted sampling.

Features:
- Twin critics with target networks
- Tanh-squashed Gaussian actor
- Automatic entropy tuning
- Replay buffer with novelty-based prioritization
- Integrated encoder training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class Actor(nn.Module):
    """
    Gaussian policy with tanh squashing: π_θ(a|z).

    Output: mean, logstd → sample → tanh → [0,1]²
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)

        # Initialize small weights for stable early training
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.logstd_head.weight, gain=0.01)

    def forward(self, latent):
        """
        Args:
            latent: [B, latent_dim]

        Returns:
            mean: [B, action_dim]
            logstd: [B, action_dim]
        """
        features = self.net(latent)
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)

        # Clip logstd for numerical stability
        logstd = torch.clamp(logstd, -20, 2)

        return mean, logstd

    def sample(self, latent, deterministic=False):
        """
        Sample action with reparameterization trick.

        Returns:
            action: [B, action_dim] in [0, 1]
            logprob: [B] log probability
        """
        mean, logstd = self.forward(latent)
        std = torch.exp(logstd)

        if deterministic:
            action_raw = mean
        else:
            # Reparameterization trick
            eps = torch.randn_like(mean)
            action_raw = mean + std * eps

        # Tanh squashing
        action = torch.tanh(action_raw)

        # Log probability with change of variables
        # log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
        logprob = (-0.5 * (eps.pow(2) + 2 * np.log(np.sqrt(2 * np.pi))) - logstd).sum(dim=1)
        logprob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1)

        # Scale to [0, 1] from [-1, 1]
        action = (action + 1) / 2

        return action, logprob


class Critic(nn.Module):
    """
    Q-function: Q_ϕ(z, a).

    Twin critics for double Q-learning (reduces overestimation).
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent, action):
        """
        Args:
            latent: [B, latent_dim]
            action: [B, action_dim]

        Returns:
            q1, q2: [B, 1] Q-values from both critics
        """
        x = torch.cat([latent, action], dim=1)
        return self.q1(x), self.q2(x)


class NoveltyReplayBuffer:
    """
    Replay buffer with novelty-weighted sampling.

    Samples are prioritized by: weight = novelty × |TD_error|
    """

    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, novelty=1.0):
        """Add transition to buffer."""
        self.buffer.append((obs, action, reward, next_obs, done, novelty))
        self.priorities.append(1.0)  # Initial priority (will update)

    def sample(self, batch_size, use_prioritization=True):
        """
        Sample batch with novelty-based prioritization.

        Returns:
            batch: Tuple of (obs, actions, rewards, next_obs, dones, novelties, indices)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if use_prioritization and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            priorities = np.abs(priorities) + 1e-6  # Avoid zero
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        batch = [self.buffer[i] for i in indices]

        obs = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_obs = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        novelties = np.array([t[5] for t in batch])

        return obs, actions, rewards, next_obs, dones, novelties, indices

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent with encoder and novelty weighting.

    Architecture:
    - Encoder f_ψ: obs → latent (128D)
    - Actor π_θ: latent → action (2D, [speed, care])
    - Critics Q_ϕ1, Q_ϕ2: (latent, action) → Q-value
    """

    def __init__(self, encoder, latent_dim=128, action_dim=2,
                 lr=3e-4, gamma=0.995, tau=5e-3,
                 buffer_capacity=int(1e6), batch_size=1024,
                 target_entropy=None, device='cpu'):
        """
        Args:
            encoder: EncoderWithAuxiliaries instance
            latent_dim: Latent dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            buffer_capacity: Replay buffer size
            batch_size: Mini-batch size
            target_entropy: Target entropy for automatic tuning (-action_dim default)
            device: torch device
        """
        self.encoder = encoder.to(device)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)  # Ensure scalar
        self.tau = float(tau)  # Ensure scalar
        self.batch_size = batch_size
        self.device = device

        # Actor
        self.actor = Actor(latent_dim, action_dim).to(device)

        # Critics
        self.critic = Critic(latent_dim, action_dim).to(device)
        self.critic_target = Critic(latent_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy temperature (automatic tuning)
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay_buffer = NoveltyReplayBuffer(capacity=buffer_capacity)

        # Training metrics
        self.training_steps = 0

    def select_action(self, obs, novelty=None, deterministic=False):
        """
        Select action from policy.

        Args:
            obs: Observation dict or tensor
            novelty: Novelty score (for buffer storage)
            deterministic: Use mean action (no sampling)

        Returns:
            action: numpy array [action_dim]
            novelty: novelty score (pass-through)
        """
        with torch.no_grad():
            if isinstance(obs, dict):
                obs_tensor = self._obs_to_tensor(obs)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            latent = self.encoder.encode(obs_tensor)
            action, _ = self.actor.sample(latent, deterministic=deterministic)

        return action.cpu().numpy()[0], novelty

    def store_transition(self, obs, action, reward, next_obs, done, novelty=1.0):
        """Store transition in replay buffer."""
        # Convert obs dict to array
        if isinstance(obs, dict):
            obs = np.array([obs['t'], obs['completed'], obs['attempts'], obs['errors']])
        if isinstance(next_obs, dict):
            next_obs = np.array([next_obs['t'], next_obs['completed'],
                                next_obs['attempts'], next_obs['errors']])

        self.replay_buffer.push(obs, action, reward, next_obs, done, novelty)

    def update(self, aux_loss_weight={'consistency': 0.1, 'contrastive': 0.1}):
        """
        SAC update with encoder auxiliary losses.

        Returns:
            metrics: Dict of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        obs, actions, rewards, next_obs, dones, novelties, indices = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        novelties_t = torch.FloatTensor(novelties).to(self.device)

        # Encode observations
        latent = self.encoder.encode(obs_t)
        next_latent = self.encoder.encode(next_obs_t)

        # --- Critic Update ---
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_logprobs = self.actor.sample(next_latent)

            # Target Q-values (use minimum of twin critics)
            q1_target, q2_target = self.critic_target(next_latent, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # Bellman backup with entropy
            target_value = rewards_t + (1 - dones_t) * self.gamma * (q_target - self.alpha * next_logprobs.unsqueeze(1))

        # Current Q-values
        q1, q2 = self.critic(latent.detach(), actions_t)  # Detach encoder gradients

        # Novelty-weighted critic loss
        weights = torch.clamp(novelties_t, 0.5, 2.0)
        weights = weights / (weights.mean() + 1e-6)

        critic_loss = (F.mse_loss(q1, target_value, reduction='none') * weights.unsqueeze(1)).mean()
        critic_loss += (F.mse_loss(q2, target_value, reduction='none') * weights.unsqueeze(1)).mean()

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update priorities
        td_errors = (target_value - q1).abs().detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors * novelties)

        # --- Actor Update ---
        # Sample actions from current policy
        new_actions, logprobs = self.actor.sample(latent)

        # Q-values for new actions
        q1_new, q2_new = self.critic(latent.detach(), new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss (maximize Q - α*entropy)
        actor_loss = (self.alpha * logprobs.unsqueeze(1) - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Entropy Temperature Update ---
        alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # --- Encoder Update (with auxiliary losses) ---
        # Re-encode to get fresh gradients for encoder training
        latent_fresh = self.encoder.encode(obs_t)
        next_latent_fresh = self.encoder.encode(next_obs_t)

        encoder_loss = 0

        # Consistency loss (only for encoders that support it)
        if hasattr(self.encoder, 'use_consistency') and self.encoder.use_consistency:
            consistency_loss = self.encoder.compute_consistency_loss(latent_fresh, next_latent_fresh)
            encoder_loss += aux_loss_weight['consistency'] * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0)

        # Contrastive loss (only for encoders that support it)
        if hasattr(self.encoder, 'use_contrastive') and self.encoder.use_contrastive:
            contrastive_loss = self.encoder.compute_contrastive_loss(latent_fresh)
            encoder_loss += aux_loss_weight['contrastive'] * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0)

        if encoder_loss > 0:
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()

        # --- Soft Update Target Networks ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.training_steps += 1

        # Return metrics
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'mean_novelty': novelties.mean(),
            'mean_weight': weights.mean().item(),
            'q_mean': q_new.mean().item()
        }

    def _obs_to_tensor(self, obs_dict):
        """Convert observation dict to tensor."""
        features = np.array([obs_dict['t'], obs_dict['completed'],
                            obs_dict['attempts'], obs_dict['errors']])
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def save(self, path):
        """Save agent state."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'training_steps': self.training_steps
        }, path)

    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        self.training_steps = checkpoint['training_steps']
