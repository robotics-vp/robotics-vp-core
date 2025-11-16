"""
Low-Level Skill Policy (π_L) for HRL.

A conditioned policy that produces actions given:
- Current observation
- Skill ID
- Skill parameters
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Stub for non-torch environments
    class nn:
        class Module:
            pass

from .skills import SkillID, SkillParams


class LowLevelSkillPolicy(nn.Module if TORCH_AVAILABLE else object):
    """
    Conditioned skill policy.

    Input: (obs, skill_id, skill_params)
    Output: action (continuous EE velocity command)

    Architecture:
    - Skill embedding layer
    - Parameter encoder
    - Shared feature encoder
    - Gaussian action head
    """

    def __init__(
        self,
        obs_dim=13,
        action_dim=3,
        num_skills=6,
        hidden_dim=256,
        skill_embed_dim=32,
        param_embed_dim=16,
        log_std_init=-0.5,
        log_std_min=-5.0,
        log_std_max=2.0
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LowLevelSkillPolicy")

        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_skills = num_skills
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Skill embedding
        self.skill_embedding = nn.Embedding(num_skills, skill_embed_dim)

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(5, param_embed_dim),
            nn.ReLU(),
        )

        # Main feature encoder
        input_dim = obs_dim + skill_embed_dim + param_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head (mean and log_std)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize log_std
        nn.init.constant_(self.log_std_head.bias, log_std_init)

        # Value head (for actor-critic)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, skill_id, skill_params):
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim) - state observation
            skill_id: (batch,) - integer skill ID (long tensor)
            skill_params: (batch, 5) - continuous parameters

        Returns:
            action_mean: (batch, action_dim)
            action_log_std: (batch, action_dim)
            value: (batch, 1)
        """
        # Encode skill ID
        skill_emb = self.skill_embedding(skill_id)  # (batch, skill_embed_dim)

        # Encode parameters
        param_emb = self.param_encoder(skill_params)  # (batch, param_embed_dim)

        # Concatenate inputs
        x = torch.cat([obs, skill_emb, param_emb], dim=-1)

        # Encode features
        features = self.encoder(x)

        # Action distribution
        action_mean = self.mean_head(features)
        action_log_std = self.log_std_head(features)
        action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)

        # Value estimate
        value = self.value_head(features)

        return action_mean, action_log_std, value

    def get_action_distribution(self, obs, skill_id, skill_params):
        """Get action distribution."""
        mean, log_std, value = self.forward(obs, skill_id, skill_params)
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist, value

    def act(self, obs, skill_id, skill_params, deterministic=False):
        """
        Sample action from policy.

        Args:
            obs: (obs_dim,) or (batch, obs_dim)
            skill_id: int or (batch,)
            skill_params: (5,) or (batch, 5)
            deterministic: bool

        Returns:
            action: (action_dim,) or (batch, action_dim)
            log_prob: () or (batch,)
            value: () or (batch,)
        """
        # Handle single samples
        single_sample = obs.dim() == 1
        if single_sample:
            obs = obs.unsqueeze(0)
            skill_id = torch.tensor([skill_id], device=obs.device, dtype=torch.long)
            skill_params = skill_params.unsqueeze(0)

        dist, value = self.get_action_distribution(obs, skill_id, skill_params)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # Clamp to valid action range
        action = torch.clamp(action, -1.0, 1.0)

        # Log probability
        log_prob = dist.log_prob(action).sum(dim=-1)

        if single_sample:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, value

    def evaluate_actions(self, obs, skill_id, skill_params, actions):
        """
        Evaluate log probability and entropy of actions.

        Used for PPO training.
        """
        dist, value = self.get_action_distribution(obs, skill_id, skill_params)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value.squeeze(-1)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'num_skills': self.num_skills,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            num_skills=checkpoint['num_skills']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


class ScriptedSkillPolicy:
    """
    Scripted fallback for skill policies.

    Provides handcrafted actions for each skill without neural network.
    Useful for generating initial training data and baselines.
    """

    def __init__(self):
        self.handle_pos = np.array([0.0, -0.42, 0.65])
        self.safe_pos = np.array([-0.3, 0.0, 0.8])
        self.vase_default_pos = np.array([0.3, 0.0, 0.8])

    def act(self, obs, skill_id, skill_params=None):
        """
        Generate scripted action for skill.

        Args:
            obs: (13,) numpy array
            skill_id: int
            skill_params: SkillParams or None

        Returns:
            action: (3,) numpy array
        """
        if skill_params is None:
            skill_params = SkillParams.default_for_skill(skill_id)

        ee_pos = obs[0:3]
        ee_vel = obs[3:6]
        drawer_frac = obs[6]
        vase_pos = obs[7:10]
        min_clearance = obs[11]

        if skill_id == SkillID.LOCATE_DRAWER:
            # Move to see drawer (minimal movement, mostly observation)
            target = np.array([0.0, -0.3, 0.7])
            action = self._move_towards(ee_pos, target, 0.3)

        elif skill_id == SkillID.LOCATE_VASE:
            # Stay still and observe
            action = np.zeros(3)

        elif skill_id == SkillID.PLAN_SAFE_APPROACH:
            # Compute safe waypoint (single step, no movement)
            action = np.zeros(3)

        elif skill_id == SkillID.GRASP_HANDLE:
            # Move towards handle
            action = self._move_towards(
                ee_pos, self.handle_pos, skill_params.approach_speed
            )
            # Apply vase avoidance
            action = self._avoid_vase(ee_pos, vase_pos, action, skill_params)

        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            # Pull drawer open
            pull_dir = np.array([0.0, -skill_params.pull_speed, 0.0])
            # Apply vase avoidance
            action = self._avoid_vase(ee_pos, vase_pos, pull_dir, skill_params)

        elif skill_id == SkillID.RETRACT_SAFE:
            # Retract to safe position
            action = self._move_towards(
                ee_pos, self.safe_pos, skill_params.retract_speed
            )

        else:
            action = np.zeros(3)

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _move_towards(self, current, target, speed):
        """Generate action to move towards target."""
        direction = target - current
        dist = np.linalg.norm(direction)

        if dist < 0.01:
            return np.zeros(3)

        direction = direction / dist
        return direction * speed

    def _avoid_vase(self, ee_pos, vase_pos, action, params):
        """Apply repulsive force from vase."""
        ee_to_vase = ee_pos - vase_pos
        distance = np.linalg.norm(ee_to_vase)

        if distance < params.target_clearance:
            # Apply repulsive force
            repulsion_strength = (params.target_clearance - distance) / params.target_clearance
            repulsion_dir = ee_to_vase / (distance + 1e-6)
            repulsion = repulsion_dir * repulsion_strength * 0.5
            action = action + repulsion

        return np.clip(action, -1.0, 1.0)
