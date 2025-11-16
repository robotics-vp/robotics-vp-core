"""
High-Level Controller (π_H) for HRL.

Selects skills and parameters based on state/vision information.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

from .skills import SkillID, SkillParams


class HighLevelController(nn.Module if TORCH_AVAILABLE else object):
    """
    High-level controller that selects skills and parameters.

    Input: (obs, z_V, risk_map, affordance_map)
    Output: (skill_id, skill_params)

    The controller operates at a lower frequency than π_L,
    selecting skills that execute for multiple time steps.
    """

    def __init__(
        self,
        obs_dim=13,
        z_v_dim=128,
        risk_map_dim=256,  # 16x16 flattened
        affordance_dim=256,
        num_skills=6,
        hidden_dim=256,
        use_vision=False
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for HighLevelController")

        super().__init__()

        self.obs_dim = obs_dim
        self.z_v_dim = z_v_dim
        self.num_skills = num_skills
        self.use_vision = use_vision

        # Feature encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        if use_vision:
            self.z_v_encoder = nn.Sequential(
                nn.Linear(z_v_dim, 64),
                nn.ReLU(),
            )

            self.risk_encoder = nn.Sequential(
                nn.Linear(risk_map_dim, 32),
                nn.ReLU(),
            )

            self.affordance_encoder = nn.Sequential(
                nn.Linear(affordance_dim, 32),
                nn.ReLU(),
            )

            combined_dim = 64 + 64 + 32 + 32
        else:
            combined_dim = 64

        # Combined encoder
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Skill selector (categorical)
        self.skill_head = nn.Linear(hidden_dim, num_skills)

        # Parameter predictor (continuous, bounded [0, 1])
        self.param_head = nn.Linear(hidden_dim, 5)

        # Value head for actor-critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, z_v=None, risk_map=None, affordance_map=None):
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)
            z_v: (batch, z_v_dim) or None
            risk_map: (batch, 16, 16) or None
            affordance_map: (batch, 16, 16) or None

        Returns:
            skill_logits: (batch, num_skills)
            skill_params: (batch, 5) in [0, 1]
            value: (batch, 1)
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Encode observations
        obs_feat = self.obs_encoder(obs)

        if self.use_vision:
            # Encode vision (if available)
            if z_v is not None:
                z_v_feat = self.z_v_encoder(z_v)
            else:
                z_v_feat = torch.zeros(batch_size, 64, device=device)

            # Encode risk map (if available)
            if risk_map is not None:
                risk_flat = risk_map.view(batch_size, -1)
                risk_feat = self.risk_encoder(risk_flat)
            else:
                risk_feat = torch.zeros(batch_size, 32, device=device)

            # Encode affordance (if available)
            if affordance_map is not None:
                aff_flat = affordance_map.view(batch_size, -1)
                aff_feat = self.affordance_encoder(aff_flat)
            else:
                aff_feat = torch.zeros(batch_size, 32, device=device)

            # Combine all features
            x = torch.cat([obs_feat, z_v_feat, risk_feat, aff_feat], dim=-1)
        else:
            x = obs_feat

        # Encode combined features
        features = self.combined(x)

        # Output heads
        skill_logits = self.skill_head(features)
        skill_params = torch.sigmoid(self.param_head(features))  # [0, 1]
        value = self.value_head(features)

        return skill_logits, skill_params, value

    def select_skill(
        self,
        obs,
        z_v=None,
        risk_map=None,
        affordance_map=None,
        temperature=1.0,
        deterministic=False
    ):
        """
        Select skill and parameters.

        Args:
            obs: (obs_dim,) or (batch, obs_dim)
            z_v: Optional vision latent
            risk_map: Optional risk map
            affordance_map: Optional affordance map
            temperature: Sampling temperature
            deterministic: Use argmax instead of sampling

        Returns:
            skill_id: int or (batch,)
            skill_params: (5,) or (batch, 5)
            log_prob: () or (batch,)
            value: () or (batch,)
        """
        # Handle single samples
        single_sample = obs.dim() == 1
        if single_sample:
            obs = obs.unsqueeze(0)
            if z_v is not None:
                z_v = z_v.unsqueeze(0)
            if risk_map is not None:
                risk_map = risk_map.unsqueeze(0)
            if affordance_map is not None:
                affordance_map = affordance_map.unsqueeze(0)

        skill_logits, skill_params, value = self.forward(
            obs, z_v, risk_map, affordance_map
        )

        # Sample or select skill
        if deterministic:
            skill_id = skill_logits.argmax(dim=-1)
            skill_probs = F.softmax(skill_logits / temperature, dim=-1)
            log_prob = torch.log(skill_probs.gather(1, skill_id.unsqueeze(-1))).squeeze(-1)
        else:
            skill_probs = F.softmax(skill_logits / temperature, dim=-1)
            dist = Categorical(skill_probs)
            skill_id = dist.sample()
            log_prob = dist.log_prob(skill_id)

        if single_sample:
            skill_id = skill_id.item()
            skill_params = skill_params.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return skill_id, skill_params, log_prob, value

    def evaluate_skills(self, obs, skill_ids, z_v=None, risk_map=None, affordance_map=None):
        """
        Evaluate log probability and entropy of skill selections.

        Used for PPO training.
        """
        skill_logits, skill_params, value = self.forward(
            obs, z_v, risk_map, affordance_map
        )

        skill_probs = F.softmax(skill_logits, dim=-1)
        dist = Categorical(skill_probs)

        log_prob = dist.log_prob(skill_ids)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1), skill_params

    def get_skill_probabilities(self, obs, z_v=None, risk_map=None, affordance_map=None):
        """Get probability distribution over skills."""
        skill_logits, _, _ = self.forward(obs, z_v, risk_map, affordance_map)
        return F.softmax(skill_logits, dim=-1)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'obs_dim': self.obs_dim,
            'z_v_dim': self.z_v_dim,
            'num_skills': self.num_skills,
            'use_vision': self.use_vision,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            obs_dim=checkpoint['obs_dim'],
            z_v_dim=checkpoint['z_v_dim'],
            num_skills=checkpoint['num_skills'],
            use_vision=checkpoint.get('use_vision', False)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


class ScriptedHighLevelController:
    """
    Scripted high-level controller for baselines.

    Uses fixed skill sequence without learning.
    """

    def __init__(self):
        self.skill_sequence = [
            SkillID.LOCATE_DRAWER,
            SkillID.LOCATE_VASE,
            SkillID.PLAN_SAFE_APPROACH,
            SkillID.GRASP_HANDLE,
            SkillID.OPEN_WITH_CLEARANCE,
            SkillID.RETRACT_SAFE,
        ]
        self.current_skill_idx = 0
        self.params = {}

    def reset(self):
        """Reset controller state."""
        self.current_skill_idx = 0
        self.params = {}

    def select_skill(self, obs=None, done_previous=True):
        """
        Select next skill in sequence.

        Args:
            obs: Current observation (optional for adaptive params)
            done_previous: Whether previous skill is done

        Returns:
            skill_id: int
            skill_params: SkillParams
            done: bool - whether sequence is complete
        """
        if done_previous and self.current_skill_idx < len(self.skill_sequence):
            skill_id = self.skill_sequence[self.current_skill_idx]

            # Adaptive parameters based on observation
            if obs is not None:
                skill_params = self._compute_adaptive_params(skill_id, obs)
            else:
                skill_params = SkillParams.default_for_skill(skill_id)

            self.current_skill_idx += 1

            sequence_done = self.current_skill_idx >= len(self.skill_sequence)

            return skill_id, skill_params, sequence_done

        # No more skills
        return -1, SkillParams(), True

    def _compute_adaptive_params(self, skill_id, obs):
        """Compute skill parameters based on current state."""
        params = SkillParams.default_for_skill(skill_id)

        # Adapt clearance based on vase proximity
        min_clearance = obs[11]
        if min_clearance < 0.2:
            # Vase is close, be more careful
            params.target_clearance = 0.2
            params.approach_speed = 0.6
            params.pull_speed = 0.4

        return params

    def get_current_skill_name(self):
        """Get name of current skill."""
        if self.current_skill_idx > 0 and self.current_skill_idx <= len(self.skill_sequence):
            skill_id = self.skill_sequence[self.current_skill_idx - 1]
            return SkillID.name(skill_id)
        return "NONE"


class HierarchicalAgent:
    """
    Complete hierarchical agent combining π_H and π_L.

    Manages skill execution, switching, and parameter passing.
    """

    def __init__(
        self,
        high_level_controller,
        low_level_policy,
        termination_detector,
        use_scripted_hl=False
    ):
        """
        Args:
            high_level_controller: HighLevelController or ScriptedHighLevelController
            low_level_policy: LowLevelSkillPolicy or ScriptedSkillPolicy
            termination_detector: SkillTerminationDetector
            use_scripted_hl: Use scripted high-level controller
        """
        self.pi_h = high_level_controller
        self.pi_l = low_level_policy
        self.termination_detector = termination_detector
        self.use_scripted_hl = use_scripted_hl

        # State
        self.current_skill_id = None
        self.current_skill_params = None
        self.skill_step_count = 0
        self.total_step_count = 0

    def reset(self):
        """Reset agent state."""
        self.current_skill_id = None
        self.current_skill_params = None
        self.skill_step_count = 0
        self.total_step_count = 0

        if self.use_scripted_hl:
            self.pi_h.reset()

    def act(self, obs, info=None, deterministic=False):
        """
        Select action given observation.

        Manages skill switching and low-level action generation.

        Args:
            obs: Current observation (numpy or torch)
            info: Info dict from environment
            deterministic: Use deterministic actions

        Returns:
            action: (3,) action array
            skill_info: dict with skill metadata
        """
        if info is None:
            info = {}

        # Check if we need to select a new skill
        need_new_skill = False

        if self.current_skill_id is None:
            need_new_skill = True
        else:
            # Check if current skill is done
            done, success, reason = self.termination_detector.is_done(
                self.current_skill_id,
                obs if isinstance(obs, np.ndarray) else obs.cpu().numpy(),
                info,
                self.skill_step_count,
                max_steps=self.current_skill_params.timeout_steps if self.current_skill_params else 100
            )
            if done:
                need_new_skill = True

        # Select new skill if needed
        if need_new_skill:
            if self.use_scripted_hl:
                skill_id, skill_params, seq_done = self.pi_h.select_skill(
                    obs if isinstance(obs, np.ndarray) else obs.cpu().numpy(),
                    done_previous=True
                )
                if skill_id == -1:
                    # Task complete
                    return np.zeros(3, dtype=np.float32), {
                        'skill_id': -1,
                        'skill_name': 'COMPLETE',
                        'task_done': True
                    }
            else:
                # Use learned controller
                obs_t = obs if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs)
                skill_id, skill_params_t, log_prob, value = self.pi_h.select_skill(
                    obs_t,
                    deterministic=deterministic
                )
                # Convert params tensor to SkillParams
                skill_params = SkillParams.from_array(skill_params_t.detach().cpu().numpy())

            self.current_skill_id = skill_id
            self.current_skill_params = skill_params
            self.skill_step_count = 0

        # Generate low-level action
        # Check if using scripted policy (no torch, has simple act method)
        if not TORCH_AVAILABLE or not hasattr(self.pi_l, 'forward'):
            action = self.pi_l.act(
                obs if isinstance(obs, np.ndarray) else obs.cpu().numpy(),
                self.current_skill_id,
                self.current_skill_params
            )
            log_prob = 0.0
        else:
            # Use learned policy
            obs_t = obs if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs)
            skill_id_t = torch.tensor(self.current_skill_id, dtype=torch.long)
            skill_params_t = torch.FloatTensor(self.current_skill_params.to_array())

            action_t, log_prob, value = self.pi_l.act(
                obs_t, skill_id_t, skill_params_t, deterministic=deterministic
            )
            action = action_t.detach().cpu().numpy() if isinstance(action_t, torch.Tensor) else action_t
            log_prob = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob

        self.skill_step_count += 1
        self.total_step_count += 1

        skill_info = {
            'skill_id': self.current_skill_id,
            'skill_name': SkillID.name(self.current_skill_id),
            'skill_step': self.skill_step_count,
            'total_step': self.total_step_count,
            'skill_params': self.current_skill_params,
            'log_prob': log_prob,
            'task_done': False
        }

        return action, skill_info
