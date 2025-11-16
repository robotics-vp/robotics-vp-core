# Phase C: HRL + VLA + SIMA Architecture

## Overview

This document defines the stable interfaces for:
- π_L (Low-level skill policies)
- π_H (High-level controller)
- Vision affordance/risk heads
- VLA transformer planner
- SIMA-2 co-agent

All interfaces must remain stable for Isaac Gym port.

---

## 1. Skill System (π_L)

### Skill IDs

```python
class SkillID:
    LOCATE_DRAWER = 0
    LOCATE_VASE = 1
    PLAN_SAFE_APPROACH = 2
    GRASP_HANDLE = 3
    OPEN_WITH_CLEARANCE = 4
    RETRACT_SAFE = 5

    NUM_SKILLS = 6

    @staticmethod
    def name(skill_id):
        names = {
            0: "LOCATE_DRAWER",
            1: "LOCATE_VASE",
            2: "PLAN_SAFE_APPROACH",
            3: "GRASP_HANDLE",
            4: "OPEN_WITH_CLEARANCE",
            5: "RETRACT_SAFE"
        }
        return names.get(skill_id, "UNKNOWN")
```

### Skill Parameters

Each skill can have continuous parameters:

```python
@dataclass
class SkillParams:
    """Parameters passed from π_H to π_L"""
    target_clearance: float = 0.15  # meters
    approach_speed: float = 0.8     # normalized [0, 1]
    grasp_force: float = 0.5        # normalized [0, 1]
    retract_distance: float = 0.3   # meters
    timeout_steps: int = 100        # max steps for skill
```

### Low-Level Policy Interface (π_L)

```python
class LowLevelSkillPolicy(nn.Module):
    """
    Conditioned skill policy.

    Input: (obs, skill_id, skill_params)
    Output: action (continuous EE velocity)
    """

    def __init__(self, obs_dim=13, action_dim=3, num_skills=6):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_skills = num_skills

        # Skill embedding
        self.skill_embedding = nn.Embedding(num_skills, 32)

        # Parameter encoder
        self.param_encoder = nn.Linear(5, 16)  # 5 skill params

        # Main network
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + 32 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Action head (Gaussian)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, skill_id, skill_params):
        """
        Args:
            obs: (batch, obs_dim) - state observation
            skill_id: (batch,) - integer skill ID
            skill_params: (batch, 5) - continuous parameters

        Returns:
            action_mean: (batch, action_dim)
            action_std: (batch, action_dim)
        """
        skill_emb = self.skill_embedding(skill_id)  # (batch, 32)
        param_emb = self.param_encoder(skill_params)  # (batch, 16)

        x = torch.cat([obs, skill_emb, param_emb], dim=-1)
        features = self.encoder(x)

        action_mean = torch.tanh(self.mean_head(features))
        action_std = self.log_std.exp().expand_as(action_mean)

        return action_mean, action_std

    def act(self, obs, skill_id, skill_params, deterministic=False):
        """Sample action from policy."""
        mean, std = self.forward(obs, skill_id, skill_params)
        if deterministic:
            return mean
        else:
            dist = Normal(mean, std)
            return dist.sample().clamp(-1, 1)
```

### Skill Termination Conditions

Each skill has a termination detector:

```python
class SkillTerminationDetector:
    """Detects when a skill has completed."""

    @staticmethod
    def is_done(skill_id, obs, info):
        """
        Args:
            skill_id: Current skill
            obs: Current observation
            info: Info dict from env

        Returns:
            done: bool
            success: bool
        """
        if skill_id == SkillID.LOCATE_DRAWER:
            # Done when EE has line of sight to drawer handle
            drawer_visible = info.get('drawer_handle_visible', True)
            return drawer_visible, drawer_visible

        elif skill_id == SkillID.LOCATE_VASE:
            # Done when vase position is known
            vase_pos = obs[7:10]
            vase_detected = np.linalg.norm(vase_pos) > 0.1
            return vase_detected, vase_detected

        elif skill_id == SkillID.PLAN_SAFE_APPROACH:
            # Done when waypoint is computed (single step)
            return True, True

        elif skill_id == SkillID.GRASP_HANDLE:
            # Done when close to handle
            ee_pos = obs[0:3]
            handle_pos = np.array([0.0, -0.42, 0.65])
            distance = np.linalg.norm(ee_pos - handle_pos)
            grasp_state = obs[12]
            success = distance < 0.05 or grasp_state > 0.5
            return success, success

        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            # Done when drawer >= 90% open
            drawer_frac = obs[6]
            vase_intact = info.get('vase_intact', True)
            success = drawer_frac >= 0.9 and vase_intact
            failure = not vase_intact
            return success or failure, success

        elif skill_id == SkillID.RETRACT_SAFE:
            # Done when EE is at safe position
            ee_pos = obs[0:3]
            safe_pos = np.array([-0.3, 0.0, 0.8])
            distance = np.linalg.norm(ee_pos - safe_pos)
            success = distance < 0.1
            return success, success

        return False, False
```

---

## 2. High-Level Controller (π_H)

### Interface

```python
class HighLevelController(nn.Module):
    """
    Selects skills and parameters given state/vision.

    Input: (obs, z_V, risk_map, affordance_map)
    Output: (skill_id, skill_params)
    """

    def __init__(
        self,
        obs_dim=13,
        z_v_dim=128,
        risk_map_dim=16*16,
        affordance_dim=16*16,
        num_skills=6
    ):
        super().__init__()

        # Feature encoders
        self.obs_encoder = nn.Linear(obs_dim, 64)
        self.z_v_encoder = nn.Linear(z_v_dim, 64)
        self.risk_encoder = nn.Linear(risk_map_dim, 32)
        self.affordance_encoder = nn.Linear(affordance_dim, 32)

        # Combined encoder
        self.combined = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Skill selector (discrete)
        self.skill_head = nn.Linear(256, num_skills)

        # Parameter predictor (continuous)
        self.param_head = nn.Linear(256, 5)  # 5 params

    def forward(self, obs, z_v=None, risk_map=None, affordance_map=None):
        """
        Args:
            obs: (batch, obs_dim)
            z_v: (batch, z_v_dim) or None
            risk_map: (batch, 16, 16) or None
            affordance_map: (batch, 16, 16) or None

        Returns:
            skill_logits: (batch, num_skills)
            skill_params: (batch, 5)
        """
        batch_size = obs.shape[0]

        # Encode observations
        obs_feat = F.relu(self.obs_encoder(obs))

        # Encode vision (if available)
        if z_v is not None:
            z_v_feat = F.relu(self.z_v_encoder(z_v))
        else:
            z_v_feat = torch.zeros(batch_size, 64, device=obs.device)

        # Encode risk map (if available)
        if risk_map is not None:
            risk_flat = risk_map.view(batch_size, -1)
            risk_feat = F.relu(self.risk_encoder(risk_flat))
        else:
            risk_feat = torch.zeros(batch_size, 32, device=obs.device)

        # Encode affordance (if available)
        if affordance_map is not None:
            aff_flat = affordance_map.view(batch_size, -1)
            aff_feat = F.relu(self.affordance_encoder(aff_flat))
        else:
            aff_feat = torch.zeros(batch_size, 32, device=obs.device)

        # Combine
        x = torch.cat([obs_feat, z_v_feat, risk_feat, aff_feat], dim=-1)
        features = self.combined(x)

        # Outputs
        skill_logits = self.skill_head(features)
        skill_params = torch.sigmoid(self.param_head(features))

        return skill_logits, skill_params

    def select_skill(self, obs, z_v=None, risk_map=None, affordance_map=None,
                     temperature=1.0, deterministic=False):
        """Select skill and parameters."""
        skill_logits, skill_params = self.forward(obs, z_v, risk_map, affordance_map)

        if deterministic:
            skill_id = skill_logits.argmax(dim=-1)
        else:
            skill_probs = F.softmax(skill_logits / temperature, dim=-1)
            skill_id = torch.multinomial(skill_probs, 1).squeeze(-1)

        return skill_id, skill_params
```

---

## 3. Vision Affordance Heads

### Risk Map Head

```python
class RiskMapHead(nn.Module):
    """
    Predicts per-pixel fragility/collision risk.

    Input: image features (from encoder)
    Output: risk map [0, 1]
    """

    def __init__(self, in_channels=128, out_size=(16, 16)):
        super().__init__()
        self.out_size = out_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.upsample = nn.Upsample(size=out_size, mode='bilinear')

    def forward(self, features):
        """
        Args:
            features: (batch, in_channels, H, W)

        Returns:
            risk_map: (batch, out_H, out_W) in [0, 1]
        """
        x = self.conv(features)  # (batch, 1, H, W)
        x = self.upsample(x)      # (batch, 1, out_H, out_W)
        return x.squeeze(1)       # (batch, out_H, out_W)
```

### Affordance Head

```python
class AffordanceHead(nn.Module):
    """
    Predicts handle graspability / interaction affordance.

    Input: image features
    Output: affordance map [0, 1]
    """

    def __init__(self, in_channels=128, out_size=(16, 16)):
        super().__init__()
        self.out_size = out_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.upsample = nn.Upsample(size=out_size, mode='bilinear')

    def forward(self, features):
        x = self.conv(features)
        x = self.upsample(x)
        return x.squeeze(1)
```

### No-Go Zone Head

```python
class NoGoZoneHead(nn.Module):
    """
    Predicts binary mask of unsafe regions.

    Input: image features + risk map
    Output: binary no-go mask
    """

    def __init__(self, in_channels=128, threshold=0.7):
        super().__init__()
        self.threshold = threshold
        self.risk_head = RiskMapHead(in_channels)

    def forward(self, features):
        risk_map = self.risk_head(features)
        no_go_mask = (risk_map > self.threshold).float()
        return no_go_mask, risk_map
```

---

## 4. VLA Transformer (Language → Skill Sequence)

### Input Schema

```python
@dataclass
class VLAInput:
    """Input to VLA transformer."""
    instruction: str  # Natural language instruction
    z_v: torch.Tensor  # (z_v_dim,) visual latent
    state: torch.Tensor  # (obs_dim,) optional state
    risk_map: torch.Tensor  # (16, 16) optional risk map
    affordance_map: torch.Tensor  # (16, 16) optional affordance
```

### Output Schema

```python
@dataclass
class VLAPlan:
    """Output from VLA transformer."""
    skill_sequence: List[int]  # [skill_id_1, skill_id_2, ...]
    skill_params: List[torch.Tensor]  # [params_1, params_2, ...]
    timing_horizons: List[int]  # [max_steps_1, max_steps_2, ...]
    confidence: List[float]  # [conf_1, conf_2, ...]
```

### Transformer Architecture

```python
class VLATransformerPlanner(nn.Module):
    """
    Language + Vision → Skill Sequence.

    Based on GPT-2 architecture with multimodal inputs.
    """

    def __init__(
        self,
        vocab_size=50257,  # GPT-2 tokenizer
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=128,
        z_v_dim=128,
        num_skills=6,
        max_plan_length=10
    ):
        super().__init__()

        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Vision encoder
        self.z_v_proj = nn.Linear(z_v_dim, embed_dim)

        # Risk/affordance encoder
        self.risk_proj = nn.Linear(16*16, embed_dim)
        self.affordance_proj = nn.Linear(16*16, embed_dim)

        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Skill sequence decoder (autoregressive)
        self.skill_head = nn.Linear(embed_dim, num_skills)
        self.param_head = nn.Linear(embed_dim, 5)
        self.timing_head = nn.Linear(embed_dim, 1)
        self.confidence_head = nn.Linear(embed_dim, 1)

        # Special tokens
        self.plan_start_token = nn.Parameter(torch.randn(1, embed_dim))
        self.plan_end_token = nn.Parameter(torch.randn(1, embed_dim))

    def encode_instruction(self, token_ids):
        """Encode text instruction."""
        # token_ids: (batch, seq_len)
        seq_len = token_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=token_ids.device)

        text_emb = self.text_embedding(token_ids) + self.pos_embedding(pos_ids)
        return text_emb  # (batch, seq_len, embed_dim)

    def encode_vision(self, z_v, risk_map=None, affordance_map=None):
        """Encode visual features."""
        batch_size = z_v.shape[0]

        z_v_emb = self.z_v_proj(z_v).unsqueeze(1)  # (batch, 1, embed_dim)

        vision_tokens = [z_v_emb]

        if risk_map is not None:
            risk_flat = risk_map.view(batch_size, -1)
            risk_emb = self.risk_proj(risk_flat).unsqueeze(1)
            vision_tokens.append(risk_emb)

        if affordance_map is not None:
            aff_flat = affordance_map.view(batch_size, -1)
            aff_emb = self.affordance_proj(aff_flat).unsqueeze(1)
            vision_tokens.append(aff_emb)

        return torch.cat(vision_tokens, dim=1)  # (batch, num_vis_tokens, embed_dim)

    def forward(self, token_ids, z_v, risk_map=None, affordance_map=None,
                max_plan_length=6):
        """
        Generate skill plan from instruction and vision.

        Returns:
            skill_sequence: (batch, max_plan_length)
            skill_params: (batch, max_plan_length, 5)
            timing: (batch, max_plan_length)
            confidence: (batch, max_plan_length)
        """
        batch_size = token_ids.shape[0]

        # Encode inputs
        text_tokens = self.encode_instruction(token_ids)
        vision_tokens = self.encode_vision(z_v, risk_map, affordance_map)

        # Add start token
        start_token = self.plan_start_token.expand(batch_size, 1, -1)

        # Concatenate all tokens
        # [text_tokens | vision_tokens | start_token]
        all_tokens = torch.cat([text_tokens, vision_tokens, start_token], dim=1)

        # Transform
        features = self.transformer(all_tokens)

        # Decode plan (simplified: use last N positions)
        # In full implementation, this would be autoregressive
        plan_features = features[:, -max_plan_length:, :]

        skill_logits = self.skill_head(plan_features)  # (batch, plan_len, num_skills)
        skill_params = torch.sigmoid(self.param_head(plan_features))
        timing = F.relu(self.timing_head(plan_features)).squeeze(-1) * 100  # scale to steps
        confidence = torch.sigmoid(self.confidence_head(plan_features)).squeeze(-1)

        skill_sequence = skill_logits.argmax(dim=-1)

        return skill_sequence, skill_params, timing, confidence
```

---

## 5. SIMA-2 Co-Agent Protocol

### Interface

```python
class SIMACoAgent:
    """
    SIMA-style co-agent that:
    1. Receives natural language instruction
    2. Generates high-level plan
    3. Produces demonstration trajectories
    4. Emits step-level narrations
    """

    def __init__(self, env, planner='scripted'):
        self.env = env
        self.planner = planner

        # Trajectory storage
        self.frames = []
        self.actions = []
        self.narrations = []
        self.states = []
        self.infos = []

    def reset(self, instruction):
        """Reset agent with new instruction."""
        self.instruction = instruction
        self.plan = self.generate_plan(instruction)
        self.current_step = 0
        self.frames = []
        self.actions = []
        self.narrations = []
        self.states = []
        self.infos = []

    def generate_plan(self, instruction):
        """
        Generate high-level plan from instruction.

        Returns:
            plan: List of (skill_name, params, narration)
        """
        # Parse instruction
        if "drawer" in instruction.lower() and "vase" in instruction.lower():
            # Standard drawer+vase task
            plan = [
                ("LOCATE_DRAWER", {}, "Looking for the drawer"),
                ("LOCATE_VASE", {}, "Identifying the vase position"),
                ("PLAN_SAFE_APPROACH", {"clearance": 0.15}, "Planning safe path around vase"),
                ("GRASP_HANDLE", {}, "Moving to grasp the drawer handle"),
                ("OPEN_WITH_CLEARANCE", {"target_clearance": 0.15}, "Carefully opening drawer while avoiding vase"),
                ("RETRACT_SAFE", {}, "Retracting to safe position"),
            ]
        else:
            # Generic plan
            plan = [("OPEN_WITH_CLEARANCE", {}, "Executing task")]

        return plan

    def step(self, obs, info):
        """
        Generate action and narration for current step.

        Returns:
            action: (3,) numpy array
            narration: str describing current action
            done: bool
        """
        # Record state
        self.states.append(obs)
        self.infos.append(info)

        # Get current skill from plan
        if self.current_step >= len(self.plan):
            return np.zeros(3), "Task complete", True

        skill_name, params, skill_narration = self.plan[self.current_step]

        # Generate action using scripted policy
        action = self._scripted_action(skill_name, obs, params)

        # Generate step-level narration
        narration = self._generate_narration(skill_name, obs, info)

        self.actions.append(action)
        self.narrations.append(narration)

        # Check if skill is done
        if self._skill_done(skill_name, obs, info):
            self.current_step += 1
            if self.current_step < len(self.plan):
                narration = f"Completed {skill_name}. " + self.plan[self.current_step][2]

        done = self.current_step >= len(self.plan)
        return action, narration, done

    def get_trajectory(self):
        """
        Get full trajectory for training.

        Returns:
            trajectory: Dict with frames, actions, narrations, etc.
        """
        return {
            'instruction': self.instruction,
            'plan': self.plan,
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'narrations': self.narrations,
            'infos': self.infos,
            'frames': self.frames,  # if rendering
        }

    def _scripted_action(self, skill_name, obs, params):
        """Generate action for skill using scripted logic."""
        ee_pos = obs[0:3]
        vase_pos = obs[7:10]
        drawer_frac = obs[6]

        if skill_name == "LOCATE_DRAWER":
            # Look towards drawer
            target = np.array([0.0, -0.3, 0.7])
            return self._move_towards(ee_pos, target, 0.3)

        elif skill_name == "LOCATE_VASE":
            # Scan for vase
            return np.array([0.0, 0.0, 0.0])

        elif skill_name == "PLAN_SAFE_APPROACH":
            # Compute waypoint (no movement, just planning)
            return np.array([0.0, 0.0, 0.0])

        elif skill_name == "GRASP_HANDLE":
            handle_pos = np.array([0.0, -0.42, 0.65])
            return self._move_towards(ee_pos, handle_pos, 0.8)

        elif skill_name == "OPEN_WITH_CLEARANCE":
            # Pull drawer while maintaining clearance
            clearance = params.get('target_clearance', 0.15)
            pull_dir = np.array([0.0, -0.6, 0.0])

            # Avoid vase
            ee_to_vase = ee_pos - vase_pos
            dist_to_vase = np.linalg.norm(ee_to_vase)
            if dist_to_vase < clearance:
                repulsion = ee_to_vase / (dist_to_vase + 1e-6) * 0.3
                pull_dir = pull_dir + repulsion

            return np.clip(pull_dir, -1, 1)

        elif skill_name == "RETRACT_SAFE":
            safe_pos = np.array([-0.3, 0.0, 0.8])
            return self._move_towards(ee_pos, safe_pos, 0.5)

        return np.zeros(3)

    def _move_towards(self, current, target, speed):
        direction = target - current
        dist = np.linalg.norm(direction)
        if dist < 0.01:
            return np.zeros(3)
        return np.clip(direction / dist * speed, -1, 1)

    def _generate_narration(self, skill_name, obs, info):
        """Generate natural language narration for current step."""
        ee_pos = obs[0:3]
        drawer_frac = obs[6]
        min_clearance = obs[11]

        if skill_name == "LOCATE_DRAWER":
            return f"Scanning environment for drawer handle"

        elif skill_name == "LOCATE_VASE":
            return f"Identified vase at position, planning avoidance"

        elif skill_name == "PLAN_SAFE_APPROACH":
            return f"Computing safe trajectory with {min_clearance:.2f}m clearance"

        elif skill_name == "GRASP_HANDLE":
            handle_pos = np.array([0.0, -0.42, 0.65])
            dist = np.linalg.norm(ee_pos - handle_pos)
            return f"Approaching handle, distance: {dist:.3f}m"

        elif skill_name == "OPEN_WITH_CLEARANCE":
            return f"Opening drawer ({drawer_frac*100:.1f}%), clearance: {min_clearance:.3f}m"

        elif skill_name == "RETRACT_SAFE":
            return f"Retracting to safe position"

        return "Executing action"

    def _skill_done(self, skill_name, obs, info):
        """Check if current skill is complete."""
        if skill_name in ["LOCATE_DRAWER", "LOCATE_VASE", "PLAN_SAFE_APPROACH"]:
            return True  # Single-step skills

        elif skill_name == "GRASP_HANDLE":
            ee_pos = obs[0:3]
            handle_pos = np.array([0.0, -0.42, 0.65])
            return np.linalg.norm(ee_pos - handle_pos) < 0.05

        elif skill_name == "OPEN_WITH_CLEARANCE":
            drawer_frac = obs[6]
            return drawer_frac >= 0.9

        elif skill_name == "RETRACT_SAFE":
            ee_pos = obs[0:3]
            safe_pos = np.array([-0.3, 0.0, 0.8])
            return np.linalg.norm(ee_pos - safe_pos) < 0.1

        return False
```

---

## 6. Training Data Format

### HRL Training Data

```python
@dataclass
class SkillTrajectory:
    """Single skill execution trajectory."""
    skill_id: int
    skill_params: np.ndarray  # (5,)
    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray       # (T, action_dim)
    rewards: np.ndarray       # (T,)
    dones: np.ndarray         # (T,)
    success: bool
```

### VLA Training Data

```python
@dataclass
class VLATrajectory:
    """Full task trajectory with language annotations."""
    instruction: str
    skill_sequence: List[int]
    skill_params: List[np.ndarray]
    narrations: List[str]
    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray       # (T, action_dim)
    z_v_sequence: np.ndarray  # (T, z_v_dim) if vision
    success: bool
```

---

## 7. Evaluation Metrics

### Per-Skill Metrics

```python
@dataclass
class SkillMetrics:
    skill_id: int
    success_rate: float
    mean_completion_time: float
    mean_clearance: float
    collision_rate: float
    energy_per_skill: float
```

### Full Task Metrics

```python
@dataclass
class TaskMetrics:
    success_rate: float
    vase_collision_rate: float
    mean_clearance: float
    mean_steps: float
    mean_energy: float
    skill_sequence_accuracy: float  # VLA metric
    plan_efficiency: float  # Actual vs optimal steps
```

---

## 8. File Structure

```
src/
├── hrl/
│   ├── __init__.py
│   ├── skills.py           # SkillID, SkillParams
│   ├── low_level_policy.py # LowLevelSkillPolicy
│   ├── high_level_controller.py # HighLevelController
│   ├── skill_termination.py # SkillTerminationDetector
│   └── hrl_trainer.py       # Training loops
├── vision/
│   ├── __init__.py
│   ├── risk_map_head.py
│   ├── affordance_head.py
│   ├── no_go_head.py
│   └── encoder_with_heads.py
├── vla/
│   ├── __init__.py
│   ├── transformer_planner.py
│   ├── tokenizer.py
│   └── vla_trainer.py
├── sima/
│   ├── __init__.py
│   ├── co_agent.py
│   ├── narrator.py
│   └── trajectory_generator.py
└── evaluation/
    ├── skill_metrics.py
    ├── task_metrics.py
    └── benchmark_runner.py

scripts/
├── train_skill_policies.py
├── train_high_level_controller.py
├── train_vla_transformer.py
├── generate_sima_trajectories.py
└── eval_feifei_benchmark.py

checkpoints/
└── hrl/
    ├── skills/
    │   ├── locate_drawer.pt
    │   ├── locate_vase.pt
    │   ├── plan_safe_approach.pt
    │   ├── grasp_handle.pt
    │   ├── open_with_clearance.pt
    │   └── retract_safe.pt
    └── high_level_controller.pt
```

---

## 9. Interface Stability Guarantees

For Isaac Gym port compatibility:

1. **SkillID constants** - Fixed integers, never change
2. **SkillParams dataclass** - Extend only, never remove fields
3. **LowLevelSkillPolicy.forward()** signature - Stable
4. **HighLevelController.select_skill()** signature - Stable
5. **SIMACoAgent.get_trajectory()** format - Stable
6. **EpisodeInfoSummary** - From Phase B, stable
7. **DrawerVaseConfig** - From environment, stable

These interfaces can be used across PyBullet, Isaac Gym, and MuJoCo backends.

---

*Architecture frozen: 2025-11-16*
*Phase C HRL/VLA/SIMA*
