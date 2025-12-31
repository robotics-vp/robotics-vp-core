"""
CtRL-Sim-style behaviour model for dynamic agents.

Implements:
- K-disks action tokenization for discretizing continuous actions
- Autoregressive behaviour model over agent actions and returns
- Exponential tilting for return-conditional sampling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from src.scene.vector_scene.graph import ObjectClass, SceneGraph, SceneObject


class KDisksActionCoder:
    """
    K-Disks action tokenizer for discretizing continuous robot actions.

    Discretizes (Δx, Δy, Δθ) into a finite vocabulary of tokens.
    Uses a hierarchical scheme:
    - K concentric disks for (Δx, Δy)
    - N angular bins for Δθ
    """

    def __init__(
        self,
        num_disks: int = 5,
        num_angles: int = 8,
        max_step: float = 2.0,
        max_rotation: float = np.pi / 4,
    ):
        """
        Args:
            num_disks: Number of distance disks (including zero)
            num_angles: Number of angle bins for direction and rotation
            max_step: Maximum step size in any direction
            max_rotation: Maximum rotation per step
        """
        self.num_disks = num_disks
        self.num_angles = num_angles
        self.max_step = max_step
        self.max_rotation = max_rotation

        # Compute disk radii (0 = stay, 1..K-1 = increasing distances)
        self.disk_radii = np.linspace(0, max_step, num_disks)

        # Angle bins for direction (uniform around circle)
        self.direction_angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        # Rotation bins
        self.rotation_bins = np.linspace(-max_rotation, max_rotation, num_angles)

        # Vocabulary size: (num_disks * num_angles) * num_angles
        # First part: disk x direction = position
        # Second part: rotation
        self.num_position_tokens = num_disks * num_angles
        self.vocab_size = self.num_position_tokens * num_angles

    def encode(self, dx: float, dy: float, dtheta: float) -> int:
        """
        Encode continuous action to discrete token.

        Args:
            dx: X displacement
            dy: Y displacement
            dtheta: Rotation in radians

        Returns:
            Token index in [0, vocab_size)
        """
        # Compute distance and direction
        dist = np.sqrt(dx ** 2 + dy ** 2)
        direction = np.arctan2(dy, dx) % (2 * np.pi)

        # Find closest disk
        disk_idx = np.argmin(np.abs(self.disk_radii - dist))

        # Find closest direction angle
        angle_diffs = np.abs(self.direction_angles - direction)
        # Handle wraparound
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        dir_idx = np.argmin(angle_diffs)

        # Find closest rotation bin
        dtheta_clipped = np.clip(dtheta, -self.max_rotation, self.max_rotation)
        rot_idx = np.argmin(np.abs(self.rotation_bins - dtheta_clipped))

        # Combine into single token
        position_token = disk_idx * self.num_angles + dir_idx
        token = position_token * self.num_angles + rot_idx

        return int(token)

    def decode(self, token: int) -> Tuple[float, float, float]:
        """
        Decode token to continuous action.

        Args:
            token: Token index

        Returns:
            Tuple of (dx, dy, dtheta)
        """
        # Extract indices
        rot_idx = token % self.num_angles
        position_token = token // self.num_angles
        dir_idx = position_token % self.num_angles
        disk_idx = position_token // self.num_angles

        # Clamp indices
        disk_idx = min(disk_idx, self.num_disks - 1)
        dir_idx = min(dir_idx, self.num_angles - 1)
        rot_idx = min(rot_idx, self.num_angles - 1)

        # Decode
        dist = self.disk_radii[disk_idx]
        direction = self.direction_angles[dir_idx]
        dtheta = self.rotation_bins[rot_idx]

        dx = dist * np.cos(direction)
        dy = dist * np.sin(direction)

        return (float(dx), float(dy), float(dtheta))

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def get_null_token(self) -> int:
        """Return token for no movement (stay in place)."""
        # Disk 0 means no movement
        return 0


@dataclass
class AgentState:
    """
    State of a single agent at a point in time.

    Attributes:
        agent_id: Unique identifier
        x, y, z: Position
        heading: Orientation in radians
        speed: Current speed
        class_id: Agent class (human, robot, forklift, etc.)
        attributes: Additional properties
    """
    agent_id: int
    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    class_id: ObjectClass = ObjectClass.HUMAN
    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_scene_object(cls, obj: SceneObject) -> "AgentState":
        """Create from SceneObject."""
        return cls(
            agent_id=obj.id,
            x=obj.x,
            y=obj.y,
            z=obj.z,
            heading=obj.heading,
            speed=obj.speed,
            class_id=obj.class_id,
            attributes=obj.attributes.copy(),
        )

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for model input."""
        # Class one-hot
        num_classes = len(ObjectClass)
        class_onehot = np.zeros(num_classes, dtype=np.float32)
        class_onehot[self.class_id.value - 1] = 1.0

        # Position and heading
        pos = np.array([self.x, self.y, self.z], dtype=np.float32)
        heading_feat = np.array([np.sin(self.heading), np.cos(self.heading)], dtype=np.float32)
        speed_arr = np.array([self.speed], dtype=np.float32)

        return np.concatenate([class_onehot, pos, heading_feat, speed_arr])

    def apply_action(self, dx: float, dy: float, dtheta: float, dt: float = 1.0) -> "AgentState":
        """Apply action to get new state."""
        new_heading = self.heading + dtheta
        new_x = self.x + dx
        new_y = self.y + dy
        new_speed = np.sqrt(dx ** 2 + dy ** 2) / dt if dt > 0 else 0.0

        return AgentState(
            agent_id=self.agent_id,
            x=new_x,
            y=new_y,
            z=self.z,
            heading=new_heading,
            speed=new_speed,
            class_id=self.class_id,
            attributes=self.attributes,
        )


@dataclass
class AgentStateBatch:
    """
    Batch of agent states at a single timestep.

    Attributes:
        agents: List of AgentState
        timestamp: Time of this snapshot
    """
    agents: List[AgentState]
    timestamp: float = 0.0

    @classmethod
    def from_scene_graph(cls, graph: SceneGraph, timestamp: float = 0.0) -> "AgentStateBatch":
        """Create from SceneGraph's dynamic objects."""
        agents = [
            AgentState.from_scene_object(obj)
            for obj in graph.objects
            if obj.speed > 0 or obj.class_id in {ObjectClass.HUMAN, ObjectClass.ROBOT, ObjectClass.FORKLIFT}
        ]
        return cls(agents=agents, timestamp=timestamp)

    def to_tensor(self, device: Optional[str] = None) -> "torch.Tensor":
        """Convert to tensor (N_agents, D_agent)."""
        if torch is None:
            raise ImportError("PyTorch is required")

        if not self.agents:
            d_agent = len(ObjectClass) + 3 + 2 + 1
            return torch.zeros((0, d_agent), dtype=torch.float32)

        features = np.stack([a.to_feature_vector() for a in self.agents])
        tensor = torch.tensor(features, dtype=torch.float32)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None


@dataclass
class SceneObjectTrajectory:
    """
    Time-indexed trajectory for a single agent.

    Attributes:
        agent_id: Agent identifier
        timestamps: List of timestamps
        positions: List of (x, y, z) tuples
        headings: List of heading values
        speeds: List of speed values
        actions: List of action tokens (optional)
    """
    agent_id: int
    timestamps: List[float] = field(default_factory=list)
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)

    def append(
        self,
        timestamp: float,
        position: Tuple[float, float, float],
        heading: float,
        speed: float,
        action: Optional[int] = None,
    ) -> None:
        """Append a new timestep to the trajectory."""
        self.timestamps.append(timestamp)
        self.positions.append(position)
        self.headings.append(heading)
        self.speeds.append(speed)
        if action is not None:
            self.actions.append(action)

    def __len__(self) -> int:
        return len(self.timestamps)


class BehaviourModel(nn.Module):
    """
    CtRL-Sim-style autoregressive model over agent actions and returns.

    Predicts next actions for all agents conditioned on:
    - Scene latent (from SceneGraphEncoder)
    - Past agent states
    - Past actions
    - Optional return-to-go for goal-conditioning
    """

    def __init__(
        self,
        scene_latent_dim: int,
        agent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_actions: int = 200,
        max_agents: int = 32,
        max_timesteps: int = 100,
        dropout: float = 0.1,
    ):
        """
        Args:
            scene_latent_dim: Dimension of scene latent from encoder
            agent_dim: Dimension of agent feature vectors
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_actions: Size of action vocabulary
            max_agents: Maximum number of agents
            max_timesteps: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.scene_latent_dim = scene_latent_dim
        self.agent_dim = agent_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_agents = max_agents
        self.max_timesteps = max_timesteps

        # Input embeddings
        self.scene_proj = nn.Linear(scene_latent_dim, hidden_dim)
        self.agent_proj = nn.Linear(agent_dim, hidden_dim)
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.return_proj = nn.Linear(1, hidden_dim)

        # Positional embeddings
        self.time_embed = nn.Embedding(max_timesteps, hidden_dim)
        self.agent_pos_embed = nn.Embedding(max_agents, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(
        self,
        scene_latent: torch.Tensor,
        agent_states: torch.Tensor,
        past_actions: torch.Tensor,
        past_returns: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict action logits for next timestep.

        Args:
            scene_latent: (B, D_scene) scene context
            agent_states: (B, T, N_agents, D_agent) agent features over time
            past_actions: (B, T, N_agents) action tokens
            past_returns: (B, T, N_agents) return-to-go values (optional)
            attention_mask: Optional attention mask

        Returns:
            (B, T, N_agents, num_actions) action logits
        """
        batch_size = agent_states.size(0)
        seq_len = agent_states.size(1)
        n_agents = agent_states.size(2)

        # Project inputs
        scene_emb = self.scene_proj(scene_latent)  # (B, D)

        # Reshape agent states for processing
        agent_flat = agent_states.view(batch_size, seq_len * n_agents, -1)  # (B, T*N, D_agent)
        agent_emb = self.agent_proj(agent_flat)  # (B, T*N, D)

        # Action embeddings
        action_flat = past_actions.view(batch_size, seq_len * n_agents)  # (B, T*N)
        action_emb = self.action_embed(action_flat)  # (B, T*N, D)

        # Time and agent position embeddings
        time_idx = torch.arange(seq_len, device=agent_states.device)
        time_idx = time_idx.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, n_agents)
        time_idx = time_idx.reshape(batch_size, seq_len * n_agents)
        time_emb = self.time_embed(time_idx)  # (B, T*N, D)

        agent_idx = torch.arange(n_agents, device=agent_states.device)
        agent_idx = agent_idx.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        agent_idx = torch.clamp(agent_idx.reshape(batch_size, seq_len * n_agents), 0, self.max_agents - 1)
        agent_pos_emb = self.agent_pos_embed(agent_idx)  # (B, T*N, D)

        # Combine embeddings
        combined = agent_emb + action_emb + time_emb + agent_pos_emb

        # Add scene context
        scene_emb = scene_emb.unsqueeze(1)  # (B, 1, D)
        combined = torch.cat([scene_emb, combined], dim=1)  # (B, 1 + T*N, D)

        # Optional return conditioning
        if past_returns is not None:
            return_flat = past_returns.view(batch_size, seq_len * n_agents, 1)
            return_emb = self.return_proj(return_flat)
            # Add to combined (skip scene token)
            combined[:, 1:, :] = combined[:, 1:, :] + return_emb

        # Transformer
        output = self.transformer(combined, mask=attention_mask)

        # Remove scene token and reshape
        output = output[:, 1:, :]  # (B, T*N, D)
        output = output.view(batch_size, seq_len, n_agents, self.hidden_dim)

        # Action logits
        logits = self.action_head(output)  # (B, T, N, num_actions)

        return logits

    def get_action_probs(
        self,
        scene_latent: torch.Tensor,
        agent_states: torch.Tensor,
        past_actions: torch.Tensor,
        past_returns: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get action probabilities with temperature scaling."""
        logits = self.forward(scene_latent, agent_states, past_actions, past_returns)
        return F.softmax(logits / temperature, dim=-1)


def exponential_tilt(
    logits: "torch.Tensor",
    returns: "torch.Tensor",
    tilt: float,
) -> "torch.Tensor":
    """
    Apply exponential tilting to action logits based on returns.

    Positive tilt favors actions leading to higher returns (cooperative).
    Negative tilt favors actions leading to lower returns (adversarial).

    Args:
        logits: (B, N, num_actions) action logits
        returns: (B, N) return values per agent
        tilt: Tilting coefficient

    Returns:
        Tilted logits
    """
    if torch is None:
        raise ImportError("PyTorch is required")

    # Normalize returns to prevent numerical issues
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Apply exponential tilt
    tilt_factor = torch.exp(tilt * returns_norm.unsqueeze(-1))
    tilted_logits = logits * tilt_factor

    return tilted_logits


def rollout_behaviour(
    graph: SceneGraph,
    behaviour_model: BehaviourModel,
    initial_states: AgentStateBatch,
    num_steps: int,
    action_coder: KDisksActionCoder,
    scene_latent: Optional["torch.Tensor"] = None,
    tilt: float = 0.0,
    dt: float = 1.0,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> List[SceneObjectTrajectory]:
    """
    Roll out agent behaviour for multiple timesteps.

    Args:
        graph: Scene graph (for context)
        behaviour_model: Trained behaviour model
        initial_states: Initial agent states
        num_steps: Number of timesteps to simulate
        action_coder: Action tokenizer
        scene_latent: Optional pre-computed scene latent
        tilt: Exponential tilting coefficient (negative = adversarial)
        dt: Time delta between steps
        temperature: Sampling temperature
        device: Target device

    Returns:
        List of SceneObjectTrajectory for each agent
    """
    if torch is None:
        raise ImportError("PyTorch is required")

    # Initialize trajectories
    trajectories = []
    for agent in initial_states.agents:
        traj = SceneObjectTrajectory(agent_id=agent.agent_id)
        traj.append(
            timestamp=initial_states.timestamp,
            position=(agent.x, agent.y, agent.z),
            heading=agent.heading,
            speed=agent.speed,
        )
        trajectories.append(traj)

    if not trajectories:
        return trajectories

    # Get scene latent if not provided
    if scene_latent is None:
        # Create placeholder scene latent
        scene_latent = torch.zeros(1, behaviour_model.scene_latent_dim)
        if device:
            scene_latent = scene_latent.to(device)

    # Initialize state tracking
    current_states = list(initial_states.agents)

    for step in range(num_steps):
        # Build agent state tensor
        agent_features = np.stack([a.to_feature_vector() for a in current_states])
        agent_tensor = torch.tensor(agent_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if device:
            agent_tensor = agent_tensor.to(device)

        # Build past actions (just use null tokens for first step)
        n_agents = len(current_states)
        if step == 0:
            past_actions = torch.zeros(1, 1, n_agents, dtype=torch.long)
        else:
            # Use actions from previous step
            last_actions = [t.actions[-1] if t.actions else 0 for t in trajectories]
            past_actions = torch.tensor(last_actions, dtype=torch.long).unsqueeze(0).unsqueeze(0)

        if device:
            past_actions = past_actions.to(device)

        # Get action logits
        with torch.no_grad():
            logits = behaviour_model(
                scene_latent,
                agent_tensor,
                past_actions,
            )  # (1, 1, N, num_actions)

        logits = logits.squeeze(0).squeeze(0)  # (N, num_actions)

        # Apply exponential tilting if specified
        if abs(tilt) > 1e-6:
            # Use simple uniform returns for now (would come from value function)
            returns = torch.zeros(n_agents)
            if device:
                returns = returns.to(device)
            logits = exponential_tilt(logits, returns, tilt)

        # Sample actions
        probs = F.softmax(logits / temperature, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)  # (N,)

        # Update states and trajectories
        timestamp = initial_states.timestamp + (step + 1) * dt
        new_states = []

        for i, (agent, action_token) in enumerate(zip(current_states, actions.cpu().numpy())):
            dx, dy, dtheta = action_coder.decode(int(action_token))
            new_agent = agent.apply_action(dx, dy, dtheta, dt)
            new_states.append(new_agent)

            trajectories[i].append(
                timestamp=timestamp,
                position=(new_agent.x, new_agent.y, new_agent.z),
                heading=new_agent.heading,
                speed=new_agent.speed,
                action=int(action_token),
            )

        current_states = new_states

    return trajectories


def create_simple_behaviour_policy(
    action_coder: KDisksActionCoder,
    speed: float = 1.0,
) -> "Callable[[AgentState], int]":
    """
    Create a simple rule-based behaviour policy.

    Useful for testing without a trained model.

    Args:
        action_coder: Action tokenizer
        speed: Target movement speed

    Returns:
        Function mapping AgentState to action token
    """
    def policy(state: AgentState) -> int:
        # Simple random walk with bias toward heading direction
        dx = speed * np.cos(state.heading) * (0.5 + 0.5 * np.random.random())
        dy = speed * np.sin(state.heading) * (0.5 + 0.5 * np.random.random())
        dtheta = np.random.uniform(-0.2, 0.2)  # Small random rotation
        return action_coder.encode(dx, dy, dtheta)

    return policy
