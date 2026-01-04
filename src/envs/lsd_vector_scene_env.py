"""
LSD Vector Scene Environment.

Combines:
- Scenario Dreamer-style vectorized scene graphs + behaviour model
- LSD-3D-style geometry + 3D Gaussian scene generation
- Integration with existing MPL/economics logging

Pipeline:
    Vector scene graph & behavior (Scenario-style)
    -> 3D geometry + texture (LSD-style)
    -> RL control
    -> MPL / econ logging
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.behaviour.ctrl_sim_like import (
    AgentStateBatch,
    BehaviourModel,
    KDisksActionCoder,
    SceneObjectTrajectory,
    create_simple_behaviour_policy,
    rollout_behaviour,
)
from src.config.econ_params import EconParams, load_econ_params


def _default_econ_params() -> EconParams:
    """Create default EconParams for LSD Vector Scene Environment."""
    return load_econ_params({
        "price_per_unit": 0.30,
        "damage_cost": 1.0,
        "energy_Wh_per_attempt": 0.01,
        "time_step_s": 1.0,
        "base_rate": 2.0,
        "p_min": 0.02,
        "k_err": 0.15,
        "q_speed": 2.0,
        "q_care": 1.5,
        "care_cost": 0.3,
        "max_steps": 500,
        "max_catastrophic_errors": 3,
        "max_error_rate_sla": 0.15,
        "min_steps_for_sla": 10,
        "zero_throughput_patience": 50,
    }, preset="toy")
from src.envs.dishwashing_env import EpisodeInfoSummary, summarize_episode_info
from src.envs.lsd3d_env.gaussian_scene import GaussianScene, mesh_to_gaussians
from src.envs.lsd3d_env.ggds import CameraRig, GGDSConfig, GGDSOptimizer, create_default_optimizer
from src.envs.lsd3d_env.proxy_geometry import Mesh, VoxelGrid, scene_graph_to_voxels, voxels_to_mesh
from src.scene.vector_scene.encoding import SceneGraphEncoder, ordered_scene_tensors
from src.scene.vector_scene.graph import (
    NodeType,
    ObjectClass,
    SceneGraph,
    SceneNode,
    SceneObject,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class SceneGraphConfig:
    """Configuration for scene graph generation."""
    topology_type: str = "WAREHOUSE_AISLES"
    num_nodes: int = 30
    num_objects: int = 12
    density: float = 0.7
    route_length: float = 100.0
    seed: Optional[int] = None


@dataclass
class VisualStyleConfig:
    """Configuration for visual style/rendering."""
    lighting: str = "DIM_INDOOR"
    clutter_level: str = "HIGH"
    material_mix: List[str] = field(default_factory=lambda: ["metal", "plastic"])
    voxel_size: float = 0.1
    default_height: float = 3.0


@dataclass
class BehaviourConfig:
    """Configuration for dynamic agent behaviour."""
    num_humans: int = 4
    num_robots: int = 1
    num_forklifts: int = 1
    tilt: float = -1.0  # Negative = adversarial
    behaviour_model_path: Optional[str] = None
    use_simple_policy: bool = True


@dataclass
class LSDVectorSceneEnvConfig:
    """Complete configuration for LSDVectorSceneEnv."""
    scene_graph_config: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    visual_style_config: VisualStyleConfig = field(default_factory=VisualStyleConfig)
    behaviour_config: BehaviourConfig = field(default_factory=BehaviourConfig)
    econ_params: Optional[EconParams] = None
    max_steps: int = 500
    time_step_s: float = 1.0
    enable_ggds: bool = False
    ggds_iterations: int = 50
    action_dim: int = 2  # (speed, care) like dishwashing

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LSDVectorSceneEnvConfig":
        """Create from dictionary config."""
        scene_graph = SceneGraphConfig(**config.get("scene_graph_config", {}))
        visual_style = VisualStyleConfig(**config.get("visual_style_config", {}))
        behaviour = BehaviourConfig(**config.get("behaviour_config", {}))

        econ_dict = config.get("econ_params")
        econ = EconParams(**econ_dict) if econ_dict else None

        return cls(
            scene_graph_config=scene_graph,
            visual_style_config=visual_style,
            behaviour_config=behaviour,
            econ_params=econ,
            max_steps=config.get("max_steps", 500),
            time_step_s=config.get("time_step_s", 1.0),
            enable_ggds=config.get("enable_ggds", False),
            ggds_iterations=config.get("ggds_iterations", 50),
            action_dim=config.get("action_dim", 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_graph_config": {
                "topology_type": self.scene_graph_config.topology_type,
                "num_nodes": self.scene_graph_config.num_nodes,
                "num_objects": self.scene_graph_config.num_objects,
                "density": self.scene_graph_config.density,
                "route_length": self.scene_graph_config.route_length,
                "seed": self.scene_graph_config.seed,
            },
            "visual_style_config": {
                "lighting": self.visual_style_config.lighting,
                "clutter_level": self.visual_style_config.clutter_level,
                "material_mix": self.visual_style_config.material_mix,
                "voxel_size": self.visual_style_config.voxel_size,
                "default_height": self.visual_style_config.default_height,
            },
            "behaviour_config": {
                "num_humans": self.behaviour_config.num_humans,
                "num_robots": self.behaviour_config.num_robots,
                "num_forklifts": self.behaviour_config.num_forklifts,
                "tilt": self.behaviour_config.tilt,
                "behaviour_model_path": self.behaviour_config.behaviour_model_path,
                "use_simple_policy": self.behaviour_config.use_simple_policy,
            },
            "max_steps": self.max_steps,
            "time_step_s": self.time_step_s,
            "enable_ggds": self.enable_ggds,
            "ggds_iterations": self.ggds_iterations,
            "action_dim": self.action_dim,
        }


def _compute_scene_id(config: LSDVectorSceneEnvConfig, graph: SceneGraph) -> str:
    """Compute a unique hash for the scene configuration + graph."""
    data = json.dumps({
        "config": config.to_dict(),
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "num_objects": len(graph.objects),
        "bbox": graph.bounding_box(),
    }, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()[:16]


def _generate_scene_graph(config: SceneGraphConfig) -> SceneGraph:
    """
    Generate a scene graph based on configuration.

    Currently supports:
    - WAREHOUSE_AISLES: Simple parallel aisles warehouse
    - Additional topology types can be added here
    """
    rng = np.random.default_rng(config.seed)

    if config.topology_type == "WAREHOUSE_AISLES":
        num_aisles = max(2, int(config.num_nodes / 5))
        aisle_length = config.route_length / max(1, num_aisles / 2)
        return SceneGraph.create_simple_warehouse(
            num_aisles=num_aisles,
            aisle_length=aisle_length,
        )

    elif config.topology_type == "KITCHEN_LAYOUT":
        # Create simple kitchen layout
        nodes = []
        edges = []
        objects = []

        # Main kitchen area
        nodes.append(SceneNode(
            id=0,
            polyline=np.array([[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]], dtype=np.float32),
            node_type=NodeType.KITCHEN_ZONE,
            width=8.0,
        ))

        # Add counters, sinks
        obj_id = 0
        for x in [2.0, 5.0, 8.0]:
            objects.append(SceneObject(
                id=obj_id,
                class_id=ObjectClass.TABLE,
                x=x, y=1.0, z=0.0,
                length=1.5, width=0.6, height=0.9,
            ))
            obj_id += 1

        objects.append(SceneObject(
            id=obj_id,
            class_id=ObjectClass.SINK,
            x=5.0, y=7.0, z=0.0,
            length=1.0, width=0.5, height=0.4,
        ))

        return SceneGraph(nodes=nodes, edges=edges, objects=objects)

    else:
        # Default to simple warehouse
        return SceneGraph.create_simple_warehouse(
            num_aisles=5,
            aisle_length=20.0,
        )


def _add_dynamic_agents(
    graph: SceneGraph,
    config: BehaviourConfig,
    rng: np.random.Generator,
) -> SceneGraph:
    """Add dynamic agents (humans, robots, forklifts) to the scene."""
    bbox = graph.bounding_box()
    min_x, min_y, max_x, max_y = bbox

    new_objects = list(graph.objects)
    obj_id = max([o.id for o in graph.objects], default=-1) + 1

    # Add humans
    for i in range(config.num_humans):
        x = rng.uniform(min_x + 1, max_x - 1)
        y = rng.uniform(min_y + 1, max_y - 1)
        new_objects.append(SceneObject(
            id=obj_id,
            class_id=ObjectClass.HUMAN,
            x=x, y=y, z=0.0,
            heading=rng.uniform(0, 2 * np.pi),
            speed=1.2,  # Walking speed
            length=0.5, width=0.5, height=1.75,
            attributes={"role": "worker", "agent_index": i},
        ))
        obj_id += 1

    # Add robots (besides the ego robot)
    for i in range(config.num_robots):
        x = rng.uniform(min_x + 1, max_x - 1)
        y = rng.uniform(min_y + 1, max_y - 1)
        new_objects.append(SceneObject(
            id=obj_id,
            class_id=ObjectClass.ROBOT,
            x=x, y=y, z=0.0,
            heading=rng.uniform(0, 2 * np.pi),
            speed=0.8,
            length=0.6, width=0.6, height=1.5,
            attributes={"robot_type": "mobile_manipulator", "agent_index": i},
        ))
        obj_id += 1

    # Add forklifts
    for i in range(config.num_forklifts):
        x = rng.uniform(min_x + 2, max_x - 2)
        y = rng.uniform(min_y + 2, max_y - 2)
        new_objects.append(SceneObject(
            id=obj_id,
            class_id=ObjectClass.FORKLIFT,
            x=x, y=y, z=0.0,
            heading=rng.uniform(0, 2 * np.pi),
            speed=2.0,  # Faster than humans
            length=2.5, width=1.2, height=2.0,
            attributes={"agent_index": i},
        ))
        obj_id += 1

    return SceneGraph(
        nodes=graph.nodes,
        edges=graph.edges,
        objects=new_objects,
        metadata=graph.metadata,
    )


class LSDVectorSceneEnv:
    """
    Environment combining LSD-3D geometry with Scenario Dreamer-style scene graphs.

    Provides:
    - Scene graph generation and encoding
    - Voxel/mesh/Gaussian scene representation
    - Dynamic agent behaviour (humans, robots, forklifts)
    - MPL/economics logging integration
    """

    def __init__(self, config: LSDVectorSceneEnvConfig):
        """
        Initialize the environment.

        Args:
            config: Complete environment configuration
        """
        self.config = config
        self.econ_params = config.econ_params or _default_econ_params()

        # Scene components
        self.graph: Optional[SceneGraph] = None
        self.voxels: Optional[VoxelGrid] = None
        self.mesh: Optional[Mesh] = None
        self.gaussian_scene: Optional[GaussianScene] = None
        self.scene_latent: Optional[Any] = None
        self.scene_id: str = ""

        # Action tokenizer for behaviour
        self.action_coder = KDisksActionCoder(
            num_disks=5,
            num_angles=8,
            max_step=2.0,
        )

        # Behaviour model (or simple policy)
        self.behaviour_model: Optional[BehaviourModel] = None
        self.simple_policy: Optional[Callable] = None
        if config.behaviour_config.use_simple_policy:
            self.simple_policy = create_simple_behaviour_policy(self.action_coder)

        # Scene encoder
        self._encoder: Optional[SceneGraphEncoder] = None

        # GGDS optimizer
        self.ggds_optimizer = create_default_optimizer(GGDSConfig(
            num_iterations=config.ggds_iterations,
        ))

        # Episode state
        self.steps = 0
        self.t = 0.0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.energy_Wh = 0.0
        self.agent_states: Optional[AgentStateBatch] = None
        self.agent_trajectories: List[SceneObjectTrajectory] = []

        # RNG
        self._rng = np.random.default_rng(config.scene_graph_config.seed)

    def _get_encoder(self) -> SceneGraphEncoder:
        """Get or create scene encoder."""
        if self._encoder is None:
            if torch is None:
                raise ImportError("PyTorch is required for scene encoding")

            # Compute input dimensions from a sample graph
            sample_graph = SceneGraph.create_simple_warehouse(num_aisles=2, aisle_length=10.0)
            tensors = ordered_scene_tensors(sample_graph)
            node_dim = tensors["node_features"].shape[-1] + tensors["node_positions"].shape[-1]
            obj_dim = tensors["object_features"].shape[-1] + tensors["object_positions"].shape[-1]

            self._encoder = SceneGraphEncoder(
                node_input_dim=node_dim,
                obj_input_dim=obj_dim,
                hidden_dim=128,
                num_layers=3,
            )

        return self._encoder

    def _build_scene(self) -> None:
        """Build complete scene from config."""
        # 1. Generate scene graph
        self.graph = _generate_scene_graph(self.config.scene_graph_config)

        # 2. Add dynamic agents
        self.graph = _add_dynamic_agents(
            self.graph,
            self.config.behaviour_config,
            self._rng,
        )

        # 3. Compute scene ID
        self.scene_id = _compute_scene_id(self.config, self.graph)

        # 4. Encode scene graph (optional - skip if dimensions don't match)
        if torch is not None:
            try:
                encoder = self._get_encoder()
                tensors = ordered_scene_tensors(self.graph)
                with torch.no_grad():
                    encoded = encoder(tensors)
                    self.scene_latent = encoded["scene_latent"]
            except (RuntimeError, Exception):
                # Skip encoding if dimension mismatch (e.g., different scene sizes)
                self.scene_latent = None

        # 5. Convert to voxels and mesh
        self.voxels = scene_graph_to_voxels(
            self.graph,
            voxel_size=self.config.visual_style_config.voxel_size,
            default_height=self.config.visual_style_config.default_height,
        )
        self.mesh = voxels_to_mesh(self.voxels)

        # 6. Initialize Gaussians from mesh
        self.gaussian_scene = mesh_to_gaussians(self.mesh)

        # 7. Optional GGDS optimization
        if self.config.enable_ggds:
            bbox = self.graph.bounding_box()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1.5)
            camera_rig = CameraRig.create_orbit(center=center, radius=10.0)
            self.gaussian_scene = self.ggds_optimizer.optimize_scene(
                self.gaussian_scene,
                camera_rig,
                prompts=[f"A {self.config.scene_graph_config.topology_type.lower().replace('_', ' ')}"],
            )

    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        # Rebuild scene
        self._build_scene()

        # Reset episode state
        self.steps = 0
        self.t = 0.0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.energy_Wh = 0.0

        # Initialize agent states
        self.agent_states = AgentStateBatch.from_scene_graph(self.graph, timestamp=0.0)

        # Initialize trajectories
        self.agent_trajectories = []
        for agent in self.agent_states.agents:
            traj = SceneObjectTrajectory(agent_id=agent.agent_id)
            traj.append(
                timestamp=0.0,
                position=(agent.x, agent.y, agent.z),
                heading=agent.heading,
                speed=agent.speed,
            )
            self.agent_trajectories.append(traj)

        return self._obs()

    def _obs(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            "t": self.t,
            "completed": self.completed,
            "attempts": self.attempts,
            "errors": self.errors,
            "scene_id": self.scene_id,
            "num_agents": len(self.agent_states.agents) if self.agent_states else 0,
        }

    def _step_agents(self) -> None:
        """Advance dynamic agents by one step."""
        if self.agent_states is None or not self.agent_states.agents:
            return

        # Use simple policy or trained model
        new_agents = []
        for agent in self.agent_states.agents:
            if self.simple_policy is not None:
                action_token = self.simple_policy(agent)
            else:
                # Would use behaviour model here
                action_token = self.action_coder.get_null_token()

            dx, dy, dtheta = self.action_coder.decode(action_token)
            new_agent = agent.apply_action(dx, dy, dtheta, self.config.time_step_s)
            new_agents.append(new_agent)

            # Update trajectory
            for traj in self.agent_trajectories:
                if traj.agent_id == agent.agent_id:
                    traj.append(
                        timestamp=self.t + self.config.time_step_s,
                        position=(new_agent.x, new_agent.y, new_agent.z),
                        heading=new_agent.heading,
                        speed=new_agent.speed,
                        action=action_token,
                    )
                    break

        self.agent_states = AgentStateBatch(
            agents=new_agents,
            timestamp=self.t + self.config.time_step_s,
        )

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """
        Step the environment.

        Args:
            action: Array of [speed, care] in [0, 1]

        Returns:
            Tuple of (observation, info, done)
        """
        prev_completed = self.completed
        prev_errors = self.errors
        prev_energy = self.energy_Wh

        # Parse action
        if np.isscalar(action):
            speed, care = float(np.clip(action, 0.0, 1.0)), 0.5
        else:
            speed = float(np.clip(action[0], 0.0, 1.0))
            care = float(np.clip(action[1], 0.0, 1.0)) if len(action) > 1 else 0.5

        # Advance dynamic agents (may create obstacles/challenges)
        self._step_agents()

        # Simulate task progress (similar to dishwashing but with scene context)
        rate_per_min = max(0.1, self.econ_params.base_rate * (0.5 + 0.5 * speed))
        rate_per_min *= (1.0 - self.econ_params.care_cost * care)

        time_step_minutes = self.config.time_step_s / 60.0
        rate_per_step = rate_per_min * time_step_minutes

        attempts = max(1, int(np.random.poisson(rate_per_step)))
        self.attempts += attempts

        # Error probability (scene complexity can affect this)
        scene_complexity = len(self.agent_states.agents) / 10.0 if self.agent_states else 0.0
        p_err = self.econ_params.p_min + self.econ_params.k_err * (
            speed ** self.econ_params.q_speed
        ) * ((1.0 - care) ** self.econ_params.q_care)
        p_err += 0.02 * scene_complexity  # More agents = harder
        p_err = float(np.clip(p_err, 0.0, 0.5))

        errs = np.random.binomial(attempts, p_err)
        self.errors += errs

        succ = max(attempts - errs, 0)
        self.completed += succ

        # Energy accounting
        delta_energy_Wh = attempts * self.econ_params.energy_Wh_per_attempt
        self.energy_Wh += delta_energy_Wh

        # Time and counters
        self.t += self.config.time_step_s
        self.steps += 1

        # Compute metrics
        delta_units = self.completed - prev_completed
        delta_errors = self.errors - prev_errors
        dt_hours = self.config.time_step_s / 3600.0
        mpl_t = (delta_units / dt_hours) if dt_hours > 0 else 0.0
        ep_t = delta_units / delta_energy_Wh if delta_energy_Wh > 0 else 0.0

        # Termination logic
        done = False
        terminated_reason = None

        if self.steps >= self.config.max_steps:
            done = True
            terminated_reason = "max_steps"

        if self.completed > 0:
            current_error_rate = self.errors / max(1, self.completed)
            if current_error_rate > self.econ_params.max_error_rate_sla:
                done = True
                terminated_reason = "sla_violation"

        obs = self._obs()

        info = {
            "succ": succ,
            "errs": errs,
            "p_err": p_err,
            "speed": speed,
            "care": care,
            "rate_per_min": rate_per_min,
            "t": self.t,
            "delta_units": delta_units,
            "delta_energy_Wh": delta_energy_Wh,
            "delta_errors": delta_errors,
            "mpl_t": mpl_t,
            "ep_t": ep_t,
            "error_rate_t": self.errors / max(1, self.completed) if self.completed > 0 else 0.0,
            "units_done": self.completed,
            "errors": self.errors,
            "energy_Wh": self.energy_Wh,
            "scene_id": self.scene_id,
            "num_agents": len(self.agent_states.agents) if self.agent_states else 0,
            "terminated_reason": terminated_reason,
            # Difficulty features for logging
            "difficulty_features": self._get_difficulty_features(),
        }

        return obs, info, done

    def _get_difficulty_features(self) -> Dict[str, float]:
        """Extract difficulty features for logging."""
        n_dynamic = len([a for a in (self.agent_states.agents if self.agent_states else [])
                        if a.class_id in {ObjectClass.HUMAN, ObjectClass.FORKLIFT}])

        return {
            "graph_density": self.config.scene_graph_config.density,
            "route_length": self.config.scene_graph_config.route_length,
            "num_dynamic_agents": float(n_dynamic),
            "tilt": self.config.behaviour_config.tilt,
            "occlusion_level": float(self.voxels.get_occupied_count()) / 1000.0 if self.voxels else 0.0,
        }

    def get_episode_log(self, info_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate episode log entry for MPL/econ reporting.

        Args:
            info_history: List of info dicts from step() calls

        Returns:
            Complete episode log entry
        """
        summary = summarize_episode_info(info_history)

        return {
            "scene_id": self.scene_id,
            "scene_graph_config": self.config.scene_graph_config.__dict__,
            "visual_style_config": {
                "lighting": self.config.visual_style_config.lighting,
                "clutter_level": self.config.visual_style_config.clutter_level,
                "material_mix": self.config.visual_style_config.material_mix,
            },
            "behaviour_config": {
                "num_humans": self.config.behaviour_config.num_humans,
                "num_robots": self.config.behaviour_config.num_robots,
                "num_forklifts": self.config.behaviour_config.num_forklifts,
                "tilt": self.config.behaviour_config.tilt,
            },
            "difficulty_features": self._get_difficulty_features(),
            "mpl_metrics": {
                "success": summary.termination_reason in {"max_steps", "success"},
                "time_to_complete": self.t,
                "energy_wh": summary.energy_Wh,
                "collisions": 0,  # Would track in more detailed sim
                "interventions": 0,
                "mpl_units_per_hour": summary.mpl_episode,
                "error_rate": summary.error_rate_episode,
                "wage_parity": summary.wage_parity,
            },
            "episode_summary": {
                "termination_reason": summary.termination_reason,
                "mpl_episode": summary.mpl_episode,
                "ep_episode": summary.ep_episode,
                "error_rate_episode": summary.error_rate_episode,
                "energy_Wh": summary.energy_Wh,
                "profit": summary.profit,
            },
        }

    def get_scene_graph(self) -> Optional[SceneGraph]:
        """Get current scene graph."""
        return self.graph

    def get_gaussian_scene(self) -> Optional[GaussianScene]:
        """Get current Gaussian scene."""
        return self.gaussian_scene

    def get_agent_trajectories(self) -> List[SceneObjectTrajectory]:
        """Get trajectories of all dynamic agents."""
        return self.agent_trajectories


def create_env_from_config_path(config_path: str) -> LSDVectorSceneEnv:
    """
    Create environment from a YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured LSDVectorSceneEnv
    """
    import yaml

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = LSDVectorSceneEnvConfig.from_dict(config_dict)
    return LSDVectorSceneEnv(config)
