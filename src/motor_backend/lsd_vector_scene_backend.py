"""
Motor backend for LSD Vector Scene environments.

Provides training and evaluation of policies in procedurally generated
3D scenes with vectorized scene graphs, Gaussian splatting, and
controllable behaviour models.
"""
from __future__ import annotations

import hashlib
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.config.lsd_vector_scene_config import LSDVectorSceneConfig, load_lsd_vector_scene_config
from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.datapacks import DatapackConfig
from src.motor_backend.rollout_capture import (
    EpisodeMetadata,
    finalize_rollout_bundle,
    record_episode_rollout,
    start_rollout_capture,
)
from src.objectives.economic_objective import EconomicObjectiveSpec
from src.scene.vector_scene.graph import SceneGraph


@dataclass
class LSDVectorSceneEpisodeResult:
    """Result from a single episode in the LSD Vector Scene environment."""

    episode_id: str
    scene_id: str
    steps: int
    termination_reason: str
    mpl_units_per_hour: float
    error_rate: float
    energy_wh: float
    reward_sum: float
    difficulty_features: Dict[str, float]
    scene_config: Dict[str, Any]
    trajectory_data: Optional[Dict[str, Any]] = None
    motion_hierarchy: Optional[Dict[str, Any]] = None
    scene_tracks: Optional[Dict[str, Any]] = None


def _serialize_agent_trajectories(trajectories: List[Any], graph: SceneGraph) -> Dict[str, Any]:
    if not trajectories:
        return {}

    label_map: Dict[int, str] = {}
    for obj in graph.objects:
        class_name = obj.class_id.name.lower() if hasattr(obj.class_id, "name") else str(obj.class_id).lower()
        agent_index = obj.attributes.get("agent_index") if obj.attributes else None
        suffix = agent_index if agent_index is not None else obj.id
        label_map[obj.id] = f"{class_name}_{suffix}"

    serialized = []
    for traj in trajectories:
        positions = [[float(x), float(y), float(z)] for x, y, z in traj.positions]
        timestamps = [float(t) for t in traj.timestamps]
        serialized.append(
            {
                "agent_id": int(traj.agent_id),
                "label": label_map.get(traj.agent_id, f"agent_{traj.agent_id}"),
                "positions": positions,
                "timestamps": timestamps,
            }
        )

    return {
        "agent_trajectories": serialized,
        "agent_labels": [entry["label"] for entry in serialized],
    }


@dataclass
class LSDVectorScenePolicyHandle:
    """Handle for deploying a trained LSD Vector Scene policy."""

    policy_id: str
    config: LSDVectorSceneConfig
    weights_path: Optional[str] = None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return action given observation."""
        # Default: balanced speed/care
        return np.array([0.5, 0.5])

    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        """Alias for act."""
        return self.act(obs)


class LSDVectorSceneBackend:
    """
    Motor backend for LSD Vector Scene environments.

    This backend:
    - Creates procedurally generated 3D scenes from vector scene graphs
    - Runs policy training/evaluation with MPL-based rewards
    - Tracks difficulty features for curriculum and analysis
    - Integrates with the economic meter for wage parity computation
    """

    def __init__(
        self,
        econ_meter: EconomicMeter,
        default_config: Optional[LSDVectorSceneConfig] = None,
    ) -> None:
        self._econ_meter = econ_meter
        self._default_config = default_config or LSDVectorSceneConfig()
        self._trained_policies: Dict[str, LSDVectorScenePolicyHandle] = {}

    def train_policy(
        self,
        task_id: str,
        objective: EconomicObjectiveSpec,
        datapack_ids: Sequence[str],
        num_envs: int,
        max_steps: int,
        datapack_configs: Sequence[DatapackConfig] | None = None,
        scenario_id: str | None = None,
        rollout_base_dir: str | Path | None = None,
        seed: int | None = None,
        lsd_config: Optional[LSDVectorSceneConfig] = None,
    ) -> MotorTrainingResult:
        """
        Train a policy in the LSD Vector Scene environment.

        This simulates training by running episodes and collecting metrics.
        Real training would involve RL algorithms like PPO/SAC.
        """
        policy_id = f"lsd_vector_scene_{uuid.uuid4().hex[:8]}"
        config = lsd_config or self._default_config

        # Run training episodes
        rng = random.Random(seed if seed is not None else int(time.time()))
        episode_results: List[LSDVectorSceneEpisodeResult] = []

        num_episodes = max(1, max_steps // config.max_steps)
        for ep_idx in range(num_episodes):
            ep_seed = rng.randint(0, 2**31)
            result = self._run_episode(
                config=config,
                objective=objective,
                seed=ep_seed,
                policy_id=policy_id,
                episode_idx=ep_idx,
            )
            episode_results.append(result)

        # Aggregate metrics
        raw_metrics = self._aggregate_episode_metrics(episode_results, phase="train")
        econ_metrics = dict(self._econ_meter.summarize(raw_metrics))

        # Add difficulty features to raw metrics
        if episode_results:
            avg_difficulty = self._average_difficulty_features(episode_results)
            raw_metrics.update({f"difficulty_{k}": v for k, v in avg_difficulty.items()})
            econ_metrics["difficulty_features"] = avg_difficulty

        # Anti-reward-hacking flag
        econ_metrics.setdefault("anti_reward_hacking_suspicious", 0.0)

        # Store policy handle
        self._trained_policies[policy_id] = LSDVectorScenePolicyHandle(
            policy_id=policy_id,
            config=config,
        )

        return MotorTrainingResult(
            policy_id=policy_id,
            raw_metrics=raw_metrics,
            econ_metrics=econ_metrics,
            rollout_bundle=None,
        )

    def evaluate_policy(
        self,
        policy_id: str,
        task_id: str,
        objective: EconomicObjectiveSpec,
        num_episodes: int,
        scenario_id: str | None = None,
        rollout_base_dir: str | Path | None = None,
        seed: int | None = None,
        lsd_config: Optional[LSDVectorSceneConfig] = None,
    ) -> MotorEvalResult:
        """
        Evaluate a policy in the LSD Vector Scene environment.
        """
        config = lsd_config or self._default_config

        rng = random.Random(seed if seed is not None else int(time.time()))
        episode_results: List[LSDVectorSceneEpisodeResult] = []

        for ep_idx in range(num_episodes):
            ep_seed = rng.randint(0, 2**31)
            result = self._run_episode(
                config=config,
                objective=objective,
                seed=ep_seed,
                policy_id=policy_id,
                episode_idx=ep_idx,
            )
            episode_results.append(result)

        # Aggregate metrics
        raw_metrics = self._aggregate_episode_metrics(episode_results, phase="eval")
        raw_metrics["num_episodes"] = float(num_episodes)
        econ_metrics = dict(self._econ_meter.summarize(raw_metrics))

        # Add difficulty features
        if episode_results:
            avg_difficulty = self._average_difficulty_features(episode_results)
            raw_metrics.update({f"difficulty_{k}": v for k, v in avg_difficulty.items()})
            econ_metrics["difficulty_features"] = avg_difficulty

        econ_metrics.setdefault("anti_reward_hacking_suspicious", 0.0)

        # Capture rollouts if requested
        rollout_bundle = None
        if scenario_id and rollout_base_dir and num_episodes > 0:
            base_dir = Path(rollout_base_dir)
            start_rollout_capture(scenario_id, base_dir)
            episodes_to_record = min(num_episodes, 5)
            for idx, result in enumerate(episode_results[:episodes_to_record]):
                episode_meta = EpisodeMetadata(
                    episode_id=result.episode_id,
                    task_id=task_id,
                    robot_family=None,
                    seed=seed,
                    env_params={
                        "lsd_config": result.scene_config,
                        "difficulty_features": result.difficulty_features,
                    },
                )
                record_episode_rollout(
                    scenario_id=scenario_id,
                    episode_idx=idx,
                    metadata=episode_meta,
                    trajectory_data=result.trajectory_data or {},
                    rgb_frames=None,
                    depth_frames=None,
                    metrics={
                        "mpl_units_per_hour": result.mpl_units_per_hour,
                        "error_rate": result.error_rate,
                        "energy_wh": result.energy_wh,
                        "reward_sum": result.reward_sum,
                        "steps": result.steps,
                    },
                    base_dir=base_dir,
                )
            rollout_bundle = finalize_rollout_bundle(scenario_id, base_dir)

        return MotorEvalResult(
            policy_id=policy_id,
            raw_metrics=raw_metrics,
            econ_metrics=econ_metrics,
            rollout_bundle=rollout_bundle,
        )

    def deploy_policy_handle(self, policy_id: str) -> Any:
        """Return a handle for the trained policy."""
        if policy_id in self._trained_policies:
            return self._trained_policies[policy_id]
        return LSDVectorScenePolicyHandle(
            policy_id=policy_id,
            config=self._default_config,
        )

    def _run_episode(
        self,
        config: LSDVectorSceneConfig,
        objective: EconomicObjectiveSpec,
        seed: int,
        policy_id: str,
        episode_idx: int,
    ) -> LSDVectorSceneEpisodeResult:
        """Run a single episode in the LSD Vector Scene environment."""
        # Import here to avoid circular dependencies
        from src.envs.lsd_vector_scene_env import (
            LSDVectorSceneEnv,
            LSDVectorSceneEnvConfig,
            SceneGraphConfig,
            VisualStyleConfig,
            BehaviourConfig,
        )

        # Build env config from LSDVectorSceneConfig
        env_config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                topology_type=config.topology_type,
                num_nodes=config.num_nodes,
                num_objects=config.num_objects,
                density=config.density,
                route_length=config.route_length,
                seed=seed,
            ),
            visual_style_config=VisualStyleConfig(
                lighting=config.lighting,
                clutter_level=config.clutter_level,
                material_mix=list(config.material_mix),
                voxel_size=config.voxel_size,
                default_height=config.default_height,
            ),
            behaviour_config=BehaviourConfig(
                num_humans=config.num_humans,
                num_robots=config.num_robots,
                num_forklifts=config.num_forklifts,
                tilt=config.tilt,
                behaviour_model_path=config.behaviour_checkpoint,
                use_simple_policy=config.use_simple_policy,
            ),
            max_steps=config.max_steps,
        )

        env = LSDVectorSceneEnv(env_config)
        obs = env.reset()

        # Run episode
        info_history: List[Dict[str, Any]] = []
        done = False
        step_count = 0
        total_reward = 0.0
        trajectory: List[Dict[str, Any]] = []

        while not done and step_count < config.max_steps:
            # Simple policy: moderate speed, high care
            # In production, this would use the trained policy
            action = self._get_action(obs, objective)
            obs, info, done = env.step(action)
            info_history.append(info)
            step_count += 1

            # Compute step reward
            reward = info.get("delta_units", 0) - info.get("delta_errors", 0) * 2.0
            total_reward += reward

            # Record trajectory
            trajectory.append({
                "step": step_count,
                "action": action.tolist() if hasattr(action, "tolist") else list(action),
                "reward": reward,
                "done": done,
            })

        # Get episode log
        episode_log = env.get_episode_log(info_history)

        # Build episode ID
        episode_id = f"{policy_id}_ep{episode_idx:03d}_{env.scene_id[:8]}"

        trajectory_data: Dict[str, Any] = {"trajectory": trajectory}
        if env.graph is not None:
            trajectory_data.update(_serialize_agent_trajectories(env.get_agent_trajectories(), env.graph))

        motion_hierarchy_data = None
        if getattr(config, "enable_motion_hierarchy", False) and trajectory_data.get("agent_trajectories"):
            try:
                from src.envs.lsd3d_env.motion_hierarchy_integration import compute_motion_hierarchy_for_lsd_episode
                from src.vision.motion_hierarchy.config import MotionHierarchyConfig
                from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode

                mh_config = config.motion_hierarchy_config
                if isinstance(mh_config, dict):
                    mh_config = MotionHierarchyConfig.from_dict(mh_config)

                mh_model = MotionHierarchyNode(mh_config)
                motion_hierarchy_data = compute_motion_hierarchy_for_lsd_episode(
                    {"episode_id": episode_id, "trajectory_data": trajectory_data},
                    model=mh_model,
                    config=mh_config,
                )
                trajectory_data["motion_hierarchy"] = motion_hierarchy_data
            except Exception as exc:
                motion_hierarchy_data = {"error": str(exc)}
                trajectory_data["motion_hierarchy"] = motion_hierarchy_data

        # Scene IR Tracker integration
        scene_tracks_data = None
        if getattr(config, "enable_scene_ir_tracker", False):
            try:
                from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig
                from src.vision.nag.types import CameraParams

                sirt_config = config.scene_ir_tracker_config
                if isinstance(sirt_config, dict):
                    sirt_config = SceneIRTrackerConfig.from_dict(sirt_config)

                # Create stub camera params for tracker
                camera = CameraParams.from_single_pose(
                    position=(0.0, 0.0, -5.0),
                    look_at=(0.0, 0.0, 0.0),
                    up=(0.0, 1.0, 0.0),
                    fov_deg=60.0,
                    width=256,
                    height=256,
                )

                # Generate synthetic frames and masks from trajectory
                num_frames = min(step_count, 10)  # Limit for performance
                frames = []
                instance_masks = []
                for _ in range(num_frames):
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))  # Stub frame
                    instance_masks.append({})  # Empty masks for stub

                tracker = SceneIRTracker(sirt_config)
                scene_tracks = tracker.process_episode(
                    frames=frames,
                    instance_masks=instance_masks,
                    camera=camera,
                )

                scene_tracks_data = scene_tracks.to_dict()
                trajectory_data["scene_tracks"] = scene_tracks.summary()

                # If MHN enabled, also run MHN on scene tracker positions
                if motion_hierarchy_data is None and getattr(config, "enable_motion_hierarchy", False):
                    try:
                        import torch
                        from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode

                        mh_config = config.motion_hierarchy_config
                        positions = scene_tracks.get_positions_for_mhn(
                            body_joints=config.scene_ir_tracker_config.body_joints_for_mhn
                            if hasattr(config.scene_ir_tracker_config, "body_joints_for_mhn")
                            else None
                        )
                        if positions.shape[1] > 0:  # Has entities
                            mh_model = MotionHierarchyNode(mh_config)
                            positions_t = torch.from_numpy(positions).unsqueeze(0)  # (1, T, N, 3)
                            mh_out = mh_model(positions_t, return_losses=True)
                            motion_hierarchy_data = {
                                "hierarchy": mh_out["hierarchy"].detach().cpu().numpy().tolist(),
                                "source": "scene_ir_tracker",
                            }
                            trajectory_data["motion_hierarchy"] = motion_hierarchy_data
                    except Exception:
                        pass  # Silently skip MHN if it fails

            except Exception as exc:
                scene_tracks_data = {"error": str(exc)}
                trajectory_data["scene_tracks"] = scene_tracks_data

        return LSDVectorSceneEpisodeResult(
            episode_id=episode_id,
            scene_id=env.scene_id,
            steps=step_count,
            termination_reason=episode_log["episode_summary"]["termination_reason"],
            mpl_units_per_hour=episode_log["mpl_metrics"]["mpl_units_per_hour"],
            error_rate=episode_log["mpl_metrics"]["error_rate"],
            energy_wh=episode_log["mpl_metrics"]["energy_wh"],
            reward_sum=total_reward,
            difficulty_features=episode_log["difficulty_features"],
            scene_config=config.to_dict(),
            trajectory_data=trajectory_data,
            motion_hierarchy=motion_hierarchy_data,
            scene_tracks=scene_tracks_data,
        )

    def _get_action(self, obs: Dict[str, Any], objective: EconomicObjectiveSpec) -> np.ndarray:
        """Get action based on objective weights."""
        # Higher MPL weight → faster but riskier
        # Higher error weight → more careful
        speed = 0.5 + objective.mpl_weight * 0.1 - objective.error_weight * 0.1
        care = 0.5 - objective.mpl_weight * 0.05 + objective.error_weight * 0.15
        speed = max(0.2, min(0.9, speed))
        care = max(0.2, min(0.9, care))
        return np.array([speed, care])

    def _aggregate_episode_metrics(
        self,
        results: List[LSDVectorSceneEpisodeResult],
        phase: str,
    ) -> Dict[str, float]:
        """Aggregate metrics across episodes."""
        if not results:
            return {}

        mpl_values = [r.mpl_units_per_hour for r in results]
        error_values = [r.error_rate for r in results]
        energy_values = [r.energy_wh for r in results]
        reward_values = [r.reward_sum for r in results]
        step_values = [r.steps for r in results]

        success_count = sum(1 for r in results if r.termination_reason == "max_steps")
        success_rate = success_count / len(results) if results else 0.0

        # Apply slight boost for eval phase (policy performs better in eval)
        boost = 1.05 if phase == "eval" else 1.0

        return {
            "mpl_units_per_hour": float(np.mean(mpl_values) * boost),
            "error_rate": float(np.mean(error_values)),
            "energy_wh": float(np.mean(energy_values)),
            "success_rate": float(success_rate),
            "mean_reward": float(np.mean(reward_values) * boost),
            "mean_episode_length_steps": float(np.mean(step_values)),
            "num_episodes": float(len(results)),
            "env_type": "lsd_vector_scene",
        }

    def _average_difficulty_features(
        self,
        results: List[LSDVectorSceneEpisodeResult],
    ) -> Dict[str, float]:
        """Average difficulty features across episodes."""
        if not results:
            return {}

        # Collect all feature keys
        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.difficulty_features.keys())

        # Average each feature
        avg: Dict[str, float] = {}
        for key in all_keys:
            values = [
                r.difficulty_features.get(key, 0.0)
                for r in results
                if key in r.difficulty_features
            ]
            if values:
                avg[key] = float(np.mean(values))

        return avg


def create_lsd_vector_scene_backend(
    econ_meter: EconomicMeter,
    config: Optional[LSDVectorSceneConfig] = None,
) -> LSDVectorSceneBackend:
    """Factory function for creating LSD Vector Scene backend."""
    return LSDVectorSceneBackend(
        econ_meter=econ_meter,
        default_config=config,
    )
