from __future__ import annotations

import random
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.economics.econ_meter import EconomicMeter
from src.envs.workcell_env import WorkcellEnv, WorkcellEnvConfig
from src.envs.workcell_env.difficulty.difficulty_features import compute_difficulty_features
from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.datapacks import DatapackBundle, DatapackConfig, DatapackProvider
from src.motor_backend.rollout_capture import (
    EpisodeMetadata,
    RolloutBundle,
    finalize_rollout_bundle,
    record_episode_rollout,
    start_rollout_capture,
)
from src.motor_backend.workcell_reward_terms import (
    WorkcellRewardTerms,
    analyze_anti_reward_hacking,
    compute_workcell_reward,
)
from src.objectives.economic_objective import EconomicObjectiveSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkcellEpisodeResult:
    """Outcome from a single workcell episode."""

    episode_id: str
    steps: int
    duration_s: float
    success: bool
    termination_reason: str
    reward_sum: float
    mean_reward: float
    error_count: int
    error_rate: float
    energy_wh: float
    mpl_units_per_hour: float
    task_progress: float
    scene_spec: Mapping[str, Any]
    trajectory_data: Mapping[str, Any]
    metrics: Mapping[str, float]
    rgb_frames: list[Any] | None = None
    depth_frames: list[Any] | None = None


@dataclass(frozen=True)
class WorkcellEnvPolicyHandle:
    """Handle for deploying a trained workcell policy."""

    policy_id: str
    config: WorkcellEnvConfig
    action_scale: float = 0.05

    def act(self, obs: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "object_id": "end_effector",
            "delta_position": (0.0, 0.0, 0.0),
        }

    def step(self, obs: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.act(obs)


class WorkcellEnvBackend:
    """Motor backend for workcell environment rollouts."""

    def __init__(
        self,
        econ_meter: EconomicMeter,
        datapack_provider: DatapackProvider,
        default_config: WorkcellEnvConfig | None = None,
    ) -> None:
        self._econ_meter = econ_meter
        self._datapack_provider = datapack_provider
        self._default_config = default_config or WorkcellEnvConfig()
        self._trained_policies: dict[str, WorkcellEnvPolicyHandle] = {}

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
    ) -> MotorTrainingResult:
        policy_id = f"workcell_env_{uuid.uuid4().hex[:8]}"
        datapack_bundle = self._datapack_provider.resolve(task_id, datapack_ids, datapack_configs)
        config = self._apply_datapack_overrides(self._default_config, datapack_bundle)
        reward_terms = self._build_reward_terms(objective)

        rng = random.Random(seed if seed is not None else int(time.time()))
        episode_results: list[WorkcellEpisodeResult] = []
        num_episodes = max(1, max_steps // max(config.max_steps, 1))
        for ep_idx in range(num_episodes):
            ep_seed = rng.randint(0, 2**31 - 1)
            episode_results.append(
                self._run_episode(
                    config=config,
                    objective=objective,
                    reward_terms=reward_terms,
                    seed=ep_seed,
                    policy_id=policy_id,
                    episode_idx=ep_idx,
                    task_id=task_id,
                )
            )

        raw_metrics = self._aggregate_episode_metrics(episode_results, phase="train")
        raw_metrics["train_steps"] = float(max_steps)
        raw_metrics["num_envs"] = float(num_envs)
        if seed is not None:
            raw_metrics["seed"] = float(seed)

        econ_metrics = dict(self._econ_meter.summarize(raw_metrics))
        econ_metrics = _apply_anti_reward_hacking_flags(econ_metrics, raw_metrics)

        rollout_bundle = None
        if scenario_id and rollout_base_dir and episode_results:
            rollout_bundle = self._capture_rollouts(
                scenario_id=scenario_id,
                base_dir=Path(rollout_base_dir),
                episode_results=episode_results,
                task_id=task_id,
                seed=seed,
                config=config,
                datapack_bundle=datapack_bundle,
                max_episodes=3,
            )

        self._trained_policies[policy_id] = WorkcellEnvPolicyHandle(policy_id=policy_id, config=config)

        return MotorTrainingResult(
            policy_id=policy_id,
            raw_metrics=raw_metrics,
            econ_metrics=econ_metrics,
            rollout_bundle=rollout_bundle,
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
    ) -> MotorEvalResult:
        config = self._default_config
        reward_terms = self._build_reward_terms(objective)

        rng = random.Random(seed if seed is not None else int(time.time()))
        episode_results: list[WorkcellEpisodeResult] = []
        for ep_idx in range(num_episodes):
            ep_seed = rng.randint(0, 2**31 - 1)
            episode_results.append(
                self._run_episode(
                    config=config,
                    objective=objective,
                    reward_terms=reward_terms,
                    seed=ep_seed,
                    policy_id=policy_id,
                    episode_idx=ep_idx,
                    task_id=task_id,
                )
            )

        raw_metrics = self._aggregate_episode_metrics(episode_results, phase="eval")
        raw_metrics["num_episodes"] = float(num_episodes)
        if seed is not None:
            raw_metrics["seed"] = float(seed)

        econ_metrics = dict(self._econ_meter.summarize(raw_metrics))
        econ_metrics = _apply_anti_reward_hacking_flags(econ_metrics, raw_metrics)

        rollout_bundle = None
        if scenario_id and rollout_base_dir and episode_results:
            rollout_bundle = self._capture_rollouts(
                scenario_id=scenario_id,
                base_dir=Path(rollout_base_dir),
                episode_results=episode_results,
                task_id=task_id,
                seed=seed,
                config=config,
                datapack_bundle=None,
                max_episodes=5,
            )

        return MotorEvalResult(
            policy_id=policy_id,
            raw_metrics=raw_metrics,
            econ_metrics=econ_metrics,
            rollout_bundle=rollout_bundle,
        )

    def deploy_policy_handle(self, policy_id: str) -> Any:
        if policy_id in self._trained_policies:
            return self._trained_policies[policy_id]
        return WorkcellEnvPolicyHandle(policy_id=policy_id, config=self._default_config)

    def _run_episode(
        self,
        *,
        config: WorkcellEnvConfig,
        objective: EconomicObjectiveSpec,
        reward_terms: WorkcellRewardTerms,
        seed: int,
        policy_id: str,
        episode_idx: int,
        task_id: str,
    ) -> WorkcellEpisodeResult:
        env = WorkcellEnv(
            config=config,
            task_id=task_id,
            robot_family="workcell",
            seed=seed,
        )
        obs = env.reset(seed=seed, task_id=task_id, robot_family="workcell")

        rng = random.Random(seed)
        actions: list[Any] = []
        states: list[Mapping[str, Any]] = []
        rewards: list[float] = []
        task_progress: list[float] = []
        info_history: list[Mapping[str, Any]] = []
        rgb_frames: list[Any] = []

        step_count = 0
        total_reward = 0.0
        total_errors = 0
        energy_wh = 0.0
        contact_force_sum = 0.0
        constraint_error_sum = 0.0
        collision_count = 0

        target_steps = _target_steps(config, objective)
        last_progress = 0.0
        success = False

        while step_count < config.max_steps:
            action = self._select_action(obs, objective, rng)
            obs, info, _ = env.step(action)
            state = env.physics_adapter.get_state()
            contact_force_sum += float(state.get("contact_force_N", 0.0))
            constraint_error_sum += float(state.get("constraint_error", 0.0))
            collision_count += int(state.get("collision_count", 0))

            progress = min(1.0, (step_count + 1) / max(target_steps, 1))
            progress_delta = max(progress - last_progress, 0.0)
            last_progress = progress
            error_flag = 1 if rng.random() < _error_probability(objective) else 0
            total_errors += error_flag

            reward_result = compute_workcell_reward(
                reward_terms,
                task_reward=info.get("reward") if isinstance(info, Mapping) else None,
                task_info=info.get("task") if isinstance(info, Mapping) else None,
                step_time_s=config.time_step_s,
                process_reward=info.get("process_reward") if isinstance(info, Mapping) else None,
                success=progress >= 1.0,
                progress=progress_delta,
                error_count=error_flag,
                tolerance_met=progress >= 1.0,
            )

            reward = reward_result.reward
            total_reward += reward
            energy_wh += _estimate_energy_wh(action, config.time_step_s)

            actions.append(_serialize_action(action))
            states.append(state)
            rewards.append(reward)
            task_progress.append(progress)
            if isinstance(info, Mapping):
                info_history.append(info)

            if config.capture_rgb_frames and len(rgb_frames) < config.render_max_frames:
                if _should_capture_frame(step_count, config):
                    frame = _render_frame(env.physics_adapter, config)
                    if frame is not None:
                        rgb_frames.append(frame)

            step_count += 1
            if progress >= 1.0:
                success = True
                break

        duration_s = float(step_count) * float(config.time_step_s)
        termination_reason = "success" if success else "max_steps"
        mean_reward = total_reward / max(step_count, 1)
        error_rate = float(total_errors) / max(step_count, 1)
        mpl_units_per_hour = _estimate_mpl_units_per_hour(success, duration_s)
        mean_contact_force = contact_force_sum / max(step_count, 1)
        mean_constraint_error = constraint_error_sum / max(step_count, 1)

        trajectory_data = {
            "scene_spec": env.scene_spec.to_dict(),
            "actions": actions,
            "states": states,
            "rewards": rewards,
            "task_graph_progress": task_progress,
            "info_history": info_history,
            "policy_id": policy_id,
        }

        metrics = {
            "reward_sum": float(total_reward),
            "mean_reward": float(mean_reward),
            "error_rate": float(error_rate),
            "energy_wh": float(energy_wh),
            "contact_force_N": float(mean_contact_force),
            "constraint_error": float(mean_constraint_error),
            "collision_count": float(collision_count),
            "mpl_units_per_hour": float(mpl_units_per_hour),
            "success_rate": 1.0 if success else 0.0,
            "steps": float(step_count),
            "episode_length_s": float(duration_s),
            "task_progress": float(task_progress[-1] if task_progress else 0.0),
        }

        episode_id = f"{policy_id}_ep{episode_idx:03d}_{env.scene_spec.workcell_id}"

        return WorkcellEpisodeResult(
            episode_id=episode_id,
            steps=step_count,
            duration_s=duration_s,
            success=success,
            termination_reason=termination_reason,
            reward_sum=total_reward,
            mean_reward=mean_reward,
            error_count=total_errors,
            error_rate=error_rate,
            energy_wh=energy_wh,
            mpl_units_per_hour=mpl_units_per_hour,
            task_progress=task_progress[-1] if task_progress else 0.0,
            scene_spec=env.scene_spec.to_dict(),
            trajectory_data=trajectory_data,
            metrics=metrics,
            rgb_frames=rgb_frames or None,
            depth_frames=None,
        )

    def _select_action(
        self,
        obs: Mapping[str, Any],
        objective: EconomicObjectiveSpec,
        rng: random.Random,
    ) -> Mapping[str, Any]:
        scale = 0.03 + 0.02 * max(0.0, objective.mpl_weight) - 0.01 * max(0.0, objective.error_weight)
        scale = max(0.01, min(0.08, scale))
        dx = rng.uniform(-scale, scale)
        dy = rng.uniform(-scale, scale)
        dz = rng.uniform(-scale * 0.5, scale * 0.5)
        return {
            "object_id": "end_effector",
            "delta_position": (dx, dy, dz),
        }

    def _aggregate_episode_metrics(
        self,
        results: list[WorkcellEpisodeResult],
        phase: str,
    ) -> dict[str, float]:
        if not results:
            return {}
        success_rate = sum(1 for r in results if r.success) / len(results)
        mean_reward = sum(r.reward_sum for r in results) / len(results)
        mean_error = sum(r.error_rate for r in results) / len(results)
        mean_energy = sum(r.energy_wh for r in results) / len(results)
        mean_duration = sum(r.duration_s for r in results) / len(results)
        mean_mpl = sum(r.mpl_units_per_hour for r in results) / len(results)
        mean_progress = sum(r.task_progress for r in results) / len(results)

        return {
            "mean_reward": float(mean_reward),
            "success_rate": float(success_rate),
            "error_rate": float(mean_error),
            "energy_wh": float(mean_energy),
            "mean_episode_length_s": float(mean_duration),
            "mpl_units_per_hour": float(mean_mpl),
            "mean_task_progress": float(mean_progress),
            "num_episodes": float(len(results)),
            "env_type": "workcell_env",
        }

    def _build_reward_terms(self, objective: EconomicObjectiveSpec) -> WorkcellRewardTerms:
        dense_progress = 0.1 + 0.05 * max(0.0, objective.mpl_weight)
        error_penalty = -0.1 - 0.05 * max(0.0, objective.error_weight)
        time_penalty = -0.01 - 0.005 * max(0.0, objective.energy_weight)
        return WorkcellRewardTerms(
            sparse_success=1.0,
            dense_progress=dense_progress,
            time_penalty=time_penalty,
            error_penalty=error_penalty,
            tolerance_bonus=0.0,
        )

    def _apply_datapack_overrides(
        self,
        config: WorkcellEnvConfig,
        datapack_bundle: DatapackBundle,
    ) -> WorkcellEnvConfig:
        if not datapack_bundle:
            return config
        merged = config.to_dict()
        for overrides in (datapack_bundle.randomization_overrides, datapack_bundle.curriculum_overrides):
            if not isinstance(overrides, Mapping):
                continue
            for key, value in overrides.items():
                if key in merged:
                    merged[key] = value
        return WorkcellEnvConfig.from_dict(merged)

    def _capture_rollouts(
        self,
        *,
        scenario_id: str,
        base_dir: Path,
        episode_results: Sequence[WorkcellEpisodeResult],
        task_id: str,
        seed: int | None,
        config: WorkcellEnvConfig,
        datapack_bundle: DatapackBundle | None,
        max_episodes: int,
    ) -> RolloutBundle:
        start_rollout_capture(scenario_id, base_dir)
        base_env_params = _build_env_params(config, datapack_bundle)
        for idx, result in enumerate(episode_results[:max_episodes]):
            env_params = dict(base_env_params)
            env_params["scene_spec"] = result.scene_spec
            episode_meta = EpisodeMetadata(
                episode_id=result.episode_id,
                task_id=task_id,
                robot_family="workcell",
                seed=seed,
                env_params=env_params,
            )
            sensor_bundle = None
            if config.capture_sensor_bundle:
                sensor_bundle = _build_sensor_bundle(
                    scene_spec=result.scene_spec,
                    states=list(result.trajectory_data.get("states", [])),
                    config=config,
                    seed=seed,
                )
            record_episode_rollout(
                scenario_id=scenario_id,
                episode_idx=idx,
                metadata=episode_meta,
                trajectory_data=result.trajectory_data,
                rgb_frames=result.rgb_frames,
                depth_frames=result.depth_frames,
                metrics=result.metrics,
                base_dir=base_dir,
                sensor_bundle=sensor_bundle,
            )
            if config.enable_scene_tracks:
                try:
                    from src.vision.scene_ir_tracker.io.scene_tracks_runner import run_scene_tracks

                    episode_dir = base_dir / scenario_id / f"episode_{idx:03d}"
                    mode = "rgb" if result.rgb_frames else "vector_proxy"
                    run_scene_tracks(
                        datapack_path=episode_dir,
                        output_path=episode_dir,
                        seed=seed,
                        max_frames=config.render_max_frames,
                        camera="front",
                        mode=mode,
                    )
                except Exception as exc:
                    logger.warning("SceneTracks generation failed for episode %s: %s", result.episode_id, exc)
        return finalize_rollout_bundle(scenario_id, base_dir)


def _serialize_action(action: Any) -> Any:
    if hasattr(action, "tolist"):
        return action.tolist()
    if isinstance(action, tuple):
        return list(action)
    if isinstance(action, dict):
        payload = dict(action)
        for key in ("delta_position", "target_position", "position", "velocity"):
            if key in payload and isinstance(payload[key], tuple):
                payload[key] = list(payload[key])
        return payload
    return action


def _estimate_energy_wh(action: Any, time_step_s: float) -> float:
    magnitude = 0.0
    if isinstance(action, dict):
        delta = action.get("delta_position") or action.get("target_position") or action.get("position")
        if isinstance(delta, (list, tuple)) and len(delta) == 3:
            magnitude = (float(delta[0]) ** 2 + float(delta[1]) ** 2 + float(delta[2]) ** 2) ** 0.5
    elif isinstance(action, (list, tuple)) and len(action) == 3:
        magnitude = (float(action[0]) ** 2 + float(action[1]) ** 2 + float(action[2]) ** 2) ** 0.5
    return max(0.0, magnitude * max(time_step_s, 0.0) * 20.0)


def _estimate_mpl_units_per_hour(success: bool, duration_s: float) -> float:
    if not success or duration_s <= 0.0:
        return 0.0
    return 3600.0 / duration_s


def _should_capture_frame(step_idx: int, config: WorkcellEnvConfig) -> bool:
    if config.render_fps <= 0:
        return False
    dt = max(config.time_step_s, 1e-6)
    stride = max(1, int(round(1.0 / (dt * config.render_fps))))
    return (step_idx % stride) == 0


def _render_frame(adapter: Any, config: WorkcellEnvConfig) -> Any | None:
    render_fn = getattr(adapter, "render", None)
    if not callable(render_fn):
        return None
    try:
        return render_fn(camera_name="front", width=config.render_width, height=config.render_height)
    except TypeError:
        try:
            return render_fn()
        except Exception:
            return None


def _build_sensor_bundle(
    *,
    scene_spec: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    config: WorkcellEnvConfig,
    seed: int | None,
) -> Any | None:
    if not states:
        return None
    indices = _select_frame_indices(len(states), config)
    if not indices:
        return None
    selected_states = [states[i] for i in indices]
    timestamps = []
    for idx, state in zip(indices, selected_states):
        ts = state.get("time_s") if isinstance(state, Mapping) else None
        if ts is None:
            ts = float(idx) * float(config.time_step_s)
        timestamps.append(float(ts))

    cameras = list(config.sensor_cameras or ("front",))
    noise_seed = config.sensor_noise_seed if config.sensor_noise_seed is not None else seed
    noise_config = dict(config.sensor_noise or {})

    rgb: dict[str, Any] = {}
    depth: dict[str, Any] = {}
    seg: dict[str, Any] = {}
    intrinsics: dict[str, Any] = {}
    extrinsics: dict[str, Any] = {}

    try:
        from src.envs.workcell_env.observations.mujoco_render import render_workcell_frames
        from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec
        from src.motor_backend.sensor_bundle import SensorBundleData

        if isinstance(scene_spec, WorkcellSceneSpec):
            spec = scene_spec
        else:
            spec = WorkcellSceneSpec.from_dict(scene_spec)

        for camera in cameras:
            frames, depth_frames, seg_frames, camera_params = render_workcell_frames(
                scene_spec=spec,
                states=selected_states,
                camera_name=camera,
                width=config.render_width,
                height=config.render_height,
                max_frames=len(selected_states),
                seed=noise_seed,
                sensor_noise=noise_config or None,
            )
            if frames:
                rgb[camera] = np.asarray(frames, dtype=np.uint8)
            if depth_frames:
                depth[camera] = np.asarray(depth_frames, dtype=np.float32)
            if seg_frames:
                seg[camera] = np.asarray(seg_frames, dtype=np.int32)
            intrinsics[camera] = {
                "fx": float(camera_params.fx),
                "fy": float(camera_params.fy),
                "cx": float(camera_params.cx),
                "cy": float(camera_params.cy),
                "width": int(camera_params.width),
                "height": int(camera_params.height),
            }
            extrinsics[camera] = np.asarray(camera_params.world_from_cam, dtype=np.float32)

        return SensorBundleData(
            cameras=cameras,
            rgb=rgb,
            depth=depth,
            seg=seg,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            timestamps_s=timestamps,
            depth_unit=config.sensor_depth_unit,
            noise_config=noise_config,
            noise_seed=noise_seed,
        )
    except Exception:
        return None


def _select_frame_indices(num_steps: int, config: WorkcellEnvConfig) -> list[int]:
    if num_steps <= 0:
        return []
    indices = [idx for idx in range(num_steps) if _should_capture_frame(idx, config)]
    if config.render_max_frames > 0:
        indices = indices[: config.render_max_frames]
    return indices


def _target_steps(config: WorkcellEnvConfig, objective: EconomicObjectiveSpec) -> int:
    speed_bias = 1.0 + 0.2 * max(0.0, objective.mpl_weight) - 0.1 * max(0.0, objective.error_weight)
    speed_bias = max(0.5, min(1.5, speed_bias))
    return max(1, int(config.max_steps / speed_bias))


def _error_probability(objective: EconomicObjectiveSpec) -> float:
    base = 0.05 + 0.02 * max(0.0, objective.error_weight) + 0.01 * max(0.0, objective.risk_weight)
    return max(0.0, min(0.3, base))


def _build_env_params(
    config: WorkcellEnvConfig,
    datapack_bundle: DatapackBundle | None,
) -> Mapping[str, Any]:
    params: dict[str, Any] = {
        "config": config.to_dict(),
        "difficulty_features": compute_difficulty_features(
            part_count=config.num_parts,
            occlusion_level=config.occlusion_level,
            tolerance_mm=config.tolerance_mm,
            max_steps=config.max_steps,
            tool_changes_required=config.tool_changes_required,
        ).to_dict(),
    }
    if datapack_bundle:
        params["datapack_ids"] = list(datapack_bundle.datapack_ids)
        params["randomization_overrides"] = dict(datapack_bundle.randomization_overrides)
        params["curriculum_overrides"] = dict(datapack_bundle.curriculum_overrides)
    return params


def _apply_anti_reward_hacking_flags(
    econ_metrics: Mapping[str, float],
    raw_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not raw_metrics:
        return econ_metrics
    suspicious, reasons, summary = analyze_anti_reward_hacking(raw_metrics)
    merged: dict[str, Any] = dict(econ_metrics)
    merged["anti_reward_hacking_suspicious"] = 1.0 if suspicious else 0.0
    if reasons:
        merged["anti_reward_hacking_reason"] = "; ".join(reasons)
    merged.setdefault("anti_reward_hacking_summary", summary)
    return merged
