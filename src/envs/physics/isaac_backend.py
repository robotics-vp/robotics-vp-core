"""Isaac shadow backend with canonical observation and summary contracts."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np

from .base_engine import PhysicsBackend
from src.envs.dishwashing_env import EpisodeInfoSummary


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


class IsaacBackend(PhysicsBackend):
    """Explicit shadow-contract Isaac backend for adapter/runtime integration.

    This backend does not pretend to be a full Isaac Sim / Isaac Gym executor.
    It provides a deterministic shadow execution loop so higher layers can
    exercise canonical observation, media, energy, and summary contracts before
    the real runtime/assets are available on the host.
    """

    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        num_envs: int = 1,
        device: str = "cuda:0",
    ):
        self.env_config = env_config or {}
        self._num_envs = max(1, int(num_envs))
        self.device = device
        self._env_name = (
            str(self.env_config.get("env_name", "isaac_default")) or "isaac_default"
        )
        self._task_name = str(self.env_config.get("task", self._env_name) or self._env_name)
        self._robot_name = str(self.env_config.get("robot", "unitree_shadow") or "unitree_shadow")
        self._mode = str(self.env_config.get("backend_mode", "shadow_contract") or "shadow_contract")
        self._max_steps = max(1, _int(self.env_config.get("max_steps", 16), 16))
        self._dt = max(1e-4, _float(self.env_config.get("dt", 1.0 / 60.0), 1.0 / 60.0))
        self._action_dim = max(1, _int(self.env_config.get("action_dim", 12), 12))
        self._obs_dim = max(self._action_dim * 4, _int(self.env_config.get("obs_dim", 0), 0))
        resolution = (
            self.env_config.get("camera_resolution")
            or self.env_config.get("image_size")
            or self.env_config.get("resolution")
            or [64, 64]
        )
        resolution_values = _sequence(resolution)
        width = _int(resolution_values[0] if resolution_values else 64, 64)
        height = _int(resolution_values[1] if len(resolution_values) > 1 else width, width)
        self._resolution = (max(8, width), max(8, height))
        self._seed = _int(self.env_config.get("seed", 0), 0)
        output_root = self.env_config.get("output_root")
        self._output_root = Path(output_root or Path("results") / "isaac_backend_shadow")
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._adapter = self._make_adapter()

        self._info_history: list[list[Dict[str, Any]]] = [[] for _ in range(self._num_envs)]
        self._step_counts: list[int] = [0 for _ in range(self._num_envs)]
        self._current_observations: list[Optional[Dict[str, Any]]] = [None for _ in range(self._num_envs)]
        self._current_adapted: list[Optional[Dict[str, Any]]] = [None for _ in range(self._num_envs)]
        self._done_flags: list[bool] = [False for _ in range(self._num_envs)]
        self._episode_ids: Dict[int, str] = {}
        self._current_episode_ids: list[Optional[str]] = [None for _ in range(self._num_envs)]
        self._media_refs: Dict[Any, Dict[str, Any]] = {}
        self._closed = False

    @property
    def engine_type(self) -> Literal["pybullet", "isaac", "ue5"]:
        return "isaac"

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, initial_state: Optional[Any] = None) -> Any:
        for env_idx in range(self._num_envs):
            self._reset_env_internal(env_idx, initial_state=initial_state)
        return self._current_observations[0]

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        return self._step_env_internal(0, action)

    def get_episode_info(self) -> EpisodeInfoSummary:
        return self._summarize_env(0)

    def get_info_history(self):
        return list(self._info_history[0])

    def close(self) -> None:
        self._closed = True

    def get_media_refs(self, key: Optional[Any] = None) -> Dict[str, Any]:
        if key is None:
            episode_id = self.get_current_episode_id(0)
            if episode_id is None:
                return {}
            return dict(self._media_refs.get(episode_id, {}))
        if key in self._media_refs:
            return dict(self._media_refs.get(key, {}))
        if isinstance(key, int):
            episode_id = self.get_current_episode_id(key)
            if episode_id and episode_id in self._media_refs:
                return dict(self._media_refs.get(episode_id, {}))
        return {}

    def set_media_refs(self, key: Any, refs: Dict[str, Any]) -> None:
        self._media_refs[key] = dict(refs)
        if isinstance(key, int):
            episode_id = self.get_current_episode_id(key)
            if episode_id:
                self._media_refs[episode_id] = dict(refs)

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._adapter = self._make_adapter()

    def render(self, mode: str = "rgb_array") -> Optional[Any]:
        current = self._current_observations[0]
        if current is None:
            return None
        if mode == "rgb_array":
            return np.asarray(current.get("rgb"))
        return None

    def get_state(self) -> Any:
        return {
            "env_name": self._env_name,
            "task": self._task_name,
            "robot": self._robot_name,
            "device": self.device,
            "mode": self._mode,
            "seed": self._seed,
            "step_counts": list(self._step_counts),
            "done_flags": list(self._done_flags),
            "episode_ids": dict(self._episode_ids),
            "max_steps": self._max_steps,
            "resolution": list(self._resolution),
        }

    def set_state(self, state: Any) -> None:
        payload = _mapping(state)
        if payload:
            self._seed = _int(payload.get("seed", self._seed), self._seed)
            self._step_counts = [
                _int(value, 0)
                for value in _sequence(payload.get("step_counts"))[: self._num_envs]
            ] or [0 for _ in range(self._num_envs)]
            self._done_flags = [
                bool(value)
                for value in _sequence(payload.get("done_flags"))[: self._num_envs]
            ] or [False for _ in range(self._num_envs)]
            episode_ids = _mapping(payload.get("episode_ids"))
            for key, value in episode_ids.items():
                try:
                    env_idx = int(key)
                except Exception:
                    continue
                if 0 <= env_idx < self._num_envs:
                    self._episode_ids[env_idx] = str(value)
                    self._current_episode_ids[env_idx] = str(value)

    def get_observation_space(self) -> Any:
        width, height = self._resolution
        return {
            "rgb": [height, width, 3],
            "depth": [height, width],
            "segmentation": [height, width],
            "joint_positions": [self._action_dim],
            "joint_velocities": [self._action_dim],
            "joint_torques": [self._action_dim],
        }

    def get_action_space(self) -> Any:
        return {
            "shape": [self._action_dim],
            "dtype": "float32",
            "range": [-1.0, 1.0],
        }

    def get_current_episode_id(self, env_idx: Optional[int] = None) -> Optional[str]:
        idx = 0 if env_idx is None else int(env_idx)
        if idx < 0 or idx >= self._num_envs:
            return None
        current = self._current_episode_ids[idx]
        if current:
            return current
        return self._episode_ids.get(idx)

    def get_config(self) -> Dict[str, Any]:
        return {
            "env_name": self._env_name,
            "engine_type": self.engine_type,
            "num_envs": self.num_envs,
            "device": self.device,
            "backend_mode": self._mode,
            "env_config": dict(self.env_config),
        }

    def get_batch_episode_info(self) -> List[EpisodeInfoSummary]:
        return [self._summarize_env(env_idx) for env_idx in range(self._num_envs)]

    def reset_env(self, env_idx: int, initial_state: Optional[Any] = None) -> Any:
        if env_idx < 0 or env_idx >= self._num_envs:
            raise IndexError(f"env_idx {env_idx} out of range for num_envs={self._num_envs}")
        return self._reset_env_internal(env_idx, initial_state=initial_state)

    def _reset_env_internal(self, env_idx: int, initial_state: Optional[Any] = None) -> Dict[str, Any]:
        episode_id = str(uuid.uuid4())
        self._episode_ids[env_idx] = episode_id
        self._current_episode_ids[env_idx] = episode_id
        self._info_history[env_idx] = []
        self._step_counts[env_idx] = 0
        self._done_flags[env_idx] = False
        observation = self._build_observation(env_idx, action=None, timestep=0, initial_state=initial_state)
        self._current_observations[env_idx] = observation
        self._current_adapted[env_idx] = self._adapt_observation(env_idx, observation, timestep=0)
        return observation

    def _step_env_internal(
        self,
        env_idx: int,
        action: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done_flags[env_idx]:
            observation = self._current_observations[env_idx] or self._build_observation(
                env_idx,
                action=action,
                timestep=self._step_counts[env_idx],
            )
            info = self._build_step_info(
                env_idx,
                action,
                observation,
                timestep=self._step_counts[env_idx],
                reward=0.0,
                done=True,
            )
            return observation, 0.0, True, info

        timestep = self._step_counts[env_idx] + 1
        observation = self._build_observation(env_idx, action=action, timestep=timestep)
        self._current_observations[env_idx] = observation
        adapted = self._adapt_observation(env_idx, observation, timestep=timestep)
        self._current_adapted[env_idx] = adapted
        reward = self._compute_reward(env_idx, action=action, timestep=timestep)
        done = timestep >= self._max_steps
        self._done_flags[env_idx] = done
        self._step_counts[env_idx] = timestep
        info = self._build_step_info(
            env_idx,
            action,
            observation,
            timestep=timestep,
            reward=reward,
            done=done,
        )
        self._info_history[env_idx].append(info)
        return observation, reward, done, info

    def _episode_sequence(self, env_idx: int) -> list[Dict[str, Any]]:
        sequence = self.env_config.get("observation_sequence")
        if isinstance(sequence, dict):
            env_sequence = sequence.get(str(env_idx), sequence.get(env_idx))
            return [dict(item) for item in _sequence(env_sequence) if isinstance(item, dict)]
        return [dict(item) for item in _sequence(sequence) if isinstance(item, dict)]

    def _build_observation(
        self,
        env_idx: int,
        *,
        action: Any,
        timestep: int,
        initial_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        sequence = self._episode_sequence(env_idx)
        if sequence:
            index = min(max(0, timestep), len(sequence) - 1)
            observation = dict(sequence[index])
            observation.setdefault("action", action)
            observation.setdefault("dt", self._dt)
            observation.setdefault("camera_name", f"isaac_shadow_cam_{env_idx}")
            return observation

        initial_payload = _mapping(initial_state)
        width, height = self._resolution
        action_vector = self._normalize_action(action)
        phase = 0.05 * float(timestep) + 0.1 * float(env_idx)
        gradient_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        gradient_y = np.linspace(0.0, 1.0, height, dtype=np.float32)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip((gradient_x[None, :] + phase) * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip((gradient_y[:, None] + 0.5 * phase) * 255.0, 0, 255).astype(
            np.uint8
        )
        rgb[..., 2] = np.uint8((37 * (env_idx + 1) + 11 * timestep) % 255)
        depth = np.full((height, width), 1.0 + 0.02 * env_idx + 0.03 * timestep, dtype=np.float32)
        segmentation = np.full((height, width), env_idx + 1, dtype=np.uint8)

        joint_positions = [
            0.05 * (env_idx + 1) + 0.01 * timestep + 0.001 * joint_idx
            for joint_idx in range(self._action_dim)
        ]
        joint_velocities = [
            0.01 * (joint_idx + 1) + 0.005 * timestep for joint_idx in range(self._action_dim)
        ]
        joint_torques = [
            0.15 + abs(action_vector[joint_idx]) * 0.4 + 0.01 * timestep
            for joint_idx in range(self._action_dim)
        ]
        camera_pose = initial_payload.get("camera_pose") or {
            "frame": "world",
            "translation": [0.0, 0.0, 1.0 + 0.05 * env_idx],
        }
        return {
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "contact_forces": [0.0 if timestep < self._max_steps else 1.0, float(env_idx)],
            "end_effector_pose": {
                "position": [0.1 + 0.01 * timestep, 0.05 * env_idx, 0.2],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "dt": self._dt,
            "camera_intrinsics": {"resolution": [width, height], "fov_deg": 90.0},
            "camera_extrinsics": camera_pose,
            "camera_pose": camera_pose,
            "camera_name": f"isaac_shadow_cam_{env_idx}",
            "tf": {
                "base_link": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                }
            },
            "action": {"joint_command": action_vector},
            "timestamp": timestep * self._dt,
            "env_idx": env_idx,
            "backend_mode": self._mode,
            "task": self._task_name,
            "robot": self._robot_name,
        }

    def _normalize_action(self, action: Any) -> list[float]:
        if isinstance(action, dict):
            values = _sequence(action.get("joint_command") or action.get("action") or action.get("value"))
        else:
            values = _sequence(action)
        if not values:
            values = [0.0 for _ in range(self._action_dim)]
        floats = [_float(value, 0.0) for value in values[: self._action_dim]]
        if len(floats) < self._action_dim:
            floats.extend([0.0 for _ in range(self._action_dim - len(floats))])
        return floats

    def _adapt_observation(
        self,
        env_idx: int,
        observation: Dict[str, Any],
        *,
        timestep: int,
    ) -> Dict[str, Any]:
        episode_id = self.get_current_episode_id(env_idx) or str(uuid.uuid4())
        adapted = self._adapter.adapt(
            observation,
            episode_id=episode_id,
            task_id=self._task_name,
            timestep=timestep,
        )
        vision_frame = adapted.get("vision_frame")
        refs = {
            "rgb_path": getattr(vision_frame, "rgb_path", None),
            "depth_path": getattr(vision_frame, "depth_path", None),
            "segmentation_path": getattr(vision_frame, "segmentation_path", None),
            "state_digest": adapted.get("state_digest"),
        }
        clean_refs = {key: value for key, value in refs.items() if value}
        self.set_media_refs(env_idx, clean_refs)
        self._media_refs[episode_id] = clean_refs
        return adapted

    def _make_adapter(self):
        # Import lazily to avoid dragging the orchestrator graph into module import.
        from src.env.isaac_adapter import IsaacAdapter

        return IsaacAdapter(
            config={
                "seed": self._seed,
                "backend": "isaac",
                "env_difficulty": _float(self.env_config.get("env_difficulty", 1.0), 1.0),
                "enable_conditioned_vision": bool(
                    self.env_config.get("enable_conditioned_vision", True)
                ),
            },
            output_root=str(self._output_root),
        )

    def _compute_reward(self, env_idx: int, *, action: Any, timestep: int) -> float:
        action_vector = self._normalize_action(action)
        action_penalty = float(np.mean(np.abs(action_vector))) if action_vector else 0.0
        progress = float(timestep) / float(self._max_steps)
        return max(0.0, 1.0 + progress - 0.2 * action_penalty - 0.02 * env_idx)

    def _build_step_info(
        self,
        env_idx: int,
        action: Any,
        observation: Dict[str, Any],
        *,
        timestep: int,
        reward: float,
        done: bool,
    ) -> Dict[str, Any]:
        adapted = self._current_adapted[env_idx] or {}
        proprio_frame = adapted.get("proprio_frame")
        energy_Wh = _float(getattr(proprio_frame, "energy_estimate_Wh", 0.0), 0.0)
        throughput_units_per_hour = float(timestep) * (3600.0 / max(self._dt, 1e-6))
        mpl_episode = float(timestep) * 10.0
        ep_episode = mpl_episode / max(energy_Wh, 1e-3)
        profit = reward - energy_Wh * _float(
            _mapping(self.env_config.get("econ_params")).get("energy_price_kWh", 0.12),
            0.12,
        )
        energy_share = {"share": 1.0, "energy_Wh": energy_Wh}
        action_vector = self._normalize_action(action)
        return {
            "terminated_reason": "success" if done else "running",
            "mpl": mpl_episode,
            "ep": ep_episode,
            "error_rate": 0.0 if done else 0.01,
            "throughput_units_per_hour": throughput_units_per_hour,
            "energy_Wh": energy_Wh,
            "energy_Wh_per_unit": energy_Wh / max(float(timestep), 1.0),
            "energy_Wh_per_hour": energy_Wh * (3600.0 / max(self._dt * max(timestep, 1), 1e-6)),
            "limb_energy_Wh": {"whole_body": energy_Wh},
            "skill_energy_Wh": {"shadow_control": energy_Wh},
            "energy_per_limb": {"whole_body": energy_share},
            "energy_per_skill": {"shadow_control": energy_share},
            "energy_per_joint": {
                f"joint_{joint_idx:02d}": {"energy_Wh": energy_Wh / float(self._action_dim)}
                for joint_idx in range(self._action_dim)
            },
            "energy_per_effector": {"whole_body": energy_share},
            "coordination_metrics": {
                "mean_abs_action": float(np.mean(np.abs(action_vector))) if action_vector else 0.0,
                "mean_joint_velocity": float(
                    np.mean(_sequence(observation.get("joint_velocities")))
                ),
                "backend_mode_shadow": 1.0,
            },
            "profit": profit,
            "wage_parity": reward / max(1.0, float(timestep)),
            "episode_id": self.get_current_episode_id(env_idx),
            "media_refs": self.get_media_refs(env_idx),
            "state_digest": adapted.get("state_digest"),
            "domain_randomization": adapted.get("domain_randomization", {}),
            "backend_mode": self._mode,
        }

    def _summarize_env(self, env_idx: int) -> EpisodeInfoSummary:
        history = self._info_history[env_idx]
        if not history:
            return EpisodeInfoSummary(
                termination_reason="not_started",
                mpl_episode=0.0,
                ep_episode=0.0,
                error_rate_episode=0.0,
                throughput_units_per_hour=0.0,
                energy_Wh=0.0,
                energy_Wh_per_unit=0.0,
                energy_Wh_per_hour=0.0,
                limb_energy_Wh={},
                skill_energy_Wh={},
                energy_per_limb={},
                energy_per_skill={},
                energy_per_joint={},
                energy_per_effector={},
                coordination_metrics={"backend_mode_shadow": 1.0},
                profit=0.0,
                episode_id=self.get_current_episode_id(env_idx) or str(uuid.uuid4()),
                media_refs=self.get_media_refs(env_idx),
                wage_parity=0.0,
            )

        last = history[-1]
        return EpisodeInfoSummary(
            termination_reason=str(last.get("terminated_reason", "unknown")),
            mpl_episode=_float(last.get("mpl", 0.0), 0.0),
            ep_episode=_float(last.get("ep", 0.0), 0.0),
            error_rate_episode=_float(last.get("error_rate", 0.0), 0.0),
            throughput_units_per_hour=_float(last.get("throughput_units_per_hour", 0.0), 0.0),
            energy_Wh=_float(last.get("energy_Wh", 0.0), 0.0),
            energy_Wh_per_unit=_float(last.get("energy_Wh_per_unit", 0.0), 0.0),
            energy_Wh_per_hour=_float(last.get("energy_Wh_per_hour", 0.0), 0.0),
            limb_energy_Wh=_mapping(last.get("limb_energy_Wh")),
            skill_energy_Wh=_mapping(last.get("skill_energy_Wh")),
            energy_per_limb=_mapping(last.get("energy_per_limb")),
            energy_per_skill=_mapping(last.get("energy_per_skill")),
            energy_per_joint=_mapping(last.get("energy_per_joint")),
            energy_per_effector=_mapping(last.get("energy_per_effector")),
            coordination_metrics=_mapping(last.get("coordination_metrics")),
            profit=_float(last.get("profit", 0.0), 0.0),
            episode_id=str(last.get("episode_id") or self.get_current_episode_id(env_idx) or uuid.uuid4()),
            media_refs=self.get_media_refs(env_idx),
            wage_parity=_float(last.get("wage_parity", 0.0), 0.0),
        )
