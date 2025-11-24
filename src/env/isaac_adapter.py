"""
Isaac adapter that normalizes simulator outputs into canonical frames.

Supports RGB + depth + segmentation, contact sensors, joint torque energy
estimates, deterministic backend selection, and domain randomization context.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.ingestion.rollout_types import ActionFrame, EnvStateDigest, ProprioFrame
from src.observation.condition_vector import ConditionVector
from src.utils.json_safe import to_json_safe
from src.vision.conditioned_adapter import ConditionedVisionAdapter
from src.vision.interfaces import VisionFrame, compute_state_digest


class IsaacAdapter:
    def __init__(self, config: Optional[Dict[str, Any]] = None, output_root: Optional[str] = None):
        self.config = config or {}
        self.seed = int(self.config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.backend = self._select_backend(self.config)
        self.enable_conditioned_vision = bool(self.config.get("enable_conditioned_vision", True))
        self.output_root = Path(output_root or Path("results") / "isaac_adapter")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.domain_randomization = self._build_domain_randomization()

    def adapt(
        self,
        observation: Dict[str, Any],
        *,
        episode_id: str,
        task_id: str,
        timestep: int = 0,
        condition_vector: Optional[ConditionVector] = None,
    ) -> Dict[str, Any]:
        episode_dir = self.output_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        vision_frame = self._build_vision_frame(observation, episode_id, task_id, timestep, episode_dir)
        proprio_frame = self._build_proprio_frame(observation, timestep)
        action_frame = self._build_action_frame(observation, timestep)
        env_digest = self._build_env_state(observation, timestep)
        energy_proxies = self._energy_proxies(
            proprio_frame.joint_torques, proprio_frame.joint_velocities, observation.get("dt", proprio_frame.metadata.get("dt", 0.0))
        )
        vision_frame.metadata.setdefault("energy_proxies", energy_proxies)
        proprio_frame.metadata.setdefault("energy_proxies", energy_proxies)
        env_digest.metadata.setdefault("energy_proxies", energy_proxies)
        env_digest.metadata.setdefault("domain_randomization", self.domain_randomization)

        vision_features = {}
        if self.enable_conditioned_vision:
            vision_features = ConditionedVisionAdapter(config={"enable_conditioning": True}).forward(vision_frame, condition_vector)

        return {
            "backend": self.backend,
            "vision_frame": vision_frame,
            "proprio_frame": proprio_frame,
            "action_frame": action_frame,
            "env_digest": env_digest,
            "vision_features": vision_features,
            "energy_proxies": energy_proxies,
            "domain_randomization": self.domain_randomization,
            "state_digest": vision_frame.state_digest,
        }

    # ----- helpers -----
    def _select_backend(self, config: Dict[str, Any]) -> str:
        mode = str(config.get("backend", config.get("engine", "isaac_stub"))).lower()
        if mode in ("isaac", "omniverse"):
            return "isaac"
        if mode in ("pybullet", "bullet"):
            return "pybullet"
        return "isaac_stub"

    def _build_domain_randomization(self) -> Dict[str, Any]:
        lighting = float(self.rng.uniform(0.5, 1.5))
        texture_variant = int(self.rng.integers(0, 8))
        return {
            "lighting": round(lighting, 4),
            "texture_variant": texture_variant,
            "seed": int(self.seed),
            "env_difficulty": float(self.config.get("env_difficulty", 1.0)),
        }

    def _build_vision_frame(
        self,
        observation: Dict[str, Any],
        episode_id: str,
        task_id: str,
        timestep: int,
        episode_dir: Path,
    ) -> VisionFrame:
        rgb = observation.get("rgb")
        depth = observation.get("depth")
        segmentation = observation.get("segmentation")
        rgb_np = np.asarray(rgb) if rgb is not None else None
        depth_np = np.asarray(depth) if depth is not None else None
        h, w, c = (0, 0, 3)
        if rgb_np is not None and rgb_np.size > 0:
            if rgb_np.ndim == 3:
                h, w, c = rgb_np.shape
            elif rgb_np.ndim == 2:
                h, w = rgb_np.shape
                c = 1
        elif depth_np is not None and depth_np.size > 0:
            h, w = depth_np.shape if depth_np.ndim >= 2 else (depth_np.size, 1)
            c = 1

        rgb_path = None
        depth_path = None
        seg_path = None
        if rgb is not None:
            rgb_path = self._write_array(episode_dir, f"{episode_id}_rgb_{timestep:04d}.json", rgb)
        if depth is not None:
            depth_path = self._write_array(episode_dir, f"{episode_id}_depth_{timestep:04d}.json", depth)
        if segmentation is not None:
            seg_path = self._write_array(episode_dir, f"{episode_id}_seg_{timestep:04d}.json", segmentation)

        intrinsics = observation.get("camera_intrinsics") or {"resolution": [int(w), int(h)], "fov_deg": 90.0}
        extrinsics = observation.get("camera_extrinsics") or observation.get("camera_pose") or {"frame": "world"}
        tf_hint = observation.get("tf") or {}
        state_digest = compute_state_digest({"tf": to_json_safe(tf_hint), "timestep": timestep})

        return VisionFrame(
            backend=self.backend,
            backend_id=self.backend,
            task_id=task_id,
            episode_id=episode_id,
            timestep=int(timestep),
            width=int(w),
            height=int(h),
            channels=int(c),
            rgb_path=rgb_path,
            depth_path=depth_path,
            segmentation_path=seg_path,
            camera_intrinsics=to_json_safe(intrinsics),
            camera_extrinsics=to_json_safe(extrinsics),
            camera_pose=to_json_safe(extrinsics),
            camera_name=str(observation.get("camera_name") or "isaac_camera"),
            dtype=str(getattr(rgb_np, "dtype", observation.get("dtype", "uint8"))),
            state_digest=state_digest,
            metadata={
                "backend": self.backend,
                "domain_randomization": self.domain_randomization,
                "tf": to_json_safe(tf_hint),
                "state_digest": state_digest,
            },
        )

    def _build_proprio_frame(self, observation: Dict[str, Any], timestep: int) -> ProprioFrame:
        positions = observation.get("joint_positions") or []
        velocities = observation.get("joint_velocities") or observation.get("joint_velocity") or []
        torques = observation.get("joint_torques") or observation.get("effort") or []
        contacts = observation.get("contact_forces") or observation.get("contacts") or []
        ee_pose = observation.get("end_effector_pose") or observation.get("ee_pose")
        dt = observation.get("dt", 0.02)
        energy_est = self._estimate_energy(torques, velocities, dt)

        contact_sensors = contacts
        try:
            contact_arr = np.asarray(contacts, dtype=float)
            contact_sensors = contact_arr.tolist()
        except Exception:
            pass

        return ProprioFrame(
            timestep=int(timestep),
            joint_positions=positions,
            joint_velocities=velocities,
            joint_torques=torques,
            contact_sensors=contact_sensors,
            end_effector_pose=ee_pose,
            energy_estimate_Wh=energy_est,
            metadata={
                "dt": float(dt),
                "backend": self.backend,
                "domain_randomization": self.domain_randomization,
            },
        )

    def _build_action_frame(self, observation: Dict[str, Any], timestep: int) -> ActionFrame:
        command = observation.get("action") or observation.get("command") or {}
        return ActionFrame(
            timestep=int(timestep),
            command=command,
            metadata={
                "backend": self.backend,
                "timestamp": observation.get("timestamp"),
            },
        )

    def _build_env_state(self, observation: Dict[str, Any], timestep: int) -> EnvStateDigest:
        tf_tree = observation.get("tf") or observation.get("transforms") or {}
        return EnvStateDigest(
            timestep=int(timestep),
            tf_tree=tf_tree,
            metadata={"backend": self.backend},
        )

    def _estimate_energy(self, torques: Any, velocities: Any, dt: float) -> float:
        try:
            tau = np.asarray(torques, dtype=float)
            vel = np.asarray(velocities, dtype=float)
            if tau.shape != vel.shape:
                vel = np.broadcast_to(vel, tau.shape)
            power = np.abs(tau * vel)
            return float(np.sum(power) * float(dt) / 3600.0)
        except Exception:
            return 0.0

    def _energy_proxies(self, torques: Any, velocities: Any, dt: float) -> Dict[str, float]:
        """
        Compute JSON-safe energy/torque proxies for downstream econ calibration.
        """
        try:
            tau = np.asarray(torques, dtype=float)
            vel = np.asarray(velocities, dtype=float)
            if tau.shape != vel.shape:
                vel = np.broadcast_to(vel, tau.shape)
            power = np.abs(tau * vel)
            energy_Wh = float(np.sum(power) * float(dt) / 3600.0)
            torque_abs = float(np.sum(np.abs(tau)))
            power_integral = float(np.sum(power) * float(dt))
            return {
                "energy_proxy_Wh": energy_Wh,
                "torque_abs_sum": torque_abs,
                "power_integral": power_integral,
            }
        except Exception:
            return {"energy_proxy_Wh": 0.0, "torque_abs_sum": 0.0, "power_integral": 0.0}

    def _write_array(self, directory: Path, filename: str, array: Any) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        with open(target, "w") as f:
            json.dump(to_json_safe(array), f, sort_keys=True)
        return str(target)

    def stable_backend_signature(self) -> str:
        payload = {
            "backend": self.backend,
            "seed": self.seed,
            "domain_randomization": self.domain_randomization,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
