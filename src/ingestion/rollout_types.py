"""
Lightweight rollout dataclasses shared by real robot ingestion and simulators.

Designed to stay JSON-safe and deterministic so SIMA-2 can consume them without
extra normalization.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.utils.json_safe import to_json_safe
from src.vision.interfaces import VisionFrame, compute_state_digest


def _safe_list(values: Optional[Any]) -> List[float]:
    if values is None:
        return []
    cleaned: List[float] = []
    for v in values:
        try:
            cleaned.append(float(v))
        except Exception:
            continue
    return cleaned


@dataclass
class ProprioFrame:
    timestep: int
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    joint_torques: List[float] = field(default_factory=list)
    contact_sensors: List[float] = field(default_factory=list)
    end_effector_pose: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    energy_estimate_Wh: float = 0.0
    state_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.state_digest is None:
            payload = {
                "joint_positions": _safe_list(self.joint_positions),
                "joint_velocities": _safe_list(self.joint_velocities),
                "joint_torques": _safe_list(self.joint_torques),
                "contact_sensors": _safe_list(self.contact_sensors),
                "end_effector_pose": to_json_safe(self.end_effector_pose),
                "energy_estimate_Wh": float(self.energy_estimate_Wh),
            }
            self.state_digest = compute_state_digest(payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["joint_positions"] = _safe_list(self.joint_positions)
        payload["joint_velocities"] = _safe_list(self.joint_velocities)
        payload["joint_torques"] = _safe_list(self.joint_torques)
        payload["contact_sensors"] = _safe_list(self.contact_sensors)
        payload["end_effector_pose"] = to_json_safe(self.end_effector_pose)
        payload["metadata"] = to_json_safe(self.metadata)
        payload["energy_estimate_Wh"] = float(self.energy_estimate_Wh)
        return payload


@dataclass
class ActionFrame:
    timestep: int
    command: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.state_digest is None:
            self.state_digest = compute_state_digest({"command": to_json_safe(self.command)})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": int(self.timestep),
            "command": to_json_safe(self.command),
            "metadata": to_json_safe(self.metadata),
            "state_digest": self.state_digest,
        }


@dataclass
class EnvStateDigest:
    timestep: int
    tf_tree: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.digest is None:
            self.digest = compute_state_digest(
                {"tf": to_json_safe(self.tf_tree), "metadata": to_json_safe(self.metadata)}
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": int(self.timestep),
            "tf_tree": to_json_safe(self.tf_tree),
            "metadata": to_json_safe(self.metadata),
            "digest": self.digest,
        }


@dataclass
class RawRollout:
    episode_id: str
    task_id: str
    vision_frames: List[VisionFrame] = field(default_factory=list)
    proprio_frames: List[ProprioFrame] = field(default_factory=list)
    action_frames: List[ActionFrame] = field(default_factory=list)
    env_digests: List[EnvStateDigest] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "vision_frames": [vf.to_dict() for vf in self.vision_frames],
            "proprio_frames": [pf.to_dict() for pf in self.proprio_frames],
            "action_frames": [af.to_dict() for af in self.action_frames],
            "env_digests": [ed.to_dict() for ed in self.env_digests],
            "metadata": to_json_safe(self.metadata),
        }

    def to_datapack_record(self, source: str = "real_robot") -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "source_type": source,
            "rollout": self.to_dict(),
            "vision_frame_count": len(self.vision_frames),
            "proprio_count": len(self.proprio_frames),
            "action_count": len(self.action_frames),
        }
