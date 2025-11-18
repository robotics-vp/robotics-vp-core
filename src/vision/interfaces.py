"""
Vision/VLA interface contracts, JSON-safe and deterministic.
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

from src.utils.json_safe import to_json_safe


def _sorted_json_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically sort keys to keep downstream hashing stable.
    """
    try:
        return json.loads(json.dumps(to_json_safe(payload), sort_keys=True))
    except Exception:
        return to_json_safe(payload)


def compute_state_digest(state: Dict[str, Any]) -> str:
    """
    Deterministic digest for backend state summaries.
    """
    serialized = json.dumps(to_json_safe(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class VisionFrame:
    backend: str
    task_id: str
    episode_id: str
    timestep: int
    backend_id: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 3
    dtype: str = "uint8"
    camera_pose: Dict[str, Any] = field(default_factory=dict)
    camera_intrinsics: Dict[str, Any] = field(default_factory=dict)
    camera_extrinsics: Dict[str, Any] = field(default_factory=dict)
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    segmentation_path: Optional[str] = None
    camera_name: Optional[str] = None
    state_digest: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_id = self.backend_id or self.backend
        # Mirror pose into extrinsics if caller only set one of them
        if self.camera_extrinsics and not self.camera_pose:
            self.camera_pose = dict(self.camera_extrinsics)
        if self.camera_pose and not self.camera_extrinsics:
            self.camera_extrinsics = dict(self.camera_pose)
        if self.state_digest is None and self.metadata.get("state") is not None:
            self.state_digest = compute_state_digest(self.metadata["state"])

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionFrame":
        return cls(**d)


@dataclass
class VisionLatent:
    backend: str
    task_id: str
    episode_id: str
    timestep: int
    latent: List[float]
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionLatent":
        return cls(**d)


@dataclass
class PolicyObservation:
    task_id: str
    episode_id: str
    timestep: int
    latent: VisionLatent
    state_summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["latent"] = self.latent.to_dict()
        return _sorted_json_safe(payload)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyObservation":
        d = dict(d)
        latent = VisionLatent.from_dict(d.pop("latent"))
        return cls(latent=latent, **d)
