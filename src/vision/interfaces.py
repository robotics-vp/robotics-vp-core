"""
Vision/VLA interface contracts, JSON-safe and deterministic.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

from src.utils.json_safe import to_json_safe


@dataclass
class VisionFrame:
    backend: str
    task_id: str
    episode_id: str
    timestep: int
    width: int = 0
    height: int = 0
    channels: int = 3
    dtype: str = "uint8"
    camera_pose: Dict[str, Any] = field(default_factory=dict)
    camera_intrinsics: Dict[str, Any] = field(default_factory=dict)
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    segmentation_path: Optional[str] = None
    camera_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))

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
        return to_json_safe(asdict(self))

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
        return to_json_safe(payload)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyObservation":
        d = dict(d)
        latent = VisionLatent.from_dict(d.pop("latent"))
        return cls(latent=latent, **d)
