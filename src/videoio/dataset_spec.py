from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class VideoClipSpec:
    episode_id: str
    env_name: str
    engine_type: str
    file_path: Optional[str] = None
    camera_pose: Optional[Dict[str, float]] = None
    media_refs: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoClipSpec":
        return cls(**d)
