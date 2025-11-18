"""
Types for RewardModel policy outputs.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from src.utils.json_safe import to_json_safe


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass
class RewardModelEpisodeScores:
    episode_id: str
    progress_estimate: float  # 0–1
    quality_score: float  # 0–1
    error_probability: float  # 0–1
    subtask_labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.progress_estimate = _clamp01(self.progress_estimate)
        self.quality_score = _clamp01(self.quality_score)
        self.error_probability = _clamp01(self.error_probability)
        self.subtask_labels = sorted({str(lbl) for lbl in self.subtask_labels if lbl})

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["subtask_labels"] = list(self.subtask_labels)
        return to_json_safe(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardModelEpisodeScores":
        return cls(
            episode_id=str(d.get("episode_id", "")),
            progress_estimate=_clamp01(d.get("progress_estimate", 0.0)),
            quality_score=_clamp01(d.get("quality_score", 0.0)),
            error_probability=_clamp01(d.get("error_probability", 0.0)),
            subtask_labels=sorted({str(x) for x in d.get("subtask_labels", []) if x}),
        )
