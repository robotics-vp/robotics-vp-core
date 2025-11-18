"""
Mobility micro-policy interfaces (advisory-only).
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Protocol

from src.utils.json_safe import to_json_safe


@dataclass
class MobilityContext:
    task_id: str
    episode_id: str
    env_name: str
    timestep: int
    pose: Dict[str, float] = field(default_factory=dict)
    contacts: Dict[str, float] = field(default_factory=dict)
    target_precision_mm: float = 5.0
    stability_margin: float = 1.0  # 0–1 heuristic
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))


@dataclass
class MobilityAdjustment:
    delta_pose: Dict[str, float] = field(default_factory=dict)  # small offsets
    stabilization_hint: str = ""
    precision_gate_passed: bool = True
    recovery_required: bool = False
    risk_level: str = "LOW"  # "LOW" | "MEDIUM" | "HIGH"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))


class MobilityPolicy(Protocol):
    def compute_adjustment(self, ctx: MobilityContext) -> MobilityAdjustment:
        ...
