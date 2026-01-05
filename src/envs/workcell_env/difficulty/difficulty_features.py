"""
Difficulty feature extraction for workcell environments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from src.envs.workcell_env.config import WorkcellEnvConfig


@dataclass(frozen=True)
class WorkcellDifficultyFeatures:
    """
    Difficulty features for a workcell episode.
    """
    part_count: int
    occlusion_proxy: float
    tolerance_factor: float
    horizon_length: int
    tool_changes: int
    assembly_depth: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "part_count": self.part_count,
            "occlusion_proxy": self.occlusion_proxy,
            "tolerance_factor": self.tolerance_factor,
            "horizon_length": self.horizon_length,
            "tool_changes": self.tool_changes,
            "assembly_depth": self.assembly_depth,
        }

    def composite_difficulty(self) -> float:
        """
        Compute a normalized composite difficulty score in [0, 1].

        Combines all difficulty features into a single scalar.
        """
        # Normalize part_count: assume 1-50 parts maps to 0-1
        part_norm = min(self.part_count / 50.0, 1.0)

        # occlusion_proxy is already 0-1
        occlusion_norm = min(max(self.occlusion_proxy, 0.0), 1.0)

        # tolerance_factor: higher = harder (1/mm). Assume 0.1-10 maps to 0-1
        tolerance_norm = min(self.tolerance_factor / 10.0, 1.0)

        # horizon_length: assume 50-500 steps maps to 0-1
        horizon_norm = min(max((self.horizon_length - 50) / 450.0, 0.0), 1.0)

        # tool_changes: assume 0-5 maps to 0-1
        tool_norm = min(self.tool_changes / 5.0, 1.0)

        # assembly_depth: assume 1-10 maps to 0-1
        depth_norm = min((self.assembly_depth - 1) / 9.0, 1.0)

        # Weighted average
        weights = [0.2, 0.15, 0.25, 0.15, 0.1, 0.15]
        values = [part_norm, occlusion_norm, tolerance_norm, horizon_norm, tool_norm, depth_norm]

        composite = sum(w * v for w, v in zip(weights, values))
        return min(max(composite, 0.0), 1.0)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkcellDifficultyFeatures":
        """Deserialize from dictionary."""
        return cls(
            part_count=data.get("part_count", 0),
            occlusion_proxy=data.get("occlusion_proxy", 0.0),
            tolerance_factor=data.get("tolerance_factor", 0.0),
            horizon_length=data.get("horizon_length", 0),
            tool_changes=data.get("tool_changes", 0),
            assembly_depth=data.get("assembly_depth", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkcellDifficultyFeatures":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def compute_difficulty_features(
    config_or_part_count: Union["WorkcellEnvConfig", int, None] = None,
    *,
    part_count: int = 0,
    occlusion_level: float = 0.0,
    tolerance_mm: float = 1.0,
    max_steps: int = 100,
    tool_changes_required: int = 0,
    assembly_depth: int = 1,
) -> WorkcellDifficultyFeatures:
    """
    Compute difficulty features from a config object or keyword arguments.

    Can be called as:
        compute_difficulty_features(config)
        compute_difficulty_features(part_count=12, tolerance_mm=2.0, ...)
    """
    # Check if first arg is a WorkcellEnvConfig
    if config_or_part_count is not None:
        # Import here to avoid circular import
        from src.envs.workcell_env.config import WorkcellEnvConfig

        if isinstance(config_or_part_count, WorkcellEnvConfig):
            cfg = config_or_part_count
            part_count = cfg.num_parts
            occlusion_level = cfg.occlusion_level
            tolerance_mm = cfg.tolerance_mm
            max_steps = cfg.max_steps
            tool_changes_required = cfg.tool_changes_required
            assembly_depth = 1  # Default; not in config
        elif isinstance(config_or_part_count, int):
            # Called with positional part_count (legacy support)
            part_count = config_or_part_count

    tolerance_factor = 1.0 / max(float(tolerance_mm), 1e-6)
    return WorkcellDifficultyFeatures(
        part_count=int(part_count),
        occlusion_proxy=float(occlusion_level),
        tolerance_factor=float(tolerance_factor),
        horizon_length=int(max_steps),
        tool_changes=int(tool_changes_required),
        assembly_depth=int(assembly_depth),
    )
