"""ConstraintSet used to condition diffusion/generation prompts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

CONSTRAINT_SET_PREFIX = "constraint_set_v1/"


@dataclass
class ConstraintSet:
    """Hard and soft constraints for a task/env manifold."""

    hard_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    kinematic_limits: Dict[str, float] = field(default_factory=dict)
    affordance_constraints: Dict[str, Any] = field(default_factory=dict)
    safety_invariants: Dict[str, Any] = field(default_factory=dict)
    geometry_priors: Dict[str, float] = field(default_factory=dict)
    source_refs: Dict[str, str] = field(default_factory=dict)
    version: str = "constraint_set_v1"

    def to_prompt_tags(self) -> list[str]:
        tags: list[str] = []
        for axis, bounds in sorted(self.hard_bounds.items()):
            if "min" in bounds:
                tags.append(f"constraint:{axis}_min:{float(bounds['min']):.3f}")
            if "max" in bounds:
                tags.append(f"constraint:{axis}_max:{float(bounds['max']):.3f}")
        for name, value in sorted(self.geometry_priors.items()):
            tags.append(f"geometry:{name}:{float(value):.3f}")
        for name, value in sorted(self.kinematic_limits.items()):
            tags.append(f"kinematic:{name}:{float(value):.3f}")
        for name, value in sorted(self.safety_invariants.items()):
            tags.append(f"safety:{name}:{str(value)}")
        return tags

    def to_structured_fields(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "hard_bounds": dict(self.hard_bounds),
            "kinematic_limits": dict(self.kinematic_limits),
            "affordance_constraints": dict(self.affordance_constraints),
            "safety_invariants": dict(self.safety_invariants),
            "geometry_priors": dict(self.geometry_priors),
            "source_refs": dict(self.source_refs),
            "prompt_tags": self.to_prompt_tags(),
        }

    @classmethod
    def from_artifacts(
        cls,
        semantic_evidence: Optional[Mapping[str, Any]] = None,
        map_first_summary: Optional[Mapping[str, Any]] = None,
        fusion_metrics: Optional[Mapping[str, Any]] = None,
    ) -> "ConstraintSet":
        semantic_evidence = semantic_evidence or {}
        map_first_summary = map_first_summary or {}
        fusion_metrics = fusion_metrics or {}

        map_quality = _safe_float(
            map_first_summary.get(
                "map_first_quality_score",
                map_first_summary.get("quality_score", 0.5),
            ),
            0.5,
        )
        disagreement = _safe_float(
            fusion_metrics.get("semantic_disagreement_vla_vs_map", 0.0),
            0.0,
        )
        confidence = _safe_float(
            fusion_metrics.get("semantic_fusion_confidence_mean", 0.0),
            0.0,
        )
        vla_conf = _safe_float(semantic_evidence.get("vla_confidence", confidence), confidence)

        hard_bounds = {
            "semantic_disagreement_vla_vs_map": {"max": max(0.05, min(1.0, disagreement + 0.2))},
            "map_first_quality_score": {"min": max(0.0, min(1.0, map_quality * 0.8))},
            "vla_confidence": {"min": max(0.0, min(1.0, vla_conf * 0.8))},
        }
        geometry_priors = {
            "bev_occupancy_plausibility": max(0.0, min(1.0, map_quality)),
            "depth_consistency": max(0.0, min(1.0, 1.0 - disagreement)),
            "feature_alignment": max(0.0, min(1.0, confidence)),
        }
        safety_invariants = {
            "respect_fragility": bool(semantic_evidence.get("fragile", False)),
            "no_unknown_collision_zones": bool(semantic_evidence.get("safety_critical", False)),
        }
        affordance_constraints = {
            "required_tags": list(semantic_evidence.get("semantic_tags", []) or []),
        }
        kinematic_limits = {
            "max_joint_velocity": _safe_float(semantic_evidence.get("max_joint_velocity", 1.0), 1.0),
            "max_gripper_force": _safe_float(semantic_evidence.get("max_gripper_force", 1.0), 1.0),
        }
        return cls(
            hard_bounds=hard_bounds,
            kinematic_limits=kinematic_limits,
            affordance_constraints=affordance_constraints,
            safety_invariants=safety_invariants,
            geometry_priors=geometry_priors,
            source_refs={
                "semantic_evidence": str(semantic_evidence.get("source", "")),
                "map_first": str(map_first_summary.get("source", "")),
                "fusion": str(fusion_metrics.get("source", "")),
            },
        )

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        payload = self.to_structured_fields()
        return {
            f"{CONSTRAINT_SET_PREFIX}version": np.array([self.version], dtype="U32"),
            f"{CONSTRAINT_SET_PREFIX}payload_json": np.array(
                [json.dumps(payload, sort_keys=True, default=str)],
                dtype="U16384",
            ),
        }

    @classmethod
    def from_npz_dict(cls, payload: Mapping[str, Any]) -> "ConstraintSet":
        json_key = f"{CONSTRAINT_SET_PREFIX}payload_json"
        if json_key in payload:
            raw = payload[json_key]
        else:
            raw = payload["payload_json"]
        decoded = json.loads(str(raw[0]))
        return cls(
            hard_bounds=dict(decoded.get("hard_bounds", {}) or {}),
            kinematic_limits=dict(decoded.get("kinematic_limits", {}) or {}),
            affordance_constraints=dict(decoded.get("affordance_constraints", {}) or {}),
            safety_invariants=dict(decoded.get("safety_invariants", {}) or {}),
            geometry_priors=dict(decoded.get("geometry_priors", {}) or {}),
            source_refs=dict(decoded.get("source_refs", {}) or {}),
            version=str(decoded.get("version", "constraint_set_v1")),
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
