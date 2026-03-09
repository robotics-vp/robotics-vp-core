"""ConstraintSet used to condition generation, pricing, and governance."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, overload

import numpy as np

CONSTRAINT_SET_PREFIX = "constraint_set_v1/"


@dataclass
class ConstraintSet:
    """Hard and soft constraints for a task/env manifold."""

    hard_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    soft_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    kinematic_limits: Dict[str, float] = field(default_factory=dict)
    affordance_constraints: Dict[str, Any] = field(default_factory=dict)
    safety_invariants: Dict[str, Any] = field(default_factory=dict)
    geometry_priors: Dict[str, float] = field(default_factory=dict)
    geometry_hints: Dict[str, Any] = field(default_factory=dict)
    semantic_evidence: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    trust_metadata: Dict[str, float] = field(default_factory=dict)
    source_refs: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "constraint_set_v1"

    def to_prompt_tags(self) -> list[str]:
        tags: list[str] = []
        for axis, bounds in sorted(self.hard_bounds.items()):
            if "min" in bounds:
                tags.append(f"constraint:{axis}_min:{float(bounds['min']):.3f}")
            if "max" in bounds:
                tags.append(f"constraint:{axis}_max:{float(bounds['max']):.3f}")
        for axis, bounds in sorted(self.soft_bounds.items()):
            if "min" in bounds:
                tags.append(f"soft_constraint:{axis}_min:{float(bounds['min']):.3f}")
            if "max" in bounds:
                tags.append(f"soft_constraint:{axis}_max:{float(bounds['max']):.3f}")
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
            "hard_constraints": dict(self.hard_bounds),
            "soft_bounds": dict(self.soft_bounds),
            "soft_constraints": dict(self.soft_bounds),
            "kinematic_limits": dict(self.kinematic_limits),
            "affordance_constraints": dict(self.affordance_constraints),
            "safety_invariants": dict(self.safety_invariants),
            "geometry_priors": dict(self.geometry_priors),
            "geometry_hints": dict(self.geometry_hints),
            "semantic_evidence": dict(self.semantic_evidence),
            "uncertainty": dict(self.uncertainty),
            "trust_metadata": dict(self.trust_metadata),
            "source_refs": dict(self.source_refs),
            "metadata": dict(self.metadata),
            "prompt_tags": self.to_prompt_tags(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_structured_fields()

    def flag_observations(self, observations: Optional[Mapping[str, Any]] = None) -> list[Dict[str, Any]]:
        observed = _flatten_observations(observations or {})
        flags: list[Dict[str, Any]] = []
        for severity, bounds in (("hard", self.hard_bounds), ("soft", self.soft_bounds)):
            for axis, spec in sorted(bounds.items()):
                obs = observed.get(axis)
                if obs is None:
                    continue
                if "min" in spec and obs < float(spec["min"]):
                    flags.append(
                        {
                            "constraint_id": f"{severity}:{axis}:below_min",
                            "severity": severity,
                            "axis": axis,
                            "flag": "below_min",
                            "threshold": float(spec["min"]),
                            "observed": float(obs),
                        }
                    )
                if "max" in spec and obs > float(spec["max"]):
                    flags.append(
                        {
                            "constraint_id": f"{severity}:{axis}:above_max",
                            "severity": severity,
                            "axis": axis,
                            "flag": "above_max",
                            "threshold": float(spec["max"]),
                            "observed": float(obs),
                        }
                    )
        for name, expected in sorted(self.safety_invariants.items()):
            if not isinstance(expected, bool):
                continue
            observed_value = observed.get(name)
            if observed_value is None:
                continue
            if bool(observed_value) != expected:
                flags.append(
                    {
                        "constraint_id": f"hard:{name}:invariant_violation",
                        "severity": "hard",
                        "axis": name,
                        "flag": "invariant_violation",
                        "threshold": expected,
                        "observed": bool(observed_value),
                    }
                )
        return sorted(flags, key=lambda item: str(item.get("constraint_id", "")))

    def constraint_flags(self, observations: Optional[Mapping[str, Any]] = None) -> list[Dict[str, Any]]:
        return self.flag_observations(observations=observations)

    def summary(self, observations: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        flags = self.flag_observations(observations=observations)
        return {
            "version": self.version,
            "hard_constraint_count": len(self.hard_bounds),
            "soft_constraint_count": len(self.soft_bounds),
            "flag_count": len(flags),
            "hard_flag_count": sum(1 for flag in flags if flag.get("severity") == "hard"),
            "soft_flag_count": sum(1 for flag in flags if flag.get("severity") == "soft"),
            "flags": flags,
        }

    @classmethod
    def from_runtime(
        cls,
        *,
        hard_constraints: Optional[Mapping[str, Mapping[str, Any]]] = None,
        soft_constraints: Optional[Mapping[str, Mapping[str, Any]]] = None,
        geometry_hints: Optional[Mapping[str, Any]] = None,
        semantic_evidence: Optional[Mapping[str, Any]] = None,
        uncertainty: Optional[Mapping[str, Any]] = None,
        trust_metadata: Optional[Mapping[str, Any]] = None,
        kinematic_limits: Optional[Mapping[str, Any]] = None,
        affordance_constraints: Optional[Mapping[str, Any]] = None,
        safety_invariants: Optional[Mapping[str, Any]] = None,
        geometry_priors: Optional[Mapping[str, Any]] = None,
        source_refs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: str = "constraint_set_v1",
    ) -> "ConstraintSet":
        return cls(
            hard_bounds=_coerce_bounds(hard_constraints),
            soft_bounds=_coerce_bounds(soft_constraints),
            kinematic_limits=_coerce_float_map(kinematic_limits),
            affordance_constraints=dict(affordance_constraints or {}),
            safety_invariants=dict(safety_invariants or {}),
            geometry_priors=_coerce_float_map(geometry_priors),
            geometry_hints=dict(geometry_hints or {}),
            semantic_evidence=dict(semantic_evidence or {}),
            uncertainty=_coerce_float_map(uncertainty),
            trust_metadata=_coerce_float_map(trust_metadata),
            source_refs={str(k): str(v) for k, v in dict(source_refs or {}).items()},
            metadata=dict(metadata or {}),
            version=version,
        )

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
        soft_bounds = {
            "semantic_fusion_confidence_mean": {"min": max(0.0, min(1.0, confidence * 0.7))},
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
            soft_bounds=soft_bounds,
            kinematic_limits=kinematic_limits,
            affordance_constraints=affordance_constraints,
            safety_invariants=safety_invariants,
            geometry_priors=geometry_priors,
            geometry_hints={
                "manifold_family": str(semantic_evidence.get("manifold_family", "unknown")),
                "occlusion_level": _safe_float(map_first_summary.get("occlusion_level", 0.0), 0.0),
            },
            semantic_evidence=dict(semantic_evidence),
            uncertainty={
                "semantic_disagreement": disagreement,
                "map_quality_inverse": max(0.0, 1.0 - map_quality),
            },
            trust_metadata={
                "fusion_confidence": confidence,
                "vla_confidence": vla_conf,
            },
            source_refs={
                "semantic_evidence": str(semantic_evidence.get("source", "")),
                "map_first": str(map_first_summary.get("source", "")),
                "fusion": str(fusion_metrics.get("source", "")),
            },
            metadata={"source": "artifact_bridge"},
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
            hard_bounds=_coerce_bounds(decoded.get("hard_bounds", decoded.get("hard_constraints", {}))),
            soft_bounds=_coerce_bounds(decoded.get("soft_bounds", decoded.get("soft_constraints", {}))),
            kinematic_limits=_coerce_float_map(decoded.get("kinematic_limits", {})),
            affordance_constraints=dict(decoded.get("affordance_constraints", {}) or {}),
            safety_invariants=dict(decoded.get("safety_invariants", {}) or {}),
            geometry_priors=_coerce_float_map(decoded.get("geometry_priors", {})),
            geometry_hints=dict(decoded.get("geometry_hints", {}) or {}),
            semantic_evidence=dict(decoded.get("semantic_evidence", {}) or {}),
            uncertainty=_coerce_float_map(decoded.get("uncertainty", {})),
            trust_metadata=_coerce_float_map(decoded.get("trust_metadata", {})),
            source_refs={str(k): str(v) for k, v in dict(decoded.get("source_refs", {}) or {}).items()},
            metadata=dict(decoded.get("metadata", {}) or {}),
            version=str(decoded.get("version", "constraint_set_v1")),
        )


def _coerce_bounds(payload: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    bounds: Dict[str, Dict[str, float]] = {}
    for axis, spec in dict(payload or {}).items():
        if not isinstance(spec, Mapping):
            continue
        axis_bounds: Dict[str, float] = {}
        if "min" in spec:
            axis_bounds["min"] = _safe_float(spec.get("min"), 0.0)
        if "max" in spec:
            axis_bounds["max"] = _safe_float(spec.get("max"), 0.0)
        if axis_bounds:
            bounds[str(axis)] = axis_bounds
    return bounds


def _coerce_float_map(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    return {str(key): _safe_float(value, 0.0) for key, value in dict(payload or {}).items()}


def _flatten_observations(payload: Mapping[str, Any]) -> Dict[str, float | bool]:
    flat: Dict[str, float | bool] = {}
    for key, value in dict(payload).items():
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                if isinstance(child_value, bool):
                    flat[str(child_key)] = child_value
                else:
                    coerced = _safe_float(child_value, None)
                    if coerced is not None:
                        flat[str(child_key)] = coerced
        elif isinstance(value, bool):
            flat[str(key)] = value
        else:
            coerced = _safe_float(value, None)
            if coerced is not None:
                flat[str(key)] = coerced
    return flat


@overload
def _safe_float(value: Any, default: None) -> Optional[float]:
    ...


@overload
def _safe_float(value: Any, default: float = 0.0) -> float:
    ...


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default
