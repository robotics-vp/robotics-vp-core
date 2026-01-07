"""Artifacts and serialization for Embodiment outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

EMBODIMENT_PROFILE_VERSION = "v1"
EMBODIMENT_PROFILE_PREFIX = "embodiment_profile_v1/"

AFFORDANCE_GRAPH_VERSION = "v1"
AFFORDANCE_GRAPH_PREFIX = "affordance_graph_v1/"

SKILL_SEGMENTS_VERSION = "v1"
SKILL_SEGMENTS_PREFIX = "skill_segments_v1/"


@dataclass
class EmbodimentSummary:
    """Compact summary for embodiment outputs."""

    w_embodiment: float = 0.0
    embodiment_quality_score: float = 0.0
    contact_coverage_pct: float = 0.0
    semantic_confidence_mean: float = 0.0
    physically_impossible_contacts: int = 0
    drift_score: float = 0.0
    trust_override_candidate: bool = False
    missing_inputs: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_embodiment": float(self.w_embodiment),
            "embodiment_quality_score": float(self.embodiment_quality_score),
            "contact_coverage_pct": float(self.contact_coverage_pct),
            "semantic_confidence_mean": float(self.semantic_confidence_mean),
            "physically_impossible_contacts": int(self.physically_impossible_contacts),
            "drift_score": float(self.drift_score),
            "trust_override_candidate": bool(self.trust_override_candidate),
            "missing_inputs": list(self.missing_inputs),
            "diagnostics": self.diagnostics,
        }


@dataclass
class EmbodimentProfileArtifact:
    """Core embodiment profile in world frame."""

    contact_matrix: np.ndarray
    contact_confidence: np.ndarray
    contact_impossible: np.ndarray
    track_ids: np.ndarray
    track_class_ids: np.ndarray
    track_labels: Optional[np.ndarray] = None
    contact_distance: Optional[np.ndarray] = None
    visibility: Optional[np.ndarray] = None
    occlusion: Optional[np.ndarray] = None
    contact_counts: Optional[np.ndarray] = None

    def to_npz(
        self,
        summary: Optional[EmbodimentSummary] = None,
        export_float16: bool = True,
    ) -> Dict[str, np.ndarray]:
        conf_dtype = np.float16 if export_float16 else np.float32
        data: Dict[str, np.ndarray] = {
            f"{EMBODIMENT_PROFILE_PREFIX}version": np.array([EMBODIMENT_PROFILE_VERSION], dtype="U8"),
            f"{EMBODIMENT_PROFILE_PREFIX}contact_matrix": self.contact_matrix.astype(bool),
            f"{EMBODIMENT_PROFILE_PREFIX}contact_confidence": self.contact_confidence.astype(conf_dtype),
            f"{EMBODIMENT_PROFILE_PREFIX}contact_impossible": self.contact_impossible.astype(bool),
            f"{EMBODIMENT_PROFILE_PREFIX}track_ids": self.track_ids.astype("U32"),
            f"{EMBODIMENT_PROFILE_PREFIX}track_class_ids": self.track_class_ids.astype(np.int32),
        }
        if self.track_labels is not None:
            data[f"{EMBODIMENT_PROFILE_PREFIX}track_labels"] = self.track_labels.astype("U64")
        if self.contact_distance is not None:
            data[f"{EMBODIMENT_PROFILE_PREFIX}contact_distance"] = self.contact_distance.astype(conf_dtype)
        if self.visibility is not None:
            data[f"{EMBODIMENT_PROFILE_PREFIX}visibility"] = self.visibility.astype(np.float32)
        if self.occlusion is not None:
            data[f"{EMBODIMENT_PROFILE_PREFIX}occlusion"] = self.occlusion.astype(np.float32)
        if self.contact_counts is not None:
            data[f"{EMBODIMENT_PROFILE_PREFIX}contact_counts"] = self.contact_counts.astype(np.float32)
        if summary is not None:
            summary_json = json.dumps(summary.to_dict())
            data[f"{EMBODIMENT_PROFILE_PREFIX}summary_json"] = np.array([summary_json], dtype="U4096")
        validate_no_object_arrays(data)
        return data


@dataclass
class AffordanceGraphArtifact:
    """Affordance graph edges derived from contacts."""

    node_ids: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_confidence: np.ndarray
    edge_support: np.ndarray
    node_class_ids: Optional[np.ndarray] = None
    node_labels: Optional[np.ndarray] = None

    def to_npz(self, export_float16: bool = True) -> Dict[str, np.ndarray]:
        conf_dtype = np.float16 if export_float16 else np.float32
        data: Dict[str, np.ndarray] = {
            f"{AFFORDANCE_GRAPH_PREFIX}version": np.array([AFFORDANCE_GRAPH_VERSION], dtype="U8"),
            f"{AFFORDANCE_GRAPH_PREFIX}node_ids": self.node_ids.astype("U32"),
            f"{AFFORDANCE_GRAPH_PREFIX}edge_index": self.edge_index.astype(np.int32),
            f"{AFFORDANCE_GRAPH_PREFIX}edge_type": self.edge_type.astype(np.int32),
            f"{AFFORDANCE_GRAPH_PREFIX}edge_confidence": self.edge_confidence.astype(conf_dtype),
            f"{AFFORDANCE_GRAPH_PREFIX}edge_support": self.edge_support.astype(conf_dtype),
        }
        if self.node_class_ids is not None:
            data[f"{AFFORDANCE_GRAPH_PREFIX}node_class_ids"] = self.node_class_ids.astype(np.int32)
        if self.node_labels is not None:
            data[f"{AFFORDANCE_GRAPH_PREFIX}node_labels"] = self.node_labels.astype("U64")
        validate_no_object_arrays(data)
        return data


@dataclass
class SkillSegmentsArtifact:
    """Segmentation of interaction primitives."""

    segment_bounds: np.ndarray
    segment_type: np.ndarray
    segment_confidence: np.ndarray
    segment_contact_pairs: np.ndarray
    segment_energy_Wh: np.ndarray
    segment_risk: np.ndarray
    segment_success: np.ndarray
    segment_labels: Optional[np.ndarray] = None

    def to_npz(self, export_float16: bool = True) -> Dict[str, np.ndarray]:
        conf_dtype = np.float16 if export_float16 else np.float32
        data: Dict[str, np.ndarray] = {
            f"{SKILL_SEGMENTS_PREFIX}version": np.array([SKILL_SEGMENTS_VERSION], dtype="U8"),
            f"{SKILL_SEGMENTS_PREFIX}segment_bounds": self.segment_bounds.astype(np.int32),
            f"{SKILL_SEGMENTS_PREFIX}segment_type": self.segment_type.astype(np.int32),
            f"{SKILL_SEGMENTS_PREFIX}segment_confidence": self.segment_confidence.astype(conf_dtype),
            f"{SKILL_SEGMENTS_PREFIX}segment_contact_pairs": self.segment_contact_pairs.astype(np.int32),
            f"{SKILL_SEGMENTS_PREFIX}segment_energy_Wh": self.segment_energy_Wh.astype(np.float32),
            f"{SKILL_SEGMENTS_PREFIX}segment_risk": self.segment_risk.astype(conf_dtype),
            f"{SKILL_SEGMENTS_PREFIX}segment_success": self.segment_success.astype(conf_dtype),
        }
        if self.segment_labels is not None:
            data[f"{SKILL_SEGMENTS_PREFIX}segment_labels"] = self.segment_labels.astype("U64")
        validate_no_object_arrays(data)
        return data


def validate_no_object_arrays(data: Dict[str, np.ndarray]) -> None:
    """Validate that no arrays have object dtype."""
    for key, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            raise ValueError(
                f"Array '{key}' has object dtype. "
                "Only numeric, bool, and unicode string dtypes allowed."
            )
