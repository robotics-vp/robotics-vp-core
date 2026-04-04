"""Typed annotation export surface for training-ready data.

This module provides the typed export layer that converts
``PerceptionGroundingWorldState`` (specifically the annotation bridge
output + scene graph) into training-ready records.  These records
become the supervision signal for:

- Graph Transformer seam training (scene-graph structure supervision)
- Annotation bridge neuralization (label quality supervision)
- Downstream training dataset formation

The export surface is the honest mechanism by which perception canonical
state becomes training-usable evidence.  Without it, neural seams train
on synthetic data only; with it, they can train on real annotation
artifacts emitted during rollout labeling.

Named consumers
---------------
- ``PerceptionSeamTrainer``: graph transformer + annotation bridge seams
- ``rollout_labeler.py``: emits annotation export records as sidecar
  artifacts alongside VLA semantic evidence
- Future replay/training pipelines that need structured scene-graph
  supervision
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Annotation export record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationExportRecord:
    """Training-ready record derived from perception canonical state.

    Captures the scene graph structure, annotation bridge labels, and
    provenance metadata in a form directly consumable by seam training
    data loaders.

    Fields are organized into three groups:
    1. **Scene graph structure** — object tokens + edges (input)
    2. **Annotation labels** — class labels, confidence, events (supervision)
    3. **Provenance** — where this record came from
    """

    # Identifiers
    record_id: str
    scene_graph_id: str
    episode_id: str
    frame_index: int

    # Scene graph structure (training input)
    object_track_ids: List[str] = field(default_factory=list)
    object_tokens: List[List[float]] = field(default_factory=list)  # (N, d_token=128)
    object_categories: List[str] = field(default_factory=list)
    object_confidences: List[float] = field(default_factory=list)
    edge_source_ids: List[str] = field(default_factory=list)
    edge_target_ids: List[str] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)
    edge_confidences: List[float] = field(default_factory=list)
    edge_features: List[List[float]] = field(default_factory=list)  # (E, d_edge=64)
    scene_summary_token: List[float] = field(default_factory=list)  # (d_summary=256,)

    # Annotation bridge labels (supervision targets)
    object_class_labels: Dict[str, str] = field(default_factory=dict)
    object_annotation_confidences: Dict[str, float] = field(default_factory=dict)
    object_affordance_hints: Dict[str, List[str]] = field(default_factory=dict)
    object_risk_hints: Dict[str, List[str]] = field(default_factory=dict)
    object_event_labels: Dict[str, List[str]] = field(default_factory=dict)
    primitive_segment_alignment_scores: List[float] = field(default_factory=list)
    failure_interpretation_tags: List[str] = field(default_factory=list)
    recovery_interpretation_tags: List[str] = field(default_factory=list)
    teacher_alignment_score: float = 0.0

    # Quality and provenance
    annotation_quality_score: float = 0.0
    source: str = "perception_compiler"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "annotation_export_record_v1"

    @property
    def n_objects(self) -> int:
        return len(self.object_track_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edge_source_ids)

    @property
    def has_annotation_labels(self) -> bool:
        return bool(self.object_class_labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "scene_graph_id": self.scene_graph_id,
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
            "object_track_ids": list(self.object_track_ids),
            "object_tokens": [[float(v) for v in tok] for tok in self.object_tokens],
            "object_categories": list(self.object_categories),
            "object_confidences": [float(v) for v in self.object_confidences],
            "edge_source_ids": list(self.edge_source_ids),
            "edge_target_ids": list(self.edge_target_ids),
            "edge_types": list(self.edge_types),
            "edge_confidences": [float(v) for v in self.edge_confidences],
            "edge_features": [[float(v) for v in feat] for feat in self.edge_features],
            "scene_summary_token": [float(v) for v in self.scene_summary_token],
            "object_class_labels": dict(self.object_class_labels),
            "object_annotation_confidences": {
                k: float(v) for k, v in self.object_annotation_confidences.items()
            },
            "object_affordance_hints": {
                k: list(v) for k, v in self.object_affordance_hints.items()
            },
            "object_risk_hints": {
                k: list(v) for k, v in self.object_risk_hints.items()
            },
            "object_event_labels": {
                k: list(v) for k, v in self.object_event_labels.items()
            },
            "primitive_segment_alignment_scores": [
                float(v) for v in self.primitive_segment_alignment_scores
            ],
            "failure_interpretation_tags": list(self.failure_interpretation_tags),
            "recovery_interpretation_tags": list(self.recovery_interpretation_tags),
            "teacher_alignment_score": float(self.teacher_alignment_score),
            "annotation_quality_score": float(self.annotation_quality_score),
            "source": self.source,
            "metadata": dict(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------


def export_annotation_record(
    perception_state: Any,
    *,
    source: str = "perception_compiler",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[AnnotationExportRecord]:
    """Convert a PerceptionGroundingWorldState into an AnnotationExportRecord.

    Returns None if the state lacks a scene graph (nothing to export).

    Args:
        perception_state: A ``PerceptionGroundingWorldState`` instance.
        source: Provenance tag for this export.
        metadata: Additional metadata to attach.

    Returns:
        An ``AnnotationExportRecord`` or None.
    """
    scene_graph = getattr(perception_state, "scene_graph", None)
    if scene_graph is None:
        return None

    graph_id = getattr(scene_graph, "graph_id", "")
    episode_id = getattr(perception_state, "episode_id", "")
    frame_index = getattr(perception_state, "frame_index", 0)

    # Extract object tokens and metadata from scene graph
    object_track_ids: List[str] = []
    object_tokens: List[List[float]] = []
    object_categories: List[str] = []
    object_confidences: List[float] = []

    for track in getattr(scene_graph, "object_tracks", []):
        object_track_ids.append(getattr(track, "track_id", ""))
        object_tokens.append(
            [float(v) for v in getattr(track, "feature_token", [])]
        )
        object_categories.append(getattr(track, "object_category", ""))
        object_confidences.append(float(getattr(track, "confidence", 0.0)))

    # Extract edges
    edge_source_ids: List[str] = []
    edge_target_ids: List[str] = []
    edge_types: List[str] = []
    edge_confidences: List[float] = []
    edge_features: List[List[float]] = []

    for edge in getattr(scene_graph, "edges", []):
        edge_source_ids.append(getattr(edge, "source_track_id", ""))
        edge_target_ids.append(getattr(edge, "target_track_id", ""))
        edge_types.append(getattr(edge, "edge_type", ""))
        edge_confidences.append(float(getattr(edge, "confidence", 0.0)))
        edge_features.append(
            [float(v) for v in getattr(edge, "edge_features", [])]
        )

    scene_summary = [
        float(v) for v in getattr(scene_graph, "scene_summary_token", [])
    ]

    # Extract annotation bridge labels if available
    registry = getattr(perception_state, "semantic_bridge_registry", None)
    annotation_bridge = (
        getattr(registry, "annotation_bridge", None) if registry else None
    )

    object_class_labels: Dict[str, str] = {}
    object_annotation_confidences: Dict[str, float] = {}
    object_affordance_hints: Dict[str, List[str]] = {}
    object_risk_hints: Dict[str, List[str]] = {}
    object_event_labels: Dict[str, List[str]] = {}
    primitive_segment_alignment_scores: List[float] = []
    failure_interpretation_tags: List[str] = []
    recovery_interpretation_tags: List[str] = []
    teacher_alignment_score = 0.0

    if annotation_bridge is not None:
        object_class_labels = dict(
            getattr(annotation_bridge, "object_class_labels", {}) or {}
        )
        object_annotation_confidences = {
            k: float(v)
            for k, v in (
                getattr(annotation_bridge, "object_confidence_scores", {}) or {}
            ).items()
        }
        object_affordance_hints = {
            k: list(v)
            for k, v in (
                getattr(annotation_bridge, "object_affordance_hints", {}) or {}
            ).items()
        }
        object_risk_hints = {
            k: list(v)
            for k, v in (
                getattr(annotation_bridge, "object_risk_hints", {}) or {}
            ).items()
        }
        object_event_labels = {
            k: list(v)
            for k, v in (
                getattr(annotation_bridge, "object_event_labels", {}) or {}
            ).items()
        }
        primitive_segment_alignment_scores = [
            float(v)
            for v in (
                getattr(annotation_bridge, "primitive_segment_alignment_scores", [])
                or []
            )
        ]
        failure_interpretation_tags = list(
            getattr(annotation_bridge, "failure_interpretation_tags", []) or []
        )
        recovery_interpretation_tags = list(
            getattr(annotation_bridge, "recovery_interpretation_tags", []) or []
        )
        teacher_alignment_score = float(
            getattr(annotation_bridge, "teacher_alignment_score", 0.0) or 0.0
        )

    # Compute annotation quality score
    n_labeled = len(object_class_labels)
    n_objects = len(object_track_ids)
    label_coverage = n_labeled / max(n_objects, 1)
    conf_mean = (
        sum(object_annotation_confidences.values())
        / max(len(object_annotation_confidences), 1)
        if object_annotation_confidences
        else 0.0
    )
    annotation_quality_score = min(
        1.0,
        0.5 * label_coverage + 0.3 * conf_mean + 0.2 * teacher_alignment_score,
    )

    record_id = f"annot_export_{uuid.uuid4().hex[:12]}"

    return AnnotationExportRecord(
        record_id=record_id,
        scene_graph_id=graph_id,
        episode_id=episode_id,
        frame_index=frame_index,
        object_track_ids=object_track_ids,
        object_tokens=object_tokens,
        object_categories=object_categories,
        object_confidences=object_confidences,
        edge_source_ids=edge_source_ids,
        edge_target_ids=edge_target_ids,
        edge_types=edge_types,
        edge_confidences=edge_confidences,
        edge_features=edge_features,
        scene_summary_token=scene_summary,
        object_class_labels=object_class_labels,
        object_annotation_confidences=object_annotation_confidences,
        object_affordance_hints=object_affordance_hints,
        object_risk_hints=object_risk_hints,
        object_event_labels=object_event_labels,
        primitive_segment_alignment_scores=primitive_segment_alignment_scores,
        failure_interpretation_tags=failure_interpretation_tags,
        recovery_interpretation_tags=recovery_interpretation_tags,
        teacher_alignment_score=teacher_alignment_score,
        annotation_quality_score=annotation_quality_score,
        source=source,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def export_annotation_records_batch(
    perception_states: Sequence[Any],
    *,
    source: str = "perception_compiler",
    min_objects: int = 1,
) -> List[AnnotationExportRecord]:
    """Export annotation records from a sequence of perception states.

    Filters out states without scene graphs or below the minimum object count.

    Args:
        perception_states: Sequence of PerceptionGroundingWorldState instances.
        source: Provenance tag.
        min_objects: Minimum objects required to produce a record.

    Returns:
        List of AnnotationExportRecord instances.
    """
    records: List[AnnotationExportRecord] = []
    for state in perception_states:
        record = export_annotation_record(state, source=source)
        if record is not None and record.n_objects >= min_objects:
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_annotation_export_json(
    path: Path,
    records: Sequence[AnnotationExportRecord],
) -> Path:
    """Save annotation export records to JSON.

    Args:
        path: Output file path.
        records: Records to serialize.

    Returns:
        The path written to.
    """
    payload = {
        "version": "annotation_export_v1",
        "record_count": len(records),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": [r.to_dict() for r in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_annotation_export_json(
    path: Path,
) -> List[AnnotationExportRecord]:
    """Load annotation export records from JSON.

    Args:
        path: Input file path.

    Returns:
        List of AnnotationExportRecord instances.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    records: List[AnnotationExportRecord] = []
    for entry in payload.get("records", []):
        records.append(AnnotationExportRecord(**{
            k: v for k, v in entry.items()
            if k in AnnotationExportRecord.__dataclass_fields__
        }))
    return records


__all__ = [
    "AnnotationExportRecord",
    "export_annotation_record",
    "export_annotation_records_batch",
    "load_annotation_export_json",
    "save_annotation_export_json",
]
