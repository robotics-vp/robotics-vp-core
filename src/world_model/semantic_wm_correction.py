"""Additive semantic world-model correction overlay.

The semantic WM builder remains deterministic. Runtime validation packets
flow into this module, which compiles a bounded correction overlay and
applies it to a copy of the world model for downstream routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.world_model.semantic_feedback_packets import WMValidationPacket
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class SemanticWMCorrectionOverlay:
    object_confidence_adjustments: Dict[str, float] = field(default_factory=dict)
    relation_confidence_adjustments: Dict[str, float] = field(default_factory=dict)
    capability_adjustments: Dict[str, float] = field(default_factory=dict)
    topology_adjustments: Dict[str, Any] = field(default_factory=dict)
    meta_node_pressure: float = 0.0
    target_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_confidence_adjustments": dict(self.object_confidence_adjustments),
            "relation_confidence_adjustments": dict(
                self.relation_confidence_adjustments
            ),
            "capability_adjustments": dict(self.capability_adjustments),
            "topology_adjustments": dict(self.topology_adjustments),
            "meta_node_pressure": float(self.meta_node_pressure),
            "target_refs": list(self.target_refs),
            "metadata": dict(self.metadata),
        }


def _coerce_packets(values: Optional[Iterable[Any]]) -> List[WMValidationPacket]:
    packets: List[WMValidationPacket] = []
    for item in values or []:
        if isinstance(item, WMValidationPacket):
            packets.append(item)
        elif isinstance(item, Mapping):
            packets.append(WMValidationPacket.from_dict(item))
    return packets


def compile_semantic_wm_correction_overlay(
    semantic_world_model: Optional[SemanticWorldModelState],
    wm_validation_packets: Optional[Iterable[Any]],
) -> SemanticWMCorrectionOverlay:
    if semantic_world_model is None:
        return SemanticWMCorrectionOverlay()
    if isinstance(semantic_world_model, Mapping):
        try:
            semantic_world_model = SemanticWorldModelState.from_dict(
                semantic_world_model
            )
        except Exception:
            return SemanticWMCorrectionOverlay()
    packets = _coerce_packets(wm_validation_packets)
    if not packets:
        return SemanticWMCorrectionOverlay()

    object_ids = {item.object_id for item in semantic_world_model.objects}
    relation_ids = {item.relation_id for item in semantic_world_model.relations}
    object_adjustments: Dict[str, float] = {}
    relation_adjustments: Dict[str, float] = {}
    max_error = 0.0
    target_refs: List[str] = []
    for packet in packets:
        error = _clip01(_safe_float(packet.error_score, 0.0))
        max_error = max(max_error, error)
        target_refs.append(packet.target_ref)
        if packet.target_ref in object_ids:
            object_adjustments[packet.target_ref] = (
                object_adjustments.get(packet.target_ref, 0.0) - 0.35 * error
            )
        if packet.target_ref in relation_ids:
            relation_adjustments[packet.target_ref] = (
                relation_adjustments.get(packet.target_ref, 0.0) - 0.4 * error
            )
        relation_ref = str(packet.metadata.get("relation_id", ""))
        if relation_ref in relation_ids:
            relation_adjustments[relation_ref] = (
                relation_adjustments.get(relation_ref, 0.0) - 0.4 * error
            )

    error_mean = sum(
        _safe_float(packet.error_score, 0.0) for packet in packets
    ) / float(len(packets))
    capability_adjustments = {
        "object_memory": -0.25 * error_mean,
        "affordance_grounding": -0.2 * error_mean,
        "meta_node_orchestration": 0.1 * error_mean,
        "risk_reasoning": 0.15 * error_mean,
    }
    topology_adjustments = {
        "wm_validation_error_rate": float(error_mean),
        "wm_validation_packet_count": int(len(packets)),
    }
    return SemanticWMCorrectionOverlay(
        object_confidence_adjustments=object_adjustments,
        relation_confidence_adjustments=relation_adjustments,
        capability_adjustments=capability_adjustments,
        topology_adjustments=topology_adjustments,
        meta_node_pressure=float(max_error),
        target_refs=sorted({ref for ref in target_refs if ref}),
        metadata={
            "packet_count": len(packets),
            "error_mean": float(error_mean),
            "high_pressure": bool(max_error >= 0.5),
        },
    )


def apply_semantic_wm_correction_overlay(
    semantic_world_model: Optional[SemanticWorldModelState],
    overlay: Optional[SemanticWMCorrectionOverlay],
) -> Optional[SemanticWorldModelState]:
    if semantic_world_model is None or overlay is None:
        return semantic_world_model
    if isinstance(semantic_world_model, Mapping):
        try:
            semantic_world_model = SemanticWorldModelState.from_dict(
                semantic_world_model
            )
        except Exception:
            return semantic_world_model
    if not overlay.metadata:
        return semantic_world_model

    corrected_objects: List[SemanticObjectState] = []
    for item in semantic_world_model.objects:
        delta = overlay.object_confidence_adjustments.get(item.object_id, 0.0)
        if delta == 0.0 and item.label in overlay.object_confidence_adjustments:
            delta = overlay.object_confidence_adjustments[item.label]
        corrected_objects.append(
            replace(item, confidence=_clip01(float(item.confidence) + float(delta)))
        )

    corrected_relations: List[SemanticRelationState] = []
    for relation in semantic_world_model.relations:
        delta = overlay.relation_confidence_adjustments.get(relation.relation_id, 0.0)
        corrected_relations.append(
            replace(
                relation, confidence=_clip01(float(relation.confidence) + float(delta))
            )
        )

    capability_scores = dict(semantic_world_model.capability_scores or {})
    for key, delta in overlay.capability_adjustments.items():
        capability_scores[key] = _clip01(
            _safe_float(capability_scores.get(key, 0.0), 0.0) + _safe_float(delta, 0.0)
        )

    topology = dict(semantic_world_model.topology or {})
    topology.update(dict(overlay.topology_adjustments or {}))

    meta_nodes = list(semantic_world_model.meta_nodes or [])
    if overlay.meta_node_pressure > 0.0:
        meta_nodes.append(
            SemanticMetaNode(
                node_id="meta:state_validation_router",
                node_type="state_validation_router",
                priority="high" if overlay.meta_node_pressure >= 0.5 else "medium",
                score=_clip01(0.3 + overlay.meta_node_pressure),
                rationale="Runtime WM validation packets requested semantic state correction",
                target_refs=list(overlay.target_refs[:8]),
                suggested_actions=[
                    "request_wm_state_validation",
                    "refresh_semantic_memory",
                ],
                metadata={"overlay": overlay.to_dict()},
            )
        )

    semantic_tags = list(
        dict.fromkeys(
            list(semantic_world_model.semantic_tags or []) + ["feedback:wm_correction"]
        )
    )
    metadata = dict(semantic_world_model.metadata or {})
    metadata["semantic_wm_correction_overlay"] = overlay.to_dict()
    return replace(
        semantic_world_model,
        objects=corrected_objects,
        relations=corrected_relations,
        meta_nodes=meta_nodes,
        capability_scores=capability_scores,
        topology=topology,
        semantic_tags=semantic_tags,
        metadata=metadata,
    )


__all__ = [
    "SemanticWMCorrectionOverlay",
    "apply_semantic_wm_correction_overlay",
    "compile_semantic_wm_correction_overlay",
]
