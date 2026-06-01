"""Learned successor/refiner over the deterministic semantic world model.

This module does not replace ``SemanticWorldModelBuilder``. It learns bounded
post-build deltas over:
  - semantic WM confidence/capability correction
  - graph-mutation proposal scoring

The learned outputs are always routed through the existing governed packet path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import json
import numpy as np

from src.world_model.semantic_feedback_packets import (
    GraphMutationProposal,
    WMValidationPacket,
)
from src.world_model.semantic_state_encoder import (
    encode_object,
    encode_relation,
    encode_wm_state_flat,
)
from src.world_model.semantic_wm_correction import SemanticWMCorrectionOverlay
from src.world_model.semantic_world_model import SemanticWorldModelState


CAPABILITY_KEYS = [
    "object_memory",
    "affordance_grounding",
    "meta_node_orchestration",
    "risk_reasoning",
]
PROPOSAL_ACTIONS = [
    "add_provisional_skill",
    "add_provisional_affordance",
    "add_object_family",
    "update_relationship",
    "mark_for_review",
]
PROPOSAL_TARGET_KINDS = [
    "skill",
    "affordance",
    "object",
    "relation",
    "review",
    "unknown",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _coerce_world_model(value: Any) -> Optional[SemanticWorldModelState]:
    if isinstance(value, SemanticWorldModelState):
        return value
    if isinstance(value, Mapping):
        try:
            return SemanticWorldModelState.from_dict(value)
        except Exception:
            return None
    return None


def _coerce_overlay(value: Any) -> SemanticWMCorrectionOverlay:
    if isinstance(value, SemanticWMCorrectionOverlay):
        return value
    if isinstance(value, Mapping):
        return SemanticWMCorrectionOverlay(
            object_confidence_adjustments={
                str(key): _safe_float(item, 0.0)
                for key, item in dict(
                    value.get("object_confidence_adjustments", {}) or {}
                ).items()
            },
            relation_confidence_adjustments={
                str(key): _safe_float(item, 0.0)
                for key, item in dict(
                    value.get("relation_confidence_adjustments", {}) or {}
                ).items()
            },
            capability_adjustments={
                str(key): _safe_float(item, 0.0)
                for key, item in dict(
                    value.get("capability_adjustments", {}) or {}
                ).items()
            },
            topology_adjustments=dict(value.get("topology_adjustments", {}) or {}),
            meta_node_pressure=_safe_float(value.get("meta_node_pressure", 0.0), 0.0),
            target_refs=[str(ref) for ref in value.get("target_refs", []) or []],
            metadata=dict(value.get("metadata", {}) or {}),
        )
    return SemanticWMCorrectionOverlay()


def _coerce_validation_packets(
    values: Optional[Iterable[Any]],
) -> List[WMValidationPacket]:
    packets: List[WMValidationPacket] = []
    for item in values or []:
        if isinstance(item, WMValidationPacket):
            packets.append(item)
        elif isinstance(item, Mapping):
            packets.append(WMValidationPacket.from_dict(item))
    return packets


def _coerce_proposals(values: Optional[Iterable[Any]]) -> List[GraphMutationProposal]:
    proposals: List[GraphMutationProposal] = []
    for item in values or []:
        if isinstance(item, GraphMutationProposal):
            proposals.append(item)
        elif isinstance(item, Mapping):
            proposals.append(
                GraphMutationProposal(
                    proposal_id=str(item.get("proposal_id", "")),
                    action=str(item.get("action", "")),
                    target_ref=str(item.get("target_ref", "")),
                    confidence=_safe_float(item.get("confidence", 0.0), 0.0),
                    rationale=str(item.get("rationale", "")),
                    source_refs=[str(ref) for ref in item.get("source_refs", []) or []],
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
    return proposals


def _validation_error_for_ref(
    packets: Sequence[WMValidationPacket],
    target_ref: str,
) -> float:
    matches = [
        _safe_float(item.error_score, 0.0)
        for item in packets
        if item.target_ref == target_ref
        or str(item.metadata.get("relation_id", "")) == target_ref
    ]
    return float(max(matches) if matches else 0.0)


def _wm_global_features(
    semantic_world_model: SemanticWorldModelState,
    *,
    correction_overlay: Optional[SemanticWMCorrectionOverlay] = None,
    feedback_summary: Optional[Mapping[str, Any]] = None,
    wm_validation_packets: Optional[Sequence[WMValidationPacket]] = None,
    mutation_proposals: Optional[Sequence[GraphMutationProposal]] = None,
) -> np.ndarray:
    overlay = correction_overlay or SemanticWMCorrectionOverlay()
    feedback = dict(feedback_summary or {})
    packets = list(wm_validation_packets or [])
    proposals = list(mutation_proposals or [])
    base = encode_wm_state_flat(semantic_world_model).astype(np.float32)
    extra = np.array(
        [
            _safe_float(overlay.meta_node_pressure, 0.0),
            float(
                np.mean(
                    [
                        abs(_safe_float(v, 0.0))
                        for v in overlay.object_confidence_adjustments.values()
                    ]
                )
                if overlay.object_confidence_adjustments
                else 0.0
            ),
            float(
                np.mean(
                    [
                        abs(_safe_float(v, 0.0))
                        for v in overlay.relation_confidence_adjustments.values()
                    ]
                )
                if overlay.relation_confidence_adjustments
                else 0.0
            ),
            min(float(len(packets)) / 8.0, 1.0),
            _safe_float(feedback.get("gap_return_mean", 0.0), 0.0),
            _safe_float(feedback.get("process_reward_mean", 0.0), 0.0),
            min(float(len(proposals)) / 8.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra])


def _one_hot(value: str, vocabulary: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(vocabulary), dtype=np.float32)
    try:
        vec[list(vocabulary).index(value)] = 1.0
    except ValueError:
        pass
    return vec


def _proposal_target_kind(target_ref: str, action: str) -> str:
    ref = str(target_ref or "")
    if action == "add_provisional_affordance" or "affordance" in ref:
        return "affordance"
    if action == "add_provisional_skill" or "skill" in ref:
        return "skill"
    if action == "add_object_family" or ref.startswith(("obj:", "object:")):
        return "object"
    if action == "update_relationship" or "rel" in ref or "relation" in ref:
        return "relation"
    if action == "mark_for_review":
        return "review"
    return "unknown"


def _infer_candidate_action(target_ref: str, validation_kind: str) -> str:
    ref = str(target_ref or "")
    kind = str(validation_kind or "")
    if "skill" in ref or "skill" in kind:
        return "add_provisional_skill"
    if "affordance" in ref or "affordance" in kind:
        return "add_provisional_affordance"
    if ref.startswith(("obj:", "object:")) or "object" in kind:
        return "add_object_family"
    if "relation" in kind:
        return "update_relationship"
    return "mark_for_review"


@dataclass(frozen=True)
class SemanticWMRefinementDataset:
    global_feature_dim: int
    object_features: List[List[float]]
    object_targets: List[float]
    relation_features: List[List[float]]
    relation_targets: List[float]
    capability_features: List[List[float]]
    capability_targets: List[List[float]]
    proposal_features: List[List[float]]
    proposal_accept_targets: List[float]
    proposal_confidence_targets: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_feature_dim": self.global_feature_dim,
            "object_features": list(self.object_features),
            "object_targets": list(self.object_targets),
            "relation_features": list(self.relation_features),
            "relation_targets": list(self.relation_targets),
            "capability_features": list(self.capability_features),
            "capability_targets": list(self.capability_targets),
            "proposal_features": list(self.proposal_features),
            "proposal_accept_targets": list(self.proposal_accept_targets),
            "proposal_confidence_targets": list(self.proposal_confidence_targets),
            "metadata": dict(self.metadata),
        }


def generate_graph_mutation_candidates(
    semantic_world_model: Any,
    *,
    base_proposals: Optional[Iterable[Any]] = None,
    wm_validation_packets: Optional[Iterable[Any]] = None,
    max_candidates: int = 16,
) -> List[GraphMutationProposal]:
    world_model = _coerce_world_model(semantic_world_model)
    if world_model is None:
        return _coerce_proposals(base_proposals)[: max(max_candidates, 0)]
    packets = _coerce_validation_packets(wm_validation_packets)
    proposals = list(_coerce_proposals(base_proposals))
    seen = {(item.action, item.target_ref) for item in proposals}
    next_index = len(proposals)
    for packet in packets:
        if _safe_float(packet.error_score, 0.0) < 0.3:
            continue
        target_ref = str(packet.metadata.get("novel_ref", packet.target_ref)).strip()
        if not target_ref:
            continue
        action = _infer_candidate_action(target_ref, packet.validation_kind)
        key = (action, target_ref)
        if key in seen:
            continue
        seen.add(key)
        next_index += 1
        proposals.append(
            GraphMutationProposal(
                proposal_id=f"learned_candidate_{next_index:03d}",
                action=action,
                target_ref=target_ref,
                confidence=_clip01(0.35 + 0.45 * _safe_float(packet.error_score, 0.0)),
                rationale=f"WM validation suggests unresolved semantic target {target_ref}",
                source_refs=[packet.target_ref],
                metadata={
                    "validation_kind": packet.validation_kind,
                    "candidate_source": "wm_validation",
                },
            )
        )
    for meta_node in list(world_model.meta_nodes or []):
        if meta_node.node_type not in {
            "ontology_router",
            "state_validation_router",
            "semantic_memory_refresh",
        }:
            continue
        for target_ref in list(meta_node.target_refs or [])[:2]:
            action = _infer_candidate_action(target_ref, meta_node.node_type)
            key = (action, target_ref)
            if key in seen:
                continue
            seen.add(key)
            next_index += 1
            proposals.append(
                GraphMutationProposal(
                    proposal_id=f"learned_candidate_{next_index:03d}",
                    action=action,
                    target_ref=target_ref,
                    confidence=_clip01(0.25 + 0.5 * _safe_float(meta_node.score, 0.0)),
                    rationale=f"Meta-node {meta_node.node_type} surfaced a semantic expansion target",
                    source_refs=[meta_node.node_id],
                    metadata={"candidate_source": "meta_node"},
                )
            )
    return proposals[: max(max_candidates, 0)]


def build_semantic_wm_refinement_dataset_from_examples(
    examples: Sequence[Mapping[str, Any]],
) -> SemanticWMRefinementDataset:
    object_features: List[List[float]] = []
    object_targets: List[float] = []
    relation_features: List[List[float]] = []
    relation_targets: List[float] = []
    capability_features: List[List[float]] = []
    capability_targets: List[List[float]] = []
    proposal_features: List[List[float]] = []
    proposal_accept_targets: List[float] = []
    proposal_confidence_targets: List[float] = []
    global_dim = 0

    for item in examples:
        world_model = _coerce_world_model(item.get("semantic_world_model"))
        if world_model is None:
            continue
        overlay = _coerce_overlay(item.get("correction_overlay"))
        feedback_summary = dict(item.get("feedback_summary", {}) or {})
        packets = _coerce_validation_packets(item.get("wm_validation_packets"))
        proposals = generate_graph_mutation_candidates(
            world_model,
            base_proposals=item.get("graph_mutation_proposals"),
            wm_validation_packets=packets,
        )
        execution_payload = dict(item.get("graph_mutation_execution", {}) or {})
        execution_records = {
            str(record.get("proposal_id", "")): str(record.get("status", "deferred"))
            for record in execution_payload.get("records", []) or []
            if isinstance(record, Mapping)
        }
        global_features = _wm_global_features(
            world_model,
            correction_overlay=overlay,
            feedback_summary=feedback_summary,
            wm_validation_packets=packets,
            mutation_proposals=proposals,
        )
        global_dim = max(global_dim, int(global_features.size))

        for obj in list(world_model.objects or []):
            target = overlay.object_confidence_adjustments.get(
                obj.object_id,
                overlay.object_confidence_adjustments.get(obj.label, 0.0),
            )
            vector = np.concatenate(
                [
                    global_features,
                    encode_object(obj),
                    np.array(
                        [
                            _validation_error_for_ref(packets, obj.object_id),
                            1.0
                            if obj.object_id in set(overlay.target_refs or [])
                            else 0.0,
                            min(float(len(obj.track_refs or [])) / 4.0, 1.0),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            object_features.append(vector.tolist())
            object_targets.append(_clip(_safe_float(target, 0.0), -0.5, 0.5))

        for rel in list(world_model.relations or []):
            target = overlay.relation_confidence_adjustments.get(rel.relation_id, 0.0)
            vector = np.concatenate(
                [
                    global_features,
                    encode_relation(rel),
                    np.array(
                        [
                            _validation_error_for_ref(packets, rel.relation_id),
                            1.0
                            if rel.relation_id in set(overlay.target_refs or [])
                            else 0.0,
                            1.0 if rel.object_id == rel.subject_id else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            relation_features.append(vector.tolist())
            relation_targets.append(_clip(_safe_float(target, 0.0), -0.5, 0.5))

        capability_features.append(global_features.tolist())
        capability_targets.append(
            [
                _clip(
                    _safe_float(overlay.capability_adjustments.get(key, 0.0), 0.0),
                    -0.5,
                    0.5,
                )
                for key in CAPABILITY_KEYS
            ]
            + [_clip01(_safe_float(overlay.meta_node_pressure, 0.0))]
        )

        target_refs = {str(ref) for ref in overlay.target_refs or []}
        for proposal in proposals:
            action = (
                proposal.action
                if proposal.action in PROPOSAL_ACTIONS
                else "mark_for_review"
            )
            target_kind = _proposal_target_kind(proposal.target_ref, action)
            target_matches = 1.0 if proposal.target_ref in target_refs else 0.0
            feature_vec = np.concatenate(
                [
                    global_features,
                    _one_hot(action, PROPOSAL_ACTIONS),
                    _one_hot(target_kind, PROPOSAL_TARGET_KINDS),
                    np.array(
                        [
                            _clip01(_safe_float(proposal.confidence, 0.0)),
                            min(float(len(proposal.source_refs or [])) / 4.0, 1.0),
                            target_matches,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            status = execution_records.get(proposal.proposal_id, "")
            accept_target = (
                1.0
                if status == "applied"
                else (1.0 if proposal.confidence >= 0.55 else 0.0)
            )
            confidence_target = (
                _clip01(_safe_float(proposal.confidence, 0.0))
                if accept_target > 0.0
                else _clip01(0.25 * _safe_float(proposal.confidence, 0.0))
            )
            proposal_features.append(feature_vec.tolist())
            proposal_accept_targets.append(float(accept_target))
            proposal_confidence_targets.append(float(confidence_target))

    return SemanticWMRefinementDataset(
        global_feature_dim=global_dim,
        object_features=object_features,
        object_targets=object_targets,
        relation_features=relation_features,
        relation_targets=relation_targets,
        capability_features=capability_features,
        capability_targets=capability_targets,
        proposal_features=proposal_features,
        proposal_accept_targets=proposal_accept_targets,
        proposal_confidence_targets=proposal_confidence_targets,
        metadata={"example_count": len(examples)},
    )


def build_semantic_wm_refinement_dataset_from_artifact_dirs(
    artifact_dirs: Sequence[str],
) -> SemanticWMRefinementDataset:
    examples: List[Dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        root = Path(artifact_dir)
        world_model_payload: Dict[str, Any] = {}
        for name in (
            "input_semantic_world_model.json",
            "corrected_semantic_world_model.json",
        ):
            path = root / name
            if path.exists():
                world_model_payload = json.loads(path.read_text(encoding="utf-8"))
                break
        if not world_model_payload:
            continue
        overlay_payload: Dict[str, Any] = {}
        overlay_path = root / "semantic_wm_correction_overlay.json"
        if overlay_path.exists():
            overlay_payload = json.loads(overlay_path.read_text(encoding="utf-8"))
        feedback_path = root / "coverage_feedback_summary.json"
        proposals_path = root / "graph_mutation_proposals.json"
        execution_path = root / "graph_mutation_execution.json"
        examples.append(
            {
                "semantic_world_model": world_model_payload,
                "correction_overlay": overlay_payload,
                "feedback_summary": json.loads(
                    feedback_path.read_text(encoding="utf-8")
                )
                if feedback_path.exists()
                else {},
                "graph_mutation_proposals": json.loads(
                    proposals_path.read_text(encoding="utf-8")
                )
                if proposals_path.exists()
                else [],
                "graph_mutation_execution": json.loads(
                    execution_path.read_text(encoding="utf-8")
                )
                if execution_path.exists()
                else {},
            }
        )
    return build_semantic_wm_refinement_dataset_from_examples(examples)


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True

    class _ScalarNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs).squeeze(-1)

    class _VectorNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

    @dataclass
    class SemanticWMRefinerPackage:
        object_model: _ScalarNet
        relation_model: _ScalarNet
        capability_model: _VectorNet
        proposal_accept_model: _ScalarNet
        proposal_confidence_model: _ScalarNet
        metadata: Dict[str, Any] = field(default_factory=dict)

        def to_checkpoint(self) -> Dict[str, Any]:
            return {
                "metadata": dict(self.metadata),
                "object_state_dict": self.object_model.state_dict(),
                "relation_state_dict": self.relation_model.state_dict(),
                "capability_state_dict": self.capability_model.state_dict(),
                "proposal_accept_state_dict": self.proposal_accept_model.state_dict(),
                "proposal_confidence_state_dict": self.proposal_confidence_model.state_dict(),
            }

        @classmethod
        def from_checkpoint(
            cls, payload: Mapping[str, Any]
        ) -> "SemanticWMRefinerPackage":
            metadata = dict(payload.get("metadata", {}) or {})
            object_model = _ScalarNet(
                int(metadata.get("object_dim", 1)), int(metadata.get("hidden_dim", 48))
            )
            relation_model = _ScalarNet(
                int(metadata.get("relation_dim", 1)),
                int(metadata.get("hidden_dim", 48)),
            )
            capability_model = _VectorNet(
                int(metadata.get("capability_dim", 1)),
                int(metadata.get("hidden_dim", 48)),
                len(CAPABILITY_KEYS) + 1,
            )
            proposal_accept_model = _ScalarNet(
                int(metadata.get("proposal_dim", 1)),
                int(metadata.get("hidden_dim", 48)),
            )
            proposal_confidence_model = _ScalarNet(
                int(metadata.get("proposal_dim", 1)),
                int(metadata.get("hidden_dim", 48)),
            )
            object_model.load_state_dict(payload["object_state_dict"])
            relation_model.load_state_dict(payload["relation_state_dict"])
            capability_model.load_state_dict(payload["capability_state_dict"])
            proposal_accept_model.load_state_dict(payload["proposal_accept_state_dict"])
            proposal_confidence_model.load_state_dict(
                payload["proposal_confidence_state_dict"]
            )
            object_model.eval()
            relation_model.eval()
            capability_model.eval()
            proposal_accept_model.eval()
            proposal_confidence_model.eval()
            return cls(
                object_model=object_model,
                relation_model=relation_model,
                capability_model=capability_model,
                proposal_accept_model=proposal_accept_model,
                proposal_confidence_model=proposal_confidence_model,
                metadata=metadata,
            )

        def predict_correction_overlay(
            self,
            semantic_world_model: Any,
            *,
            wm_validation_packets: Optional[Iterable[Any]] = None,
            feedback_summary: Optional[Mapping[str, Any]] = None,
        ) -> SemanticWMCorrectionOverlay:
            world_model = _coerce_world_model(semantic_world_model)
            if world_model is None:
                return SemanticWMCorrectionOverlay()
            packets = _coerce_validation_packets(wm_validation_packets)
            global_features = _wm_global_features(
                world_model,
                feedback_summary=feedback_summary,
                wm_validation_packets=packets,
            )
            object_adjustments: Dict[str, float] = {}
            relation_adjustments: Dict[str, float] = {}
            target_refs: List[str] = []

            with torch.no_grad():
                for obj in list(world_model.objects or []):
                    vector = np.concatenate(
                        [
                            global_features,
                            encode_object(obj),
                            np.array(
                                [
                                    _validation_error_for_ref(packets, obj.object_id),
                                    1.0
                                    if _validation_error_for_ref(packets, obj.object_id)
                                    > 0.0
                                    else 0.0,
                                    min(float(len(obj.track_refs or [])) / 4.0, 1.0),
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    )
                    tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
                    delta = _clip(float(self.object_model(tensor).item()), -0.5, 0.5)
                    if abs(delta) >= 0.02:
                        object_adjustments[obj.object_id] = delta
                        target_refs.append(obj.object_id)
                for rel in list(world_model.relations or []):
                    vector = np.concatenate(
                        [
                            global_features,
                            encode_relation(rel),
                            np.array(
                                [
                                    _validation_error_for_ref(packets, rel.relation_id),
                                    1.0
                                    if _validation_error_for_ref(
                                        packets, rel.relation_id
                                    )
                                    > 0.0
                                    else 0.0,
                                    1.0 if rel.object_id == rel.subject_id else 0.0,
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    )
                    tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
                    delta = _clip(float(self.relation_model(tensor).item()), -0.5, 0.5)
                    if abs(delta) >= 0.02:
                        relation_adjustments[rel.relation_id] = delta
                        target_refs.append(rel.relation_id)
                capability_vec = torch.tensor(
                    global_features, dtype=torch.float32
                ).unsqueeze(0)
                capability_outputs = (
                    self.capability_model(capability_vec).squeeze(0).tolist()
                )
            capability_adjustments = {
                key: _clip(_safe_float(value, 0.0), -0.5, 0.5)
                for key, value in zip(
                    CAPABILITY_KEYS, capability_outputs[: len(CAPABILITY_KEYS)]
                )
            }
            pressure = _clip01(
                _safe_float(capability_outputs[len(CAPABILITY_KEYS)], 0.0)
            )
            pressure = max(
                pressure,
                max([abs(value) for value in object_adjustments.values()] + [0.0]),
                max([abs(value) for value in relation_adjustments.values()] + [0.0]),
            )
            return SemanticWMCorrectionOverlay(
                object_confidence_adjustments=object_adjustments,
                relation_confidence_adjustments=relation_adjustments,
                capability_adjustments=capability_adjustments,
                topology_adjustments={
                    "learned_refiner_active": True,
                    "learned_refiner_target_count": len(set(target_refs)),
                },
                meta_node_pressure=float(pressure),
                target_refs=sorted(set(target_refs)),
                metadata={
                    "source": "semantic_wm_refiner",
                    "model_type": "torch",
                    "predicted_target_count": len(set(target_refs)),
                },
            )

        def score_graph_mutation_proposals(
            self,
            semantic_world_model: Any,
            proposals: Optional[Iterable[Any]],
            *,
            wm_validation_packets: Optional[Iterable[Any]] = None,
            feedback_summary: Optional[Mapping[str, Any]] = None,
            min_confidence: float = 0.55,
        ) -> List[GraphMutationProposal]:
            world_model = _coerce_world_model(semantic_world_model)
            if world_model is None:
                return []
            packets = _coerce_validation_packets(wm_validation_packets)
            candidates = generate_graph_mutation_candidates(
                world_model,
                base_proposals=proposals,
                wm_validation_packets=packets,
            )
            global_features = _wm_global_features(
                world_model,
                feedback_summary=feedback_summary,
                wm_validation_packets=packets,
                mutation_proposals=candidates,
            )
            scored: List[GraphMutationProposal] = []
            with torch.no_grad():
                for proposal in candidates:
                    action = (
                        proposal.action
                        if proposal.action in PROPOSAL_ACTIONS
                        else "mark_for_review"
                    )
                    target_kind = _proposal_target_kind(proposal.target_ref, action)
                    feature_vec = np.concatenate(
                        [
                            global_features,
                            _one_hot(action, PROPOSAL_ACTIONS),
                            _one_hot(target_kind, PROPOSAL_TARGET_KINDS),
                            np.array(
                                [
                                    _clip01(_safe_float(proposal.confidence, 0.0)),
                                    min(
                                        float(len(proposal.source_refs or [])) / 4.0,
                                        1.0,
                                    ),
                                    1.0
                                    if _validation_error_for_ref(
                                        packets, proposal.target_ref
                                    )
                                    > 0.0
                                    else 0.0,
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    )
                    tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)
                    accept = _clip01(
                        float(torch.sigmoid(self.proposal_accept_model(tensor)).item())
                    )
                    confidence = _clip01(
                        float(
                            torch.sigmoid(self.proposal_confidence_model(tensor)).item()
                        )
                    )
                    merged_confidence = _clip01(
                        0.4 * _safe_float(proposal.confidence, 0.0) + 0.6 * confidence
                    )
                    if max(accept, merged_confidence) < min_confidence:
                        continue
                    scored.append(
                        GraphMutationProposal(
                            proposal_id=proposal.proposal_id,
                            action=action,
                            target_ref=proposal.target_ref,
                            confidence=max(accept, merged_confidence),
                            rationale=proposal.rationale,
                            source_refs=list(proposal.source_refs),
                            metadata={
                                **dict(proposal.metadata or {}),
                                "source": "semantic_wm_refiner",
                                "accept_score": float(accept),
                                "predicted_confidence": float(merged_confidence),
                            },
                        )
                    )
            return scored

    def _train_scalar_net(
        features: Sequence[Sequence[float]],
        targets: Sequence[float],
        *,
        hidden_dim: int,
        epochs: int,
        learning_rate: float,
    ) -> _ScalarNet:
        model = _ScalarNet(len(features[0]), hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        inputs = torch.tensor(np.asarray(features, dtype=np.float32))
        target_tensor = torch.tensor(np.asarray(targets, dtype=np.float32))
        for _ in range(max(int(epochs), 1)):
            preds = model(inputs)
            loss = loss_fn(preds, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def _train_vector_net(
        features: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        *,
        hidden_dim: int,
        epochs: int,
        learning_rate: float,
    ) -> _VectorNet:
        model = _VectorNet(len(features[0]), hidden_dim, len(targets[0]))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        inputs = torch.tensor(np.asarray(features, dtype=np.float32))
        target_tensor = torch.tensor(np.asarray(targets, dtype=np.float32))
        for _ in range(max(int(epochs), 1)):
            preds = model(inputs)
            loss = loss_fn(preds, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def train_semantic_wm_refiner_package(
        dataset: SemanticWMRefinementDataset,
        *,
        epochs: int = 24,
        learning_rate: float = 1e-3,
        hidden_dim: int = 48,
    ) -> SemanticWMRefinerPackage:
        if not dataset.object_features and not dataset.relation_features:
            raise ValueError("no semantic WM refinement samples available")
        object_features = dataset.object_features or [
            [0.0] * max(dataset.global_feature_dim, 1)
        ]
        object_targets = dataset.object_targets or [0.0]
        relation_features = dataset.relation_features or [
            [0.0] * max(dataset.global_feature_dim, 1)
        ]
        relation_targets = dataset.relation_targets or [0.0]
        capability_features = dataset.capability_features or [
            [0.0] * max(dataset.global_feature_dim, 1)
        ]
        capability_targets = dataset.capability_targets or [
            [0.0] * (len(CAPABILITY_KEYS) + 1)
        ]
        proposal_features = dataset.proposal_features or [
            [0.0] * max(dataset.global_feature_dim, 1)
        ]
        proposal_accept_targets = dataset.proposal_accept_targets or [0.0]
        proposal_confidence_targets = dataset.proposal_confidence_targets or [0.0]
        package = SemanticWMRefinerPackage(
            object_model=_train_scalar_net(
                object_features,
                object_targets,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
            relation_model=_train_scalar_net(
                relation_features,
                relation_targets,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
            capability_model=_train_vector_net(
                capability_features,
                capability_targets,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
            proposal_accept_model=_train_scalar_net(
                proposal_features,
                proposal_accept_targets,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
            proposal_confidence_model=_train_scalar_net(
                proposal_features,
                proposal_confidence_targets,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
            metadata={
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "object_dim": len(object_features[0]),
                "relation_dim": len(relation_features[0]),
                "capability_dim": len(capability_features[0]),
                "proposal_dim": len(proposal_features[0]),
                "dataset_metadata": dict(dataset.metadata),
            },
        )
        return package

    def shadow_fit_semantic_wm_refiner_package(
        semantic_world_model: Any,
        *,
        correction_overlay: Optional[Any] = None,
        feedback_summary: Optional[Mapping[str, Any]] = None,
        wm_validation_packets: Optional[Iterable[Any]] = None,
        graph_mutation_proposals: Optional[Iterable[Any]] = None,
        graph_mutation_execution: Optional[Mapping[str, Any]] = None,
        min_object_samples: int = 1,
    ) -> Optional[SemanticWMRefinerPackage]:
        world_model = _coerce_world_model(semantic_world_model)
        if (
            world_model is None
            or len(list(world_model.objects or [])) < min_object_samples
        ):
            return None
        dataset = build_semantic_wm_refinement_dataset_from_examples(
            [
                {
                    "semantic_world_model": world_model,
                    "correction_overlay": _coerce_overlay(correction_overlay).to_dict(),
                    "feedback_summary": dict(feedback_summary or {}),
                    "wm_validation_packets": [
                        item.to_dict()
                        for item in _coerce_validation_packets(wm_validation_packets)
                    ],
                    "graph_mutation_proposals": [
                        item.to_dict()
                        for item in _coerce_proposals(graph_mutation_proposals)
                    ],
                    "graph_mutation_execution": dict(graph_mutation_execution or {}),
                }
            ]
        )
        if not dataset.object_features:
            return None
        try:
            return train_semantic_wm_refiner_package(
                dataset, epochs=16, learning_rate=2e-3
            )
        except Exception:
            return None


except Exception:
    TORCH_AVAILABLE = False

    @dataclass
    class SemanticWMRefinerPackage:  # type: ignore[no-redef]
        metadata: Dict[str, Any] = field(default_factory=dict)

        def to_checkpoint(self) -> Dict[str, Any]:
            return {"metadata": dict(self.metadata)}

        @classmethod
        def from_checkpoint(
            cls, payload: Mapping[str, Any]
        ) -> "SemanticWMRefinerPackage":
            return cls(metadata=dict(payload.get("metadata", {}) or {}))

        def predict_correction_overlay(
            self,
            semantic_world_model: Any,
            *,
            wm_validation_packets: Optional[Iterable[Any]] = None,
            feedback_summary: Optional[Mapping[str, Any]] = None,
        ) -> SemanticWMCorrectionOverlay:
            world_model = _coerce_world_model(semantic_world_model)
            if world_model is None:
                return SemanticWMCorrectionOverlay()
            packets = _coerce_validation_packets(wm_validation_packets)
            adjustments = {
                obj.object_id: -0.15 * _validation_error_for_ref(packets, obj.object_id)
                for obj in list(world_model.objects or [])
                if _validation_error_for_ref(packets, obj.object_id) > 0.0
            }
            return SemanticWMCorrectionOverlay(
                object_confidence_adjustments=adjustments,
                relation_confidence_adjustments={},
                capability_adjustments={},
                topology_adjustments={"learned_refiner_active": True},
                meta_node_pressure=max(
                    [abs(value) for value in adjustments.values()] + [0.0]
                ),
                target_refs=sorted(adjustments.keys()),
                metadata={"source": "semantic_wm_refiner", "model_type": "fallback"},
            )

        def score_graph_mutation_proposals(
            self,
            semantic_world_model: Any,
            proposals: Optional[Iterable[Any]],
            *,
            wm_validation_packets: Optional[Iterable[Any]] = None,
            feedback_summary: Optional[Mapping[str, Any]] = None,
            min_confidence: float = 0.55,
        ) -> List[GraphMutationProposal]:
            world_model = _coerce_world_model(semantic_world_model)
            if world_model is None:
                return []
            packets = _coerce_validation_packets(wm_validation_packets)
            candidates = generate_graph_mutation_candidates(
                world_model, base_proposals=proposals, wm_validation_packets=packets
            )
            scored: List[GraphMutationProposal] = []
            for proposal in candidates:
                confidence = max(
                    _clip01(_safe_float(proposal.confidence, 0.0)),
                    _clip01(
                        0.3
                        + 0.5 * _validation_error_for_ref(packets, proposal.target_ref)
                    ),
                )
                if confidence < min_confidence:
                    continue
                scored.append(
                    GraphMutationProposal(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=proposal.target_ref,
                        confidence=confidence,
                        rationale=proposal.rationale,
                        source_refs=list(proposal.source_refs),
                        metadata={
                            **dict(proposal.metadata or {}),
                            "source": "semantic_wm_refiner",
                            "model_type": "fallback",
                        },
                    )
                )
            return scored

    def train_semantic_wm_refiner_package(  # type: ignore[misc,no-redef]
        *args: Any, **kwargs: Any
    ) -> SemanticWMRefinerPackage:
        raise ImportError("train_semantic_wm_refiner_package requires torch")

    def shadow_fit_semantic_wm_refiner_package(  # type: ignore[misc,no-redef]
        *args: Any, **kwargs: Any
    ) -> Optional[SemanticWMRefinerPackage]:
        return None


def merge_semantic_wm_correction_overlays(
    base_overlay: Optional[Any],
    learned_overlay: Optional[Any],
    *,
    learned_weight: float = 0.35,
) -> SemanticWMCorrectionOverlay:
    base = _coerce_overlay(base_overlay)
    learned = _coerce_overlay(learned_overlay)
    if not learned.metadata:
        return base
    weight = _clip01(learned_weight)
    inv_weight = 1.0 - weight
    object_keys = set(base.object_confidence_adjustments) | set(
        learned.object_confidence_adjustments
    )
    relation_keys = set(base.relation_confidence_adjustments) | set(
        learned.relation_confidence_adjustments
    )
    capability_keys = set(base.capability_adjustments) | set(
        learned.capability_adjustments
    )
    return SemanticWMCorrectionOverlay(
        object_confidence_adjustments={
            key: _clip(
                inv_weight
                * _safe_float(base.object_confidence_adjustments.get(key, 0.0), 0.0)
                + weight
                * _safe_float(learned.object_confidence_adjustments.get(key, 0.0), 0.0),
                -0.5,
                0.5,
            )
            for key in sorted(object_keys)
        },
        relation_confidence_adjustments={
            key: _clip(
                inv_weight
                * _safe_float(base.relation_confidence_adjustments.get(key, 0.0), 0.0)
                + weight
                * _safe_float(
                    learned.relation_confidence_adjustments.get(key, 0.0), 0.0
                ),
                -0.5,
                0.5,
            )
            for key in sorted(relation_keys)
        },
        capability_adjustments={
            key: _clip(
                inv_weight * _safe_float(base.capability_adjustments.get(key, 0.0), 0.0)
                + weight
                * _safe_float(learned.capability_adjustments.get(key, 0.0), 0.0),
                -0.5,
                0.5,
            )
            for key in sorted(capability_keys)
        },
        topology_adjustments={
            **dict(base.topology_adjustments or {}),
            **dict(learned.topology_adjustments or {}),
            "learned_overlay_blend_weight": float(weight),
        },
        meta_node_pressure=max(
            _safe_float(base.meta_node_pressure, 0.0),
            _safe_float(learned.meta_node_pressure, 0.0),
        ),
        target_refs=sorted(
            set(list(base.target_refs or []) + list(learned.target_refs or []))
        ),
        metadata={
            **dict(base.metadata or {}),
            "learned_overlay": learned.to_dict(),
            "merged_with_learned_refiner": True,
        },
    )


def merge_graph_mutation_proposals(
    base_proposals: Optional[Iterable[Any]],
    learned_proposals: Optional[Iterable[Any]],
    *,
    learned_min_confidence: float = 0.55,
) -> List[GraphMutationProposal]:
    merged: Dict[tuple[str, str], GraphMutationProposal] = {}
    for item in _coerce_proposals(base_proposals):
        merged[(item.action, item.target_ref)] = item
    for item in _coerce_proposals(learned_proposals):
        if _safe_float(item.confidence, 0.0) < learned_min_confidence:
            continue
        key = (item.action, item.target_ref)
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            continue
        merged[key] = GraphMutationProposal(
            proposal_id=existing.proposal_id or item.proposal_id,
            action=existing.action,
            target_ref=existing.target_ref,
            confidence=max(
                _safe_float(existing.confidence, 0.0), _safe_float(item.confidence, 0.0)
            ),
            rationale=existing.rationale or item.rationale,
            source_refs=sorted(
                set(list(existing.source_refs or []) + list(item.source_refs or []))
            ),
            metadata={
                **dict(existing.metadata or {}),
                "learned_refiner": dict(item.metadata or {}),
            },
        )
    return list(merged.values())


__all__ = [
    "CAPABILITY_KEYS",
    "PROPOSAL_ACTIONS",
    "PROPOSAL_TARGET_KINDS",
    "SemanticWMRefinementDataset",
    "SemanticWMRefinerPackage",
    "build_semantic_wm_refinement_dataset_from_artifact_dirs",
    "build_semantic_wm_refinement_dataset_from_examples",
    "generate_graph_mutation_candidates",
    "merge_graph_mutation_proposals",
    "merge_semantic_wm_correction_overlays",
    "shadow_fit_semantic_wm_refiner_package",
    "train_semantic_wm_refiner_package",
]
