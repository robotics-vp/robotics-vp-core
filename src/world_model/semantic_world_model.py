"""Additive semantic world-model scaffolding for Stage 1 and runtime fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import json
import numpy as np

from src.evidence.belief_state import BeliefState
from src.evidence.teacher_trace import TeacherTrace
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.vision.map_first_supervision.semantics import parse_vla_semantic_evidence
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.world_model.governed_video_world_model import (
    GovernedVideoHypothesis,
    VideoStateSnapshot,
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or []) if str(value)]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_mean(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return float(default)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _normalize_label(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


SEMANTIC_TOKEN_RULES: Dict[str, list[str]] = {
    "drawer": ["drawer", "object:drawer", "affordance:open", "affordance:grasp_handle"],
    "handle": ["drawer", "object:drawer", "part:handle", "affordance:grasp_handle"],
    "open": ["affordance:open", "intent:open"],
    "close": ["affordance:close", "intent:close"],
    "vase": [
        "vase",
        "object:vase",
        "fragile",
        "risk:fragility",
        "constraint:avoid_collision",
    ],
    "fragile": ["fragile", "risk:fragility", "constraint:avoid_collision"],
    "safety": ["safety", "constraint:avoid_collision", "priority:safety"],
    "collision": ["constraint:avoid_collision", "risk:collision"],
    "energy": ["energy_efficient", "objective:energy"],
    "efficient": ["energy_efficient", "objective:energy"],
    "recover": ["error_recovery", "mode:recovery"],
    "error": ["error_recovery", "mode:recovery"],
    "precision": ["high_precision", "quality:precision", "affordance:align"],
    "careful": ["high_precision", "quality:precision"],
    "fast": ["high_speed", "objective:throughput"],
    "throughput": ["high_speed", "objective:throughput"],
    "pick": ["object:workpiece", "affordance:pick"],
    "place": ["object:workpiece", "affordance:place"],
    "bench": ["bench", "region:bench"],
    "workcell": ["workpiece", "object:workpiece", "region:bench"],
    "human": ["teacher:human"],
    "expert": ["teacher:expert"],
}

OBJECT_PRIORS: Dict[str, Dict[str, Any]] = {
    "robot_arm": {
        "label": "robot_arm",
        "category": "agent",
        "aliases": ["manipulator", "arm"],
        "affordances": ["reach", "transport", "stabilize"],
        "state_tags": ["actuated"],
        "risk_tags": [],
    },
    "gripper": {
        "label": "gripper",
        "category": "end_effector",
        "aliases": ["tool"],
        "affordances": ["grasp", "release", "probe"],
        "state_tags": ["contact_capable"],
        "risk_tags": [],
    },
    "workspace": {
        "label": "workspace",
        "category": "scene_region",
        "aliases": ["scene", "workcell"],
        "affordances": ["clear_path", "recenter", "observe"],
        "state_tags": ["shared_context"],
        "risk_tags": [],
    },
    "drawer": {
        "label": "drawer",
        "category": "container",
        "aliases": ["cabinet_drawer"],
        "affordances": ["open", "close", "grasp_handle"],
        "state_tags": ["occluding"],
        "risk_tags": [],
    },
    "vase": {
        "label": "vase",
        "category": "fragile_object",
        "aliases": ["fragile_vessel"],
        "affordances": ["avoid_contact", "stabilize"],
        "state_tags": ["fragile"],
        "risk_tags": ["fragility", "tip_over"],
    },
    "workpiece": {
        "label": "workpiece",
        "category": "manipulated_object",
        "aliases": ["part", "component"],
        "affordances": ["pick", "place", "align"],
        "state_tags": ["movable"],
        "risk_tags": [],
    },
    "bench": {
        "label": "bench",
        "category": "support_surface",
        "aliases": ["table"],
        "affordances": ["support", "stage", "place"],
        "state_tags": ["static"],
        "risk_tags": [],
    },
}

TAG_TO_OBJECT = {
    "drawer": "drawer",
    "object:drawer": "drawer",
    "vase": "vase",
    "object:vase": "vase",
    "workpiece": "workpiece",
    "object:workpiece": "workpiece",
    "bench": "bench",
    "region:bench": "bench",
}


@dataclass(frozen=True)
class SemanticObjectState:
    object_id: str
    label: str
    category: str
    confidence: float
    salience: float
    aliases: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)
    state_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    track_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "category": self.category,
            "confidence": float(self.confidence),
            "salience": float(self.salience),
            "aliases": list(self.aliases),
            "affordances": list(self.affordances),
            "state_tags": list(self.state_tags),
            "risk_tags": list(self.risk_tags),
            "track_refs": list(self.track_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticObjectState":
        return cls(
            object_id=str(payload.get("object_id", "")),
            label=str(payload.get("label", "")),
            category=str(payload.get("category", "")),
            confidence=float(payload.get("confidence", 0.0)),
            salience=float(payload.get("salience", 0.0)),
            aliases=_strings(payload.get("aliases")),
            affordances=_strings(payload.get("affordances")),
            state_tags=_strings(payload.get("state_tags")),
            risk_tags=_strings(payload.get("risk_tags")),
            track_refs=_strings(payload.get("track_refs")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class SemanticRelationState:
    relation_id: str
    subject_id: str
    relation_type: str
    object_id: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "subject_id": self.subject_id,
            "relation_type": self.relation_type,
            "object_id": self.object_id,
            "confidence": float(self.confidence),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRelationState":
        return cls(
            relation_id=str(payload.get("relation_id", "")),
            subject_id=str(payload.get("subject_id", "")),
            relation_type=str(payload.get("relation_type", "")),
            object_id=str(payload.get("object_id", "")),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class SemanticMetaNode:
    node_id: str
    node_type: str
    priority: str
    score: float
    rationale: str
    target_refs: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "priority": self.priority,
            "score": float(self.score),
            "rationale": self.rationale,
            "target_refs": list(self.target_refs),
            "suggested_actions": list(self.suggested_actions),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticMetaNode":
        return cls(
            node_id=str(payload.get("node_id", "")),
            node_type=str(payload.get("node_type", "")),
            priority=str(payload.get("priority", "low")),
            score=float(payload.get("score", 0.0)),
            rationale=str(payload.get("rationale", "")),
            target_refs=_strings(payload.get("target_refs")),
            suggested_actions=_strings(payload.get("suggested_actions")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class SemanticWorldModelState:
    world_model_id: str
    episode_id: str
    task_id: str
    objective_preset: str
    semantic_tags: list[str]
    objects: list[SemanticObjectState] = field(default_factory=list)
    relations: list[SemanticRelationState] = field(default_factory=list)
    meta_nodes: list[SemanticMetaNode] = field(default_factory=list)
    capability_scores: Dict[str, float] = field(default_factory=dict)
    topology: Dict[str, Any] = field(default_factory=dict)
    functional_roles: Dict[str, Any] = field(default_factory=dict)
    risk_register: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "semantic_world_model_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_model_id": self.world_model_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "objective_preset": self.objective_preset,
            "semantic_tags": list(self.semantic_tags),
            "objects": [item.to_dict() for item in self.objects],
            "relations": [item.to_dict() for item in self.relations],
            "meta_nodes": [item.to_dict() for item in self.meta_nodes],
            "capability_scores": _float_mapping(self.capability_scores),
            "topology": _mapping(self.topology),
            "functional_roles": _mapping(self.functional_roles),
            "risk_register": _mapping(self.risk_register),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticWorldModelState":
        return cls(
            world_model_id=str(payload.get("world_model_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            task_id=str(payload.get("task_id", "")),
            objective_preset=str(payload.get("objective_preset", "")),
            semantic_tags=_strings(payload.get("semantic_tags")),
            objects=[
                SemanticObjectState.from_dict(item)
                for item in payload.get("objects", []) or []
            ],
            relations=[
                SemanticRelationState.from_dict(item)
                for item in payload.get("relations", []) or []
            ],
            meta_nodes=[
                SemanticMetaNode.from_dict(item)
                for item in payload.get("meta_nodes", []) or []
            ],
            capability_scores=_float_mapping(payload.get("capability_scores")),
            topology=_mapping(payload.get("topology")),
            functional_roles=_mapping(payload.get("functional_roles")),
            risk_register=_mapping(payload.get("risk_register")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "semantic_world_model_v1")),
        )

    def embed(self, encoder: Optional[Any] = None) -> np.ndarray:
        """Return fixed-dim embedding of this WM state.

        Parameters
        ----------
        encoder : SemanticStateEncoder, optional
            When provided, uses the learned set-transformer encoder.
            Falls back to deterministic mean-pool embedding.

        Returns
        -------
        np.ndarray of shape (embed_dim,)
        """
        if encoder is not None:
            try:
                import torch

                with torch.no_grad():
                    return encoder.encode_state(self).detach().numpy()
            except Exception:
                pass
        # Fallback to flat deterministic encoding
        from src.world_model.semantic_state_encoder import encode_wm_state_flat

        return encode_wm_state_flat(self)


@dataclass(frozen=True)
class SemanticWorldModelConfig:
    max_meta_nodes: int = 8
    meta_activation_floor: float = 0.2


class SemanticWorldModelBuilder:
    """Deterministic object-centric semantic state builder."""

    def __init__(self, config: Optional[SemanticWorldModelConfig] = None) -> None:
        self.config = config or SemanticWorldModelConfig()

    def infer_seed_tags(
        self,
        video_ref: Mapping[str, Any],
        base_tags: Optional[Sequence[Any]] = None,
    ) -> list[str]:
        tags = set(_strings(base_tags))
        tags.update(
            {
                "robot_arm",
                "gripper",
                "workspace",
                "object:robot_arm",
                "object:gripper",
                "region:workspace",
            }
        )
        tokens = self._tokenize_video_ref(video_ref)
        for token in tokens:
            for pattern, rule_tags in SEMANTIC_TOKEN_RULES.items():
                if pattern in token:
                    tags.update(rule_tags)
        metadata = video_ref.get("metadata", {})
        if isinstance(metadata, Mapping):
            if bool(metadata.get("success")) is False:
                tags.update({"error_recovery", "mode:recovery"})
            duration_s = float(metadata.get("duration_s", 0.0) or 0.0)
            if duration_s >= 20.0:
                tags.add("horizon:long")
            elif duration_s > 0.0:
                tags.add("horizon:short")
        demonstrator = str(video_ref.get("demonstrator", "")).lower()
        if "human" in demonstrator:
            tags.add("teacher:human")
        if "expert" in demonstrator:
            tags.add("teacher:expert")
        return sorted(tags)

    def build_from_stage1(
        self,
        *,
        video_ref: Mapping[str, Any],
        belief_state: BeliefState,
        video_state_snapshot: Optional[VideoStateSnapshot],
        hypotheses: Optional[Sequence[GovernedVideoHypothesis]],
        constraint_set: Optional[Mapping[str, Any]],
        objective_preset: str,
        semantic_tags: Optional[Sequence[Any]] = None,
        scene_tracks_payload: Optional[Any] = None,
        teacher_trace: Optional[Any] = None,
        vla_semantic_evidence: Optional[Any] = None,
        semantic_fusion_summary: Optional[Mapping[str, Any]] = None,
        stage2_ontology_proposals: Optional[Sequence[Any]] = None,
        stage2_task_refinements: Optional[Sequence[Any]] = None,
        stage2_tags: Optional[Sequence[Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SemanticWorldModelState:
        tags = self.infer_seed_tags(
            video_ref, base_tags=semantic_tags or belief_state.semantic_tags
        )
        return self._build_world_model(
            episode_id=str(video_ref.get("episode_id", belief_state.episode_id)),
            task_id=str(
                video_ref.get(
                    "task_type", video_ref.get("task_id", belief_state.episode_id)
                )
            ),
            objective_preset=objective_preset,
            semantic_tags=tags,
            belief_state=belief_state,
            video_state_snapshot=video_state_snapshot,
            hypotheses=hypotheses,
            constraint_set=constraint_set,
            scene_tracks_payload=scene_tracks_payload,
            teacher_trace=teacher_trace,
            vla_semantic_evidence=vla_semantic_evidence,
            semantic_fusion_summary=semantic_fusion_summary,
            stage2_ontology_proposals=stage2_ontology_proposals,
            stage2_task_refinements=stage2_task_refinements,
            stage2_tags=stage2_tags,
            artifact_refs=artifact_refs,
            metadata={
                "task_type": str(video_ref.get("task_type", "")),
                "instruction": str(video_ref.get("instruction", "")),
                "source_type": str(video_ref.get("source_type", "")),
                **_mapping(video_ref.get("metadata")),
                **_mapping(metadata),
            },
        )

    def build_from_runtime_fusion(
        self,
        *,
        episode_id: str,
        task_id: str,
        objective_preset: str,
        belief_state: BeliefState,
        semantic_tags: Optional[Sequence[Any]] = None,
        scene_tracks_payload: Optional[Any] = None,
        teacher_trace: Optional[Any] = None,
        vla_semantic_evidence: Optional[Any] = None,
        semantic_fusion_summary: Optional[Mapping[str, Any]] = None,
        stage2_ontology_proposals: Optional[Sequence[Any]] = None,
        stage2_task_refinements: Optional[Sequence[Any]] = None,
        stage2_tags: Optional[Sequence[Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SemanticWorldModelState:
        seed_tags = set(_strings(semantic_tags) or belief_state.semantic_tags)
        seed_tags.update({"robot_arm", "gripper", "workspace"})
        return self._build_world_model(
            episode_id=str(episode_id),
            task_id=str(task_id),
            objective_preset=objective_preset,
            semantic_tags=sorted(seed_tags),
            belief_state=belief_state,
            video_state_snapshot=None,
            hypotheses=None,
            constraint_set=None,
            scene_tracks_payload=scene_tracks_payload,
            teacher_trace=teacher_trace,
            vla_semantic_evidence=vla_semantic_evidence,
            semantic_fusion_summary=semantic_fusion_summary,
            stage2_ontology_proposals=stage2_ontology_proposals,
            stage2_task_refinements=stage2_task_refinements,
            stage2_tags=stage2_tags,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )

    def _build_world_model(
        self,
        *,
        episode_id: str,
        task_id: str,
        objective_preset: str,
        semantic_tags: Sequence[str],
        belief_state: BeliefState,
        video_state_snapshot: Optional[VideoStateSnapshot],
        hypotheses: Optional[Sequence[GovernedVideoHypothesis]],
        constraint_set: Optional[Mapping[str, Any]],
        scene_tracks_payload: Optional[Any],
        teacher_trace: Optional[Any],
        vla_semantic_evidence: Optional[Any],
        semantic_fusion_summary: Optional[Mapping[str, Any]],
        stage2_ontology_proposals: Optional[Sequence[Any]],
        stage2_task_refinements: Optional[Sequence[Any]],
        stage2_tags: Optional[Sequence[Any]],
        artifact_refs: Optional[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]],
    ) -> SemanticWorldModelState:
        grounded_scene = self._build_grounded_scene(
            scene_tracks_payload=scene_tracks_payload,
            teacher_trace=teacher_trace,
            vla_semantic_evidence=vla_semantic_evidence,
            semantic_tags=semantic_tags,
        )
        resolved_semantic_tags = sorted(
            set(_strings(semantic_tags)) | set(grounded_scene["semantic_tags"])
        )
        objects = self._build_objects(
            semantic_tags=resolved_semantic_tags,
            belief_state=belief_state,
            semantic_fusion_summary=semantic_fusion_summary,
            stage2_ontology_proposals=stage2_ontology_proposals,
            stage2_tags=stage2_tags,
            grounded_objects=grounded_scene["objects"],
        )
        relations = self._build_relations(
            semantic_tags=resolved_semantic_tags,
            objects=objects,
            hypotheses=hypotheses,
            grounded_relations=grounded_scene["relations"],
        )
        meta_nodes = self._build_meta_nodes(
            semantic_tags=resolved_semantic_tags,
            belief_state=belief_state,
            objects=objects,
            relations=relations,
            hypotheses=hypotheses,
            constraint_set=constraint_set,
            semantic_fusion_summary=semantic_fusion_summary,
            stage2_ontology_proposals=stage2_ontology_proposals,
            stage2_task_refinements=stage2_task_refinements,
            stage2_tags=stage2_tags,
        )
        capability_scores = self._build_capabilities(
            semantic_tags=resolved_semantic_tags,
            belief_state=belief_state,
            objects=objects,
            relations=relations,
            meta_nodes=meta_nodes,
            semantic_fusion_summary=semantic_fusion_summary,
            stage2_ontology_proposals=stage2_ontology_proposals,
            stage2_task_refinements=stage2_task_refinements,
            stage2_tags=stage2_tags,
        )
        topology = self._build_topology(
            objects=objects,
            relations=relations,
            meta_nodes=meta_nodes,
            capability_scores=capability_scores,
        )
        risk_register = self._build_risk_register(
            semantic_tags=resolved_semantic_tags,
            objects=objects,
            constraint_set=constraint_set,
            meta_nodes=meta_nodes,
        )
        functional_roles = self._build_functional_roles(
            semantic_tags=resolved_semantic_tags,
            hypotheses=hypotheses,
            meta_nodes=meta_nodes,
            capability_scores=capability_scores,
            semantic_fusion_summary=semantic_fusion_summary,
            stage2_ontology_proposals=stage2_ontology_proposals,
            stage2_task_refinements=stage2_task_refinements,
        )
        payload = {
            "episode_id": episode_id,
            "task_id": task_id,
            "objective_preset": objective_preset,
            "semantic_tags": list(resolved_semantic_tags),
            "topology": topology,
            "capability_scores": capability_scores,
            "risk_register": risk_register,
        }
        world_model_id = f"semantic_world_{sha256_json(payload)[:16]}"
        video_state_metadata = (
            video_state_snapshot.metadata if video_state_snapshot is not None else {}
        )
        return SemanticWorldModelState(
            world_model_id=world_model_id,
            episode_id=str(episode_id),
            task_id=str(task_id),
            objective_preset=str(objective_preset),
            semantic_tags=resolved_semantic_tags,
            objects=objects,
            relations=relations,
            meta_nodes=meta_nodes,
            capability_scores=capability_scores,
            topology=topology,
            functional_roles=functional_roles,
            risk_register=risk_register,
            artifact_refs={
                **_mapping(belief_state.artifact_refs),
                **_mapping(
                    video_state_snapshot.artifact_refs if video_state_snapshot else None
                ),
                **_mapping(artifact_refs),
            },
            provenance={
                "belief_id": belief_state.belief_id,
                "video_state_id": getattr(video_state_snapshot, "state_id", ""),
                "hypothesis_modes": [item.mode for item in (hypotheses or [])],
                "stage2_ontology_count": len(stage2_ontology_proposals or []),
                "stage2_refinement_count": len(stage2_task_refinements or []),
                "stage2_tag_count": len(stage2_tags or []),
                "grounded_track_count": int(
                    grounded_scene["summary"].get("track_count", 0)
                ),
            },
            metadata={
                **_mapping(video_state_metadata),
                "grounded_scene": grounded_scene["summary"],
                **_mapping(metadata),
            },
        )

    def _tokenize_video_ref(self, video_ref: Mapping[str, Any]) -> list[str]:
        fields = [
            str(video_ref.get("task_type", "")),
            str(video_ref.get("instruction", "")),
            str(video_ref.get("video_path", "")),
            str(video_ref.get("demonstrator", "")),
            str(video_ref.get("source_type", "")),
        ]
        metadata = video_ref.get("metadata", {})
        if isinstance(metadata, Mapping):
            fields.extend(str(value) for value in metadata.values())
            fields.extend(str(tag) for tag in metadata.get("semantic_tags", []) or [])
        tokens: list[str] = []
        for item in fields:
            cleaned = item.replace("/", " ").replace("_", " ").replace("-", " ").lower()
            tokens.extend(part for part in cleaned.split() if part)
        return tokens

    def _track_refs(
        self, semantic_fusion_summary: Optional[Mapping[str, Any]]
    ) -> list[str]:
        if not isinstance(semantic_fusion_summary, Mapping):
            return []
        raw = semantic_fusion_summary.get("track_ids") or semantic_fusion_summary.get(
            "tracks"
        )
        return _strings(raw)

    def _load_npz_payload(self, path_like: Any) -> Optional[Dict[str, Any]]:
        path = Path(str(path_like))
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
        except Exception:
            return None
        return {key: data[key] for key in data.files}

    def _load_json_payload(self, path_like: Any) -> Optional[Dict[str, Any]]:
        path = Path(str(path_like))
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _coerce_scene_tracks_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if isinstance(payload, (str, Path)):
            loaded = self._load_npz_payload(payload)
            return self._coerce_scene_tracks_payload(loaded)
        if not isinstance(payload, Mapping):
            return None
        if any(str(key).startswith("scene_tracks_v1/") for key in payload.keys()):
            return {
                str(key): value
                for key, value in payload.items()
                if str(key).startswith("scene_tracks_v1/")
            }
        nested = payload.get("scene_tracks_v1")
        if isinstance(nested, Mapping):
            return {f"scene_tracks_v1/{key}": value for key, value in nested.items()}
        scene_tracks_path = payload.get("scene_tracks_path") or payload.get(
            "scene_tracks_npz"
        )
        if scene_tracks_path:
            loaded = self._load_npz_payload(scene_tracks_path)
            return self._coerce_scene_tracks_payload(loaded)
        if {"track_ids", "poses_t"} <= set(str(key) for key in payload.keys()):
            return {f"scene_tracks_v1/{key}": value for key, value in payload.items()}
        return None

    def _coerce_teacher_trace(self, payload: Any) -> Optional[TeacherTrace]:
        if payload is None:
            return None
        if isinstance(payload, TeacherTrace):
            return payload
        if isinstance(payload, (str, Path)):
            loaded = self._load_json_payload(payload)
            return self._coerce_teacher_trace(loaded)
        if isinstance(payload, Mapping):
            try:
                return TeacherTrace.from_dict(payload)
            except Exception:
                return None
        return None

    def _coerce_vla_semantic_evidence(
        self,
        payload: Any,
        scene_track_ids: Optional[np.ndarray] = None,
    ) -> Any:
        if payload is None:
            return None
        if isinstance(payload, (str, Path)):
            loaded = self._load_npz_payload(payload)
            return self._coerce_vla_semantic_evidence(
                loaded, scene_track_ids=scene_track_ids
            )
        return parse_vla_semantic_evidence(payload, scene_track_ids=scene_track_ids)

    def _teacher_tags(self, teacher_trace: Optional[TeacherTrace]) -> list[str]:
        if teacher_trace is None:
            return []
        tags = set(_strings(teacher_trace.metadata.get("semantic_tags")))
        for step in teacher_trace.steps:
            tags.update(_strings(step.semantic_tags))
        instruction = str(teacher_trace.instruction or "").lower()
        for pattern, rule_tags in SEMANTIC_TOKEN_RULES.items():
            if pattern in instruction:
                tags.update(rule_tags)
        if teacher_trace.summary.get("teacher_confidence_mean", 0.0) > 0.0:
            tags.add("teacher:available")
        return sorted(tags)

    def _teacher_object_refs(self, teacher_trace: Optional[TeacherTrace]) -> list[str]:
        if teacher_trace is None:
            return []
        refs = set(_strings(teacher_trace.metadata.get("object_refs")))
        for step in teacher_trace.steps:
            refs.update(_strings(step.metadata.get("object_refs")))
        return sorted(_normalize_label(ref_) for ref_ in refs if _normalize_label(ref_))

    def _teacher_affordances(self, teacher_trace: Optional[TeacherTrace]) -> list[str]:
        if teacher_trace is None:
            return []
        affordances = set(_strings(teacher_trace.metadata.get("affordance_hints")))
        for step in teacher_trace.steps:
            affordances.update(_strings(step.metadata.get("affordance_hints")))
        affordances.update(self._instruction_affordances(teacher_trace.instruction))
        return sorted(
            _normalize_label(value) for value in affordances if _normalize_label(value)
        )

    def _teacher_risk_hints(self, teacher_trace: Optional[TeacherTrace]) -> list[str]:
        if teacher_trace is None:
            return []
        risks = set(_strings(teacher_trace.metadata.get("risk_hints")))
        for step in teacher_trace.steps:
            risks.update(_strings(step.metadata.get("risk_hints")))
        risks.update(
            tag.split("risk:", 1)[1]
            for tag in self._teacher_tags(teacher_trace)
            if tag.startswith("risk:")
        )
        return sorted(
            _normalize_label(value) for value in risks if _normalize_label(value)
        )

    def _instruction_affordances(self, instruction: str) -> list[str]:
        instruction = str(instruction or "").lower()
        affordances: set[str] = set()
        for token, rule_tags in SEMANTIC_TOKEN_RULES.items():
            if token in instruction:
                affordances.update(
                    tag.split("affordance:", 1)[1]
                    for tag in rule_tags
                    if tag.startswith("affordance:")
                )
        return sorted(affordances)

    def _grounded_label_to_tags(self, label: str, category: str) -> set[str]:
        normalized = _normalize_label(label)
        tags = {f"object:{normalized}"}
        if normalized in SEMANTIC_TOKEN_RULES:
            tags.update(SEMANTIC_TOKEN_RULES[normalized])
        if category == "human_body":
            tags.add("teacher:human")
        if category in {"support_surface", "scene_region"}:
            tags.add(f"region:{normalized}")
        return tags

    def _derive_track_label(
        self,
        *,
        track_id: str,
        entity_type: int,
        class_id: int,
        class_names: Optional[Sequence[str]],
    ) -> str:
        if class_names and 0 <= int(class_id) < len(class_names):
            return _normalize_label(str(class_names[int(class_id)]))
        if int(entity_type) == 1:
            return "human_body"
        normalized = _normalize_label(track_id)
        return normalized if normalized else "unknown_object"

    def _build_grounded_scene(
        self,
        *,
        scene_tracks_payload: Any,
        teacher_trace: Any,
        vla_semantic_evidence: Any,
        semantic_tags: Sequence[str],
    ) -> Dict[str, Any]:
        scene_tracks_dict = self._coerce_scene_tracks_payload(scene_tracks_payload)
        teacher = self._coerce_teacher_trace(teacher_trace)
        scene_tracks = None
        if scene_tracks_dict is not None:
            try:
                scene_tracks = deserialize_scene_tracks_v1(scene_tracks_dict)
            except Exception:
                scene_tracks = None
        vla_evidence = self._coerce_vla_semantic_evidence(
            vla_semantic_evidence,
            scene_track_ids=getattr(scene_tracks, "track_ids", None),
        )
        if scene_tracks is None:
            return {
                "objects": [],
                "relations": [],
                "semantic_tags": sorted(
                    set(_strings(semantic_tags)) | set(self._teacher_tags(teacher))
                ),
                "summary": {
                    "grounding_mode": "heuristic_fallback",
                    "track_count": 0,
                    "teacher_trace_present": bool(teacher),
                    "vla_semantic_evidence_present": bool(vla_evidence),
                },
            }
        scene_tracks_dict = scene_tracks_dict or {}

        track_ids = np.asarray(scene_tracks.track_ids)
        entity_types = np.asarray(scene_tracks.entity_types)
        class_ids = np.asarray(scene_tracks.class_ids)
        class_names = getattr(scene_tracks, "class_names", None)
        poses_t = (
            np.asarray(scene_tracks.poses_t, dtype=np.float32)
            if getattr(scene_tracks, "poses_t", None) is not None
            else np.zeros((0, 0, 3), dtype=np.float32)
        )
        visibility = (
            np.asarray(scene_tracks.visibility, dtype=np.float32)
            if getattr(scene_tracks, "visibility", None) is not None
            else np.zeros((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
        )
        occlusion = (
            np.asarray(scene_tracks.occlusion, dtype=np.float32)
            if getattr(scene_tracks, "occlusion", None) is not None
            else np.zeros((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
        )
        ir_loss = (
            np.asarray(scene_tracks.ir_loss, dtype=np.float32)
            if getattr(scene_tracks, "ir_loss", None) is not None
            else np.zeros((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
        )
        converged = (
            np.asarray(scene_tracks.converged, dtype=np.float32)
            if getattr(scene_tracks, "converged", None) is not None
            else np.ones((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
        )
        teacher_tags = self._teacher_tags(teacher)
        instruction = ""
        if teacher is not None:
            instruction = str(teacher.instruction or "")
        instruction_affordances = self._instruction_affordances(instruction)
        teacher_object_refs = set(self._teacher_object_refs(teacher))
        teacher_affordances = set(self._teacher_affordances(teacher))
        teacher_risk_hints = set(self._teacher_risk_hints(teacher))
        track_sem_conf = {}
        if (
            vla_evidence is not None
            and getattr(vla_evidence, "class_probs", None) is not None
        ):
            probs = np.asarray(vla_evidence.class_probs, dtype=np.float32)
            if probs.ndim == 3 and probs.shape[1] == len(track_ids):
                track_sem_conf = {
                    str(track_ids[idx]): float(
                        np.mean(np.max(probs[:, idx, :], axis=-1))
                    )
                    for idx in range(len(track_ids))
                }
        if (
            vla_evidence is not None
            and getattr(vla_evidence, "confidence", None) is not None
        ):
            conf_arr = np.asarray(vla_evidence.confidence, dtype=np.float32)
            if conf_arr.ndim >= 2 and conf_arr.shape[1] == len(track_ids):
                track_sem_conf.update(
                    {
                        str(track_ids[idx]): max(
                            track_sem_conf.get(str(track_ids[idx]), 0.0),
                            float(np.mean(conf_arr[:, idx])),
                        )
                        for idx in range(len(track_ids))
                    }
                )
        track_label_source = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_label_source", [])
            )
        ]
        track_categories = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_category", [])
            )
        ]
        track_label_confidence = np.asarray(
            scene_tracks_dict.get(
                "scene_tracks_v1/track_label_confidence",
                np.zeros((len(track_ids),), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        track_source_instance_ids = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_source_instance_id", [])
            )
        ]
        track_source_object_ids = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_source_object_id", [])
            )
        ]
        track_hint_object_ids = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_hint_object_id", [])
            )
        ]
        raw_track_tags = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_semantic_tags_json", [])
            )
        ]
        raw_track_affordances = [
            str(value)
            for value in list(
                scene_tracks_dict.get("scene_tracks_v1/track_affordances_json", [])
            )
        ]

        grounded_objects: list[SemanticObjectState] = []
        grounded_tags: set[str] = set(_strings(semantic_tags))
        if bool(scene_tracks_dict):
            grounded_tags.add("scene_tracks:present")
        summary = getattr(scene_tracks, "summary", None) or {}
        quality_score = _safe_float(
            summary.get("quality_score", summary.get("scene_ir_quality", 0.0))
        )
        for idx, track_id in enumerate(track_ids):
            label = self._derive_track_label(
                track_id=str(track_id),
                entity_type=int(entity_types[idx]) if idx < len(entity_types) else 0,
                class_id=int(class_ids[idx]) if idx < len(class_ids) else -1,
                class_names=class_names,
            )
            normalized_label = _normalize_label(label)
            label_source = (
                track_label_source[idx] if idx < len(track_label_source) else ""
            )
            label_confidence = (
                float(track_label_confidence[idx])
                if idx < track_label_confidence.shape[0]
                else 0.0
            )
            source_instance_id = (
                _normalize_label(track_source_instance_ids[idx])
                if idx < len(track_source_instance_ids)
                else ""
            )
            source_object_id = (
                _normalize_label(track_source_object_ids[idx])
                if idx < len(track_source_object_ids)
                else ""
            )
            hint_object_id = source_object_id or (
                _normalize_label(track_hint_object_ids[idx])
                if idx < len(track_hint_object_ids)
                else ""
            )
            extra_track_tags: set[str] = set()
            extra_track_affordances: set[str] = set()
            if idx < len(raw_track_tags):
                try:
                    extra_track_tags.update(_strings(json.loads(raw_track_tags[idx])))
                except Exception:
                    pass
            if idx < len(raw_track_affordances):
                try:
                    extra_track_affordances.update(
                        _strings(json.loads(raw_track_affordances[idx]))
                    )
                except Exception:
                    pass
            prior = OBJECT_PRIORS.get(normalized_label, {})
            visibility_mean = _safe_mean(
                visibility[:, idx]
                if visibility.ndim >= 2 and idx < visibility.shape[1]
                else [0.0]
            )
            occlusion_mean = _safe_mean(
                occlusion[:, idx]
                if occlusion.ndim >= 2 and idx < occlusion.shape[1]
                else [0.0]
            )
            ir_loss_mean = _safe_mean(
                ir_loss[:, idx]
                if ir_loss.ndim >= 2 and idx < ir_loss.shape[1]
                else [0.0]
            )
            converged_rate = _safe_mean(
                converged[:, idx]
                if converged.ndim >= 2 and idx < converged.shape[1]
                else [1.0],
                1.0,
            )
            position_seq = (
                poses_t[:, idx, :]
                if poses_t.ndim == 3 and idx < poses_t.shape[1]
                else np.zeros((1, 3), dtype=np.float32)
            )
            motion_score = 0.0
            if position_seq.shape[0] > 1:
                diffs = np.diff(position_seq, axis=0)
                motion_score = float(np.mean(np.linalg.norm(diffs, axis=-1)))
            teacher_match = float(
                bool(
                    any(
                        normalized_label in tag or tag.endswith(normalized_label)
                        for tag in teacher_tags
                    )
                    or normalized_label in teacher_object_refs
                    or (bool(hint_object_id) and hint_object_id in teacher_object_refs)
                )
            )
            semantic_conf = max(
                track_sem_conf.get(str(track_id), 0.0), label_confidence * 0.75
            )
            confidence = _clip01(
                0.2
                + 0.3 * visibility_mean
                + 0.15 * (1.0 - occlusion_mean)
                + 0.15 * converged_rate
                + 0.1 * (1.0 - min(ir_loss_mean, 1.0))
                + 0.1 * semantic_conf
            )
            salience = _clip01(
                0.15
                + 0.25 * visibility_mean
                + 0.2 * min(motion_score * 4.0, 1.0)
                + 0.15 * teacher_match
                + 0.1 * semantic_conf
                + 0.15 * max(quality_score, 0.0)
            )
            state_tags = list(prior.get("state_tags", []))
            affordances = list(prior.get("affordances", []))
            risk_tags = list(prior.get("risk_tags", []))
            resolved_category = (
                track_categories[idx]
                if idx < len(track_categories) and str(track_categories[idx]).strip()
                else ""
            )
            if motion_score > 0.05:
                state_tags.append("dynamic_track")
            else:
                state_tags.append("static_track")
            if occlusion_mean > 0.4:
                state_tags.append("partially_occluded")
                grounded_tags.add("scene:occluded")
            if int(entity_types[idx]) == 1:
                state_tags.append("human_present")
                grounded_tags.add("human_present")
            for affordance in instruction_affordances:
                if normalized_label in instruction.lower():
                    affordances.append(affordance)
            if (
                normalized_label in teacher_object_refs
                or hint_object_id in teacher_object_refs
            ):
                affordances.extend(teacher_affordances)
                risk_tags.extend(teacher_risk_hints)
            if "fragile" in teacher_tags and normalized_label in {
                "vase",
                "glass",
                "fragile_object",
            }:
                risk_tags.append("fragility")
            for tag in extra_track_tags:
                if tag.startswith("risk:"):
                    risk_tags.append(tag.split("risk:", 1)[1])
                elif tag.startswith("affordance:"):
                    affordances.append(tag.split("affordance:", 1)[1])
                else:
                    state_tags.append(tag)
            affordances.extend(extra_track_affordances)
            if label_source:
                state_tags.append(f"label_source:{_normalize_label(label_source)}")
            category = str(
                resolved_category
                or prior.get(
                    "category",
                    "human_body" if int(entity_types[idx]) == 1 else "tracked_object",
                )
            )
            object_id = f"track:{track_id}"
            grounded_tags.update(
                self._grounded_label_to_tags(normalized_label, category)
            )
            grounded_tags.update(_strings(list(extra_track_tags)))
            grounded_objects.append(
                SemanticObjectState(
                    object_id=object_id,
                    label=normalized_label,
                    category=category,
                    confidence=confidence,
                    salience=salience,
                    aliases=_strings(prior.get("aliases")),
                    affordances=sorted(set(_strings(affordances))),
                    state_tags=sorted(set(_strings(state_tags))),
                    risk_tags=sorted(set(_strings(risk_tags))),
                    track_refs=[str(track_id)],
                    metadata={
                        "track_id": str(track_id),
                        "class_id": int(class_ids[idx]) if idx < len(class_ids) else -1,
                        "class_name": normalized_label,
                        "entity_type": int(entity_types[idx])
                        if idx < len(entity_types)
                        else 0,
                        "visibility_mean": visibility_mean,
                        "occlusion_mean": occlusion_mean,
                        "converged_rate": converged_rate,
                        "ir_loss_mean": ir_loss_mean,
                        "motion_score": motion_score,
                        "mean_position": position_seq.mean(axis=0).tolist()
                        if position_seq.size
                        else [0.0, 0.0, 0.0],
                        "teacher_match": bool(teacher_match),
                        "semantic_confidence": semantic_conf,
                        "label_source": label_source,
                        "label_confidence": label_confidence,
                        "hint_object_id": hint_object_id,
                        "source_object_id": source_object_id,
                        "source_instance_id": source_instance_id,
                    },
                )
            )

        grounded_relations: list[SemanticRelationState] = []
        for obj in grounded_objects:
            relation_id = (
                f"rel_{sha256_json([obj.object_id, 'inside', 'workspace'])[:12]}"
            )
            grounded_relations.append(
                SemanticRelationState(
                    relation_id=relation_id,
                    subject_id=obj.object_id,
                    relation_type="inside",
                    object_id="workspace",
                    confidence=_clip01(0.7 + 0.2 * obj.confidence),
                    metadata={"source": "scene_tracks"},
                )
            )
        for idx, subject in enumerate(grounded_objects):
            pos_i = np.asarray(
                subject.metadata.get("mean_position", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            motion_i = _safe_float(subject.metadata.get("motion_score", 0.0))
            for jdx in range(idx + 1, len(grounded_objects)):
                target = grounded_objects[jdx]
                pos_j = np.asarray(
                    target.metadata.get("mean_position", [0.0, 0.0, 0.0]),
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(pos_i - pos_j))
                if distance <= 0.25:
                    relation_id = f"rel_{sha256_json([subject.object_id, 'near', target.object_id])[:12]}"
                    grounded_relations.append(
                        SemanticRelationState(
                            relation_id=relation_id,
                            subject_id=subject.object_id,
                            relation_type="near",
                            object_id=target.object_id,
                            confidence=_clip01(0.85 - distance),
                            metadata={"distance_m": distance, "source": "scene_tracks"},
                        )
                    )
                motion_j = _safe_float(target.metadata.get("motion_score", 0.0))
                if (
                    distance <= 0.12
                    and abs(motion_i - motion_j) <= 0.03
                    and max(motion_i, motion_j) > 0.02
                ):
                    relation_id = f"rel_{sha256_json([subject.object_id, 'moves_with', target.object_id])[:12]}"
                    grounded_relations.append(
                        SemanticRelationState(
                            relation_id=relation_id,
                            subject_id=subject.object_id,
                            relation_type="moves_with",
                            object_id=target.object_id,
                            confidence=_clip01(0.7 + 0.1 * (motion_i + motion_j)),
                            metadata={"distance_m": distance, "source": "scene_tracks"},
                        )
                    )
                if {"support_surface", "scene_region"} & {
                    subject.category,
                    target.category,
                } and abs(float(pos_i[2] - pos_j[2])) <= 0.15:
                    support = (
                        subject
                        if subject.category in {"support_surface", "scene_region"}
                        else target
                    )
                    resting = target if support is subject else subject
                    relation_id = f"rel_{sha256_json([resting.object_id, 'rests_on', support.object_id])[:12]}"
                    grounded_relations.append(
                        SemanticRelationState(
                            relation_id=relation_id,
                            subject_id=resting.object_id,
                            relation_type="rests_on",
                            object_id=support.object_id,
                            confidence=_clip01(0.75 - min(distance, 0.2)),
                            metadata={"distance_m": distance, "source": "scene_tracks"},
                        )
                    )

        grounding_tags = sorted(grounded_tags | set(teacher_tags))
        grounding_summary = {
            "grounding_mode": "scene_tracks",
            "track_count": int(len(track_ids)),
            "grounded_object_count": int(len(grounded_objects)),
            "grounded_relation_count": int(len(grounded_relations)),
            "teacher_trace_present": bool(teacher),
            "vla_semantic_evidence_present": bool(vla_evidence),
            "training_eligible": bool(summary.get("training_eligible", False)),
            "scene_ir_quality": quality_score,
            "semantic_density_score": _safe_float(
                summary.get("semantic_density_score", 0.0)
            ),
            "semantic_grounding_ready": bool(summary.get("grounding_ready", False)),
        }
        return {
            "objects": grounded_objects,
            "relations": grounded_relations,
            "semantic_tags": grounding_tags,
            "summary": grounding_summary,
        }

    def _build_objects(
        self,
        *,
        semantic_tags: Sequence[str],
        belief_state: BeliefState,
        semantic_fusion_summary: Optional[Mapping[str, Any]],
        stage2_ontology_proposals: Optional[Sequence[Any]],
        stage2_tags: Optional[Sequence[Any]],
        grounded_objects: Optional[Sequence[SemanticObjectState]] = None,
    ) -> list[SemanticObjectState]:
        tags = set(_strings(semantic_tags))
        object_names = {"robot_arm", "gripper", "workspace"}
        for tag in tags:
            object_name = TAG_TO_OBJECT.get(tag)
            if object_name:
                object_names.add(object_name)
        for proposal in stage2_ontology_proposals or []:
            target_object_id = getattr(proposal, "target_object_id", None)
            if target_object_id:
                object_names.add(str(target_object_id))
        for record in stage2_tags or []:
            tag_dict = to_json_safe(record)
            if isinstance(tag_dict, dict):
                for key in ("object_name", "target_object_id"):
                    if tag_dict.get(key):
                        object_names.add(str(tag_dict[key]))

        confidence_mean = float(
            belief_state.state_vector.get("evidence_confidence_mean", 0.0)
        )
        coverage = float(belief_state.state_vector.get("evidence_coverage", 0.0))
        disagreement = float(
            belief_state.state_vector.get("evidence_disagreement_mean", 0.0)
        )
        track_refs = self._track_refs(semantic_fusion_summary)
        objects: list[SemanticObjectState] = list(grounded_objects or [])
        existing_ids = {item.object_id for item in objects}

        for object_name in sorted(object_names):
            if object_name in existing_ids:
                continue
            prior = OBJECT_PRIORS.get(object_name, {})
            explicit_tag = (
                1.0 if object_name in tags or f"object:{object_name}" in tags else 0.0
            )
            risk_tags = list(prior.get("risk_tags", []))
            state_tags = list(prior.get("state_tags", []))
            affordances = list(prior.get("affordances", []))
            if (
                "fragile" in tags
                and object_name == "vase"
                and "fragility" not in risk_tags
            ):
                risk_tags.append("fragility")
            if "high_precision" in tags and object_name in {"gripper", "workpiece"}:
                state_tags.append("precision_sensitive")
            if "mode:recovery" in tags and object_name in {
                "gripper",
                "drawer",
                "workpiece",
            }:
                state_tags.append("recovery_context")
            confidence = _clip01(
                0.25 + 0.35 * confidence_mean + 0.25 * explicit_tag + 0.15 * coverage
            )
            salience = _clip01(
                0.2
                + 0.2 * explicit_tag
                + 0.2 * float(bool(risk_tags))
                + 0.15 * float(object_name in {"drawer", "vase", "workpiece"})
                + 0.15 * (1.0 - disagreement)
            )
            object_track_refs = (
                track_refs
                if object_name in {"workspace", "workpiece", "drawer", "vase"}
                else []
            )
            objects.append(
                SemanticObjectState(
                    object_id=str(object_name),
                    label=str(prior.get("label", object_name)),
                    category=str(prior.get("category", "semantic_entity")),
                    confidence=confidence,
                    salience=salience,
                    aliases=_strings(prior.get("aliases")),
                    affordances=sorted(set(_strings(affordances))),
                    state_tags=sorted(set(_strings(state_tags))),
                    risk_tags=sorted(set(_strings(risk_tags))),
                    track_refs=object_track_refs,
                    metadata={
                        "explicit_tag": bool(explicit_tag),
                        "coverage": coverage,
                        "disagreement": disagreement,
                    },
                )
            )
        return sorted(objects, key=lambda item: item.object_id)

    def _build_relations(
        self,
        *,
        semantic_tags: Sequence[str],
        objects: Sequence[SemanticObjectState],
        hypotheses: Optional[Sequence[GovernedVideoHypothesis]],
        grounded_relations: Optional[Sequence[SemanticRelationState]] = None,
    ) -> list[SemanticRelationState]:
        object_ids = {item.object_id for item in objects}
        hypothesis_modes = {item.mode for item in (hypotheses or [])}
        relations: list[tuple[str, str, str, float, Dict[str, Any]]] = [
            ("robot_arm", "controls", "gripper", 0.95, {}),
            ("gripper", "operates_in", "workspace", 0.9, {}),
        ]
        if "drawer" in object_ids:
            relations.append(
                ("gripper", "acts_on", "drawer", 0.82, {"affordance": "open"})
            )
            relations.append(("drawer", "inside", "workspace", 0.88, {}))
        if "vase" in object_ids:
            relations.append(("vase", "inside", "workspace", 0.86, {}))
            relations.append(
                ("gripper", "avoid_contact", "vase", 0.92, {"safety": True})
            )
        if {"bench", "workpiece"} <= object_ids:
            relations.append(("workpiece", "rests_on", "bench", 0.8, {}))
            relations.append(
                (
                    "gripper",
                    "transfers",
                    "workpiece",
                    0.84,
                    {"affordance": "pick_place"},
                )
            )
        if "semantic_disambiguation" in hypothesis_modes:
            relations.append(
                (
                    "gripper",
                    "reobserve",
                    "workspace",
                    0.72,
                    {"mode": "semantic_disambiguation"},
                )
            )
        if "recovery_branch" in hypothesis_modes:
            target = (
                "drawer"
                if "drawer" in object_ids
                else "workpiece"
                if "workpiece" in object_ids
                else "workspace"
            )
            relations.append(
                ("gripper", "recovers_from", target, 0.78, {"mode": "recovery_branch"})
            )
        if "energy_saver_retiming" in hypothesis_modes:
            relations.append(
                (
                    "robot_arm",
                    "retimes_for",
                    "workspace",
                    0.65,
                    {"mode": "energy_saver_retiming"},
                )
            )

        result: list[SemanticRelationState] = list(grounded_relations or [])
        seen_relation_ids = {item.relation_id for item in result}
        for subject_id, relation_type, object_id, confidence, metadata in relations:
            if subject_id not in object_ids or object_id not in object_ids:
                continue
            relation_id = f"rel_{sha256_json([subject_id, relation_type, object_id, metadata])[:12]}"
            if relation_id in seen_relation_ids:
                continue
            result.append(
                SemanticRelationState(
                    relation_id=relation_id,
                    subject_id=subject_id,
                    relation_type=relation_type,
                    object_id=object_id,
                    confidence=_clip01(confidence),
                    metadata=metadata,
                )
            )
            seen_relation_ids.add(relation_id)
        return sorted(
            result,
            key=lambda item: (item.subject_id, item.relation_type, item.object_id),
        )

    def _build_meta_nodes(
        self,
        *,
        semantic_tags: Sequence[str],
        belief_state: BeliefState,
        objects: Sequence[SemanticObjectState],
        relations: Sequence[SemanticRelationState],
        hypotheses: Optional[Sequence[GovernedVideoHypothesis]],
        constraint_set: Optional[Mapping[str, Any]],
        semantic_fusion_summary: Optional[Mapping[str, Any]],
        stage2_ontology_proposals: Optional[Sequence[Any]],
        stage2_task_refinements: Optional[Sequence[Any]],
        stage2_tags: Optional[Sequence[Any]],
    ) -> list[SemanticMetaNode]:
        tags = set(_strings(semantic_tags))
        confidence = float(
            belief_state.state_vector.get("evidence_confidence_mean", 0.0)
        )
        disagreement = float(
            belief_state.state_vector.get("evidence_disagreement_mean", 0.0)
        )
        coverage = float(belief_state.state_vector.get("evidence_coverage", 0.0))
        teacher_alignment = float(
            belief_state.state_vector.get("teacher_alignment", 0.0)
        )
        hard_bounds = (
            constraint_set.get("hard_bounds", {})
            if isinstance(constraint_set, Mapping)
            else {}
        )
        hypothesis_modes = {item.mode for item in (hypotheses or [])}
        has_risk_object = any(item.risk_tags for item in objects)
        has_recovery = (
            "error_recovery" in tags
            or "mode:recovery" in tags
            or "recovery_branch" in hypothesis_modes
        )
        has_fusion = bool(semantic_fusion_summary)
        has_stage2 = bool(
            stage2_ontology_proposals or stage2_task_refinements or stage2_tags
        )

        node_specs = [
            (
                "semantic_memory_refresh",
                _clip01(
                    (1.0 - coverage) * 0.55
                    + disagreement * 0.3
                    + (1.0 - confidence) * 0.15
                ),
                ["belief_state", "semantic_snapshot"],
                ["refresh_scene_memory", "request_additional_evidence"],
                "Coverage or confidence is low enough that semantic memory should be refreshed.",
            ),
            (
                "risk_triage",
                _clip01(
                    0.25
                    + 0.35 * float(has_risk_object)
                    + 0.2 * min(len(hard_bounds), 4) / 4.0
                    + 0.2 * float("safety" in tags)
                ),
                [item.object_id for item in objects if item.risk_tags],
                ["tighten_meta_node_attention", "prioritize_fragility_review"],
                "Fragility, collision, or safety constraints are active.",
            ),
            (
                "recovery_router",
                _clip01(
                    0.2
                    + 0.45 * float(has_recovery)
                    + 0.2 * disagreement
                    + 0.15 * (1.0 - teacher_alignment)
                ),
                ["gripper", "workspace"],
                ["route_recovery_supervision", "collect_recovery_counterfactuals"],
                "Recovery signatures are present in tags or governed hypotheses.",
            ),
            (
                "affordance_router",
                _clip01(
                    0.2
                    + 0.08 * min(sum(len(item.affordances) for item in objects), 8)
                    + 0.18 * coverage
                ),
                [item.object_id for item in objects if item.affordances],
                ["prioritize_affordance_alignment", "emit_affordance_sidecars"],
                "Object affordances are rich enough to drive meta-node routing.",
            ),
            (
                "fusion_bridge",
                _clip01(
                    0.15
                    + 0.55 * float(has_fusion)
                    + 0.15 * confidence
                    + 0.15 * coverage
                ),
                ["semantic_fusion", "belief_state"],
                ["materialize_runtime_backbone", "persist_fusion_summary"],
                "Runtime fusion evidence is available for backbone materialization.",
            ),
            (
                "ontology_router",
                _clip01(
                    0.15
                    + 0.35 * float(bool(stage2_ontology_proposals))
                    + 0.2 * float(has_stage2)
                    + 0.1 * coverage
                ),
                [
                    getattr(item, "proposal_id", "")
                    for item in (stage2_ontology_proposals or [])
                ],
                ["route_ontology_advisories", "align_object_vocabulary"],
                "Stage 2 ontology proposals should be translated into meta-node work.",
            ),
            (
                "task_graph_router",
                _clip01(
                    0.12
                    + 0.4 * float(bool(stage2_task_refinements))
                    + 0.2 * float(len(relations) > 4)
                    + 0.1 * float(bool(hypothesis_modes))
                ),
                [
                    getattr(item, "proposal_id", "")
                    for item in (stage2_task_refinements or [])
                ],
                ["route_task_graph_review", "emit_subtask_reconciliation"],
                "Task graph refinements should stay advisory but remain visible to orchestration.",
            ),
            (
                "efficiency_router",
                _clip01(
                    0.18
                    + 0.3
                    * float(
                        "objective:throughput" in tags or "objective:energy" in tags
                    )
                    + 0.18 * teacher_alignment
                ),
                ["robot_arm", "gripper"],
                ["bias_toward_efficiency_meta_nodes", "review_energy_tradeoffs"],
                "Objective mix indicates efficiency or energy pressure.",
            ),
        ]

        nodes: list[SemanticMetaNode] = []
        for node_type, score, target_refs, actions, rationale in node_specs:
            if score < self.config.meta_activation_floor:
                continue
            priority = (
                "critical"
                if score >= 0.85
                else "high"
                if score >= 0.65
                else "medium"
                if score >= 0.4
                else "low"
            )
            node_id = f"meta_{sha256_json([node_type, target_refs, actions])[:12]}"
            nodes.append(
                SemanticMetaNode(
                    node_id=node_id,
                    node_type=node_type,
                    priority=priority,
                    score=score,
                    rationale=rationale,
                    target_refs=target_refs,
                    suggested_actions=actions,
                    metadata={
                        "confidence": confidence,
                        "coverage": coverage,
                        "disagreement": disagreement,
                    },
                )
            )
        return sorted(nodes, key=lambda item: (-item.score, item.node_type))[
            : self.config.max_meta_nodes
        ]

    def _build_capabilities(
        self,
        *,
        semantic_tags: Sequence[str],
        belief_state: BeliefState,
        objects: Sequence[SemanticObjectState],
        relations: Sequence[SemanticRelationState],
        meta_nodes: Sequence[SemanticMetaNode],
        semantic_fusion_summary: Optional[Mapping[str, Any]],
        stage2_ontology_proposals: Optional[Sequence[Any]],
        stage2_task_refinements: Optional[Sequence[Any]],
        stage2_tags: Optional[Sequence[Any]],
    ) -> Dict[str, float]:
        coverage = float(belief_state.state_vector.get("evidence_coverage", 0.0))
        confidence = float(
            belief_state.state_vector.get("evidence_confidence_mean", 0.0)
        )
        disagreement = float(
            belief_state.state_vector.get("evidence_disagreement_mean", 0.0)
        )
        tags = set(_strings(semantic_tags))
        relation_scale = min(len(relations), 8) / 8.0
        object_scale = min(len(objects), 6) / 6.0
        meta_scale = min(len(meta_nodes), 6) / 6.0
        capability_scores = {
            "object_memory": _clip01(
                0.25 + 0.25 * object_scale + 0.2 * coverage + 0.15 * confidence
            ),
            "relation_graph": _clip01(
                0.2
                + 0.3 * relation_scale
                + 0.15 * coverage
                + 0.1 * (1.0 - disagreement)
            ),
            "affordance_grounding": _clip01(
                0.2
                + 0.25 * min(sum(len(item.affordances) for item in objects), 8) / 8.0
                + 0.1 * float(any("affordance:" in tag for tag in tags))
                + 0.1 * confidence
            ),
            "risk_reasoning": _clip01(
                0.22
                + 0.25 * float(any(item.risk_tags for item in objects))
                + 0.12 * float("safety" in tags or "constraint:avoid_collision" in tags)
                + 0.1 * coverage
            ),
            "recovery_reasoning": _clip01(
                0.18
                + 0.3 * float("error_recovery" in tags or "mode:recovery" in tags)
                + 0.12 * confidence
                + 0.1 * coverage
            ),
            "meta_node_orchestration": _clip01(
                0.2 + 0.35 * meta_scale + 0.15 * float(bool(meta_nodes))
            ),
            "fusion_bridge": _clip01(
                0.15
                + 0.45 * float(bool(semantic_fusion_summary))
                + 0.15 * coverage
                + 0.1 * confidence
            ),
            "stage2_bridge": _clip01(
                0.12
                + 0.2 * float(bool(stage2_ontology_proposals))
                + 0.2 * float(bool(stage2_task_refinements))
                + 0.18 * float(bool(stage2_tags))
            ),
        }
        return dict(sorted(capability_scores.items(), key=lambda item: item[0]))

    def _build_topology(
        self,
        *,
        objects: Sequence[SemanticObjectState],
        relations: Sequence[SemanticRelationState],
        meta_nodes: Sequence[SemanticMetaNode],
        capability_scores: Mapping[str, Any],
    ) -> Dict[str, Any]:
        high_risk_objects = [item.object_id for item in objects if item.risk_tags]
        uncertain_objects = [
            item.object_id for item in objects if item.confidence < 0.55
        ]
        grounded_objects = [
            item.object_id for item in objects if item.object_id.startswith("track:")
        ]
        return {
            "object_count": len(objects),
            "grounded_track_object_count": len(grounded_objects),
            "relation_count": len(relations),
            "meta_node_count": len(meta_nodes),
            "high_risk_objects": high_risk_objects,
            "uncertain_objects": uncertain_objects,
            "active_capabilities": [
                key for key, value in capability_scores.items() if float(value) >= 0.5
            ],
        }

    def _build_risk_register(
        self,
        *,
        semantic_tags: Sequence[str],
        objects: Sequence[SemanticObjectState],
        constraint_set: Optional[Mapping[str, Any]],
        meta_nodes: Sequence[SemanticMetaNode],
    ) -> Dict[str, Any]:
        tags = set(_strings(semantic_tags))
        hard_bounds = (
            constraint_set.get("hard_bounds", {})
            if isinstance(constraint_set, Mapping)
            else {}
        )
        active_nodes = [
            item.node_type
            for item in meta_nodes
            if item.node_type in {"risk_triage", "recovery_router"}
        ]
        return {
            "risk_tags": sorted(
                {risk for item in objects for risk in item.risk_tags}
                | {
                    tag
                    for tag in tags
                    if tag.startswith("risk:") or tag.startswith("constraint:")
                }
            ),
            "constraint_keys": sorted(hard_bounds.keys()),
            "high_risk_objects": [item.object_id for item in objects if item.risk_tags],
            "active_meta_nodes": active_nodes,
        }

    def _build_functional_roles(
        self,
        *,
        semantic_tags: Sequence[str],
        hypotheses: Optional[Sequence[GovernedVideoHypothesis]],
        meta_nodes: Sequence[SemanticMetaNode],
        capability_scores: Mapping[str, Any],
        semantic_fusion_summary: Optional[Mapping[str, Any]],
        stage2_ontology_proposals: Optional[Sequence[Any]],
        stage2_task_refinements: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        return {
            "scene_memory": {
                "tags": [
                    tag
                    for tag in semantic_tags
                    if tag.startswith("object:") or tag.startswith("region:")
                ],
                "capability": float(capability_scores.get("object_memory", 0.0)),
            },
            "action_bridge": {
                "hypothesis_modes": [item.mode for item in (hypotheses or [])],
                "capability": float(capability_scores.get("affordance_grounding", 0.0)),
            },
            "runtime_bridge": {
                "has_semantic_fusion": bool(semantic_fusion_summary),
                "capability": float(capability_scores.get("fusion_bridge", 0.0)),
            },
            "stage2_bridge": {
                "ontology_count": len(stage2_ontology_proposals or []),
                "task_refinement_count": len(stage2_task_refinements or []),
                "capability": float(capability_scores.get("stage2_bridge", 0.0)),
            },
            "meta_nodes": {
                "active": [item.node_type for item in meta_nodes],
                "capability": float(
                    capability_scores.get("meta_node_orchestration", 0.0)
                ),
            },
        }


__all__ = [
    "SemanticMetaNode",
    "SemanticObjectState",
    "SemanticRelationState",
    "SemanticWorldModelBuilder",
    "SemanticWorldModelConfig",
    "SemanticWorldModelState",
]
