#!/usr/bin/env python3
"""
Stage 1 Pipeline: Real Video → Diffusion Planning → VLA Distillation → DataPackMeta

Connects:
1. Video references (real demonstrations)
2. Governed diffusion runtime (augmented-clip plans based on semantic tags)
3. VLA controller (skill plan generation)
4. DataPackMeta creation (for downstream RL training)

No GPU-backed video materialization is required for the planning path, but the
runtime now reports explicit provider truth instead of pretending a stub is a
real backend.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.diffusion import (
    DiffusionProposal,
    VideoDiffusionRuntime,
    VideoDiffusionRuntimeConfig,
)
from src.evidence import (
    EvidenceBus,
    EvidenceRecord,
    belief_state_from_evidence_bus,
    build_benchmark_gate_report,
    build_execution_preconditions,
    build_execution_work_order,
    collect_benchmark_gating_signals,
)
from src.evidence.scene_tracks_truth import scene_tracks_truth_from_metadata
from src.governance import governance_trace_sidecar_payload
from src.runtime import decision_ledger_sidecar_payload, event_spine_sidecar_payload
from src.evidence.teacher_trace import (
    TeacherTrace,
    build_teacher_provider_truth,
    save_teacher_trace_json,
)
from src.vla.transformer_planner import VLATransformerPlanner, VLAInput, VLAPlan
from src.vla.teacher_runtime import (
    TeacherActionEnvelope,
    TeacherAdapterContract,
    save_teacher_action_envelope_json,
    save_teacher_adapter_contract_json,
)
from src.valuation.datapack_schema import (
    DataPackMeta,
    ConditionProfile,
    ObjectiveProfile,
    GuidanceProfile,
    AttributionProfile,
)
from src.regal.gen_plausibility import RegalGenPlausibilityNode
from src.regal.base import RegalDecision
from src.semantic.runtime_backbone import (
    SemanticRuntimeBackbone,
    build_orchestrator_control_plane_context,
)
from src.world_model import GovernedVideoWorldModel, SemanticWorldModelBuilder
from src.world_model.governed_video_supervision import (
    build_governed_video_supervision_bundle,
)
from src.vision.reconstruction import (
    build_four_d_reconstruction_sidecar,
    build_reconstruction_grounding_report,
    save_four_d_reconstruction_sidecar,
    save_reconstruction_grounding_report,
)


SEMANTIC_KEYWORDS = {
    "drawer": ["drawer", "handle", "grasp", "open"],
    "vase": ["vase", "fragile", "avoid_collision", "safety"],
    "safety": ["safety", "avoid_collision"],
    "energy": ["energy_efficient"],
    "recover": ["error_recovery"],
    "error": ["error_recovery"],
    "precision": ["high_precision", "careful"],
    "fast": ["high_speed"],
}


def _metadata_dict(video_ref: Dict[str, Any]) -> Dict[str, Any]:
    metadata = video_ref.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _sensor_bundle_metadata(video_ref: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _metadata_dict(video_ref)
    if isinstance(video_ref.get("sensor_bundle"), dict):
        return dict(video_ref["sensor_bundle"])
    if isinstance(metadata.get("sensor_bundle"), dict):
        return dict(metadata["sensor_bundle"])

    camera_name = str(video_ref.get("camera") or metadata.get("camera") or "front")
    intrinsics_ref = (
        video_ref.get("intrinsics_ref")
        or video_ref.get("camera_intrinsics_ref")
        or metadata.get("intrinsics_ref")
        or metadata.get("camera_intrinsics_ref")
    )
    extrinsics_ref = (
        video_ref.get("extrinsics_ref")
        or video_ref.get("camera_extrinsics_ref")
        or metadata.get("extrinsics_ref")
        or metadata.get("camera_extrinsics_ref")
    )
    return {
        "cameras": [camera_name],
        "intrinsics": {camera_name: intrinsics_ref} if intrinsics_ref else {},
        "extrinsics": {camera_name: extrinsics_ref} if extrinsics_ref else {},
        "depth_unit": metadata.get("depth_unit", "unknown"),
    }


def _camera_calibration_class(video_ref: Dict[str, Any]) -> str:
    sensor_bundle = _sensor_bundle_metadata(video_ref)
    cameras = [str(camera) for camera in list(sensor_bundle.get("cameras", []) or [])]
    if not cameras:
        return "camera_missing"
    intrinsics = (
        sensor_bundle.get("intrinsics")
        if isinstance(sensor_bundle.get("intrinsics"), dict)
        else {}
    )
    extrinsics = (
        sensor_bundle.get("extrinsics")
        if isinstance(sensor_bundle.get("extrinsics"), dict)
        else {}
    )
    calibrated = sum(
        1 for camera in cameras if intrinsics.get(camera) and extrinsics.get(camera)
    )
    if calibrated == len(cameras):
        return "camera_calibrated"
    if calibrated > 0 or any(
        intrinsics.get(camera) or extrinsics.get(camera) for camera in cameras
    ):
        return "camera_partial"
    return "camera_missing"


def _scene_tracks_truth_payload(video_ref: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(_metadata_dict(video_ref))
    for key in (
        "scene_tracks_v1",
        "scene_tracks_path",
        "scene_tracks_npz",
        "scene_tracks",
    ):
        value = video_ref.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    return payload


def _runtime_field(video_ref: Dict[str, Any], *keys: str, default: str = "") -> str:
    metadata = _metadata_dict(video_ref)
    for key in keys:
        if video_ref.get(key):
            return str(video_ref.get(key))
        if metadata.get(key):
            return str(metadata.get(key))
    return str(default)


def _scene_tracks_backend(video_ref: Dict[str, Any]) -> str:
    explicit = _runtime_field(video_ref, "scene_tracks_backend")
    truth = scene_tracks_truth_from_metadata(
        _scene_tracks_truth_payload(video_ref),
        explicit_backend=explicit,
    )
    return str(truth.get("scene_tracks_backend", "") or "unavailable")


def _vision_backbone_selected(video_ref: Dict[str, Any]) -> str:
    return _runtime_field(
        video_ref,
        "vision_backbone_selected",
        "openvla_vision_backbone_selected",
        "teacher_runtime_vision_backbone_selected",
        default="unavailable",
    )


def _teacher_runtime_backend_selected(video_ref: Dict[str, Any]) -> str:
    return _runtime_field(
        video_ref,
        "teacher_runtime_backend_selected",
        "openvla_backend_selected",
        "teacher_backend_selected",
        default="real" if _teacher_runtime_live(video_ref) else "unavailable",
    )


def _semantic_grounding_mode(
    video_ref: Dict[str, Any], semantic_world_model: Optional[Any] = None
) -> str:
    explicit = _runtime_field(video_ref, "semantic_grounding_mode", "grounding_mode")
    if explicit:
        return explicit
    if semantic_world_model is not None:
        grounded_scene = getattr(semantic_world_model, "metadata", {}).get(
            "grounded_scene", {}
        )
        if isinstance(grounded_scene, dict) and grounded_scene.get("grounding_mode"):
            return str(grounded_scene["grounding_mode"])
    return (
        "non_heuristic"
        if _scene_tracks_backend(video_ref) == "real"
        else "heuristic_fallback"
    )


def _scene_tracks_non_stub(video_ref: Dict[str, Any]) -> bool:
    truth = scene_tracks_truth_from_metadata(
        _scene_tracks_truth_payload(video_ref),
        explicit_backend=_runtime_field(video_ref, "scene_tracks_backend"),
    )
    return bool(truth.get("scene_tracks_non_stub", False))


def _teacher_runtime_live(video_ref: Dict[str, Any]) -> bool:
    metadata = _metadata_dict(video_ref)
    explicit = metadata.get("future_training_signals", {})
    if isinstance(explicit, dict) and "teacher_runtime_live" in explicit:
        return bool(explicit["teacher_runtime_live"])
    return bool(
        video_ref.get("teacher_trace")
        or video_ref.get("teacher_trace_path")
        or metadata.get("teacher_trace")
        or metadata.get("teacher_trace_path")
        or video_ref.get("teacher_contract_path")
        or metadata.get("teacher_contract_path")
        or video_ref.get("teacher_action_path")
        or metadata.get("teacher_action_path")
        or video_ref.get("teacher_action_envelope_path")
        or metadata.get("teacher_action_envelope_path")
    )


def _teacher_manifest_payload(video_ref: Dict[str, Any], key: str) -> Any:
    metadata = _metadata_dict(video_ref)
    return video_ref.get(key) or metadata.get(key)


def _teacher_action_payload(video_ref: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("teacher_action", "teacher_action_payload", "vla_action"):
        payload = _teacher_manifest_payload(video_ref, key)
        if isinstance(payload, dict):
            return dict(payload)
    envelope_payload = _teacher_manifest_payload(video_ref, "teacher_action_envelope")
    if isinstance(envelope_payload, dict):
        action = envelope_payload.get("action")
        if isinstance(action, dict):
            payload = dict(action)
            for key in ("confidence", "available", "vla_available", "failure_mode"):
                if key in envelope_payload and key not in payload:
                    payload[key] = envelope_payload[key]
            return payload
    return {}


def _load_manifest_teacher_trace(video_ref: Dict[str, Any]) -> Optional[TeacherTrace]:
    trace_payload = _teacher_manifest_payload(video_ref, "teacher_trace")
    if isinstance(trace_payload, dict):
        try:
            return TeacherTrace.from_dict(trace_payload)
        except Exception:
            return None
    trace_path = _teacher_manifest_payload(video_ref, "teacher_trace_path")
    if trace_path:
        try:
            return TeacherTrace.from_dict(
                json.loads(Path(str(trace_path)).read_text(encoding="utf-8"))
            )
        except Exception:
            return None
    return None


def _stage1_teacher_runtime_artifacts(
    video_ref: Dict[str, Any],
    semantic_tags: List[str],
) -> Dict[str, Any]:
    """Build explicit Stage-1 teacher artifacts without invoking a GPU teacher."""

    metadata = _metadata_dict(video_ref)
    teacher_id = str(_teacher_manifest_payload(video_ref, "teacher_id") or "openvla")
    model_name = str(
        _teacher_manifest_payload(video_ref, "teacher_model_name")
        or _teacher_manifest_payload(video_ref, "teacher_model")
        or "openvla/openvla-7b"
    )
    instruction = str(
        video_ref.get("instruction")
        or metadata.get("instruction")
        or "Execute the task safely."
    )
    backend_selected = _teacher_runtime_backend_selected(video_ref)
    backend_policy = str(
        _teacher_manifest_payload(video_ref, "teacher_backend_policy") or "manifest"
    )
    vision_backbone_selected = _vision_backbone_selected(video_ref)
    action_payload = _teacher_action_payload(video_ref)
    manifest_trace = _load_manifest_teacher_trace(video_ref)
    available = bool(
        (action_payload or manifest_trace is not None)
        and backend_selected == "real"
        and bool(metadata.get("teacher_runtime_available", True))
    )
    failure_reason = str(
        _teacher_manifest_payload(video_ref, "teacher_failure_reason")
        or _teacher_manifest_payload(video_ref, "teacher_availability_reason")
        or ("" if available else "teacher_action_missing")
    )
    confidence = float(
        action_payload.get("confidence", 0.0)
        or (
            manifest_trace.summary.get("teacher_confidence_mean", 0.0)
            if manifest_trace
            else 0.0
        )
    )
    provider_truth = build_teacher_provider_truth(
        provider_id=teacher_id,
        provider_name=model_name,
        available=available,
        backend_selected=backend_selected
        if backend_selected
        else ("real" if available else "unavailable"),
        fallback_mode=failure_reason,
        confidence=confidence if available else 0.0,
        metadata={
            "backend_policy": backend_policy,
            "vision_backbone_selected": vision_backbone_selected,
            "failure_reason": failure_reason,
            "source": "stage1_manifest",
        },
    )
    contract = TeacherAdapterContract(
        teacher_id=teacher_id,
        model_name=model_name,
        modality="action_semantics",
        advisory_only=True,
        available=available,
        metadata={
            "source": "stage1_manifest",
            "backend_selected": backend_selected
            if backend_selected
            else ("real" if available else "unavailable"),
            "backend_policy": backend_policy,
            "vision_backbone_selected": vision_backbone_selected,
            "availability_reason": failure_reason,
            "instruction": instruction,
            "manifest_teacher_trace_present": manifest_trace is not None,
            "manifest_teacher_action_present": bool(action_payload),
        },
        provider_truth=provider_truth,
    )
    if available:
        envelope_payload = dict(action_payload)
        envelope_payload.setdefault("vla_available", True)
        envelope_payload.setdefault(
            "confidence", confidence if confidence > 0.0 else 0.35
        )
        semantic_hint_tags = sorted(
            {
                str(tag)
                for tag in list(semantic_tags)
                + list(metadata.get("teacher_semantic_tags", []) or [])
                + list(metadata.get("semantic_tags", []) or [])
            }
            | {"teacher:available"}
        )
        envelope = TeacherActionEnvelope(
            teacher_id=teacher_id,
            model_name=model_name,
            instruction=instruction,
            available=True,
            action=envelope_payload,
            confidence=float(envelope_payload.get("confidence", 0.35)),
            failure_mode="",
            semantic_tags=semantic_hint_tags,
            object_refs=[str(value) for value in metadata.get("object_refs", []) or []],
            affordance_hints=[
                str(value) for value in metadata.get("affordance_hints", []) or []
            ],
            risk_hints=[str(value) for value in metadata.get("risk_hints", []) or []],
            provenance={
                "contract_id": contract.contract_id,
                "source": "stage1_manifest",
            },
            metadata={
                "backend_selected": backend_selected,
                "backend_policy": backend_policy,
                "vision_backbone_selected": vision_backbone_selected,
                "source": "stage1_manifest",
            },
            provider_truth=provider_truth,
        )
    else:
        envelope = TeacherActionEnvelope.unavailable(
            teacher_id=teacher_id,
            model_name=model_name,
            instruction=instruction,
            failure_mode=failure_reason,
            metadata={
                "contract_id": contract.contract_id,
                "backend_selected": backend_selected or "unavailable",
                "backend_policy": backend_policy,
                "vision_backbone_selected": vision_backbone_selected,
                "failure_reason": failure_reason,
                "source": "stage1_manifest",
            },
        )

    if manifest_trace is not None:
        trace = manifest_trace
    else:
        trace = TeacherTrace.from_vla_action(
            episode_id=str(video_ref["episode_id"]),
            instruction=instruction,
            semantic_tags=sorted(set(semantic_tags + list(envelope.semantic_tags))),
            action=envelope.to_vla_payload(),
            teacher_id=teacher_id,
            timestamp=str(video_ref.get("timestamp", "")),
            availability_reason=envelope.failure_mode,
            provider_truth=envelope.provider_truth,
        )
    return {"contract": contract, "envelope": envelope, "trace": trace}


def _future_training_signals(
    video_ref: Dict[str, Any],
    semantic_world_model: Any,
    sidecar_paths: Dict[str, Any],
    benchmark_gate: Optional[Any] = None,
) -> Dict[str, bool]:
    metadata = _metadata_dict(video_ref)
    explicit = metadata.get("future_training_signals", {})
    if not isinstance(explicit, dict):
        explicit = {}
    scene_tracks_truth = scene_tracks_truth_from_metadata(
        _scene_tracks_truth_payload(video_ref),
        explicit_backend=_runtime_field(video_ref, "scene_tracks_backend"),
    )
    topology = getattr(semantic_world_model, "topology", {}) or {}
    grounded_track_count = 0
    if isinstance(topology, dict):
        grounded_track_count = int(topology.get("grounded_track_object_count", 0) or 0)
    benchmark_signals = collect_benchmark_gating_signals(
        _stage1_benchmark_metadata(video_ref, semantic_world_model)
    )
    reconstruction_report = _load_json_sidecar(
        sidecar_paths.get("reconstruction_grounding_report_path")
    )
    derived = {
        "replay_roundtrip_complete": False,
        "promotion_trace_complete": all(
            sidecar_paths.get(key)
            for key in (
                "event_spine_path",
                "decision_ledger_path",
                "governance_trace_path",
                "counterfactual_eval_path",
                "value_target_pack_path",
            )
        ),
        "teacher_runtime_live": bool(
            _teacher_runtime_live(video_ref)
            or sidecar_paths.get("teacher_trace_path")
            or sidecar_paths.get("teacher_contract_path")
            or sidecar_paths.get("teacher_action_path")
        ),
        "teacher_runtime_contract_complete": bool(
            sidecar_paths.get("teacher_trace_path")
            and sidecar_paths.get("teacher_contract_path")
            and sidecar_paths.get("teacher_action_path")
        ),
        "scene_tracks_non_stub": bool(
            scene_tracks_truth.get("scene_tracks_non_stub", False)
        ),
        "scene_tracks_training_eligible": bool(
            scene_tracks_truth.get("scene_tracks_training_eligible", False)
        ),
        "semantic_grounding_non_heuristic": bool(
            scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
        ),
        "semantic_memory_grounded": grounded_track_count > 0,
        "reconstruction_calibrated": reconstruction_report.get("calibration_class")
        == "camera_calibrated",
        "reconstruction_real_grounded": reconstruction_report.get("grounding_class")
        == "real_scene_tracks_joined",
        "reconstruction_training_eligible": bool(
            reconstruction_report.get("training_eligible", False)
        ),
        "benchmark_gate_ready": bool(getattr(benchmark_gate, "ready", False))
        if benchmark_gate is not None
        else False,
        "budget_settlement_live": False,
    }
    derived.update(
        {
            str(key): bool(value)
            for key, value in benchmark_signals.items()
            if isinstance(value, bool)
        }
    )
    derived.update(
        {
            str(key): bool(value)
            for key, value in explicit.items()
            if str(key)
            not in {
                "scene_tracks_non_stub",
                "scene_tracks_training_eligible",
                "semantic_grounding_non_heuristic",
                "semantic_grounding_ready",
            }
        }
    )
    return dict(sorted(derived.items()))


def _load_json_sidecar(path_value: Any) -> Dict[str, Any]:
    if not path_value:
        return {}
    try:
        payload = json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _future_training_artifacts(video_ref: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _metadata_dict(video_ref)
    explicit = metadata.get("future_training_artifacts", {})
    if not isinstance(explicit, dict):
        explicit = {}
    artifacts = {
        str(key): value
        for key, value in explicit.items()
        if value not in (None, "", [], {})
    }
    if metadata.get("training_runtime_manifest"):
        artifacts["training_runtime_manifest"] = metadata["training_runtime_manifest"]
    if metadata.get("promotion_ledger_ref"):
        artifacts["promotion_ledger_ref"] = metadata["promotion_ledger_ref"]
    return dict(sorted(artifacts.items()))


def _stage1_benchmark_metadata(
    video_ref: Dict[str, Any], semantic_world_model: Any
) -> Dict[str, Any]:
    grounded_scene = {}
    if semantic_world_model is not None:
        grounded_scene = dict(
            getattr(semantic_world_model, "metadata", {}).get("grounded_scene", {})
            or {}
        )
    calibration_class = _camera_calibration_class(video_ref)
    return {
        "scene_tracks_backend": _scene_tracks_backend(video_ref),
        "teacher_runtime_backend_selected": _teacher_runtime_backend_selected(
            video_ref
        ),
        "vision_backbone_selected": _vision_backbone_selected(video_ref),
        "semantic_grounding_mode": _semantic_grounding_mode(
            video_ref, semantic_world_model
        ),
        "reconstruction_calibrated": calibration_class == "camera_calibrated",
        "reconstruction_calibration_class": calibration_class,
        "semantic_memory_grounded": bool(
            grounded_scene.get("grounding_ready", False)
            or int(
                getattr(semantic_world_model, "topology", {}).get(
                    "grounded_track_object_count", 0
                )
                or 0
            )
            > 0
        ),
        "grounded_track_object_count": int(
            getattr(semantic_world_model, "topology", {}).get(
                "grounded_track_object_count", 0
            )
            or 0
        ),
        "semantic_world_model_summary": {
            "topology": dict(getattr(semantic_world_model, "topology", {}) or {}),
            "grounded_track_object_count": int(
                getattr(semantic_world_model, "topology", {}).get(
                    "grounded_track_object_count", 0
                )
                or 0
            ),
        },
    }


def build_stage1_benchmark_gate(
    video_ref: Dict[str, Any], semantic_world_model: Any
) -> Any:
    return build_benchmark_gate_report(
        subject_id=str(video_ref.get("episode_id", "")),
        subject_kind="stage1_video_diffusion_benchmark_gate",
        metadata=_stage1_benchmark_metadata(video_ref, semantic_world_model),
        require_real_scene_tracks=True,
        require_teacher_runtime=False,
        require_vision_backbone=True,
        require_camera_calibration=True,
    )


def simulate_real_video_reference(index: int = 0) -> Dict[str, Any]:
    """
    Deterministic fallback when no real video manifest is supplied.

    In production, callers should pass a manifest of real video references.
    """
    episode_id = f"real_demo_sim_{index:03d}"
    duration_s = 12.0 + 2.5 * float(index % 4)
    success = (index % 5) != 4
    task_type = "drawer_vase" if index % 2 == 0 else "workcell_pick_place"

    return {
        "episode_id": episode_id,
        "video_path": f"/data/demonstrations/{episode_id}.mp4",
        "depth_path": f"/data/demonstrations/{episode_id}_depth.npy",
        "timestamp": float(1_700_000_000 + index),
        "task_type": task_type,
        "demonstrator": "human_expert",
        "source_type": "simulated_reference",
        "instruction": (
            "Open the drawer without hitting the vase."
            if task_type == "drawer_vase"
            else "Move the part safely across the bench."
        ),
        "metadata": {
            "duration_s": duration_s,
            "success": success,
            "num_frames": int(duration_s * 30.0),
            "scene_tracks_backend": "unavailable",
            "vision_backbone_selected": "unavailable",
            "teacher_runtime_backend_selected": "unavailable",
            "semantic_grounding_mode": "heuristic_fallback",
            "future_training_signals": {
                "scene_tracks_non_stub": False,
                "teacher_runtime_live": False,
            },
        },
    }


def _normalize_video_reference(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    episode_id = str(record.get("episode_id") or f"video_manifest_{index:03d}")
    video_path = str(record.get("video_path") or record.get("rgb_video_path") or "")
    task_type = str(
        record.get("task_type") or metadata.get("task_type") or "drawer_vase"
    )
    instruction = str(record.get("instruction") or metadata.get("instruction") or "")
    normalized = {
        "episode_id": episode_id,
        "video_path": video_path,
        "depth_path": record.get("depth_path"),
        "timestamp": float(record.get("timestamp", time.time())),
        "task_type": task_type,
        "demonstrator": str(record.get("demonstrator", "unknown")),
        "instruction": instruction,
        "source_type": str(record.get("source_type", "video_manifest")),
        "metadata": metadata,
    }
    passthrough_keys = (
        "camera",
        "scene_tracks_v1",
        "scene_tracks_path",
        "scene_tracks_npz",
        "teacher_trace",
        "teacher_trace_path",
        "teacher_action",
        "teacher_action_payload",
        "teacher_action_envelope",
        "teacher_action_path",
        "teacher_action_envelope_path",
        "teacher_contract_path",
        "teacher_id",
        "teacher_model",
        "teacher_model_name",
        "teacher_backend_policy",
        "teacher_failure_reason",
        "teacher_availability_reason",
        "vla_semantic_evidence",
        "vla_semantic_evidence_path",
        "sensor_bundle",
        "intrinsics_ref",
        "extrinsics_ref",
        "camera_intrinsics_ref",
        "camera_extrinsics_ref",
        "calibration_ref",
        "calibration_path",
    )
    for key in passthrough_keys:
        if key in record:
            normalized[key] = record[key]
    if "semantic_tags" in record and isinstance(record["semantic_tags"], list):
        normalized["metadata"] = dict(
            metadata, semantic_tags=list(record["semantic_tags"])
        )
    return normalized


def load_video_references(
    num_videos: int, manifest_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    if manifest_path is None:
        return [simulate_real_video_reference(index=i) for i in range(num_videos)]

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Video manifest not found: {path}")
    if path.suffix.lower() == ".jsonl":
        raw_records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            raw_records = payload
        elif isinstance(payload, dict):
            raw_records = payload.get("videos", [])
        else:
            raw_records = []

    references = [
        _normalize_video_reference(record, index=i)
        for i, record in enumerate(raw_records[:num_videos])
        if isinstance(record, dict)
    ]
    if len(references) < num_videos:
        references.extend(
            simulate_real_video_reference(index=len(references) + i)
            for i in range(num_videos - len(references))
        )
    return references[:num_videos]


def _tokenize_video_reference(video_ref: Dict[str, Any]) -> List[str]:
    fields = [
        str(video_ref.get("task_type", "")),
        str(video_ref.get("instruction", "")),
        str(video_ref.get("video_path", "")),
        str(video_ref.get("demonstrator", "")),
    ]
    metadata = video_ref.get("metadata", {})
    if isinstance(metadata, dict):
        fields.extend(str(value) for value in metadata.values())
        fields.extend(str(tag) for tag in metadata.get("semantic_tags", []) or [])
    tokens: List[str] = []
    for field in fields:
        cleaned = field.replace("/", " ").replace("_", " ").replace("-", " ").lower()
        tokens.extend(part for part in cleaned.split() if part)
    return tokens


def extract_semantic_tags_from_video(video_ref: Dict[str, Any]) -> List[str]:
    """
    Extract semantic seed tags from manifest metadata and instructions.

    This remains deterministic and lightweight, but it now emits a small
    object/affordance/risk vocabulary that can seed a semantic world model
    rather than only a flat keyword list.
    """
    builder = SemanticWorldModelBuilder()
    base_tags = set(builder.infer_seed_tags(video_ref))
    tokens = _tokenize_video_reference(video_ref)
    for token in tokens:
        for key, tags in SEMANTIC_KEYWORDS.items():
            if key in token:
                base_tags.update(tags)

    metadata = video_ref.get("metadata", {})
    if isinstance(metadata, dict):
        base_tags.update(str(tag) for tag in metadata.get("semantic_tags", []) or [])
        if bool(metadata.get("success")) is False:
            base_tags.add("error_recovery")
        duration_s = float(metadata.get("duration_s", 0.0) or 0.0)
        if duration_s >= 20.0:
            base_tags.add("long_horizon")
        if duration_s <= 8.0:
            base_tags.add("short_horizon")

    return sorted(base_tags)


def build_video_evidence(
    video_ref: Dict[str, Any],
    semantic_tags: List[str],
    objective_preset: str,
) -> tuple[EvidenceBus, Any]:
    metadata = video_ref.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    path_exists = Path(str(video_ref.get("video_path", ""))).exists()
    reference_conf = 0.85 if path_exists else 0.35
    semantic_conf = min(0.95, 0.35 + 0.04 * float(len(semantic_tags)))
    disagreement = 0.0 if metadata.get("success", True) else 0.2

    evidence_bus = EvidenceBus(
        [
            EvidenceRecord.from_components(
                episode_id=str(video_ref["episode_id"]),
                timestamp=str(video_ref.get("timestamp", "")),
                source=str(video_ref.get("source_type", "video_reference")),
                kind="video_reference",
                confidence=reference_conf,
                disagreement=0.0,
                metrics={
                    "video_path_exists": 1.0 if path_exists else 0.0,
                    "duration_s": float(metadata.get("duration_s", 0.0) or 0.0),
                },
                payload={"task_type": video_ref.get("task_type", "")},
                artifact_refs={"video_path": str(video_ref.get("video_path", ""))},
            ),
            EvidenceRecord.from_components(
                episode_id=str(video_ref["episode_id"]),
                timestamp=str(video_ref.get("timestamp", "")),
                source="stage1_semantic_extractor",
                kind="semantic_tags",
                confidence=semantic_conf,
                disagreement=disagreement,
                metrics={
                    "semantic_tag_count": float(len(semantic_tags)),
                    "teacher_confidence_mean": semantic_conf
                    if "human" in str(video_ref.get("demonstrator", "")).lower()
                    else 0.0,
                },
                payload={
                    "semantic_tags": semantic_tags,
                    "objective_preset": objective_preset,
                },
                artifact_refs={"video_path": str(video_ref.get("video_path", ""))},
            ),
        ]
    )
    belief_state = belief_state_from_evidence_bus(
        evidence_bus=evidence_bus,
        episode_id=str(video_ref["episode_id"]),
        timestamp=str(video_ref.get("timestamp", "")),
        semantic_tags=semantic_tags,
        extra_state={
            "geometry_quality": reference_conf,
            "semantic_quality": semantic_conf,
        },
        metadata={"source_type": video_ref.get("source_type", "video_reference")},
        artifact_refs={"video_path": str(video_ref.get("video_path", ""))},
    )
    return evidence_bus, belief_state


def build_stage1_constraint_set(
    semantic_tags: List[str],
    objective_preset: str,
    belief_state: Any,
) -> Dict[str, Any]:
    hard_bounds: Dict[str, Any] = {}
    if {"fragile", "avoid_collision", "safety"} & set(semantic_tags):
        hard_bounds["clearance_m"] = {"min": 0.05}
        hard_bounds["semantic_disagreement"] = {"max": 0.3}
    if objective_preset == "energy_saver":
        hard_bounds["energy_profile"] = {"target": 0.5}
    return {
        "hard_bounds": hard_bounds,
        "belief_state_id": getattr(belief_state, "belief_id", ""),
        "evidence_coverage": getattr(belief_state, "state_vector", {}).get(
            "evidence_coverage", 0.0
        ),
    }


def write_stage1_sidecars(
    output_dir: str,
    video_ref: Dict[str, Any],
    evidence_bus: EvidenceBus,
    belief_state: Any,
    snapshot: Any,
    hypotheses: List[Any],
    semantic_world_model: Any,
    semantic_snapshot: Any,
    orchestrator_advisory: Any,
    semantic_tags: List[str],
    objective_preset: str,
    constraint_set: Dict[str, Any],
    benchmark_gate: Optional[Any] = None,
    teacher_runtime_artifacts: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, str], Any]:
    sidecar_dir = Path(output_dir) / "governed_video"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    episode_id = str(video_ref["episode_id"])
    evidence_bus_path = sidecar_dir / f"{episode_id}_evidence_bus_v1.json"
    belief_state_path = sidecar_dir / f"{episode_id}_belief_state_v1.json"
    snapshot_path = sidecar_dir / f"{episode_id}_video_state_v1.json"
    hypotheses_path = sidecar_dir / f"{episode_id}_hypotheses_v1.json"
    semantic_world_model_path = (
        sidecar_dir / f"{episode_id}_semantic_world_model_v1.json"
    )
    semantic_snapshot_path = sidecar_dir / f"{episode_id}_semantic_snapshot_v1.json"
    orchestrator_advisory_path = (
        sidecar_dir / f"{episode_id}_orchestrator_advisory_v1.json"
    )
    control_plane_context_path = (
        sidecar_dir / f"{episode_id}_control_plane_context_v1.json"
    )
    evidence_bus_path.write_text(json.dumps(evidence_bus.to_dict(), indent=2))
    belief_state_path.write_text(json.dumps(belief_state.to_dict(), indent=2))
    snapshot_path.write_text(json.dumps(snapshot.to_dict(), indent=2))
    hypotheses_path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
            },
            indent=2,
        )
    )
    semantic_world_model_path.write_text(
        json.dumps(semantic_world_model.to_dict(), indent=2)
    )
    semantic_snapshot_path.write_text(json.dumps(semantic_snapshot.to_dict(), indent=2))
    orchestrator_advisory_path.write_text(
        json.dumps(orchestrator_advisory.to_json(), indent=2)
    )
    control_plane_context_path.write_text(
        json.dumps(
            build_orchestrator_control_plane_context(
                semantic_world_model=semantic_world_model,
                semantic_snapshot=semantic_snapshot,
                orchestrator_advisory=orchestrator_advisory,
                artifact_refs={
                    "semantic_world_model_path": str(semantic_world_model_path),
                    "semantic_snapshot_path": str(semantic_snapshot_path),
                    "orchestrator_advisory_path": str(orchestrator_advisory_path),
                },
            ),
            indent=2,
        )
    )
    sidecar_paths = {
        "evidence_bus_path": str(evidence_bus_path),
        "belief_state_path": str(belief_state_path),
        "video_state_path": str(snapshot_path),
        "hypotheses_path": str(hypotheses_path),
        "semantic_world_model_path": str(semantic_world_model_path),
        "semantic_snapshot_path": str(semantic_snapshot_path),
        "orchestrator_advisory_path": str(orchestrator_advisory_path),
        "control_plane_context_path": str(control_plane_context_path),
    }
    if teacher_runtime_artifacts:
        teacher_contract = teacher_runtime_artifacts.get("contract")
        teacher_envelope = teacher_runtime_artifacts.get("envelope")
        teacher_trace = teacher_runtime_artifacts.get("trace")
        teacher_contract_path = sidecar_dir / f"{episode_id}_teacher_contract_v1.json"
        teacher_action_path = (
            sidecar_dir / f"{episode_id}_teacher_action_envelope_v1.json"
        )
        teacher_trace_path = sidecar_dir / f"{episode_id}_teacher_trace_v1.json"
        if isinstance(teacher_contract, TeacherAdapterContract):
            save_teacher_adapter_contract_json(teacher_contract_path, teacher_contract)
            sidecar_paths["teacher_contract_path"] = str(teacher_contract_path)
        if isinstance(teacher_envelope, TeacherActionEnvelope):
            save_teacher_action_envelope_json(teacher_action_path, teacher_envelope)
            sidecar_paths["teacher_action_path"] = str(teacher_action_path)
            sidecar_paths["teacher_action_envelope_path"] = str(teacher_action_path)
        if isinstance(teacher_trace, TeacherTrace):
            save_teacher_trace_json(teacher_trace_path, teacher_trace)
            sidecar_paths["teacher_trace_path"] = str(teacher_trace_path)
    metadata = video_ref.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    frame_count = int(metadata.get("num_frames", 0) or 0)
    reconstruction_path = sidecar_dir / f"{episode_id}_reconstruction_sidecar_v1.json"
    reconstruction_grounding_report_path = (
        sidecar_dir / f"{episode_id}_reconstruction_grounding_report_v1.json"
    )
    sensor_bundle_meta = _sensor_bundle_metadata(video_ref)
    scene_tracks_ref = (
        video_ref.get("scene_tracks_path")
        or video_ref.get("scene_tracks_npz")
        or metadata.get("scene_tracks_path")
        or metadata.get("scene_tracks_npz")
    )
    if not scene_tracks_ref and (
        video_ref.get("scene_tracks_v1") or metadata.get("scene_tracks_v1")
    ):
        scene_tracks_ref = "inline:scene_tracks_v1"
    reconstruction_sidecar = build_four_d_reconstruction_sidecar(
        episode_id=episode_id,
        source_type=str(video_ref.get("source_type", "video_reference")),
        media_refs=[
            ref
            for ref in [video_ref.get("video_path"), video_ref.get("depth_path")]
            if ref
        ],
        sensor_bundle_meta=sensor_bundle_meta,
        frame_count=frame_count,
        frame_range=[0, max(0, frame_count - 1)] if frame_count else None,
        geometry_refs={
            "video_state_path": sidecar_paths["video_state_path"],
            "hypotheses_path": sidecar_paths["hypotheses_path"],
            "depth_path": str(video_ref.get("depth_path", "")),
            "scene_tracks_path": str(scene_tracks_ref or ""),
        },
        evidence_refs={
            "evidence_bus_path": sidecar_paths["evidence_bus_path"],
            "belief_state_path": sidecar_paths["belief_state_path"],
            "teacher_trace_path": sidecar_paths.get("teacher_trace_path", ""),
        },
        quality={
            "geometry_quality": float(
                belief_state.state_vector.get("geometry_quality", 0.0)
            ),
            "evidence_coverage": float(
                belief_state.state_vector.get("evidence_coverage", 0.0)
            ),
        },
        metadata={
            "task_type": str(video_ref.get("task_type", "")),
            "scene_tracks_backend": _scene_tracks_backend(video_ref),
            "semantic_grounding_mode": _semantic_grounding_mode(
                video_ref, semantic_world_model
            ),
            "vision_backbone_selected": _vision_backbone_selected(video_ref),
            "teacher_runtime_backend_selected": _teacher_runtime_backend_selected(
                video_ref
            ),
            "calibration_ref": str(
                video_ref.get("calibration_ref")
                or video_ref.get("calibration_path")
                or metadata.get("calibration_ref")
                or metadata.get("calibration_path")
                or ""
            ),
        },
    )
    save_four_d_reconstruction_sidecar(reconstruction_path, reconstruction_sidecar)
    sidecar_paths["reconstruction_sidecar_path"] = str(reconstruction_path)
    reconstruction_grounding_report = build_reconstruction_grounding_report(
        sidecar=reconstruction_sidecar,
        sidecar_path=reconstruction_path,
        scene_tracks_backend=_scene_tracks_backend(video_ref),
        semantic_grounding_mode=_semantic_grounding_mode(
            video_ref, semantic_world_model
        ),
        vision_backbone_selected=_vision_backbone_selected(video_ref),
        teacher_runtime_backend_selected=_teacher_runtime_backend_selected(video_ref),
        metadata={"task_type": str(video_ref.get("task_type", ""))},
    )
    save_reconstruction_grounding_report(
        reconstruction_grounding_report_path,
        reconstruction_grounding_report,
    )
    sidecar_paths["reconstruction_grounding_report_path"] = str(
        reconstruction_grounding_report_path
    )

    supervision_bundle = build_governed_video_supervision_bundle(
        run_id=f"stage1_governed_{episode_id}",
        video_ref=video_ref,
        semantic_tags=semantic_tags,
        belief_state=belief_state,
        snapshot=snapshot,
        hypotheses=hypotheses,
        objective_preset=objective_preset,
        constraint_set=constraint_set,
        sidecar_refs=sidecar_paths,
        value_ledger_path=sidecar_dir / "governed_video_value_ledger.jsonl",
    )
    runtime_packet_path = sidecar_dir / f"{episode_id}_runtime_packet_v1.json"
    pricing_tick_path = sidecar_dir / f"{episode_id}_pricing_tick_v1.json"
    branch_eval_path = sidecar_dir / f"{episode_id}_branch_evaluations_v1.json"
    event_spine_path = sidecar_dir / f"{episode_id}_event_spine_v1.json"
    decision_ledger_path = sidecar_dir / f"{episode_id}_decision_ledger_v1.json"
    governance_trace_path = sidecar_dir / f"{episode_id}_governance_trace_v1.json"
    counterfactual_eval_path = sidecar_dir / f"{episode_id}_counterfactual_eval_v1.json"
    value_target_pack_path = sidecar_dir / f"{episode_id}_value_target_pack_v1.json"
    value_ledger_receipt_path = (
        sidecar_dir / f"{episode_id}_value_ledger_receipt_v1.json"
    )
    benchmark_gate_path = sidecar_dir / f"{episode_id}_benchmark_gate_v1.json"

    runtime_packet_path.write_text(
        json.dumps(supervision_bundle.runtime_packet.to_dict(), indent=2)
    )
    pricing_tick_path.write_text(
        json.dumps(supervision_bundle.pricing_tick.to_dict(), indent=2)
    )
    branch_eval_path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "branch_evaluations": [
                    evaluation.to_dict()
                    for evaluation in supervision_bundle.branch_evaluations
                ],
            },
            indent=2,
        )
    )
    event_spine_path.write_text(
        json.dumps(
            event_spine_sidecar_payload(
                run_id=f"stage1_governed_{episode_id}",
                events=supervision_bundle.events,
            ),
            indent=2,
        )
    )
    decision_ledger_path.write_text(
        json.dumps(
            decision_ledger_sidecar_payload(
                run_id=f"stage1_governed_{episode_id}",
                decisions=supervision_bundle.decisions,
            ),
            indent=2,
        )
    )
    governance_trace_path.write_text(
        json.dumps(
            governance_trace_sidecar_payload(
                run_id=f"stage1_governed_{episode_id}",
                traces=supervision_bundle.governance_traces,
            ),
            indent=2,
        )
    )
    counterfactual_eval_path.write_text(
        json.dumps(supervision_bundle.counterfactual_eval.to_dict(), indent=2)
    )
    value_target_pack_path.write_text(
        json.dumps(supervision_bundle.value_target_pack.to_dict(), indent=2)
    )
    value_ledger_receipt_path.write_text(
        json.dumps(supervision_bundle.value_ledger_receipt.to_dict(), indent=2)
    )
    if benchmark_gate is not None:
        benchmark_gate_path.write_text(json.dumps(benchmark_gate.to_dict(), indent=2))
    sidecar_paths.update(
        {
            "runtime_packet_path": str(runtime_packet_path),
            "pricing_tick_path": str(pricing_tick_path),
            "branch_evaluations_path": str(branch_eval_path),
            "event_spine_path": str(event_spine_path),
            "decision_ledger_path": str(decision_ledger_path),
            "governance_trace_path": str(governance_trace_path),
            "counterfactual_eval_path": str(counterfactual_eval_path),
            "value_target_pack_path": str(value_target_pack_path),
            "value_ledger_receipt_path": str(value_ledger_receipt_path),
            "benchmark_gate_path": str(benchmark_gate_path)
            if benchmark_gate is not None
            else "",
        }
    )
    return sidecar_paths, supervision_bundle


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def generate_diffusion_proposals(
    video_ref: Dict[str, Any],
    semantic_tags: List[str],
    diffusion_runtime: VideoDiffusionRuntime,
    world_model: GovernedVideoWorldModel,
    belief_state: Any,
    objective_preset: str = "balanced",
    num_proposals: int = 3,
) -> tuple[List[DiffusionProposal], Any, List[Any], Dict[str, Any], Dict[str, Any]]:
    """
    Generate governed video hypotheses and render them into proposals.
    """
    constraint_set = build_stage1_constraint_set(
        semantic_tags, objective_preset, belief_state
    )
    snapshot = world_model.build_state_snapshot(
        episode_id=str(video_ref["episode_id"]),
        timestamp=str(video_ref.get("timestamp", "")),
        belief_state=belief_state,
        objective_preset=objective_preset,
        semantic_tags=semantic_tags,
        media_refs=[str(video_ref["video_path"])],
        artifact_refs={"video_path": str(video_ref["video_path"])},
        metadata={"task_type": video_ref.get("task_type", "")},
    )
    hypotheses = world_model.propose_hypotheses(
        snapshot=snapshot,
        constraint_set=constraint_set,
    )
    routing_context = {
        "routing_source": "governed_video_world_model",
        "objective_preset": objective_preset,
        "benchmark_gate_ready": _scene_tracks_backend(video_ref) == "real"
        and _vision_backbone_selected(video_ref) == "real",
        "scene_tracks_backend": _scene_tracks_backend(video_ref),
        "teacher_runtime_backend_selected": _teacher_runtime_backend_selected(
            video_ref
        ),
        "vision_backbone_selected": _vision_backbone_selected(video_ref),
        "semantic_grounding_mode": _semantic_grounding_mode(video_ref),
        "evidence_coverage": float(
            snapshot.state_features.get("evidence_coverage", 0.0)
        ),
        "semantic_disagreement": float(
            snapshot.state_features.get("evidence_disagreement_mean", 0.0)
        ),
        "constraint_pressure": float(
            len(dict(constraint_set.get("hard_bounds", {}) or {}))
        )
        / 6.0,
        "governed_hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
    }
    proposals = diffusion_runtime.propose_augmented_clips(
        episode_id=video_ref["episode_id"],
        media_refs=[video_ref["video_path"]],
        semantic_tags=semantic_tags,
        objective_preset=objective_preset,
        constraint_set=constraint_set,
        hypotheses=[hypothesis.to_dict() for hypothesis in hypotheses],
        routing_context=routing_context,
        num_proposals=num_proposals,
    )
    return proposals, snapshot, hypotheses, constraint_set, routing_context


def extract_vla_plan_from_proposal(
    proposal: DiffusionProposal,
    vla_planner: VLATransformerPlanner,
) -> VLAPlan:
    """
    Generate VLA skill plan based on diffusion proposal.

    The VLA planner takes the augmentation type and semantic tags
    to generate an appropriate skill sequence.
    """
    # Build instruction from proposal
    instruction = f"{proposal.augmentation_type} with "
    instruction += ", ".join(proposal.semantic_tags[:5])

    vla_input = VLAInput(
        instruction=instruction,
        # In production, would include actual visual features
    )

    plan = vla_planner.plan(vla_input)
    return plan


def create_datapack_from_pipeline(
    video_ref: Dict[str, Any],
    semantic_tags: List[str],
    diffusion_proposal: DiffusionProposal,
    vla_plan: VLAPlan,
    semantic_world_model: Any,
    orchestrator_advisory: Any,
    objective_preset: str = "balanced",
    benchmark_gate: Optional[Dict[str, Any]] = None,
    execution_work_order: Optional[Dict[str, Any]] = None,
) -> DataPackMeta:
    """
    Create DataPackMeta from Stage 1 pipeline outputs.

    This datapack can be used for downstream RL training.
    """
    # Determine objective vector from preset
    if objective_preset == "throughput":
        objective_vector = [2.0, 1.0, 0.5, 1.0, 0.0]
    elif objective_preset == "safety":
        objective_vector = [1.0, 1.0, 0.5, 3.0, 0.0]
    elif objective_preset == "energy_saver":
        objective_vector = [1.0, 1.0, 2.0, 1.0, 0.0]
    else:  # balanced
        objective_vector = [1.0, 1.0, 1.0, 1.0, 0.0]

    # Create profiles using actual schema
    condition_profile = ConditionProfile(
        task_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",  # Would be real robot backend
        world_id="stage1_world",
        vase_offset=(0.0, 0.0, 0.0),
        drawer_friction=0.3,
        lighting_profile="normal",
        occlusion_level=0.0,
        econ_preset="drawer_vase",
        price_per_unit=5.0,
        vase_break_cost=50.0,
        energy_price_kWh=diffusion_proposal.econ_context.get("energy_price_kWh", 0.12),
        objective_vector=objective_vector[:3],  # ConditionProfile uses 3-dim vector
        tags={
            "fragile": "fragile" in semantic_tags,
            "safety_critical": "safety" in semantic_tags,
            "source": "stage1_diffusion",
        },
    )

    objective_profile = ObjectiveProfile(
        env_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",
        task_type=video_ref.get("task_type", "unknown"),
        customer_segment=diffusion_proposal.econ_context.get(
            "customer_segment", "balanced"
        ),
        market_region="US",
        objective_vector=objective_vector,
        wage_human=diffusion_proposal.econ_context.get("wage_human", 18.0),
        energy_price_kWh=diffusion_proposal.econ_context.get("energy_price_kWh", 0.12),
    )

    # Determine quality and main driver from semantic tags and objective
    benchmark_ready = bool(dict(benchmark_gate or {}).get("ready", False))
    is_good = diffusion_proposal.estimated_novelty > 0.4 and benchmark_ready
    if not benchmark_ready:
        quality_label = "shadow_only"
    else:
        quality_label = (
            "high_value" if diffusion_proposal.estimated_novelty > 0.6 else "medium"
        )

    # Determine main driver from semantic tags
    if "safety" in semantic_tags:
        main_driver = "safety_margin"
    elif "energy_efficient" in semantic_tags:
        main_driver = "energy_efficiency"
    elif "high_speed" in semantic_tags or "throughput" in objective_preset:
        main_driver = "throughput_gain"
    else:
        main_driver = "error_reduction"

    guidance_profile = GuidanceProfile(
        is_good=is_good,
        quality_label=quality_label,
        env_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",
        task_type=video_ref.get("task_type", "drawer_vase"),
        customer_segment=diffusion_proposal.econ_context.get(
            "customer_segment", "balanced"
        ),
        objective_vector=objective_vector,
        main_driver=main_driver,
        delta_mpl=diffusion_proposal.estimated_novelty * 5.0,
        delta_error=-0.01 if "safety" in semantic_tags else 0.0,
        delta_energy_Wh=-0.5 if "energy_efficient" in semantic_tags else 0.0,
        delta_J=diffusion_proposal.estimated_novelty * 2.0,
        semantic_tags=semantic_tags,
        orchestrator_plan_id=None,
        orchestrator_step_index=None,
    )

    # Attribution based on diffusion novelty and VLA confidence
    # Get mean confidence, handling potentially empty lists
    vla_conf = vla_plan.confidence
    if isinstance(vla_conf, (list, np.ndarray)) and len(vla_conf) > 0:
        mean_conf = float(np.mean(vla_conf))
    else:
        mean_conf = 0.5  # Default

    tier = (
        0
        if not benchmark_ready
        else 2
        if diffusion_proposal.estimated_novelty > 0.6
        else 1
    )

    econ_semantic_tags = list(semantic_tags)
    econ_semantic_tags.append(f"objective:{objective_preset}")
    econ_semantic_tags.append(f"aug:{diffusion_proposal.augmentation_type}")
    econ_semantic_tags.append(f"routing:{diffusion_proposal.routing_source}")
    if benchmark_ready:
        econ_semantic_tags.append("benchmark:ready")
    else:
        econ_semantic_tags.append("benchmark:shadow_only")
    for capability, score in sorted(
        getattr(semantic_world_model, "capability_scores", {}).items(),
        key=lambda item: item[0],
    ):
        if float(score) >= 0.45:
            econ_semantic_tags.append(f"capability:{capability}")
    for node_type, score in sorted(
        getattr(orchestrator_advisory, "meta_node_weights", {}).items(),
        key=lambda item: item[0],
    ):
        if float(score) >= 0.4:
            econ_semantic_tags.append(f"meta_node:{node_type}")
    semantic_quality = float(max(0.0, min(1.0, diffusion_proposal.estimated_novelty)))
    if not benchmark_ready:
        semantic_quality = float(min(semantic_quality, 0.35))

    effective_trust = mean_conf
    if not benchmark_ready:
        effective_trust = min(effective_trust, 0.45)

    attribution_profile = AttributionProfile(
        env_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",
        delta_mpl=diffusion_proposal.estimated_novelty
        * 5.0,  # Novelty correlates with learning
        delta_error=-0.01 if "safety" in semantic_tags else 0.0,
        delta_J=diffusion_proposal.estimated_novelty * 2.0,
        trust_score=effective_trust,
        w_econ=0.8,
        tier=tier,
    )

    datapack = DataPackMeta(
        pack_id=f"stage1_{video_ref['episode_id']}_{int(time.time())}",
        task_name=video_ref.get("task_type", "drawer_vase"),
        env_type="pybullet",
        schema_version="2.0-stage1",
        condition=condition_profile,
        objective_profile=objective_profile,
        guidance_profile=guidance_profile,
        attribution=attribution_profile,
        semantic_tags=semantic_tags
        + [f"vla_skill_{s}" for s in vla_plan.skill_sequence[:3]],
        econ_semantic_tags=econ_semantic_tags,
        semantic_quality=semantic_quality,
        agent_profile={
            "policy": "stage1_vla",
            "source_type": "stage1_diffusion_vla",
            "semantic_world_model_id": getattr(
                semantic_world_model, "world_model_id", ""
            ),
            "meta_node_weights": getattr(
                orchestrator_advisory, "meta_node_weights", {}
            ),
            "diffusion_routing_source": diffusion_proposal.routing_source,
            "diffusion_routing_score": diffusion_proposal.routing_score,
            "diffusion_backend_selected": diffusion_proposal.diffusion_backend_selected,
            "diffusion_backend_policy": diffusion_proposal.diffusion_backend_policy,
            "diffusion_materialization_mode": diffusion_proposal.diffusion_materialization_mode,
            "benchmark_admission_mode": (
                dict(execution_work_order or {}).get(
                    "recommended_mode", "shadow_stage1_datapack"
                )
            ),
        },
        signal_bundle={
            "semantic_world_model": {
                "world_model_id": getattr(semantic_world_model, "world_model_id", ""),
                "topology": getattr(semantic_world_model, "topology", {}),
                "capability_scores": getattr(
                    semantic_world_model, "capability_scores", {}
                ),
            },
            "meta_nodes": getattr(orchestrator_advisory, "meta_node_weights", {}),
            "benchmark_gate": dict(benchmark_gate or {}),
            "diffusion_routing": {
                "routing_source": diffusion_proposal.routing_source,
                "routing_score": diffusion_proposal.routing_score,
                "source_hypothesis_id": diffusion_proposal.source_hypothesis_id,
                "benchmark_gate_ready": diffusion_proposal.benchmark_gate_ready,
                "diffusion_backend_selected": diffusion_proposal.diffusion_backend_selected,
                "diffusion_backend_policy": diffusion_proposal.diffusion_backend_policy,
                "diffusion_materialization_mode": diffusion_proposal.diffusion_materialization_mode,
                "diffusion_provider_truth": dict(
                    diffusion_proposal.diffusion_provider_truth or {}
                ),
            },
        },
    )

    return datapack


def run_stage1_pipeline(
    num_videos: int = 5,
    proposals_per_video: int = 3,
    objective_preset: str = "balanced",
    output_dir: str = "results/stage1_pipeline",
    video_manifest: Optional[str] = None,
    plausibility_node: Optional[RegalGenPlausibilityNode] = None,
    diffusion_backend_policy: str = "auto",
    diffusion_model_ref: Optional[str] = None,
    diffusion_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full Stage 1 pipeline.

    Returns:
        results: Dict with pipeline outputs and statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    diffusion_runtime = VideoDiffusionRuntime(
        VideoDiffusionRuntimeConfig(
            model_ref=str(diffusion_model_ref or ""),
            device=str(diffusion_device or "cuda"),
            backend_policy=str(diffusion_backend_policy or "auto"),
        )
    )
    vla_planner = VLATransformerPlanner()
    plausibility_node = plausibility_node or RegalGenPlausibilityNode()
    world_model = GovernedVideoWorldModel()
    semantic_world_model_builder = SemanticWorldModelBuilder()
    semantic_backbone = SemanticRuntimeBackbone({"write_to_file": False})

    all_datapacks = []
    all_proposals = []
    all_plans = []
    pipeline_log = []
    admission_records: List[Dict[str, Any]] = []
    generated_proposal_count = 0
    video_refs = load_video_references(
        num_videos=num_videos, manifest_path=video_manifest
    )

    print(f"Running Stage 1 pipeline with {len(video_refs)} videos...")

    for i, video_ref in enumerate(video_refs):
        print(f"\n--- Video {i + 1}/{len(video_refs)} ---")

        # Step 1: Load real or manifest-backed video reference
        print(f"  Video: {video_ref['episode_id']}")

        # Step 2: Extract semantic tags
        semantic_tags = extract_semantic_tags_from_video(video_ref)
        print(f"  Tags: {semantic_tags[:5]}...")

        # Step 3: Build evidence and belief state
        evidence_bus, belief_state = build_video_evidence(
            video_ref, semantic_tags, objective_preset
        )
        teacher_runtime_artifacts = _stage1_teacher_runtime_artifacts(
            video_ref, semantic_tags
        )

        # Step 4: Generate governed hypotheses, then render proposals
        proposals, snapshot, hypotheses, constraint_set, routing_context = (
            generate_diffusion_proposals(
                video_ref,
                semantic_tags,
                diffusion_runtime,
                world_model,
                belief_state,
                objective_preset,
                proposals_per_video,
            )
        )
        semantic_world_model = semantic_world_model_builder.build_from_stage1(
            video_ref=video_ref,
            belief_state=belief_state,
            video_state_snapshot=snapshot,
            hypotheses=hypotheses,
            constraint_set=constraint_set,
            objective_preset=objective_preset,
            semantic_tags=semantic_tags,
            scene_tracks_payload=(
                video_ref.get("scene_tracks_v1")
                or video_ref.get("scene_tracks_path")
                or video_ref.get("scene_tracks_npz")
                or video_ref.get("metadata", {}).get("scene_tracks_v1")
                or video_ref.get("metadata", {}).get("scene_tracks_path")
                or video_ref.get("metadata", {}).get("scene_tracks_npz")
            ),
            teacher_trace=(
                teacher_runtime_artifacts.get("trace")
                or video_ref.get("teacher_trace")
                or video_ref.get("teacher_trace_path")
                or video_ref.get("metadata", {}).get("teacher_trace")
                or video_ref.get("metadata", {}).get("teacher_trace_path")
            ),
            vla_semantic_evidence=(
                video_ref.get("vla_semantic_evidence")
                or video_ref.get("vla_semantic_evidence_path")
                or video_ref.get("metadata", {}).get("vla_semantic_evidence")
                or video_ref.get("metadata", {}).get("vla_semantic_evidence_path")
            ),
            artifact_refs={"video_path": str(video_ref.get("video_path", ""))},
        )
        semantic_backbone_result = semantic_backbone.build(
            task_id=str(video_ref.get("task_type", video_ref.get("episode_id", ""))),
            objective_preset=objective_preset,
            semantic_world_model=semantic_world_model,
            runtime_metrics={
                "avg_energy_cost": 0.0,
                "avg_error_rate": 0.0
                if video_ref.get("metadata", {}).get("success", True)
                else 1.0,
                "avg_wage_parity": 1.0,
                "avg_mpl_units_per_hour": float(len(semantic_tags)),
                "expected_delta_mpl": float(
                    max(
                        (proposal.estimated_novelty for proposal in proposals),
                        default=0.0,
                    )
                ),
                "recovery_segment_fraction": 1.0
                if {"error_recovery", "mode:recovery"} & set(semantic_tags)
                else 0.0,
            },
            frontier_episodes=[str(video_ref.get("episode_id", ""))],
            metadata={
                "source_stage": "stage1",
                "video_episode_id": str(video_ref.get("episode_id", "")),
            },
            backends=["governed_video"],
        )
        generated_proposal_count += len(proposals)
        benchmark_gate = build_stage1_benchmark_gate(video_ref, semantic_world_model)
        sidecar_paths, supervision_bundle = write_stage1_sidecars(
            output_dir,
            video_ref,
            evidence_bus,
            belief_state,
            snapshot,
            hypotheses,
            semantic_world_model,
            semantic_backbone_result.semantic_snapshot,
            semantic_backbone_result.orchestrator_advisory,
            semantic_tags,
            objective_preset,
            constraint_set,
            benchmark_gate=benchmark_gate,
            teacher_runtime_artifacts=teacher_runtime_artifacts,
        )
        print(f"  Generated {len(proposals)} diffusion proposals")
        future_training_signals = _future_training_signals(
            video_ref,
            semantic_world_model,
            sidecar_paths,
            benchmark_gate=benchmark_gate,
        )
        future_training_artifacts = _future_training_artifacts(video_ref)
        if sidecar_paths.get("benchmark_gate_path"):
            future_training_artifacts["benchmark_gate_path"] = sidecar_paths[
                "benchmark_gate_path"
            ]

        # Step 5: For each proposal, generate VLA plan and create datapack
        for j, proposal in enumerate(proposals):
            print(f"    Proposal {j + 1}: {proposal.augmentation_type}")

            plausibility_context = {
                "map_first_quality_score": max(
                    0.0,
                    min(
                        1.0,
                        belief_state.state_vector.get(
                            "geometry_quality", proposal.confidence
                        ),
                    ),
                ),
                "semantic_disagreement_vla_vs_map": max(
                    0.0,
                    min(
                        1.0,
                        belief_state.state_vector.get(
                            "evidence_disagreement_mean", 1.0 - proposal.confidence
                        ),
                    ),
                ),
                "vla_evidence_coverage": max(
                    0.0,
                    min(
                        1.0,
                        belief_state.state_vector.get(
                            "evidence_coverage", len(semantic_tags) / 12.0
                        ),
                    ),
                ),
            }
            plausibility_report = plausibility_node.evaluate(plausibility_context)
            execution_preconditions = build_execution_preconditions(
                subject_id=proposal.proposal_id,
                subject_kind="governed_video_proposal",
                artifact_refs={
                    **sidecar_paths,
                    **future_training_artifacts,
                    "proposal_id": proposal.proposal_id,
                    "hypothesis_mode": proposal.augmentation_type,
                },
                required_artifact_refs=[
                    "belief_state_path",
                    "video_state_path",
                    "reconstruction_sidecar_path",
                    "reconstruction_grounding_report_path",
                    "runtime_packet_path",
                    "governance_trace_path",
                    "counterfactual_eval_path",
                    "value_target_pack_path",
                    "teacher_contract_path",
                    "teacher_action_path",
                    "teacher_trace_path",
                    "event_spine_path",
                    "decision_ledger_path",
                ],
                signal_values={
                    **plausibility_context,
                    "estimated_novelty": float(proposal.estimated_novelty),
                    "diffusion_routing_score": float(proposal.routing_score),
                    **future_training_signals,
                },
                soft_boolean_signals={key: True for key in future_training_signals},
                soft_required_artifact_refs=list(future_training_artifacts.keys()),
                blocked_reasons=plausibility_report.reason_codes
                if plausibility_report.decision == RegalDecision.BLOCK
                else [],
                metadata={
                    "video_id": video_ref["episode_id"],
                    "counterfactual_eval_id": supervision_bundle.counterfactual_eval.eval_id,
                    "value_target_pack_id": supervision_bundle.value_target_pack.pack_id,
                    "future_training_signals": future_training_signals,
                    "future_training_artifacts": future_training_artifacts,
                },
            )
            work_order = build_execution_work_order(
                order_type="governed_video_admission",
                subject_id=proposal.proposal_id,
                subject_kind="governed_video_proposal",
                decision=(
                    "capture_negative_supervision"
                    if plausibility_report.decision == RegalDecision.BLOCK
                    else "admit_datapack"
                    if benchmark_gate.ready
                    else "admit_shadow_datapack"
                ),
                priority=float(max(0.0, proposal.estimated_novelty)),
                recommended_mode=(
                    "negative_counterexample"
                    if plausibility_report.decision == RegalDecision.BLOCK
                    else "stage1_datapack"
                    if benchmark_gate.ready
                    else "shadow_stage1_datapack"
                ),
                readiness=execution_preconditions,
                reasons=plausibility_report.reason_codes or ["plausibility_ok"],
                artifact_refs={
                    **sidecar_paths,
                    "proposal_id": proposal.proposal_id,
                },
                metadata={
                    "video_id": video_ref["episode_id"],
                    "augmentation_type": proposal.augmentation_type,
                },
            )
            admission_record = {
                "video_id": video_ref["episode_id"],
                "proposal_id": proposal.proposal_id,
                "augmentation_type": proposal.augmentation_type,
                "blocked": plausibility_report.decision == RegalDecision.BLOCK,
                "plausibility_gate": plausibility_report.to_dict(),
                "execution_preconditions": execution_preconditions.to_dict(),
                "execution_work_order": work_order.to_dict(),
                "benchmark_gate": benchmark_gate.to_dict(),
                "routing_source": proposal.routing_source,
                "routing_score": float(proposal.routing_score),
                "source_hypothesis_id": proposal.source_hypothesis_id,
                "diffusion_backend_selected": proposal.diffusion_backend_selected,
                "diffusion_backend_policy": proposal.diffusion_backend_policy,
                "diffusion_materialization_mode": proposal.diffusion_materialization_mode,
                "diffusion_provider_truth": dict(
                    proposal.diffusion_provider_truth or {}
                ),
                "diffusion_routing_context": dict(routing_context),
                "video_state_id": snapshot.state_id,
                "counterfactual_eval_id": supervision_bundle.counterfactual_eval.eval_id,
                "value_target_pack_id": supervision_bundle.value_target_pack.pack_id,
                "future_training_signals": future_training_signals,
                "future_training_artifacts": future_training_artifacts,
                **sidecar_paths,
            }
            admission_records.append(admission_record)
            if plausibility_report.decision == RegalDecision.BLOCK:
                print(
                    "      Skipped by plausibility gate:",
                    ",".join(plausibility_report.reason_codes),
                )
                pipeline_log.append(dict(admission_record))
                continue

            # Generate VLA plan
            vla_plan = extract_vla_plan_from_proposal(proposal, vla_planner)
            if (
                isinstance(vla_plan.confidence, (list, np.ndarray))
                and len(vla_plan.confidence) > 0
            ):
                vla_mean_confidence = float(np.mean(vla_plan.confidence))
            else:
                vla_mean_confidence = 0.5
            print(f"      VLA skills: {vla_plan.skill_sequence[:3]}...")

            # Create datapack
            datapack = create_datapack_from_pipeline(
                video_ref,
                semantic_tags,
                proposal,
                vla_plan,
                semantic_world_model,
                semantic_backbone_result.orchestrator_advisory,
                objective_preset,
                benchmark_gate=benchmark_gate.to_dict(),
                execution_work_order=work_order.to_dict(),
            )
            datapack.regal_annotations = {
                "gen_plausibility": plausibility_report.to_dict(),
                "governed_video_state_id": snapshot.state_id,
                "governed_video_constraint_set": constraint_set,
                "governed_video_hypothesis_mode": proposal.augmentation_type,
                "governed_video_counterfactual_eval_id": supervision_bundle.counterfactual_eval.eval_id,
                "governed_video_value_target_pack_id": supervision_bundle.value_target_pack.pack_id,
                "semantic_world_model_id": semantic_world_model.world_model_id,
                "semantic_capabilities": semantic_world_model.capability_scores,
                "orchestrator_meta_nodes": semantic_backbone_result.orchestrator_advisory.meta_node_weights,
                "execution_preconditions": execution_preconditions.to_dict(),
                "execution_work_order_id": work_order.work_order_id,
                "benchmark_gate": benchmark_gate.to_dict(),
                "diffusion_routing_source": proposal.routing_source,
                "diffusion_routing_score": float(proposal.routing_score),
                "source_hypothesis_id": proposal.source_hypothesis_id,
                "diffusion_backend_selected": proposal.diffusion_backend_selected,
                "diffusion_backend_policy": proposal.diffusion_backend_policy,
                "diffusion_materialization_mode": proposal.diffusion_materialization_mode,
            }
            datapack.episode_metrics["execution_preconditions"] = (
                execution_preconditions.to_dict()
            )
            datapack.episode_metrics["execution_work_order"] = work_order.to_dict()
            datapack.episode_metrics["benchmark_gate"] = benchmark_gate.to_dict()
            datapack.episode_metrics["scene_tracks_backend"] = _scene_tracks_backend(
                video_ref
            )
            datapack.episode_metrics["vision_backbone_selected"] = (
                _vision_backbone_selected(video_ref)
            )
            datapack.episode_metrics["teacher_runtime_backend_selected"] = (
                _teacher_runtime_backend_selected(video_ref)
            )
            datapack.episode_metrics["diffusion_routing_score"] = float(
                proposal.routing_score
            )
            datapack.episode_metrics["diffusion_backend_selected"] = (
                proposal.diffusion_backend_selected
            )
            datapack.episode_metrics["diffusion_backend_policy"] = (
                proposal.diffusion_backend_policy
            )
            datapack.episode_metrics["diffusion_materialization_mode"] = (
                proposal.diffusion_materialization_mode
            )
            datapack.episode_metrics["diffusion_provider_truth"] = dict(
                proposal.diffusion_provider_truth or {}
            )
            print(f"      DataPack: {datapack.pack_id}")
            print(
                f"      Tier: {datapack.attribution.tier}, Trust: {datapack.attribution.trust_score:.3f}"
            )

            all_datapacks.append(datapack)
            all_proposals.append(proposal)
            all_plans.append(vla_plan)

            # Log
            pipeline_log.append(
                {
                    **dict(admission_record),
                    "semantic_tags": semantic_tags,
                    "vla_skills": vla_plan.skill_sequence[:5],
                    "vla_confidence": vla_mean_confidence,
                    "datapack_id": datapack.pack_id,
                    "tier": datapack.attribution.tier,
                    "trust_score": datapack.attribution.trust_score,
                    "estimated_novelty": proposal.estimated_novelty,
                    "semantic_world_model_id": semantic_world_model.world_model_id,
                    "semantic_capabilities": semantic_world_model.capability_scores,
                    "meta_node_weights": semantic_backbone_result.orchestrator_advisory.meta_node_weights,
                    "constraint_set": constraint_set,
                    "benchmark_gate": benchmark_gate.to_dict(),
                    "routing_source": proposal.routing_source,
                    "routing_score": float(proposal.routing_score),
                    "diffusion_backend_selected": proposal.diffusion_backend_selected,
                    "diffusion_backend_policy": proposal.diffusion_backend_policy,
                    "blocked": False,
                }
            )

    # Save outputs
    # 1. Datapacks
    datapacks_path = os.path.join(output_dir, "datapacks.json")
    # Use the to_dict method for proper serialization
    datapacks_data = [dp.to_dict() for dp in all_datapacks]
    with open(datapacks_path, "w") as f:
        json.dump(datapacks_data, f, indent=2)
    print(f"\nSaved {len(all_datapacks)} datapacks to {datapacks_path}")

    # 1b. Econ/Semantic advisory tags
    econ_semantic_path = os.path.join(output_dir, "stage1_econ_semantic_tags.jsonl")
    with open(econ_semantic_path, "w") as f:
        for dp in all_datapacks:
            line = {
                "pack_id": dp.pack_id,
                "econ_semantic_tags": dp.econ_semantic_tags or [],
                "semantic_quality": dp.semantic_quality
                if dp.semantic_quality is not None
                else (dp.attribution.trust_score if dp.attribution else None),
            }
            f.write(json.dumps(line) + "\n")
    print(f"Saved econ/semantic tag advisory file to {econ_semantic_path}")

    # 2. Pipeline log
    log_path = os.path.join(output_dir, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(pipeline_log, f, indent=2)
    print(f"Saved pipeline log to {log_path}")
    admission_log_path = (
        Path(output_dir) / "governed_video" / "proposal_admission_v1.jsonl"
    )
    _write_jsonl(admission_log_path, admission_records)
    print(f"Saved governed video admission log to {admission_log_path}")

    # Compute statistics
    tier_counts = {0: 0, 1: 0, 2: 0}
    avg_trust = 0.0
    avg_novelty = 0.0
    augmentation_types = {}
    routing_sources = {}
    diffusion_backends = {}

    completed_entries = [entry for entry in pipeline_log if not entry.get("blocked")]
    for entry in completed_entries:
        tier_counts[entry["tier"]] = tier_counts.get(entry["tier"], 0) + 1
        avg_trust += entry["trust_score"]
        avg_novelty += entry["estimated_novelty"]
        aug_type = entry["augmentation_type"]
        augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1
        routing_source = str(entry.get("routing_source", "unknown"))
        routing_sources[routing_source] = routing_sources.get(routing_source, 0) + 1
        diffusion_backend = str(entry.get("diffusion_backend_selected", "unknown"))
        diffusion_backends[diffusion_backend] = (
            diffusion_backends.get(diffusion_backend, 0) + 1
        )

    if completed_entries:
        avg_trust /= len(completed_entries)
        avg_novelty /= len(completed_entries)

    stats = {
        "total_videos": len(video_refs),
        "total_proposals": generated_proposal_count,
        "total_datapacks": len(all_datapacks),
        "blocked_proposals": sum(1 for row in admission_records if row.get("blocked")),
        "admitted_proposals": sum(
            1 for row in admission_records if not row.get("blocked")
        ),
        "benchmark_ready_proposals": sum(
            1
            for row in admission_records
            if dict(row.get("benchmark_gate", {}) or {}).get("ready")
        ),
        "shadow_only_proposals": sum(
            1
            for row in admission_records
            if not row.get("blocked")
            and not dict(row.get("benchmark_gate", {}) or {}).get("ready")
        ),
        "executable_work_orders": sum(
            1
            for row in admission_records
            if dict(row.get("execution_work_order", {}) or {}).get("ready")
        ),
        "tier_distribution": tier_counts,
        "avg_trust_score": avg_trust,
        "avg_novelty": avg_novelty,
        "augmentation_type_distribution": augmentation_types,
        "routing_source_distribution": routing_sources,
        "diffusion_backend_distribution": diffusion_backends,
        "objective_preset": objective_preset,
        "proposal_admission_log": str(admission_log_path),
    }

    stats_path = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 Pipeline: Video → Diffusion → VLA → DataPack"
    )
    parser.add_argument(
        "--num-videos", type=int, default=5, help="Number of video references"
    )
    parser.add_argument(
        "--proposals-per-video",
        type=int,
        default=3,
        help="Diffusion proposals per video",
    )
    parser.add_argument(
        "--objective-preset",
        type=str,
        default="balanced",
        choices=["balanced", "throughput", "safety", "energy_saver"],
    )
    parser.add_argument("--output-dir", type=str, default="results/stage1_pipeline")
    parser.add_argument(
        "--video-manifest",
        type=str,
        default=None,
        help="Optional JSON/JSONL manifest of real video references",
    )
    parser.add_argument(
        "--diffusion-backend-policy",
        type=str,
        default="auto",
        choices=["auto", "real", "disabled", "stub"],
        help="Real-or-unavailable policy for the diffusion materialization provider.",
    )
    parser.add_argument(
        "--diffusion-model-ref",
        type=str,
        default=None,
        help="Optional local/cached diffusers model ref for real video diffusion bring-up.",
    )
    parser.add_argument(
        "--diffusion-device",
        type=str,
        default="cuda",
        help="Device to use when a real diffusion provider is available.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Stage 1 Pipeline: Real Video → Diffusion → VLA → DataPackMeta")
    print("=" * 70)
    print(f"Videos: {args.num_videos}")
    print(f"Proposals per video: {args.proposals_per_video}")
    print(f"Objective preset: {args.objective_preset}")
    print("=" * 70)

    stats = run_stage1_pipeline(
        num_videos=args.num_videos,
        proposals_per_video=args.proposals_per_video,
        objective_preset=args.objective_preset,
        output_dir=args.output_dir,
        video_manifest=args.video_manifest,
        diffusion_backend_policy=args.diffusion_backend_policy,
        diffusion_model_ref=args.diffusion_model_ref,
        diffusion_device=args.diffusion_device,
    )

    print("\n" + "=" * 70)
    print("Stage 1 Pipeline Summary")
    print("=" * 70)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"Total diffusion proposals: {stats['total_proposals']}")
    print(f"Total datapacks created: {stats['total_datapacks']}")
    print(f"Tier distribution: {stats['tier_distribution']}")
    print(f"Average trust score: {stats['avg_trust_score']:.3f}")
    print(f"Average novelty: {stats['avg_novelty']:.3f}")
    print(f"Augmentation types: {stats['augmentation_type_distribution']}")
    print("\nStage 1 pipeline complete! Datapacks ready for downstream RL training.")


if __name__ == "__main__":
    main()
