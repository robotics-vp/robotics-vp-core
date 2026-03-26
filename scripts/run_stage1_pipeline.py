#!/usr/bin/env python3
"""
Stage 1 Pipeline: Real Video → Diffusion → VLA Distillation → DataPackMeta

Connects:
1. Video references (real demonstrations)
2. Diffusion stub (augmented clips based on semantic tags)
3. VLA controller (skill plan generation)
4. DataPackMeta creation (for downstream RL training)

No GPU, no actual generation - just structural correctness.
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

from src.diffusion.real_video_diffusion_stub import (
    VideoDiffusionStub,
    DiffusionProposal,
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
from src.governance import governance_trace_sidecar_payload
from src.runtime import decision_ledger_sidecar_payload, event_spine_sidecar_payload
from src.vla.transformer_planner import VLATransformerPlanner, VLAInput, VLAPlan
from src.valuation.datapack_schema import (
    DataPackMeta,
    ConditionProfile,
    ObjectiveProfile,
    GuidanceProfile,
    AttributionProfile,
)
from src.regal.gen_plausibility import RegalGenPlausibilityNode
from src.regal.base import RegalDecision
from src.semantic.runtime_backbone import SemanticRuntimeBackbone
from src.world_model import GovernedVideoWorldModel, SemanticWorldModelBuilder
from src.world_model.governed_video_supervision import build_governed_video_supervision_bundle
from src.vision.reconstruction import (
    build_four_d_reconstruction_sidecar,
    save_four_d_reconstruction_sidecar,
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
    if explicit:
        return explicit
    metadata = _metadata_dict(video_ref)
    scene_tracks_meta = metadata.get("scene_tracks_metadata", {})
    if isinstance(scene_tracks_meta, dict):
        runner = scene_tracks_meta.get("runner", {})
        run_config = runner.get("run_config", {}) if isinstance(runner, dict) else {}
        if run_config.get("use_stub_adapters") is False:
            return "real"
        if run_config.get("use_stub_adapters") is True:
            return "stub"
    if video_ref.get("scene_tracks_v1") or video_ref.get("scene_tracks_path") or video_ref.get("scene_tracks_npz"):
        return "real"
    if metadata.get("scene_tracks_v1") or metadata.get("scene_tracks_path") or metadata.get("scene_tracks_npz"):
        return "real"
    return "unavailable"


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


def _semantic_grounding_mode(video_ref: Dict[str, Any], semantic_world_model: Optional[Any] = None) -> str:
    explicit = _runtime_field(video_ref, "semantic_grounding_mode", "grounding_mode")
    if explicit:
        return explicit
    if semantic_world_model is not None:
        grounded_scene = getattr(semantic_world_model, "metadata", {}).get("grounded_scene", {})
        if isinstance(grounded_scene, dict) and grounded_scene.get("grounding_mode"):
            return str(grounded_scene["grounding_mode"])
    return "non_heuristic" if _scene_tracks_backend(video_ref) == "real" else "heuristic_fallback"


def _scene_tracks_non_stub(video_ref: Dict[str, Any]) -> bool:
    backend = _scene_tracks_backend(video_ref)
    if backend:
        return backend in {"real", "passthrough"}
    metadata = _metadata_dict(video_ref)
    explicit = metadata.get("future_training_signals", {})
    if isinstance(explicit, dict) and "scene_tracks_non_stub" in explicit:
        return bool(explicit["scene_tracks_non_stub"])
    if "scene_tracks_non_stub" in metadata:
        return bool(metadata["scene_tracks_non_stub"])
    scene_tracks_meta = metadata.get("scene_tracks_metadata", {})
    if isinstance(scene_tracks_meta, dict):
        runner = scene_tracks_meta.get("runner", {})
        run_config = runner.get("run_config", {}) if isinstance(runner, dict) else {}
        execution = scene_tracks_meta.get("execution_preconditions", {})
        if isinstance(run_config, dict) and run_config.get("use_stub_adapters") is False:
            return bool(not isinstance(execution, dict) or execution.get("ready", True))
    return False


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
    )


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
    topology = getattr(semantic_world_model, "topology", {}) or {}
    grounded_track_count = 0
    if isinstance(topology, dict):
        grounded_track_count = int(topology.get("grounded_track_object_count", 0) or 0)
    benchmark_signals = collect_benchmark_gating_signals(
        _stage1_benchmark_metadata(video_ref, semantic_world_model)
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
        "teacher_runtime_live": _teacher_runtime_live(video_ref),
        "scene_tracks_non_stub": _scene_tracks_non_stub(video_ref),
        "semantic_memory_grounded": grounded_track_count > 0,
        "benchmark_gate_ready": bool(getattr(benchmark_gate, "ready", False)) if benchmark_gate is not None else False,
        "budget_settlement_live": False,
    }
    derived.update(
        {
            str(key): bool(value)
            for key, value in benchmark_signals.items()
            if isinstance(value, bool)
        }
    )
    derived.update({str(key): bool(value) for key, value in explicit.items()})
    return dict(sorted(derived.items()))


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


def _stage1_benchmark_metadata(video_ref: Dict[str, Any], semantic_world_model: Any) -> Dict[str, Any]:
    grounded_scene = {}
    if semantic_world_model is not None:
        grounded_scene = dict(getattr(semantic_world_model, "metadata", {}).get("grounded_scene", {}) or {})
    return {
        "scene_tracks_backend": _scene_tracks_backend(video_ref),
        "teacher_runtime_backend_selected": _teacher_runtime_backend_selected(video_ref),
        "vision_backbone_selected": _vision_backbone_selected(video_ref),
        "semantic_grounding_mode": _semantic_grounding_mode(video_ref, semantic_world_model),
        "semantic_memory_grounded": bool(
            grounded_scene.get("grounding_ready", False)
            or int(getattr(semantic_world_model, "topology", {}).get("grounded_track_object_count", 0) or 0) > 0
        ),
        "grounded_track_object_count": int(
            getattr(semantic_world_model, "topology", {}).get("grounded_track_object_count", 0) or 0
        ),
        "semantic_world_model_summary": {
            "topology": dict(getattr(semantic_world_model, "topology", {}) or {}),
            "grounded_track_object_count": int(
                getattr(semantic_world_model, "topology", {}).get("grounded_track_object_count", 0) or 0
            ),
        },
    }


def build_stage1_benchmark_gate(video_ref: Dict[str, Any], semantic_world_model: Any) -> Any:
    return build_benchmark_gate_report(
        subject_id=str(video_ref.get("episode_id", "")),
        subject_kind="stage1_video_diffusion_benchmark_gate",
        metadata=_stage1_benchmark_metadata(video_ref, semantic_world_model),
        require_real_scene_tracks=True,
        require_teacher_runtime=False,
        require_vision_backbone=True,
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
    task_type = str(record.get("task_type") or metadata.get("task_type") or "drawer_vase")
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
    if "semantic_tags" in record and isinstance(record["semantic_tags"], list):
        normalized["metadata"] = dict(metadata, semantic_tags=list(record["semantic_tags"]))
    return normalized


def load_video_references(num_videos: int, manifest_path: Optional[str] = None) -> List[Dict[str, Any]]:
    if manifest_path is None:
        return [simulate_real_video_reference(index=i) for i in range(num_videos)]

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Video manifest not found: {path}")
    if path.suffix.lower() == ".jsonl":
        raw_records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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
                    "teacher_confidence_mean": semantic_conf if "human" in str(video_ref.get("demonstrator", "")).lower() else 0.0,
                },
                payload={"semantic_tags": semantic_tags, "objective_preset": objective_preset},
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
        "evidence_coverage": getattr(belief_state, "state_vector", {}).get("evidence_coverage", 0.0),
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
) -> tuple[Dict[str, str], Any]:
    sidecar_dir = Path(output_dir) / "governed_video"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    episode_id = str(video_ref["episode_id"])
    evidence_bus_path = sidecar_dir / f"{episode_id}_evidence_bus_v1.json"
    belief_state_path = sidecar_dir / f"{episode_id}_belief_state_v1.json"
    snapshot_path = sidecar_dir / f"{episode_id}_video_state_v1.json"
    hypotheses_path = sidecar_dir / f"{episode_id}_hypotheses_v1.json"
    semantic_world_model_path = sidecar_dir / f"{episode_id}_semantic_world_model_v1.json"
    semantic_snapshot_path = sidecar_dir / f"{episode_id}_semantic_snapshot_v1.json"
    orchestrator_advisory_path = sidecar_dir / f"{episode_id}_orchestrator_advisory_v1.json"
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
    semantic_world_model_path.write_text(json.dumps(semantic_world_model.to_dict(), indent=2))
    semantic_snapshot_path.write_text(json.dumps(semantic_snapshot.to_dict(), indent=2))
    orchestrator_advisory_path.write_text(json.dumps(orchestrator_advisory.to_json(), indent=2))
    sidecar_paths = {
        "evidence_bus_path": str(evidence_bus_path),
        "belief_state_path": str(belief_state_path),
        "video_state_path": str(snapshot_path),
        "hypotheses_path": str(hypotheses_path),
        "semantic_world_model_path": str(semantic_world_model_path),
        "semantic_snapshot_path": str(semantic_snapshot_path),
        "orchestrator_advisory_path": str(orchestrator_advisory_path),
    }
    metadata = video_ref.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    frame_count = int(metadata.get("num_frames", 0) or 0)
    reconstruction_path = sidecar_dir / f"{episode_id}_reconstruction_sidecar_v1.json"
    sensor_bundle_meta = metadata.get("sensor_bundle") if isinstance(metadata.get("sensor_bundle"), dict) else None
    if sensor_bundle_meta is None:
        camera_name = str(video_ref.get("camera", "front"))
        sensor_bundle_meta = {
            "cameras": [camera_name],
            "intrinsics": {camera_name: metadata.get("intrinsics_ref", f"intrinsics://{camera_name}")},
            "extrinsics": {camera_name: metadata.get("extrinsics_ref", f"extrinsics://{camera_name}")},
            "depth_unit": metadata.get("depth_unit", "unknown"),
        }
    reconstruction_sidecar = build_four_d_reconstruction_sidecar(
        episode_id=episode_id,
        source_type=str(video_ref.get("source_type", "video_reference")),
        media_refs=[ref for ref in [video_ref.get("video_path"), video_ref.get("depth_path")] if ref],
        sensor_bundle_meta=sensor_bundle_meta,
        frame_count=frame_count,
        frame_range=[0, max(0, frame_count - 1)] if frame_count else None,
        geometry_refs={
            "video_state_path": sidecar_paths["video_state_path"],
            "hypotheses_path": sidecar_paths["hypotheses_path"],
            "depth_path": str(video_ref.get("depth_path", "")),
        },
        evidence_refs={
            "evidence_bus_path": sidecar_paths["evidence_bus_path"],
            "belief_state_path": sidecar_paths["belief_state_path"],
        },
        quality={
            "geometry_quality": float(belief_state.state_vector.get("geometry_quality", 0.0)),
            "evidence_coverage": float(belief_state.state_vector.get("evidence_coverage", 0.0)),
        },
        metadata={"task_type": str(video_ref.get("task_type", ""))},
    )
    save_four_d_reconstruction_sidecar(reconstruction_path, reconstruction_sidecar)
    sidecar_paths["reconstruction_sidecar_path"] = str(reconstruction_path)

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
    value_ledger_receipt_path = sidecar_dir / f"{episode_id}_value_ledger_receipt_v1.json"
    benchmark_gate_path = sidecar_dir / f"{episode_id}_benchmark_gate_v1.json"

    runtime_packet_path.write_text(json.dumps(supervision_bundle.runtime_packet.to_dict(), indent=2))
    pricing_tick_path.write_text(json.dumps(supervision_bundle.pricing_tick.to_dict(), indent=2))
    branch_eval_path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "branch_evaluations": [evaluation.to_dict() for evaluation in supervision_bundle.branch_evaluations],
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
    counterfactual_eval_path.write_text(json.dumps(supervision_bundle.counterfactual_eval.to_dict(), indent=2))
    value_target_pack_path.write_text(json.dumps(supervision_bundle.value_target_pack.to_dict(), indent=2))
    value_ledger_receipt_path.write_text(json.dumps(supervision_bundle.value_ledger_receipt.to_dict(), indent=2))
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
            "benchmark_gate_path": str(benchmark_gate_path) if benchmark_gate is not None else "",
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
    diffusion_stub: VideoDiffusionStub,
    world_model: GovernedVideoWorldModel,
    belief_state: Any,
    objective_preset: str = "balanced",
    num_proposals: int = 3,
) -> tuple[List[DiffusionProposal], Any, List[Any], Dict[str, Any], Dict[str, Any]]:
    """
    Generate governed video hypotheses and render them into proposals.
    """
    constraint_set = build_stage1_constraint_set(semantic_tags, objective_preset, belief_state)
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
        "teacher_runtime_backend_selected": _teacher_runtime_backend_selected(video_ref),
        "vision_backbone_selected": _vision_backbone_selected(video_ref),
        "semantic_grounding_mode": _semantic_grounding_mode(video_ref),
        "evidence_coverage": float(snapshot.state_features.get("evidence_coverage", 0.0)),
        "semantic_disagreement": float(snapshot.state_features.get("evidence_disagreement_mean", 0.0)),
        "constraint_pressure": float(len(dict(constraint_set.get("hard_bounds", {}) or {}))) / 6.0,
        "governed_hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
    }
    proposals = diffusion_stub.propose_augmented_clips(
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
        customer_segment=diffusion_proposal.econ_context.get("customer_segment", "balanced"),
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
        quality_label = "high_value" if diffusion_proposal.estimated_novelty > 0.6 else "medium"

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
        customer_segment=diffusion_proposal.econ_context.get("customer_segment", "balanced"),
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

    tier = 0 if not benchmark_ready else 2 if diffusion_proposal.estimated_novelty > 0.6 else 1

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
        delta_mpl=diffusion_proposal.estimated_novelty * 5.0,  # Novelty correlates with learning
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
        semantic_tags=semantic_tags + [f"vla_skill_{s}" for s in vla_plan.skill_sequence[:3]],
        econ_semantic_tags=econ_semantic_tags,
        semantic_quality=semantic_quality,
        agent_profile={
            "policy": "stage1_vla",
            "source_type": "stage1_diffusion_vla",
            "semantic_world_model_id": getattr(semantic_world_model, "world_model_id", ""),
            "meta_node_weights": getattr(orchestrator_advisory, "meta_node_weights", {}),
            "diffusion_routing_source": diffusion_proposal.routing_source,
            "diffusion_routing_score": diffusion_proposal.routing_score,
            "benchmark_admission_mode": (
                dict(execution_work_order or {}).get("recommended_mode", "shadow_stage1_datapack")
            ),
        },
        signal_bundle={
            "semantic_world_model": {
                "world_model_id": getattr(semantic_world_model, "world_model_id", ""),
                "topology": getattr(semantic_world_model, "topology", {}),
                "capability_scores": getattr(semantic_world_model, "capability_scores", {}),
            },
            "meta_nodes": getattr(orchestrator_advisory, "meta_node_weights", {}),
            "benchmark_gate": dict(benchmark_gate or {}),
            "diffusion_routing": {
                "routing_source": diffusion_proposal.routing_source,
                "routing_score": diffusion_proposal.routing_score,
                "source_hypothesis_id": diffusion_proposal.source_hypothesis_id,
                "benchmark_gate_ready": diffusion_proposal.benchmark_gate_ready,
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
) -> Dict[str, Any]:
    """
    Run full Stage 1 pipeline.

    Returns:
        results: Dict with pipeline outputs and statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    diffusion_stub = VideoDiffusionStub()
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
    video_refs = load_video_references(num_videos=num_videos, manifest_path=video_manifest)

    print(f"Running Stage 1 pipeline with {len(video_refs)} videos...")

    for i, video_ref in enumerate(video_refs):
        print(f"\n--- Video {i+1}/{len(video_refs)} ---")

        # Step 1: Load real or manifest-backed video reference
        print(f"  Video: {video_ref['episode_id']}")

        # Step 2: Extract semantic tags
        semantic_tags = extract_semantic_tags_from_video(video_ref)
        print(f"  Tags: {semantic_tags[:5]}...")

        # Step 3: Build evidence and belief state
        evidence_bus, belief_state = build_video_evidence(video_ref, semantic_tags, objective_preset)

        # Step 4: Generate governed hypotheses, then render proposals
        proposals, snapshot, hypotheses, constraint_set, routing_context = generate_diffusion_proposals(
            video_ref,
            semantic_tags,
            diffusion_stub,
            world_model,
            belief_state,
            objective_preset,
            proposals_per_video,
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
                video_ref.get("teacher_trace")
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
                "avg_error_rate": 0.0 if video_ref.get("metadata", {}).get("success", True) else 1.0,
                "avg_wage_parity": 1.0,
                "avg_mpl_units_per_hour": float(len(semantic_tags)),
                "expected_delta_mpl": float(max((proposal.estimated_novelty for proposal in proposals), default=0.0)),
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
            future_training_artifacts["benchmark_gate_path"] = sidecar_paths["benchmark_gate_path"]

        # Step 5: For each proposal, generate VLA plan and create datapack
        for j, proposal in enumerate(proposals):
            print(f"    Proposal {j+1}: {proposal.augmentation_type}")

            plausibility_context = {
                "map_first_quality_score": max(
                    0.0,
                    min(1.0, belief_state.state_vector.get("geometry_quality", proposal.confidence)),
                ),
                "semantic_disagreement_vla_vs_map": max(
                    0.0,
                    min(
                        1.0,
                        belief_state.state_vector.get("evidence_disagreement_mean", 1.0 - proposal.confidence),
                    ),
                ),
                "vla_evidence_coverage": max(
                    0.0,
                    min(1.0, belief_state.state_vector.get("evidence_coverage", len(semantic_tags) / 12.0)),
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
                    "runtime_packet_path",
                    "governance_trace_path",
                    "counterfactual_eval_path",
                    "value_target_pack_path",
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
                blocked_reasons=plausibility_report.reason_codes if plausibility_report.decision == RegalDecision.BLOCK else [],
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
            if isinstance(vla_plan.confidence, (list, np.ndarray)) and len(vla_plan.confidence) > 0:
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
            }
            datapack.episode_metrics["execution_preconditions"] = execution_preconditions.to_dict()
            datapack.episode_metrics["execution_work_order"] = work_order.to_dict()
            datapack.episode_metrics["benchmark_gate"] = benchmark_gate.to_dict()
            datapack.episode_metrics["scene_tracks_backend"] = _scene_tracks_backend(video_ref)
            datapack.episode_metrics["vision_backbone_selected"] = _vision_backbone_selected(video_ref)
            datapack.episode_metrics["teacher_runtime_backend_selected"] = _teacher_runtime_backend_selected(video_ref)
            datapack.episode_metrics["diffusion_routing_score"] = float(proposal.routing_score)
            print(f"      DataPack: {datapack.pack_id}")
            print(f"      Tier: {datapack.attribution.tier}, Trust: {datapack.attribution.trust_score:.3f}")

            all_datapacks.append(datapack)
            all_proposals.append(proposal)
            all_plans.append(vla_plan)

            # Log
            pipeline_log.append({
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
                "blocked": False,
            })

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
                "semantic_quality": dp.semantic_quality if dp.semantic_quality is not None else (
                    dp.attribution.trust_score if dp.attribution else None
                ),
            }
            f.write(json.dumps(line) + "\n")
    print(f"Saved econ/semantic tag advisory file to {econ_semantic_path}")

    # 2. Pipeline log
    log_path = os.path.join(output_dir, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(pipeline_log, f, indent=2)
    print(f"Saved pipeline log to {log_path}")
    admission_log_path = Path(output_dir) / "governed_video" / "proposal_admission_v1.jsonl"
    _write_jsonl(admission_log_path, admission_records)
    print(f"Saved governed video admission log to {admission_log_path}")

    # Compute statistics
    tier_counts = {0: 0, 1: 0, 2: 0}
    avg_trust = 0.0
    avg_novelty = 0.0
    augmentation_types = {}
    routing_sources = {}

    completed_entries = [entry for entry in pipeline_log if not entry.get("blocked")]
    for entry in completed_entries:
        tier_counts[entry["tier"]] = tier_counts.get(entry["tier"], 0) + 1
        avg_trust += entry["trust_score"]
        avg_novelty += entry["estimated_novelty"]
        aug_type = entry["augmentation_type"]
        augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1
        routing_source = str(entry.get("routing_source", "unknown"))
        routing_sources[routing_source] = routing_sources.get(routing_source, 0) + 1

    if completed_entries:
        avg_trust /= len(completed_entries)
        avg_novelty /= len(completed_entries)

    stats = {
        "total_videos": len(video_refs),
        "total_proposals": generated_proposal_count,
        "total_datapacks": len(all_datapacks),
        "blocked_proposals": sum(1 for row in admission_records if row.get("blocked")),
        "admitted_proposals": sum(1 for row in admission_records if not row.get("blocked")),
        "benchmark_ready_proposals": sum(
            1
            for row in admission_records
            if dict(row.get("benchmark_gate", {}) or {}).get("ready")
        ),
        "shadow_only_proposals": sum(
            1
            for row in admission_records
            if not row.get("blocked") and not dict(row.get("benchmark_gate", {}) or {}).get("ready")
        ),
        "executable_work_orders": sum(
            1 for row in admission_records
            if dict(row.get("execution_work_order", {}) or {}).get("ready")
        ),
        "tier_distribution": tier_counts,
        "avg_trust_score": avg_trust,
        "avg_novelty": avg_novelty,
        "augmentation_type_distribution": augmentation_types,
        "routing_source_distribution": routing_sources,
        "objective_preset": objective_preset,
        "proposal_admission_log": str(admission_log_path),
    }

    stats_path = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Pipeline: Video → Diffusion → VLA → DataPack")
    parser.add_argument("--num-videos", type=int, default=5, help="Number of video references")
    parser.add_argument("--proposals-per-video", type=int, default=3, help="Diffusion proposals per video")
    parser.add_argument("--objective-preset", type=str, default="balanced",
                        choices=["balanced", "throughput", "safety", "energy_saver"])
    parser.add_argument("--output-dir", type=str, default="results/stage1_pipeline")
    parser.add_argument("--video-manifest", type=str, default=None, help="Optional JSON/JSONL manifest of real video references")
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
