"""Runtime semantic backbone bridging world-model state into snapshot and advisory outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.evidence.benchmark_gating import collect_benchmark_gating_signals
from src.evidence.preconditions import build_execution_preconditions
from src.orchestrator.semantic_orchestrator_v2 import (
    OrchestratorAdvisory,
    SemanticOrchestratorV2,
)
from src.semantic.models import EconSlice, MetaTransformerSlice, SemanticSnapshot
from src.world_model.semantic_world_model import SemanticWorldModelState


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


def _semantic_world_model_summary(world_model: Optional[SemanticWorldModelState]) -> Dict[str, Any]:
    if world_model is None:
        return {}
    return {
        "world_model_id": world_model.world_model_id,
        "version": world_model.version,
        "topology": world_model.topology,
        "capability_scores": world_model.capability_scores,
        "meta_nodes": {
            item.node_type: item.score
            for item in world_model.meta_nodes
        },
    }


def _semantic_runtime_truth(
    *,
    semantic_world_model: SemanticWorldModelState,
    runtime_metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    grounded_scene = _mapping(semantic_world_model.metadata.get("grounded_scene"))
    topology = _mapping(semantic_world_model.topology)
    scene_tracks_backend = str(
        metadata.get("scene_tracks_backend")
        or ("real" if grounded_scene.get("grounding_mode") == "scene_tracks" else "")
    )
    teacher_runtime_backend_selected = str(
        metadata.get("teacher_runtime_backend_selected")
        or ("real" if grounded_scene.get("teacher_trace_present", False) else "unavailable")
    )
    vision_backbone_selected = str(
        metadata.get("vision_backbone_selected")
        or metadata.get("openvla_vision_backbone_selected")
        or ("real" if grounded_scene.get("vla_semantic_evidence_present", False) else "unavailable")
    )
    grounded_track_object_count = int(
        topology.get("grounded_track_object_count")
        or grounded_scene.get("grounded_object_count")
        or 0
    )
    semantic_grounding_ready = bool(grounded_scene.get("semantic_grounding_ready", False))
    semantic_fusion_confidence = _safe_float(
        runtime_metrics.get(
            "semantic_fusion_confidence_mean",
            metadata.get("semantic_fusion_confidence_mean", 0.0),
        ),
    )
    quality_score = max(
        _safe_float(metadata.get("quality_score", 0.0)),
        _safe_float(grounded_scene.get("scene_ir_quality", 0.0)),
        semantic_fusion_confidence,
    )
    truth = {
        "scene_tracks_backend": scene_tracks_backend,
        "teacher_runtime_backend_selected": teacher_runtime_backend_selected,
        "vision_backbone_selected": vision_backbone_selected,
        "semantic_grounding_mode": (
            "non_heuristic"
            if semantic_grounding_ready
            else str(grounded_scene.get("grounding_mode", "heuristic_fallback") or "heuristic_fallback")
        ),
        "semantic_memory_grounded": bool(grounded_track_object_count > 0 or semantic_grounding_ready),
        "grounded_track_object_count": grounded_track_object_count,
        "teacher_runtime_live": bool(grounded_scene.get("teacher_trace_present", False)),
        "semantic_fusion_confidence_mean": semantic_fusion_confidence,
        "quality_score": quality_score,
    }
    benchmark_signals = collect_benchmark_gating_signals(truth)
    execution_preconditions = build_execution_preconditions(
        subject_id=semantic_world_model.world_model_id,
        subject_kind="semantic_runtime_snapshot",
        signal_values={
            **benchmark_signals,
            "semantic_fusion_ready": semantic_fusion_confidence > 0.0,
            "semantic_fusion_confidence_mean": semantic_fusion_confidence,
            "quality_score": quality_score,
        },
        required_boolean_signals={"semantic_grounding_non_heuristic": True},
        soft_boolean_signals={
            "teacher_runtime_real": True,
            "vision_backbone_real": True,
            "semantic_fusion_ready": True,
        },
        metadata={"grounded_scene": grounded_scene},
    )
    return {
        **truth,
        "benchmark_signals": benchmark_signals,
        "execution_preconditions": execution_preconditions.to_dict(),
    }


def build_orchestrator_control_plane_context(
    *,
    semantic_world_model: SemanticWorldModelState,
    semantic_snapshot: SemanticSnapshot,
    orchestrator_advisory: OrchestratorAdvisory,
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot_metadata = _mapping(getattr(semantic_snapshot, "metadata", {}))
    benchmark_signals = _mapping(snapshot_metadata.get("benchmark_signals"))
    execution_preconditions = _mapping(snapshot_metadata.get("execution_preconditions"))
    semantic_runtime_truth = _mapping(snapshot_metadata.get("semantic_runtime_truth"))
    return {
        "schema_version": "orchestrator_control_plane_context_v1",
        "receipt_kind": "orchestrator_control_plane_context_v1",
        "authority_class": "canonical_metadata",
        "decision_scope": "semantic_runtime_control_plane",
        "reward_math_mutation": False,
        "task_id": str(semantic_snapshot.task_id),
        "semantic_world_model_id": semantic_world_model.world_model_id,
        "semantic_world_model_summary": _semantic_world_model_summary(semantic_world_model),
        "focus_objective_presets": list(orchestrator_advisory.focus_objective_presets),
        "sampler_strategy_overrides": dict(orchestrator_advisory.sampler_strategy_overrides),
        "datapack_priority_tags": list(orchestrator_advisory.datapack_priority_tags),
        "safety_emphasis": float(orchestrator_advisory.safety_emphasis),
        "execution_mode": str(orchestrator_advisory.execution_mode),
        "policy_source": str(orchestrator_advisory.policy_source),
        "promotion_stage": orchestrator_advisory.promotion_stage,
        "meta_node_weights": dict(orchestrator_advisory.meta_node_weights),
        "activation_plan": dict(orchestrator_advisory.activation_plan),
        "activation_work_order": dict(orchestrator_advisory.activation_work_order or {}),
        "helper_trace": dict(orchestrator_advisory.helper_trace or {}),
        "benchmark_signals": benchmark_signals,
        "execution_preconditions": execution_preconditions,
        "semantic_runtime_truth": semantic_runtime_truth,
        "artifact_refs": _mapping(artifact_refs),
        "metadata": {
            "snapshot_runtime_backbone": str(snapshot_metadata.get("runtime_backbone", "")),
            "active_meta_nodes": sorted(
                [
                    str(node_type)
                    for node_type, score in dict(orchestrator_advisory.meta_node_weights).items()
                    if _safe_float(score) > 0.0
                ]
            ),
            "capability_keys": sorted(
                str(key) for key in dict(semantic_world_model.capability_scores or {}).keys()
            ),
        },
    }


@dataclass(frozen=True)
class RuntimeSemanticBackboneResult:
    semantic_world_model: SemanticWorldModelState
    semantic_snapshot: SemanticSnapshot
    orchestrator_advisory: OrchestratorAdvisory


class SemanticRuntimeBackbone:
    """Materialize a shared semantic packet for runtime consumers."""

    def __init__(self, orchestrator_config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(orchestrator_config or {})
        config.setdefault("write_to_file", False)
        self.orchestrator = SemanticOrchestratorV2(config)

    def build(
        self,
        *,
        task_id: str,
        objective_preset: str,
        semantic_world_model: SemanticWorldModelState,
        stage2_ontology_proposals: Optional[Sequence[Any]] = None,
        stage2_task_refinements: Optional[Sequence[Any]] = None,
        stage2_tags: Optional[Sequence[Any]] = None,
        recap_summary: Optional[Dict[str, Any]] = None,
        runtime_metrics: Optional[Mapping[str, Any]] = None,
        frontier_episodes: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        backends: Optional[Sequence[str]] = None,
    ) -> RuntimeSemanticBackboneResult:
        runtime_metrics = _mapping(runtime_metrics)
        metadata_payload = _mapping(metadata)
        topology = semantic_world_model.topology or {}
        capabilities = semantic_world_model.capability_scores or {}
        runtime_truth = _semantic_runtime_truth(
            semantic_world_model=semantic_world_model,
            runtime_metrics=runtime_metrics,
            metadata=metadata_payload,
        )
        econ_slice = EconSlice(
            task_id=task_id,
            avg_mpl_units_per_hour=_safe_float(runtime_metrics.get("mpl", runtime_metrics.get("avg_mpl_units_per_hour", 0.0))),
            avg_wage_parity=_safe_float(runtime_metrics.get("wage_parity", runtime_metrics.get("avg_wage_parity", 1.0)), 1.0),
            avg_energy_cost=_safe_float(runtime_metrics.get("energy_cost", runtime_metrics.get("avg_energy_cost", 0.0))),
            avg_error_rate=_safe_float(runtime_metrics.get("error_rate", runtime_metrics.get("avg_error_rate", 0.0))),
            frontier_episodes=[str(item) for item in (frontier_episodes or [])],
            metadata={
                "runtime_metrics": runtime_metrics,
                "semantic_world_model_topology": topology,
                "semantic_runtime_truth": runtime_truth,
            },
        )
        meta_slice = MetaTransformerSlice(
            task_id=task_id,
            objective_vectors=[{"semantic_world_model_capability_mean": _safe_float(sum(capabilities.values()) / max(len(capabilities), 1))}],
            presets=[objective_preset],
            expected_deltas={
                "expected_delta_mpl": _safe_float(runtime_metrics.get("expected_delta_mpl", 0.0)),
                "expected_delta_error": _safe_float(runtime_metrics.get("expected_delta_error", 0.0)),
                "expected_delta_energy_Wh": _safe_float(runtime_metrics.get("expected_delta_energy_Wh", 0.0)),
            },
            backends=[str(item) for item in (backends or [])],
            metadata={
                "semantic_world_model_id": semantic_world_model.world_model_id,
                "active_meta_nodes": [item.node_type for item in semantic_world_model.meta_nodes],
                "semantic_runtime_truth": runtime_truth,
            },
        )
        snapshot = SemanticSnapshot(
            task_id=str(task_id),
            ontology_proposals=list(stage2_ontology_proposals or []),
            task_refinements=list(stage2_task_refinements or []),
            semantic_tags=list(stage2_tags or []),
            econ_slice=econ_slice,
            meta_slice=meta_slice,
            semantic_world_model=semantic_world_model,
            num_segments=int(topology.get("relation_count", 0)),
            segment_types={"meta_node_count": int(topology.get("meta_node_count", 0))},
            subtask_label_histogram={},
            mobility_drift_rate=_safe_float(runtime_metrics.get("mobility_drift_rate", 0.0)),
            recovery_segment_fraction=_safe_float(
                runtime_metrics.get(
                    "recovery_segment_fraction",
                    1.0 if any(node.node_type == "recovery_router" for node in semantic_world_model.meta_nodes) else 0.0,
                )
            ),
            metadata={
                "runtime_backbone": "semantic_runtime_backbone_v1",
                "recap": recap_summary or {},
                "semantic_world_model": semantic_world_model.to_dict(),
                "semantic_world_model_summary": _semantic_world_model_summary(semantic_world_model),
                "semantic_runtime_truth": runtime_truth,
                "benchmark_signals": runtime_truth["benchmark_signals"],
                "execution_preconditions": runtime_truth["execution_preconditions"],
                **runtime_metrics,
                **metadata_payload,
            },
        ).sorted_copy()
        advisory = self.orchestrator.propose(snapshot)
        return RuntimeSemanticBackboneResult(
            semantic_world_model=semantic_world_model,
            semantic_snapshot=snapshot,
            orchestrator_advisory=advisory,
        )

    def write_sidecars(
        self,
        *,
        output_dir: str | Path,
        file_stem: str,
        result: RuntimeSemanticBackboneResult,
    ) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        world_model_path = output_path / f"{file_stem}_semantic_world_model_v1.json"
        snapshot_path = output_path / f"{file_stem}_semantic_snapshot_v1.json"
        advisory_path = output_path / f"{file_stem}_orchestrator_advisory_v1.json"
        control_plane_context_path = output_path / f"{file_stem}_control_plane_context_v1.json"
        world_model_path.write_text(json.dumps(result.semantic_world_model.to_dict(), indent=2))
        snapshot_path.write_text(json.dumps(result.semantic_snapshot.to_dict(), indent=2))
        advisory_path.write_text(json.dumps(result.orchestrator_advisory.to_json(), indent=2))
        control_plane_context_path.write_text(
            json.dumps(
                build_orchestrator_control_plane_context(
                    semantic_world_model=result.semantic_world_model,
                    semantic_snapshot=result.semantic_snapshot,
                    orchestrator_advisory=result.orchestrator_advisory,
                    artifact_refs={
                        "semantic_world_model_path": str(world_model_path),
                        "semantic_snapshot_path": str(snapshot_path),
                        "orchestrator_advisory_path": str(advisory_path),
                    },
                ),
                indent=2,
            )
        )
        return {
            "semantic_world_model_path": str(world_model_path),
            "semantic_snapshot_path": str(snapshot_path),
            "orchestrator_advisory_path": str(advisory_path),
            "control_plane_context_path": str(control_plane_context_path),
        }


__all__ = [
    "RuntimeSemanticBackboneResult",
    "SemanticRuntimeBackbone",
    "build_orchestrator_control_plane_context",
]
