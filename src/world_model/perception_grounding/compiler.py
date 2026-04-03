"""Compiler for the canonical Perception / Grounding world state.

This is the first functional Phase 2 compiler: it consumes real upstream
inputs that already exist in the repo (scene tracks, belief state, teacher/VLA
semantic evidence) and compiles them into WM-owned canonical state plus the
first heuristic semantic-bridge outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evidence.belief_state import BeliefState
from src.world_model.semantic_world_model import SemanticWorldModelBuilder

from .common import clip01, mapping, stable_id, strings
from .receipts import EvidenceFusionReceipt, ProviderInvocationReceipt
from .promotion import (
    resolve_evidence_fusion_helper,
    resolve_graph_transformer_helper,
    resolve_provider_adapter_helper,
    resolve_semantic_bridge_helper,
    resolve_temporal_grounding_helper,
)
from .semantic_bridges import (
    AnnotationSemanticBridgeState,
    EconomicSemanticBridgeState,
    EmbodimentSemanticBridgeState,
    SemanticBridgeRegistry,
    SimSynthSemanticBridgeState,
)
from .state import (
    DatasetSurfaceState,
    DeploymentResourceSurface,
    EvidenceRoutingState,
    ObjectTrackState,
    PerceptionGroundingWorldState,
    ProviderSurfaceState,
    SceneEdge,
    SceneGraphState,
    TaskMeasurementSurface,
    TemporalGroundingState,
)


def _empty_belief_state(
    *,
    episode_id: str,
    semantic_tags: Sequence[Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> BeliefState:
    payload = {
        "episode_id": episode_id,
        "semantic_tags": list(strings(semantic_tags)),
        "metadata": mapping(metadata),
    }
    return BeliefState(
        belief_id=f"belief_{stable_id('perception_grounding', episode_id, str(payload))}",
        episode_id=episode_id,
        timestamp=str(mapping(metadata).get("timestamp", "unknown")),
        semantic_tags=strings(semantic_tags),
        state_vector={},
        uncertainty={},
        evidence_refs=[],
        artifact_refs=mapping(mapping(metadata).get("artifact_refs")),
        provenance={"source": "perception_grounding_compiler"},
        metadata=mapping(metadata),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _infer_provider_ids(
    *,
    scene_tracks_payload: Optional[Any],
    vla_semantic_evidence: Optional[Any],
    teacher_trace: Optional[Any],
) -> list[str]:
    providers: list[str] = []
    if scene_tracks_payload is not None:
        providers.append("scene_tracks")
    if vla_semantic_evidence is not None:
        providers.append("vla_semantic_evidence")
    if teacher_trace is not None:
        providers.append("teacher_trace")
    providers.append("vision_backbone_stub")
    return providers


def _provider_surface(
    *,
    state_id: str,
    scene_tracks_payload: Optional[Any],
    vla_semantic_evidence: Optional[Any],
    teacher_trace: Optional[Any],
) -> ProviderSurfaceState:
    provider_ids = _infer_provider_ids(
        scene_tracks_payload=scene_tracks_payload,
        vla_semantic_evidence=vla_semantic_evidence,
        teacher_trace=teacher_trace,
    )
    provider_kinds = {
        "scene_tracks": "scene_tracks",
        "vla_semantic_evidence": "teacher_semantics",
        "teacher_trace": "teacher_trace",
        "vision_backbone_stub": "vision_backbone",
    }
    provider_availability = {
        provider_id: (
            "available"
            if provider_id != "vision_backbone_stub" or True
            else "unavailable"
        )
        for provider_id in provider_ids
    }
    provider_truth_class = {
        "scene_tracks": "provider_backed",
        "vla_semantic_evidence": "advisory_evidence",
        "teacher_trace": "advisory_evidence",
        "vision_backbone_stub": "stub_smoke_only",
    }
    sensor_modalities = {
        "scene_tracks": ["rgb", "depth", "pose"],
        "vla_semantic_evidence": ["semantic_tokens"],
        "teacher_trace": ["action_semantics"],
        "vision_backbone_stub": ["rgb"],
    }
    return ProviderSurfaceState(
        surface_id=f"provider_surface_{state_id}",
        provider_ids=provider_ids,
        provider_kinds={
            provider_id: provider_kinds[provider_id]
            for provider_id in provider_ids
            if provider_id in provider_kinds
        },
        provider_availability={
            provider_id: provider_availability[provider_id]
            for provider_id in provider_ids
            if provider_id in provider_availability
        },
        provider_truth_class={
            provider_id: provider_truth_class[provider_id]
            for provider_id in provider_ids
            if provider_id in provider_truth_class
        },
        sensor_modalities={
            provider_id: sensor_modalities[provider_id]
            for provider_id in provider_ids
            if provider_id in sensor_modalities
        },
        vectorized_runtime_supported=bool(scene_tracks_payload is not None),
        provider_batch_capacity=max(1, len(provider_ids)),
        provider_latency_budget_ms=40.0,
        metadata={
            "provider_surface_mode": "heuristic_compiled",
            "load_bearing_sources": provider_ids,
        },
    )


def _dataset_surface(
    *,
    state_id: str,
    scene_tracks_payload: Optional[Any],
    metadata: Optional[Mapping[str, Any]],
) -> DatasetSurfaceState:
    metadata_payload = mapping(metadata)
    sensor_inventory = ["rgb"]
    if isinstance(scene_tracks_payload, Mapping):
        sensor_inventory.append("scene_tracks")
        if "scene_tracks_v1/poses_t" in scene_tracks_payload or "poses_t" in scene_tracks_payload:
            sensor_inventory.append("pose")
    return DatasetSurfaceState(
        surface_id=f"dataset_surface_{state_id}",
        dataset_ids=strings(metadata_payload.get("dataset_ids")) or ["runtime_episode"],
        world_inventory_ids=strings(metadata_payload.get("world_inventory_ids")) or ["runtime_scene"],
        split_name=str(metadata_payload.get("split_name", "runtime")),
        sensor_inventory=sensor_inventory,
        scene_hierarchy_levels=["object", "relation", "scene"],
        available_sequences=1,
        calibration_assets_present=bool(scene_tracks_payload is not None),
        annotation_sources=["teacher_runtime", "vla_semantic_evidence"]
        if scene_tracks_payload is not None
        else [],
        metadata={"source": "perception_grounding_compiler"},
    )


def _task_measurements(
    *,
    state_id: str,
    task_id: str,
    semantic_state: Any,
    object_count: int,
    edge_count: int,
    fusion_confidence: float,
) -> TaskMeasurementSurface:
    grounding_quality = clip01(
        max(
            _safe_float(getattr(semantic_state, "capability_scores", {}).get("grounding_quality"), 0.0),
            fusion_confidence,
        )
    )
    temporal_stability = clip01(
        _safe_float(getattr(semantic_state, "topology", {}).get("temporal_stability", 0.0), 0.0)
    )
    semantic_density = clip01(min(object_count, 8) / 8.0 + min(edge_count, 8) / 16.0)
    measurement_values = {
        "grounding_quality": grounding_quality,
        "temporal_stability": temporal_stability,
        "semantic_density": semantic_density,
        "object_count_norm": clip01(object_count / 8.0),
    }
    measurement_status = {
        key: ("good" if value >= 0.65 else "degraded" if value >= 0.35 else "poor")
        for key, value in measurement_values.items()
    }
    return TaskMeasurementSurface(
        surface_id=f"task_measurement_{state_id}",
        task_id=task_id or "perception_shadow_runtime",
        measurement_names=list(measurement_values.keys()),
        measurement_values=measurement_values,
        measurement_status=measurement_status,
        vector_env_count=1,
        measurement_window_frames=max(
            1,
            int(_safe_float(getattr(semantic_state, "topology", {}).get("track_count"), object_count)),
        ),
        metadata={
            "measurement_mode": "compiler_shadow_runtime",
            "semantic_world_model_id": getattr(semantic_state, "world_model_id", ""),
        },
    )


def _dense_token(values: Sequence[float], target_dim: int = 8) -> list[float]:
    padded = list(values[:target_dim])
    while len(padded) < target_dim:
        padded.append(0.0)
    return [float(v) for v in padded]


def _track_feature_token(
    *,
    object_id: str,
    confidence: float,
    salience: float,
    affordance_count: int,
    risk_count: int,
) -> list[float]:
    hashed = stable_id("obj_tok", object_id)
    hash_vals = [int(hashed[idx : idx + 2], 16) / 255.0 for idx in range(0, min(len(hashed), 16), 2)]
    return _dense_token(
        [
            confidence,
            salience,
            clip01(affordance_count / 4.0),
            clip01(risk_count / 3.0),
            *hash_vals[:4],
        ],
        target_dim=8,
    )


def _semantic_object_to_track(item: Any) -> ObjectTrackState:
    track_id = strings(getattr(item, "track_refs", []))[0] if strings(getattr(item, "track_refs", [])) else str(getattr(item, "object_id", "unknown"))
    metadata = mapping(getattr(item, "metadata", {}))
    return ObjectTrackState(
        track_id=track_id,
        object_label=str(getattr(item, "label", "")),
        object_category=str(getattr(item, "category", "")),
        confidence=clip01(_safe_float(getattr(item, "confidence", 0.0), 0.0)),
        epistemic_uncertainty=clip01(1.0 - _safe_float(getattr(item, "confidence", 0.0), 0.0)),
        feature_token=_track_feature_token(
            object_id=str(getattr(item, "object_id", "")),
            confidence=_safe_float(getattr(item, "confidence", 0.0), 0.0),
            salience=_safe_float(getattr(item, "salience", 0.0), 0.0),
            affordance_count=len(strings(getattr(item, "affordances", []))),
            risk_count=len(strings(getattr(item, "risk_tags", []))),
        ),
        provider_sources=["scene_tracks", "semantic_world_model"],
        visibility=clip01(_safe_float(metadata.get("visibility_mean"), 0.5)),
        occlusion_score=clip01(_safe_float(metadata.get("occlusion_mean"), 0.0)),
        temporal_persistence_frames=int(_safe_float(metadata.get("track_length"), 1.0,)),
        first_seen_frame=int(_safe_float(metadata.get("first_seen_frame"), 0.0)),
        last_seen_frame=int(_safe_float(metadata.get("last_seen_frame"), 0.0)),
        occlusion_state=str(metadata.get("occlusion_state", "visible")),
        reidentification_confidence=clip01(_safe_float(metadata.get("teacher_match"), 0.0) * 0.5 + 0.5),
        affordance_hints=strings(getattr(item, "affordances", [])),
        risk_hints=strings(getattr(item, "risk_tags", [])),
        metadata={
            "semantic_object_id": str(getattr(item, "object_id", "")),
            "aliases": strings(getattr(item, "aliases", [])),
            "state_tags": strings(getattr(item, "state_tags", [])),
            **metadata,
        },
    )


def _semantic_relation_to_edge(item: Any, object_id_to_track_id: Mapping[str, str]) -> SceneEdge:
    subject_id = str(getattr(item, "subject_id", ""))
    object_id = str(getattr(item, "object_id", ""))
    relation_type = str(getattr(item, "relation_type", "semantic_relation"))
    return SceneEdge(
        edge_id=str(getattr(item, "relation_id", stable_id(subject_id, relation_type, object_id))),
        source_track_id=str(object_id_to_track_id.get(subject_id, subject_id)),
        target_track_id=str(object_id_to_track_id.get(object_id, object_id)),
        edge_type=relation_type,
        confidence=clip01(_safe_float(getattr(item, "confidence", 0.0), 0.0)),
        spatial_distance=_safe_float(mapping(getattr(item, "metadata", {})).get("distance"), 0.0),
        edge_features=_dense_token(
            [
                _safe_float(getattr(item, "confidence", 0.0), 0.0),
                1.0 if relation_type in {"contact", "acts_on"} else 0.5,
            ],
            target_dim=4,
        ),
        metadata=mapping(getattr(item, "metadata", {})),
    )


def _scene_graph(semantic_state: Any, graph_helper: Mapping[str, Any]) -> SceneGraphState:
    object_tracks = [_semantic_object_to_track(item) for item in getattr(semantic_state, "objects", [])]
    object_id_to_track_id = {
        str(track.metadata.get("semantic_object_id", track.track_id)): track.track_id for track in object_tracks
    }
    edges = [
        _semantic_relation_to_edge(item, object_id_to_track_id)
        for item in getattr(semantic_state, "relations", [])
    ]
    edge_type_counts: dict[str, int] = {}
    for edge in edges:
        edge_type_counts[edge.edge_type] = edge_type_counts.get(edge.edge_type, 0) + 1
    object_count = len(object_tracks)
    edge_count = len(edges)
    graph_density = 0.0
    if object_count > 1:
        graph_density = float(edge_count) / float(object_count * (object_count - 1))
    capability_scores = mapping(getattr(semantic_state, "capability_scores", {}))
    scene_summary_token = _dense_token(
        [
            float(object_count),
            float(edge_count),
            clip01(graph_density),
            clip01(_safe_float(capability_scores.get("grounding_quality"), 0.0)),
            clip01(_safe_float(capability_scores.get("affordance_grounding"), 0.0)),
            clip01(_safe_float(capability_scores.get("risk_reasoning"), 0.0)),
            clip01(graph_helper.get("helper_weight", 0.0)),
            clip01(_safe_float(getattr(semantic_state, "topology", {}).get("temporal_stability"), 0.0)),
        ],
        target_dim=8,
    )
    return SceneGraphState(
        graph_id=f"scene_graph_{getattr(semantic_state, 'world_model_id', stable_id('scene_graph', object_count, edge_count))}",
        object_tracks=object_tracks,
        edges=edges,
        scene_summary_token=scene_summary_token,
        object_count=object_count,
        edge_count=edge_count,
        edge_type_counts=edge_type_counts,
        graph_density=graph_density,
        metadata={
            "source_world_model_id": getattr(semantic_state, "world_model_id", ""),
            "semantic_tags": strings(getattr(semantic_state, "semantic_tags", [])),
            "helper_stage": str(graph_helper.get("promotion_stage", "heuristic_fallback")),
            "functional_roles": mapping(getattr(semantic_state, "functional_roles", {})),
            "risk_register": mapping(getattr(semantic_state, "risk_register", {})),
        },
    )


def _temporal_grounding(
    *,
    state_id: str,
    frame_index: int,
    scene_graph: SceneGraphState,
    semantic_state: Any,
    helper_status: Mapping[str, Any],
) -> TemporalGroundingState:
    topology = mapping(getattr(semantic_state, "topology", {}))
    total_tracks = max(scene_graph.object_count, int(_safe_float(topology.get("track_count"), scene_graph.object_count)))
    visible_tracks = scene_graph.object_count
    occluded_tracks = len(
        [
            track
            for track in scene_graph.object_tracks
            if "partially_occluded" in strings(track.metadata.get("state_tags"))
        ]
    )
    return TemporalGroundingState(
        grounding_id=f"temporal_grounding_{state_id}",
        frame_index=frame_index,
        total_tracks=total_tracks,
        visible_tracks=visible_tracks,
        occluded_tracks=occluded_tracks,
        lost_tracks=max(0, total_tracks - visible_tracks),
        recovered_tracks=int(_safe_float(topology.get("recovered_track_count"), 0.0)),
        id_switch_count=int(_safe_float(topology.get("id_switch_count"), 0.0)),
        temporal_coherence_score=clip01(_safe_float(topology.get("temporal_stability"), 0.0)),
        prediction_quality_score=clip01(_safe_float(topology.get("grounding_confidence"), 0.0)),
        memory_token_count=scene_graph.object_count,
        helper_posture=str(helper_status.get("posture", "auto")),
        helper_promotion_stage=str(helper_status.get("promotion_stage", "heuristic_fallback")),
        metadata={
            "source_world_model_id": getattr(semantic_state, "world_model_id", ""),
            "topology": topology,
        },
    )


def _invoke_provider_adapter_seam(
    *,
    state_id: str,
    provider_id: str,
    provider_kind: str,
    seam: Any,
    seam_input: Any,
    benchmark_signals: Mapping[str, Any],
) -> Tuple[Optional[Any], ProviderInvocationReceipt]:
    """Invoke a provider adapter seam and emit an invocation receipt.

    Args:
        state_id: State ID for receipt generation.
        provider_id: Provider identifier (e.g., "sam_3_1", "dinov2").
        provider_kind: Provider kind (e.g., "sam_calibration", "vision_backbone_projection").
        seam: The neural seam module (torch.nn.Module).
        seam_input: Input data for the seam forward pass.
        benchmark_signals: Benchmark signals for promotion resolution.

    Returns:
        Tuple of (seam output or None, ProviderInvocationReceipt).
    """
    import time

    helper_status = resolve_provider_adapter_helper(
        provider_kind=provider_kind,
        loading_posture="auto",
        benchmark_signals=benchmark_signals,
    )
    promotion_stage = str(helper_status.get("promotion_stage", "raw_provider_output"))

    # If not promoted or no seam, return fallback receipt
    if promotion_stage != "promoted" or seam is None:
        return None, ProviderInvocationReceipt(
            receipt_id=f"provider_invocation_{provider_id}_{state_id}",
            provider_id=provider_id,
            provider_kind=provider_kind,
            invocation_status="skipped",
            fallback_used=True,
            fallback_reason="seam_not_promoted" if seam is not None else "seam_not_available",
            metadata={
                "promotion_stage": promotion_stage,
                "helper_status": dict(helper_status),
            },
        )

    # Invoke the seam
    start_time = time.perf_counter()
    output = None
    invocation_status = "success"
    fallback_used = False
    fallback_reason = ""
    output_quality = 0.0
    output_token_count = 0

    try:
        import torch

        with torch.no_grad():
            output = seam(seam_input) if not isinstance(seam_input, tuple) else seam(*seam_input)

        # Extract quality/token count from output if available
        if isinstance(output, dict):
            if "calibrated_confidence" in output:
                output_quality = clip01(float(output["calibrated_confidence"].mean().item()))
            elif "temporal_confidence" in output:
                output_quality = clip01(float(output["temporal_confidence"].mean().item()))
            else:
                output_quality = 0.7  # Default for successful projection
        elif hasattr(output, "shape"):
            output_token_count = int(output.shape[0] if output.dim() >= 1 else 1)
            output_quality = 0.7

    except Exception as e:
        invocation_status = "error"
        fallback_used = True
        fallback_reason = str(e)[:200]
        output = None

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    receipt = ProviderInvocationReceipt(
        receipt_id=f"provider_invocation_{provider_id}_{state_id}",
        provider_id=provider_id,
        provider_kind=provider_kind,
        invocation_status=invocation_status,
        output_quality_score=output_quality,
        latency_ms=latency_ms,
        output_token_count=output_token_count,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        metadata={
            "promotion_stage": promotion_stage,
            "seam_type": type(seam).__name__ if seam else "none",
            "helper_status": dict(helper_status),
        },
    )

    return output, receipt


def _evidence_routing(
    *,
    state_id: str,
    belief_state: BeliefState,
    provider_surface: ProviderSurfaceState,
    helper_status: Mapping[str, Any],
    object_count: int,
    edge_count: int,
    evidence_fusion_seam: Optional[Any] = None,
) -> Tuple[EvidenceRoutingState, EvidenceFusionReceipt]:
    """Route and fuse provider evidence, branching on promotion stage.

    When ``promotion_stage == "promoted"`` and a neural seam is provided,
    the seam forward pass produces fusion weights and confidence.
    Otherwise the heuristic weighted-sum path executes as before.

    Always emits an ``EvidenceFusionReceipt`` recording which path was
    taken, the resulting weights, and confidence.
    """
    provider_ids = list(provider_surface.provider_ids)
    promotion_stage = str(
        helper_status.get("promotion_stage", "heuristic_fallback")
    )
    fusion_method = "semantic_world_model_heuristic_fusion"
    neural_seam_used = False

    # Extract belief signals (used by both paths)
    semantic_quality = clip01(
        _safe_float(belief_state.state_vector.get("semantic_quality"), 0.0)
    )
    coverage = clip01(
        _safe_float(belief_state.state_vector.get("evidence_coverage"), 0.0)
    )
    disagreement = clip01(
        _safe_float(
            belief_state.state_vector.get("evidence_disagreement_mean"), 0.0
        )
    )

    # --- Neural seam path (promoted + seam available) ---
    if promotion_stage == "promoted" and evidence_fusion_seam is not None:
        try:
            import torch

            from .neural_seams import encode_provider_features

            provider_features = encode_provider_features(
                provider_ids=provider_ids,
                provider_kinds=dict(provider_surface.provider_kinds),
                provider_availability=dict(provider_surface.provider_availability),
                provider_truth_class=dict(provider_surface.provider_truth_class),
                semantic_quality=semantic_quality,
                coverage=coverage,
                disagreement=disagreement,
                object_count_norm=clip01(object_count / 8.0),
                edge_count_norm=clip01(edge_count / 8.0),
            )

            with torch.no_grad():
                weights_tensor, confidence_tensor = evidence_fusion_seam(
                    provider_features
                )

            weights_list = weights_tensor.tolist()
            normalized = {
                pid: float(w)
                for pid, w in zip(provider_ids, weights_list)
            }
            fusion_confidence = clip01(float(confidence_tensor.item()))
            fusion_method = "neural_evidence_fusion_seam"
            neural_seam_used = True
        except Exception:
            # Any neural seam failure → fall through to heuristic
            neural_seam_used = False

    # --- Heuristic fallback path ---
    if not neural_seam_used:
        contributions: dict[str, float] = {}
        if "scene_tracks" in provider_ids:
            contributions["scene_tracks"] = 0.55
        if "vla_semantic_evidence" in provider_ids:
            contributions["vla_semantic_evidence"] = 0.25
        if "teacher_trace" in provider_ids:
            contributions["teacher_trace"] = 0.15
        if "vision_backbone_stub" in provider_ids:
            contributions["vision_backbone_stub"] = 0.05
        contribution_total = sum(contributions.values()) or 1.0
        normalized = {
            key: value / contribution_total
            for key, value in contributions.items()
        }
        structure_bonus = clip01(
            0.5 * min(object_count, 6) / 6.0
            + 0.5 * min(edge_count, 8) / 8.0
        )
        fusion_confidence = clip01(
            0.45 * semantic_quality + 0.25 * coverage + 0.3 * structure_bonus
        )

    routing_state = EvidenceRoutingState(
        routing_id=f"evidence_routing_{state_id}",
        provider_contributions=normalized,
        fusion_method=fusion_method,
        fusion_confidence=fusion_confidence,
        fusion_disagreement=disagreement,
        provider_availability=provider_surface.provider_availability,
        helper_posture=str(helper_status.get("posture", "auto")),
        helper_promotion_stage=promotion_stage,
        metadata={
            "belief_id": belief_state.belief_id,
            "semantic_quality": semantic_quality,
            "coverage": coverage,
            "neural_seam_used": neural_seam_used,
        },
    )

    receipt = EvidenceFusionReceipt(
        receipt_id=f"evidence_fusion_receipt_{state_id}",
        fusion_method=fusion_method,
        provider_ids=provider_ids,
        provider_weights=dict(normalized),
        fusion_confidence=fusion_confidence,
        fusion_disagreement=disagreement,
        output_object_count=object_count,
        output_edge_count=edge_count,
        helper_posture=str(helper_status.get("posture", "auto")),
        metadata={
            "neural_seam_used": neural_seam_used,
            "promotion_stage": promotion_stage,
        },
    )

    return routing_state, receipt


def _semantic_bridge_registry(
    *,
    state_id: str,
    scene_graph: SceneGraphState,
    semantic_state: Any,
    evidence_routing: EvidenceRoutingState,
    deployment_resource_surface: DeploymentResourceSurface,
    benchmark_signals: Mapping[str, Any],
) -> SemanticBridgeRegistry:
    objects = scene_graph.object_tracks
    edges = scene_graph.edges
    tag_set = set(strings(getattr(semantic_state, "semantic_tags", [])))
    helper_sim = resolve_semantic_bridge_helper(
        bridge_kind="sim_synth",
        loading_posture="auto",
        benchmark_signals=benchmark_signals,
    )
    helper_embodiment = resolve_semantic_bridge_helper(
        bridge_kind="embodiment",
        loading_posture="auto",
        benchmark_signals=benchmark_signals,
    )
    helper_annotation = resolve_semantic_bridge_helper(
        bridge_kind="annotation",
        loading_posture="auto",
        benchmark_signals=benchmark_signals,
    )
    helper_economic = resolve_semantic_bridge_helper(
        bridge_kind="economic",
        loading_posture="auto",
        benchmark_signals=benchmark_signals,
    )
    branch_relevance_scores = [
        clip01(
            0.25
            + 0.25 * float(bool(track.affordance_hints))
            + 0.2 * float(bool(track.risk_hints))
            + 0.15 * float(track.object_category in {"container", "fragile_object", "manipulated_object"})
            + 0.15 * float("object:" in f"object:{track.object_label}")
        )
        for track in objects
    ]
    object_preservation_scores = [
        clip01(0.55 * track.confidence + 0.3 * (1.0 - track.epistemic_uncertainty) + 0.15 * track.visibility)
        for track in objects
    ]
    physics_edge_weights = {
        edge.edge_id: clip01(
            edge.confidence
            * (
                1.0
                if edge.edge_type in {"contact", "acts_on"}
                else 0.8 if edge.edge_type in {"containment", "supports"}
                else 0.5
            )
        )
        for edge in edges
    }
    sim_bridge = SimSynthSemanticBridgeState(
        bridge_id=f"sim_synth_bridge_{state_id}",
        source_graph_id=scene_graph.graph_id,
        physics_object_tokens=[track.feature_token for track in objects],
        physics_edge_weights=physics_edge_weights,
        branch_relevance_scores=branch_relevance_scores,
        object_preservation_scores=object_preservation_scores,
        diffusion_conditioning_features=_dense_token(
            [
                clip01(np.mean(branch_relevance_scores) if branch_relevance_scores else 0.0),
                clip01(np.mean(object_preservation_scores) if object_preservation_scores else 0.0),
                evidence_routing.fusion_confidence,
                scene_graph.graph_density,
            ],
            target_dim=8,
        ),
        contact_topology_summary={
            "contact_like_edge_count": len(
                [edge for edge in edges if edge.edge_type in {"contact", "acts_on", "supports"}]
            ),
            "bridge_preconditions": [
                "object_preservation",
                "branch_evaluation",
                "diffusion_conditioning",
            ],
        },
        helper_posture=str(helper_sim.get("posture", "auto")),
        helper_promotion_stage=str(helper_sim.get("promotion_stage", "heuristic_fallback")),
        metadata={"downstream_wm": "sim_synth_physics"},
    )
    embodiment_bridge = EmbodimentSemanticBridgeState(
        bridge_id=f"embodiment_bridge_{state_id}",
        source_graph_id=scene_graph.graph_id,
        per_object_affordance_scores={
            track.track_id: clip01(0.35 + 0.15 * len(track.affordance_hints) + 0.2 * track.confidence)
            for track in objects
        },
        per_object_affordance_classes={
            track.track_id: track.affordance_hints for track in objects
        },
        body_object_pairwise_scores={
            "g1_default_body": {
                track.track_id: clip01(
                    0.3 + 0.2 * len(track.affordance_hints) + 0.15 * track.visibility
                )
                for track in objects
            }
        },
        action_feasibility_summary={
            "deployment_posture": deployment_resource_surface.deployment_posture,
            "bridge_preconditions": [
                "affordance_projection",
                "action_relevance",
                "body_object_pairing",
            ],
        },
        resource_conditioned=deployment_resource_surface.deployment_posture != "unavailable",
        embodiment_dof=29,
        helper_posture=str(helper_embodiment.get("posture", "auto")),
        helper_promotion_stage=str(helper_embodiment.get("promotion_stage", "heuristic_fallback")),
        metadata={"downstream_wm": "embodiment_actuation"},
    )
    annotation_bridge = AnnotationSemanticBridgeState(
        bridge_id=f"annotation_bridge_{state_id}",
        source_graph_id=scene_graph.graph_id,
        object_class_labels={track.track_id: track.object_label for track in objects},
        object_confidence_scores={track.track_id: track.confidence for track in objects},
        object_affordance_hints={track.track_id: track.affordance_hints for track in objects},
        object_risk_hints={track.track_id: track.risk_hints for track in objects},
        primitive_segment_alignment_scores=[
            clip01(0.45 + 0.2 * float(bool(track.affordance_hints)) + 0.15 * track.confidence)
            for track in objects
        ],
        object_event_labels={
            track.track_id: [
                *[f"affordance:{hint}" for hint in track.affordance_hints],
                *[f"risk:{hint}" for hint in track.risk_hints],
            ]
            for track in objects
        },
        failure_interpretation_tags=sorted(
            {
                "risk:fragility" if any("fragility" in track.risk_hints for track in objects) else "",
                "recovery_needed" if "mode:recovery" in tag_set else "",
            }
            - {""}
        ),
        recovery_interpretation_tags=sorted(
            {"mode:recovery"} if "mode:recovery" in tag_set else set()
        ),
        teacher_alignment_score=clip01(
            _safe_float(getattr(semantic_state, "capability_scores", {}).get("teacher_alignment"), 0.0)
            or _safe_float(getattr(semantic_state, "topology", {}).get("grounding_confidence"), 0.0)
        ),
        helper_posture=str(helper_annotation.get("posture", "auto")),
        helper_promotion_stage=str(helper_annotation.get("promotion_stage", "heuristic_fallback")),
        metadata={
            "downstream_wm": "annotation_evidence",
            "bridge_preconditions": [
                "rollout_labeling",
                "failure_recovery_labeling",
                "semantic_dataset_crosswalk",
            ],
        },
    )
    economic_bridge = EconomicSemanticBridgeState(
        bridge_id=f"economic_bridge_{state_id}",
        source_graph_id=scene_graph.graph_id,
        economic_summary_token=_dense_token(
            [
                clip01(scene_graph.object_count / 8.0),
                clip01(scene_graph.edge_count / 8.0),
                clip01(len(tag_set) / 12.0),
                evidence_routing.fusion_confidence,
            ],
            target_dim=8,
        ),
        semantic_density=clip01((scene_graph.object_count + scene_graph.edge_count) / 12.0),
        object_diversity=clip01(len({track.object_category for track in objects}) / 6.0),
        affordance_richness=clip01(sum(len(track.affordance_hints) for track in objects) / 10.0),
        grounding_confidence=evidence_routing.fusion_confidence,
        temporal_stability=clip01(
            _safe_float(getattr(semantic_state, "topology", {}).get("temporal_stability"), 0.0)
        ),
        concept_coverage=clip01(len(tag_set) / 12.0),
        num_query_tokens=16,
        helper_posture=str(helper_economic.get("posture", "auto")),
        helper_promotion_stage=str(helper_economic.get("promotion_stage", "heuristic_fallback")),
        metadata={
            "downstream_wm": "economic",
            "bridge_preconditions": [
                "grounding_quality_pricing",
                "semantic_contribution_estimation",
                "allocation_governance_inputs",
            ],
        },
    )
    return SemanticBridgeRegistry(
        registry_id=f"semantic_bridge_registry_{state_id}",
        source_graph_id=scene_graph.graph_id,
        sim_synth_bridge=sim_bridge,
        embodiment_bridge=embodiment_bridge,
        annotation_bridge=annotation_bridge,
        economic_bridge=economic_bridge,
        semantic_vla_successor_status="distributed_bridge_family",
        metadata={
            "source_world_model_id": getattr(semantic_state, "world_model_id", ""),
            "bridge_family_mode": "heuristic_shadow_runtime",
        },
    )


def compile_perception_grounding_world_state(
    *,
    episode_id: str,
    task_id: str,
    frame_index: int = 0,
    semantic_tags: Optional[Sequence[Any]] = None,
    belief_state: Optional[BeliefState] = None,
    scene_tracks_payload: Optional[Any] = None,
    teacher_trace: Optional[Any] = None,
    vla_semantic_evidence: Optional[Any] = None,
    semantic_fusion_summary: Optional[Mapping[str, Any]] = None,
    deployment_resource_surface: Optional[DeploymentResourceSurface] = None,
    benchmark_signals: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    objective_preset: str = "balanced",
    # Neural seams (optional, invoked when promoted)
    evidence_fusion_seam: Optional[Any] = None,
    sam_calibration_seam: Optional[Any] = None,
    vision_backbone_projection_seam: Optional[Any] = None,
    depth_metric_calibration_seam: Optional[Any] = None,
    vjepa_temporal_alignment_seam: Optional[Any] = None,
    # Provider adapter inputs (optional, passed to seams when available)
    sam_mask_features: Optional[Any] = None,
    sam_raw_confidence: Optional[Any] = None,
    backbone_features: Optional[Any] = None,
    depth_map: Optional[Any] = None,
    camera_intrinsics: Optional[Any] = None,
    vjepa_tokens: Optional[Any] = None,
    wm_object_tokens: Optional[Any] = None,
) -> PerceptionGroundingWorldState:
    """Compile the first loop-facing Perception / Grounding WM state.

    Provider adapter seams are invoked when:
    - The seam is provided (not None)
    - The corresponding input data is provided
    - The seam's promotion posture is "promoted" based on benchmark signals

    When invoked, seams emit ProviderInvocationReceipt to the receipts list.
    """

    resolved_tags = strings(semantic_tags)
    resolved_belief = belief_state or _empty_belief_state(
        episode_id=episode_id,
        semantic_tags=resolved_tags,
        metadata=metadata,
    )
    benchmark_payload = mapping(benchmark_signals)
    semantic_builder = SemanticWorldModelBuilder()
    semantic_state = semantic_builder.build_from_runtime_fusion(
        episode_id=episode_id,
        task_id=task_id,
        objective_preset=objective_preset,
        belief_state=resolved_belief,
        semantic_tags=resolved_tags,
        scene_tracks_payload=scene_tracks_payload,
        teacher_trace=teacher_trace,
        vla_semantic_evidence=vla_semantic_evidence,
        semantic_fusion_summary=semantic_fusion_summary,
        artifact_refs=resolved_belief.artifact_refs,
        metadata=metadata,
    )
    state_id = f"perception_grounding_{semantic_state.world_model_id}"
    graph_helper = resolve_graph_transformer_helper(
        loading_posture="auto",
        benchmark_signals=benchmark_payload,
    )
    scene_graph = _scene_graph(semantic_state, graph_helper)
    provider_surface = _provider_surface(
        state_id=state_id,
        scene_tracks_payload=scene_tracks_payload,
        vla_semantic_evidence=vla_semantic_evidence,
        teacher_trace=teacher_trace,
    )
    evidence_helper = resolve_evidence_fusion_helper(
        loading_posture="auto",
        benchmark_signals=benchmark_payload,
    )
    evidence_routing, evidence_fusion_receipt = _evidence_routing(
        state_id=state_id,
        belief_state=resolved_belief,
        provider_surface=provider_surface,
        helper_status=evidence_helper,
        object_count=scene_graph.object_count,
        edge_count=scene_graph.edge_count,
        evidence_fusion_seam=evidence_fusion_seam,
    )
    temporal_helper = resolve_temporal_grounding_helper(
        loading_posture="auto",
        benchmark_signals=benchmark_payload,
    )

    # --- Invoke provider adapter seams ---
    provider_adapter_receipts: List[ProviderInvocationReceipt] = []
    provider_adapter_outputs: dict[str, Any] = {}

    # SAM calibration seam
    if sam_calibration_seam is not None and sam_mask_features is not None:
        seam_input = (sam_mask_features, sam_raw_confidence) if sam_raw_confidence is not None else (sam_mask_features,)
        output, receipt = _invoke_provider_adapter_seam(
            state_id=state_id,
            provider_id="sam_3_1",
            provider_kind="sam_calibration",
            seam=sam_calibration_seam,
            seam_input=seam_input,
            benchmark_signals=benchmark_payload,
        )
        provider_adapter_receipts.append(receipt)
        if output is not None:
            provider_adapter_outputs["sam_calibration"] = output

    # Vision backbone projection seam
    if vision_backbone_projection_seam is not None and backbone_features is not None:
        output, receipt = _invoke_provider_adapter_seam(
            state_id=state_id,
            provider_id="dinov2_vit_l_14",
            provider_kind="vision_backbone_projection",
            seam=vision_backbone_projection_seam,
            seam_input=backbone_features,
            benchmark_signals=benchmark_payload,
        )
        provider_adapter_receipts.append(receipt)
        if output is not None:
            provider_adapter_outputs["vision_backbone_projection"] = output

    # Depth metric calibration seam
    if depth_metric_calibration_seam is not None and depth_map is not None:
        seam_input = (depth_map, camera_intrinsics) if camera_intrinsics is not None else (depth_map,)
        output, receipt = _invoke_provider_adapter_seam(
            state_id=state_id,
            provider_id="depth_anything_v2",
            provider_kind="depth_metric_calibration",
            seam=depth_metric_calibration_seam,
            seam_input=seam_input,
            benchmark_signals=benchmark_payload,
        )
        provider_adapter_receipts.append(receipt)
        if output is not None:
            provider_adapter_outputs["depth_metric_calibration"] = output

    # V-JEPA temporal alignment seam
    if vjepa_temporal_alignment_seam is not None and vjepa_tokens is not None:
        # wm_object_tokens defaults to scene_graph tokens if not provided
        wm_tokens = wm_object_tokens
        if wm_tokens is None and scene_graph.object_tracks:
            import torch
            wm_tokens = torch.tensor(
                [track.feature_token for track in scene_graph.object_tracks],
                dtype=torch.float32,
            )
        if wm_tokens is not None:
            output, receipt = _invoke_provider_adapter_seam(
                state_id=state_id,
                provider_id="vjepa2",
                provider_kind="vjepa_temporal_alignment",
                seam=vjepa_temporal_alignment_seam,
                seam_input=(vjepa_tokens, wm_tokens),
                benchmark_signals=benchmark_payload,
            )
            provider_adapter_receipts.append(receipt)
            if output is not None:
                provider_adapter_outputs["vjepa_temporal_alignment"] = output

    deployment_surface = deployment_resource_surface or DeploymentResourceSurface(
        surface_id=f"deployment_resource_{state_id}",
        deployment_posture="unavailable",
        metadata={"source": "perception_grounding_compiler_default"},
    )
    bridge_registry = _semantic_bridge_registry(
        state_id=state_id,
        scene_graph=scene_graph,
        semantic_state=semantic_state,
        evidence_routing=evidence_routing,
        deployment_resource_surface=deployment_surface,
        benchmark_signals=benchmark_payload,
    )
    return PerceptionGroundingWorldState(
        state_id=state_id,
        frame_index=frame_index,
        episode_id=episode_id,
        scene_graph=scene_graph,
        temporal_grounding=_temporal_grounding(
            state_id=state_id,
            frame_index=frame_index,
            scene_graph=scene_graph,
            semantic_state=semantic_state,
            helper_status=temporal_helper,
        ),
        evidence_routing=evidence_routing,
        provider_surface=provider_surface,
        dataset_surface=_dataset_surface(
            state_id=state_id,
            scene_tracks_payload=scene_tracks_payload,
            metadata=metadata,
        ),
        task_measurements=_task_measurements(
            state_id=state_id,
            task_id=task_id,
            semantic_state=semantic_state,
            object_count=scene_graph.object_count,
            edge_count=scene_graph.edge_count,
            fusion_confidence=evidence_routing.fusion_confidence,
        ),
        deployment_resource_surface=deployment_surface,
        semantic_bridge_registry=bridge_registry,
        input_context={
            "belief_id": resolved_belief.belief_id,
            "semantic_world_model_id": semantic_state.world_model_id,
            "semantic_tags": list(getattr(semantic_state, "semantic_tags", [])),
        },
        maturity_stage="shadow_runtime",
        metadata={
            "source_world_model_id": semantic_state.world_model_id,
            "semantic_topology": mapping(getattr(semantic_state, "topology", {})),
            "semantic_capability_scores": mapping(getattr(semantic_state, "capability_scores", {})),
            "semantic_functional_roles": mapping(getattr(semantic_state, "functional_roles", {})),
            "semantic_risk_register": mapping(getattr(semantic_state, "risk_register", {})),
            "graph_helper_status": graph_helper,
            "evidence_helper_status": evidence_helper,
            "temporal_helper_status": temporal_helper,
            "evidence_fusion_receipt": evidence_fusion_receipt.to_dict(),
            "provider_adapter_receipts": [r.to_dict() for r in provider_adapter_receipts],
            "provider_adapter_outputs_available": list(provider_adapter_outputs.keys()),
            **mapping(metadata),
        },
    )


# ---------------------------------------------------------------------------
# Compilation result with receipts
# ---------------------------------------------------------------------------


@dataclass
class PerceptionCompilationResult:
    """Result of perception grounding compilation with explicit receipts.

    The ``state`` field is the canonical ``PerceptionGroundingWorldState``.
    The ``receipts`` field carries typed receipt objects from the
    compilation pass — primarily ``EvidenceFusionReceipt`` now, with
    provider/deployment/bridge receipts added in later tranches.

    Downstream consumers that only need state can ignore receipts.
    Training/replay/promotion-gate consumers should inspect receipts.
    """

    state: PerceptionGroundingWorldState
    receipts: List[Any] = field(default_factory=list)


def compile_perception_grounding_with_receipts(
    **kwargs: Any,
) -> PerceptionCompilationResult:
    """Compile Perception / Grounding WM state and return receipts.

    Accepts the same keyword arguments as
    ``compile_perception_grounding_world_state``.  Returns a
    ``PerceptionCompilationResult`` with both the state and the
    typed receipts from the compilation pass.

    Receipts include:
    - EvidenceFusionReceipt: evidence routing and fusion quality
    - ProviderInvocationReceipt: per-seam invocation status and quality
    """
    state = compile_perception_grounding_world_state(**kwargs)
    receipts: list[Any] = []

    # Extract the evidence fusion receipt from state metadata
    efr_dict = state.metadata.get("evidence_fusion_receipt")
    if efr_dict is not None:
        receipts.append(
            EvidenceFusionReceipt(
                receipt_id=str(efr_dict.get("receipt_id", "")),
                fusion_method=str(efr_dict.get("fusion_method", "")),
                provider_ids=list(efr_dict.get("provider_ids", [])),
                provider_weights=dict(efr_dict.get("provider_weights", {})),
                fusion_confidence=float(efr_dict.get("fusion_confidence", 0.0)),
                fusion_disagreement=float(
                    efr_dict.get("fusion_disagreement", 0.0)
                ),
                output_object_count=int(
                    efr_dict.get("output_object_count", 0)
                ),
                output_edge_count=int(efr_dict.get("output_edge_count", 0)),
                helper_posture=str(efr_dict.get("helper_posture", "")),
                metadata=dict(efr_dict.get("metadata", {})),
            )
        )

    # Extract provider adapter receipts from state metadata
    par_list = state.metadata.get("provider_adapter_receipts", [])
    for par_dict in par_list:
        if isinstance(par_dict, dict):
            receipts.append(
                ProviderInvocationReceipt(
                    receipt_id=str(par_dict.get("receipt_id", "")),
                    provider_id=str(par_dict.get("provider_id", "")),
                    provider_kind=str(par_dict.get("provider_kind", "")),
                    invocation_status=str(par_dict.get("invocation_status", "")),
                    output_quality_score=float(par_dict.get("output_quality_score", 0.0)),
                    latency_ms=float(par_dict.get("latency_ms", 0.0)),
                    output_token_count=int(par_dict.get("output_token_count", 0)),
                    fallback_used=bool(par_dict.get("fallback_used", False)),
                    fallback_reason=str(par_dict.get("fallback_reason", "")),
                    metadata=dict(par_dict.get("metadata", {})),
                )
            )

    return PerceptionCompilationResult(state=state, receipts=receipts)


__all__ = [
    "PerceptionCompilationResult",
    "compile_perception_grounding_with_receipts",
    "compile_perception_grounding_world_state",
]
