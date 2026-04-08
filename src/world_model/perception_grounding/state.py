"""Typed state objects for the Perception / Grounding world model.

This module defines the canonical state surfaces owned by the Perception /
Grounding WM.  Every state object follows the frozen-dataclass-with-
serialization pattern established by ``sim_synth_physics/state.py``.

Ownership boundaries
--------------------
- Canonical scene/grounding state is WM-owned.
- Provider outputs (SAM masks, DINOv2 features, depth maps, V-JEPA predictions)
  enter through typed provider contracts and are fused into WM-owned state.
- Habitat-style dataset/world inventory, provider/runtime, and measurement
  surfaces are represented as typed lower-WM state, not as a giant env object.
- Deployment and resource posture is typed lower-WM state first; only later does
  the Economic WM elevate those surfaces into allocatable objects.
- External providers are never native truth owners.

Neuralization placement
-----------------------
- Graph Transformer (WM-native) operates over ``ObjectTrackState`` tokens and
  ``SceneEdge`` edges to produce the canonical scene graph.
- Provider adapters (provider-backed interpretation layers) produce calibrated
  object tokens from frozen backbones.
- Temporal grounding module (WM-native causal transformer) maintains object
  persistence across frames.
- Evidence fusion module (WM-native set transformer) merges heterogeneous
  provider evidence tokens into canonical object state.
- All learned components expose ``disabled|auto|required`` promotion posture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import clip01, mapping, strings
from .semantic_bridges import SemanticBridgeRegistry


# ---------------------------------------------------------------------------
# Per-object canonical track state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectTrackState:
    """Canonical per-object persistent track state.

    This is the atomic representational unit of the Perception / Grounding WM.
    Each object has a persistent track across frames with:
    - 3D pose (4x4 homogeneous, or empty if 2D-only)
    - provider-fused feature token (d=128, or raw confidence if pre-fusion)
    - identity / label / category information
    - calibrated confidence and uncertainty from provider fusion
    - temporal persistence metadata (first-seen, last-seen, occlusion state)

    Downstream consumers:
    - Graph Transformer consumes these as node features
    - SimSynth WM consumes for branch evaluation (object preservation)
    - Embodiment WM consumes for affordance estimation
    - Annotation bridge consumes for rollout labeling
    """

    track_id: str
    object_label: str
    object_category: str
    confidence: float
    epistemic_uncertainty: float
    pose_3d: List[float] = field(default_factory=list)
    scale_3d: List[float] = field(default_factory=list)
    feature_token: List[float] = field(default_factory=list)
    provider_sources: List[str] = field(default_factory=list)
    visibility: float = 1.0
    occlusion_score: float = 0.0
    temporal_persistence_frames: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    occlusion_state: str = "visible"
    reidentification_confidence: float = 1.0
    affordance_hints: List[str] = field(default_factory=list)
    risk_hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "object_track_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_label": self.object_label,
            "object_category": self.object_category,
            "confidence": clip01(self.confidence),
            "epistemic_uncertainty": clip01(self.epistemic_uncertainty),
            "pose_3d": [float(v) for v in self.pose_3d],
            "scale_3d": [float(v) for v in self.scale_3d],
            "feature_token": [float(v) for v in self.feature_token],
            "provider_sources": strings(self.provider_sources),
            "visibility": clip01(self.visibility),
            "occlusion_score": clip01(self.occlusion_score),
            "temporal_persistence_frames": int(self.temporal_persistence_frames),
            "first_seen_frame": int(self.first_seen_frame),
            "last_seen_frame": int(self.last_seen_frame),
            "occlusion_state": self.occlusion_state,
            "reidentification_confidence": clip01(self.reidentification_confidence),
            "affordance_hints": strings(self.affordance_hints),
            "risk_hints": strings(self.risk_hints),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Scene graph edges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneEdge:
    """Typed spatial/temporal edge between two tracked objects.

    Edge types follow the neuralization doctrine:
    - spatial_adjacency: objects within proximity threshold
    - contact: objects in physical contact (from depth/physics)
    - containment: one object inside another (drawer contains cup)
    - occlusion: one object occluding another
    - temporal_co_occurrence: objects consistently co-present
    - affordance_relation: functional relation (handle affords grasp)

    These become explicit edge types in the Graph Transformer attention
    structure, with learned edge embeddings (d=64) per type.
    """

    edge_id: str
    source_track_id: str
    target_track_id: str
    edge_type: str
    confidence: float
    spatial_distance: float = 0.0
    edge_features: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "scene_edge_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_track_id": self.source_track_id,
            "target_track_id": self.target_track_id,
            "edge_type": self.edge_type,
            "confidence": clip01(self.confidence),
            "spatial_distance": float(self.spatial_distance),
            "edge_features": [float(v) for v in self.edge_features],
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Temporal grounding state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalGroundingState:
    """Temporal persistence and continuity state for the scene.

    Owned by the causal transformer temporal grounding module.
    Tracks scene-level temporal coherence, identity stability,
    and predictive state quality.

    Neuralization: 3-10M param causal transformer. Governed locally
    by Perception/Grounding WM. Promotion posture: disabled|auto|required.
    """

    grounding_id: str
    frame_index: int
    total_tracks: int
    visible_tracks: int
    occluded_tracks: int
    lost_tracks: int
    recovered_tracks: int
    id_switch_count: int = 0
    temporal_coherence_score: float = 0.0
    prediction_quality_score: float = 0.0
    memory_token_count: int = 0
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "temporal_grounding_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounding_id": self.grounding_id,
            "frame_index": int(self.frame_index),
            "total_tracks": int(self.total_tracks),
            "visible_tracks": int(self.visible_tracks),
            "occluded_tracks": int(self.occluded_tracks),
            "lost_tracks": int(self.lost_tracks),
            "recovered_tracks": int(self.recovered_tracks),
            "id_switch_count": int(self.id_switch_count),
            "temporal_coherence_score": clip01(self.temporal_coherence_score),
            "prediction_quality_score": clip01(self.prediction_quality_score),
            "memory_token_count": int(self.memory_token_count),
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Scene-level summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneGraphState:
    """Canonical scene graph: objects + edges + scene summary.

    This is the primary output of the Graph Transformer and the
    canonical scene representation consumed by all downstream bridges.

    Output dimensions per neuralization doctrine:
    - Per-object token: d=128
    - Per-edge embedding: d=64
    - Scene summary token: d=256
    """

    graph_id: str
    object_tracks: List[ObjectTrackState] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)
    scene_summary_token: List[float] = field(default_factory=list)
    object_count: int = 0
    edge_count: int = 0
    edge_type_counts: Dict[str, int] = field(default_factory=dict)
    graph_density: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "scene_graph_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "object_tracks": [t.to_dict() for t in self.object_tracks],
            "edges": [e.to_dict() for e in self.edges],
            "scene_summary_token": [float(v) for v in self.scene_summary_token],
            "object_count": int(self.object_count),
            "edge_count": int(self.edge_count),
            "edge_type_counts": {str(k): int(v) for k, v in self.edge_type_counts.items()},
            "graph_density": float(self.graph_density),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Habitat-inspired provider / dataset / measurement surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSurfaceState:
    """Typed provider/runtime inventory for the current perception window.

    This is the Perception/Grounding WM's explicit provider-surface layer:
    the equivalent of the simulator/provider inventory concept, but kept
    under WM-owned typed state instead of an environment object.

    Downstream consumers:
    - live provider bring-up and runtime planners
    - replay/training export surfaces
    - later deployment-resource and Economic WM ingestion
    """

    surface_id: str
    provider_ids: List[str] = field(default_factory=list)
    provider_kinds: Dict[str, str] = field(default_factory=dict)
    provider_availability: Dict[str, str] = field(default_factory=dict)
    provider_truth_class: Dict[str, str] = field(default_factory=dict)
    sensor_modalities: Dict[str, List[str]] = field(default_factory=dict)
    vectorized_runtime_supported: bool = False
    provider_batch_capacity: int = 0
    provider_latency_budget_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "provider_surface_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "provider_ids": strings(self.provider_ids),
            "provider_kinds": {str(k): str(v) for k, v in self.provider_kinds.items()},
            "provider_availability": {
                str(k): str(v) for k, v in self.provider_availability.items()
            },
            "provider_truth_class": {
                str(k): str(v) for k, v in self.provider_truth_class.items()
            },
            "sensor_modalities": {
                str(k): strings(v) for k, v in self.sensor_modalities.items()
            },
            "vectorized_runtime_supported": bool(self.vectorized_runtime_supported),
            "provider_batch_capacity": int(self.provider_batch_capacity),
            "provider_latency_budget_ms": float(self.provider_latency_budget_ms),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class DatasetSurfaceState:
    """Typed dataset/world inventory surface for perception inputs.

    Habitat contributes the useful pattern here: dataset/world inventory is
    a separate concern from provider execution. The WM owns the typed summary.
    """

    surface_id: str
    dataset_ids: List[str] = field(default_factory=list)
    world_inventory_ids: List[str] = field(default_factory=list)
    split_name: str = ""
    sensor_inventory: List[str] = field(default_factory=list)
    scene_hierarchy_levels: List[str] = field(default_factory=list)
    available_sequences: int = 0
    calibration_assets_present: bool = False
    annotation_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "dataset_surface_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "dataset_ids": strings(self.dataset_ids),
            "world_inventory_ids": strings(self.world_inventory_ids),
            "split_name": self.split_name,
            "sensor_inventory": strings(self.sensor_inventory),
            "scene_hierarchy_levels": strings(self.scene_hierarchy_levels),
            "available_sequences": int(self.available_sequences),
            "calibration_assets_present": bool(self.calibration_assets_present),
            "annotation_sources": strings(self.annotation_sources),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class TaskMeasurementSurface:
    """Typed task/measurement surface for perception quality and eval.

    This keeps measurements explicit and replayable instead of hiding them
    inside ad hoc eval code. The pattern is inspired by Habitat's measures,
    but the ownership remains WM-native.
    """

    surface_id: str
    task_id: str
    measurement_names: List[str] = field(default_factory=list)
    measurement_values: Dict[str, float] = field(default_factory=dict)
    measurement_status: Dict[str, str] = field(default_factory=dict)
    vector_env_count: int = 0
    measurement_window_frames: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "task_measurement_surface_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "task_id": self.task_id,
            "measurement_names": strings(self.measurement_names),
            "measurement_values": {
                str(k): float(v) for k, v in self.measurement_values.items()
            },
            "measurement_status": {
                str(k): str(v) for k, v in self.measurement_status.items()
            },
            "vector_env_count": int(self.vector_env_count),
            "measurement_window_frames": int(self.measurement_window_frames),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Evidence routing state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRoutingState:
    """Evidence routing ownership for the Perception / Grounding WM.

    Tracks which providers contributed, how evidence was fused, and
    what confidence/uncertainty the fusion produced.  This is the
    typed metadata that downstream consumers use to assess perception
    quality.

    Neuralization: Set Transformer (Perceiver-style), 2-5M params.
    Governed locally.  Promotion: disabled → heuristic fusion (existing
    semantic_fusion.py MVP) → learned fusion.
    """

    routing_id: str
    provider_contributions: Dict[str, float] = field(default_factory=dict)
    fusion_method: str = "heuristic_weighted"
    fusion_confidence: float = 0.0
    fusion_disagreement: float = 0.0
    provider_availability: Dict[str, str] = field(default_factory=dict)
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "evidence_routing_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_id": self.routing_id,
            "provider_contributions": {str(k): float(v) for k, v in self.provider_contributions.items()},
            "fusion_method": self.fusion_method,
            "fusion_confidence": clip01(self.fusion_confidence),
            "fusion_disagreement": clip01(self.fusion_disagreement),
            "provider_availability": {str(k): str(v) for k, v in self.provider_availability.items()},
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Deployment/resource posture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeEnvelopeState:
    """Onboard/companion compute availability for perception runtime."""

    profile_id: str
    on_device_available: bool = False
    companion_available: bool = False
    placement_class: str = "on_device_only"
    latency_budget_ms: float = 0.0
    qos_class: str = "best_effort"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "compute_envelope_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "on_device_available": bool(self.on_device_available),
            "companion_available": bool(self.companion_available),
            "placement_class": self.placement_class,
            "latency_budget_ms": float(self.latency_budget_ms),
            "qos_class": self.qos_class,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class InferenceCapacityState:
    """Inference headroom and provider-capacity posture."""

    profile_id: str
    provider_capacity_by_id: Dict[str, float] = field(default_factory=dict)
    headroom_fraction: float = 0.0
    batch_headroom: int = 0
    max_parallel_providers: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "inference_capacity_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "provider_capacity_by_id": {
                str(k): float(v) for k, v in self.provider_capacity_by_id.items()
            },
            "headroom_fraction": clip01(self.headroom_fraction),
            "batch_headroom": int(self.batch_headroom),
            "max_parallel_providers": int(self.max_parallel_providers),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BatteryState:
    """Concrete battery posture relevant to perception inference."""

    battery_id: str
    charge_fraction: float = 0.0
    reserve_fraction: float = 0.0
    projected_runtime_minutes: float = 0.0
    charging_state: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "battery_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_id": self.battery_id,
            "charge_fraction": clip01(self.charge_fraction),
            "reserve_fraction": clip01(self.reserve_fraction),
            "projected_runtime_minutes": float(self.projected_runtime_minutes),
            "charging_state": self.charging_state,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ThermalState:
    """Thermal headroom and throttling posture."""

    thermal_id: str
    thermal_headroom_fraction: float = 0.0
    throttled: bool = False
    max_temperature_c: float = 0.0
    thermal_zone: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "thermal_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thermal_id": self.thermal_id,
            "thermal_headroom_fraction": clip01(self.thermal_headroom_fraction),
            "throttled": bool(self.throttled),
            "max_temperature_c": float(self.max_temperature_c),
            "thermal_zone": self.thermal_zone,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class DeploymentResourceSurface:
    """Typed deployment/resource posture for perception runtime.

    This surface is intentionally lower-WM-owned first. It captures runtime
    feasibility, availability, and headroom before later economic allocation.
    """

    surface_id: str
    compute_envelope: Optional[ComputeEnvelopeState] = None
    inference_capacity: Optional[InferenceCapacityState] = None
    battery_state: Optional[BatteryState] = None
    thermal_state: Optional[ThermalState] = None
    bandwidth_mbps: float = 0.0
    companion_compute_available: bool = False
    deployment_posture: str = "unavailable"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "deployment_resource_surface_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "compute_envelope": (
                None if self.compute_envelope is None else self.compute_envelope.to_dict()
            ),
            "inference_capacity": (
                None if self.inference_capacity is None else self.inference_capacity.to_dict()
            ),
            "battery_state": (
                None if self.battery_state is None else self.battery_state.to_dict()
            ),
            "thermal_state": (
                None if self.thermal_state is None else self.thermal_state.to_dict()
            ),
            "bandwidth_mbps": float(self.bandwidth_mbps),
            "companion_compute_available": bool(self.companion_compute_available),
            "deployment_posture": self.deployment_posture,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Top-level Perception / Grounding WM state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionGroundingWorldState:
    """Top-level canonical state for the Perception / Grounding WM.

    This is the single typed surface that downstream WMs consume:
    - Sim / Synth / Physics WM consumes ``scene_graph`` for branch evaluation
    - Embodiment / Actuation WM consumes ``scene_graph`` for affordance estimation
    - Annotation / Evidence bridge consumes for rollout labeling
    - Economic WM later consumes scene summary for allocation

    Ownership boundaries:
    - Provider outputs are fused through ``evidence_routing`` into ``scene_graph``
    - ``temporal_grounding`` tracks persistence across frames
    - ``scene_graph`` is the canonical truth surface — everything else is provenance

    Maturity ladder position: starts at ``schema_only``, targets ``shadow_runtime``
    before Phase 2 closure.
    """

    state_id: str
    frame_index: int
    episode_id: str
    scene_graph: Optional[SceneGraphState] = None
    temporal_grounding: Optional[TemporalGroundingState] = None
    evidence_routing: Optional[EvidenceRoutingState] = None
    provider_surface: Optional[ProviderSurfaceState] = None
    dataset_surface: Optional[DatasetSurfaceState] = None
    task_measurements: Optional[TaskMeasurementSurface] = None
    deployment_resource_surface: Optional[DeploymentResourceSurface] = None
    semantic_bridge_registry: Optional[SemanticBridgeRegistry] = None
    input_context: Dict[str, Any] = field(default_factory=dict)
    maturity_stage: str = "schema_only"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "perception_grounding_world_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "frame_index": int(self.frame_index),
            "episode_id": self.episode_id,
            "scene_graph": None if self.scene_graph is None else self.scene_graph.to_dict(),
            "temporal_grounding": (
                None if self.temporal_grounding is None else self.temporal_grounding.to_dict()
            ),
            "evidence_routing": (
                None if self.evidence_routing is None else self.evidence_routing.to_dict()
            ),
            "provider_surface": (
                None if self.provider_surface is None else self.provider_surface.to_dict()
            ),
            "dataset_surface": (
                None if self.dataset_surface is None else self.dataset_surface.to_dict()
            ),
            "task_measurements": (
                None if self.task_measurements is None else self.task_measurements.to_dict()
            ),
            "deployment_resource_surface": (
                None
                if self.deployment_resource_surface is None
                else self.deployment_resource_surface.to_dict()
            ),
            "semantic_bridge_registry": (
                None
                if self.semantic_bridge_registry is None
                else self.semantic_bridge_registry.to_dict()
            ),
            "input_context": mapping(self.input_context),
            "maturity_stage": self.maturity_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


__all__ = [
    "BatteryState",
    "ComputeEnvelopeState",
    "DatasetSurfaceState",
    "DeploymentResourceSurface",
    "EvidenceRoutingState",
    "InferenceCapacityState",
    "ObjectTrackState",
    "PerceptionGroundingWorldState",
    "ProviderSurfaceState",
    "SceneEdge",
    "SceneGraphState",
    "TaskMeasurementSurface",
    "ThermalState",
    "TemporalGroundingState",
]
