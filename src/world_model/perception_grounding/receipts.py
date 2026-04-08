"""Receipt contracts for the Perception / Grounding WM.

Every provider invocation, fusion decision, calibration check, and
evidence routing action should emit a typed receipt.  These receipts
feed replay, training, promotion gates, and downstream WM consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .common import clip01, mapping, strings


@dataclass(frozen=True)
class ProviderInvocationReceipt:
    """Receipt for a single perception provider invocation.

    Emitted each time a provider (SAM, DINOv2, depth, V-JEPA, etc.)
    is invoked or skipped.  Records availability, execution status,
    output quality, and latency.
    """

    receipt_id: str
    provider_id: str
    provider_kind: str
    invocation_status: str
    output_quality_score: float = 0.0
    latency_ms: float = 0.0
    output_token_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "provider_invocation_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "invocation_status": self.invocation_status,
            "output_quality_score": clip01(self.output_quality_score),
            "latency_ms": float(self.latency_ms),
            "output_token_count": int(self.output_token_count),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ProviderAvailabilityReceipt:
    """Receipt for provider-surface availability and install posture.

    This is the provider-surface counterpart to invocation receipts: it records
    whether a provider is locally available, install-ready, or blocked before
    any inference attempt is made.
    """

    receipt_id: str
    provider_surface_id: str
    provider_id: str
    availability_status: str
    install_status: str = "unknown"
    provider_truth_class: str = "unavailable"
    sensor_modalities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "provider_availability_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "provider_surface_id": self.provider_surface_id,
            "provider_id": self.provider_id,
            "availability_status": self.availability_status,
            "install_status": self.install_status,
            "provider_truth_class": self.provider_truth_class,
            "sensor_modalities": strings(self.sensor_modalities),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class GroundingCalibrationReceipt:
    """Calibration evidence for perception grounding quality.

    Emitted when the WM evaluates the quality of its grounding state
    against available ground truth, cross-provider agreement, or
    downstream consumer feedback.
    """

    receipt_id: str
    calibration_method: str
    grounding_accuracy: float = 0.0
    spatial_accuracy: float = 0.0
    temporal_consistency: float = 0.0
    provider_agreement: float = 0.0
    cross_provider_disagreement: float = 0.0
    downstream_task_correlation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "grounding_calibration_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "calibration_method": self.calibration_method,
            "grounding_accuracy": clip01(self.grounding_accuracy),
            "spatial_accuracy": clip01(self.spatial_accuracy),
            "temporal_consistency": clip01(self.temporal_consistency),
            "provider_agreement": clip01(self.provider_agreement),
            "cross_provider_disagreement": clip01(self.cross_provider_disagreement),
            "downstream_task_correlation": float(self.downstream_task_correlation),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class InferenceHeadroomReceipt:
    """Receipt for runtime headroom and provider-capacity posture."""

    receipt_id: str
    deployment_surface_id: str
    provider_id: str
    headroom_fraction: float = 0.0
    estimated_latency_ms: float = 0.0
    on_device_available: bool = False
    companion_available: bool = False
    bandwidth_mbps: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "inference_headroom_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "deployment_surface_id": self.deployment_surface_id,
            "provider_id": self.provider_id,
            "headroom_fraction": clip01(self.headroom_fraction),
            "estimated_latency_ms": float(self.estimated_latency_ms),
            "on_device_available": bool(self.on_device_available),
            "companion_available": bool(self.companion_available),
            "bandwidth_mbps": float(self.bandwidth_mbps),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EvidenceFusionReceipt:
    """Receipt for one evidence fusion pass.

    Emitted each time the evidence routing module fuses provider outputs
    into canonical object state.  Records which providers contributed,
    fusion weights, quality, and disagreement.
    """

    receipt_id: str
    fusion_method: str
    provider_ids: List[str] = field(default_factory=list)
    provider_weights: Dict[str, float] = field(default_factory=dict)
    fusion_confidence: float = 0.0
    fusion_disagreement: float = 0.0
    output_object_count: int = 0
    output_edge_count: int = 0
    helper_posture: str = "disabled"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "evidence_fusion_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "fusion_method": self.fusion_method,
            "provider_ids": strings(self.provider_ids),
            "provider_weights": {str(k): float(v) for k, v in self.provider_weights.items()},
            "fusion_confidence": clip01(self.fusion_confidence),
            "fusion_disagreement": clip01(self.fusion_disagreement),
            "output_object_count": int(self.output_object_count),
            "output_edge_count": int(self.output_edge_count),
            "helper_posture": self.helper_posture,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class DeploymentResourceReceipt:
    """Receipt for deployment-resource readiness and explicit blockers."""

    receipt_id: str
    deployment_surface_id: str
    deployment_posture: str
    compute_ready: bool = False
    battery_ready: bool = False
    thermal_ready: bool = False
    bottleneck_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "deployment_resource_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "deployment_surface_id": self.deployment_surface_id,
            "deployment_posture": self.deployment_posture,
            "compute_ready": bool(self.compute_ready),
            "battery_ready": bool(self.battery_ready),
            "thermal_ready": bool(self.thermal_ready),
            "bottleneck_ids": strings(self.bottleneck_ids),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class TemporalGroundingReceipt:
    """Receipt for temporal grounding / scene persistence evaluation.

    Emitted each time the temporal grounding module processes a new frame
    or evaluates temporal persistence quality.
    """

    receipt_id: str
    frame_index: int
    tracks_maintained: int = 0
    tracks_lost: int = 0
    tracks_recovered: int = 0
    id_switches: int = 0
    temporal_coherence_score: float = 0.0
    prediction_accuracy: float = 0.0
    helper_posture: str = "disabled"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "temporal_grounding_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "frame_index": int(self.frame_index),
            "tracks_maintained": int(self.tracks_maintained),
            "tracks_lost": int(self.tracks_lost),
            "tracks_recovered": int(self.tracks_recovered),
            "id_switches": int(self.id_switches),
            "temporal_coherence_score": clip01(self.temporal_coherence_score),
            "prediction_accuracy": clip01(self.prediction_accuracy),
            "helper_posture": self.helper_posture,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class PerceptionContributionReceipt:
    """Perception-side contribution receipt for downstream economic consumption.

    Emitted per episode/window summarizing the perception WM's
    contribution to downstream usefulness.  Economic WM will later
    consume these typed receipts for allocation and valuation.
    """

    receipt_id: str
    episode_id: str
    grounding_quality: float = 0.0
    semantic_yield: float = 0.0
    calibration_confidence: float = 0.0
    action_relevance_prior: float = 0.0
    novelty_score: float = 0.0
    temporal_stability: float = 0.0
    provider_count: int = 0
    object_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "perception_contribution_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "episode_id": self.episode_id,
            "grounding_quality": clip01(self.grounding_quality),
            "semantic_yield": clip01(self.semantic_yield),
            "calibration_confidence": clip01(self.calibration_confidence),
            "action_relevance_prior": clip01(self.action_relevance_prior),
            "novelty_score": clip01(self.novelty_score),
            "temporal_stability": clip01(self.temporal_stability),
            "provider_count": int(self.provider_count),
            "object_count": int(self.object_count),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class GraphTransformerShadowReceipt:
    """Shadow receipt for the SceneGraphTransformerSeam — Phase 2 plasticity gate.

    Framing: Phase 2 shadow-runtime maturation.  The graph transformer seam
    is a real shadow successor to the heuristic scene-graph path — a
    Perception-owned canonical semantic-state refinement, not bridge
    neuralization, not an Embodiment concern, not a transport-layer model.

    The heuristic scene graph remains canonical for now.  The seam earns
    promotion by **benchmark evidence** (annotation-export supervision,
    held-out label agreement, downstream usefulness), not by imitating the
    heuristic.  This implements selective consolidation / promotion gating
    per the plasticity gating doctrine.

    Field groups
    ------------
    - **Intrinsic seam quality**: the seam's own confidence and coherence.
    - **Shadow comparison** (diagnostic only): disagreement monitoring
      between heuristic and learned paths.  These are NOT promotion evidence.
    - **Promotion evidence** (benchmark-gated): annotation-export supervision,
      held-out label agreement, downstream usefulness.  Absent (zero + flag)
      until benchmark data flows.
    - **Promotion gate**: gate_score is meaningful only when
      benchmark_evidence_present is True.  Without benchmark evidence,
      promotion_eligible is always False.

    IMPORTANT — provisional evidence marking:
    If ``evidence_source_provisional`` is True, the benchmark evidence was
    derived from heuristic object tokens rather than provider-backed tokens.
    Provisional evidence can support shadow monitoring, but MUST NOT
    produce promotion_eligible=True.

    Emitted every compilation pass when a graph transformer seam is
    provided, regardless of promotion stage.
    """

    receipt_id: str
    seam_id: str
    promotion_stage: str  # heuristic_fallback | shadow_monitoring | benchmark_gated
    posture: str  # disabled | auto | required

    # --- Intrinsic seam quality ---
    graph_confidence: float = 0.0
    mean_edge_weight: float = 0.0

    # --- Shadow comparison (diagnostic, NOT promotion evidence) ---
    edge_overlap_fraction: float = 0.0
    node_token_cosine_similarity: float = 0.0
    edge_weight_correlation: float = 0.0
    confidence_delta: float = 0.0
    edge_count_heuristic: int = 0
    edge_count_learned: int = 0
    node_count: int = 0

    # --- Promotion evidence (benchmark-gated) ---
    # Populated only when real benchmark data is available.
    benchmark_evidence_present: bool = False
    evidence_source_provisional: bool = False
    annotation_supervision_score: float = 0.0
    held_out_label_agreement: float = 0.0
    downstream_usefulness_score: float = 0.0
    receipt_consistency: float = 0.0

    # --- Runtime ---
    latency_ms: float = 0.0
    param_count: int = 0

    # --- Promotion gate ---
    # gate_score reflects intrinsic quality without benchmark evidence.
    # promotion_eligible is always False without benchmark evidence.
    promotion_eligible: bool = False
    gate_score: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "graph_transformer_shadow_receipt_v3"

    def to_dict(self) -> Dict[str, Any]:
        effective_eligible = (
            self.promotion_eligible
            and self.benchmark_evidence_present
            and not self.evidence_source_provisional
        )
        return {
            "receipt_id": self.receipt_id,
            "seam_id": self.seam_id,
            "promotion_stage": self.promotion_stage,
            "posture": self.posture,
            "graph_confidence": clip01(self.graph_confidence),
            "mean_edge_weight": clip01(self.mean_edge_weight),
            "edge_overlap_fraction": clip01(self.edge_overlap_fraction),
            "node_token_cosine_similarity": float(self.node_token_cosine_similarity),
            "edge_weight_correlation": float(self.edge_weight_correlation),
            "confidence_delta": float(self.confidence_delta),
            "edge_count_heuristic": int(self.edge_count_heuristic),
            "edge_count_learned": int(self.edge_count_learned),
            "node_count": int(self.node_count),
            "benchmark_evidence_present": bool(self.benchmark_evidence_present),
            "evidence_source_provisional": bool(self.evidence_source_provisional),
            "annotation_supervision_score": clip01(self.annotation_supervision_score),
            "held_out_label_agreement": clip01(self.held_out_label_agreement),
            "downstream_usefulness_score": clip01(self.downstream_usefulness_score),
            "receipt_consistency": clip01(self.receipt_consistency),
            "latency_ms": float(self.latency_ms),
            "param_count": int(self.param_count),
            "promotion_eligible": bool(effective_eligible),
            "gate_score": clip01(self.gate_score),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class AnnotationBridgeShadowReceipt:
    """Shadow receipt for the AnnotationBridgeProjectionSeam.

    Same plasticity-gating pattern as GraphTransformerShadowReceipt.
    The annotation bridge projection is a Perception-owned canonical
    projection from object tokens to annotation labels (class, confidence,
    affordance).

    Promotion evidence is **benchmark-gated**: promotion_eligible is always
    False when benchmark_evidence_present is False.

    IMPORTANT — provisional evidence marking:
    If ``evidence_source_provisional`` is True, the benchmark evidence was
    derived from heuristic object tokens rather than real provider-backed
    features.  Provisional evidence is diagnostic only and MUST NOT be
    treated as honest promotion evidence.  When provisional,
    ``promotion_eligible`` is forced False regardless of gate_score.
    """

    receipt_id: str
    seam_id: str
    promotion_stage: str  # heuristic_fallback | shadow_monitoring | benchmark_gated
    posture: str  # disabled | auto | required

    # --- Projection quality ---
    class_accuracy: float = 0.0
    confidence_mae: float = 1.0
    affordance_accuracy: float = 0.0

    # --- Promotion evidence (benchmark-gated) ---
    benchmark_evidence_present: bool = False
    evidence_source_provisional: bool = True  # True until real provider features
    annotation_supervision_score: float = 0.0
    held_out_label_agreement: float = 0.0
    downstream_usefulness_score: float = 0.0
    receipt_consistency: float = 0.0

    # --- Runtime ---
    latency_ms: float = 0.0
    param_count: int = 0

    # --- Promotion gate ---
    promotion_eligible: bool = False
    gate_score: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "annotation_bridge_shadow_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        # Enforce: provisional evidence cannot promote
        effective_eligible = (
            self.promotion_eligible
            and self.benchmark_evidence_present
            and not self.evidence_source_provisional
        )
        return {
            "receipt_id": self.receipt_id,
            "seam_id": self.seam_id,
            "promotion_stage": self.promotion_stage,
            "posture": self.posture,
            "class_accuracy": clip01(self.class_accuracy),
            "confidence_mae": float(self.confidence_mae),
            "affordance_accuracy": clip01(self.affordance_accuracy),
            "benchmark_evidence_present": bool(self.benchmark_evidence_present),
            "evidence_source_provisional": bool(self.evidence_source_provisional),
            "annotation_supervision_score": clip01(self.annotation_supervision_score),
            "held_out_label_agreement": clip01(self.held_out_label_agreement),
            "downstream_usefulness_score": clip01(self.downstream_usefulness_score),
            "receipt_consistency": clip01(self.receipt_consistency),
            "latency_ms": float(self.latency_ms),
            "param_count": int(self.param_count),
            "promotion_eligible": effective_eligible,
            "gate_score": clip01(self.gate_score),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


__all__ = [
    "AnnotationBridgeShadowReceipt",
    "DeploymentResourceReceipt",
    "EvidenceFusionReceipt",
    "GraphTransformerShadowReceipt",
    "GroundingCalibrationReceipt",
    "InferenceHeadroomReceipt",
    "PerceptionContributionReceipt",
    "ProviderAvailabilityReceipt",
    "ProviderInvocationReceipt",
    "TemporalGroundingReceipt",
]
