"""Typed provider contracts for the Perception / Grounding WM.

Each external perception provider (SAM 3/3.1, DINOv2, SigLIP, V-JEPA 2,
DepthAnythingV2, UniDepth, SceneTracks) has a typed contract that specifies:

- provider identity and capability
- availability and loading posture (disabled | auto | required)
- provider truth class (real | unavailable | stub_smoke_only)
- calibration status
- capacity / latency expectations
- fallback semantics

Providers are NOT native truth owners.  The Perception / Grounding WM
owns canonical downstream state.  Providers contribute evidence through
typed invocation → receipt → fusion → canonical state.

Neuralization placement:
- Each provider contract carries a ``learned_adapter_posture`` indicating
  whether the corresponding learned projection/calibration head is active.
- Adapter parameters are governed by the Perception / Grounding WM.
- Provider weights (SAM, DINOv2, V-JEPA 2, depth) are always frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .common import mapping


# ---------------------------------------------------------------------------
# Base provider contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionProviderContract:
    """Typed contract for a single perception provider.

    This is the base contract type.  Provider-specific contracts
    extend it with additional fields.
    """

    provider_id: str
    provider_kind: str
    provider_family: str
    availability: str
    provider_truth_class: str
    loading_posture: str = "disabled"
    learned_adapter_posture: str = "disabled"
    adapter_promotion_stage: str = "not_loaded"
    model_name: str = ""
    model_version: str = ""
    weights_available: bool = False
    weights_path: str = ""
    gpu_required: bool = True
    expected_latency_ms: float = 0.0
    expected_output_dim: int = 0
    fallback_posture: str = "unavailable"
    fallback_reason: str = ""
    calibration_status: str = "uncalibrated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "perception_provider_contract_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_family": self.provider_family,
            "availability": self.availability,
            "provider_truth_class": self.provider_truth_class,
            "loading_posture": self.loading_posture,
            "learned_adapter_posture": self.learned_adapter_posture,
            "adapter_promotion_stage": self.adapter_promotion_stage,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_available": bool(self.weights_available),
            "weights_path": self.weights_path,
            "gpu_required": bool(self.gpu_required),
            "expected_latency_ms": float(self.expected_latency_ms),
            "expected_output_dim": int(self.expected_output_dim),
            "fallback_posture": self.fallback_posture,
            "fallback_reason": self.fallback_reason,
            "calibration_status": self.calibration_status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# SAM 3 / 3.1 provider contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SAMProviderContract:
    """Typed contract for SAM 3/3.1 concept segmentation and video tracking.

    SAM 3/3.1 provides:
    - image predictor: concept-conditioned segmentation from prompts
    - video predictor: video object tracking with memory/multiplex
    - mask-level object evidence with per-mask confidence

    Learned components (governed by Perception/Grounding WM):
    - calibration head (500K-2M params): calibrated_confidence,
      epistemic_uncertainty, prompt_satisfaction_score
    - mask-to-token projector (500K-1M params): cross-attention
      converting SAM masks + features into d=128 object tokens

    Provider truth: external.  SAM outputs are fused, not canonical.
    """

    provider_id: str = "sam_3_1"
    provider_kind: str = "concept_segmentation"
    provider_family: str = "sam"
    availability: str = "unavailable"
    provider_truth_class: str = "unavailable"
    loading_posture: str = "disabled"
    model_name: str = "sam2.1_hiera_large"
    model_version: str = "3.1"
    weights_available: bool = False
    weights_path: str = ""
    gpu_required: bool = True
    image_predictor_available: bool = False
    video_predictor_available: bool = False
    memory_mode: str = "disabled"
    multiplex_mode: str = "disabled"
    max_objects: int = 64
    expected_latency_ms: float = 50.0
    expected_mask_dim: int = 256
    calibration_head_posture: str = "disabled"
    mask_to_token_projector_posture: str = "disabled"
    fallback_posture: str = "scene_tracks_only"
    fallback_reason: str = "sam_weights_or_gpu_unavailable"
    calibration_status: str = "uncalibrated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "sam_provider_contract_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_family": self.provider_family,
            "availability": self.availability,
            "provider_truth_class": self.provider_truth_class,
            "loading_posture": self.loading_posture,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_available": bool(self.weights_available),
            "weights_path": self.weights_path,
            "gpu_required": bool(self.gpu_required),
            "image_predictor_available": bool(self.image_predictor_available),
            "video_predictor_available": bool(self.video_predictor_available),
            "memory_mode": self.memory_mode,
            "multiplex_mode": self.multiplex_mode,
            "max_objects": int(self.max_objects),
            "expected_latency_ms": float(self.expected_latency_ms),
            "expected_mask_dim": int(self.expected_mask_dim),
            "calibration_head_posture": self.calibration_head_posture,
            "mask_to_token_projector_posture": self.mask_to_token_projector_posture,
            "fallback_posture": self.fallback_posture,
            "fallback_reason": self.fallback_reason,
            "calibration_status": self.calibration_status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Vision backbone provider contract (DINOv2 / SigLIP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionBackboneProviderContract:
    """Typed contract for frozen vision backbone feature extraction.

    DINOv2 or SigLIP as backbone.  Backbone weights are always frozen.
    The only learned component is the projection head (1-5M params):
    2-layer MLP, d_in=backbone_dim, d_hidden=512, d_out=128.

    Provider truth: external.  Features are provider-backed, not native truth.
    """

    provider_id: str = "dinov2_vit_l_14"
    provider_kind: str = "vision_backbone"
    provider_family: str = "dinov2"
    availability: str = "unavailable"
    provider_truth_class: str = "unavailable"
    loading_posture: str = "disabled"
    model_name: str = "dinov2_vitl14"
    model_version: str = "v2"
    weights_available: bool = False
    weights_path: str = ""
    gpu_required: bool = True
    backbone_dim: int = 1024
    projection_output_dim: int = 128
    projection_head_posture: str = "disabled"
    expected_latency_ms: float = 15.0
    fallback_posture: str = "deterministic_stub"
    fallback_reason: str = "backbone_weights_or_gpu_unavailable"
    calibration_status: str = "uncalibrated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "vision_backbone_provider_contract_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_family": self.provider_family,
            "availability": self.availability,
            "provider_truth_class": self.provider_truth_class,
            "loading_posture": self.loading_posture,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_available": bool(self.weights_available),
            "weights_path": self.weights_path,
            "gpu_required": bool(self.gpu_required),
            "backbone_dim": int(self.backbone_dim),
            "projection_output_dim": int(self.projection_output_dim),
            "projection_head_posture": self.projection_head_posture,
            "expected_latency_ms": float(self.expected_latency_ms),
            "fallback_posture": self.fallback_posture,
            "fallback_reason": self.fallback_reason,
            "calibration_status": self.calibration_status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# V-JEPA 2 temporal prediction provider contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VJEPAProviderContract:
    """Typed contract for V-JEPA 2 temporal state prediction.

    Dual-homing: serves both Perception/Grounding WM (temporal state)
    and Sim/Synth/Physics WM (future estimation for branch evaluation).

    Frozen V-JEPA 2 backbone + learned projection (2-5M params):
    cross-attention from V-JEPA latent tokens to WM object tokens.

    Governance: Perception WM owns adapter parameters.
    SimSynth WM provides auxiliary supervised targets.
    """

    provider_id: str = "vjepa2"
    provider_kind: str = "temporal_prediction"
    provider_family: str = "vjepa"
    availability: str = "unavailable"
    provider_truth_class: str = "unavailable"
    loading_posture: str = "disabled"
    model_name: str = "vjepa2_vitl"
    model_version: str = "2.0"
    weights_available: bool = False
    weights_path: str = ""
    gpu_required: bool = True
    upstream_repo: str = "facebookresearch/vjepa2"
    latent_dim: int = 1024
    projection_output_dim: int = 128
    temporal_alignment_head_posture: str = "disabled"
    projection_posture: str = "disabled"
    expected_latency_ms: float = 100.0
    action_conditioned: bool = False
    fallback_posture: str = "planning_only"
    fallback_reason: str = "vjepa2_weights_or_gpu_unavailable"
    calibration_status: str = "uncalibrated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "vjepa_provider_contract_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_family": self.provider_family,
            "availability": self.availability,
            "provider_truth_class": self.provider_truth_class,
            "loading_posture": self.loading_posture,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_available": bool(self.weights_available),
            "weights_path": self.weights_path,
            "gpu_required": bool(self.gpu_required),
            "upstream_repo": self.upstream_repo,
            "latent_dim": int(self.latent_dim),
            "projection_output_dim": int(self.projection_output_dim),
            "temporal_alignment_head_posture": self.temporal_alignment_head_posture,
            "projection_posture": self.projection_posture,
            "expected_latency_ms": float(self.expected_latency_ms),
            "action_conditioned": bool(self.action_conditioned),
            "fallback_posture": self.fallback_posture,
            "fallback_reason": self.fallback_reason,
            "calibration_status": self.calibration_status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Depth provider contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthProviderContract:
    """Typed contract for monocular/stereo depth estimation.

    Frozen depth model (DepthAnythingV2 or UniDepth) + learned metric
    calibration head (500K-1M params): scale_correction, shift_correction.

    Provider truth: external.  Depth is calibrated, not native truth.
    """

    provider_id: str = "depth_anything_v2"
    provider_kind: str = "metric_depth"
    provider_family: str = "depth_anything"
    availability: str = "unavailable"
    provider_truth_class: str = "unavailable"
    loading_posture: str = "disabled"
    model_name: str = "depth_anything_v2_vitl"
    model_version: str = "2.0"
    weights_available: bool = False
    weights_path: str = ""
    gpu_required: bool = True
    metric_calibration_head_posture: str = "disabled"
    expected_latency_ms: float = 20.0
    supports_stereo: bool = False
    camera_intrinsics_required: bool = True
    fallback_posture: str = "scene_tracks_geometry_only"
    fallback_reason: str = "depth_weights_or_gpu_unavailable"
    calibration_status: str = "uncalibrated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "depth_provider_contract_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_family": self.provider_family,
            "availability": self.availability,
            "provider_truth_class": self.provider_truth_class,
            "loading_posture": self.loading_posture,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_available": bool(self.weights_available),
            "weights_path": self.weights_path,
            "gpu_required": bool(self.gpu_required),
            "metric_calibration_head_posture": self.metric_calibration_head_posture,
            "expected_latency_ms": float(self.expected_latency_ms),
            "supports_stereo": bool(self.supports_stereo),
            "camera_intrinsics_required": bool(self.camera_intrinsics_required),
            "fallback_posture": self.fallback_posture,
            "fallback_reason": self.fallback_reason,
            "calibration_status": self.calibration_status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Provider registry state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionProviderRegistry:
    """Registry of all perception provider contracts for one planning window.

    This typed surface lets the WM, downstream consumers, and receipts
    see exactly which providers are available, unavailable, or stub-only.
    """

    registry_id: str
    providers: List[PerceptionProviderContract] = field(default_factory=list)
    sam_contract: SAMProviderContract = field(default_factory=SAMProviderContract)
    vision_backbone_contract: VisionBackboneProviderContract = field(
        default_factory=VisionBackboneProviderContract
    )
    vjepa_contract: VJEPAProviderContract = field(default_factory=VJEPAProviderContract)
    depth_contract: DepthProviderContract = field(default_factory=DepthProviderContract)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "perception_provider_registry_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "providers": [p.to_dict() for p in self.providers],
            "sam_contract": self.sam_contract.to_dict(),
            "vision_backbone_contract": self.vision_backbone_contract.to_dict(),
            "vjepa_contract": self.vjepa_contract.to_dict(),
            "depth_contract": self.depth_contract.to_dict(),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


__all__ = [
    "DepthProviderContract",
    "PerceptionProviderContract",
    "PerceptionProviderRegistry",
    "SAMProviderContract",
    "VJEPAProviderContract",
    "VisionBackboneProviderContract",
]
