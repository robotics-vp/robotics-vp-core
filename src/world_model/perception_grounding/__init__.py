"""Perception / Grounding World Model — Phase 2.

Canonical owner of:
- Scene / grounding state (object tracks, spatial relations, scene graph)
- Temporal grounding / continuity (object persistence across frames)
- Provider truth (SAM 3/3.1, DINOv2/SigLIP, V-JEPA 2, Depth)
- Provider/dataset/task/deployment-resource surfaces for honest runtime truth
- Evidence routing / calibration
- Perception-side contribution signals for downstream WMs
- **Semantic bridge family** — per-WM semantic transformations that
  replace the monolithic SemanticVLA posture

Semantic architecture
---------------------
The canonical semantic substrate (``SceneGraphState``) is WM-owned.
Downstream WMs consume semantics through **bridge layers** that transform
the substrate into WM-native form:

- ``SimSynthSemanticBridgeState``: physics-relevant semantics for branch
  evaluation, diffusion conditioning, object preservation
- ``EmbodimentSemanticBridgeState``: affordance/action-relevance semantics
  for bodily feasibility and grasp planning
- ``AnnotationSemanticBridgeState``: object-linked primitive/event labeling,
  failure interpretation, training dataset formation
- ``EconomicSemanticBridgeState``: fixed-dim summaries for allocation and
  governance

This distributed bridge family is the explicit successor to ``SemanticVLA``.
"""

from .provider_contracts import (
    DepthProviderContract,
    PerceptionProviderContract,
    PerceptionProviderRegistry,
    SAMProviderContract,
    VisionBackboneProviderContract,
    VJEPAProviderContract,
)
from .promotion import (
    resolve_evidence_fusion_helper,
    resolve_graph_transformer_helper,
    resolve_semantic_bridge_helper,
    resolve_temporal_grounding_helper,
)
from .receipts import (
    DeploymentResourceReceipt,
    EvidenceFusionReceipt,
    GroundingCalibrationReceipt,
    InferenceHeadroomReceipt,
    PerceptionContributionReceipt,
    ProviderAvailabilityReceipt,
    ProviderInvocationReceipt,
    TemporalGroundingReceipt,
)
from .semantic_bridges import (
    AnnotationSemanticBridgeState,
    EconomicSemanticBridgeState,
    EmbodimentSemanticBridgeState,
    SemanticBridgeReceipt,
    SemanticBridgeRegistry,
    SimSynthSemanticBridgeState,
)
from .state import (
    BatteryState,
    ComputeEnvelopeState,
    DatasetSurfaceState,
    DeploymentResourceSurface,
    EvidenceRoutingState,
    InferenceCapacityState,
    ObjectTrackState,
    PerceptionGroundingWorldState,
    ProviderSurfaceState,
    SceneEdge,
    SceneGraphState,
    TaskMeasurementSurface,
    ThermalState,
    TemporalGroundingState,
)

__all__ = [
    # State
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
    # Semantic bridges
    "AnnotationSemanticBridgeState",
    "EconomicSemanticBridgeState",
    "EmbodimentSemanticBridgeState",
    "SemanticBridgeReceipt",
    "SemanticBridgeRegistry",
    "SimSynthSemanticBridgeState",
    # Receipts
    "DeploymentResourceReceipt",
    "EvidenceFusionReceipt",
    "GroundingCalibrationReceipt",
    "InferenceHeadroomReceipt",
    "PerceptionContributionReceipt",
    "ProviderAvailabilityReceipt",
    "ProviderInvocationReceipt",
    "TemporalGroundingReceipt",
    # Provider contracts
    "DepthProviderContract",
    "PerceptionProviderContract",
    "PerceptionProviderRegistry",
    "SAMProviderContract",
    "VisionBackboneProviderContract",
    "VJEPAProviderContract",
    # Promotion
    "resolve_evidence_fusion_helper",
    "resolve_graph_transformer_helper",
    "resolve_semantic_bridge_helper",
    "resolve_temporal_grounding_helper",
]
