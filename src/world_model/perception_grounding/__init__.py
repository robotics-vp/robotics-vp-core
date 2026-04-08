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
- **Embodiment-facing shadow consumer** — typed perception→embodiment
  shadow consumption proving Perception outputs matter to embodiment logic

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

Internal subsystem decomposition
---------------------------------
The Perception / Grounding WM is NOT a flat package.  It contains named
internal subsystems with clear ownership boundaries, typed receipts, and
bounded neural successor paths.

1. **Object / Track Persistence** (``state.py``: ``ObjectTrackState``)
   - Owns: per-object persistent track state (identity, pose, features,
     confidence, temporal persistence, affordance/risk hints)
   - Receipts: ``TemporalGroundingReceipt``
   - Neural successor: causal transformer, 3-10M params, predictive loss
     on future state, contrastive on identity
   - Consumers: all downstream WMs via scene graph
   - NOT: an object detector (that is a provider)

2. **Scene Graph / Relation State** (``state.py``: ``SceneGraphState``,
   ``SceneEdge``)
   - Owns: canonical scene graph (objects + typed edges + summary token)
   - Receipts: ``GroundingCalibrationReceipt``
   - Neural successor: Graph Transformer, 5-10M params, supervised on
     relation accuracy + downstream branch evaluation
   - Consumers: all semantic bridges, all downstream WMs
   - NOT: a visual relationship detector (provider evidence, not truth)

3. **Temporal Grounding** (``state.py``: ``TemporalGroundingState``)
   - Owns: frame-to-frame scene persistence and continuity metrics
   - Receipts: ``TemporalGroundingReceipt``
   - Neural successor: causal transformer, 3-10M params
   - Consumers: scene graph, annotation bridge, embodiment shadow consumer
   - NOT: a video predictor (that is V-JEPA's role as a provider)

4. **Evidence Routing / Fusion** (``state.py``: ``EvidenceRoutingState``;
   ``neural_seams.py``: ``EvidenceFusionSeam``)
   - Owns: provider fusion weights, confidence, disagreement, method
   - Receipts: ``EvidenceFusionReceipt``
   - Neural successor: set transformer / perceiver, 5-50M params,
     supervised on provider agreement + downstream task correlation
   - Consumers: scene graph construction, all downstream quality signals
   - NOT: a provider scheduler (that is a deployment concern)

5. **Affordance / Action-Relevance Bridge Surface**
   (``semantic_bridges.py``: ``EmbodimentSemanticBridgeState``;
   ``embodiment_shadow_consumer.py``: ``EmbodimentShadowSurface``)
   - Owns: body-relevant perception semantics (affordance scores,
     reachability, obstruction, contact feasibility, risk hints)
   - Receipts: ``EmbodimentShadowConsumptionReceipt``, ``SemanticBridgeReceipt``
   - Neural successor: cross-attention from body state to object tokens +
     bipartite body-object attention, 1-10M params, supervised on
     affordance/grasp/contact prediction
   - Consumers: Embodiment / Actuation WM (Phase 3)
   - NOT: action proposal, inverse dynamics, or control policy

6. **Provider / Runtime / Deployment-Resource Truth**
   (``state.py``: ``ProviderSurfaceState``, ``DeploymentResourceSurface``,
   ``ComputeEnvelopeState``, ``InferenceCapacityState``, ``BatteryState``,
   ``ThermalState``; ``provider_contracts.py``: all contracts)
   - Owns: provider availability, truth class, deployment readiness,
     compute/battery/thermal posture
   - Receipts: ``ProviderAvailabilityReceipt``, ``InferenceHeadroomReceipt``,
     ``DeploymentResourceReceipt``, ``ProviderInvocationReceipt``
   - Neural successor: none needed (deterministic runtime truth)
   - Consumers: evidence routing, embodiment shadow consumer, economic bridge
   - NOT: an allocator or governor (that is Economic WM's role)

7. **Replay / Export / Bridge Registry Surfaces**
   (``semantic_bridges.py``: ``SemanticBridgeRegistry``,
   ``SimSynthSemanticBridgeState``, ``AnnotationSemanticBridgeState``,
   ``EconomicSemanticBridgeState``)
   - Owns: per-WM semantic bridge outputs, bridge posture, registry state
   - Receipts: ``SemanticBridgeReceipt``, ``PerceptionContributionReceipt``
   - Neural successor: per-bridge learned layers (see bridge docstrings)
   - Consumers: SimSynth WM, Annotation pipeline, Economic WM
   - NOT: the downstream WMs themselves (bridges transform, not replace)

Boundary rules
--------------
- No mother-latent: subsystems maintain typed boundaries, not one blob.
- No provider-owned truth: providers propose; Perception owns compiled truth.
- No bridge that becomes the downstream WM.
- No economic pre-collapse: economic-facing outputs are typed receipts, not
  allocative policy.
- No ungoverned fast→slow leakage: per-step noise emits receipts, does not
  overwrite cross-WM governance state.
"""

from .embodiment_shadow_consumer import (
    EmbodimentShadowConsumptionReceipt,
    EmbodimentShadowSurface,
    ObjectActionRelevance,
    consume_perception_for_embodiment,
)
from .provider_contracts import (
    DepthProviderContract,
    PerceptionProviderContract,
    PerceptionProviderRegistry,
    SAMProviderContract,
    VisionBackboneProviderContract,
    VJEPAProviderContract,
)
from .compiler import (
    PerceptionCompilationResult,
    compile_perception_grounding_with_receipts,
    compile_perception_grounding_world_state,
)
from .annotation_export import (
    AnnotationExportRecord,
    export_annotation_record,
    export_annotation_records_batch,
    load_annotation_export_json,
    save_annotation_export_json,
)
from .benchmark_evidence import (
    PERCEPTION_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
    PerceptionBenchmarkEvidence,
    build_perception_benchmark_evidence,
    load_perception_benchmark_evidence,
    write_perception_benchmark_evidence,
)
from .neural_seams import (
    AnnotationBridgeProjectionSeam,
    EDGE_TYPE_VOCAB,
    DepthMetricCalibrationSeam,
    EvidenceFusionSeam,
    SAMCalibrationSeam,
    SceneGraphTransformerSeam,
    VisionBackboneProjectionSeam,
    VJEPATemporalAlignmentSeam,
    encode_provider_features,
)
from .promotion import (
    resolve_annotation_bridge_helper,
    resolve_evidence_fusion_helper,
    resolve_graph_transformer_helper,
    resolve_provider_adapter_helper,
    resolve_semantic_bridge_helper,
    resolve_temporal_grounding_helper,
)
from .seam_registry import (
    PerceptionSeamRegistry,
    SeamDescriptor,
    create_default_registry,
)
from .receipts import (
    AnnotationBridgeShadowReceipt,
    DeploymentResourceReceipt,
    EvidenceFusionReceipt,
    GraphTransformerShadowReceipt,
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
    # Embodiment shadow consumer
    "EmbodimentShadowConsumptionReceipt",
    "EmbodimentShadowSurface",
    "ObjectActionRelevance",
    "consume_perception_for_embodiment",
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
    # Provider contracts
    "DepthProviderContract",
    "PerceptionProviderContract",
    "PerceptionProviderRegistry",
    "SAMProviderContract",
    "VisionBackboneProviderContract",
    "VJEPAProviderContract",
    # Compiler
    "PerceptionCompilationResult",
    "compile_perception_grounding_with_receipts",
    "compile_perception_grounding_world_state",
    # Annotation export
    "AnnotationExportRecord",
    "export_annotation_record",
    "export_annotation_records_batch",
    "load_annotation_export_json",
    "save_annotation_export_json",
    # Benchmark evidence
    "PERCEPTION_BENCHMARK_EVIDENCE_SCHEMA_VERSION",
    "PerceptionBenchmarkEvidence",
    "build_perception_benchmark_evidence",
    "load_perception_benchmark_evidence",
    "write_perception_benchmark_evidence",
    # Neural seams
    "AnnotationBridgeProjectionSeam",
    "DepthMetricCalibrationSeam",
    "EDGE_TYPE_VOCAB",
    "EvidenceFusionSeam",
    "SAMCalibrationSeam",
    "SceneGraphTransformerSeam",
    "VisionBackboneProjectionSeam",
    "VJEPATemporalAlignmentSeam",
    "encode_provider_features",
    # Seam registry
    "PerceptionSeamRegistry",
    "SeamDescriptor",
    "create_default_registry",
    # Promotion
    "resolve_annotation_bridge_helper",
    "resolve_evidence_fusion_helper",
    "resolve_graph_transformer_helper",
    "resolve_provider_adapter_helper",
    "resolve_semantic_bridge_helper",
    "resolve_temporal_grounding_helper",
]
