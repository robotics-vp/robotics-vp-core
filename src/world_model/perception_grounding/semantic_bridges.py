"""Semantic bridge family: WM-native semantic transformations.

The canonical semantic substrate lives in the Perception / Grounding WM
(``SceneGraphState`` — object tokens, edges, scene summary).  Downstream
WMs must not consume this substrate raw.  Instead, each consuming WM has
a **semantic bridge layer** that transforms canonical semantics into the
WM-native semantic form required by that consumer.

This module defines the typed state and contract surfaces for each bridge.
The bridges are the **semantic superposition mechanism**: the same
object-relational-temporal content exists simultaneously in multiple
WM-specific forms, connected by learnable bridges whose quality is
measured by downstream WM performance and (later) transport fidelity.

Bridge family
-------------
1. **Semantic→SimSynthPhysics**: topology-preserving cross-attention +
   spatial graph convolution.  Re-weights edges by physics relevance.
   Governed by Sim/Synth/Physics WM.  2-5M params.

2. **Semantic→Embodiment**: cross-attention from body state to object
   tokens + bipartite body-object attention for affordance scores.
   Governed by Embodiment/Actuation WM.  1-10M params (scales with DoF).

3. **Semantic→Annotation/Evidence**: projection heads + shallow attention
   for object-linked primitive/event labeling, failure interpretation,
   and training dataset formation.  Governed by Perception/Grounding WM.
   500K-2M params.

4. **Semantic→Economic**: perceiver-style cross-attention with learned
   economic query tokens producing fixed-dimensional summaries.
   Governed by Economic WM.  1-3M params.

``SemanticVLA`` successor posture
---------------------------------
``SemanticVLA`` is an insufficient placeholder.  Its successor is NOT one
monolithic model.  It is the distributed semantic bridge family defined
here, built from:

- Perception-WM canonical object/track state (the substrate)
- per-WM semantic bridge layers (the transformations)
- teacher/runtime semantic proposals (provider evidence, not truth)
- affordance / role inference (Embodiment bridge)
- primitive/action segmentation crosswalk (Annotation bridge)
- semantic evidence fusion (Evidence routing, existing in state.py)

Training / shaping story
------------------------
- **Before Phase 6**: bridges are shaped by supervised/predictive losses
  tied to downstream WM task performance (branch eval accuracy, affordance
  prediction, annotation labeling quality, economic value correlation).
- **Phase 6+**: transport fidelity provides explicit gradient signals that
  shape bridge parameters to produce transportable representations.
- **RL placement**: bridges should NOT be trained with direct RL on task
  reward.  They are middleware — supervised/contrastive/predictive training
  is correct for structural fidelity.  Indirect RL shaping comes later
  through Economic WM allocation performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import clip01, mapping, strings


# ---------------------------------------------------------------------------
# Semantic bridge output state types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimSynthSemanticBridgeState:
    """Output of the Semantic→SimSynthPhysics bridge.

    Transforms canonical scene graph into physics-relevant semantic form
    for branch evaluation, sim agenda compilation, diffusion conditioning,
    and object-preservation checking.

    Architecture: topology-preserving cross-attention from sim planning
    context to semantic object tokens, followed by spatial graph convolution
    that re-weights edges by physics relevance (contact > occlusion >
    containment for physics planning).

    Governing WM: Sim/Synth/Physics WM governs output vocabulary, capacity,
    training objectives.  Perception/Grounding WM governs input.

    Capacity: 2-5M params.  Underpowered <500K.  Overbuilt >10M.
    Training: branch evaluation prediction + diffusion conditioning quality.
    No direct RL.  Supervised on branch outcome receipts.
    """

    bridge_id: str
    source_graph_id: str
    physics_object_tokens: List[List[float]] = field(default_factory=list)
    physics_edge_weights: Dict[str, float] = field(default_factory=dict)
    branch_relevance_scores: List[float] = field(default_factory=list)
    object_preservation_scores: List[float] = field(default_factory=list)
    diffusion_conditioning_features: List[float] = field(default_factory=list)
    contact_topology_summary: Dict[str, Any] = field(default_factory=dict)
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "sim_synth_semantic_bridge_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "source_graph_id": self.source_graph_id,
            "physics_object_tokens": [
                [float(v) for v in tok] for tok in self.physics_object_tokens
            ],
            "physics_edge_weights": {
                str(k): float(v) for k, v in self.physics_edge_weights.items()
            },
            "branch_relevance_scores": [float(v) for v in self.branch_relevance_scores],
            "object_preservation_scores": [float(v) for v in self.object_preservation_scores],
            "diffusion_conditioning_features": [
                float(v) for v in self.diffusion_conditioning_features
            ],
            "contact_topology_summary": mapping(self.contact_topology_summary),
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentSemanticBridgeState:
    """Output of the Semantic→Embodiment bridge (affordance projection).

    Transforms canonical scene graph into body-relevant affordance structure
    for action feasibility, grasp planning, manipulation/locomotion relevance.

    Architecture: cross-attention from embodiment state (body config,
    end-effector state, capability profile, resource constraints) to
    semantic object tokens.  Followed by per-object affordance heads and
    bipartite body-object attention for pairwise affordance scores.

    Governing WM: Embodiment/Actuation WM governs output vocabulary,
    capacity, training.  Perception/Grounding WM governs input.

    Capacity: 1-3M (gripper) → 5-10M (bimanual humanoid).
    Scales with embodiment DoF, not scene complexity.
    Training: affordance classification + grasp/contact prediction +
    action success correlation.  Supervised, not RL.
    """

    bridge_id: str
    source_graph_id: str
    per_object_affordance_scores: Dict[str, float] = field(default_factory=dict)
    per_object_affordance_classes: Dict[str, List[str]] = field(default_factory=dict)
    body_object_pairwise_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    action_feasibility_summary: Dict[str, Any] = field(default_factory=dict)
    resource_conditioned: bool = False
    embodiment_dof: int = 0
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_semantic_bridge_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "source_graph_id": self.source_graph_id,
            "per_object_affordance_scores": {
                str(k): float(v) for k, v in self.per_object_affordance_scores.items()
            },
            "per_object_affordance_classes": {
                str(k): strings(v) for k, v in self.per_object_affordance_classes.items()
            },
            "body_object_pairwise_scores": {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in self.body_object_pairwise_scores.items()
            },
            "action_feasibility_summary": mapping(self.action_feasibility_summary),
            "resource_conditioned": bool(self.resource_conditioned),
            "embodiment_dof": int(self.embodiment_dof),
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class AnnotationSemanticBridgeState:
    """Output of the Semantic→Annotation/Evidence bridge.

    Transforms canonical scene graph into structured evidence payloads for
    rollout labeling, semantic evidence, annotation crosswalk, and training
    dataset formation.

    Although architecturally lighter than other bridges, this function is
    **load-bearing**: it is the primary mechanism by which semantic state
    becomes training-usable evidence.

    Architecture: projection heads (MLPs / shallow attention) from semantic
    object tokens → class labels, confidence scores, bounding regions,
    affordance hints, risk hints, primitive-segment alignment scores,
    object-linked event labels, failure/recovery interpretation tags.

    Governing WM: Perception/Grounding WM.
    Capacity: 500K-2M params.
    Training: annotation labeling accuracy + primitive-segment alignment +
    downstream training dataset quality.
    """

    bridge_id: str
    source_graph_id: str
    object_class_labels: Dict[str, str] = field(default_factory=dict)
    object_confidence_scores: Dict[str, float] = field(default_factory=dict)
    object_affordance_hints: Dict[str, List[str]] = field(default_factory=dict)
    object_risk_hints: Dict[str, List[str]] = field(default_factory=dict)
    primitive_segment_alignment_scores: List[float] = field(default_factory=list)
    object_event_labels: Dict[str, List[str]] = field(default_factory=dict)
    failure_interpretation_tags: List[str] = field(default_factory=list)
    recovery_interpretation_tags: List[str] = field(default_factory=list)
    teacher_alignment_score: float = 0.0
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "annotation_semantic_bridge_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "source_graph_id": self.source_graph_id,
            "object_class_labels": {str(k): str(v) for k, v in self.object_class_labels.items()},
            "object_confidence_scores": {
                str(k): clip01(float(v)) for k, v in self.object_confidence_scores.items()
            },
            "object_affordance_hints": {
                str(k): strings(v) for k, v in self.object_affordance_hints.items()
            },
            "object_risk_hints": {
                str(k): strings(v) for k, v in self.object_risk_hints.items()
            },
            "primitive_segment_alignment_scores": [
                float(v) for v in self.primitive_segment_alignment_scores
            ],
            "object_event_labels": {
                str(k): strings(v) for k, v in self.object_event_labels.items()
            },
            "failure_interpretation_tags": strings(self.failure_interpretation_tags),
            "recovery_interpretation_tags": strings(self.recovery_interpretation_tags),
            "teacher_alignment_score": clip01(self.teacher_alignment_score),
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EconomicSemanticBridgeState:
    """Output of the Semantic→Economic bridge (semantic receipt encoder).

    Transforms canonical scene graph into fixed-dimensional summaries for
    Economic WM allocation, pricing, and governance decisions.

    Architecture: perceiver-style cross-attention from learned economic
    query tokens (16-64) to semantic object tokens → fixed-dimensional
    economic-semantic summaries.

    Governing WM: Economic WM governs output vocabulary and training.
    Perception/Grounding WM governs input.
    Capacity: 1-3M params.  Overbuilt >5M.
    Training: economic value prediction + allocation quality.
    Supervised on economic receipts.
    """

    bridge_id: str
    source_graph_id: str
    economic_summary_token: List[float] = field(default_factory=list)
    semantic_density: float = 0.0
    object_diversity: float = 0.0
    affordance_richness: float = 0.0
    grounding_confidence: float = 0.0
    temporal_stability: float = 0.0
    concept_coverage: float = 0.0
    num_query_tokens: int = 0
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "economic_semantic_bridge_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "source_graph_id": self.source_graph_id,
            "economic_summary_token": [float(v) for v in self.economic_summary_token],
            "semantic_density": float(self.semantic_density),
            "object_diversity": float(self.object_diversity),
            "affordance_richness": clip01(self.affordance_richness),
            "grounding_confidence": clip01(self.grounding_confidence),
            "temporal_stability": clip01(self.temporal_stability),
            "concept_coverage": clip01(self.concept_coverage),
            "num_query_tokens": int(self.num_query_tokens),
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Semantic bridge registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticBridgeRegistry:
    """Registry of all semantic bridge states for one planning window.

    This is the typed surface that shows which bridges are active,
    heuristic-only, or promoted-to-learned for the current window.
    It replaces the monolithic SemanticVLA posture with a distributed
    semantic bridge family.
    """

    registry_id: str
    source_graph_id: str = ""
    sim_synth_bridge: Optional[SimSynthSemanticBridgeState] = None
    embodiment_bridge: Optional[EmbodimentSemanticBridgeState] = None
    annotation_bridge: Optional[AnnotationSemanticBridgeState] = None
    economic_bridge: Optional[EconomicSemanticBridgeState] = None
    semantic_vla_successor_status: str = "distributed_bridge_family"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "semantic_bridge_registry_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "source_graph_id": self.source_graph_id,
            "sim_synth_bridge": (
                None if self.sim_synth_bridge is None else self.sim_synth_bridge.to_dict()
            ),
            "embodiment_bridge": (
                None if self.embodiment_bridge is None else self.embodiment_bridge.to_dict()
            ),
            "annotation_bridge": (
                None if self.annotation_bridge is None else self.annotation_bridge.to_dict()
            ),
            "economic_bridge": (
                None if self.economic_bridge is None else self.economic_bridge.to_dict()
            ),
            "semantic_vla_successor_status": self.semantic_vla_successor_status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Semantic bridge receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticBridgeReceipt:
    """Receipt for one semantic bridge invocation.

    Emitted each time a bridge transforms canonical semantics into
    WM-native form.  Records which bridge, quality, downstream
    usefulness signals, and helper posture.
    """

    receipt_id: str
    bridge_kind: str
    source_graph_id: str
    output_quality_score: float = 0.0
    downstream_usefulness_score: float = 0.0
    helper_posture: str = "disabled"
    helper_promotion_stage: str = "heuristic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "semantic_bridge_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "bridge_kind": self.bridge_kind,
            "source_graph_id": self.source_graph_id,
            "output_quality_score": clip01(self.output_quality_score),
            "downstream_usefulness_score": clip01(self.downstream_usefulness_score),
            "helper_posture": self.helper_posture,
            "helper_promotion_stage": self.helper_promotion_stage,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


__all__ = [
    "AnnotationSemanticBridgeState",
    "EconomicSemanticBridgeState",
    "EmbodimentSemanticBridgeState",
    "SemanticBridgeReceipt",
    "SemanticBridgeRegistry",
    "SimSynthSemanticBridgeState",
]
