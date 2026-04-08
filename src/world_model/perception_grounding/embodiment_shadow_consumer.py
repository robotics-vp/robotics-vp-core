"""Embodiment-facing shadow consumer for Perception / Grounding WM outputs.

This module implements the Phase 2 downstream consumer that proves Perception
state is consumable by embodiment-relevant logic.  It is shaped by the
Embodiment / Actuation WM doctrine (``docs/actuation_embodiment_world_model.md``)
but does NOT implement the Embodiment WM itself.

What this is
------------
- A typed, additive, receipt-emitting shadow consumer
- Produces body-relevant perception summaries (action relevance, reachability,
  obstruction, affordance feasibility, contact preconditions, risk hints)
- Non-sovereign: does not replace, override, or rewrite any downstream WM
- Replayable: all inputs and outputs are typed frozen dataclasses
- Shadow mode: advisory only, no control authority

What this is NOT
----------------
- NOT the Embodiment / Actuation WM (Phase 3)
- NOT a planner, controller, or action proposal head
- NOT inverse dynamics, retargeting, or skill generation
- NOT a replacement ontology for body-aware control
- NOT sovereign over any downstream WM state

Subsystem placement
-------------------
This consumer sits at the **affordance / action-relevance bridge surface**
within the Perception / Grounding WM.  It transforms the embodiment bridge
state (already built by the compiler) into a downstream-consumable shadow
surface that later Embodiment WM subsystems (Contact/Affordance Graph Builder,
Capability/Embodiment State Surface) can read during Phase 3 bring-up.

Neuralization successor
-----------------------
Architecture family: cross-attention from embodiment state to object tokens
+ bipartite body-object attention (same as EmbodimentSemanticBridgeState).
Capacity band: 1-3M (gripper) → 5-10M (bimanual humanoid).
Training objective: supervised on affordance classification, grasp/contact
prediction, action success correlation.  NOT direct RL.
Current posture: heuristic shadow (disabled|auto|required applies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import clip01, mapping, stable_id, strings


# ---------------------------------------------------------------------------
# Typed output surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectActionRelevance:
    """Per-object body-relevant action summary derived from Perception state.

    Subsystem: affordance / action-relevance bridge surface.
    Downstream consumers: Embodiment WM Contact/Affordance Graph Builder (Phase 3).
    NOT: action proposal, inverse dynamics, or control policy output.
    """

    track_id: str
    object_label: str
    reachability_score: float = 0.0
    obstruction_score: float = 0.0
    affordance_feasibility: float = 0.0
    contact_precondition_met: float = 0.0
    misalignment_risk: float = 0.0
    perception_confidence: float = 0.0
    epistemic_uncertainty: float = 0.0
    affordance_classes: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_label": self.object_label,
            "reachability_score": clip01(self.reachability_score),
            "obstruction_score": clip01(self.obstruction_score),
            "affordance_feasibility": clip01(self.affordance_feasibility),
            "contact_precondition_met": clip01(self.contact_precondition_met),
            "misalignment_risk": clip01(self.misalignment_risk),
            "perception_confidence": clip01(self.perception_confidence),
            "epistemic_uncertainty": clip01(self.epistemic_uncertainty),
            "affordance_classes": strings(self.affordance_classes),
            "risk_flags": strings(self.risk_flags),
            "metadata": mapping(self.metadata),
        }


@dataclass(frozen=True)
class EmbodimentShadowSurface:
    """Typed perception → embodiment shadow consumption surface.

    This is the principal output of the embodiment-facing shadow consumer.
    It packages Perception state into a form that the future Embodiment WM
    can consume directly, without re-querying the raw scene graph.

    Subsystem: affordance / action-relevance bridge surface.
    Ownership: Perception / Grounding WM compiles this; Embodiment / Actuation
    WM will consume it (shadow mode now, bounded runtime authority later).

    Boundary rules:
    - Output stays at the level of body-relevant perception semantics.
    - Does NOT contain action proposals, control policy, or inverse dynamics.
    - Does NOT become a hidden Embodiment WM.
    - Economic fields stay as compact typed summaries, not allocative policy.
    """

    surface_id: str
    source_state_id: str
    source_episode_id: str
    frame_index: int

    # Per-object action relevance
    object_action_relevances: List[ObjectActionRelevance] = field(
        default_factory=list
    )

    # Scene-level embodiment summaries
    scene_contact_feasibility: float = 0.0
    scene_affordance_coverage: float = 0.0
    scene_obstruction_severity: float = 0.0
    actionable_object_count: int = 0
    obstructed_object_count: int = 0

    # Body-object engagement summary (per body config)
    body_object_engagement_summary: Dict[str, float] = field(default_factory=dict)

    # Resource/deployment readiness for embodiment runtime
    resource_readiness: Dict[str, Any] = field(default_factory=dict)

    # Provider truth posture relevant to embodiment decisions
    provider_truth_for_embodiment: Dict[str, Any] = field(default_factory=dict)

    # Latency/headroom for real-time embodiment consumption
    latency_headroom_for_embodiment: Dict[str, Any] = field(default_factory=dict)

    # Evidence quality summary for embodiment trust
    evidence_quality_for_embodiment: Dict[str, Any] = field(default_factory=dict)

    # Shadow posture
    consumption_mode: str = "shadow_advisory"
    authority_level: str = "none"

    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_shadow_surface_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "source_state_id": self.source_state_id,
            "source_episode_id": self.source_episode_id,
            "frame_index": int(self.frame_index),
            "object_action_relevances": [
                r.to_dict() for r in self.object_action_relevances
            ],
            "scene_contact_feasibility": clip01(self.scene_contact_feasibility),
            "scene_affordance_coverage": clip01(self.scene_affordance_coverage),
            "scene_obstruction_severity": clip01(self.scene_obstruction_severity),
            "actionable_object_count": int(self.actionable_object_count),
            "obstructed_object_count": int(self.obstructed_object_count),
            "body_object_engagement_summary": {
                str(k): float(v)
                for k, v in self.body_object_engagement_summary.items()
            },
            "resource_readiness": mapping(self.resource_readiness),
            "provider_truth_for_embodiment": mapping(self.provider_truth_for_embodiment),
            "latency_headroom_for_embodiment": mapping(
                self.latency_headroom_for_embodiment
            ),
            "evidence_quality_for_embodiment": mapping(
                self.evidence_quality_for_embodiment
            ),
            "consumption_mode": self.consumption_mode,
            "authority_level": self.authority_level,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbodimentShadowConsumptionReceipt:
    """Receipt for one embodiment shadow consumption pass.

    Emitted each time Perception state is consumed by the embodiment-facing
    shadow consumer.  Records input quality, output coverage, and any
    degraded-quality behavior due to provider unavailability.
    """

    receipt_id: str
    source_state_id: str
    source_episode_id: str
    object_count_consumed: int = 0
    actionable_object_count: int = 0
    obstructed_object_count: int = 0
    scene_contact_feasibility: float = 0.0
    scene_affordance_coverage: float = 0.0
    provider_truth_available: bool = False
    deployment_posture: str = "unavailable"
    evidence_fusion_confidence: float = 0.0
    reduced_quality: bool = False
    reduced_quality_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_shadow_consumption_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "source_state_id": self.source_state_id,
            "source_episode_id": self.source_episode_id,
            "object_count_consumed": int(self.object_count_consumed),
            "actionable_object_count": int(self.actionable_object_count),
            "obstructed_object_count": int(self.obstructed_object_count),
            "scene_contact_feasibility": clip01(self.scene_contact_feasibility),
            "scene_affordance_coverage": clip01(self.scene_affordance_coverage),
            "provider_truth_available": bool(self.provider_truth_available),
            "deployment_posture": self.deployment_posture,
            "evidence_fusion_confidence": clip01(self.evidence_fusion_confidence),
            "reduced_quality": bool(self.reduced_quality),
            "reduced_quality_reason": self.reduced_quality_reason,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Consumer implementation
# ---------------------------------------------------------------------------


def _object_action_relevance(
    track: Any,
    bridge_affordance_score: float,
    bridge_affordance_classes: List[str],
    body_pairwise_score: float,
    deployment_posture: str,
) -> ObjectActionRelevance:
    """Derive per-object action relevance from Perception state + bridge.

    Heuristic shadow path.  Neuralization successor: cross-attention from
    body config to object tokens (1-3M params, supervised on grasp success).
    """
    confidence = float(getattr(track, "confidence", 0.0))
    uncertainty = float(getattr(track, "epistemic_uncertainty", 0.5))
    visibility = float(getattr(track, "visibility", 0.5))
    occlusion = float(getattr(track, "occlusion_score", 0.0))
    affordance_hints = list(getattr(track, "affordance_hints", []))
    risk_hints = list(getattr(track, "risk_hints", []))

    # Reachability: combines visibility, bridge pairwise score, confidence
    reachability = clip01(
        0.35 * visibility
        + 0.30 * body_pairwise_score
        + 0.20 * confidence
        + 0.15 * (1.0 - occlusion)
    )

    # Obstruction: high occlusion + low visibility + risk flags
    obstruction = clip01(
        0.45 * occlusion
        + 0.30 * (1.0 - visibility)
        + 0.25 * min(len(risk_hints), 3) / 3.0
    )

    # Affordance feasibility: bridge score modulated by confidence and deployment
    deployment_factor = 0.8 if deployment_posture != "unavailable" else 0.4
    affordance_feasibility = clip01(
        bridge_affordance_score * 0.50
        + confidence * 0.25
        + deployment_factor * 0.25
    )

    # Contact precondition: needs visibility + low uncertainty + some affordance
    contact_met = clip01(
        0.30 * visibility
        + 0.25 * (1.0 - uncertainty)
        + 0.25 * min(len(affordance_hints), 3) / 3.0
        + 0.20 * confidence
    )

    # Misalignment risk: uncertainty + occlusion + risk hints
    misalignment = clip01(
        0.35 * uncertainty
        + 0.30 * occlusion
        + 0.35 * min(len(risk_hints), 3) / 3.0
    )

    # Embodiment-specific risk flags
    risk_flags: list[str] = []
    if occlusion > 0.5:
        risk_flags.append("high_occlusion")
    if uncertainty > 0.5:
        risk_flags.append("high_epistemic_uncertainty")
    if any("fragil" in h for h in risk_hints):
        risk_flags.append("fragility_contact_risk")
    if visibility < 0.4:
        risk_flags.append("low_visibility")
    if deployment_posture == "unavailable":
        risk_flags.append("deployment_unavailable")

    return ObjectActionRelevance(
        track_id=str(getattr(track, "track_id", "")),
        object_label=str(getattr(track, "object_label", "")),
        reachability_score=reachability,
        obstruction_score=obstruction,
        affordance_feasibility=affordance_feasibility,
        contact_precondition_met=contact_met,
        misalignment_risk=misalignment,
        perception_confidence=confidence,
        epistemic_uncertainty=uncertainty,
        affordance_classes=bridge_affordance_classes or affordance_hints,
        risk_flags=risk_flags,
        metadata={
            "bridge_affordance_score": bridge_affordance_score,
            "body_pairwise_score": body_pairwise_score,
            "deployment_posture": deployment_posture,
        },
    )


def consume_perception_for_embodiment(
    state: Any,
) -> tuple[EmbodimentShadowSurface, EmbodimentShadowConsumptionReceipt]:
    """Consume Perception / Grounding WM state for embodiment-relevant shadow output.

    This is the typed entry point for embodiment-facing shadow consumption.
    It reads the canonical Perception state (scene graph, embodiment bridge,
    evidence routing, deployment resource surface) and produces:

    - Per-object action relevance summaries
    - Scene-level contact feasibility and affordance coverage
    - Body-object engagement summaries
    - Resource/deployment readiness for embodiment runtime
    - Provider truth posture relevant to embodiment decisions
    - Evidence quality summary for embodiment trust

    All outputs are shadow/advisory.  No control authority is asserted.

    Args:
        state: ``PerceptionGroundingWorldState`` from the compiler.

    Returns:
        Tuple of (EmbodimentShadowSurface, EmbodimentShadowConsumptionReceipt).
    """
    state_id = str(getattr(state, "state_id", "unknown"))
    episode_id = str(getattr(state, "episode_id", "unknown"))
    frame_index = int(getattr(state, "frame_index", 0))
    surface_id = f"embodiment_shadow_{stable_id('embodiment_shadow', state_id)}"

    # --- Extract perception subsystems ---
    scene_graph = getattr(state, "scene_graph", None)
    bridge_registry = getattr(state, "semantic_bridge_registry", None)
    evidence_routing = getattr(state, "evidence_routing", None)
    deployment_surface = getattr(state, "deployment_resource_surface", None)
    provider_surface = getattr(state, "provider_surface", None)

    # --- Determine reduced-quality posture ---
    reduced_quality = False
    reduced_quality_reason = ""

    if scene_graph is None or not getattr(scene_graph, "object_tracks", []):
        reduced_quality = True
        reduced_quality_reason = "no_scene_graph_or_empty_tracks"

    embod_bridge = (
        getattr(bridge_registry, "embodiment_bridge", None)
        if bridge_registry is not None
        else None
    )
    if embod_bridge is None:
        reduced_quality = True
        reduced_quality_reason = reduced_quality_reason or "no_embodiment_bridge"

    deployment_posture = (
        str(getattr(deployment_surface, "deployment_posture", "unavailable"))
        if deployment_surface is not None
        else "unavailable"
    )

    # --- Build per-object action relevances ---
    object_relevances: list[ObjectActionRelevance] = []
    tracks = list(getattr(scene_graph, "object_tracks", [])) if scene_graph else []
    bridge_affordance_scores = (
        dict(getattr(embod_bridge, "per_object_affordance_scores", {}))
        if embod_bridge
        else {}
    )
    bridge_affordance_classes = (
        dict(getattr(embod_bridge, "per_object_affordance_classes", {}))
        if embod_bridge
        else {}
    )
    body_pairwise = {}
    if embod_bridge is not None:
        bops = dict(getattr(embod_bridge, "body_object_pairwise_scores", {}))
        # Use first body config (typically "g1_default_body")
        if bops:
            first_body = next(iter(bops.values()))
            body_pairwise = dict(first_body) if isinstance(first_body, dict) else {}

    for track in tracks:
        tid = str(getattr(track, "track_id", ""))
        object_relevances.append(
            _object_action_relevance(
                track=track,
                bridge_affordance_score=float(
                    bridge_affordance_scores.get(tid, 0.3)
                ),
                bridge_affordance_classes=list(
                    bridge_affordance_classes.get(tid, [])
                ),
                body_pairwise_score=float(body_pairwise.get(tid, 0.3)),
                deployment_posture=deployment_posture,
            )
        )

    # --- Scene-level summaries ---
    actionable_count = sum(
        1 for r in object_relevances if r.affordance_feasibility > 0.4
    )
    obstructed_count = sum(
        1 for r in object_relevances if r.obstruction_score > 0.5
    )
    scene_contact_feasibility = clip01(
        sum(r.contact_precondition_met for r in object_relevances)
        / max(len(object_relevances), 1)
    )
    scene_affordance_coverage = clip01(
        sum(r.affordance_feasibility for r in object_relevances)
        / max(len(object_relevances), 1)
    )
    scene_obstruction_severity = clip01(
        sum(r.obstruction_score for r in object_relevances)
        / max(len(object_relevances), 1)
    )

    # --- Body-object engagement summary ---
    body_engagement: dict[str, float] = {}
    if embod_bridge is not None:
        bops = dict(getattr(embod_bridge, "body_object_pairwise_scores", {}))
        for body_id, scores in bops.items():
            if isinstance(scores, dict):
                vals = [float(v) for v in scores.values()]
                body_engagement[str(body_id)] = clip01(
                    sum(vals) / max(len(vals), 1)
                )

    # --- Resource readiness ---
    resource_readiness: dict[str, Any] = {
        "deployment_posture": deployment_posture,
        "compute_available": False,
        "companion_available": False,
        "latency_budget_ms": 0.0,
    }
    if deployment_surface is not None:
        compute_env = getattr(deployment_surface, "compute_envelope", None)
        if compute_env is not None:
            resource_readiness["compute_available"] = bool(
                getattr(compute_env, "on_device_available", False)
            )
            resource_readiness["companion_available"] = bool(
                getattr(compute_env, "companion_available", False)
            )
            resource_readiness["latency_budget_ms"] = float(
                getattr(compute_env, "latency_budget_ms", 0.0)
            )
        resource_readiness["bandwidth_mbps"] = float(
            getattr(deployment_surface, "bandwidth_mbps", 0.0)
        )

    # --- Provider truth for embodiment ---
    provider_truth: dict[str, Any] = {"providers_available": {}, "truth_classes": {}}
    if provider_surface is not None:
        provider_truth["providers_available"] = dict(
            getattr(provider_surface, "provider_availability", {})
        )
        provider_truth["truth_classes"] = dict(
            getattr(provider_surface, "provider_truth_class", {})
        )
        provider_truth["provider_count"] = len(
            getattr(provider_surface, "provider_ids", [])
        )
        # Any provider that is stub-only degrades embodiment trust
        stub_providers = [
            pid
            for pid, tc in provider_truth["truth_classes"].items()
            if tc in ("stub_smoke_only", "unavailable")
        ]
        provider_truth["stub_only_providers"] = stub_providers
        provider_truth["all_providers_real"] = len(stub_providers) == 0

    # --- Latency headroom ---
    latency_headroom: dict[str, Any] = {
        "perception_latency_budget_ms": 0.0,
        "headroom_fraction": 0.0,
    }
    if deployment_surface is not None:
        inf_cap = getattr(deployment_surface, "inference_capacity", None)
        if inf_cap is not None:
            latency_headroom["headroom_fraction"] = float(
                getattr(inf_cap, "headroom_fraction", 0.0)
            )
        if provider_surface is not None:
            latency_headroom["perception_latency_budget_ms"] = float(
                getattr(provider_surface, "provider_latency_budget_ms", 0.0)
            )

    # --- Evidence quality for embodiment trust ---
    fusion_confidence = 0.0
    fusion_disagreement = 0.0
    fusion_method = "unknown"
    if evidence_routing is not None:
        fusion_confidence = float(
            getattr(evidence_routing, "fusion_confidence", 0.0)
        )
        fusion_disagreement = float(
            getattr(evidence_routing, "fusion_disagreement", 0.0)
        )
        fusion_method = str(getattr(evidence_routing, "fusion_method", "unknown"))

    evidence_quality: dict[str, Any] = {
        "fusion_confidence": fusion_confidence,
        "fusion_disagreement": fusion_disagreement,
        "fusion_method": fusion_method,
        "embodiment_trust_score": clip01(
            fusion_confidence * 0.6 + (1.0 - fusion_disagreement) * 0.4
        ),
    }

    # --- Build the shadow surface ---
    surface = EmbodimentShadowSurface(
        surface_id=surface_id,
        source_state_id=state_id,
        source_episode_id=episode_id,
        frame_index=frame_index,
        object_action_relevances=object_relevances,
        scene_contact_feasibility=scene_contact_feasibility,
        scene_affordance_coverage=scene_affordance_coverage,
        scene_obstruction_severity=scene_obstruction_severity,
        actionable_object_count=actionable_count,
        obstructed_object_count=obstructed_count,
        body_object_engagement_summary=body_engagement,
        resource_readiness=resource_readiness,
        provider_truth_for_embodiment=provider_truth,
        latency_headroom_for_embodiment=latency_headroom,
        evidence_quality_for_embodiment=evidence_quality,
        consumption_mode="shadow_advisory",
        authority_level="none",
        metadata={
            "consumer_version": "embodiment_shadow_consumer_v1",
            "reduced_quality": reduced_quality,
            "reduced_quality_reason": reduced_quality_reason,
            "bridge_posture": (
                str(getattr(embod_bridge, "helper_posture", "disabled"))
                if embod_bridge
                else "no_bridge"
            ),
            "bridge_promotion_stage": (
                str(
                    getattr(
                        embod_bridge, "helper_promotion_stage", "heuristic_fallback"
                    )
                )
                if embod_bridge
                else "no_bridge"
            ),
        },
    )

    # --- Emit receipt ---
    receipt = EmbodimentShadowConsumptionReceipt(
        receipt_id=f"embodiment_shadow_receipt_{stable_id('embod_receipt', state_id)}",
        source_state_id=state_id,
        source_episode_id=episode_id,
        object_count_consumed=len(object_relevances),
        actionable_object_count=actionable_count,
        obstructed_object_count=obstructed_count,
        scene_contact_feasibility=scene_contact_feasibility,
        scene_affordance_coverage=scene_affordance_coverage,
        provider_truth_available=bool(
            provider_truth.get("all_providers_real", False)
        ),
        deployment_posture=deployment_posture,
        evidence_fusion_confidence=fusion_confidence,
        reduced_quality=reduced_quality,
        reduced_quality_reason=reduced_quality_reason,
        metadata={
            "consumption_mode": "shadow_advisory",
            "body_configs_evaluated": list(body_engagement.keys()),
        },
    )

    return surface, receipt


__all__ = [
    "EmbodimentShadowConsumptionReceipt",
    "EmbodimentShadowSurface",
    "ObjectActionRelevance",
    "consume_perception_for_embodiment",
]
