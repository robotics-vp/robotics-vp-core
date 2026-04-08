"""Promotion / demotion machinery for Perception / Grounding WM helpers.

Same pattern as ``sim_synth_physics/promotion.py``: shared demotion logic
plus per-subsystem resolvers with ``disabled|auto|required`` posture.

Subsystems with promotion/demotion:
- Graph Transformer (canonical scene graph)
- Temporal grounding module (causal transformer)
- Evidence fusion module (set transformer / perceiver)
- Per-provider learned adapters (SAM calibration, DINOv2 projection, etc.)
- Semantic bridge layers (per-WM semantic transformations)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _check_demotion(
    benchmark_gate: Dict[str, Any],
    evidence_signals: Dict[str, Any],
) -> tuple[bool, str]:
    """Shared demotion check for all perception helper resolvers.

    Demotion triggers:
    - ``benchmark_gate_revoked``: explicit gate revocation
    - ``evidence_failure``: evidence quality below acceptable threshold
    - ``recent_failure_rate`` exceeding ``demotion_failure_threshold``
    """
    if not evidence_signals:
        return False, ""
    if bool(evidence_signals.get("benchmark_gate_revoked", False)):
        return True, "benchmark_gate_revoked"
    if bool(evidence_signals.get("evidence_failure", False)):
        return True, "evidence_failure"
    failure_rate = float(evidence_signals.get("recent_failure_rate", 0.0) or 0.0)
    failure_threshold = float(
        benchmark_gate.get("demotion_failure_threshold", 0.5) or 0.5
    )
    if failure_rate > failure_threshold:
        return True, f"failure_rate_{failure_rate:.2f}_exceeds_{failure_threshold:.2f}"
    return False, ""


def resolve_graph_transformer_helper(
    *,
    loading_posture: str,
    benchmark_signals: Mapping[str, Any],
    evidence_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve Graph Transformer helper posture.

    The Graph Transformer is the core learned module that produces
    the canonical scene graph from object tokens and edges.

    - ``disabled``: heuristic scene graph (existing SemanticWorldModel path)
    - ``auto``: learned graph transformer if benchmark-ready, else heuristic
    - ``required``: must use learned graph transformer or fail
    """
    posture = str(loading_posture or "disabled")
    benchmark_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    benchmark_gate = dict(benchmark_signals.get("benchmark_gate", {}) or {})

    if posture == "disabled":
        return {
            "helper_active": False,
            "promotion_stage": "heuristic_fallback",
            "helper_weight": 0.0,
            "posture": "disabled",
        }

    if posture == "required":
        if benchmark_ready:
            should_demote, reason = _check_demotion(
                benchmark_gate, dict(evidence_signals or {})
            )
            if should_demote:
                return {
                    "helper_active": True,
                    "promotion_stage": "demoted_to_shadow",
                    "helper_weight": 0.25,
                    "posture": "required",
                    "demotion_reason": reason,
                }
            return {
                "helper_active": True,
                "promotion_stage": "promoted",
                "helper_weight": 1.0,
                "posture": "required",
            }
        return {
            "helper_active": False,
            "promotion_stage": "required_but_not_ready",
            "helper_weight": 0.0,
            "posture": "required",
        }

    # auto
    if benchmark_ready:
        should_demote, reason = _check_demotion(
            benchmark_gate, dict(evidence_signals or {})
        )
        if should_demote:
            return {
                "helper_active": True,
                "promotion_stage": "demoted_to_shadow",
                "helper_weight": 0.25,
                "posture": "auto",
                "demotion_reason": reason,
            }
        return {
            "helper_active": True,
            "promotion_stage": "promoted",
            "helper_weight": 1.0,
            "posture": "auto",
        }
    return {
        "helper_active": False,
        "promotion_stage": "heuristic_fallback",
        "helper_weight": 0.0,
        "posture": "auto",
    }


def resolve_temporal_grounding_helper(
    *,
    loading_posture: str,
    benchmark_signals: Mapping[str, Any],
    evidence_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve temporal grounding helper posture.

    The temporal grounding module is the causal transformer that
    maintains object persistence across frames.

    - ``disabled``: SceneTracks Kalman tracking (existing heuristic)
    - ``auto``: learned temporal module if benchmark-ready
    - ``required``: must use learned temporal module or fail
    """
    posture = str(loading_posture or "disabled")
    benchmark_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    benchmark_gate = dict(benchmark_signals.get("benchmark_gate", {}) or {})

    if posture == "disabled":
        return {
            "helper_active": False,
            "promotion_stage": "heuristic_fallback",
            "helper_weight": 0.0,
            "posture": "disabled",
        }

    if posture == "required":
        if benchmark_ready:
            should_demote, reason = _check_demotion(
                benchmark_gate, dict(evidence_signals or {})
            )
            if should_demote:
                return {
                    "helper_active": True,
                    "promotion_stage": "demoted_to_shadow",
                    "helper_weight": 0.25,
                    "posture": "required",
                    "demotion_reason": reason,
                }
            return {
                "helper_active": True,
                "promotion_stage": "promoted",
                "helper_weight": 1.0,
                "posture": "required",
            }
        return {
            "helper_active": False,
            "promotion_stage": "required_but_not_ready",
            "helper_weight": 0.0,
            "posture": "required",
        }

    # auto
    if benchmark_ready:
        should_demote, reason = _check_demotion(
            benchmark_gate, dict(evidence_signals or {})
        )
        if should_demote:
            return {
                "helper_active": True,
                "promotion_stage": "demoted_to_shadow",
                "helper_weight": 0.25,
                "posture": "auto",
                "demotion_reason": reason,
            }
        return {
            "helper_active": True,
            "promotion_stage": "promoted",
            "helper_weight": 1.0,
            "posture": "auto",
        }
    return {
        "helper_active": False,
        "promotion_stage": "heuristic_fallback",
        "helper_weight": 0.0,
        "posture": "auto",
    }


def resolve_evidence_fusion_helper(
    *,
    loading_posture: str,
    benchmark_signals: Mapping[str, Any],
    evidence_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve evidence fusion helper posture.

    The evidence fusion module is the set transformer / perceiver
    that fuses heterogeneous provider evidence into canonical object state.

    - ``disabled``: heuristic weighted fusion (existing semantic_fusion.py MVP)
    - ``auto``: learned fusion if benchmark-ready
    - ``required``: must use learned fusion or fail
    """
    posture = str(loading_posture or "disabled")
    benchmark_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    benchmark_gate = dict(benchmark_signals.get("benchmark_gate", {}) or {})

    if posture == "disabled":
        return {
            "helper_active": False,
            "promotion_stage": "heuristic_fallback",
            "helper_weight": 0.0,
            "posture": "disabled",
        }

    if posture == "required":
        if benchmark_ready:
            should_demote, reason = _check_demotion(
                benchmark_gate, dict(evidence_signals or {})
            )
            if should_demote:
                return {
                    "helper_active": True,
                    "promotion_stage": "demoted_to_shadow",
                    "helper_weight": 0.25,
                    "posture": "required",
                    "demotion_reason": reason,
                }
            return {
                "helper_active": True,
                "promotion_stage": "promoted",
                "helper_weight": 1.0,
                "posture": "required",
            }
        return {
            "helper_active": False,
            "promotion_stage": "required_but_not_ready",
            "helper_weight": 0.0,
            "posture": "required",
        }

    # auto
    if benchmark_ready:
        should_demote, reason = _check_demotion(
            benchmark_gate, dict(evidence_signals or {})
        )
        if should_demote:
            return {
                "helper_active": True,
                "promotion_stage": "demoted_to_shadow",
                "helper_weight": 0.25,
                "posture": "auto",
                "demotion_reason": reason,
            }
        return {
            "helper_active": True,
            "promotion_stage": "promoted",
            "helper_weight": 1.0,
            "posture": "auto",
        }
    return {
        "helper_active": False,
        "promotion_stage": "heuristic_fallback",
        "helper_weight": 0.0,
        "posture": "auto",
    }


def resolve_semantic_bridge_helper(
    *,
    bridge_kind: str,
    loading_posture: str,
    benchmark_signals: Mapping[str, Any],
    evidence_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a semantic bridge layer's promotion posture.

    Each semantic bridge transforms canonical Perception/Grounding WM
    semantics into the native form required by a consuming WM.  Bridges
    are trained with supervised/predictive losses, not direct RL.

    ``bridge_kind`` identifies which bridge:
    - ``sim_synth``: Semantic→SimSynthPhysics (topology cross-attention)
    - ``embodiment``: Semantic→Embodiment (affordance bipartite attention)
    - ``annotation``: Semantic→Annotation (projection heads)
    - ``economic``: Semantic→Economic (perceiver query tokens)

    Promotion posture:
    - ``disabled``: heuristic bridge (static projection or passthrough)
    - ``auto``: learned bridge if benchmark-ready, else heuristic
    - ``required``: must use learned bridge or fail
    """
    posture = str(loading_posture or "disabled")
    benchmark_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    benchmark_gate = dict(benchmark_signals.get("benchmark_gate", {}) or {})

    base = {"bridge_kind": str(bridge_kind)}

    if posture == "disabled":
        return {
            **base,
            "helper_active": False,
            "promotion_stage": "heuristic_fallback",
            "helper_weight": 0.0,
            "posture": "disabled",
        }

    if posture == "required":
        if benchmark_ready:
            should_demote, reason = _check_demotion(
                benchmark_gate, dict(evidence_signals or {})
            )
            if should_demote:
                return {
                    **base,
                    "helper_active": True,
                    "promotion_stage": "demoted_to_shadow",
                    "helper_weight": 0.25,
                    "posture": "required",
                    "demotion_reason": reason,
                }
            return {
                **base,
                "helper_active": True,
                "promotion_stage": "promoted",
                "helper_weight": 1.0,
                "posture": "required",
            }
        return {
            **base,
            "helper_active": False,
            "promotion_stage": "required_but_not_ready",
            "helper_weight": 0.0,
            "posture": "required",
        }

    # auto
    if benchmark_ready:
        should_demote, reason = _check_demotion(
            benchmark_gate, dict(evidence_signals or {})
        )
        if should_demote:
            return {
                **base,
                "helper_active": True,
                "promotion_stage": "demoted_to_shadow",
                "helper_weight": 0.25,
                "posture": "auto",
                "demotion_reason": reason,
            }
        return {
            **base,
            "helper_active": True,
            "promotion_stage": "promoted",
            "helper_weight": 1.0,
            "posture": "auto",
        }
    return {
        **base,
        "helper_active": False,
        "promotion_stage": "heuristic_fallback",
        "helper_weight": 0.0,
        "posture": "auto",
    }


def resolve_provider_adapter_helper(
    *,
    provider_kind: str,
    loading_posture: str,
    benchmark_signals: Mapping[str, Any],
    evidence_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a provider adapter neural seam's promotion posture.

    Provider adapters are learned projection/calibration heads that sit
    between frozen external providers (SAM, DINOv2, V-JEPA 2, Depth) and
    the canonical WM state.  They are governed by Perception/Grounding WM.

    ``provider_kind`` identifies which adapter:
    - ``sam_calibration``: SAM mask confidence calibration head
    - ``vision_backbone_projection``: DINOv2/SigLIP feature projection
    - ``depth_metric_calibration``: Depth scale/shift calibration
    - ``vjepa_temporal_alignment``: V-JEPA temporal cross-attention

    Promotion posture:
    - ``disabled``: raw provider output (no calibration/projection)
    - ``auto``: learned adapter if benchmark-ready, else raw
    - ``required``: must use learned adapter or fail
    """
    posture = str(loading_posture or "disabled")
    benchmark_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    benchmark_gate = dict(benchmark_signals.get("benchmark_gate", {}) or {})

    base = {"provider_kind": str(provider_kind)}

    if posture == "disabled":
        return {
            **base,
            "helper_active": False,
            "promotion_stage": "raw_provider_output",
            "helper_weight": 0.0,
            "posture": "disabled",
        }

    if posture == "required":
        if benchmark_ready:
            should_demote, reason = _check_demotion(
                benchmark_gate, dict(evidence_signals or {})
            )
            if should_demote:
                return {
                    **base,
                    "helper_active": True,
                    "promotion_stage": "demoted_to_shadow",
                    "helper_weight": 0.25,
                    "posture": "required",
                    "demotion_reason": reason,
                }
            return {
                **base,
                "helper_active": True,
                "promotion_stage": "promoted",
                "helper_weight": 1.0,
                "posture": "required",
            }
        return {
            **base,
            "helper_active": False,
            "promotion_stage": "required_but_not_ready",
            "helper_weight": 0.0,
            "posture": "required",
        }

    # auto
    if benchmark_ready:
        should_demote, reason = _check_demotion(
            benchmark_gate, dict(evidence_signals or {})
        )
        if should_demote:
            return {
                **base,
                "helper_active": True,
                "promotion_stage": "demoted_to_shadow",
                "helper_weight": 0.25,
                "posture": "auto",
                "demotion_reason": reason,
            }
        return {
            **base,
            "helper_active": True,
            "promotion_stage": "promoted",
            "helper_weight": 1.0,
            "posture": "auto",
        }
    return {
        **base,
        "helper_active": False,
        "promotion_stage": "raw_provider_output",
        "helper_weight": 0.0,
        "posture": "auto",
    }


def resolve_annotation_bridge_helper(
    *,
    loading_posture: str,
    benchmark_signals: Mapping[str, Any],
    evidence_signals: Optional[Mapping[str, Any]] = None,
    evidence_source_provisional: bool = True,
) -> Dict[str, Any]:
    """Resolve Annotation Bridge Projection helper posture.

    The annotation bridge projection is the Perception-owned projection
    from object tokens to annotation labels.  Same disabled|auto|required
    posture pattern as other resolvers, with one additional constraint:

    If ``evidence_source_provisional`` is True (default), benchmark
    evidence was derived from heuristic object tokens, not real
    provider-backed features.  Provisional evidence blocks promotion
    regardless of benchmark gate score — the seam runs in shadow mode
    only, emitting diagnostic receipts.
    """
    posture = str(loading_posture or "disabled")
    benchmark_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    benchmark_gate = dict(benchmark_signals.get("benchmark_gate", {}) or {})

    if posture == "disabled":
        return {
            "helper_active": False,
            "promotion_stage": "heuristic_fallback",
            "helper_weight": 0.0,
            "posture": "disabled",
            "evidence_source_provisional": evidence_source_provisional,
        }

    # Provisional evidence blocks promotion even if benchmark gate passes
    effective_benchmark_ready = benchmark_ready and not evidence_source_provisional

    if posture == "required":
        if effective_benchmark_ready:
            should_demote, reason = _check_demotion(
                benchmark_gate, dict(evidence_signals or {})
            )
            if should_demote:
                return {
                    "helper_active": True,
                    "promotion_stage": "demoted_to_shadow",
                    "helper_weight": 0.25,
                    "posture": "required",
                    "demotion_reason": reason,
                    "evidence_source_provisional": evidence_source_provisional,
                }
            return {
                "helper_active": True,
                "promotion_stage": "promoted",
                "helper_weight": 1.0,
                "posture": "required",
                "evidence_source_provisional": evidence_source_provisional,
            }
        stage = "shadow_monitoring" if benchmark_ready else "required_but_not_ready"
        return {
            "helper_active": benchmark_ready,
            "promotion_stage": stage,
            "helper_weight": 0.0,
            "posture": "required",
            "evidence_source_provisional": evidence_source_provisional,
        }

    # auto
    if effective_benchmark_ready:
        should_demote, reason = _check_demotion(
            benchmark_gate, dict(evidence_signals or {})
        )
        if should_demote:
            return {
                "helper_active": True,
                "promotion_stage": "demoted_to_shadow",
                "helper_weight": 0.25,
                "posture": "auto",
                "demotion_reason": reason,
                "evidence_source_provisional": evidence_source_provisional,
            }
        return {
            "helper_active": True,
            "promotion_stage": "promoted",
            "helper_weight": 1.0,
            "posture": "auto",
            "evidence_source_provisional": evidence_source_provisional,
        }
    # shadow_monitoring if benchmark data exists but is provisional
    stage = "shadow_monitoring" if benchmark_ready else "heuristic_fallback"
    return {
        "helper_active": benchmark_ready,
        "promotion_stage": stage,
        "helper_weight": 0.0,
        "posture": "auto",
        "evidence_source_provisional": evidence_source_provisional,
    }


__all__ = [
    "resolve_annotation_bridge_helper",
    "resolve_evidence_fusion_helper",
    "resolve_graph_transformer_helper",
    "resolve_provider_adapter_helper",
    "resolve_semantic_bridge_helper",
    "resolve_temporal_grounding_helper",
]
