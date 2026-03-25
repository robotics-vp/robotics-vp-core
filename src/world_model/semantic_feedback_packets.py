"""Packetized semantic feedback surfaces for the runtime coverage loop.

This module closes the additive cybernetic gap without handing authority
directly to ad hoc heuristics. It compiles runtime outcome/validation
signals into typed packets that downstream meta-nodes and transformer
shells can route as bounded work orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def parse_edge_key(edge_key: str) -> Tuple[str, str]:
    parts = str(edge_key or "").split(" -> ", 1)
    if len(parts) != 2:
        return ("", "")
    return (parts[0].strip(), parts[1].strip())


@dataclass(frozen=True)
class CoverageOutcomePacket:
    """Observed downstream effect of a coverage fill attempt."""

    edge_key: str
    fill_method: str
    coverage_delta: float = 0.0
    process_reward_delta: float = 0.0
    policy_eval_delta: float = 0.0
    quality_score: float = 0.0
    cost_score: float = 0.0
    backend_health_score: float = 1.0
    governance_status: str = "approved"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def edge(self) -> Tuple[str, str]:
        return parse_edge_key(self.edge_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_key": self.edge_key,
            "fill_method": self.fill_method,
            "coverage_delta": float(self.coverage_delta),
            "process_reward_delta": float(self.process_reward_delta),
            "policy_eval_delta": float(self.policy_eval_delta),
            "quality_score": float(self.quality_score),
            "cost_score": float(self.cost_score),
            "backend_health_score": float(self.backend_health_score),
            "governance_status": self.governance_status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoverageOutcomePacket":
        return cls(
            edge_key=str(payload.get("edge_key", "")),
            fill_method=str(payload.get("fill_method", "")),
            coverage_delta=_safe_float(payload.get("coverage_delta", 0.0)),
            process_reward_delta=_safe_float(payload.get("process_reward_delta", 0.0)),
            policy_eval_delta=_safe_float(payload.get("policy_eval_delta", 0.0)),
            quality_score=_safe_float(payload.get("quality_score", 0.0)),
            cost_score=_safe_float(payload.get("cost_score", 0.0)),
            backend_health_score=_safe_float(payload.get("backend_health_score", 1.0)),
            governance_status=str(payload.get("governance_status", "approved")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class WMValidationPacket:
    """Observed mismatch between semantic-WM state and runtime outcomes."""

    target_ref: str
    validation_kind: str
    predicted_value: str = ""
    observed_value: str = ""
    error_score: float = 0.0
    severity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_ref": self.target_ref,
            "validation_kind": self.validation_kind,
            "predicted_value": self.predicted_value,
            "observed_value": self.observed_value,
            "error_score": float(self.error_score),
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMValidationPacket":
        return cls(
            target_ref=str(payload.get("target_ref", "")),
            validation_kind=str(payload.get("validation_kind", "")),
            predicted_value=str(payload.get("predicted_value", "")),
            observed_value=str(payload.get("observed_value", "")),
            error_score=_safe_float(payload.get("error_score", 0.0)),
            severity=str(payload.get("severity", "medium")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class GraphMutationProposal:
    """Bounded graph mutation request for the skill/primitive coverage graph."""

    proposal_id: str
    action: str
    target_ref: str
    confidence: float
    rationale: str
    source_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "target_ref": self.target_ref,
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "source_refs": list(self.source_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SemanticCoverageFeedback:
    """Compiled return-path summary for coverage and semantic routing."""

    feedback_summary: Dict[str, Any] = field(default_factory=dict)
    edge_metadata: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    edge_economic_overlay: Dict[Tuple[str, str], float] = field(default_factory=dict)
    edge_trust_overlay: Dict[Tuple[str, str], float] = field(default_factory=dict)
    edge_readiness_overlay: Dict[Tuple[str, str], float] = field(default_factory=dict)
    trust_calibration_overlay: Dict[str, float] = field(default_factory=dict)
    econ_calibration_overlay: Dict[str, float] = field(default_factory=dict)
    wm_validation_summary: Dict[str, Any] = field(default_factory=dict)
    graph_mutation_proposals: List[GraphMutationProposal] = field(default_factory=list)


def _coerce_coverage_outcomes(
    values: Optional[Iterable[Any]],
) -> List[CoverageOutcomePacket]:
    packets: List[CoverageOutcomePacket] = []
    for item in values or []:
        if isinstance(item, CoverageOutcomePacket):
            packets.append(item)
        elif isinstance(item, Mapping):
            packets.append(CoverageOutcomePacket.from_dict(item))
    return packets


def _coerce_wm_validations(
    values: Optional[Iterable[Any]],
) -> List[WMValidationPacket]:
    packets: List[WMValidationPacket] = []
    for item in values or []:
        if isinstance(item, WMValidationPacket):
            packets.append(item)
        elif isinstance(item, Mapping):
            packets.append(WMValidationPacket.from_dict(item))
    return packets


def _average(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def compile_semantic_coverage_feedback(
    *,
    coverage_outcomes: Optional[Iterable[Any]] = None,
    wm_validation_packets: Optional[Iterable[Any]] = None,
    fill_outcome_records: Optional[Sequence[Any]] = None,
    process_reward_summaries: Optional[Sequence[Mapping[str, Any]]] = None,
    governance_traces: Optional[Sequence[Mapping[str, Any]]] = None,
    stage2_ontology_proposals: Optional[Sequence[Any]] = None,
    econ_signals: Optional[Mapping[str, Any]] = None,
    trust_state: Optional[Mapping[str, Any]] = None,
    backend_health_reports: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SemanticCoverageFeedback:
    """Compile cybernetic return-path signals into typed overlay summaries."""
    outcome_packets = _coerce_coverage_outcomes(coverage_outcomes)
    validation_packets = _coerce_wm_validations(wm_validation_packets)
    fill_records = list(fill_outcome_records or [])
    process_rewards = [dict(item or {}) for item in (process_reward_summaries or [])]
    governance = [dict(item or {}) for item in (governance_traces or [])]
    backend_health = [dict(item or {}) for item in (backend_health_reports or [])]

    blocked_edges: set[Tuple[str, str]] = set()
    blocked_edge_keys: set[str] = set()
    for trace in governance:
        outcome = str(trace.get("outcome", trace.get("status", ""))).lower()
        edge_key = str(trace.get("edge_key", "")).strip()
        if edge_key:
            edge = parse_edge_key(edge_key)
            if edge != ("", ""):
                if outcome in {"veto", "blocked", "deny"}:
                    blocked_edges.add(edge)
                    blocked_edge_keys.add(edge_key)
        for raw_edge in trace.get("blocked_edges", []) or []:
            edge = parse_edge_key(str(raw_edge))
            if edge != ("", ""):
                blocked_edges.add(edge)
                blocked_edge_keys.add(str(raw_edge))

    edge_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
    edge_econ_overlay: Dict[Tuple[str, str], float] = {}
    edge_trust_overlay: Dict[Tuple[str, str], float] = {}
    edge_readiness_overlay: Dict[Tuple[str, str], float] = {}

    outcome_quality_values: List[float] = []
    gap_return_values: List[float] = []
    process_reward_values: List[float] = []
    policy_delta_values: List[float] = []

    for packet in outcome_packets:
        edge = packet.edge
        if edge == ("", ""):
            continue
        outcome_quality_values.append(float(packet.quality_score))
        process_reward_values.append(float(packet.process_reward_delta))
        policy_delta_values.append(float(packet.policy_eval_delta))
        gap_return = float(packet.coverage_delta + 0.5 * packet.process_reward_delta + 0.5 * packet.policy_eval_delta)
        gap_return_values.append(gap_return)
        edge_metadata[edge] = {
            **edge_metadata.get(edge, {}),
            "latest_fill_method": packet.fill_method,
            "quality_score": float(packet.quality_score),
            "process_reward_delta": float(packet.process_reward_delta),
            "policy_eval_delta": float(packet.policy_eval_delta),
            "backend_health_score": float(packet.backend_health_score),
            "governance_status": packet.governance_status,
        }
        edge_econ_overlay[edge] = _clip01(
            0.5
            + 0.35 * float(packet.policy_eval_delta)
            + 0.25 * float(packet.process_reward_delta)
            - 0.15 * float(packet.cost_score)
        )
        edge_trust_overlay[edge] = _clip01(
            0.45
            + 0.35 * float(packet.quality_score)
            + 0.2 * float(packet.backend_health_score)
            - 0.25 * float(packet.metadata.get("disagreement_score", 0.0))
        )
        edge_readiness_overlay[edge] = _clip01(
            0.45
            + 0.25 * float(packet.quality_score)
            + 0.15 * float(packet.backend_health_score)
            + 0.15 * float(packet.coverage_delta)
        )

    for record in fill_records:
        edge = parse_edge_key(getattr(record, "edge_key", ""))
        if edge == ("", ""):
            continue
        marginal_value = _safe_float(getattr(record, "marginal_value", 0.0))
        edge_metadata.setdefault(edge, {})
        edge_metadata[edge]["marginal_value"] = marginal_value
        edge_econ_overlay[edge] = _clip01(max(edge_econ_overlay.get(edge, 0.0), 0.5 + 0.25 * marginal_value))
        edge_trust_overlay[edge] = _clip01(
            max(
                edge_trust_overlay.get(edge, 0.0),
                0.4 + 0.3 * _safe_float(getattr(record, "quality_score", 0.0)),
            )
        )
        edge_readiness_overlay[edge] = _clip01(
            max(
                edge_readiness_overlay.get(edge, 0.0),
                0.35 + 0.35 * _safe_float(getattr(record, "coverage_delta", 0.0)),
            )
        )

    for edge in blocked_edges:
        edge_metadata.setdefault(edge, {})
        edge_metadata[edge]["governance_blocked"] = True
        edge_readiness_overlay[edge] = 0.0

    validation_errors = [float(packet.error_score) for packet in validation_packets]
    severe_errors = [
        packet
        for packet in validation_packets
        if packet.severity in {"high", "critical"} or packet.error_score >= 0.5
    ]
    validation_summary = {
        "packet_count": len(validation_packets),
        "error_rate": float(_average(validation_errors, 0.0)),
        "high_error_count": len(severe_errors),
        "top_targets": [packet.target_ref for packet in severe_errors[:6]],
        "dominant_error_kinds": sorted(
            {packet.validation_kind for packet in severe_errors or validation_packets}
        )[:6],
    }

    for packet in severe_errors:
        edge = parse_edge_key(str(packet.metadata.get("edge_key", "")))
        if edge != ("", ""):
            edge_metadata.setdefault(edge, {})
            edge_metadata[edge]["wm_validation_pressure"] = float(packet.error_score)
            edge_readiness_overlay[edge] = min(edge_readiness_overlay.get(edge, 1.0), _clip01(0.5 - 0.4 * packet.error_score))

    process_reward_mean = _average(
        [_safe_float(item.get("phi_star", 0.0)) * _safe_float(item.get("confidence", 0.0)) for item in process_rewards],
        0.0,
    )
    process_reward_delta_mean = _average(
        [_safe_float(item.get("phi_star_delta", item.get("phi_star", 0.0))) for item in process_rewards],
        0.0,
    )
    backend_health_mean = _average(
        [
            _safe_float(item.get("evidence_density_score", item.get("backend_health_score", 0.0)))
            for item in backend_health
        ],
        _average([packet.backend_health_score for packet in outcome_packets], 1.0 if not outcome_packets else 0.0),
    )

    global_econ = _safe_float((econ_signals or {}).get("urgency", (econ_signals or {}).get("mpl_urgency", 0.0)), 0.0)
    global_w_econ = _safe_float((econ_signals or {}).get("w_econ", 0.0), 0.0)
    global_trust = _safe_float((trust_state or {}).get("calibration_score", 0.0), 0.0)

    trust_overlay = {
        "mean_signal": _clip01(0.4 * global_trust + 0.35 * _average(list(edge_trust_overlay.values()), global_trust) + 0.25 * backend_health_mean),
        "backend_health_mean": float(backend_health_mean),
        "wm_validation_penalty": float(min(validation_summary["error_rate"], 1.0)),
        "blocked_edge_fraction": (
            len(blocked_edges) / float(max(len(outcome_packets) + len(fill_records), 1))
        ),
    }
    econ_overlay = {
        "mean_signal": _clip01(
            0.35 * global_w_econ
            + 0.3 * global_econ
            + 0.2 * _average(list(edge_econ_overlay.values()), global_w_econ)
            + 0.15 * max(process_reward_mean, 0.0)
        ),
        "gap_return_mean": float(_average(gap_return_values, 0.0)),
        "policy_delta_mean": float(_average(policy_delta_values, 0.0)),
        "process_reward_mean": float(process_reward_mean),
    }

    mutation_proposals: List[GraphMutationProposal] = []
    proposal_index = 0
    for item in stage2_ontology_proposals or []:
        proposal_type = str(getattr(getattr(item, "proposal_type", None), "value", getattr(item, "proposal_type", "unknown")))
        target_ref = str(
            getattr(item, "target_object_id", None)
            or getattr(item, "target_affordance_type", None)
            or getattr(item, "target_skill_id", None)
            or proposal_type
        )
        action = "mark_for_review"
        if "affordance" in proposal_type:
            action = "add_provisional_affordance"
        elif "skill" in proposal_type:
            action = "add_provisional_skill"
        elif "relationship" in proposal_type:
            action = "update_relationship"
        elif "object_category" in proposal_type:
            action = "add_object_family"
        proposal_index += 1
        mutation_proposals.append(
            GraphMutationProposal(
                proposal_id=f"graph_mutation_{proposal_index:03d}",
                action=action,
                target_ref=target_ref,
                confidence=_safe_float(getattr(item, "confidence", 0.5), 0.5),
                rationale=str(getattr(item, "rationale", proposal_type)),
                source_refs=[str(getattr(item, "proposal_id", ""))] if getattr(item, "proposal_id", None) else [],
                metadata={
                    "proposal_type": proposal_type,
                    "priority": str(getattr(getattr(item, "priority", None), "value", getattr(item, "priority", ""))),
                },
            )
        )

    seen_targets = {proposal.target_ref for proposal in mutation_proposals}
    for packet in severe_errors:
        novelty_hint = str(packet.metadata.get("novel_ref", packet.target_ref)).strip()
        if not novelty_hint or novelty_hint in seen_targets:
            continue
        seen_targets.add(novelty_hint)
        proposal_index += 1
        mutation_proposals.append(
            GraphMutationProposal(
                proposal_id=f"graph_mutation_{proposal_index:03d}",
                action="add_provisional_skill" if "skill" in novelty_hint else "mark_for_review",
                target_ref=novelty_hint,
                confidence=_clip01(0.35 + 0.5 * float(packet.error_score)),
                rationale=f"Runtime validation exposed unsupported semantic state for {novelty_hint}",
                source_refs=[packet.target_ref],
                metadata={"validation_kind": packet.validation_kind, "severity": packet.severity},
            )
        )

    feedback_summary = {
        "coverage_outcome_count": len(outcome_packets),
        "fill_outcome_count": len(fill_records),
        "wm_validation_count": len(validation_packets),
        "wm_validation_error_rate": float(validation_summary["error_rate"]),
        "governance_blocked_count": len(blocked_edges),
        "blocked_edge_keys": sorted(blocked_edge_keys)[:16],
        "process_reward_mean": float(process_reward_mean),
        "process_reward_delta_mean": float(process_reward_delta_mean),
        "outcome_quality_mean": float(_average(outcome_quality_values, 0.0)),
        "gap_return_mean": float(_average(gap_return_values, 0.0)),
        "trust_overlay_mean": float(trust_overlay["mean_signal"]),
        "econ_overlay_mean": float(econ_overlay["mean_signal"]),
        "graph_mutation_pressure": float(len(mutation_proposals)),
        "graph_mutation_actions": [proposal.action for proposal in mutation_proposals[:8]],
    }

    return SemanticCoverageFeedback(
        feedback_summary=feedback_summary,
        edge_metadata=edge_metadata,
        edge_economic_overlay=edge_econ_overlay,
        edge_trust_overlay=edge_trust_overlay,
        edge_readiness_overlay=edge_readiness_overlay,
        trust_calibration_overlay=trust_overlay,
        econ_calibration_overlay=econ_overlay,
        wm_validation_summary=validation_summary,
        graph_mutation_proposals=mutation_proposals,
    )


__all__ = [
    "CoverageOutcomePacket",
    "GraphMutationProposal",
    "SemanticCoverageFeedback",
    "WMValidationPacket",
    "compile_semantic_coverage_feedback",
    "parse_edge_key",
]
