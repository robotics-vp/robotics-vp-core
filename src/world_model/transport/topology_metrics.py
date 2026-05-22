"""Topology and causal preservation metrics for Phase-6 transport scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import WMTransportBridgeContract

WM_TRANSPORT_TOPOLOGY_METRICS_VERSION = "wm_transport_topology_metrics_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, float(numerator) / float(denominator)))


@dataclass(frozen=True)
class WMTransportTopologyMetrics:
    """Decomposed local topology/causal/actionability metrics for a bridge."""

    metrics_id: str
    contract_id: str
    bridge_key: str
    topology_field_coverage: float
    causal_edge_coverage: float
    semantic_field_coverage: float
    actionability_field_coverage: float
    governance_constraint_coverage: float
    aggregate_score: float
    authority_class: str = "transport_topology_metrics_only"
    training_executed: bool = False
    promotion_eligible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_TOPOLOGY_METRICS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics_id": self.metrics_id,
            "version": self.version,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "topology_field_coverage": float(self.topology_field_coverage),
            "causal_edge_coverage": float(self.causal_edge_coverage),
            "semantic_field_coverage": float(self.semantic_field_coverage),
            "actionability_field_coverage": float(self.actionability_field_coverage),
            "governance_constraint_coverage": float(
                self.governance_constraint_coverage
            ),
            "aggregate_score": float(self.aggregate_score),
            "authority_class": self.authority_class,
            "training_executed": bool(self.training_executed),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportTopologyMetrics":
        return cls(
            metrics_id=str(payload.get("metrics_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            topology_field_coverage=float(payload.get("topology_field_coverage", 0.0)),
            causal_edge_coverage=float(payload.get("causal_edge_coverage", 0.0)),
            semantic_field_coverage=float(payload.get("semantic_field_coverage", 0.0)),
            actionability_field_coverage=float(
                payload.get("actionability_field_coverage", 0.0)
            ),
            governance_constraint_coverage=float(
                payload.get("governance_constraint_coverage", 0.0)
            ),
            aggregate_score=float(payload.get("aggregate_score", 0.0)),
            authority_class=str(
                payload.get("authority_class", "transport_topology_metrics_only")
            ),
            training_executed=bool(payload.get("training_executed", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_TOPOLOGY_METRICS_VERSION)),
        )


def compute_wm_transport_topology_metrics(
    contract: WMTransportBridgeContract,
) -> WMTransportTopologyMetrics:
    """Compute deterministic local topology metrics from contract structure."""

    topology_fields = contract.topology_map.topology_fields
    causal_edges = contract.causal_map.dependency_edges
    semantic_fields = contract.ontology_mapping.required_fields
    actionability_fields = contract.ontology_mapping.actionability_fields
    governance_constraints = contract.ontology_mapping.governance_constraints

    topology = (
        _ratio(len(topology_fields), len(topology_fields)) if topology_fields else 0.0
    )
    causal = _ratio(len(causal_edges), len(causal_edges)) if causal_edges else 0.0
    semantic = (
        _ratio(len(semantic_fields), len(semantic_fields)) if semantic_fields else 0.0
    )
    actionability = (
        _ratio(len(actionability_fields), len(actionability_fields))
        if actionability_fields
        else 0.0
    )
    governance = (
        _ratio(len(governance_constraints), len(governance_constraints))
        if governance_constraints
        else 0.0
    )
    validity_gate = 1.0 if contract.structurally_valid else 0.0
    aggregate = validity_gate * (
        0.3 * topology
        + 0.2 * causal
        + 0.2 * semantic
        + 0.15 * actionability
        + 0.15 * governance
    )
    payload = {
        "contract_id": contract.contract_id,
        "bridge_key": contract.bridge_key,
        "aggregate_score": aggregate,
    }
    return WMTransportTopologyMetrics(
        metrics_id=f"wm_transport_topology_metrics_{sha256_json(payload)[:16]}",
        contract_id=contract.contract_id,
        bridge_key=contract.bridge_key,
        topology_field_coverage=topology,
        causal_edge_coverage=causal,
        semantic_field_coverage=semantic,
        actionability_field_coverage=actionability,
        governance_constraint_coverage=governance,
        aggregate_score=aggregate,
        metadata={
            "structurally_valid": contract.structurally_valid,
            "raw_hidden_state_transport": contract.raw_hidden_state_transport,
        },
    )


__all__ = [
    "WM_TRANSPORT_TOPOLOGY_METRICS_VERSION",
    "WMTransportTopologyMetrics",
    "compute_wm_transport_topology_metrics",
]
