"""Phase-6.4 advisory runtime and decomposed evaluation for WM transport.

The runtime surfaces here are local scaffolding only. They emit proposals,
invocations, receipts, decomposed evaluation reports, and shadow outcome join
slots. They do not train transport weights, write weights, invoke providers,
execute hardware, control live policy, mutate reward math, or promote outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import WMTransportBridgeContract
from src.world_model.transport.neural_manifest import (
    WMTransportNeuralArchitectureManifest,
)
from src.world_model.transport.roundtrip import WMTransportRoundTripReceipt
from src.world_model.transport.training import WMTransportTrainerScaffoldManifest

WM_TRANSPORT_SHADOW_JOIN_SLOT_VERSION = "wm_transport_shadow_outcome_join_slot_v1"
WM_TRANSPORT_PROPOSAL_VERSION = "wm_transport_proposal_v1"
WM_TRANSPORT_INVOCATION_VERSION = "wm_transport_invocation_v1"
WM_TRANSPORT_RECEIPT_VERSION = "wm_transport_receipt_v1"
WM_TRANSPORT_DECOMPOSED_EVAL_REPORT_VERSION = (
    "wm_transport_decomposed_eval_report_v1"
)
WM_TRANSPORT_ADVISORY_RUNTIME_REPORT_VERSION = (
    "wm_transport_advisory_runtime_report_v1"
)
WM_TRANSPORT_UNITREE_EVENT_SPINE_JOIN_VERSION = (
    "wm_transport_unitree_event_spine_join_v1"
)

DENIED_TRANSPORT_RUNTIME_AUTHORITIES = (
    "training_execution",
    "weight_write",
    "provider_execution",
    "hardware_execution",
    "live_policy_control",
    "reward_math_mutation",
    "target_receiver_bypass",
    "promotion_decision",
)

PHASE64_ADVISORY_RUNTIME_BLOCKERS = (
    "transport_weights_not_trained",
    "gpu_transport_training_not_run",
    "provider_or_hardware_transport_evidence_missing",
    "topology_latency_evaluation_not_run",
    "promotion_grade_downstream_benchmark_missing",
    "live_authority_not_granted",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _shadow_payload(item: Any) -> Dict[str, Any]:
    if hasattr(item, "to_dict"):
        return _mapping(item.to_dict())
    if isinstance(item, Mapping):
        return _mapping(item)
    return {}


@dataclass(frozen=True)
class WMTransportShadowOutcomeJoinSlot:
    """Join slot from a transport receipt to an available shadow outcome."""

    slot_id: str
    proposal_id: str
    contract_id: str
    bridge_key: str
    target_wm: str
    join_status: str
    shadow_outcome_receipt_id: str = ""
    work_order_id: str = ""
    allocation_label: str = ""
    recommended_action: str = ""
    local_structural_outcome_available: bool = False
    promotion_grade_outcome: bool = False
    outcome_metrics: Dict[str, float] = field(default_factory=dict)
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    authority_class: str = "transport_shadow_outcome_join_slot_only"
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_SHADOW_JOIN_SLOT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "target_wm": self.target_wm,
            "join_status": self.join_status,
            "shadow_outcome_receipt_id": self.shadow_outcome_receipt_id,
            "work_order_id": self.work_order_id,
            "allocation_label": self.allocation_label,
            "recommended_action": self.recommended_action,
            "local_structural_outcome_available": bool(
                self.local_structural_outcome_available
            ),
            "promotion_grade_outcome": bool(self.promotion_grade_outcome),
            "outcome_metrics": _float_dict(self.outcome_metrics),
            "evidence_refs": _mapping(self.evidence_refs),
            "authority_class": self.authority_class,
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportShadowOutcomeJoinSlot":
        return cls(
            slot_id=str(payload.get("slot_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            target_wm=str(payload.get("target_wm", "")),
            join_status=str(payload.get("join_status", "")),
            shadow_outcome_receipt_id=str(
                payload.get("shadow_outcome_receipt_id", "")
            ),
            work_order_id=str(payload.get("work_order_id", "")),
            allocation_label=str(payload.get("allocation_label", "")),
            recommended_action=str(payload.get("recommended_action", "")),
            local_structural_outcome_available=bool(
                payload.get("local_structural_outcome_available", False)
            ),
            promotion_grade_outcome=bool(payload.get("promotion_grade_outcome", False)),
            outcome_metrics=_float_dict(payload.get("outcome_metrics", {})),
            evidence_refs=_mapping(payload.get("evidence_refs")),
            authority_class=str(
                payload.get(
                    "authority_class", "transport_shadow_outcome_join_slot_only"
                )
            ),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_SHADOW_JOIN_SLOT_VERSION)
            ),
        )


@dataclass(frozen=True)
class WMTransportUnitreeEventSpineJoin:
    """Lower-WM event-spine label joined to one advisory transport proposal."""

    join_id: str
    proposal_id: str
    contract_id: str
    bridge_key: str
    source_wm: str
    target_wm: str
    join_status: str
    event_spine_ref: str
    event_count: int
    event_ids: list[str] = field(default_factory=list)
    event_kinds: list[str] = field(default_factory=list)
    lower_wm_label_only: bool = True
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_UNITREE_EVENT_SPINE_JOIN_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "join_id": self.join_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_wm": self.source_wm,
            "target_wm": self.target_wm,
            "join_status": self.join_status,
            "event_spine_ref": self.event_spine_ref,
            "event_count": int(self.event_count),
            "event_ids": list(self.event_ids),
            "event_kinds": list(self.event_kinds),
            "lower_wm_label_only": bool(self.lower_wm_label_only),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportUnitreeEventSpineJoin":
        return cls(
            join_id=str(payload.get("join_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_wm=str(payload.get("source_wm", "")),
            target_wm=str(payload.get("target_wm", "")),
            join_status=str(payload.get("join_status", "")),
            event_spine_ref=str(payload.get("event_spine_ref", "")),
            event_count=int(payload.get("event_count", 0) or 0),
            event_ids=[
                str(item) for item in list(payload.get("event_ids", []) or [])
            ],
            event_kinds=[
                str(item) for item in list(payload.get("event_kinds", []) or [])
            ],
            lower_wm_label_only=bool(payload.get("lower_wm_label_only", True)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", WM_TRANSPORT_UNITREE_EVENT_SPINE_JOIN_VERSION
                )
            ),
        )


@dataclass(frozen=True)
class TransportProposal:
    """Advisory proposal for one adjacent-WM transport invocation."""

    proposal_id: str
    contract_id: str
    bridge_key: str
    source_wm: str
    target_wm: str
    source_object_ref: str
    source_object_version: str
    target_intake_ref: str
    target_state_version: str
    source_exporter_id: str
    target_receiver_id: str
    topology_fields: list[str] = field(default_factory=list)
    causal_edges: list[str] = field(default_factory=list)
    required_semantic_fields: list[str] = field(default_factory=list)
    governance_constraints: list[str] = field(default_factory=list)
    uncertainty_profile_id: str = ""
    provenance_id: str = ""
    shadow_outcome_slot_id: str = ""
    authority_class: str = "transport_proposal_advisory_only"
    advisory_only: bool = True
    receiver_required: bool = True
    target_receiver_bypassed: bool = False
    denied_authority: list[str] = field(
        default_factory=lambda: list(DENIED_TRANSPORT_RUNTIME_AUTHORITIES)
    )
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_PROPOSAL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "version": self.version,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_wm": self.source_wm,
            "target_wm": self.target_wm,
            "source_object_ref": self.source_object_ref,
            "source_object_version": self.source_object_version,
            "target_intake_ref": self.target_intake_ref,
            "target_state_version": self.target_state_version,
            "source_exporter_id": self.source_exporter_id,
            "target_receiver_id": self.target_receiver_id,
            "topology_fields": list(self.topology_fields),
            "causal_edges": list(self.causal_edges),
            "required_semantic_fields": list(self.required_semantic_fields),
            "governance_constraints": list(self.governance_constraints),
            "uncertainty_profile_id": self.uncertainty_profile_id,
            "provenance_id": self.provenance_id,
            "shadow_outcome_slot_id": self.shadow_outcome_slot_id,
            "authority_class": self.authority_class,
            "advisory_only": bool(self.advisory_only),
            "receiver_required": bool(self.receiver_required),
            "target_receiver_bypassed": bool(self.target_receiver_bypassed),
            "denied_authority": list(self.denied_authority),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_wm=str(payload.get("source_wm", "")),
            target_wm=str(payload.get("target_wm", "")),
            source_object_ref=str(payload.get("source_object_ref", "")),
            source_object_version=str(payload.get("source_object_version", "")),
            target_intake_ref=str(payload.get("target_intake_ref", "")),
            target_state_version=str(payload.get("target_state_version", "")),
            source_exporter_id=str(payload.get("source_exporter_id", "")),
            target_receiver_id=str(payload.get("target_receiver_id", "")),
            topology_fields=[
                str(item) for item in list(payload.get("topology_fields", []) or [])
            ],
            causal_edges=[
                str(item) for item in list(payload.get("causal_edges", []) or [])
            ],
            required_semantic_fields=[
                str(item)
                for item in list(payload.get("required_semantic_fields", []) or [])
            ],
            governance_constraints=[
                str(item)
                for item in list(payload.get("governance_constraints", []) or [])
            ],
            uncertainty_profile_id=str(payload.get("uncertainty_profile_id", "")),
            provenance_id=str(payload.get("provenance_id", "")),
            shadow_outcome_slot_id=str(payload.get("shadow_outcome_slot_id", "")),
            authority_class=str(
                payload.get("authority_class", "transport_proposal_advisory_only")
            ),
            advisory_only=bool(payload.get("advisory_only", True)),
            receiver_required=bool(payload.get("receiver_required", True)),
            target_receiver_bypassed=bool(
                payload.get("target_receiver_bypassed", False)
            ),
            denied_authority=[
                str(item) for item in list(payload.get("denied_authority", []) or [])
            ],
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_PROPOSAL_VERSION)),
        )


@dataclass(frozen=True)
class TransportInvocation:
    """Local advisory invocation record for one transport proposal."""

    invocation_id: str
    proposal_id: str
    contract_id: str
    bridge_key: str
    source_exporter_id: str
    bridge_contract_id: str
    target_receiver_id: str
    neural_manifest_id: str
    trainer_scaffold_id: str
    runtime_mode: str = "local_advisory_shadow"
    invocation_status: str = "receipt_emitted"
    operation_sequence: list[str] = field(default_factory=list)
    authority_class: str = "transport_invocation_advisory_only"
    advisory_only: bool = True
    target_receiver_bypassed: bool = False
    denied_authority: list[str] = field(
        default_factory=lambda: list(DENIED_TRANSPORT_RUNTIME_AUTHORITIES)
    )
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_INVOCATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_exporter_id": self.source_exporter_id,
            "bridge_contract_id": self.bridge_contract_id,
            "target_receiver_id": self.target_receiver_id,
            "neural_manifest_id": self.neural_manifest_id,
            "trainer_scaffold_id": self.trainer_scaffold_id,
            "runtime_mode": self.runtime_mode,
            "invocation_status": self.invocation_status,
            "operation_sequence": list(self.operation_sequence),
            "authority_class": self.authority_class,
            "advisory_only": bool(self.advisory_only),
            "target_receiver_bypassed": bool(self.target_receiver_bypassed),
            "denied_authority": list(self.denied_authority),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportInvocation":
        return cls(
            invocation_id=str(payload.get("invocation_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_exporter_id=str(payload.get("source_exporter_id", "")),
            bridge_contract_id=str(payload.get("bridge_contract_id", "")),
            target_receiver_id=str(payload.get("target_receiver_id", "")),
            neural_manifest_id=str(payload.get("neural_manifest_id", "")),
            trainer_scaffold_id=str(payload.get("trainer_scaffold_id", "")),
            runtime_mode=str(payload.get("runtime_mode", "local_advisory_shadow")),
            invocation_status=str(payload.get("invocation_status", "receipt_emitted")),
            operation_sequence=[
                str(item)
                for item in list(payload.get("operation_sequence", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "transport_invocation_advisory_only")
            ),
            advisory_only=bool(payload.get("advisory_only", True)),
            target_receiver_bypassed=bool(
                payload.get("target_receiver_bypassed", False)
            ),
            denied_authority=[
                str(item) for item in list(payload.get("denied_authority", []) or [])
            ],
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_INVOCATION_VERSION)),
        )


@dataclass(frozen=True)
class TransportReceipt:
    """Receipt for one advisory transport proposal and invocation."""

    receipt_id: str
    proposal_id: str
    invocation_id: str
    eval_report_id: str
    contract_id: str
    bridge_key: str
    source_wm: str
    target_wm: str
    transformed_object_ref: str
    target_receiver_id: str
    receiver_actionable: bool
    topology_survived: bool
    uncertainty_survived: bool
    provenance_preserved: bool
    governance_satisfied: bool
    topology_preservation_score: float
    uncertainty_calibration_score: float
    receiver_actionability_score: float
    provenance_score: float
    governance_score: float
    aggregate_score: float
    shadow_outcome_slot: WMTransportShadowOutcomeJoinSlot
    authority_class: str = "transport_receipt_advisory_only"
    advisory_only: bool = True
    target_receiver_bypassed: bool = False
    denied_authority: list[str] = field(
        default_factory=lambda: list(DENIED_TRANSPORT_RUNTIME_AUTHORITIES)
    )
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_RECEIPT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "invocation_id": self.invocation_id,
            "eval_report_id": self.eval_report_id,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_wm": self.source_wm,
            "target_wm": self.target_wm,
            "transformed_object_ref": self.transformed_object_ref,
            "target_receiver_id": self.target_receiver_id,
            "receiver_actionable": bool(self.receiver_actionable),
            "topology_survived": bool(self.topology_survived),
            "uncertainty_survived": bool(self.uncertainty_survived),
            "provenance_preserved": bool(self.provenance_preserved),
            "governance_satisfied": bool(self.governance_satisfied),
            "topology_preservation_score": float(self.topology_preservation_score),
            "uncertainty_calibration_score": float(
                self.uncertainty_calibration_score
            ),
            "receiver_actionability_score": float(
                self.receiver_actionability_score
            ),
            "provenance_score": float(self.provenance_score),
            "governance_score": float(self.governance_score),
            "aggregate_score": float(self.aggregate_score),
            "shadow_outcome_slot": self.shadow_outcome_slot.to_dict(),
            "authority_class": self.authority_class,
            "advisory_only": bool(self.advisory_only),
            "target_receiver_bypassed": bool(self.target_receiver_bypassed),
            "denied_authority": list(self.denied_authority),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            invocation_id=str(payload.get("invocation_id", "")),
            eval_report_id=str(payload.get("eval_report_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_wm=str(payload.get("source_wm", "")),
            target_wm=str(payload.get("target_wm", "")),
            transformed_object_ref=str(payload.get("transformed_object_ref", "")),
            target_receiver_id=str(payload.get("target_receiver_id", "")),
            receiver_actionable=bool(payload.get("receiver_actionable", False)),
            topology_survived=bool(payload.get("topology_survived", False)),
            uncertainty_survived=bool(payload.get("uncertainty_survived", False)),
            provenance_preserved=bool(payload.get("provenance_preserved", False)),
            governance_satisfied=bool(payload.get("governance_satisfied", False)),
            topology_preservation_score=float(
                payload.get("topology_preservation_score", 0.0)
            ),
            uncertainty_calibration_score=float(
                payload.get("uncertainty_calibration_score", 0.0)
            ),
            receiver_actionability_score=float(
                payload.get("receiver_actionability_score", 0.0)
            ),
            provenance_score=float(payload.get("provenance_score", 0.0)),
            governance_score=float(payload.get("governance_score", 0.0)),
            aggregate_score=float(payload.get("aggregate_score", 0.0)),
            shadow_outcome_slot=WMTransportShadowOutcomeJoinSlot.from_dict(
                dict(payload.get("shadow_outcome_slot", {}) or {})
            ),
            authority_class=str(
                payload.get("authority_class", "transport_receipt_advisory_only")
            ),
            advisory_only=bool(payload.get("advisory_only", True)),
            target_receiver_bypassed=bool(
                payload.get("target_receiver_bypassed", False)
            ),
            denied_authority=[
                str(item) for item in list(payload.get("denied_authority", []) or [])
            ],
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class WMTransportDecomposedEvalReport:
    """Bridge/receiver/downstream/joint/interaction eval report."""

    eval_report_id: str
    proposal_id: str
    invocation_id: str
    receipt_id: str
    contract_id: str
    bridge_key: str
    bridge_only_score: float
    receiver_only_score: float
    downstream_only_score: float
    joint_score: float
    interaction_effect: float
    interaction_class: str
    shadow_outcome_join_status: str
    shadow_outcome_receipt_id: str = ""
    terms: Dict[str, float] = field(default_factory=dict)
    authority_class: str = "transport_decomposed_eval_report_only"
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_DECOMPOSED_EVAL_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_report_id": self.eval_report_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "invocation_id": self.invocation_id,
            "receipt_id": self.receipt_id,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "bridge_only_score": float(self.bridge_only_score),
            "receiver_only_score": float(self.receiver_only_score),
            "downstream_only_score": float(self.downstream_only_score),
            "joint_score": float(self.joint_score),
            "interaction_effect": float(self.interaction_effect),
            "interaction_class": self.interaction_class,
            "shadow_outcome_join_status": self.shadow_outcome_join_status,
            "shadow_outcome_receipt_id": self.shadow_outcome_receipt_id,
            "terms": _float_dict(self.terms),
            "authority_class": self.authority_class,
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportDecomposedEvalReport":
        return cls(
            eval_report_id=str(payload.get("eval_report_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            invocation_id=str(payload.get("invocation_id", "")),
            receipt_id=str(payload.get("receipt_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            bridge_only_score=float(payload.get("bridge_only_score", 0.0)),
            receiver_only_score=float(payload.get("receiver_only_score", 0.0)),
            downstream_only_score=float(payload.get("downstream_only_score", 0.0)),
            joint_score=float(payload.get("joint_score", 0.0)),
            interaction_effect=float(payload.get("interaction_effect", 0.0)),
            interaction_class=str(payload.get("interaction_class", "")),
            shadow_outcome_join_status=str(
                payload.get("shadow_outcome_join_status", "")
            ),
            shadow_outcome_receipt_id=str(
                payload.get("shadow_outcome_receipt_id", "")
            ),
            terms=_float_dict(payload.get("terms", {})),
            authority_class=str(
                payload.get(
                    "authority_class", "transport_decomposed_eval_report_only"
                )
            ),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_DECOMPOSED_EVAL_REPORT_VERSION)
            ),
        )


@dataclass(frozen=True)
class WMTransportAdvisoryRuntimeReport:
    """Top-level report for the Phase-6.4 advisory runtime artifact pass."""

    report_id: str
    neural_manifest_id: str
    trainer_scaffold_id: str
    proposal_count: int
    invocation_count: int
    receipt_count: int
    eval_report_count: int
    shadow_join_slot_count: int
    joined_shadow_outcome_count: int
    status: str
    authority_class: str = "transport_advisory_runtime_report_only"
    ready_for_decomposed_eval: bool = False
    ready_for_training: bool = False
    ready_for_gpu_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_authority: list[str] = field(
        default_factory=lambda: list(DENIED_TRANSPORT_RUNTIME_AUTHORITIES)
    )
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_ADVISORY_RUNTIME_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "neural_manifest_id": self.neural_manifest_id,
            "trainer_scaffold_id": self.trainer_scaffold_id,
            "proposal_count": int(self.proposal_count),
            "invocation_count": int(self.invocation_count),
            "receipt_count": int(self.receipt_count),
            "eval_report_count": int(self.eval_report_count),
            "shadow_join_slot_count": int(self.shadow_join_slot_count),
            "joined_shadow_outcome_count": int(self.joined_shadow_outcome_count),
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_decomposed_eval": bool(self.ready_for_decomposed_eval),
            "ready_for_training": bool(self.ready_for_training),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_authority": list(self.denied_authority),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportAdvisoryRuntimeReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            neural_manifest_id=str(payload.get("neural_manifest_id", "")),
            trainer_scaffold_id=str(payload.get("trainer_scaffold_id", "")),
            proposal_count=int(payload.get("proposal_count", 0) or 0),
            invocation_count=int(payload.get("invocation_count", 0) or 0),
            receipt_count=int(payload.get("receipt_count", 0) or 0),
            eval_report_count=int(payload.get("eval_report_count", 0) or 0),
            shadow_join_slot_count=int(payload.get("shadow_join_slot_count", 0) or 0),
            joined_shadow_outcome_count=int(
                payload.get("joined_shadow_outcome_count", 0) or 0
            ),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get(
                    "authority_class", "transport_advisory_runtime_report_only"
                )
            ),
            ready_for_decomposed_eval=bool(
                payload.get("ready_for_decomposed_eval", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            ready_for_gpu_training=bool(
                payload.get("ready_for_gpu_training", False)
            ),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_authority=[
                str(item) for item in list(payload.get("denied_authority", []) or [])
            ],
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_ADVISORY_RUNTIME_REPORT_VERSION)
            ),
        )


def _proposal_for_contract(
    contract: WMTransportBridgeContract, *, shadow_slot_id: str
) -> TransportProposal:
    payload = {
        "contract_id": contract.contract_id,
        "bridge_key": contract.bridge_key,
        "source_ref": contract.source_endpoint.state_ref,
        "target_ref": contract.target_endpoint.state_ref,
    }
    return TransportProposal(
        proposal_id=f"wm_transport_proposal_{sha256_json(payload)[:16]}",
        contract_id=contract.contract_id,
        bridge_key=contract.bridge_key,
        source_wm=contract.source_endpoint.wm_key,
        target_wm=contract.target_endpoint.wm_key,
        source_object_ref=contract.source_endpoint.state_ref,
        source_object_version=contract.source_endpoint.state_version,
        target_intake_ref=contract.target_endpoint.state_ref,
        target_state_version=contract.target_endpoint.state_version,
        source_exporter_id=contract.source_endpoint.transformer_id,
        target_receiver_id=contract.target_endpoint.transformer_id,
        topology_fields=list(contract.topology_map.topology_fields),
        causal_edges=list(contract.causal_map.dependency_edges),
        required_semantic_fields=list(contract.ontology_mapping.required_fields),
        governance_constraints=list(contract.ontology_mapping.governance_constraints),
        uncertainty_profile_id=contract.uncertainty_profile.profile_id,
        provenance_id=contract.provenance.provenance_id,
        shadow_outcome_slot_id=shadow_slot_id,
        blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
        metadata={
            "phase": "6.4_transport_advisory_runtime",
            "proposal_claim": "advisory_only",
            "contract_structurally_valid": contract.structurally_valid,
        },
    )


def _shadow_slot(
    *,
    proposal_id: str,
    contract: WMTransportBridgeContract,
    shadow_receipts: list[Dict[str, Any]],
    economic_join_index: int,
) -> WMTransportShadowOutcomeJoinSlot:
    base = {
        "proposal_id": proposal_id,
        "contract_id": contract.contract_id,
        "bridge_key": contract.bridge_key,
    }
    if contract.target_endpoint.wm_key != "economic":
        return WMTransportShadowOutcomeJoinSlot(
            slot_id=f"wm_transport_shadow_slot_{sha256_json(base)[:16]}",
            proposal_id=proposal_id,
            contract_id=contract.contract_id,
            bridge_key=contract.bridge_key,
            target_wm=contract.target_endpoint.wm_key,
            join_status="not_applicable_non_economic_target",
            blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
            metadata={"shadow_outcome_join_reason": "target_wm_is_not_economic"},
        )
    if not shadow_receipts:
        return WMTransportShadowOutcomeJoinSlot(
            slot_id=f"wm_transport_shadow_slot_{sha256_json(base)[:16]}",
            proposal_id=proposal_id,
            contract_id=contract.contract_id,
            bridge_key=contract.bridge_key,
            target_wm=contract.target_endpoint.wm_key,
            join_status="awaiting_shadow_outcome_receipt",
            blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
            metadata={"shadow_outcome_join_reason": "no_local_shadow_outcome_file"},
        )

    receipt = shadow_receipts[economic_join_index % len(shadow_receipts)]
    observed = _float_dict(receipt.get("observed_effects", {}))
    comparison = _float_dict(receipt.get("comparison_metrics", {}))
    metrics = {
        **observed,
        **{f"comparison_{key}": value for key, value in comparison.items()},
    }
    return WMTransportShadowOutcomeJoinSlot(
        slot_id=f"wm_transport_shadow_slot_{sha256_json({**base, **receipt})[:16]}",
        proposal_id=proposal_id,
        contract_id=contract.contract_id,
        bridge_key=contract.bridge_key,
        target_wm=contract.target_endpoint.wm_key,
        join_status="joined_local_structural_shadow_outcome",
        shadow_outcome_receipt_id=str(receipt.get("receipt_id", "")),
        work_order_id=str(receipt.get("work_order_id", "")),
        allocation_label=str(receipt.get("allocation_label", "")),
        recommended_action=str(receipt.get("recommended_action", "")),
        local_structural_outcome_available=True,
        promotion_grade_outcome=False,
        outcome_metrics=metrics,
        evidence_refs=_mapping(receipt.get("evidence_refs")),
        blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
        metadata={
            "shadow_outcome_join_reason": "economic_target_and_local_structural_receipt_available",
            "promotion_grade_evidence": False,
        },
    )


def _invocation_for_proposal(
    *,
    proposal: TransportProposal,
    neural_manifest: WMTransportNeuralArchitectureManifest,
    trainer_manifest: WMTransportTrainerScaffoldManifest,
) -> TransportInvocation:
    payload = {
        "proposal_id": proposal.proposal_id,
        "neural_manifest_id": neural_manifest.manifest_id,
        "trainer_scaffold_id": trainer_manifest.trainer_scaffold_id,
    }
    return TransportInvocation(
        invocation_id=f"wm_transport_invocation_{sha256_json(payload)[:16]}",
        proposal_id=proposal.proposal_id,
        contract_id=proposal.contract_id,
        bridge_key=proposal.bridge_key,
        source_exporter_id=proposal.source_exporter_id,
        bridge_contract_id=proposal.contract_id,
        target_receiver_id=proposal.target_receiver_id,
        neural_manifest_id=neural_manifest.manifest_id,
        trainer_scaffold_id=trainer_manifest.trainer_scaffold_id,
        operation_sequence=[
            "read_source_typed_object",
            "apply_source_exporter_contract",
            "apply_isomorphic_bridge_contract",
            "apply_target_receiver_contract",
            "emit_advisory_receipt",
        ],
        blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
        metadata={
            "phase": "6.4_transport_advisory_runtime",
            "runtime_claim": "local_receipt_materialization_only",
        },
    )


def _downstream_score(slot: WMTransportShadowOutcomeJoinSlot) -> float:
    if not slot.local_structural_outcome_available:
        return 0.0
    metrics = slot.outcome_metrics
    structural = metrics.get("local_structural_loop_closed", 0.0)
    supervision = metrics.get("supervision_record_coverage", 0.0)
    target = metrics.get("value_target_pack_coverage", 0.0)
    ledger = metrics.get("value_ledger_receipt_coverage", 0.0)
    promotion = metrics.get("comparison_promotion_grade_evidence_observed", 0.0)
    return _clamp(
        0.3 * structural
        + 0.3 * supervision
        + 0.2 * target
        + 0.15 * ledger
        + 0.05 * promotion
    )


def _scores(
    *,
    roundtrip: WMTransportRoundTripReceipt,
    slot: WMTransportShadowOutcomeJoinSlot,
) -> Dict[str, float]:
    topology = roundtrip.topology_metrics.aggregate_score
    uncertainty = roundtrip.uncertainty_calibration.calibration_score
    source = roundtrip.source_reconstruction_score
    receiver = roundtrip.target_receiver_actionability_score
    downstream = _downstream_score(slot)
    bridge_only = _clamp(0.45 * topology + 0.35 * uncertainty + 0.2 * source)
    receiver_only = _clamp(receiver)
    joint = _clamp(roundtrip.aggregate_score)
    downstream_factor = downstream if slot.local_structural_outcome_available else 1.0
    independent = _clamp(bridge_only * receiver_only * downstream_factor)
    interaction = max(-1.0, min(1.0, joint - independent))
    return {
        "bridge_only_score": bridge_only,
        "receiver_only_score": receiver_only,
        "downstream_only_score": downstream,
        "joint_score": joint,
        "interaction_effect": interaction,
        "topology_score": _clamp(topology),
        "uncertainty_score": _clamp(uncertainty),
        "source_export_score": _clamp(source),
        "roundtrip_score": _clamp(roundtrip.roundtrip_consistency_score),
    }


def _interaction_class(effect: float) -> str:
    if effect > 0.05:
        return "positive_interaction"
    if effect < -0.05:
        return "negative_interaction"
    return "neutral_interaction"


def _eval_report(
    *,
    proposal: TransportProposal,
    invocation: TransportInvocation,
    receipt_id: str,
    roundtrip: WMTransportRoundTripReceipt,
    shadow_slot: WMTransportShadowOutcomeJoinSlot,
) -> WMTransportDecomposedEvalReport:
    scores = _scores(roundtrip=roundtrip, slot=shadow_slot)
    payload = {
        "proposal_id": proposal.proposal_id,
        "invocation_id": invocation.invocation_id,
        "contract_id": proposal.contract_id,
        "scores": scores,
    }
    return WMTransportDecomposedEvalReport(
        eval_report_id=f"wm_transport_eval_{sha256_json(payload)[:16]}",
        proposal_id=proposal.proposal_id,
        invocation_id=invocation.invocation_id,
        receipt_id=receipt_id,
        contract_id=proposal.contract_id,
        bridge_key=proposal.bridge_key,
        bridge_only_score=scores["bridge_only_score"],
        receiver_only_score=scores["receiver_only_score"],
        downstream_only_score=scores["downstream_only_score"],
        joint_score=scores["joint_score"],
        interaction_effect=scores["interaction_effect"],
        interaction_class=_interaction_class(scores["interaction_effect"]),
        shadow_outcome_join_status=shadow_slot.join_status,
        shadow_outcome_receipt_id=shadow_slot.shadow_outcome_receipt_id,
        terms=scores,
        blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
        metadata={
            "phase": "6.4_transport_decomposed_eval",
            "bridge_receiver_downstream_decomposed": True,
            "promotion_claim": False,
        },
    )


def _receipt(
    *,
    proposal: TransportProposal,
    invocation: TransportInvocation,
    eval_report_id: str,
    roundtrip: WMTransportRoundTripReceipt,
    shadow_slot: WMTransportShadowOutcomeJoinSlot,
) -> TransportReceipt:
    topology = roundtrip.topology_metrics.aggregate_score
    uncertainty = roundtrip.uncertainty_calibration.calibration_score
    receiver = roundtrip.target_receiver_actionability_score
    provenance = 1.0
    governance = roundtrip.topology_metrics.governance_constraint_coverage
    aggregate = _clamp(
        0.25 * topology
        + 0.2 * uncertainty
        + 0.25 * receiver
        + 0.15 * provenance
        + 0.15 * governance
    )
    payload = {
        "proposal_id": proposal.proposal_id,
        "invocation_id": invocation.invocation_id,
        "eval_report_id": eval_report_id,
        "aggregate": aggregate,
    }
    return TransportReceipt(
        receipt_id=f"wm_transport_receipt_{sha256_json(payload)[:16]}",
        proposal_id=proposal.proposal_id,
        invocation_id=invocation.invocation_id,
        eval_report_id=eval_report_id,
        contract_id=proposal.contract_id,
        bridge_key=proposal.bridge_key,
        source_wm=proposal.source_wm,
        target_wm=proposal.target_wm,
        transformed_object_ref=proposal.target_intake_ref,
        target_receiver_id=proposal.target_receiver_id,
        receiver_actionable=receiver > 0.0,
        topology_survived=topology > 0.0,
        uncertainty_survived=uncertainty > 0.0,
        provenance_preserved=True,
        governance_satisfied=governance > 0.0,
        topology_preservation_score=topology,
        uncertainty_calibration_score=uncertainty,
        receiver_actionability_score=receiver,
        provenance_score=provenance,
        governance_score=governance,
        aggregate_score=aggregate,
        shadow_outcome_slot=shadow_slot,
        blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
        metadata={
            "phase": "6.4_transport_advisory_receipt",
            "receipt_claim": "local_advisory_only",
            "downstream_shadow_outcome_join_status": shadow_slot.join_status,
        },
    )


def build_wm_transport_advisory_runtime(
    *,
    contracts: Iterable[WMTransportBridgeContract],
    roundtrip_receipts: Iterable[WMTransportRoundTripReceipt],
    neural_manifest: WMTransportNeuralArchitectureManifest,
    trainer_manifest: WMTransportTrainerScaffoldManifest,
    shadow_outcome_receipts: Optional[Iterable[Any]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    WMTransportAdvisoryRuntimeReport,
    list[TransportProposal],
    list[TransportInvocation],
    list[TransportReceipt],
    list[WMTransportDecomposedEvalReport],
]:
    """Build Phase-6.4 advisory runtime artifacts from existing scaffolds."""

    receipt_by_contract = {item.contract_id: item for item in roundtrip_receipts}
    shadow_receipts = [_shadow_payload(item) for item in shadow_outcome_receipts or []]
    proposals: list[TransportProposal] = []
    invocations: list[TransportInvocation] = []
    receipts: list[TransportReceipt] = []
    eval_reports: list[WMTransportDecomposedEvalReport] = []
    economic_join_index = 0

    for contract in contracts:
        proposal_seed = {
            "contract_id": contract.contract_id,
            "bridge_key": contract.bridge_key,
            "source_ref": contract.source_endpoint.state_ref,
            "target_ref": contract.target_endpoint.state_ref,
        }
        proposal_id = f"wm_transport_proposal_{sha256_json(proposal_seed)[:16]}"
        shadow_slot = _shadow_slot(
            proposal_id=proposal_id,
            contract=contract,
            shadow_receipts=shadow_receipts,
            economic_join_index=economic_join_index,
        )
        if contract.target_endpoint.wm_key == "economic":
            economic_join_index += 1
        proposal = _proposal_for_contract(
            contract, shadow_slot_id=shadow_slot.slot_id
        )
        invocation = _invocation_for_proposal(
            proposal=proposal,
            neural_manifest=neural_manifest,
            trainer_manifest=trainer_manifest,
        )
        roundtrip = receipt_by_contract[contract.contract_id]
        receipt_seed = {
            "proposal_id": proposal.proposal_id,
            "invocation_id": invocation.invocation_id,
            "contract_id": proposal.contract_id,
            "bridge_key": proposal.bridge_key,
        }
        provisional_receipt_id = (
            f"wm_transport_receipt_{sha256_json(receipt_seed)[:16]}"
        )
        eval_report = _eval_report(
            proposal=proposal,
            invocation=invocation,
            receipt_id=provisional_receipt_id,
            roundtrip=roundtrip,
            shadow_slot=shadow_slot,
        )
        receipt = _receipt(
            proposal=proposal,
            invocation=invocation,
            eval_report_id=eval_report.eval_report_id,
            roundtrip=roundtrip,
            shadow_slot=shadow_slot,
        )
        eval_report = WMTransportDecomposedEvalReport(
            **{
                **eval_report.__dict__,
                "receipt_id": receipt.receipt_id,
            }
        )
        proposals.append(proposal)
        invocations.append(invocation)
        receipts.append(receipt)
        eval_reports.append(eval_report)

    joined = sum(
        1
        for item in receipts
        if item.shadow_outcome_slot.join_status == "joined_local_structural_shadow_outcome"
    )
    expected_count = len(proposals)
    status = (
        "ok"
        if expected_count
        and expected_count == len(invocations) == len(receipts) == len(eval_reports)
        and not trainer_manifest.training_executed
        and not trainer_manifest.weights_written
        and not trainer_manifest.promotion_eligible
        and not trainer_manifest.reward_math_mutation
        else "blocked"
    )
    payload = {
        "neural_manifest_id": neural_manifest.manifest_id,
        "trainer_scaffold_id": trainer_manifest.trainer_scaffold_id,
        "proposal_ids": [item.proposal_id for item in proposals],
        "eval_report_ids": [item.eval_report_id for item in eval_reports],
    }
    report = WMTransportAdvisoryRuntimeReport(
        report_id=f"wm_transport_advisory_runtime_{sha256_json(payload)[:16]}",
        neural_manifest_id=neural_manifest.manifest_id,
        trainer_scaffold_id=trainer_manifest.trainer_scaffold_id,
        proposal_count=len(proposals),
        invocation_count=len(invocations),
        receipt_count=len(receipts),
        eval_report_count=len(eval_reports),
        shadow_join_slot_count=len(receipts),
        joined_shadow_outcome_count=joined,
        status=status,
        ready_for_decomposed_eval=status == "ok",
        blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
        aggregate_counts={
            "proposal_count": float(len(proposals)),
            "invocation_count": float(len(invocations)),
            "receipt_count": float(len(receipts)),
            "eval_report_count": float(len(eval_reports)),
            "shadow_join_slot_count": float(len(receipts)),
            "joined_shadow_outcome_count": float(joined),
            "available_shadow_outcome_count": float(len(shadow_receipts)),
        },
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "phase": "6.4_transport_advisory_runtime",
            "boundary": "advisory proposals, receipts, decomposed eval only",
            **_mapping(metadata),
        },
    )
    return report, proposals, invocations, receipts, eval_reports


def _event_rows(event_spine_payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        _mapping(row)
        for row in list(event_spine_payload.get("events", []) or [])
        if isinstance(row, Mapping)
    ]


def build_wm_transport_unitree_event_spine_joins(
    *,
    proposals: Iterable[TransportProposal],
    event_spine_payload: Mapping[str, Any],
    event_spine_ref: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> list[WMTransportUnitreeEventSpineJoin]:
    """Build lower-WM Unitree event-spine labels for Phase-6.4 eval rows."""

    events = _event_rows(event_spine_payload)
    event_ids = [str(row.get("event_id", "")) for row in events if row.get("event_id")]
    event_kinds = sorted(
        {str(row.get("event_kind", "")) for row in events if row.get("event_kind")}
    )
    rows: list[WMTransportUnitreeEventSpineJoin] = []
    for proposal in proposals:
        payload = {
            "proposal_id": proposal.proposal_id,
            "contract_id": proposal.contract_id,
            "event_spine_ref": event_spine_ref,
            "event_ids": event_ids,
        }
        rows.append(
            WMTransportUnitreeEventSpineJoin(
                join_id=f"wm_transport_unitree_event_join_{sha256_json(payload)[:16]}",
                proposal_id=proposal.proposal_id,
                contract_id=proposal.contract_id,
                bridge_key=proposal.bridge_key,
                source_wm=proposal.source_wm,
                target_wm=proposal.target_wm,
                join_status=(
                    "joined_unitree_event_spine_ref"
                    if events and event_spine_ref
                    else "awaiting_unitree_event_spine_ref"
                ),
                event_spine_ref=event_spine_ref,
                event_count=len(events),
                event_ids=event_ids,
                event_kinds=event_kinds,
                blockers=list(PHASE64_ADVISORY_RUNTIME_BLOCKERS),
                metadata={
                    "phase": "6.4_transport_unitree_event_spine_join",
                    "label_scope": "lower_wm_event_label_only",
                    "event_spine_run_id": str(event_spine_payload.get("run_id", "")),
                    **_mapping(metadata),
                },
            )
        )
    return rows


def _join_metadata(
    join: Optional[WMTransportUnitreeEventSpineJoin],
) -> Dict[str, Any]:
    if join is None:
        return {}
    return {
        "unitree_event_spine_join_id": join.join_id,
        "unitree_event_spine_join_status": join.join_status,
        "unitree_event_spine_ref": join.event_spine_ref,
        "unitree_event_count": join.event_count,
        "unitree_event_kinds": list(join.event_kinds),
        "unitree_event_lower_wm_label_only": True,
    }


def attach_unitree_event_spine_joins_to_advisory_runtime(
    *,
    report: WMTransportAdvisoryRuntimeReport,
    proposals: Iterable[TransportProposal],
    receipts: Iterable[TransportReceipt],
    eval_reports: Iterable[WMTransportDecomposedEvalReport],
    unitree_event_spine_joins: Iterable[WMTransportUnitreeEventSpineJoin],
) -> tuple[
    WMTransportAdvisoryRuntimeReport,
    list[TransportProposal],
    list[TransportReceipt],
    list[WMTransportDecomposedEvalReport],
]:
    """Attach Unitree event-spine join refs to existing advisory artifacts."""

    joins_by_proposal = {
        row.proposal_id: row for row in list(unitree_event_spine_joins)
    }
    enriched_proposals = [
        TransportProposal(
            **{
                **proposal.__dict__,
                "metadata": {
                    **dict(proposal.metadata),
                    **_join_metadata(joins_by_proposal.get(proposal.proposal_id)),
                },
            }
        )
        for proposal in proposals
    ]
    enriched_receipts = [
        TransportReceipt(
            **{
                **receipt.__dict__,
                "metadata": {
                    **dict(receipt.metadata),
                    **_join_metadata(joins_by_proposal.get(receipt.proposal_id)),
                },
            }
        )
        for receipt in receipts
    ]
    enriched_evals = [
        WMTransportDecomposedEvalReport(
            **{
                **eval_report.__dict__,
                "metadata": {
                    **dict(eval_report.metadata),
                    **_join_metadata(joins_by_proposal.get(eval_report.proposal_id)),
                },
            }
        )
        for eval_report in eval_reports
    ]
    joins = list(joins_by_proposal.values())
    joined = sum(
        1 for row in joins if row.join_status == "joined_unitree_event_spine_ref"
    )
    report = WMTransportAdvisoryRuntimeReport(
        **{
            **report.__dict__,
            "aggregate_counts": {
                **dict(report.aggregate_counts),
                "unitree_event_spine_join_count": float(len(joins)),
                "joined_unitree_event_spine_count": float(joined),
                "unitree_event_count": float(
                    max((row.event_count for row in joins), default=0)
                ),
            },
            "metadata": {
                **dict(report.metadata),
                "unitree_event_spine_joined": bool(joined),
                "unitree_event_spine_join_count": len(joins),
                "unitree_event_spine_lower_wm_label_only": True,
            },
        }
    )
    return report, enriched_proposals, enriched_receipts, enriched_evals


def build_wm_transport_advisory_runtime_with_unitree_event_spine(
    *,
    contracts: Iterable[WMTransportBridgeContract],
    roundtrip_receipts: Iterable[WMTransportRoundTripReceipt],
    neural_manifest: WMTransportNeuralArchitectureManifest,
    trainer_manifest: WMTransportTrainerScaffoldManifest,
    event_spine_payload: Mapping[str, Any],
    event_spine_ref: str,
    shadow_outcome_receipts: Optional[Iterable[Any]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    WMTransportAdvisoryRuntimeReport,
    list[TransportProposal],
    list[TransportInvocation],
    list[TransportReceipt],
    list[WMTransportDecomposedEvalReport],
    list[WMTransportUnitreeEventSpineJoin],
]:
    """Build advisory runtime artifacts and attach Unitree event-spine refs."""

    report, proposals, invocations, receipts, eval_reports = (
        build_wm_transport_advisory_runtime(
            contracts=contracts,
            roundtrip_receipts=roundtrip_receipts,
            neural_manifest=neural_manifest,
            trainer_manifest=trainer_manifest,
            shadow_outcome_receipts=shadow_outcome_receipts,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )
    )
    joins = build_wm_transport_unitree_event_spine_joins(
        proposals=proposals,
        event_spine_payload=event_spine_payload,
        event_spine_ref=event_spine_ref,
        metadata=metadata,
    )
    report, proposals, receipts, eval_reports = (
        attach_unitree_event_spine_joins_to_advisory_runtime(
            report=report,
            proposals=proposals,
            receipts=receipts,
            eval_reports=eval_reports,
            unitree_event_spine_joins=joins,
        )
    )
    return report, proposals, invocations, receipts, eval_reports, joins


def save_wm_transport_advisory_runtime(
    *,
    report_path: str | Path,
    report: WMTransportAdvisoryRuntimeReport,
    proposals_path: str | Path,
    proposals: Iterable[TransportProposal],
    invocations_path: str | Path,
    invocations: Iterable[TransportInvocation],
    receipts_path: str | Path,
    receipts: Iterable[TransportReceipt],
    eval_reports_path: str | Path,
    eval_reports: Iterable[WMTransportDecomposedEvalReport],
    unitree_event_spine_joins_path: str | Path | None = None,
    unitree_event_spine_joins: Iterable[WMTransportUnitreeEventSpineJoin] = (),
) -> None:
    _write_json(report_path, report.to_dict())
    _write_jsonl(proposals_path, [item.to_dict() for item in proposals])
    _write_jsonl(invocations_path, [item.to_dict() for item in invocations])
    _write_jsonl(receipts_path, [item.to_dict() for item in receipts])
    _write_jsonl(eval_reports_path, [item.to_dict() for item in eval_reports])
    if unitree_event_spine_joins_path is not None:
        _write_jsonl(
            unitree_event_spine_joins_path,
            [item.to_dict() for item in unitree_event_spine_joins],
        )


def load_wm_transport_advisory_runtime_report(
    path: str | Path,
) -> WMTransportAdvisoryRuntimeReport:
    return WMTransportAdvisoryRuntimeReport.from_dict(_load_json(path))


def load_wm_transport_proposals(path: str | Path) -> list[TransportProposal]:
    return [TransportProposal.from_dict(row) for row in _load_jsonl(path)]


def load_wm_transport_invocations(path: str | Path) -> list[TransportInvocation]:
    return [TransportInvocation.from_dict(row) for row in _load_jsonl(path)]


def load_wm_transport_receipts(path: str | Path) -> list[TransportReceipt]:
    return [TransportReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_wm_transport_decomposed_eval_reports(
    path: str | Path,
) -> list[WMTransportDecomposedEvalReport]:
    return [WMTransportDecomposedEvalReport.from_dict(row) for row in _load_jsonl(path)]


def load_wm_transport_unitree_event_spine_joins(
    path: str | Path,
) -> list[WMTransportUnitreeEventSpineJoin]:
    return [
        WMTransportUnitreeEventSpineJoin.from_dict(row) for row in _load_jsonl(path)
    ]


__all__ = [
    "DENIED_TRANSPORT_RUNTIME_AUTHORITIES",
    "PHASE64_ADVISORY_RUNTIME_BLOCKERS",
    "WM_TRANSPORT_ADVISORY_RUNTIME_REPORT_VERSION",
    "WM_TRANSPORT_DECOMPOSED_EVAL_REPORT_VERSION",
    "WM_TRANSPORT_INVOCATION_VERSION",
    "WM_TRANSPORT_PROPOSAL_VERSION",
    "WM_TRANSPORT_RECEIPT_VERSION",
    "WM_TRANSPORT_SHADOW_JOIN_SLOT_VERSION",
    "WM_TRANSPORT_UNITREE_EVENT_SPINE_JOIN_VERSION",
    "TransportInvocation",
    "TransportProposal",
    "TransportReceipt",
    "WMTransportAdvisoryRuntimeReport",
    "WMTransportDecomposedEvalReport",
    "WMTransportShadowOutcomeJoinSlot",
    "WMTransportUnitreeEventSpineJoin",
    "attach_unitree_event_spine_joins_to_advisory_runtime",
    "build_wm_transport_advisory_runtime",
    "build_wm_transport_advisory_runtime_with_unitree_event_spine",
    "build_wm_transport_unitree_event_spine_joins",
    "load_wm_transport_advisory_runtime_report",
    "load_wm_transport_decomposed_eval_reports",
    "load_wm_transport_invocations",
    "load_wm_transport_proposals",
    "load_wm_transport_receipts",
    "load_wm_transport_unitree_event_spine_joins",
    "save_wm_transport_advisory_runtime",
]
