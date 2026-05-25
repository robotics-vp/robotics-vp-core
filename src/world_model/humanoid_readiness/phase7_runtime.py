"""Shadow runtime/event-spine wiring for Phase 7 Meta-Regal surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.runtime.event_spine import (
    DecisionLedgerEntry,
    RuntimeEvent,
    decision_ledger_sidecar_payload,
    event_spine_sidecar_payload,
)
from src.world_model.humanoid_readiness.common import (
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase7 import (
    DENIED_PHASE7_AUTHORITIES,
    PHASE7_REMAINING_BLOCKERS,
    Phase7ConflictOverrideReceipt,
    Phase7ControlFieldSlot,
    Phase7MetaRegalControlScaffoldReport,
    load_phase7_conflict_override_receipts,
    load_phase7_control_field_slots,
    load_phase7_meta_regal_control_scaffold_report,
)
from src.world_model.humanoid_readiness.phase7_signal_adapters import (
    Phase7GovernanceNodeSignalReceipt,
)

PHASE7_SHADOW_RUNTIME_WIRING_REPORT_VERSION = (
    "phase7_shadow_runtime_wiring_report_v1"
)
PHASE7_CONTROL_FIELD_RUNTIME_RECEIPT_VERSION = (
    "phase7_control_field_runtime_receipt_v1"
)
PHASE7_CONFLICT_RUNTIME_JOIN_RECEIPT_VERSION = (
    "phase7_conflict_runtime_join_receipt_v1"
)


def _phase7_runtime_denied_gates(
    extra: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    gates = {
        "training_executed": False,
        "weights_written": False,
        "provider_executed": False,
        "hardware_executed": False,
        "unitree_sim_runtime_executed": False,
        "live_policy_control": False,
        "reward_math_mutation": False,
        "promotion_eligible": False,
        "phase7_runtime_authority": False,
        "live_cross_wm_control": False,
        "hard_veto_dispatch": False,
        "lower_wm_replacement": False,
        "scalar_governance_collapse": False,
    }
    gates.update({str(key): bool(value) for key, value in dict(extra or {}).items()})
    return gates


@dataclass(frozen=True)
class Phase7ControlFieldRuntimeReceipt:
    receipt_id: str
    slot_id: str
    field_key: str
    runtime_event_id: str
    decision_id: str
    runtime_packet_id: str | None
    contract_id: str | None
    composition_mode: str
    target_surface: str
    source_node_ids: list[str] = field(default_factory=list)
    node_signal_receipt_ids: list[str] = field(default_factory=list)
    lower_wm_signal_backed: bool = False
    output_authority: str = "shadow_field_only"
    shadow_only: bool = True
    live_dispatch_allowed: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_CONTROL_FIELD_RUNTIME_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "slot_id": self.slot_id,
            "field_key": self.field_key,
            "runtime_event_id": self.runtime_event_id,
            "decision_id": self.decision_id,
            "runtime_packet_id": self.runtime_packet_id,
            "contract_id": self.contract_id,
            "composition_mode": self.composition_mode,
            "target_surface": self.target_surface,
            "source_node_ids": list(self.source_node_ids),
            "node_signal_receipt_ids": list(self.node_signal_receipt_ids),
            "lower_wm_signal_backed": bool(self.lower_wm_signal_backed),
            "output_authority": self.output_authority,
            "shadow_only": bool(self.shadow_only),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7ControlFieldRuntimeReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            slot_id=str(payload.get("slot_id", "")),
            field_key=str(payload.get("field_key", "")),
            runtime_event_id=str(payload.get("runtime_event_id", "")),
            decision_id=str(payload.get("decision_id", "")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            contract_id=payload.get("contract_id"),
            composition_mode=str(payload.get("composition_mode", "")),
            target_surface=str(payload.get("target_surface", "")),
            source_node_ids=strings(payload.get("source_node_ids")),
            node_signal_receipt_ids=strings(payload.get("node_signal_receipt_ids")),
            lower_wm_signal_backed=bool(
                payload.get("lower_wm_signal_backed", False)
            ),
            output_authority=str(payload.get("output_authority", "shadow_field_only")),
            shadow_only=bool(payload.get("shadow_only", True)),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(
                payload.get("version", PHASE7_CONTROL_FIELD_RUNTIME_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase7ConflictRuntimeJoinReceipt:
    receipt_id: str
    conflict_receipt_id: str
    conflict_key: str
    runtime_event_id: str
    decision_id: str
    runtime_packet_id: str | None
    contract_id: str | None
    source_node_ids: list[str]
    related_control_field_event_ids: list[str] = field(default_factory=list)
    node_signal_receipt_ids: list[str] = field(default_factory=list)
    lower_wm_signal_backed: bool = False
    composition_mode: str = ""
    override_policy: str = ""
    severity_prior: float = 0.0
    shadow_only: bool = True
    hard_veto_dispatch: bool = False
    live_dispatch_allowed: bool = False
    promotion_eligible: bool = False
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_CONFLICT_RUNTIME_JOIN_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "conflict_receipt_id": self.conflict_receipt_id,
            "conflict_key": self.conflict_key,
            "runtime_event_id": self.runtime_event_id,
            "decision_id": self.decision_id,
            "runtime_packet_id": self.runtime_packet_id,
            "contract_id": self.contract_id,
            "source_node_ids": list(self.source_node_ids),
            "related_control_field_event_ids": list(
                self.related_control_field_event_ids
            ),
            "node_signal_receipt_ids": list(self.node_signal_receipt_ids),
            "lower_wm_signal_backed": bool(self.lower_wm_signal_backed),
            "composition_mode": self.composition_mode,
            "override_policy": self.override_policy,
            "severity_prior": float(self.severity_prior),
            "shadow_only": bool(self.shadow_only),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7ConflictRuntimeJoinReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            conflict_receipt_id=str(payload.get("conflict_receipt_id", "")),
            conflict_key=str(payload.get("conflict_key", "")),
            runtime_event_id=str(payload.get("runtime_event_id", "")),
            decision_id=str(payload.get("decision_id", "")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            contract_id=payload.get("contract_id"),
            source_node_ids=strings(payload.get("source_node_ids")),
            related_control_field_event_ids=strings(
                payload.get("related_control_field_event_ids")
            ),
            node_signal_receipt_ids=strings(payload.get("node_signal_receipt_ids")),
            lower_wm_signal_backed=bool(
                payload.get("lower_wm_signal_backed", False)
            ),
            composition_mode=str(payload.get("composition_mode", "")),
            override_policy=str(payload.get("override_policy", "")),
            severity_prior=float(payload.get("severity_prior", 0.0) or 0.0),
            shadow_only=bool(payload.get("shadow_only", True)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(
                payload.get("version", PHASE7_CONFLICT_RUNTIME_JOIN_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase7ShadowRuntimeWiringReport:
    report_id: str
    phase7_scaffold_report_id: str
    run_id: str
    episode_id: str
    status: str
    control_field_runtime_receipt_count: int
    conflict_runtime_join_receipt_count: int
    runtime_event_count: int
    decision_ledger_entry_count: int
    shadow_event_spine_wiring_executed: bool
    decision_ledger_wiring_executed: bool
    local_shadow_runtime_wiring_complete: bool
    node_signal_receipt_count: int = 0
    lower_wm_signal_backed: bool = False
    phase7_authority_granted: bool = False
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_phase7_runtime_denied_gates)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_SHADOW_RUNTIME_WIRING_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase7_scaffold_report_id": self.phase7_scaffold_report_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "status": self.status,
            "control_field_runtime_receipt_count": int(
                self.control_field_runtime_receipt_count
            ),
            "conflict_runtime_join_receipt_count": int(
                self.conflict_runtime_join_receipt_count
            ),
            "node_signal_receipt_count": int(self.node_signal_receipt_count),
            "lower_wm_signal_backed": bool(self.lower_wm_signal_backed),
            "runtime_event_count": int(self.runtime_event_count),
            "decision_ledger_entry_count": int(self.decision_ledger_entry_count),
            "shadow_event_spine_wiring_executed": bool(
                self.shadow_event_spine_wiring_executed
            ),
            "decision_ledger_wiring_executed": bool(
                self.decision_ledger_wiring_executed
            ),
            "local_shadow_runtime_wiring_complete": bool(
                self.local_shadow_runtime_wiring_complete
            ),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": _phase7_runtime_denied_gates(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7ShadowRuntimeWiringReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase7_scaffold_report_id=str(
                payload.get("phase7_scaffold_report_id", "")
            ),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            status=str(payload.get("status", "blocked")),
            control_field_runtime_receipt_count=int(
                payload.get("control_field_runtime_receipt_count", 0) or 0
            ),
            conflict_runtime_join_receipt_count=int(
                payload.get("conflict_runtime_join_receipt_count", 0) or 0
            ),
            node_signal_receipt_count=int(
                payload.get("node_signal_receipt_count", 0) or 0
            ),
            lower_wm_signal_backed=bool(
                payload.get("lower_wm_signal_backed", False)
            ),
            runtime_event_count=int(payload.get("runtime_event_count", 0) or 0),
            decision_ledger_entry_count=int(
                payload.get("decision_ledger_entry_count", 0) or 0
            ),
            shadow_event_spine_wiring_executed=bool(
                payload.get("shadow_event_spine_wiring_executed", False)
            ),
            decision_ledger_wiring_executed=bool(
                payload.get("decision_ledger_wiring_executed", False)
            ),
            local_shadow_runtime_wiring_complete=bool(
                payload.get("local_shadow_runtime_wiring_complete", False)
            ),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates=_phase7_runtime_denied_gates(payload.get("denied_gates")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE7_SHADOW_RUNTIME_WIRING_REPORT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase7RuntimeScaffoldInputs:
    report: Phase7MetaRegalControlScaffoldReport
    control_fields: list[Phase7ControlFieldSlot]
    conflict_receipts: list[Phase7ConflictOverrideReceipt]


def load_phase7_runtime_scaffold_inputs(
    scaffold_dir: str | Path,
) -> Phase7RuntimeScaffoldInputs:
    root = Path(scaffold_dir)
    return Phase7RuntimeScaffoldInputs(
        report=load_phase7_meta_regal_control_scaffold_report(
            root / "phase7_meta_regal_control_scaffold_report_v1.json"
        ),
        control_fields=load_phase7_control_field_slots(
            root / "phase7_control_field_slots_v1.jsonl"
        ),
        conflict_receipts=load_phase7_conflict_override_receipts(
            root / "phase7_conflict_override_receipts_v1.jsonl"
        ),
    )


def build_phase7_shadow_runtime_wiring(
    *,
    phase7_report: Phase7MetaRegalControlScaffoldReport,
    control_fields: Sequence[Phase7ControlFieldSlot],
    conflict_receipts: Sequence[Phase7ConflictOverrideReceipt],
    run_id: str,
    episode_id: str,
    timestamp: str,
    runtime_packet_id: str | None,
    contract_id: str | None,
    node_signal_receipts: Sequence[Phase7GovernanceNodeSignalReceipt] = (),
    artifact_refs: Mapping[str, Any] | None = None,
    event_sequence_start: int = 0,
    decision_sequence_start: int = 0,
) -> tuple[
    Phase7ShadowRuntimeWiringReport,
    list[Phase7ControlFieldRuntimeReceipt],
    list[Phase7ConflictRuntimeJoinReceipt],
    list[RuntimeEvent],
    list[DecisionLedgerEntry],
    int,
    int,
]:
    denied = list(DENIED_PHASE7_AUTHORITIES)
    refs = mapping(artifact_refs)
    events: list[RuntimeEvent] = []
    decisions: list[DecisionLedgerEntry] = []
    field_receipts: list[Phase7ControlFieldRuntimeReceipt] = []
    conflict_join_receipts: list[Phase7ConflictRuntimeJoinReceipt] = []
    event_sequence_idx = int(event_sequence_start)
    decision_sequence_idx = int(decision_sequence_start)
    signal_ids_by_surface_id = _signal_ids_by_surface_id(node_signal_receipts)

    for field_slot in control_fields:
        node_signal_ids = _node_signal_receipt_ids(
            source_node_ids=field_slot.source_node_ids,
            signal_ids_by_surface_id=signal_ids_by_surface_id,
        )
        lower_wm_signal_backed = bool(node_signal_ids)
        event = RuntimeEvent.from_components(
            run_id=run_id,
            episode_id=episode_id,
            timestamp=timestamp,
            event_kind="phase7_control_field_shadow_emitted",
            sequence_idx=event_sequence_idx,
            scope={
                "scope_kind": "episode",
                "phase": "phase7_meta_regal_control_wm",
                "field_key": field_slot.field_key,
                "target_surface": field_slot.target_surface,
            },
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            receipt_label_refs=[field_slot.slot_id],
            artifact_refs=refs,
            provenance={
                "advisor": {
                    "component": "phase7_meta_regal_shadow_runtime",
                    "authority": "shadow_control_field_only",
                }
            },
            metadata={
                "slot_id": field_slot.slot_id,
                "composition_mode": field_slot.composition_mode,
                "source_node_ids": list(field_slot.source_node_ids),
                "node_signal_receipt_ids": node_signal_ids,
                "lower_wm_signal_backed": lower_wm_signal_backed,
                "output_authority": field_slot.output_authority,
                "shadow_only": True,
                "live_dispatch_allowed": False,
                "hard_veto_dispatch": False,
                "phase7_authority_granted": False,
                "reward_math_mutation": False,
                "promotion_eligible": False,
            },
        )
        events.append(event)
        event_sequence_idx += 1
        decision = DecisionLedgerEntry.from_components(
            run_id=run_id,
            episode_id=episode_id,
            timestamp=timestamp,
            decision_kind="phase7_control_field_shadow_recorded",
            outcome="shadow_only_no_dispatch",
            sequence_idx=decision_sequence_idx,
            scope=event.scope,
            reasons=[
                "phase7_control_field_connected_to_event_spine",
                "live_dispatch_authority_denied",
            ],
            source_event_ids=[event.event_id],
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            receipt_label_refs=[field_slot.slot_id],
            artifact_refs=refs,
            provenance=event.provenance,
            metadata={
                "field_key": field_slot.field_key,
                "composition_mode": field_slot.composition_mode,
                "node_signal_receipt_ids": node_signal_ids,
                "lower_wm_signal_backed": lower_wm_signal_backed,
                "shadow_only": True,
                "live_dispatch_allowed": False,
            },
        )
        decisions.append(decision)
        decision_sequence_idx += 1
        receipt_payload = {
            "slot_id": field_slot.slot_id,
            "event_id": event.event_id,
            "decision_id": decision.decision_id,
            "run_id": run_id,
            "episode_id": episode_id,
        }
        field_receipts.append(
            Phase7ControlFieldRuntimeReceipt(
                receipt_id=stable_id("phase7_field_runtime", receipt_payload),
                slot_id=field_slot.slot_id,
                field_key=field_slot.field_key,
                runtime_event_id=event.event_id,
                decision_id=decision.decision_id,
                runtime_packet_id=runtime_packet_id,
                contract_id=contract_id,
                composition_mode=field_slot.composition_mode,
                target_surface=field_slot.target_surface,
                source_node_ids=list(field_slot.source_node_ids),
                node_signal_receipt_ids=node_signal_ids,
                lower_wm_signal_backed=lower_wm_signal_backed,
                output_authority=field_slot.output_authority,
                denied_authority=denied,
            )
        )

    for conflict_receipt in conflict_receipts:
        node_signal_ids = _node_signal_receipt_ids(
            source_node_ids=conflict_receipt.source_node_ids,
            signal_ids_by_surface_id=signal_ids_by_surface_id,
        )
        lower_wm_signal_backed = bool(node_signal_ids)
        related_field_event_ids = _related_control_field_event_ids(
            conflict_receipt=conflict_receipt,
            field_receipts=field_receipts,
            control_fields=control_fields,
        )
        event = RuntimeEvent.from_components(
            run_id=run_id,
            episode_id=episode_id,
            timestamp=timestamp,
            event_kind="phase7_conflict_override_shadow_joined",
            sequence_idx=event_sequence_idx,
            scope={
                "scope_kind": "episode",
                "phase": "phase7_meta_regal_control_wm",
                "conflict_key": conflict_receipt.conflict_key,
                "composition_mode": conflict_receipt.composition_mode,
            },
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            receipt_label_refs=[
                conflict_receipt.receipt_id,
                *related_field_event_ids,
            ],
            artifact_refs=refs,
            provenance={
                "critic": {
                    "component": "phase7_meta_regal_shadow_runtime",
                    "authority": "shadow_conflict_join_only",
                }
            },
            metadata={
                "conflict_receipt_id": conflict_receipt.receipt_id,
                "source_node_ids": list(conflict_receipt.source_node_ids),
                "node_signal_receipt_ids": node_signal_ids,
                "lower_wm_signal_backed": lower_wm_signal_backed,
                "override_policy": conflict_receipt.override_policy,
                "severity_prior": float(conflict_receipt.severity_prior),
                "related_control_field_event_ids": related_field_event_ids,
                "shadow_only": True,
                "live_dispatch_allowed": False,
                "hard_veto_dispatch": False,
                "phase7_authority_granted": False,
                "reward_math_mutation": False,
                "promotion_eligible": False,
            },
        )
        events.append(event)
        event_sequence_idx += 1
        decision = DecisionLedgerEntry.from_components(
            run_id=run_id,
            episode_id=episode_id,
            timestamp=timestamp,
            decision_kind="phase7_conflict_override_shadow_recorded",
            outcome="shadow_joined_no_override_dispatch",
            sequence_idx=decision_sequence_idx,
            scope=event.scope,
            reasons=[
                "phase7_conflict_receipt_joined_to_event_spine",
                "hard_veto_dispatch_denied",
                "live_dispatch_authority_denied",
            ],
            source_event_ids=[event.event_id, *related_field_event_ids],
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            receipt_label_refs=[conflict_receipt.receipt_id],
            artifact_refs=refs,
            provenance=event.provenance,
            metadata={
                "conflict_key": conflict_receipt.conflict_key,
                "composition_mode": conflict_receipt.composition_mode,
                "override_policy": conflict_receipt.override_policy,
                "node_signal_receipt_ids": node_signal_ids,
                "lower_wm_signal_backed": lower_wm_signal_backed,
                "shadow_only": True,
                "hard_veto_dispatch": False,
            },
        )
        decisions.append(decision)
        decision_sequence_idx += 1
        receipt_payload = {
            "conflict_receipt_id": conflict_receipt.receipt_id,
            "event_id": event.event_id,
            "decision_id": decision.decision_id,
            "run_id": run_id,
            "episode_id": episode_id,
        }
        conflict_join_receipts.append(
            Phase7ConflictRuntimeJoinReceipt(
                receipt_id=stable_id("phase7_conflict_runtime", receipt_payload),
                conflict_receipt_id=conflict_receipt.receipt_id,
                conflict_key=conflict_receipt.conflict_key,
                runtime_event_id=event.event_id,
                decision_id=decision.decision_id,
                runtime_packet_id=runtime_packet_id,
                contract_id=contract_id,
                source_node_ids=list(conflict_receipt.source_node_ids),
                related_control_field_event_ids=related_field_event_ids,
                node_signal_receipt_ids=node_signal_ids,
                lower_wm_signal_backed=lower_wm_signal_backed,
                composition_mode=conflict_receipt.composition_mode,
                override_policy=conflict_receipt.override_policy,
                severity_prior=float(conflict_receipt.severity_prior),
                denied_authority=denied,
            )
        )

    input_ready = (
        phase7_report.status == "ok"
        and phase7_report.local_phase7_scaffold_complete
        and phase7_report.ready_for_runtime_wiring
        and not phase7_report.phase7_authority_granted
    )
    signal_backing_required = bool(node_signal_receipts)
    signal_backing_complete = not signal_backing_required or (
        all(receipt.lower_wm_signal_backed for receipt in field_receipts)
        and all(receipt.lower_wm_signal_backed for receipt in conflict_join_receipts)
    )
    complete = (
        input_ready
        and len(field_receipts) == len(control_fields)
        and len(conflict_join_receipts) == len(conflict_receipts)
        and len(events) == len(decisions)
        and len(events) == len(control_fields) + len(conflict_receipts)
        and signal_backing_complete
    )
    report_payload = {
        "phase7_scaffold_report_id": phase7_report.report_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "control_field_runtime_receipt_count": len(field_receipts),
        "conflict_runtime_join_receipt_count": len(conflict_join_receipts),
        "node_signal_receipt_count": len(node_signal_receipts),
        "runtime_event_count": len(events),
        "artifact_refs": refs,
    }
    report = Phase7ShadowRuntimeWiringReport(
        report_id=stable_id("phase7_shadow_runtime", report_payload),
        phase7_scaffold_report_id=phase7_report.report_id,
        run_id=run_id,
        episode_id=episode_id,
        status="ok" if complete else "blocked",
        control_field_runtime_receipt_count=len(field_receipts),
        conflict_runtime_join_receipt_count=len(conflict_join_receipts),
        runtime_event_count=len(events),
        decision_ledger_entry_count=len(decisions),
        shadow_event_spine_wiring_executed=complete,
        decision_ledger_wiring_executed=complete,
        local_shadow_runtime_wiring_complete=complete,
        node_signal_receipt_count=len(node_signal_receipts),
        lower_wm_signal_backed=signal_backing_required and signal_backing_complete,
        denied_gates=_phase7_runtime_denied_gates(),
        remaining_blockers=list(PHASE7_REMAINING_BLOCKERS),
        artifact_refs=refs,
    )
    return (
        report,
        field_receipts,
        conflict_join_receipts,
        events,
        decisions,
        event_sequence_idx,
        decision_sequence_idx,
    )


def _related_control_field_event_ids(
    *,
    conflict_receipt: Phase7ConflictOverrideReceipt,
    field_receipts: Sequence[Phase7ControlFieldRuntimeReceipt],
    control_fields: Sequence[Phase7ControlFieldSlot],
) -> list[str]:
    slot_sources = {field.slot_id: set(field.source_node_ids) for field in control_fields}
    conflict_sources = set(conflict_receipt.source_node_ids)
    related: list[str] = []
    for field_receipt in field_receipts:
        if conflict_sources.intersection(slot_sources.get(field_receipt.slot_id, set())):
            related.append(field_receipt.runtime_event_id)
    return related


def _signal_ids_by_surface_id(
    receipts: Sequence[Phase7GovernanceNodeSignalReceipt],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for receipt in receipts:
        if not receipt.lower_wm_receipt_backed:
            continue
        output.setdefault(receipt.surface_id, []).append(receipt.signal_id)
    return output


def _node_signal_receipt_ids(
    *,
    source_node_ids: Sequence[str],
    signal_ids_by_surface_id: Mapping[str, Sequence[str]],
) -> list[str]:
    signal_ids: list[str] = []
    for source_node_id in source_node_ids:
        signal_ids.extend(signal_ids_by_surface_id.get(source_node_id, ()))
    return sorted(dict.fromkeys(signal_ids))


def save_phase7_shadow_runtime_wiring(
    output_dir: str | Path,
    report: Phase7ShadowRuntimeWiringReport,
    field_receipts: Sequence[Phase7ControlFieldRuntimeReceipt],
    conflict_join_receipts: Sequence[Phase7ConflictRuntimeJoinReceipt],
    runtime_events: Sequence[RuntimeEvent],
    decision_entries: Sequence[DecisionLedgerEntry],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase7_shadow_runtime_wiring_report_v1.json",
        "control_field_runtime_receipts_path": output
        / "phase7_control_field_runtime_receipts_v1.jsonl",
        "conflict_runtime_join_receipts_path": output
        / "phase7_conflict_runtime_join_receipts_v1.jsonl",
        "runtime_event_spine_path": output / "phase7_runtime_event_spine_v1.json",
        "runtime_decision_ledger_path": output
        / "phase7_runtime_decision_ledger_v1.json",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(
        paths["control_field_runtime_receipts_path"],
        [receipt.to_dict() for receipt in field_receipts],
    )
    write_jsonl(
        paths["conflict_runtime_join_receipts_path"],
        [receipt.to_dict() for receipt in conflict_join_receipts],
    )
    write_json(
        paths["runtime_event_spine_path"],
        event_spine_sidecar_payload(run_id=report.run_id, events=runtime_events),
    )
    write_json(
        paths["runtime_decision_ledger_path"],
        decision_ledger_sidecar_payload(
            run_id=report.run_id,
            decisions=decision_entries,
        ),
    )
    return {key: str(value) for key, value in paths.items()}


def load_phase7_shadow_runtime_wiring_report(
    path: str | Path,
) -> Phase7ShadowRuntimeWiringReport:
    return Phase7ShadowRuntimeWiringReport.from_dict(load_json(path))


def load_phase7_control_field_runtime_receipts(
    path: str | Path,
) -> list[Phase7ControlFieldRuntimeReceipt]:
    return [Phase7ControlFieldRuntimeReceipt.from_dict(row) for row in load_jsonl(path)]


def load_phase7_conflict_runtime_join_receipts(
    path: str | Path,
) -> list[Phase7ConflictRuntimeJoinReceipt]:
    return [Phase7ConflictRuntimeJoinReceipt.from_dict(row) for row in load_jsonl(path)]
