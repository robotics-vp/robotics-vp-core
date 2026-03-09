"""Canonical runtime packet scaffolding for economic-world-model readiness."""

from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.event_spine import (
    DecisionLedgerEntry,
    RuntimeEvent,
    decision_ledger_sidecar_payload,
    event_spine_sidecar_payload,
)
from src.runtime.observation_adapter_v2 import ObservationAdapterV2
from src.runtime.packets import (
    ContractPacket,
    RuntimePacket,
    SchemaRef,
    build_contract_packet,
    runtime_packet_from_record,
    runtime_packet_sidecar_payload,
)

__all__ = [
    "ActionAdapterV2",
    "DecisionLedgerEntry",
    "ContractPacket",
    "ObservationAdapterV2",
    "RuntimeEvent",
    "RuntimePacket",
    "SchemaRef",
    "build_contract_packet",
    "decision_ledger_sidecar_payload",
    "event_spine_sidecar_payload",
    "runtime_packet_from_record",
    "runtime_packet_sidecar_payload",
]
