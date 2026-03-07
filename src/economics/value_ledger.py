"""Sparse, auditable shadow value ledger backed by deterministic JSONL."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.constraints.constraint_set import ConstraintSet
from src.economics.econ_tensor import EconTensor
from src.economics.pricing_sentinel import PricingTick
from src.objectives.runtime_builder import summarize_objective_tensor
from src.objectives.tensor import ObjectiveTensor
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


@dataclass(frozen=True)
class ValueLedgerReceipt:
    """Single sparse economic receipt for a meaningful shadow event."""

    ledger_event_id: str
    event_type: str
    run_id: str
    episode_id: str
    objective_profile_id: str
    objective_tensor_summary: Dict[str, Any]
    econ_tensor_summary: Dict[str, Any]
    pricing_tick_summary: Dict[str, Any]
    constraint_summary: Dict[str, Any]
    regal_decision_summary: Dict[str, Any]
    datapack_id: Optional[str]
    source_domain: str
    timestamp: str
    provenance_hashes: Dict[str, str]
    schema_ids: Dict[str, str]
    receipt_hash: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_event_id": self.ledger_event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "objective_profile_id": self.objective_profile_id,
            "objective_tensor_summary": dict(self.objective_tensor_summary),
            "econ_tensor_summary": dict(self.econ_tensor_summary),
            "pricing_tick_summary": dict(self.pricing_tick_summary),
            "constraint_summary": dict(self.constraint_summary),
            "regal_decision_summary": dict(self.regal_decision_summary),
            "datapack_id": self.datapack_id,
            "source_domain": self.source_domain,
            "timestamp": self.timestamp,
            "provenance_hashes": dict(self.provenance_hashes),
            "schema_ids": dict(self.schema_ids),
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueLedgerReceipt":
        return cls(
            ledger_event_id=str(payload.get("ledger_event_id", "")),
            event_type=str(payload.get("event_type", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            objective_profile_id=str(payload.get("objective_profile_id", "")),
            objective_tensor_summary=dict(payload.get("objective_tensor_summary", {}) or {}),
            econ_tensor_summary=dict(payload.get("econ_tensor_summary", {}) or {}),
            pricing_tick_summary=dict(payload.get("pricing_tick_summary", {}) or {}),
            constraint_summary=dict(payload.get("constraint_summary", {}) or {}),
            regal_decision_summary=dict(payload.get("regal_decision_summary", {}) or {}),
            datapack_id=payload.get("datapack_id"),
            source_domain=str(payload.get("source_domain", "")),
            timestamp=str(payload.get("timestamp", "")),
            provenance_hashes={str(k): str(v) for k, v in dict(payload.get("provenance_hashes", {}) or {}).items()},
            schema_ids={str(k): str(v) for k, v in dict(payload.get("schema_ids", {}) or {}).items()},
            receipt_hash=str(payload.get("receipt_hash", "")),
        )


class ValueLedger:
    """Append-only JSONL ledger for sparse shadow accounting receipts."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, receipt: ValueLedgerReceipt) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")

    def load(self) -> List[ValueLedgerReceipt]:
        if not self.ledger_path.exists():
            return []
        receipts: List[ValueLedgerReceipt] = []
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    receipts.append(ValueLedgerReceipt.from_dict(json.loads(line)))
        return receipts

    def build_receipt(
        self,
        *,
        event_type: str,
        run_id: str,
        episode_id: str,
        objective_profile_id: str,
        objective_tensor: ObjectiveTensor | Mapping[str, Any],
        econ_tensor: EconTensor | Mapping[str, Any],
        pricing_tick: PricingTick | Mapping[str, Any],
        constraint_set: ConstraintSet | Mapping[str, Any],
        regal_decision_summary: Mapping[str, Any],
        datapack_id: Optional[str],
        source_domain: str,
        timestamp: str,
    ) -> ValueLedgerReceipt:
        objective_summary = summarize_objective_tensor(objective_tensor) if isinstance(objective_tensor, ObjectiveTensor) else dict(objective_tensor)
        econ_summary = summarize_econ_tensor(econ_tensor)
        pricing_summary = pricing_tick.to_dict() if isinstance(pricing_tick, PricingTick) else dict(pricing_tick)
        constraint_summary = (
            constraint_set.summary() if isinstance(constraint_set, ConstraintSet) else dict(constraint_set)
        )
        regal_summary = dict(regal_decision_summary or {})
        schema_ids = {
            "objective_tensor": str(objective_summary.get("schema_id", "")),
            "econ_tensor": str(econ_summary.get("schema_id", "")),
            "constraint_set": str(constraint_summary.get("version", "")),
            "pricing_policy": str(pricing_summary.get("metadata", {}).get("policy_id", "")),
        }
        provenance_hashes = {
            "objective_tensor": str(objective_summary.get("schema_hash", sha256_json(objective_summary))),
            "econ_tensor": str(econ_summary.get("schema_hash", sha256_json(econ_summary))),
            "pricing_tick": str(pricing_summary.get("metadata", {}).get("policy_hash", sha256_json(pricing_summary))),
            "regal": sha256_json(regal_summary),
        }
        core = {
            "event_type": event_type,
            "run_id": run_id,
            "episode_id": episode_id,
            "objective_profile_id": objective_profile_id,
            "objective_tensor_summary": to_json_safe(objective_summary),
            "econ_tensor_summary": to_json_safe(econ_summary),
            "pricing_tick_summary": to_json_safe(pricing_summary),
            "constraint_summary": to_json_safe(constraint_summary),
            "regal_decision_summary": to_json_safe(regal_summary),
            "datapack_id": datapack_id,
            "source_domain": source_domain,
            "timestamp": timestamp,
            "provenance_hashes": provenance_hashes,
            "schema_ids": schema_ids,
        }
        receipt_hash = sha256_json(core)
        return ValueLedgerReceipt(
            ledger_event_id=f"ledger_{receipt_hash[:16]}",
            event_type=event_type,
            run_id=run_id,
            episode_id=episode_id,
            objective_profile_id=objective_profile_id,
            objective_tensor_summary=objective_summary,
            econ_tensor_summary=econ_summary,
            pricing_tick_summary=pricing_summary,
            constraint_summary=constraint_summary,
            regal_decision_summary=regal_summary,
            datapack_id=datapack_id,
            source_domain=source_domain,
            timestamp=timestamp,
            provenance_hashes=provenance_hashes,
            schema_ids=schema_ids,
            receipt_hash=receipt_hash,
        )


def summarize_econ_tensor(econ_tensor: EconTensor | Mapping[str, Any]) -> Dict[str, Any]:
    """Create a compact, stable econ summary for ledgers and reports."""

    if isinstance(econ_tensor, EconTensor):
        axes = {
            axis: float(econ_tensor.values[index])
            for index, axis in enumerate(econ_tensor.schema.axes)
        }
        schema_id = econ_tensor.schema.schema_id
    else:
        axes = {
            str(key): float(value)
            for key, value in dict(econ_tensor).items()
            if isinstance(value, (int, float))
        }
        if "axes" in econ_tensor and "values" in econ_tensor:
            axes = {
                str(axis): float(econ_tensor["values"][index])
                for index, axis in enumerate(econ_tensor["axes"])
            }
        schema_id = str(econ_tensor.get("schema_id", "")) if isinstance(econ_tensor, Mapping) else ""
    summary = {
        "schema_id": schema_id,
        "axes": axes,
    }
    summary["schema_hash"] = sha256_json(summary)
    return summary
