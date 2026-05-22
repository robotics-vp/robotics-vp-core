"""Round-trip and receiver-actionability receipts for Phase-6 transport."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import WMTransportBridgeContract
from src.world_model.transport.topology_metrics import (
    WMTransportTopologyMetrics,
    compute_wm_transport_topology_metrics,
)
from src.world_model.transport.uncertainty import (
    WMTransportUncertaintyCalibration,
    calibrate_wm_transport_uncertainty,
)
from src.world_model.transport.wm_transformers import PerWMTransportTransformerRegistry

WM_TRANSPORT_ROUNDTRIP_RECEIPT_VERSION = "wm_transport_roundtrip_receipt_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class WMTransportRoundTripReceipt:
    """Local receipt for contract/receiver round-trip shape compatibility."""

    receipt_id: str
    contract_id: str
    bridge_key: str
    source_exporter_id: str
    target_receiver_id: str
    topology_metrics: WMTransportTopologyMetrics
    uncertainty_calibration: WMTransportUncertaintyCalibration
    source_reconstruction_score: float
    target_receiver_actionability_score: float
    roundtrip_consistency_score: float
    aggregate_score: float
    authority_class: str = "transport_roundtrip_receipt_only"
    training_executed: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_ROUNDTRIP_RECEIPT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_exporter_id": self.source_exporter_id,
            "target_receiver_id": self.target_receiver_id,
            "topology_metrics": self.topology_metrics.to_dict(),
            "uncertainty_calibration": self.uncertainty_calibration.to_dict(),
            "source_reconstruction_score": float(self.source_reconstruction_score),
            "target_receiver_actionability_score": float(
                self.target_receiver_actionability_score
            ),
            "roundtrip_consistency_score": float(self.roundtrip_consistency_score),
            "aggregate_score": float(self.aggregate_score),
            "authority_class": self.authority_class,
            "training_executed": bool(self.training_executed),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportRoundTripReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_exporter_id=str(payload.get("source_exporter_id", "")),
            target_receiver_id=str(payload.get("target_receiver_id", "")),
            topology_metrics=WMTransportTopologyMetrics.from_dict(
                dict(payload.get("topology_metrics", {}) or {})
            ),
            uncertainty_calibration=WMTransportUncertaintyCalibration.from_dict(
                dict(payload.get("uncertainty_calibration", {}) or {})
            ),
            source_reconstruction_score=float(
                payload.get("source_reconstruction_score", 0.0)
            ),
            target_receiver_actionability_score=float(
                payload.get("target_receiver_actionability_score", 0.0)
            ),
            roundtrip_consistency_score=float(
                payload.get("roundtrip_consistency_score", 0.0)
            ),
            aggregate_score=float(payload.get("aggregate_score", 0.0)),
            authority_class=str(
                payload.get("authority_class", "transport_roundtrip_receipt_only")
            ),
            training_executed=bool(payload.get("training_executed", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_ROUNDTRIP_RECEIPT_VERSION)),
        )


def build_wm_transport_roundtrip_receipts(
    *,
    contracts: Iterable[WMTransportBridgeContract],
    transformer_registry: PerWMTransportTransformerRegistry,
) -> list[WMTransportRoundTripReceipt]:
    transformers = transformer_registry.by_id()
    receipts: list[WMTransportRoundTripReceipt] = []
    for contract in contracts:
        source_exporter_id = contract.source_endpoint.transformer_id
        target_receiver_id = contract.target_endpoint.transformer_id
        exporter_present = source_exporter_id in transformers
        receiver_present = target_receiver_id in transformers
        topology = compute_wm_transport_topology_metrics(contract)
        uncertainty = calibrate_wm_transport_uncertainty(contract)
        source_score = (
            1.0 if contract.source_endpoint.state_ref and exporter_present else 0.0
        )
        receiver_score = (
            1.0 if contract.target_endpoint.state_ref and receiver_present else 0.0
        )
        roundtrip_score = (
            0.35 * source_score
            + 0.35 * receiver_score
            + 0.2 * topology.aggregate_score
            + 0.1 * uncertainty.calibration_score
        )
        aggregate = max(0.0, min(1.0, roundtrip_score))
        payload = {
            "contract_id": contract.contract_id,
            "source_exporter_id": source_exporter_id,
            "target_receiver_id": target_receiver_id,
            "aggregate_score": aggregate,
        }
        receipts.append(
            WMTransportRoundTripReceipt(
                receipt_id=f"wm_transport_roundtrip_{sha256_json(payload)[:16]}",
                contract_id=contract.contract_id,
                bridge_key=contract.bridge_key,
                source_exporter_id=source_exporter_id,
                target_receiver_id=target_receiver_id,
                topology_metrics=topology,
                uncertainty_calibration=uncertainty,
                source_reconstruction_score=source_score,
                target_receiver_actionability_score=receiver_score,
                roundtrip_consistency_score=aggregate,
                aggregate_score=aggregate,
                blockers=[
                    "gpu_transport_training_not_run",
                    "receiver_actionability_benchmark_missing",
                    "provider_or_hardware_roundtrip_not_run",
                    "promotion_grade_transport_benchmark_missing",
                ],
                metadata={
                    "phase": "6.2_roundtrip_topology_uncertainty_receipt",
                    "bridge_only_training_claim": False,
                    "receiver_training_claim": False,
                },
            )
        )
    return receipts


def save_wm_transport_roundtrip_receipts(
    path: str | Path, receipts: Iterable[WMTransportRoundTripReceipt]
) -> None:
    _write_jsonl(path, [receipt.to_dict() for receipt in receipts])


def load_wm_transport_roundtrip_receipts(
    path: str | Path,
) -> list[WMTransportRoundTripReceipt]:
    return [WMTransportRoundTripReceipt.from_dict(row) for row in _load_jsonl(path)]


__all__ = [
    "WM_TRANSPORT_ROUNDTRIP_RECEIPT_VERSION",
    "WMTransportRoundTripReceipt",
    "build_wm_transport_roundtrip_receipts",
    "load_wm_transport_roundtrip_receipts",
    "save_wm_transport_roundtrip_receipts",
]
