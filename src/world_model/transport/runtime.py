"""Advisory runtime report for Phase-6.0-6.2 transport scaffolding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import WMTransportContractPack
from src.world_model.transport.training_rows import WMTransportTrainingManifest
from src.world_model.transport.wm_transformers import PerWMTransportTransformerRegistry

WM_TRANSPORT_PHASE6_SCAFFOLD_REPORT_VERSION = "wm_transport_phase6_scaffold_report_v1"


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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WMTransportPhase6ScaffoldReport:
    """Single top-level report for the local Phase-6.0-6.2 scaffold pass."""

    report_id: str
    contract_pack_id: str
    transformer_registry_id: str
    training_manifest_id: str
    contract_count: int
    transformer_count: int
    roundtrip_receipt_count: int
    training_row_count: int
    status: str
    authority_class: str = "phase6_transport_scaffold_report_only"
    ready_for_phase6_3_neural_scaffold: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_PHASE6_SCAFFOLD_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "contract_pack_id": self.contract_pack_id,
            "transformer_registry_id": self.transformer_registry_id,
            "training_manifest_id": self.training_manifest_id,
            "contract_count": int(self.contract_count),
            "transformer_count": int(self.transformer_count),
            "roundtrip_receipt_count": int(self.roundtrip_receipt_count),
            "training_row_count": int(self.training_row_count),
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_phase6_3_neural_scaffold": bool(
                self.ready_for_phase6_3_neural_scaffold
            ),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportPhase6ScaffoldReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            contract_pack_id=str(payload.get("contract_pack_id", "")),
            transformer_registry_id=str(payload.get("transformer_registry_id", "")),
            training_manifest_id=str(payload.get("training_manifest_id", "")),
            contract_count=int(payload.get("contract_count", 0) or 0),
            transformer_count=int(payload.get("transformer_count", 0) or 0),
            roundtrip_receipt_count=int(payload.get("roundtrip_receipt_count", 0) or 0),
            training_row_count=int(payload.get("training_row_count", 0) or 0),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "phase6_transport_scaffold_report_only")
            ),
            ready_for_phase6_3_neural_scaffold=bool(
                payload.get("ready_for_phase6_3_neural_scaffold", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_PHASE6_SCAFFOLD_REPORT_VERSION)
            ),
        )


def build_wm_transport_phase6_scaffold_report(
    *,
    contract_pack: WMTransportContractPack,
    transformer_registry: PerWMTransportTransformerRegistry,
    training_manifest: WMTransportTrainingManifest,
    roundtrip_receipt_count: int,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportPhase6ScaffoldReport:
    status = (
        "ok"
        if (
            contract_pack.ready_for_phase6_rows
            and transformer_registry.ready_for_roundtrip_eval
            and training_manifest.ready_for_trainer_scaffold
            and roundtrip_receipt_count == contract_pack.contract_count
        )
        else "blocked"
    )
    payload = {
        "contract_pack_id": contract_pack.pack_id,
        "transformer_registry_id": transformer_registry.registry_id,
        "training_manifest_id": training_manifest.manifest_id,
        "roundtrip_receipt_count": roundtrip_receipt_count,
    }
    return WMTransportPhase6ScaffoldReport(
        report_id=f"wm_transport_phase6_report_{sha256_json(payload)[:16]}",
        contract_pack_id=contract_pack.pack_id,
        transformer_registry_id=transformer_registry.registry_id,
        training_manifest_id=training_manifest.manifest_id,
        contract_count=contract_pack.contract_count,
        transformer_count=transformer_registry.transformer_count,
        roundtrip_receipt_count=roundtrip_receipt_count,
        training_row_count=training_manifest.row_count,
        status=status,
        ready_for_phase6_3_neural_scaffold=status == "ok",
        blockers=sorted(
            set(
                [
                    *contract_pack.blockers,
                    *transformer_registry.blockers,
                    *training_manifest.blockers,
                    "gpu_transport_training_not_run",
                    "phase6_3_neural_scaffold_not_built",
                ]
            )
        ),
        aggregate_counts={
            "contract_count": float(contract_pack.contract_count),
            "transformer_count": float(transformer_registry.transformer_count),
            "roundtrip_receipt_count": float(roundtrip_receipt_count),
            "training_row_count": float(training_manifest.row_count),
        },
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "phase": "6.0_6.2_transport_scaffold",
            "boundary": "contracts, rows, receipts only; no learned transport run",
            **_mapping(metadata),
        },
    )


def save_wm_transport_phase6_scaffold_report(
    path: str | Path, report: WMTransportPhase6ScaffoldReport
) -> None:
    _write_json(path, report.to_dict())


def load_wm_transport_phase6_scaffold_report(
    path: str | Path,
) -> WMTransportPhase6ScaffoldReport:
    return WMTransportPhase6ScaffoldReport.from_dict(_load_json(path))


__all__ = [
    "WM_TRANSPORT_PHASE6_SCAFFOLD_REPORT_VERSION",
    "WMTransportPhase6ScaffoldReport",
    "build_wm_transport_phase6_scaffold_report",
    "load_wm_transport_phase6_scaffold_report",
    "save_wm_transport_phase6_scaffold_report",
]
