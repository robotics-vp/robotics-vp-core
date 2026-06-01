"""Phase-6 transport training row materialization.

Rows are contract and receipt scaffolds only: no GPU training, weight writes,
provider execution, promotion, or live authority is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import (
    WMTransportBridgeContract,
    WMTransportContractPack,
)
from src.world_model.transport.roundtrip import WMTransportRoundTripReceipt
from src.world_model.transport.wm_transformers import PerWMTransportTransformerRegistry

WM_TRANSPORT_TRAINING_ROW_VERSION = "wm_transport_training_row_v1"
WM_TRANSPORT_TRAINING_MANIFEST_VERSION = "wm_transport_training_manifest_v1"

ROW_FAMILIES = (
    "wm_transport_pair_row_v1",
    "wm_transport_roundtrip_row_v1",
    "wm_transport_topology_alignment_row_v1",
    "wm_transport_causal_dependency_row_v1",
    "wm_transport_uncertainty_calibration_row_v1",
    "wm_transport_downstream_yield_row_v1",
    "wm_transport_postmortem_counterfactual_row_v1",
    "wm_receiver_transformer_row_v1",
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


@dataclass(frozen=True)
class WMTransportTrainingRow:
    """One row for bridge/receiver training scaffolds."""

    row_id: str
    row_family: str
    contract_id: str
    bridge_key: str
    source_wm: str
    target_wm: str
    source_exporter_id: str
    target_receiver_id: str
    source_state_ref: str
    target_state_ref: str
    feature_vector: Dict[str, float] = field(default_factory=dict)
    target_vector: Dict[str, float] = field(default_factory=dict)
    loss_families: list[str] = field(default_factory=list)
    authority_class: str = "transport_training_row_only"
    ready_for_trainer_scaffold: bool = True
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_TRAINING_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "version": self.version,
            "row_family": self.row_family,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_wm": self.source_wm,
            "target_wm": self.target_wm,
            "source_exporter_id": self.source_exporter_id,
            "target_receiver_id": self.target_receiver_id,
            "source_state_ref": self.source_state_ref,
            "target_state_ref": self.target_state_ref,
            "feature_vector": _float_dict(self.feature_vector),
            "target_vector": _float_dict(self.target_vector),
            "loss_families": list(self.loss_families),
            "authority_class": self.authority_class,
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportTrainingRow":
        return cls(
            row_id=str(payload.get("row_id", "")),
            row_family=str(payload.get("row_family", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_wm=str(payload.get("source_wm", "")),
            target_wm=str(payload.get("target_wm", "")),
            source_exporter_id=str(payload.get("source_exporter_id", "")),
            target_receiver_id=str(payload.get("target_receiver_id", "")),
            source_state_ref=str(payload.get("source_state_ref", "")),
            target_state_ref=str(payload.get("target_state_ref", "")),
            feature_vector=_float_dict(payload.get("feature_vector", {})),
            target_vector=_float_dict(payload.get("target_vector", {})),
            loss_families=[
                str(item) for item in list(payload.get("loss_families", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "transport_training_row_only")
            ),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", True)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_TRAINING_ROW_VERSION)),
        )


@dataclass(frozen=True)
class WMTransportTrainingManifest:
    """Manifest for Phase-6.1 transport row materialization."""

    manifest_id: str
    contract_pack_id: str
    transformer_registry_id: str
    row_count: int
    rows_path: str
    row_family_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "blocked"
    authority_class: str = "transport_training_manifest_only"
    ready_for_trainer_scaffold: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_TRAINING_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "contract_pack_id": self.contract_pack_id,
            "transformer_registry_id": self.transformer_registry_id,
            "row_count": int(self.row_count),
            "rows_path": self.rows_path,
            "row_family_counts": {
                str(key): int(value) for key, value in self.row_family_counts.items()
            },
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportTrainingManifest":
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            contract_pack_id=str(payload.get("contract_pack_id", "")),
            transformer_registry_id=str(payload.get("transformer_registry_id", "")),
            row_count=int(payload.get("row_count", 0) or 0),
            rows_path=str(payload.get("rows_path", "")),
            row_family_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("row_family_counts", {}) or {}
                ).items()
            },
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "transport_training_manifest_only")
            ),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_TRAINING_MANIFEST_VERSION)),
        )


def _losses_for_family(row_family: str) -> list[str]:
    return {
        "wm_transport_pair_row_v1": ["translation_reconstruction_loss"],
        "wm_transport_roundtrip_row_v1": ["roundtrip_consistency_loss"],
        "wm_transport_topology_alignment_row_v1": ["topology_preservation_loss"],
        "wm_transport_causal_dependency_row_v1": ["causal_edge_preservation_loss"],
        "wm_transport_uncertainty_calibration_row_v1": [
            "uncertainty_nll_brier_ece_loss"
        ],
        "wm_transport_downstream_yield_row_v1": ["downstream_yield_proxy_loss"],
        "wm_transport_postmortem_counterfactual_row_v1": [
            "postmortem_counterfactual_improvement_loss"
        ],
        "wm_receiver_transformer_row_v1": ["target_native_actionability_loss"],
    }.get(row_family, ["transport_scaffold_loss"])


def _feature_vector(
    contract: WMTransportBridgeContract, receipt: WMTransportRoundTripReceipt
) -> Dict[str, float]:
    return {
        "structurally_valid": 1.0 if contract.structurally_valid else 0.0,
        "adjacent_allowed": 1.0 if contract.adjacent_allowed else 0.0,
        "source_confidence": receipt.uncertainty_calibration.source_confidence,
        "target_confidence": receipt.uncertainty_calibration.target_confidence,
        "topology_field_coverage": receipt.topology_metrics.topology_field_coverage,
        "causal_edge_coverage": receipt.topology_metrics.causal_edge_coverage,
        "semantic_field_coverage": receipt.topology_metrics.semantic_field_coverage,
        "receiver_actionability": receipt.target_receiver_actionability_score,
    }


def _target_vector(
    row_family: str, receipt: WMTransportRoundTripReceipt
) -> Dict[str, float]:
    base = {
        "roundtrip_consistency_score": receipt.roundtrip_consistency_score,
        "aggregate_score": receipt.aggregate_score,
        "promotion_eligible": 0.0,
    }
    if row_family == "wm_transport_topology_alignment_row_v1":
        base["target_topology_preservation"] = receipt.topology_metrics.aggregate_score
    elif row_family == "wm_transport_uncertainty_calibration_row_v1":
        base["target_calibration_score"] = (
            receipt.uncertainty_calibration.calibration_score
        )
    elif row_family == "wm_receiver_transformer_row_v1":
        base["target_receiver_actionability"] = (
            receipt.target_receiver_actionability_score
        )
    elif row_family == "wm_transport_downstream_yield_row_v1":
        base["target_downstream_yield_proxy"] = receipt.aggregate_score
    return base


def build_wm_transport_training_rows(
    *,
    contract_pack: WMTransportContractPack,
    contracts: Iterable[WMTransportBridgeContract],
    transformer_registry: PerWMTransportTransformerRegistry,
    roundtrip_receipts: Iterable[WMTransportRoundTripReceipt],
    rows_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[WMTransportTrainingManifest, list[WMTransportTrainingRow]]:
    receipts_by_contract = {item.contract_id: item for item in roundtrip_receipts}
    rows: list[WMTransportTrainingRow] = []
    for contract in contracts:
        receipt = receipts_by_contract[contract.contract_id]
        for row_family in ROW_FAMILIES:
            row_payload = {
                "row_family": row_family,
                "contract_id": contract.contract_id,
                "bridge_key": contract.bridge_key,
            }
            rows.append(
                WMTransportTrainingRow(
                    row_id=f"wm_transport_row_{sha256_json(row_payload)[:16]}",
                    row_family=row_family,
                    contract_id=contract.contract_id,
                    bridge_key=contract.bridge_key,
                    source_wm=contract.source_endpoint.wm_key,
                    target_wm=contract.target_endpoint.wm_key,
                    source_exporter_id=contract.source_endpoint.transformer_id,
                    target_receiver_id=contract.target_endpoint.transformer_id,
                    source_state_ref=contract.source_endpoint.state_ref,
                    target_state_ref=contract.target_endpoint.state_ref,
                    feature_vector=_feature_vector(contract, receipt),
                    target_vector=_target_vector(row_family, receipt),
                    loss_families=_losses_for_family(row_family),
                    blockers=[
                        "gpu_transport_training_not_run",
                        "provider_or_hardware_transport_evidence_missing",
                        "promotion_grade_transport_benchmark_missing",
                    ],
                    source_refs={
                        "contract_id": contract.contract_id,
                        "roundtrip_receipt_id": receipt.receipt_id,
                        "transformer_registry_id": transformer_registry.registry_id,
                    },
                    metadata={
                        "training_claim": False,
                        "row_source": "phase6_0_6_2_local_scaffold",
                    },
                )
            )
    family_counts = {family: 0 for family in ROW_FAMILIES}
    for row in rows:
        family_counts[row.row_family] = family_counts.get(row.row_family, 0) + 1
    status = "ok" if rows and all(family_counts.values()) else "blocked"
    payload: dict[str, Any] = {
        "contract_pack_id": contract_pack.pack_id,
        "registry_id": transformer_registry.registry_id,
        "row_ids": [row.row_id for row in rows],
    }
    manifest = WMTransportTrainingManifest(
        manifest_id=f"wm_transport_training_manifest_{sha256_json(payload)[:16]}",
        contract_pack_id=contract_pack.pack_id,
        transformer_registry_id=transformer_registry.registry_id,
        row_count=len(rows),
        rows_path=str(rows_path),
        row_family_counts=family_counts,
        status=status,
        ready_for_trainer_scaffold=status == "ok",
        blockers=[
            "gpu_transport_training_not_run",
            "cross_wm_corpus_density_not_proven",
            "promotion_grade_transport_benchmark_missing",
        ],
        aggregate_counts={
            "row_count": float(len(rows)),
            "contract_count": float(contract_pack.contract_count),
            "row_family_count": float(len(family_counts)),
        },
        artifact_refs={
            **_mapping(artifact_refs),
            "rows_path": str(rows_path),
        },
        metadata={
            "phase": "6.1_transport_training_rows",
            "boundary": "row materialization only; no training",
            **_mapping(metadata),
        },
    )
    return manifest, rows


def save_wm_transport_training_rows(
    *,
    manifest_path: str | Path,
    manifest: WMTransportTrainingManifest,
    rows: Iterable[WMTransportTrainingRow],
) -> None:
    _write_json(manifest_path, manifest.to_dict())
    _write_jsonl(manifest.rows_path, [row.to_dict() for row in rows])


def load_wm_transport_training_manifest(
    path: str | Path,
) -> WMTransportTrainingManifest:
    return WMTransportTrainingManifest.from_dict(_load_json(path))


def load_wm_transport_training_rows(path: str | Path) -> list[WMTransportTrainingRow]:
    return [WMTransportTrainingRow.from_dict(row) for row in _load_jsonl(path)]


__all__ = [
    "ROW_FAMILIES",
    "WM_TRANSPORT_TRAINING_MANIFEST_VERSION",
    "WM_TRANSPORT_TRAINING_ROW_VERSION",
    "WMTransportTrainingManifest",
    "WMTransportTrainingRow",
    "build_wm_transport_training_rows",
    "load_wm_transport_training_manifest",
    "load_wm_transport_training_rows",
    "save_wm_transport_training_rows",
]
