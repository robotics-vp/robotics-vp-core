"""Per-WM exporter and receiver transformer scaffolds for Phase 6 transport."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import WMTransportBridgeContract

PER_WM_TRANSPORT_TRANSFORMER_VERSION = "per_wm_transport_transformer_v1"
PER_WM_TRANSPORT_TRANSFORMER_REGISTRY_VERSION = (
    "per_wm_transport_transformer_registry_v1"
)

LOSS_FAMILIES_BY_WM = {
    "perception_grounding": [
        "spatial_temporal_alignment_loss",
        "calibration_consistency_loss",
        "concept_presence_confidence_loss",
    ],
    "sim_synth_physics": [
        "physical_plausibility_loss",
        "branch_yield_prediction_loss",
        "constraint_topology_loss",
    ],
    "embodiment_actuation": [
        "kinematic_feasibility_loss",
        "morphology_consistency_loss",
        "actionability_calibration_loss",
    ],
    "economic": [
        "counterfactual_value_fit_loss",
        "pareto_quality_proxy_loss",
        "shadow_outcome_correlation_loss",
    ],
    "lower_wm_bundle": [
        "source_composition_reconstruction_loss",
        "functional_contribution_loss",
        "provenance_preservation_loss",
    ],
}


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PerWMTransportTransformer:
    """One WM-local exporter or receiver transformer scaffold."""

    transformer_id: str
    wm_key: str
    direction: str
    role: str
    accepted_surfaces: list[str] = field(default_factory=list)
    emitted_surfaces: list[str] = field(default_factory=list)
    loss_families: list[str] = field(default_factory=list)
    training_rows: list[str] = field(default_factory=list)
    authority_class: str = "per_wm_transformer_scaffold_only"
    training_ready: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = PER_WM_TRANSPORT_TRANSFORMER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformer_id": self.transformer_id,
            "version": self.version,
            "wm_key": self.wm_key,
            "direction": self.direction,
            "role": self.role,
            "accepted_surfaces": list(self.accepted_surfaces),
            "emitted_surfaces": list(self.emitted_surfaces),
            "loss_families": list(self.loss_families),
            "training_rows": list(self.training_rows),
            "authority_class": self.authority_class,
            "training_ready": bool(self.training_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerWMTransportTransformer":
        return cls(
            transformer_id=str(payload.get("transformer_id", "")),
            wm_key=str(payload.get("wm_key", "")),
            direction=str(payload.get("direction", "")),
            role=str(payload.get("role", "")),
            accepted_surfaces=[
                str(item) for item in list(payload.get("accepted_surfaces", []) or [])
            ],
            emitted_surfaces=[
                str(item) for item in list(payload.get("emitted_surfaces", []) or [])
            ],
            loss_families=[
                str(item) for item in list(payload.get("loss_families", []) or [])
            ],
            training_rows=[
                str(item) for item in list(payload.get("training_rows", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "per_wm_transformer_scaffold_only")
            ),
            training_ready=bool(payload.get("training_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", PER_WM_TRANSPORT_TRANSFORMER_VERSION)),
        )


@dataclass(frozen=True)
class PerWMTransportTransformerRegistry:
    """Registry proving every contract has explicit exporter and receiver posture."""

    registry_id: str
    contract_pack_id: str
    transformer_count: int
    exporter_count: int
    receiver_count: int
    missing_transformer_count: int
    transformers: list[PerWMTransportTransformer] = field(default_factory=list)
    status: str = "blocked"
    authority_class: str = "per_wm_transformer_registry_only"
    ready_for_roundtrip_eval: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = PER_WM_TRANSPORT_TRANSFORMER_REGISTRY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "version": self.version,
            "contract_pack_id": self.contract_pack_id,
            "transformer_count": int(self.transformer_count),
            "exporter_count": int(self.exporter_count),
            "receiver_count": int(self.receiver_count),
            "missing_transformer_count": int(self.missing_transformer_count),
            "transformers": [item.to_dict() for item in self.transformers],
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_roundtrip_eval": bool(self.ready_for_roundtrip_eval),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PerWMTransportTransformerRegistry":
        return cls(
            registry_id=str(payload.get("registry_id", "")),
            contract_pack_id=str(payload.get("contract_pack_id", "")),
            transformer_count=int(payload.get("transformer_count", 0) or 0),
            exporter_count=int(payload.get("exporter_count", 0) or 0),
            receiver_count=int(payload.get("receiver_count", 0) or 0),
            missing_transformer_count=int(
                payload.get("missing_transformer_count", 0) or 0
            ),
            transformers=[
                PerWMTransportTransformer.from_dict(item)
                for item in list(payload.get("transformers", []) or [])
            ],
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "per_wm_transformer_registry_only")
            ),
            ready_for_roundtrip_eval=bool(
                payload.get("ready_for_roundtrip_eval", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", PER_WM_TRANSPORT_TRANSFORMER_REGISTRY_VERSION)
            ),
        )

    def by_id(self) -> Dict[str, PerWMTransportTransformer]:
        return {item.transformer_id: item for item in self.transformers}


def _transformer(
    *, endpoint_surfaces: Iterable[str], wm_key: str, direction: str, role: str
) -> PerWMTransportTransformer:
    accepted = (
        list(endpoint_surfaces)
        if direction == "export"
        else [
            "WMTransportOntologyObject",
            "IsomorphicTransportBridgeOutput",
        ]
    )
    emitted = (
        [
            "WMTransportOntologyObject",
            "IsomorphicTransportBridgeInput",
        ]
        if direction == "export"
        else list(endpoint_surfaces)
    )
    id_payload = {
        "wm_key": wm_key,
        "direction": direction,
        "version": PER_WM_TRANSPORT_TRANSFORMER_VERSION,
    }
    return PerWMTransportTransformer(
        transformer_id=f"wm_transport_{direction}_{wm_key}_{sha256_json(id_payload)[:10]}",
        wm_key=wm_key,
        direction=direction,
        role=role,
        accepted_surfaces=accepted,
        emitted_surfaces=emitted,
        loss_families=LOSS_FAMILIES_BY_WM.get(wm_key, ["native_reconstruction_loss"]),
        training_rows=["wm_receiver_transformer_row_v1"],
        blockers=[
            "gpu_transformer_training_not_run",
            "target_wm_actionability_eval_not_run",
            "promotion_grade_receiver_benchmark_missing",
        ],
        metadata={
            "training_claim": False,
            "authority": "advisory_only",
        },
    )


def build_per_wm_transformer_registry(
    *,
    contract_pack_id: str,
    contracts: Iterable[WMTransportBridgeContract],
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> PerWMTransportTransformerRegistry:
    transformers_by_key: Dict[tuple[str, str], PerWMTransportTransformer] = {}
    missing = 0
    for contract in contracts:
        source = contract.source_endpoint
        target = contract.target_endpoint
        exporter = _transformer(
            endpoint_surfaces=source.surfaces,
            wm_key=source.wm_key,
            direction="export",
            role="source_exporter",
        )
        receiver = _transformer(
            endpoint_surfaces=target.surfaces,
            wm_key=target.wm_key,
            direction="receive",
            role="target_receiver",
        )
        transformers_by_key[(exporter.wm_key, exporter.direction)] = exporter
        transformers_by_key[(receiver.wm_key, receiver.direction)] = receiver
        if not contract.exporter_required or not exporter.transformer_id:
            missing += 1
        if not contract.receiver_required or not receiver.transformer_id:
            missing += 1
    transformers = sorted(
        transformers_by_key.values(), key=lambda item: (item.wm_key, item.direction)
    )
    exporter_count = sum(1 for item in transformers if item.direction == "export")
    receiver_count = sum(1 for item in transformers if item.direction == "receive")
    status = "ok" if transformers and missing == 0 else "blocked"
    payload = {
        "contract_pack_id": contract_pack_id,
        "transformer_ids": [item.transformer_id for item in transformers],
    }
    return PerWMTransportTransformerRegistry(
        registry_id=f"wm_transport_transformer_registry_{sha256_json(payload)[:16]}",
        contract_pack_id=contract_pack_id,
        transformer_count=len(transformers),
        exporter_count=exporter_count,
        receiver_count=receiver_count,
        missing_transformer_count=missing,
        transformers=transformers,
        status=status,
        ready_for_roundtrip_eval=status == "ok",
        blockers=[
            "gpu_transformer_training_not_run",
            "receiver_actionability_benchmarks_missing",
            "transport_bridge_training_not_run",
        ],
        aggregate_counts={
            "transformer_count": float(len(transformers)),
            "exporter_count": float(exporter_count),
            "receiver_count": float(receiver_count),
            "missing_transformer_count": float(missing),
        },
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "phase": "6.0_per_wm_transformer_scaffold",
            "boundary": "transformer registry only; no trained weights",
            **_mapping(metadata),
        },
    )


def save_per_wm_transformer_registry(
    path: str | Path, registry: PerWMTransportTransformerRegistry
) -> None:
    _write_json(path, registry.to_dict())


def load_per_wm_transformer_registry(
    path: str | Path,
) -> PerWMTransportTransformerRegistry:
    return PerWMTransportTransformerRegistry.from_dict(_load_json(path))


__all__ = [
    "PER_WM_TRANSPORT_TRANSFORMER_REGISTRY_VERSION",
    "PER_WM_TRANSPORT_TRANSFORMER_VERSION",
    "PerWMTransportTransformer",
    "PerWMTransportTransformerRegistry",
    "build_per_wm_transformer_registry",
    "load_per_wm_transformer_registry",
    "save_per_wm_transformer_registry",
]
