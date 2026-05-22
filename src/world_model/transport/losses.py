"""Loss schema and CPU-smoke loss ledger for Phase-6.3 transport training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.neural_manifest import (
    WMTransportNeuralArchitectureManifest,
)
from src.world_model.transport.training_rows import (
    WMTransportTrainingManifest,
    WMTransportTrainingRow,
)

WM_TRANSPORT_LOSS_DEFINITION_VERSION = "wm_transport_loss_definition_v1"
WM_TRANSPORT_LOSS_LEDGER_VERSION = "wm_transport_loss_ledger_v1"

LOSS_BLOCKERS = (
    "gpu_transport_training_not_run",
    "losses_defined_not_optimized",
    "promotion_grade_transport_benchmark_missing",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


@dataclass(frozen=True)
class WMTransportLossDefinition:
    """One named loss in the Phase-6.3 trainer scaffold."""

    loss_id: str
    loss_key: str
    role: str
    row_families: list[str] = field(default_factory=list)
    component_keys: list[str] = field(default_factory=list)
    formula: str = "defined_not_optimized"
    default_weight: float = 1.0
    optimization_status: str = "defined_not_optimized"
    uses_rl_style_signal: bool = False
    direct_policy_rl: bool = False
    authority_class: str = "transport_loss_definition_only"
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_LOSS_DEFINITION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss_id": self.loss_id,
            "version": self.version,
            "loss_key": self.loss_key,
            "role": self.role,
            "row_families": list(self.row_families),
            "component_keys": list(self.component_keys),
            "formula": self.formula,
            "default_weight": float(self.default_weight),
            "optimization_status": self.optimization_status,
            "uses_rl_style_signal": bool(self.uses_rl_style_signal),
            "direct_policy_rl": bool(self.direct_policy_rl),
            "authority_class": self.authority_class,
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportLossDefinition":
        return cls(
            loss_id=str(payload.get("loss_id", "")),
            loss_key=str(payload.get("loss_key", "")),
            role=str(payload.get("role", "")),
            row_families=[
                str(item) for item in list(payload.get("row_families", []) or [])
            ],
            component_keys=[
                str(item) for item in list(payload.get("component_keys", []) or [])
            ],
            formula=str(payload.get("formula", "defined_not_optimized")),
            default_weight=float(payload.get("default_weight", 1.0)),
            optimization_status=str(
                payload.get("optimization_status", "defined_not_optimized")
            ),
            uses_rl_style_signal=bool(payload.get("uses_rl_style_signal", False)),
            direct_policy_rl=bool(payload.get("direct_policy_rl", False)),
            authority_class=str(
                payload.get("authority_class", "transport_loss_definition_only")
            ),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_LOSS_DEFINITION_VERSION)),
        )


@dataclass(frozen=True)
class WMTransportLossLedger:
    """Loss ledger emitted by the Phase-6.3 neural scaffold."""

    ledger_id: str
    neural_manifest_id: str
    training_manifest_id: str
    loss_count: int
    definitions: list[WMTransportLossDefinition] = field(default_factory=list)
    total_default_weight: float = 0.0
    status: str = "blocked"
    authority_class: str = "transport_loss_ledger_only"
    ready_for_cpu_smoke_forward: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    direct_policy_rl: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_LOSS_LEDGER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "version": self.version,
            "neural_manifest_id": self.neural_manifest_id,
            "training_manifest_id": self.training_manifest_id,
            "loss_count": int(self.loss_count),
            "definitions": [definition.to_dict() for definition in self.definitions],
            "total_default_weight": float(self.total_default_weight),
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_cpu_smoke_forward": bool(self.ready_for_cpu_smoke_forward),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "direct_policy_rl": bool(self.direct_policy_rl),
            "blockers": list(self.blockers),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportLossLedger":
        return cls(
            ledger_id=str(payload.get("ledger_id", "")),
            neural_manifest_id=str(payload.get("neural_manifest_id", "")),
            training_manifest_id=str(payload.get("training_manifest_id", "")),
            loss_count=int(payload.get("loss_count", 0) or 0),
            definitions=[
                WMTransportLossDefinition.from_dict(item)
                for item in list(payload.get("definitions", []) or [])
            ],
            total_default_weight=float(payload.get("total_default_weight", 0.0)),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "transport_loss_ledger_only")
            ),
            ready_for_cpu_smoke_forward=bool(
                payload.get("ready_for_cpu_smoke_forward", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            direct_policy_rl=bool(payload.get("direct_policy_rl", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_LOSS_LEDGER_VERSION)),
        )


def _definition(
    *,
    loss_key: str,
    role: str,
    row_families: Iterable[str],
    component_keys: Iterable[str],
    formula: str,
    default_weight: float,
    uses_rl_style_signal: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportLossDefinition:
    payload = {
        "loss_key": loss_key,
        "row_families": list(row_families),
        "component_keys": list(component_keys),
        "version": WM_TRANSPORT_LOSS_DEFINITION_VERSION,
    }
    return WMTransportLossDefinition(
        loss_id=f"wm_transport_loss_{sha256_json(payload)[:16]}",
        loss_key=loss_key,
        role=role,
        row_families=list(row_families),
        component_keys=list(component_keys),
        formula=formula,
        default_weight=float(default_weight),
        uses_rl_style_signal=uses_rl_style_signal,
        direct_policy_rl=False,
        blockers=list(LOSS_BLOCKERS),
        metadata={"training_claim": False, **_mapping(metadata)},
    )


def build_wm_transport_loss_ledger(
    *,
    neural_manifest: WMTransportNeuralArchitectureManifest,
    training_manifest: WMTransportTrainingManifest,
    training_rows: Iterable[WMTransportTrainingRow],
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportLossLedger:
    component_by_loss: Dict[str, list[str]] = {}
    for component in neural_manifest.components:
        for loss in component.loss_families:
            component_by_loss.setdefault(loss, []).append(component.component_key)
    definitions = [
        _definition(
            loss_key="source_export_reconstruction_loss",
            role="forces exporter to preserve source-WM-native typed information",
            row_families=["wm_transport_pair_row_v1"],
            component_keys=component_by_loss.get(
                "source_export_reconstruction_loss", []
            ),
            formula="masked_l1(source_native_fields, decoded_source_fields)",
            default_weight=0.8,
        ),
        _definition(
            loss_key="translation_reconstruction_loss",
            role="reconstruct target ontology fields after bridge transport",
            row_families=["wm_transport_pair_row_v1"],
            component_keys=component_by_loss.get("translation_reconstruction_loss", []),
            formula="masked_l1(target_ontology_fields, transported_fields)",
            default_weight=1.0,
        ),
        _definition(
            loss_key="topological_contrastive_alignment_loss",
            role="separate valid adjacent-WM topology pairs from topology-breaking negatives",
            row_families=["wm_transport_topology_alignment_row_v1"],
            component_keys=component_by_loss.get(
                "topological_contrastive_alignment_loss", []
            ),
            formula="info_nce(valid_adjacent_pair, topology_breaking_negatives)",
            default_weight=1.0,
        ),
        _definition(
            loss_key="topology_preservation_loss",
            role="preserve explicit topology fields across bridge transport",
            row_families=["wm_transport_topology_alignment_row_v1"],
            component_keys=component_by_loss.get("topology_preservation_loss", []),
            formula="bce_or_l1(expected_topology_edges, transported_topology_edges)",
            default_weight=0.9,
        ),
        _definition(
            loss_key="causal_edge_preservation_loss",
            role="preserve causal/dependency edges separately from topology fields",
            row_families=["wm_transport_causal_dependency_row_v1"],
            component_keys=component_by_loss.get("causal_edge_preservation_loss", []),
            formula="bce(expected_dependency_edges, transported_dependency_edges)",
            default_weight=0.8,
        ),
        _definition(
            loss_key="target_native_reconstruction_loss",
            role="force receiver outputs to match target-WM-native intake vocabulary",
            row_families=["wm_receiver_transformer_row_v1"],
            component_keys=component_by_loss.get(
                "target_native_reconstruction_loss", []
            ),
            formula="masked_l1(target_native_fields, receiver_output_fields)",
            default_weight=1.0,
        ),
        _definition(
            loss_key="receiver_actionability_loss",
            role="score whether target WM can actually use transported output",
            row_families=["wm_receiver_transformer_row_v1"],
            component_keys=component_by_loss.get("receiver_actionability_loss", []),
            formula="bce(target_actionability_label, receiver_actionability_score)",
            default_weight=1.0,
        ),
        _definition(
            loss_key="roundtrip_consistency_loss",
            role="maintain source-to-target-to-source and target-to-source-to-target cycle consistency",
            row_families=["wm_transport_roundtrip_row_v1"],
            component_keys=component_by_loss.get("roundtrip_consistency_loss", []),
            formula="l1(source_cycle, source_identity) + l1(target_cycle, target_identity)",
            default_weight=0.7,
        ),
        _definition(
            loss_key="uncertainty_nll_brier_ece_loss",
            role="calibrate transport confidence and abstention posture",
            row_families=["wm_transport_uncertainty_calibration_row_v1"],
            component_keys=component_by_loss.get("uncertainty_nll_brier_ece_loss", []),
            formula="nll + brier + ece_proxy",
            default_weight=0.6,
        ),
        _definition(
            loss_key="provenance_consistency_loss",
            role="ensure provenance and WM identity survive transport",
            row_families=["wm_transport_pair_row_v1"],
            component_keys=component_by_loss.get("provenance_consistency_loss", []),
            formula="cross_entropy(source_wm_id, transported_wm_id) + provenance_ref_mismatch_penalty",
            default_weight=0.5,
        ),
        _definition(
            loss_key="governance_constraint_satisfaction_loss",
            role="penalize authority, governance, and receiver-boundary violations",
            row_families=["wm_receiver_transformer_row_v1", "wm_transport_pair_row_v1"],
            component_keys=component_by_loss.get(
                "governance_constraint_satisfaction_loss", []
            ),
            formula="constraint_violation_penalty(no_raw_hidden_state, advisory_only, receiver_required)",
            default_weight=0.9,
        ),
        _definition(
            loss_key="downstream_yield_proxy_loss",
            role="predict downstream shadow usefulness without direct policy control",
            row_families=["wm_transport_downstream_yield_row_v1"],
            component_keys=component_by_loss.get("downstream_yield_proxy_loss", []),
            formula="huber(predicted_shadow_yield, observed_shadow_yield_proxy)",
            default_weight=0.4,
            uses_rl_style_signal=True,
            metadata={"rl_signal_role": "offline_label_or_sample_weight_only"},
        ),
        _definition(
            loss_key="postmortem_counterfactual_improvement_loss",
            role="predict postmortem counterfactual improvement from transported evidence",
            row_families=["wm_transport_postmortem_counterfactual_row_v1"],
            component_keys=component_by_loss.get(
                "postmortem_counterfactual_improvement_loss", []
            ),
            formula="huber(predicted_counterfactual_improvement, postmortem_improvement_proxy)",
            default_weight=0.4,
            uses_rl_style_signal=True,
            metadata={"rl_signal_role": "postmortem_label_only"},
        ),
        _definition(
            loss_key="contextual_bandit_shadow_ranking_loss",
            role="rank shadow transport choices offline; not a bridge policy objective",
            row_families=["wm_transport_downstream_yield_row_v1"],
            component_keys=component_by_loss.get(
                "contextual_bandit_shadow_ranking_loss", []
            ),
            formula="pairwise_rank_loss(chosen_shadow_transport, alternative_transport)",
            default_weight=0.25,
            uses_rl_style_signal=True,
            metadata={"direct_policy_rl": False},
        ),
    ]
    rows = list(training_rows)
    payload = {
        "neural_manifest_id": neural_manifest.manifest_id,
        "training_manifest_id": training_manifest.manifest_id,
        "loss_keys": [definition.loss_key for definition in definitions],
    }
    return WMTransportLossLedger(
        ledger_id=f"wm_transport_loss_ledger_{sha256_json(payload)[:16]}",
        neural_manifest_id=neural_manifest.manifest_id,
        training_manifest_id=training_manifest.manifest_id,
        loss_count=len(definitions),
        definitions=definitions,
        total_default_weight=sum(
            definition.default_weight for definition in definitions
        ),
        status="ok" if definitions and rows else "blocked",
        ready_for_cpu_smoke_forward=bool(definitions and rows),
        blockers=list(LOSS_BLOCKERS),
        aggregate_counts={
            "loss_count": float(len(definitions)),
            "rl_style_signal_loss_count": float(
                sum(1 for definition in definitions if definition.uses_rl_style_signal)
            ),
            "direct_policy_rl_loss_count": float(
                sum(1 for definition in definitions if definition.direct_policy_rl)
            ),
            "training_row_count": float(len(rows)),
        },
        metadata={
            "transport_is_policy": False,
            "rl_style_signals_are_advisory": True,
            **_mapping(metadata),
        },
    )


def save_wm_transport_loss_ledger(
    path: str | Path, ledger: WMTransportLossLedger
) -> None:
    _write_json(path, ledger.to_dict())


def load_wm_transport_loss_ledger(path: str | Path) -> WMTransportLossLedger:
    return WMTransportLossLedger.from_dict(_load_json(path))


__all__ = [
    "LOSS_BLOCKERS",
    "WM_TRANSPORT_LOSS_DEFINITION_VERSION",
    "WM_TRANSPORT_LOSS_LEDGER_VERSION",
    "WMTransportLossDefinition",
    "WMTransportLossLedger",
    "build_wm_transport_loss_ledger",
    "load_wm_transport_loss_ledger",
    "save_wm_transport_loss_ledger",
]
