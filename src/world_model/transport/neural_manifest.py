"""Phase-6.3 neural architecture manifest for cross-WM transport.

This is a topology and training-contract artifact. It names future learned
exporter, bridge, receiver, calibration, topology, and critic components over
Phase-6 transport rows. It does not instantiate weights, run training, grant
transport authority, or promote any bridge.
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
from src.world_model.transport.training_rows import (
    ROW_FAMILIES,
    WMTransportTrainingManifest,
    WMTransportTrainingRow,
)
from src.world_model.transport.wm_transformers import PerWMTransportTransformerRegistry

WM_TRANSPORT_NEURAL_COMPONENT_SPEC_VERSION = "wm_transport_neural_component_spec_v1"
WM_TRANSPORT_NEURAL_ARCHITECTURE_MANIFEST_VERSION = (
    "wm_transport_neural_architecture_manifest_v1"
)

PHASE6_3_NEURAL_BLOCKERS = (
    "gpu_transport_training_not_run",
    "cross_wm_corpus_density_not_proven",
    "provider_or_hardware_transport_evidence_missing",
    "topology_latency_evaluation_not_run",
    "promotion_grade_transport_benchmark_missing",
    "weights_not_written",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WMTransportNeuralComponentSpec:
    """One future learned component in the Phase-6 transport stack."""

    component_id: str
    component_key: str
    role: str
    model_family: str
    architecture_pattern: str
    input_surfaces: list[str] = field(default_factory=list)
    output_surfaces: list[str] = field(default_factory=list)
    training_rows: list[str] = field(default_factory=list)
    training_signals: list[str] = field(default_factory=list)
    loss_families: list[str] = field(default_factory=list)
    promotion_gates: list[str] = field(default_factory=list)
    authority_boundary: str = "advisory_transport_only_until_promoted"
    authority_class: str = "transport_neural_scaffold_only"
    runtime_plane: str = "gpu_train_required"
    training_ready: bool = False
    promotion_eligible: bool = False
    estimated_parameter_band: str = "unknown"
    training_stage: str = "phase6_3_scaffold"
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_NEURAL_COMPONENT_SPEC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "version": self.version,
            "component_key": self.component_key,
            "role": self.role,
            "component_role": self.role,
            "model_family": self.model_family,
            "architecture_pattern": self.architecture_pattern,
            "input_surfaces": list(self.input_surfaces),
            "output_surfaces": list(self.output_surfaces),
            "training_rows": list(self.training_rows),
            "training_signals": list(self.training_signals),
            "loss_families": list(self.loss_families),
            "promotion_gates": list(self.promotion_gates),
            "authority_boundary": self.authority_boundary,
            "authority_class": self.authority_class,
            "runtime_plane": self.runtime_plane,
            "training_ready": bool(self.training_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "estimated_parameter_band": self.estimated_parameter_band,
            "training_stage": self.training_stage,
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportNeuralComponentSpec":
        return cls(
            component_id=str(payload.get("component_id", "")),
            component_key=str(payload.get("component_key", "")),
            role=str(payload.get("role", payload.get("component_role", ""))),
            model_family=str(payload.get("model_family", "")),
            architecture_pattern=str(payload.get("architecture_pattern", "")),
            input_surfaces=[
                str(item) for item in list(payload.get("input_surfaces", []) or [])
            ],
            output_surfaces=[
                str(item) for item in list(payload.get("output_surfaces", []) or [])
            ],
            training_rows=[
                str(item) for item in list(payload.get("training_rows", []) or [])
            ],
            training_signals=[
                str(item) for item in list(payload.get("training_signals", []) or [])
            ],
            loss_families=[
                str(item) for item in list(payload.get("loss_families", []) or [])
            ],
            promotion_gates=[
                str(item) for item in list(payload.get("promotion_gates", []) or [])
            ],
            authority_boundary=str(
                payload.get(
                    "authority_boundary", "advisory_transport_only_until_promoted"
                )
            ),
            authority_class=str(
                payload.get("authority_class", "transport_neural_scaffold_only")
            ),
            runtime_plane=str(payload.get("runtime_plane", "gpu_train_required")),
            training_ready=bool(payload.get("training_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            estimated_parameter_band=str(
                payload.get("estimated_parameter_band", "unknown")
            ),
            training_stage=str(payload.get("training_stage", "phase6_3_scaffold")),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_NEURAL_COMPONENT_SPEC_VERSION)
            ),
        )


@dataclass(frozen=True)
class WMTransportNeuralArchitectureManifest:
    """Manifest for future learned transport components."""

    manifest_id: str
    contract_pack_id: str
    transformer_registry_id: str
    training_manifest_id: str
    architecture_stage: str
    components: list[WMTransportNeuralComponentSpec] = field(default_factory=list)
    row_families: list[str] = field(default_factory=list)
    input_contracts: list[str] = field(default_factory=list)
    output_contracts: list[str] = field(default_factory=list)
    loss_families: list[str] = field(default_factory=list)
    training_blockers: list[str] = field(default_factory=list)
    provider_blockers: list[str] = field(default_factory=list)
    gpu_training_ready: bool = False
    provider_bringup_ready: bool = False
    ready_for_trainer_scaffold: bool = False
    ready_for_gpu_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    live_policy_control: bool = False
    raw_hidden_state_transport: bool = False
    authority_class: str = "transport_neural_manifest_only"
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_NEURAL_ARCHITECTURE_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "contract_pack_id": self.contract_pack_id,
            "transformer_registry_id": self.transformer_registry_id,
            "training_manifest_id": self.training_manifest_id,
            "architecture_stage": self.architecture_stage,
            "components": [component.to_dict() for component in self.components],
            "row_families": list(self.row_families),
            "input_contracts": list(self.input_contracts),
            "output_contracts": list(self.output_contracts),
            "loss_families": list(self.loss_families),
            "training_blockers": list(self.training_blockers),
            "provider_blockers": list(self.provider_blockers),
            "gpu_training_ready": bool(self.gpu_training_ready),
            "provider_bringup_ready": bool(self.provider_bringup_ready),
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "live_policy_control": bool(self.live_policy_control),
            "raw_hidden_state_transport": bool(self.raw_hidden_state_transport),
            "authority_class": self.authority_class,
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportNeuralArchitectureManifest":
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            contract_pack_id=str(payload.get("contract_pack_id", "")),
            transformer_registry_id=str(payload.get("transformer_registry_id", "")),
            training_manifest_id=str(payload.get("training_manifest_id", "")),
            architecture_stage=str(payload.get("architecture_stage", "planned")),
            components=[
                WMTransportNeuralComponentSpec.from_dict(item)
                for item in list(payload.get("components", []) or [])
            ],
            row_families=[
                str(item) for item in list(payload.get("row_families", []) or [])
            ],
            input_contracts=[
                str(item) for item in list(payload.get("input_contracts", []) or [])
            ],
            output_contracts=[
                str(item) for item in list(payload.get("output_contracts", []) or [])
            ],
            loss_families=[
                str(item) for item in list(payload.get("loss_families", []) or [])
            ],
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            provider_blockers=[
                str(item) for item in list(payload.get("provider_blockers", []) or [])
            ],
            gpu_training_ready=bool(payload.get("gpu_training_ready", False)),
            provider_bringup_ready=bool(payload.get("provider_bringup_ready", False)),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", False)
            ),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            raw_hidden_state_transport=bool(
                payload.get("raw_hidden_state_transport", False)
            ),
            authority_class=str(
                payload.get("authority_class", "transport_neural_manifest_only")
            ),
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", WM_TRANSPORT_NEURAL_ARCHITECTURE_MANIFEST_VERSION
                )
            ),
        )


def _component(
    *,
    component_key: str,
    role: str,
    model_family: str,
    architecture_pattern: str,
    input_surfaces: Iterable[str],
    output_surfaces: Iterable[str],
    training_rows: Iterable[str],
    training_signals: Iterable[str],
    loss_families: Iterable[str],
    promotion_gates: Iterable[str],
    estimated_parameter_band: str,
    blockers: Iterable[str],
    runtime_plane: str = "gpu_train_required",
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportNeuralComponentSpec:
    payload = {
        "component_key": component_key,
        "model_family": model_family,
        "architecture_pattern": architecture_pattern,
        "version": WM_TRANSPORT_NEURAL_COMPONENT_SPEC_VERSION,
    }
    return WMTransportNeuralComponentSpec(
        component_id=f"wm_transport_neural_component_{sha256_json(payload)[:16]}",
        component_key=component_key,
        role=role,
        model_family=model_family,
        architecture_pattern=architecture_pattern,
        input_surfaces=list(input_surfaces),
        output_surfaces=list(output_surfaces),
        training_rows=list(training_rows),
        training_signals=list(training_signals),
        loss_families=list(loss_families),
        promotion_gates=list(promotion_gates),
        estimated_parameter_band=estimated_parameter_band,
        blockers=list(blockers),
        runtime_plane=runtime_plane,
        metadata={"training_claim": False, **_mapping(metadata)},
    )


def _planned_components(
    *,
    contracts: list[WMTransportBridgeContract],
    transformer_registry: PerWMTransportTransformerRegistry,
) -> list[WMTransportNeuralComponentSpec]:
    bridge_keys = _unique(contract.bridge_key for contract in contracts)
    wm_keys = _unique(
        [
            *(contract.source_endpoint.wm_key for contract in contracts),
            *(contract.target_endpoint.wm_key for contract in contracts),
        ]
    )
    common_gates = [
        "phase6_transport_contract_pack_ok",
        "per_wm_receiver_transformer_registry_ok",
        "roundtrip_receipts_materialized",
        "transport_training_rows_materialized",
        "gpu_training_runtime_receipt",
        "provider_or_hardware_transport_receipts",
        "promotion_grade_downstream_benchmark_evidence",
        "no_raw_hidden_state_transport",
        "no_reward_math_mutation",
    ]
    blockers = list(PHASE6_3_NEURAL_BLOCKERS)
    return [
        _component(
            component_key="typed_source_exporter_bank",
            role="Encode source WM-native canonical objects into WM-transport ontology objects without erasing source ownership.",
            model_family="per_wm_typed_object_encoder_bank",
            architecture_pattern="Small per-WM transformer/MLP encoders over typed fields, provenance tokens, and topology fields; one exporter family per source WM.",
            input_surfaces=[
                "source_endpoint",
                "source_native_surfaces",
                "provenance_fields",
            ],
            output_surfaces=["WMTransportOntologyObject", "IsomorphicBridgeInput"],
            training_rows=[
                "wm_transport_pair_row_v1",
                "wm_receiver_transformer_row_v1",
            ],
            training_signals=[
                "source_native_reconstruction",
                "provenance_preservation",
            ],
            loss_families=[
                "source_export_reconstruction_loss",
                "provenance_consistency_loss",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band=f"{len(wm_keys)} x 100K-2M",
            blockers=blockers,
            metadata={"wm_keys": wm_keys},
        ),
        _component(
            component_key="isomorphic_transport_bridge",
            role="Transport ontology objects across adjacent WMs while preserving topology, causal structure, uncertainty, provenance, and semantic compatibility.",
            model_family="relation_aware_graph_set_transformer",
            architecture_pattern="Bounded graph/set transformer over typed ontology nodes, topology edges, causal edges, uncertainty tokens, and bridge identity embeddings.",
            input_surfaces=[
                "WMTransportOntologyObject",
                "TransportTopologyMap",
                "TransportCausalMap",
            ],
            output_surfaces=[
                "IsomorphicBridgeOutput",
                "TransportCompatibilityEmbedding",
            ],
            training_rows=[
                "wm_transport_pair_row_v1",
                "wm_transport_topology_alignment_row_v1",
                "wm_transport_causal_dependency_row_v1",
            ],
            training_signals=[
                "valid_adjacent_wm_pairs",
                "topology_contrastive_positives_and_negatives",
                "causal_dependency_preservation",
            ],
            loss_families=[
                "translation_reconstruction_loss",
                "topological_contrastive_alignment_loss",
                "topology_preservation_loss",
                "causal_edge_preservation_loss",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="2M-30M",
            blockers=blockers,
            metadata={"bridge_keys": bridge_keys},
        ),
        _component(
            component_key="target_receiver_transformer_bank",
            role="Decode transported objects into target-WM-native state, rows, actionability constraints, and receipts.",
            model_family="per_wm_receiver_transformer_bank",
            architecture_pattern="Target-specific transformer heads conditioned on target WM vocabulary, target actionability surface, and receiver authority gates.",
            input_surfaces=[
                "IsomorphicBridgeOutput",
                "target_endpoint",
                "target_wm_contract",
            ],
            output_surfaces=[
                "TargetNativeTransportIntake",
                "ReceiverActionabilityScore",
            ],
            training_rows=[
                "wm_receiver_transformer_row_v1",
                "wm_transport_roundtrip_row_v1",
            ],
            training_signals=["target_native_reconstruction", "receiver_actionability"],
            loss_families=[
                "target_native_reconstruction_loss",
                "receiver_actionability_loss",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band=f"{transformer_registry.receiver_count} x 100K-3M",
            blockers=blockers,
            metadata={"receiver_count": transformer_registry.receiver_count},
        ),
        _component(
            component_key="roundtrip_cycle_decoder",
            role="Score and reconstruct source/target cycle consistency through exporter, bridge, and receiver paths.",
            model_family="cycle_consistency_decoder",
            architecture_pattern="Light decoder over source/export/bridge/receiver embeddings with explicit source and target reconstruction heads.",
            input_surfaces=["IsomorphicBridgeOutput", "TargetNativeTransportIntake"],
            output_surfaces=["RoundTripReconstruction", "CycleConsistencyScore"],
            training_rows=["wm_transport_roundtrip_row_v1"],
            training_signals=["roundtrip_receipts", "source_target_cycle_pairs"],
            loss_families=["roundtrip_consistency_loss"],
            promotion_gates=common_gates,
            estimated_parameter_band="500K-5M",
            blockers=blockers,
        ),
        _component(
            component_key="topology_causal_preservation_heads",
            role="Predict topology and causal/dependency preservation separately from semantic translation quality.",
            model_family="topology_causal_auxiliary_heads",
            architecture_pattern="Multi-head graph-edge classifiers and relation-preservation regressors over transported topology/causal tokens.",
            input_surfaces=[
                "TransportTopologyMap",
                "TransportCausalMap",
                "IsomorphicBridgeOutput",
            ],
            output_surfaces=["TopologyPreservationScore", "CausalPreservationScore"],
            training_rows=[
                "wm_transport_topology_alignment_row_v1",
                "wm_transport_causal_dependency_row_v1",
            ],
            training_signals=["topology_field_coverage", "causal_edge_coverage"],
            loss_families=[
                "topology_preservation_loss",
                "causal_edge_preservation_loss",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="250K-3M",
            blockers=blockers,
        ),
        _component(
            component_key="transport_uncertainty_calibrator",
            role="Calibrate epistemic/aleatoric uncertainty over bridge and receiver outputs.",
            model_family="temperature_calibrated_uncertainty_head",
            architecture_pattern="Small calibration head producing confidence, ECE/Brier proxies, and abstention metadata over bridge/receiver outputs.",
            input_surfaces=["TransportUncertaintyProfile", "RoundTripReceipt"],
            output_surfaces=[
                "TransportConfidence",
                "CalibrationScore",
                "AbstentionRecommendation",
            ],
            training_rows=["wm_transport_uncertainty_calibration_row_v1"],
            training_signals=[
                "uncertainty_calibration_receipts",
                "receiver_actionability",
            ],
            loss_families=["uncertainty_nll_brier_ece_loss"],
            promotion_gates=common_gates,
            estimated_parameter_band="50K-500K",
            blockers=blockers,
        ),
        _component(
            component_key="governance_actionability_classifier",
            role="Classify whether transported objects satisfy governance, provenance, authority, and target-WM actionability constraints.",
            model_family="constraint_satisfaction_classifier",
            architecture_pattern="Small classifier over governance constraints, provenance tokens, receiver outputs, and denied-authority gates.",
            input_surfaces=[
                "TransportProvenance",
                "GovernanceConstraints",
                "TargetNativeTransportIntake",
            ],
            output_surfaces=["GovernanceSatisfactionScore", "AuthorityViolationRisk"],
            training_rows=[
                "wm_receiver_transformer_row_v1",
                "wm_transport_pair_row_v1",
            ],
            training_signals=["governance_satisfaction", "authority_violation_absence"],
            loss_families=["governance_constraint_satisfaction_loss"],
            promotion_gates=common_gates,
            estimated_parameter_band="100K-1M",
            blockers=blockers,
        ),
        _component(
            component_key="downstream_shadow_transport_critic",
            role="Estimate downstream shadow usefulness without turning transport into a direct reward-seeking policy.",
            model_family="offline_shadow_critic_and_contextual_ranker",
            architecture_pattern="Offline critic/ranker over shadow outcomes, postmortem counterfactuals, and economic-yield proxies; advisory weights only.",
            input_surfaces=[
                "RoundTripReceipt",
                "ShadowOutcomeReceipt",
                "EconomicYieldProxy",
            ],
            output_surfaces=[
                "DownstreamYieldProxy",
                "SampleWeight",
                "TransportPriorityHint",
            ],
            training_rows=[
                "wm_transport_downstream_yield_row_v1",
                "wm_transport_postmortem_counterfactual_row_v1",
            ],
            training_signals=[
                "counterfactual_improvement",
                "downstream_economic_yield",
                "contextual_bandit_shadow_ranking",
            ],
            loss_families=[
                "downstream_yield_proxy_loss",
                "postmortem_counterfactual_improvement_loss",
                "contextual_bandit_shadow_ranking_loss",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="500K-5M",
            blockers=blockers,
            metadata={
                "direct_policy_rl": False,
                "rl_signal_role": "sample_weights_constraints_and_labels_only",
            },
        ),
    ]


def build_wm_transport_neural_architecture_manifest(
    *,
    contract_pack: WMTransportContractPack,
    contracts: Iterable[WMTransportBridgeContract],
    transformer_registry: PerWMTransportTransformerRegistry,
    training_manifest: WMTransportTrainingManifest,
    training_rows: Iterable[WMTransportTrainingRow],
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportNeuralArchitectureManifest:
    contracts_list = list(contracts)
    rows_list = list(training_rows)
    components = _planned_components(
        contracts=contracts_list,
        transformer_registry=transformer_registry,
    )
    input_contracts = _unique(
        [
            *(
                surface
                for component in components
                for surface in component.input_surfaces
            ),
            *(row.row_family for row in rows_list),
        ]
    )
    output_contracts = _unique(
        surface for component in components for surface in component.output_surfaces
    )
    loss_families = _unique(
        loss for component in components for loss in component.loss_families
    )
    training_blockers = _unique(
        [*PHASE6_3_NEURAL_BLOCKERS, *(b for c in components for b in c.blockers)]
    )
    provider_blockers = [
        blocker
        for blocker in training_blockers
        if "provider" in blocker or "hardware" in blocker
    ]
    payload = {
        "contract_pack_id": contract_pack.pack_id,
        "transformer_registry_id": transformer_registry.registry_id,
        "training_manifest_id": training_manifest.manifest_id,
        "component_ids": [component.component_id for component in components],
        "version": WM_TRANSPORT_NEURAL_ARCHITECTURE_MANIFEST_VERSION,
    }
    return WMTransportNeuralArchitectureManifest(
        manifest_id=f"wm_transport_neural_manifest_{sha256_json(payload)[:16]}",
        contract_pack_id=contract_pack.pack_id,
        transformer_registry_id=transformer_registry.registry_id,
        training_manifest_id=training_manifest.manifest_id,
        architecture_stage="phase6_3_neural_scaffold",
        components=components,
        row_families=list(ROW_FAMILIES),
        input_contracts=input_contracts,
        output_contracts=output_contracts,
        loss_families=loss_families,
        training_blockers=training_blockers,
        provider_blockers=provider_blockers,
        ready_for_trainer_scaffold=bool(
            contract_pack.ready_for_phase6_rows
            and transformer_registry.ready_for_roundtrip_eval
            and training_manifest.ready_for_trainer_scaffold
            and rows_list
        ),
        aggregate_counts={
            "component_count": float(len(components)),
            "contract_count": float(contract_pack.contract_count),
            "transformer_count": float(transformer_registry.transformer_count),
            "training_row_count": float(len(rows_list)),
            "loss_family_count": float(len(loss_families)),
            "gpu_train_required_count": float(
                sum(
                    1
                    for component in components
                    if component.runtime_plane == "gpu_train_required"
                )
            ),
        },
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "boundary": "neural topology manifest only; no weights, training, provider, hardware, or promotion claim",
            "training_shape": "source_exporter_to_isomorphic_bridge_to_target_receiver",
            "direct_task_reward_rl_for_transport": False,
            **_mapping(metadata),
        },
    )


def save_wm_transport_neural_architecture_manifest(
    path: str | Path, manifest: WMTransportNeuralArchitectureManifest
) -> None:
    _write_json(path, manifest.to_dict())


def load_wm_transport_neural_architecture_manifest(
    path: str | Path,
) -> WMTransportNeuralArchitectureManifest:
    return WMTransportNeuralArchitectureManifest.from_dict(_load_json(path))


__all__ = [
    "PHASE6_3_NEURAL_BLOCKERS",
    "WM_TRANSPORT_NEURAL_ARCHITECTURE_MANIFEST_VERSION",
    "WM_TRANSPORT_NEURAL_COMPONENT_SPEC_VERSION",
    "WMTransportNeuralArchitectureManifest",
    "WMTransportNeuralComponentSpec",
    "build_wm_transport_neural_architecture_manifest",
    "load_wm_transport_neural_architecture_manifest",
    "save_wm_transport_neural_architecture_manifest",
]
