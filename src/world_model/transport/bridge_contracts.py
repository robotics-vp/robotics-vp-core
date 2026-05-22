"""Phase-6 typed cross-WM transport bridge contracts.

These contracts are structural middleware only. They define allowed adjacent-WM
transport shapes, source exporter / target receiver requirements, ontology
mapping, topology/causal fields, uncertainty, provenance, and authority gates.
They do not train models, run providers, promote outputs, or transport raw
hidden states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.lower_wm_maturity_sweep import (
    EconomicWMLowerWMMaturityRow,
    EconomicWMLowerWMMaturitySweep,
    load_economic_wm_lower_wm_maturity_rows,
    load_economic_wm_lower_wm_maturity_sweep,
)
from src.world_model.economic_world_model.phase5_local_prep import (
    EconomicWMPhase5LocalPrepManifest,
    load_economic_wm_phase5_local_prep_manifest,
)

WM_TRANSPORT_ENDPOINT_VERSION = "wm_transport_endpoint_v1"
WM_TRANSPORT_ONTOLOGY_MAPPING_VERSION = "wm_transport_ontology_mapping_v1"
WM_TRANSPORT_TOPOLOGY_MAP_VERSION = "wm_transport_topology_map_v1"
WM_TRANSPORT_CAUSAL_MAP_VERSION = "wm_transport_causal_map_v1"
WM_TRANSPORT_UNCERTAINTY_PROFILE_VERSION = "wm_transport_uncertainty_profile_v1"
WM_TRANSPORT_PROVENANCE_VERSION = "wm_transport_provenance_v1"
WM_TRANSPORT_BRIDGE_CONTRACT_VERSION = "wm_transport_bridge_contract_v1"
WM_TRANSPORT_CONTRACT_PACK_VERSION = "wm_transport_contract_pack_v1"

PHASE6_TRANSPORT_BLOCKERS = (
    "gpu_transport_training_not_run",
    "cross_wm_corpus_density_not_proven",
    "topology_latency_evaluation_not_run",
    "provider_or_hardware_transport_evidence_missing",
    "promotion_grade_downstream_benchmark_missing",
)

# Phase 6.0 is deliberately adjacent-only. The lower-WM bundle is an aggregation
# endpoint, not arbitrary all-to-all hidden-state transport.
ALLOWED_ADJACENT_BRIDGES = {
    "perception_grounding_to_sim_synth_physics",
    "sim_synth_physics_to_embodiment_actuation",
    "embodiment_actuation_to_economic",
    "lower_wm_bundle_to_economic",
}

STATE_VERSION_BY_WM = {
    "perception_grounding": "perception_grounding_world_state_v1",
    "sim_synth_physics": "sim_synth_physics_world_state_v1",
    "embodiment_actuation": "embodiment_actuation_world_state_v1",
    "lower_wm_bundle": "lower_wm_bundle_transport_object_v1",
    "economic": "economic_state_v1",
}

TARGET_SURFACES_BY_WM = {
    "sim_synth_physics": [
        "SimSynthPhysicsWorldState",
        "BranchEvaluationSeed",
        "PhysicalPlausibilityConstraints",
    ],
    "embodiment_actuation": [
        "EmbodimentActuationWorldState",
        "MorphologyConsistencySurface",
        "ActionabilitySurface",
    ],
    "economic": [
        "EconomicState",
        "BottleneckMap",
        "ShadowPriceFieldSeed",
        "AllocationEnvelopeSeed",
    ],
}

SOURCE_SURFACES_BY_WM = {
    "perception_grounding": [
        "PerceptionGroundingWorldState",
        "SceneTrackSurface",
        "CalibrationSurface",
    ],
    "sim_synth_physics": [
        "SimSynthPhysicsWorldState",
        "PhysicalPlausibilitySurface",
        "BranchYieldSurface",
    ],
    "embodiment_actuation": [
        "EmbodimentActuationWorldState",
        "KinematicFeasibilitySurface",
        "ResourceEnvelopeSurface",
    ],
    "lower_wm_bundle": [
        "PerceptionGroundingWorldState",
        "SimSynthPhysicsWorldState",
        "EmbodimentActuationWorldState",
        "DatapackCompositionRow",
    ],
}

TOPOLOGY_FIELDS_BY_BRIDGE = {
    "perception_grounding_to_sim_synth_physics": [
        "scene_entities",
        "spatial_relations",
        "calibration_frame",
        "temporal_tracks",
    ],
    "sim_synth_physics_to_embodiment_actuation": [
        "contact_graph",
        "kinematic_chain",
        "constraint_edges",
        "control_rate_surface",
    ],
    "embodiment_actuation_to_economic": [
        "resource_envelope",
        "battery_thermal_compute_edges",
        "actionability_constraints",
    ],
    "lower_wm_bundle_to_economic": [
        "cross_wm_source_composition",
        "functional_contribution_edges",
        "resource_receipt_links",
        "counterfactual_value_links",
    ],
}

CAUSAL_FIELDS_BY_BRIDGE = {
    "perception_grounding_to_sim_synth_physics": [
        "observation_to_branch_constraint",
        "calibration_to_physical_plausibility",
    ],
    "sim_synth_physics_to_embodiment_actuation": [
        "physics_constraint_to_actionability",
        "contact_prediction_to_control_feasibility",
    ],
    "embodiment_actuation_to_economic": [
        "resource_spend_to_allocation_pressure",
        "control_feasibility_to_value_realization",
    ],
    "lower_wm_bundle_to_economic": [
        "source_mixture_to_marginal_utility",
        "lower_wm_maturity_to_allocation_admissibility",
    ],
}


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
class TransportEndpoint:
    """One typed endpoint of an adjacent-WM transport bridge."""

    wm_key: str
    state_version: str
    state_ref: str
    state_id: str = ""
    endpoint_role: str = "source"
    transformer_id: str = ""
    surfaces: list[str] = field(default_factory=list)
    version: str = WM_TRANSPORT_ENDPOINT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wm_key": self.wm_key,
            "version": self.version,
            "state_version": self.state_version,
            "state_ref": self.state_ref,
            "state_id": self.state_id,
            "endpoint_role": self.endpoint_role,
            "transformer_id": self.transformer_id,
            "surfaces": list(self.surfaces),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportEndpoint":
        return cls(
            wm_key=str(payload.get("wm_key", "")),
            state_version=str(payload.get("state_version", "")),
            state_ref=str(payload.get("state_ref", "")),
            state_id=str(payload.get("state_id", "")),
            endpoint_role=str(payload.get("endpoint_role", "source")),
            transformer_id=str(payload.get("transformer_id", "")),
            surfaces=[str(item) for item in list(payload.get("surfaces", []) or [])],
            version=str(payload.get("version", WM_TRANSPORT_ENDPOINT_VERSION)),
        )


@dataclass(frozen=True)
class TransportOntologyMapping:
    """Typed semantic/governance mapping required for one bridge."""

    mapping_id: str
    bridge_key: str
    source_terms: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    actionability_fields: list[str] = field(default_factory=list)
    governance_constraints: list[str] = field(default_factory=list)
    version: str = WM_TRANSPORT_ONTOLOGY_MAPPING_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mapping_id": self.mapping_id,
            "version": self.version,
            "bridge_key": self.bridge_key,
            "source_terms": list(self.source_terms),
            "target_terms": list(self.target_terms),
            "required_fields": list(self.required_fields),
            "actionability_fields": list(self.actionability_fields),
            "governance_constraints": list(self.governance_constraints),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportOntologyMapping":
        return cls(
            mapping_id=str(payload.get("mapping_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_terms=[
                str(item) for item in list(payload.get("source_terms", []) or [])
            ],
            target_terms=[
                str(item) for item in list(payload.get("target_terms", []) or [])
            ],
            required_fields=[
                str(item) for item in list(payload.get("required_fields", []) or [])
            ],
            actionability_fields=[
                str(item)
                for item in list(payload.get("actionability_fields", []) or [])
            ],
            governance_constraints=[
                str(item)
                for item in list(payload.get("governance_constraints", []) or [])
            ],
            version=str(payload.get("version", WM_TRANSPORT_ONTOLOGY_MAPPING_VERSION)),
        )


@dataclass(frozen=True)
class TransportTopologyMap:
    """Topology-preservation expectations for a bridge."""

    topology_map_id: str
    bridge_key: str
    topology_fields: list[str] = field(default_factory=list)
    edge_families: list[str] = field(default_factory=list)
    preservation_required: bool = True
    version: str = WM_TRANSPORT_TOPOLOGY_MAP_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_map_id": self.topology_map_id,
            "version": self.version,
            "bridge_key": self.bridge_key,
            "topology_fields": list(self.topology_fields),
            "edge_families": list(self.edge_families),
            "preservation_required": bool(self.preservation_required),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportTopologyMap":
        return cls(
            topology_map_id=str(payload.get("topology_map_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            topology_fields=[
                str(item) for item in list(payload.get("topology_fields", []) or [])
            ],
            edge_families=[
                str(item) for item in list(payload.get("edge_families", []) or [])
            ],
            preservation_required=bool(payload.get("preservation_required", True)),
            version=str(payload.get("version", WM_TRANSPORT_TOPOLOGY_MAP_VERSION)),
        )


@dataclass(frozen=True)
class TransportCausalMap:
    """Causal/dependency-preservation expectations for a bridge."""

    causal_map_id: str
    bridge_key: str
    dependency_edges: list[str] = field(default_factory=list)
    preservation_required: bool = True
    version: str = WM_TRANSPORT_CAUSAL_MAP_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "causal_map_id": self.causal_map_id,
            "version": self.version,
            "bridge_key": self.bridge_key,
            "dependency_edges": list(self.dependency_edges),
            "preservation_required": bool(self.preservation_required),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportCausalMap":
        return cls(
            causal_map_id=str(payload.get("causal_map_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            dependency_edges=[
                str(item) for item in list(payload.get("dependency_edges", []) or [])
            ],
            preservation_required=bool(payload.get("preservation_required", True)),
            version=str(payload.get("version", WM_TRANSPORT_CAUSAL_MAP_VERSION)),
        )


@dataclass(frozen=True)
class TransportUncertaintyProfile:
    """Uncertainty profile carried by the bridge contract."""

    profile_id: str
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    calibration_required: bool = True
    confidence: float = 0.0
    version: str = WM_TRANSPORT_UNCERTAINTY_PROFILE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "epistemic_uncertainty": float(self.epistemic_uncertainty),
            "aleatoric_uncertainty": float(self.aleatoric_uncertainty),
            "calibration_required": bool(self.calibration_required),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportUncertaintyProfile":
        return cls(
            profile_id=str(payload.get("profile_id", "")),
            epistemic_uncertainty=float(payload.get("epistemic_uncertainty", 0.0)),
            aleatoric_uncertainty=float(payload.get("aleatoric_uncertainty", 0.0)),
            calibration_required=bool(payload.get("calibration_required", True)),
            confidence=float(payload.get("confidence", 0.0)),
            version=str(
                payload.get("version", WM_TRANSPORT_UNCERTAINTY_PROFILE_VERSION)
            ),
        )


@dataclass(frozen=True)
class TransportProvenance:
    """Explicit provenance for one contract."""

    provenance_id: str
    source_refs: Dict[str, Any] = field(default_factory=dict)
    wm_identity_preserved: bool = True
    phase5_followup_blockers_preserved: bool = True
    version: str = WM_TRANSPORT_PROVENANCE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "version": self.version,
            "source_refs": _mapping(self.source_refs),
            "wm_identity_preserved": bool(self.wm_identity_preserved),
            "phase5_followup_blockers_preserved": bool(
                self.phase5_followup_blockers_preserved
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportProvenance":
        return cls(
            provenance_id=str(payload.get("provenance_id", "")),
            source_refs=_mapping(payload.get("source_refs")),
            wm_identity_preserved=bool(payload.get("wm_identity_preserved", True)),
            phase5_followup_blockers_preserved=bool(
                payload.get("phase5_followup_blockers_preserved", True)
            ),
            version=str(payload.get("version", WM_TRANSPORT_PROVENANCE_VERSION)),
        )


@dataclass(frozen=True)
class WMTransportBridgeContract:
    """One adjacent-WM transport contract."""

    contract_id: str
    bridge_key: str
    source_endpoint: TransportEndpoint
    target_endpoint: TransportEndpoint
    ontology_mapping: TransportOntologyMapping
    topology_map: TransportTopologyMap
    causal_map: TransportCausalMap
    uncertainty_profile: TransportUncertaintyProfile
    provenance: TransportProvenance
    authority_class: str = "transport_contract_only"
    advisory_only: bool = True
    receiver_required: bool = True
    exporter_required: bool = True
    raw_hidden_state_transport: bool = False
    training_ready: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_BRIDGE_CONTRACT_VERSION

    @property
    def adjacent_allowed(self) -> bool:
        return self.bridge_key in ALLOWED_ADJACENT_BRIDGES

    @property
    def structurally_valid(self) -> bool:
        return (
            self.adjacent_allowed
            and self.receiver_required
            and self.exporter_required
            and not self.raw_hidden_state_transport
            and bool(self.source_endpoint.state_ref)
            and bool(self.target_endpoint.state_ref)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "bridge_key": self.bridge_key,
            "source_endpoint": self.source_endpoint.to_dict(),
            "target_endpoint": self.target_endpoint.to_dict(),
            "ontology_mapping": self.ontology_mapping.to_dict(),
            "topology_map": self.topology_map.to_dict(),
            "causal_map": self.causal_map.to_dict(),
            "uncertainty_profile": self.uncertainty_profile.to_dict(),
            "provenance": self.provenance.to_dict(),
            "authority_class": self.authority_class,
            "advisory_only": bool(self.advisory_only),
            "receiver_required": bool(self.receiver_required),
            "exporter_required": bool(self.exporter_required),
            "raw_hidden_state_transport": bool(self.raw_hidden_state_transport),
            "adjacent_allowed": bool(self.adjacent_allowed),
            "structurally_valid": bool(self.structurally_valid),
            "training_ready": bool(self.training_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportBridgeContract":
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_endpoint=TransportEndpoint.from_dict(
                dict(payload.get("source_endpoint", {}) or {})
            ),
            target_endpoint=TransportEndpoint.from_dict(
                dict(payload.get("target_endpoint", {}) or {})
            ),
            ontology_mapping=TransportOntologyMapping.from_dict(
                dict(payload.get("ontology_mapping", {}) or {})
            ),
            topology_map=TransportTopologyMap.from_dict(
                dict(payload.get("topology_map", {}) or {})
            ),
            causal_map=TransportCausalMap.from_dict(
                dict(payload.get("causal_map", {}) or {})
            ),
            uncertainty_profile=TransportUncertaintyProfile.from_dict(
                dict(payload.get("uncertainty_profile", {}) or {})
            ),
            provenance=TransportProvenance.from_dict(
                dict(payload.get("provenance", {}) or {})
            ),
            authority_class=str(
                payload.get("authority_class", "transport_contract_only")
            ),
            advisory_only=bool(payload.get("advisory_only", True)),
            receiver_required=bool(payload.get("receiver_required", True)),
            exporter_required=bool(payload.get("exporter_required", True)),
            raw_hidden_state_transport=bool(
                payload.get("raw_hidden_state_transport", False)
            ),
            training_ready=bool(payload.get("training_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_BRIDGE_CONTRACT_VERSION)),
        )


@dataclass(frozen=True)
class WMTransportContractPack:
    """Manifest for a Phase-6.0 bridge contract pack."""

    pack_id: str
    maturity_sweep_id: str
    phase5_manifest_id: str
    contract_count: int
    structurally_valid_count: int
    receiver_required_count: int
    exporter_required_count: int
    contract_path: str
    status: str
    authority_class: str = "transport_contract_pack_only"
    ready_for_phase6_rows: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_CONTRACT_PACK_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "version": self.version,
            "maturity_sweep_id": self.maturity_sweep_id,
            "phase5_manifest_id": self.phase5_manifest_id,
            "contract_count": int(self.contract_count),
            "structurally_valid_count": int(self.structurally_valid_count),
            "receiver_required_count": int(self.receiver_required_count),
            "exporter_required_count": int(self.exporter_required_count),
            "contract_path": self.contract_path,
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_phase6_rows": bool(self.ready_for_phase6_rows),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WMTransportContractPack":
        return cls(
            pack_id=str(payload.get("pack_id", "")),
            maturity_sweep_id=str(payload.get("maturity_sweep_id", "")),
            phase5_manifest_id=str(payload.get("phase5_manifest_id", "")),
            contract_count=int(payload.get("contract_count", 0) or 0),
            structurally_valid_count=int(
                payload.get("structurally_valid_count", 0) or 0
            ),
            receiver_required_count=int(payload.get("receiver_required_count", 0) or 0),
            exporter_required_count=int(payload.get("exporter_required_count", 0) or 0),
            contract_path=str(payload.get("contract_path", "")),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "transport_contract_pack_only")
            ),
            ready_for_phase6_rows=bool(payload.get("ready_for_phase6_rows", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", WM_TRANSPORT_CONTRACT_PACK_VERSION)),
        )


def _transformer_id(wm_key: str, direction: str) -> str:
    payload = {
        "wm_key": wm_key,
        "direction": direction,
        "version": "per_wm_transport_transformer_v1",
    }
    return f"wm_transport_{direction}_{wm_key}_{sha256_json(payload)[:10]}"


def _ontology_mapping(
    bridge_key: str, source_wm: str, target_wm: str
) -> TransportOntologyMapping:
    payload = {"bridge_key": bridge_key, "source_wm": source_wm, "target_wm": target_wm}
    source_terms = SOURCE_SURFACES_BY_WM.get(source_wm, [source_wm])
    target_terms = TARGET_SURFACES_BY_WM.get(target_wm, [target_wm])
    return TransportOntologyMapping(
        mapping_id=f"wm_transport_mapping_{sha256_json(payload)[:16]}",
        bridge_key=bridge_key,
        source_terms=source_terms,
        target_terms=target_terms,
        required_fields=sorted(set([*source_terms, *target_terms])),
        actionability_fields=[
            "target_native_intake_available",
            "receiver_transformer_required",
            "advisory_authority_only",
        ],
        governance_constraints=[
            "no_reward_math_mutation",
            "no_live_policy_control",
            "no_raw_hidden_state_transport",
            "phase5_followup_blockers_preserved",
        ],
    )


def _topology_map(bridge_key: str) -> TransportTopologyMap:
    fields = TOPOLOGY_FIELDS_BY_BRIDGE.get(bridge_key, [])
    payload = {"bridge_key": bridge_key, "fields": fields}
    return TransportTopologyMap(
        topology_map_id=f"wm_transport_topology_{sha256_json(payload)[:16]}",
        bridge_key=bridge_key,
        topology_fields=fields,
        edge_families=[f"{field}_edge" for field in fields],
    )


def _causal_map(bridge_key: str) -> TransportCausalMap:
    edges = CAUSAL_FIELDS_BY_BRIDGE.get(bridge_key, [])
    payload = {"bridge_key": bridge_key, "edges": edges}
    return TransportCausalMap(
        causal_map_id=f"wm_transport_causal_{sha256_json(payload)[:16]}",
        bridge_key=bridge_key,
        dependency_edges=edges,
    )


def _uncertainty_profile(
    bridge_key: str, source_row: Optional[EconomicWMLowerWMMaturityRow]
) -> TransportUncertaintyProfile:
    maturity_score = float(source_row.maturity_score if source_row else 0.5)
    confidence = max(0.0, min(1.0, maturity_score))
    payload = {"bridge_key": bridge_key, "confidence": confidence}
    return TransportUncertaintyProfile(
        profile_id=f"wm_transport_uncertainty_{sha256_json(payload)[:16]}",
        epistemic_uncertainty=1.0 - confidence,
        aleatoric_uncertainty=0.25
        if source_row and source_row.artifact_exists
        else 0.5,
        calibration_required=True,
        confidence=confidence,
    )


def _endpoint_from_maturity(
    row: EconomicWMLowerWMMaturityRow, *, role: str
) -> TransportEndpoint:
    return TransportEndpoint(
        wm_key=row.wm_key,
        state_version=row.observed_version or STATE_VERSION_BY_WM.get(row.wm_key, ""),
        state_ref=row.artifact_path,
        state_id=row.state_id,
        endpoint_role=role,
        transformer_id=_transformer_id(
            row.wm_key, "export" if role == "source" else "receive"
        ),
        surfaces=(
            SOURCE_SURFACES_BY_WM.get(row.wm_key, [])
            if role == "source"
            else TARGET_SURFACES_BY_WM.get(row.wm_key, [])
        ),
    )


def _synthetic_endpoint(
    *, wm_key: str, role: str, state_ref: str, state_id: str
) -> TransportEndpoint:
    return TransportEndpoint(
        wm_key=wm_key,
        state_version=STATE_VERSION_BY_WM.get(wm_key, ""),
        state_ref=state_ref,
        state_id=state_id,
        endpoint_role=role,
        transformer_id=_transformer_id(
            wm_key, "export" if role == "source" else "receive"
        ),
        surfaces=(
            SOURCE_SURFACES_BY_WM.get(wm_key, [])
            if role == "source"
            else TARGET_SURFACES_BY_WM.get(wm_key, [])
        ),
    )


def _contract(
    *,
    bridge_key: str,
    source_endpoint: TransportEndpoint,
    target_endpoint: TransportEndpoint,
    maturity_row: Optional[EconomicWMLowerWMMaturityRow],
    source_refs: Mapping[str, Any],
) -> WMTransportBridgeContract:
    payload = {
        "bridge_key": bridge_key,
        "source": source_endpoint.to_dict(),
        "target": target_endpoint.to_dict(),
    }
    blockers = list(PHASE6_TRANSPORT_BLOCKERS)
    return WMTransportBridgeContract(
        contract_id=f"wm_transport_contract_{sha256_json(payload)[:16]}",
        bridge_key=bridge_key,
        source_endpoint=source_endpoint,
        target_endpoint=target_endpoint,
        ontology_mapping=_ontology_mapping(
            bridge_key, source_endpoint.wm_key, target_endpoint.wm_key
        ),
        topology_map=_topology_map(bridge_key),
        causal_map=_causal_map(bridge_key),
        uncertainty_profile=_uncertainty_profile(bridge_key, maturity_row),
        provenance=TransportProvenance(
            provenance_id=f"wm_transport_provenance_{sha256_json(source_refs)[:16]}",
            source_refs=_mapping(source_refs),
        ),
        blockers=blockers,
        metadata={
            "phase": "6.0_contract_scaffold",
            "training_claim": False,
            "provider_claim": False,
            "hardware_claim": False,
        },
    )


def build_wm_transport_contract_pack(
    *,
    maturity_sweep: EconomicWMLowerWMMaturitySweep,
    maturity_rows: Iterable[EconomicWMLowerWMMaturityRow],
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
    contract_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[WMTransportContractPack, list[WMTransportBridgeContract]]:
    """Build Phase-6.0 adjacent bridge contracts from Phase-5 maturity rows."""

    by_source: Dict[str, Dict[str, EconomicWMLowerWMMaturityRow]] = {}
    for row in maturity_rows:
        if row.ready_for_phase6_contracts:
            by_source.setdefault(row.source_row_id, {})[row.wm_key] = row

    contracts: list[WMTransportBridgeContract] = []
    for source_row_id, rows_by_wm in sorted(by_source.items()):
        perception = rows_by_wm.get("perception_grounding")
        sim = rows_by_wm.get("sim_synth_physics")
        embodiment = rows_by_wm.get("embodiment_actuation")
        common_refs = {
            "source_row_id": source_row_id,
            "phase5_manifest_id": phase5_manifest.manifest_id,
            "maturity_sweep_id": maturity_sweep.sweep_id,
        }
        if perception and sim:
            contracts.append(
                _contract(
                    bridge_key="perception_grounding_to_sim_synth_physics",
                    source_endpoint=_endpoint_from_maturity(perception, role="source"),
                    target_endpoint=_endpoint_from_maturity(sim, role="target"),
                    maturity_row=perception,
                    source_refs=common_refs,
                )
            )
        if sim and embodiment:
            contracts.append(
                _contract(
                    bridge_key="sim_synth_physics_to_embodiment_actuation",
                    source_endpoint=_endpoint_from_maturity(sim, role="source"),
                    target_endpoint=_endpoint_from_maturity(embodiment, role="target"),
                    maturity_row=sim,
                    source_refs=common_refs,
                )
            )
        if embodiment:
            contracts.append(
                _contract(
                    bridge_key="embodiment_actuation_to_economic",
                    source_endpoint=_endpoint_from_maturity(embodiment, role="source"),
                    target_endpoint=_synthetic_endpoint(
                        wm_key="economic",
                        role="target",
                        state_ref=str(phase5_manifest.composition_rows_path),
                        state_id=source_row_id,
                    ),
                    maturity_row=embodiment,
                    source_refs=common_refs,
                )
            )
        if perception and sim and embodiment:
            contracts.append(
                _contract(
                    bridge_key="lower_wm_bundle_to_economic",
                    source_endpoint=_synthetic_endpoint(
                        wm_key="lower_wm_bundle",
                        role="source",
                        state_ref=str(phase5_manifest.composition_rows_path),
                        state_id=source_row_id,
                    ),
                    target_endpoint=_synthetic_endpoint(
                        wm_key="economic",
                        role="target",
                        state_ref=str(phase5_manifest.composition_rows_path),
                        state_id=source_row_id,
                    ),
                    maturity_row=embodiment,
                    source_refs={
                        **common_refs,
                        "perception_state_ref": perception.artifact_path,
                        "sim_synth_state_ref": sim.artifact_path,
                        "embodiment_state_ref": embodiment.artifact_path,
                    },
                )
            )

    valid_count = sum(1 for contract in contracts if contract.structurally_valid)
    receiver_count = sum(1 for contract in contracts if contract.receiver_required)
    exporter_count = sum(1 for contract in contracts if contract.exporter_required)
    status = "ok" if contracts and valid_count == len(contracts) else "blocked"
    payload = {
        "maturity_sweep_id": maturity_sweep.sweep_id,
        "phase5_manifest_id": phase5_manifest.manifest_id,
        "contract_ids": [contract.contract_id for contract in contracts],
    }
    pack = WMTransportContractPack(
        pack_id=f"wm_transport_contract_pack_{sha256_json(payload)[:16]}",
        maturity_sweep_id=maturity_sweep.sweep_id,
        phase5_manifest_id=phase5_manifest.manifest_id,
        contract_count=len(contracts),
        structurally_valid_count=valid_count,
        receiver_required_count=receiver_count,
        exporter_required_count=exporter_count,
        contract_path=str(contract_path),
        status=status,
        ready_for_phase6_rows=status == "ok",
        blockers=list(PHASE6_TRANSPORT_BLOCKERS),
        aggregate_counts={
            "contract_count": float(len(contracts)),
            "structurally_valid_count": float(valid_count),
            "receiver_required_count": float(receiver_count),
            "exporter_required_count": float(exporter_count),
            "raw_hidden_state_transport_count": float(
                sum(1 for contract in contracts if contract.raw_hidden_state_transport)
            ),
            "source_row_count": float(len(by_source)),
        },
        artifact_refs={
            **_mapping(artifact_refs),
            "contract_path": str(contract_path),
            "phase5_manifest_path": str(artifact_refs.get("phase5_manifest_path", ""))
            if artifact_refs
            else "",
            "maturity_sweep_path": str(artifact_refs.get("maturity_sweep_path", ""))
            if artifact_refs
            else "",
        },
        metadata={
            "phase": "6.0_contract_scaffold",
            "boundary": "typed adjacent transport contracts only; no training/promotion",
            **_mapping(metadata),
        },
    )
    return pack, contracts


def save_wm_transport_contract_pack(
    *,
    pack_path: str | Path,
    pack: WMTransportContractPack,
    contracts: Iterable[WMTransportBridgeContract],
) -> None:
    _write_json(pack_path, pack.to_dict())
    _write_jsonl(pack.contract_path, [contract.to_dict() for contract in contracts])


def load_wm_transport_contract_pack(path: str | Path) -> WMTransportContractPack:
    return WMTransportContractPack.from_dict(_load_json(path))


def load_wm_transport_bridge_contracts(
    path: str | Path,
) -> list[WMTransportBridgeContract]:
    return [WMTransportBridgeContract.from_dict(row) for row in _load_jsonl(path)]


def build_wm_transport_contract_pack_from_paths(
    *,
    maturity_sweep_path: str | Path,
    maturity_rows_path: str | Path,
    phase5_manifest_path: str | Path,
    pack_path: str | Path,
    contract_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportContractPack:
    maturity_sweep = load_economic_wm_lower_wm_maturity_sweep(maturity_sweep_path)
    maturity_rows = load_economic_wm_lower_wm_maturity_rows(maturity_rows_path)
    phase5_manifest = load_economic_wm_phase5_local_prep_manifest(phase5_manifest_path)
    pack, contracts = build_wm_transport_contract_pack(
        maturity_sweep=maturity_sweep,
        maturity_rows=maturity_rows,
        phase5_manifest=phase5_manifest,
        contract_path=contract_path,
        artifact_refs={
            "maturity_sweep_path": str(maturity_sweep_path),
            "maturity_rows_path": str(maturity_rows_path),
            "phase5_manifest_path": str(phase5_manifest_path),
            "pack_path": str(pack_path),
        },
        metadata=metadata,
    )
    save_wm_transport_contract_pack(pack_path=pack_path, pack=pack, contracts=contracts)
    return pack


__all__ = [
    "ALLOWED_ADJACENT_BRIDGES",
    "PHASE6_TRANSPORT_BLOCKERS",
    "WM_TRANSPORT_BRIDGE_CONTRACT_VERSION",
    "WM_TRANSPORT_CONTRACT_PACK_VERSION",
    "TransportCausalMap",
    "TransportEndpoint",
    "TransportOntologyMapping",
    "TransportProvenance",
    "TransportTopologyMap",
    "TransportUncertaintyProfile",
    "WMTransportBridgeContract",
    "WMTransportContractPack",
    "build_wm_transport_contract_pack",
    "build_wm_transport_contract_pack_from_paths",
    "load_wm_transport_bridge_contracts",
    "load_wm_transport_contract_pack",
    "save_wm_transport_contract_pack",
]
