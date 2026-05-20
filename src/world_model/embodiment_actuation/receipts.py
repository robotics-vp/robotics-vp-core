"""Receipt contracts for the Embodiment / Actuation world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import clip01, mapping, safe_float, strings


@dataclass(frozen=True)
class EmbodimentReceipt:
    receipt_id: str
    state_id: str
    status: str = "shadow_recorded"
    truth_class: str = "advisory"
    source_refs: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    downstream_preconditions: list[str] = field(default_factory=list)
    authority_level: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "state_id": self.state_id,
            "status": self.status,
            "truth_class": self.truth_class,
            "source_refs": mapping(self.source_refs),
            "missing_evidence": strings(self.missing_evidence),
            "degraded_reasons": strings(self.degraded_reasons),
            "downstream_preconditions": strings(self.downstream_preconditions),
            "authority_level": self.authority_level,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentCompilationReceipt(EmbodimentReceipt):
    compiled_surface_count: int = 0
    receipt_count: int = 0
    compilation_mode: str = "shadow_advisory"
    version: str = "embodiment_compilation_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "compiled_surface_count": int(self.compiled_surface_count),
                "receipt_count": int(self.receipt_count),
                "compilation_mode": self.compilation_mode,
            }
        )
        return payload


@dataclass(frozen=True)
class CapabilityProfileReceipt(EmbodimentReceipt):
    robot_family: str = "unknown"
    action_space_count: int = 0
    sensor_modality_count: int = 0
    version: str = "capability_profile_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "robot_family": self.robot_family,
                "action_space_count": int(self.action_space_count),
                "sensor_modality_count": int(self.sensor_modality_count),
            }
        )
        return payload


@dataclass(frozen=True)
class ActionSpaceValidationReceipt(EmbodimentReceipt):
    schema_id: str = ""
    dimension: int = 0
    validation_status: str = "unavailable"
    version: str = "action_space_validation_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "schema_id": self.schema_id,
                "dimension": int(self.dimension),
                "validation_status": self.validation_status,
            }
        )
        return payload


@dataclass(frozen=True)
class ObservationInterfaceReceipt(EmbodimentReceipt):
    schema_id: str = ""
    sensor_ref_count: int = 0
    proprio_field_count: int = 0
    validation_status: str = "unavailable"
    version: str = "observation_interface_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "schema_id": self.schema_id,
                "sensor_ref_count": int(self.sensor_ref_count),
                "proprio_field_count": int(self.proprio_field_count),
                "validation_status": self.validation_status,
            }
        )
        return payload


@dataclass(frozen=True)
class ContactAffordanceReceipt(EmbodimentReceipt):
    node_count: int = 0
    edge_count: int = 0
    scene_contact_feasibility: float = 0.0
    scene_affordance_coverage: float = 0.0
    version: str = "contact_affordance_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "node_count": int(self.node_count),
                "edge_count": int(self.edge_count),
                "scene_contact_feasibility": clip01(self.scene_contact_feasibility),
                "scene_affordance_coverage": clip01(self.scene_affordance_coverage),
            }
        )
        return payload


@dataclass(frozen=True)
class LocalDynamicsReceipt(EmbodimentReceipt):
    forecast_mode: str = "heuristic_shadow"
    confidence: float = 0.0
    promotion_stage: str = "heuristic_fallback"
    version: str = "local_dynamics_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "forecast_mode": self.forecast_mode,
                "confidence": clip01(self.confidence),
                "promotion_stage": self.promotion_stage,
            }
        )
        return payload


@dataclass(frozen=True)
class InverseRetargetReceipt(EmbodimentReceipt):
    retarget_mode: str = "shadow_unavailable"
    readiness_score: float = 0.0
    version: str = "inverse_retarget_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "retarget_mode": self.retarget_mode,
                "readiness_score": clip01(self.readiness_score),
            }
        )
        return payload


@dataclass(frozen=True)
class ActionProposalReceipt(EmbodimentReceipt):
    proposal_mode: str = "shadow_summary"
    proposal_count: int = 0
    action_feasibility_score: float = 0.0
    version: str = "action_proposal_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "proposal_mode": self.proposal_mode,
                "proposal_count": int(self.proposal_count),
                "action_feasibility_score": clip01(self.action_feasibility_score),
            }
        )
        return payload


@dataclass(frozen=True)
class SafetyEnvelopeReceipt(EmbodimentReceipt):
    safety_status: str = "external_blocked"
    margin_fraction: float = 0.0
    watchdog_ref_present: bool = False
    latency_profile_ref_present: bool = False
    version: str = "safety_envelope_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "safety_status": self.safety_status,
                "margin_fraction": clip01(self.margin_fraction),
                "watchdog_ref_present": bool(self.watchdog_ref_present),
                "latency_profile_ref_present": bool(self.latency_profile_ref_present),
            }
        )
        return payload


@dataclass(frozen=True)
class EmbodimentDriftReceipt(EmbodimentReceipt):
    drift_score: float = 0.0
    calibration_due: bool = False
    version: str = "embodiment_drift_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "drift_score": clip01(self.drift_score),
                "calibration_due": bool(self.calibration_due),
            }
        )
        return payload


@dataclass(frozen=True)
class CalibrationTargetReceipt(EmbodimentReceipt):
    priority_score: float = 0.0
    target_count: int = 0
    version: str = "calibration_target_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "priority_score": clip01(self.priority_score),
                "target_count": int(self.target_count),
            }
        )
        return payload


@dataclass(frozen=True)
class EmbodimentCostReceipt(EmbodimentReceipt):
    energy_wh: float = 0.0
    risk_score: float = 0.0
    latency_ms: float = 0.0
    version: str = "embodiment_cost_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "energy_wh": max(0.0, safe_float(self.energy_wh)),
                "risk_score": clip01(self.risk_score),
                "latency_ms": max(0.0, safe_float(self.latency_ms)),
            }
        )
        return payload


@dataclass(frozen=True)
class SimEmbodimentTransferReceipt(EmbodimentReceipt):
    transfer_status: str = "shadow_advisory"
    action_feasibility_score: float = 0.0
    retargeting_readiness_score: float = 0.0
    drift_score: float = 0.0
    version: str = "sim_embodiment_transfer_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "transfer_status": self.transfer_status,
                "action_feasibility_score": clip01(self.action_feasibility_score),
                "retargeting_readiness_score": clip01(self.retargeting_readiness_score),
                "drift_score": clip01(self.drift_score),
            }
        )
        return payload
