"""No-hardware Phase 3.5 readiness audit for bipedal chassis work.

This module consumes the local bipedal chassis scaffold and adds the remaining
work that can be done before GPUs, sim runtime, or hardware exist: robot asset
intake contracts, kinematic consistency validators, joint-limit validation
receipts, balance-geometry checks, whole-body replay rows, and a local closure
audit. It does not claim calibrated assets, Unitree sim execution, hardware
execution, training, promotion, or live policy authority.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .bipedal_chassis import (
    BipedalChassisScaffoldReport,
    BipedalSupportState,
    BalanceEnvelopeReceipt,
    HumanoidChassisProfile,
    HumanoidFrameTree,
    JointLimitEnvelope,
    WholeBodyActionSchema,
    WholeBodyObservationSchema,
)
from .common import mapping, safe_float, stable_id, strings

PHASE35_BIPEDAL_READINESS_AUDIT_VERSION = "phase35_bipedal_readiness_audit_v1"
HUMANOID_ROBOT_ASSET_CONTRACT_VERSION = "humanoid_robot_asset_contract_v1"
ROBOT_ASSET_PARSE_RECEIPT_VERSION = "robot_asset_parse_receipt_v1"
KINEMATIC_CONSISTENCY_REPORT_VERSION = "kinematic_consistency_report_v1"
JOINT_VECTOR_VALIDATION_RECEIPT_VERSION = "joint_vector_validation_receipt_v1"
BALANCE_GEOMETRY_REPORT_VERSION = "balance_geometry_report_v1"
WHOLE_BODY_REPLAY_ROW_VERSION = "whole_body_replay_row_v1"

DENIED_BIPEDAL_READINESS_AUTHORITIES = (
    "hardware_calibrated_limits",
    "unitree_sim_runtime_executed",
    "provider_executed",
    "hardware_executed",
    "training_executed",
    "weights_written",
    "live_policy_control",
    "reward_math_mutation",
    "promotion_eligible",
)

PHASE35_BIPEDAL_READINESS_BLOCKERS = (
    "real_unitree_asset_files_missing_or_unparsed",
    "hardware_joint_limit_validation_missing",
    "calibrated_limb_transforms_missing",
    "measured_imu_contact_balance_streams_missing",
    "unitree_sim_runtime_evidence_missing",
    "whole_body_replay_corpus_missing",
    "promotion_grade_balance_benchmark_missing",
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_BIPEDAL_READINESS_AUTHORITIES}


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(row, sort_keys=True) for row in rows]
    target.write_text("\n".join(serialized) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class HumanoidRobotAssetContract:
    contract_id: str
    robot_family: str
    variant: str
    expected_formats: list[str] = field(default_factory=list)
    asset_paths: list[str] = field(default_factory=list)
    asset_status: str = "assets_unavailable"
    parser_status: str = "not_run_without_assets"
    required_asset_roles: list[str] = field(default_factory=list)
    calibration_refs: list[str] = field(default_factory=list)
    real_asset_parsed: bool = False
    hardware_calibrated_limits: bool = False
    authority_class: str = "asset_intake_contract_only"
    missing_evidence: list[str] = field(default_factory=list)
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    version: str = HUMANOID_ROBOT_ASSET_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "robot_family": self.robot_family,
            "variant": self.variant,
            "expected_formats": strings(self.expected_formats),
            "asset_paths": strings(self.asset_paths),
            "asset_status": self.asset_status,
            "parser_status": self.parser_status,
            "required_asset_roles": strings(self.required_asset_roles),
            "calibration_refs": strings(self.calibration_refs),
            "real_asset_parsed": bool(self.real_asset_parsed),
            "hardware_calibrated_limits": bool(self.hardware_calibrated_limits),
            "authority_class": self.authority_class,
            "missing_evidence": strings(self.missing_evidence),
            "denied_gates": dict(self.denied_gates),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidRobotAssetContract":
        real_asset_parsed = bool(payload.get("real_asset_parsed", False))
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            robot_family=str(payload.get("robot_family", "unknown")),
            variant=str(payload.get("variant", "")),
            expected_formats=strings(payload.get("expected_formats")),
            asset_paths=strings(payload.get("asset_paths")),
            asset_status=str(payload.get("asset_status", "assets_unavailable")),
            parser_status=str(payload.get("parser_status", "not_run_without_assets")),
            required_asset_roles=strings(payload.get("required_asset_roles")),
            calibration_refs=strings(payload.get("calibration_refs")),
            real_asset_parsed=real_asset_parsed,
            hardware_calibrated_limits=bool(
                payload.get("hardware_calibrated_limits", False)
            ),
            authority_class=str(
                payload.get("authority_class", "asset_intake_contract_only")
            ),
            missing_evidence=strings(payload.get("missing_evidence")),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(payload.get("denied_gates", {}) or {}).items()
                },
            },
            version=str(payload.get("version", HUMANOID_ROBOT_ASSET_CONTRACT_VERSION)),
        )


@dataclass(frozen=True)
class RobotAssetParseReceipt:
    receipt_id: str
    contract_id: str
    asset_path: str
    asset_format: str
    status: str
    parser_kind: str = "stdlib_xml_contract_parser"
    extracted_joint_names: list[str] = field(default_factory=list)
    extracted_frame_names: list[str] = field(default_factory=list)
    extracted_limit_count: int = 0
    parse_errors: list[str] = field(default_factory=list)
    real_asset_parsed: bool = False
    hardware_calibrated_limits: bool = False
    authority_class: str = "asset_parse_receipt_only"
    version: str = ROBOT_ASSET_PARSE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "contract_id": self.contract_id,
            "asset_path": self.asset_path,
            "asset_format": self.asset_format,
            "status": self.status,
            "parser_kind": self.parser_kind,
            "extracted_joint_names": strings(self.extracted_joint_names),
            "extracted_frame_names": strings(self.extracted_frame_names),
            "extracted_limit_count": int(self.extracted_limit_count),
            "parse_errors": strings(self.parse_errors),
            "real_asset_parsed": bool(self.real_asset_parsed),
            "hardware_calibrated_limits": bool(self.hardware_calibrated_limits),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RobotAssetParseReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            asset_path=str(payload.get("asset_path", "")),
            asset_format=str(payload.get("asset_format", "unknown")),
            status=str(payload.get("status", "")),
            parser_kind=str(
                payload.get("parser_kind", "stdlib_xml_contract_parser")
            ),
            extracted_joint_names=strings(payload.get("extracted_joint_names")),
            extracted_frame_names=strings(payload.get("extracted_frame_names")),
            extracted_limit_count=int(payload.get("extracted_limit_count", 0) or 0),
            parse_errors=strings(payload.get("parse_errors")),
            real_asset_parsed=bool(payload.get("real_asset_parsed", False)),
            hardware_calibrated_limits=bool(
                payload.get("hardware_calibrated_limits", False)
            ),
            authority_class=str(
                payload.get("authority_class", "asset_parse_receipt_only")
            ),
            version=str(payload.get("version", ROBOT_ASSET_PARSE_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class KinematicConsistencyReport:
    report_id: str
    chassis_id: str
    status: str
    controlled_joint_count: int
    action_dimension: int
    joint_limit_envelope_count: int
    frame_count: int
    minimum_21dof_invariant_passed: bool
    action_channel_alignment_passed: bool
    joint_limit_coverage_passed: bool
    frame_tree_acyclic: bool
    frame_tree_orphan_free: bool
    left_right_limb_symmetry_passed: bool
    asset_joint_alignment_status: str
    missing_joint_names: list[str] = field(default_factory=list)
    extra_asset_joint_names: list[str] = field(default_factory=list)
    missing_limit_joint_names: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    authority_class: str = "kinematic_consistency_report_only"
    version: str = KINEMATIC_CONSISTENCY_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "status": self.status,
            "controlled_joint_count": int(self.controlled_joint_count),
            "action_dimension": int(self.action_dimension),
            "joint_limit_envelope_count": int(self.joint_limit_envelope_count),
            "frame_count": int(self.frame_count),
            "minimum_21dof_invariant_passed": bool(
                self.minimum_21dof_invariant_passed
            ),
            "action_channel_alignment_passed": bool(
                self.action_channel_alignment_passed
            ),
            "joint_limit_coverage_passed": bool(self.joint_limit_coverage_passed),
            "frame_tree_acyclic": bool(self.frame_tree_acyclic),
            "frame_tree_orphan_free": bool(self.frame_tree_orphan_free),
            "left_right_limb_symmetry_passed": bool(
                self.left_right_limb_symmetry_passed
            ),
            "asset_joint_alignment_status": self.asset_joint_alignment_status,
            "missing_joint_names": strings(self.missing_joint_names),
            "extra_asset_joint_names": strings(self.extra_asset_joint_names),
            "missing_limit_joint_names": strings(self.missing_limit_joint_names),
            "blockers": strings(self.blockers),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KinematicConsistencyReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            status=str(payload.get("status", "blocked")),
            controlled_joint_count=int(payload.get("controlled_joint_count", 0) or 0),
            action_dimension=int(payload.get("action_dimension", 0) or 0),
            joint_limit_envelope_count=int(
                payload.get("joint_limit_envelope_count", 0) or 0
            ),
            frame_count=int(payload.get("frame_count", 0) or 0),
            minimum_21dof_invariant_passed=bool(
                payload.get("minimum_21dof_invariant_passed", False)
            ),
            action_channel_alignment_passed=bool(
                payload.get("action_channel_alignment_passed", False)
            ),
            joint_limit_coverage_passed=bool(
                payload.get("joint_limit_coverage_passed", False)
            ),
            frame_tree_acyclic=bool(payload.get("frame_tree_acyclic", False)),
            frame_tree_orphan_free=bool(
                payload.get("frame_tree_orphan_free", False)
            ),
            left_right_limb_symmetry_passed=bool(
                payload.get("left_right_limb_symmetry_passed", False)
            ),
            asset_joint_alignment_status=str(
                payload.get("asset_joint_alignment_status", "not_checked")
            ),
            missing_joint_names=strings(payload.get("missing_joint_names")),
            extra_asset_joint_names=strings(payload.get("extra_asset_joint_names")),
            missing_limit_joint_names=strings(payload.get("missing_limit_joint_names")),
            blockers=strings(payload.get("blockers")),
            authority_class=str(
                payload.get("authority_class", "kinematic_consistency_report_only")
            ),
            version=str(payload.get("version", KINEMATIC_CONSISTENCY_REPORT_VERSION)),
        )


@dataclass(frozen=True)
class JointVectorValidationReceipt:
    receipt_id: str
    chassis_id: str
    validation_kind: str
    status: str
    checked_joint_count: int
    violation_count: int
    violation_joint_names: list[str] = field(default_factory=list)
    vector_truth_class: str = "synthetic_local_probe"
    live_policy_control: bool = False
    promotion_eligible: bool = False
    authority_class: str = "joint_vector_validation_receipt_only"
    version: str = JOINT_VECTOR_VALIDATION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "validation_kind": self.validation_kind,
            "status": self.status,
            "checked_joint_count": int(self.checked_joint_count),
            "violation_count": int(self.violation_count),
            "violation_joint_names": strings(self.violation_joint_names),
            "vector_truth_class": self.vector_truth_class,
            "live_policy_control": bool(self.live_policy_control),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JointVectorValidationReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            validation_kind=str(payload.get("validation_kind", "")),
            status=str(payload.get("status", "")),
            checked_joint_count=int(payload.get("checked_joint_count", 0) or 0),
            violation_count=int(payload.get("violation_count", 0) or 0),
            violation_joint_names=strings(payload.get("violation_joint_names")),
            vector_truth_class=str(
                payload.get("vector_truth_class", "synthetic_local_probe")
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get("authority_class", "joint_vector_validation_receipt_only")
            ),
            version=str(payload.get("version", JOINT_VECTOR_VALIDATION_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class BalanceGeometryReport:
    report_id: str
    chassis_id: str
    support_state_id: str
    support_phase: str
    status: str
    polygon_area_m2: float
    com_inside_support: Optional[bool] = None
    zmp_inside_support: Optional[bool] = None
    cop_inside_support: Optional[bool] = None
    computed_from_measured_streams: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "balance_geometry_report_only"
    version: str = BALANCE_GEOMETRY_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "support_state_id": self.support_state_id,
            "support_phase": self.support_phase,
            "status": self.status,
            "polygon_area_m2": safe_float(self.polygon_area_m2),
            "com_inside_support": self.com_inside_support,
            "zmp_inside_support": self.zmp_inside_support,
            "cop_inside_support": self.cop_inside_support,
            "computed_from_measured_streams": bool(
                self.computed_from_measured_streams
            ),
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BalanceGeometryReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            support_state_id=str(payload.get("support_state_id", "")),
            support_phase=str(payload.get("support_phase", "")),
            status=str(payload.get("status", "")),
            polygon_area_m2=float(payload.get("polygon_area_m2", 0.0) or 0.0),
            com_inside_support=payload.get("com_inside_support"),
            zmp_inside_support=payload.get("zmp_inside_support"),
            cop_inside_support=payload.get("cop_inside_support"),
            computed_from_measured_streams=bool(
                payload.get("computed_from_measured_streams", False)
            ),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "balance_geometry_report_only")
            ),
            version=str(payload.get("version", BALANCE_GEOMETRY_REPORT_VERSION)),
        )


@dataclass(frozen=True)
class WholeBodyReplayRow:
    row_id: str
    chassis_id: str
    posture_tag: str
    support_state_id: str
    balance_receipt_id: str
    observation_schema_ref: str
    action_schema_ref: str
    joint_limit_validation_receipt_ids: list[str] = field(default_factory=list)
    asset_contract_id: str = ""
    kinematic_report_id: str = ""
    balance_geometry_report_id: str = ""
    joint_names: list[str] = field(default_factory=list)
    floating_base_slot: dict[str, Any] = field(default_factory=dict)
    resource_timing_refs: list[str] = field(default_factory=list)
    promotion_scope: str = "shadow_replay_schema_only"
    ready_for_training: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    promotion_eligible: bool = False
    version: str = WHOLE_BODY_REPLAY_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "posture_tag": self.posture_tag,
            "support_state_id": self.support_state_id,
            "balance_receipt_id": self.balance_receipt_id,
            "observation_schema_ref": self.observation_schema_ref,
            "action_schema_ref": self.action_schema_ref,
            "joint_limit_validation_receipt_ids": strings(
                self.joint_limit_validation_receipt_ids
            ),
            "asset_contract_id": self.asset_contract_id,
            "kinematic_report_id": self.kinematic_report_id,
            "balance_geometry_report_id": self.balance_geometry_report_id,
            "joint_names": strings(self.joint_names),
            "floating_base_slot": mapping(self.floating_base_slot),
            "resource_timing_refs": strings(self.resource_timing_refs),
            "promotion_scope": self.promotion_scope,
            "ready_for_training": bool(self.ready_for_training),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WholeBodyReplayRow":
        return cls(
            row_id=str(payload.get("row_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            support_state_id=str(payload.get("support_state_id", "")),
            balance_receipt_id=str(payload.get("balance_receipt_id", "")),
            observation_schema_ref=str(payload.get("observation_schema_ref", "")),
            action_schema_ref=str(payload.get("action_schema_ref", "")),
            joint_limit_validation_receipt_ids=strings(
                payload.get("joint_limit_validation_receipt_ids")
            ),
            asset_contract_id=str(payload.get("asset_contract_id", "")),
            kinematic_report_id=str(payload.get("kinematic_report_id", "")),
            balance_geometry_report_id=str(
                payload.get("balance_geometry_report_id", "")
            ),
            joint_names=strings(payload.get("joint_names")),
            floating_base_slot=mapping(payload.get("floating_base_slot")),
            resource_timing_refs=strings(payload.get("resource_timing_refs")),
            promotion_scope=str(
                payload.get("promotion_scope", "shadow_replay_schema_only")
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", WHOLE_BODY_REPLAY_ROW_VERSION)),
        )


@dataclass(frozen=True)
class Phase35BipedalReadinessAudit:
    audit_id: str
    chassis_report_id: str
    chassis_id: str
    status: str
    local_asset_ingestion_contract_present: bool
    asset_parse_receipt_count: int
    real_asset_parsed: bool
    kinematic_validators_present: bool
    joint_vector_validation_receipt_count: int
    balance_geometry_report_count: int
    whole_body_replay_row_count: int
    phase35_no_gpu_no_hardware_prepared: bool
    ready_for_unitree_runtime: bool = False
    ready_for_training: bool = False
    hardware_calibrated_limits: bool = False
    unitree_sim_runtime_executed: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    training_executed: bool = False
    weights_written: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    closed_local_surfaces: list[str] = field(default_factory=list)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE35_BIPEDAL_READINESS_AUDIT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "version": self.version,
            "chassis_report_id": self.chassis_report_id,
            "chassis_id": self.chassis_id,
            "status": self.status,
            "local_asset_ingestion_contract_present": bool(
                self.local_asset_ingestion_contract_present
            ),
            "asset_parse_receipt_count": int(self.asset_parse_receipt_count),
            "real_asset_parsed": bool(self.real_asset_parsed),
            "kinematic_validators_present": bool(self.kinematic_validators_present),
            "joint_vector_validation_receipt_count": int(
                self.joint_vector_validation_receipt_count
            ),
            "balance_geometry_report_count": int(
                self.balance_geometry_report_count
            ),
            "whole_body_replay_row_count": int(self.whole_body_replay_row_count),
            "phase35_no_gpu_no_hardware_prepared": bool(
                self.phase35_no_gpu_no_hardware_prepared
            ),
            "ready_for_unitree_runtime": bool(self.ready_for_unitree_runtime),
            "ready_for_training": bool(self.ready_for_training),
            "hardware_calibrated_limits": bool(self.hardware_calibrated_limits),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": dict(self.denied_gates),
            "closed_local_surfaces": strings(self.closed_local_surfaces),
            "remaining_blockers": strings(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase35BipedalReadinessAudit":
        real_asset_parsed = bool(payload.get("real_asset_parsed", False))
        return cls(
            audit_id=str(payload.get("audit_id", "")),
            chassis_report_id=str(payload.get("chassis_report_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            status=str(payload.get("status", "blocked")),
            local_asset_ingestion_contract_present=bool(
                payload.get("local_asset_ingestion_contract_present", False)
            ),
            asset_parse_receipt_count=int(
                payload.get("asset_parse_receipt_count", 0) or 0
            ),
            real_asset_parsed=real_asset_parsed,
            kinematic_validators_present=bool(
                payload.get("kinematic_validators_present", False)
            ),
            joint_vector_validation_receipt_count=int(
                payload.get("joint_vector_validation_receipt_count", 0) or 0
            ),
            balance_geometry_report_count=int(
                payload.get("balance_geometry_report_count", 0) or 0
            ),
            whole_body_replay_row_count=int(
                payload.get("whole_body_replay_row_count", 0) or 0
            ),
            phase35_no_gpu_no_hardware_prepared=bool(
                payload.get("phase35_no_gpu_no_hardware_prepared", False)
            ),
            ready_for_unitree_runtime=bool(
                payload.get("ready_for_unitree_runtime", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            hardware_calibrated_limits=bool(
                payload.get("hardware_calibrated_limits", False)
            ),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(payload.get("denied_gates", {}) or {}).items()
                },
            },
            closed_local_surfaces=strings(payload.get("closed_local_surfaces")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE35_BIPEDAL_READINESS_AUDIT_VERSION)
            ),
        )


def build_humanoid_robot_asset_contract(
    *,
    chassis: HumanoidChassisProfile,
    asset_paths: Optional[Iterable[str | Path]] = None,
) -> HumanoidRobotAssetContract:
    paths = [str(Path(path)) for path in list(asset_paths or []) if str(path)]
    existing = [path for path in paths if Path(path).exists()]
    status = "assets_present_unparsed" if existing else "assets_unavailable"
    parser_status = "ready_to_parse" if existing else "not_run_without_assets"
    return HumanoidRobotAssetContract(
        contract_id=stable_id(
            "humanoid_asset_contract",
            {"chassis_id": chassis.chassis_id, "asset_paths": paths},
        ),
        robot_family=chassis.robot_family,
        variant=chassis.variant,
        expected_formats=["urdf", "mjcf", "xml", "usd", "srdf"],
        asset_paths=paths,
        asset_status=status,
        parser_status=parser_status,
        required_asset_roles=[
            "robot_description",
            "joint_map",
            "joint_limits",
            "collision_geometry",
            "frame_transforms",
            "calibration_refs",
        ],
        calibration_refs=["calibration_ref_required"],
        real_asset_parsed=False,
        hardware_calibrated_limits=False,
        missing_evidence=list(PHASE35_BIPEDAL_READINESS_BLOCKERS),
        denied_gates=_denied_gates(),
    )


def _asset_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "xml":
        return "mjcf_xml"
    if suffix in {"urdf", "usd", "srdf"}:
        return suffix
    return suffix or "unknown"


def _parse_xml_asset(path: Path) -> tuple[list[str], list[str], int, list[str]]:
    errors: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        return [], [], 0, [f"xml_parse_error:{exc}"]
    joint_names: list[str] = []
    frame_names: list[str] = []
    limit_count = 0
    for elem in root.iter():
        tag = str(elem.tag).split("}")[-1]
        name = elem.attrib.get("name")
        if tag == "joint" and name:
            joint_names.append(name)
            if elem.find("limit") is not None or "range" in elem.attrib:
                limit_count += 1
        if tag in {"link", "body"} and name:
            frame_names.append(name)
    return joint_names, frame_names, limit_count, errors


def parse_robot_asset_contract(
    contract: HumanoidRobotAssetContract,
) -> list[RobotAssetParseReceipt]:
    paths = [Path(path) for path in contract.asset_paths]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return [
            RobotAssetParseReceipt(
                receipt_id=stable_id(
                    "robot_asset_parse_receipt",
                    {"contract_id": contract.contract_id, "status": "no_assets"},
                ),
                contract_id=contract.contract_id,
                asset_path="",
                asset_format="none",
                status="unavailable_no_asset_paths",
                parser_kind="asset_contract_ready_no_asset_files",
                parse_errors=["asset_paths_missing_or_unavailable"],
                real_asset_parsed=False,
            )
        ]
    receipts: list[RobotAssetParseReceipt] = []
    for path in existing:
        fmt = _asset_format(path)
        if fmt in {"urdf", "mjcf_xml", "srdf"}:
            joints, frames, limit_count, errors = _parse_xml_asset(path)
            status = "parsed_local_asset" if not errors else "parse_error"
        else:
            joints, frames, limit_count = [], [], 0
            errors = [f"parser_not_available_for:{fmt}"]
            status = "unsupported_asset_format"
        receipts.append(
            RobotAssetParseReceipt(
                receipt_id=stable_id(
                    "robot_asset_parse_receipt",
                    {
                        "contract_id": contract.contract_id,
                        "asset_path": str(path),
                        "status": status,
                    },
                ),
                contract_id=contract.contract_id,
                asset_path=str(path),
                asset_format=fmt,
                status=status,
                extracted_joint_names=joints,
                extracted_frame_names=frames,
                extracted_limit_count=limit_count,
                parse_errors=errors,
                real_asset_parsed=status == "parsed_local_asset",
                hardware_calibrated_limits=False,
            )
        )
    return receipts


def build_kinematic_consistency_report(
    *,
    chassis: HumanoidChassisProfile,
    frame_tree: HumanoidFrameTree,
    joint_limits: list[JointLimitEnvelope],
    action_schema: WholeBodyActionSchema,
    parse_receipts: list[RobotAssetParseReceipt],
) -> KinematicConsistencyReport:
    joint_names = set(chassis.joint_names)
    limit_names = {limit.joint_name for limit in joint_limits}
    asset_names = {
        name
        for receipt in parse_receipts
        if receipt.real_asset_parsed
        for name in receipt.extracted_joint_names
    }
    missing_limits = sorted(joint_names - limit_names)
    missing_asset_names = sorted(joint_names - asset_names) if asset_names else []
    extra_asset_names = sorted(asset_names - joint_names) if asset_names else []
    asset_status = (
        "aligned"
        if asset_names and not missing_asset_names and not extra_asset_names
        else "no_real_asset_to_compare"
        if not asset_names
        else "mismatch"
    )
    limb_groups = chassis.limb_groups
    left_right_symmetry = (
        len(limb_groups.get("left_leg", [])) == len(limb_groups.get("right_leg", []))
        and len(limb_groups.get("left_arm", [])) == len(limb_groups.get("right_arm", []))
        and len(limb_groups.get("left_hand", []))
        == len(limb_groups.get("right_hand", []))
    )
    checks = [
        chassis.controlled_joint_count >= 21,
        action_schema.action_dimension == chassis.controlled_joint_count,
        len(joint_limits) == chassis.controlled_joint_count and not missing_limits,
        not frame_tree.cycle_detected,
        not frame_tree.orphan_frame_ids,
        left_right_symmetry,
        asset_status in {"aligned", "no_real_asset_to_compare"},
    ]
    blockers = [] if all(checks) else ["local_kinematic_consistency_failed"]
    if asset_status == "no_real_asset_to_compare":
        blockers.append("real_robot_asset_not_available_for_alignment")
    return KinematicConsistencyReport(
        report_id=stable_id(
            "kinematic_consistency_report",
            {"chassis_id": chassis.chassis_id, "asset_status": asset_status},
        ),
        chassis_id=chassis.chassis_id,
        status="ok_contract_only" if all(checks) else "blocked",
        controlled_joint_count=chassis.controlled_joint_count,
        action_dimension=action_schema.action_dimension,
        joint_limit_envelope_count=len(joint_limits),
        frame_count=frame_tree.frame_count,
        minimum_21dof_invariant_passed=chassis.controlled_joint_count >= 21,
        action_channel_alignment_passed=(
            action_schema.action_dimension == chassis.controlled_joint_count
        ),
        joint_limit_coverage_passed=not missing_limits
        and len(joint_limits) == chassis.controlled_joint_count,
        frame_tree_acyclic=not frame_tree.cycle_detected,
        frame_tree_orphan_free=not frame_tree.orphan_frame_ids,
        left_right_limb_symmetry_passed=left_right_symmetry,
        asset_joint_alignment_status=asset_status,
        missing_joint_names=missing_asset_names,
        extra_asset_joint_names=extra_asset_names,
        missing_limit_joint_names=missing_limits,
        blockers=blockers,
    )


def validate_joint_vector(
    *,
    chassis_id: str,
    joint_limits: list[JointLimitEnvelope],
    validation_kind: str,
    positions_by_joint: Mapping[str, float],
) -> JointVectorValidationReceipt:
    violations: list[str] = []
    for limit in joint_limits:
        value = safe_float(positions_by_joint.get(limit.joint_name, 0.0))
        if value < limit.lower_rad or value > limit.upper_rad:
            violations.append(limit.joint_name)
    status = "ok" if not violations else "violations_observed"
    return JointVectorValidationReceipt(
        receipt_id=stable_id(
            "joint_vector_validation",
            {
                "chassis_id": chassis_id,
                "validation_kind": validation_kind,
                "violations": violations,
            },
        ),
        chassis_id=chassis_id,
        validation_kind=validation_kind,
        status=status,
        checked_joint_count=len(joint_limits),
        violation_count=len(violations),
        violation_joint_names=violations,
    )


def build_joint_vector_validation_receipts(
    chassis_id: str,
    joint_limits: list[JointLimitEnvelope],
) -> list[JointVectorValidationReceipt]:
    neutral = {
        limit.joint_name: (limit.lower_rad + limit.upper_rad) / 2.0
        for limit in joint_limits
    }
    violation = dict(neutral)
    if joint_limits:
        first = joint_limits[0]
        violation[first.joint_name] = first.upper_rad + 0.5
    return [
        validate_joint_vector(
            chassis_id=chassis_id,
            joint_limits=joint_limits,
            validation_kind="neutral_planning_vector",
            positions_by_joint=neutral,
        ),
        validate_joint_vector(
            chassis_id=chassis_id,
            joint_limits=joint_limits,
            validation_kind="synthetic_limit_violation_probe",
            positions_by_joint=violation,
        ),
    ]


def _polygon_area(points: list[Mapping[str, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += safe_float(point.get("x")) * safe_float(nxt.get("y"))
        area -= safe_float(nxt.get("x")) * safe_float(point.get("y"))
    return abs(area) / 2.0


def _point_in_polygon(point: Mapping[str, float], polygon: list[Mapping[str, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x = safe_float(point.get("x"))
    y = safe_float(point.get("y"))
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi = safe_float(current.get("x"))
        yi = safe_float(current.get("y"))
        previous = polygon[j]
        xj = safe_float(previous.get("x"))
        yj = safe_float(previous.get("y"))
        intersects = (yi > y) != (yj > y) and x < (
            (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def build_balance_geometry_reports(
    support_states: list[BipedalSupportState],
) -> list[BalanceGeometryReport]:
    reports: list[BalanceGeometryReport] = []
    for support in support_states:
        polygon = support.support_polygon_vertices
        area = _polygon_area(polygon)
        com_inside = (
            _point_in_polygon(support.com_projection_xy, polygon)
            if support.com_projection_xy
            else None
        )
        zmp_inside = (
            _point_in_polygon(support.zmp_xy, polygon) if support.zmp_xy else None
        )
        cop_inside = (
            _point_in_polygon(support.cop_xy, polygon) if support.cop_xy else None
        )
        status = (
            "geometry_computable_schema_slot"
            if area > 0.0 and com_inside is not None
            else "awaiting_measured_balance_streams"
        )
        reports.append(
            BalanceGeometryReport(
                report_id=stable_id(
                    "balance_geometry_report",
                    {
                        "support_state_id": support.support_state_id,
                        "support_phase": support.support_phase,
                    },
                ),
                chassis_id=support.chassis_id,
                support_state_id=support.support_state_id,
                support_phase=support.support_phase,
                status=status,
                polygon_area_m2=area,
                com_inside_support=com_inside,
                zmp_inside_support=zmp_inside,
                cop_inside_support=cop_inside,
                computed_from_measured_streams=False,
                missing_evidence=support.missing_evidence,
            )
        )
    return reports


def build_whole_body_replay_rows(
    *,
    chassis: HumanoidChassisProfile,
    observation_schema: WholeBodyObservationSchema,
    action_schema: WholeBodyActionSchema,
    support_states: list[BipedalSupportState],
    balance_receipts: list[BalanceEnvelopeReceipt],
    joint_vector_receipts: list[JointVectorValidationReceipt],
    asset_contract: HumanoidRobotAssetContract,
    kinematic_report: KinematicConsistencyReport,
    balance_geometry_reports: list[BalanceGeometryReport],
) -> list[WholeBodyReplayRow]:
    receipt_by_support = {
        receipt.support_state_id: receipt for receipt in balance_receipts
    }
    geometry_by_support = {
        report.support_state_id: report for report in balance_geometry_reports
    }
    validation_ids = [receipt.receipt_id for receipt in joint_vector_receipts]
    rows: list[WholeBodyReplayRow] = []
    for support in support_states:
        balance = receipt_by_support.get(support.support_state_id)
        geometry = geometry_by_support.get(support.support_state_id)
        rows.append(
            WholeBodyReplayRow(
                row_id=stable_id(
                    "whole_body_replay_row",
                    {
                        "chassis_id": chassis.chassis_id,
                        "support_state_id": support.support_state_id,
                    },
                ),
                chassis_id=chassis.chassis_id,
                posture_tag=chassis.posture_tag,
                support_state_id=support.support_state_id,
                balance_receipt_id=balance.receipt_id if balance else "",
                observation_schema_ref=observation_schema.schema_id,
                action_schema_ref=action_schema.schema_id,
                joint_limit_validation_receipt_ids=validation_ids,
                asset_contract_id=asset_contract.contract_id,
                kinematic_report_id=kinematic_report.report_id,
                balance_geometry_report_id=geometry.report_id if geometry else "",
                joint_names=chassis.joint_names,
                floating_base_slot={
                    "pose": "awaiting_runtime_packet_or_replay",
                    "velocity": "awaiting_runtime_packet_or_replay",
                    "truth_class": "schema_slot",
                },
                resource_timing_refs=[
                    "phase35_capacity_band_contracts",
                    "phase4_control_loop_contracts",
                    "phase4_companion_compute_contracts",
                ],
            )
        )
    return rows


def build_phase35_bipedal_readiness_audit(
    *,
    chassis_report: BipedalChassisScaffoldReport,
    chassis: HumanoidChassisProfile,
    frame_tree: HumanoidFrameTree,
    joint_limits: list[JointLimitEnvelope],
    observation_schema: WholeBodyObservationSchema,
    action_schema: WholeBodyActionSchema,
    support_states: list[BipedalSupportState],
    balance_receipts: list[BalanceEnvelopeReceipt],
    asset_paths: Optional[Iterable[str | Path]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    Phase35BipedalReadinessAudit,
    HumanoidRobotAssetContract,
    list[RobotAssetParseReceipt],
    KinematicConsistencyReport,
    list[JointVectorValidationReceipt],
    list[BalanceGeometryReport],
    list[WholeBodyReplayRow],
]:
    asset_contract = build_humanoid_robot_asset_contract(
        chassis=chassis,
        asset_paths=asset_paths,
    )
    parse_receipts = parse_robot_asset_contract(asset_contract)
    real_asset_parsed = any(receipt.real_asset_parsed for receipt in parse_receipts)
    kinematic_report = build_kinematic_consistency_report(
        chassis=chassis,
        frame_tree=frame_tree,
        joint_limits=joint_limits,
        action_schema=action_schema,
        parse_receipts=parse_receipts,
    )
    joint_vector_receipts = build_joint_vector_validation_receipts(
        chassis.chassis_id,
        joint_limits,
    )
    balance_geometry_reports = build_balance_geometry_reports(support_states)
    replay_rows = build_whole_body_replay_rows(
        chassis=chassis,
        observation_schema=observation_schema,
        action_schema=action_schema,
        support_states=support_states,
        balance_receipts=balance_receipts,
        joint_vector_receipts=joint_vector_receipts,
        asset_contract=asset_contract,
        kinematic_report=kinematic_report,
        balance_geometry_reports=balance_geometry_reports,
    )
    local_prepared = (
        chassis_report.local_structural_scaffold_complete
        and bool(asset_contract.contract_id)
        and len(parse_receipts) >= 1
        and kinematic_report.status == "ok_contract_only"
        and len(joint_vector_receipts) >= 2
        and len(balance_geometry_reports) == len(support_states)
        and len(replay_rows) == len(support_states)
    )
    audit_payload = {
        "chassis_report_id": chassis_report.report_id,
        "chassis_id": chassis.chassis_id,
        "real_asset_parsed": real_asset_parsed,
        "replay_row_count": len(replay_rows),
    }
    audit = Phase35BipedalReadinessAudit(
        audit_id=stable_id("phase35_bipedal_readiness", audit_payload),
        chassis_report_id=chassis_report.report_id,
        chassis_id=chassis.chassis_id,
        status="ok" if local_prepared else "blocked",
        local_asset_ingestion_contract_present=bool(asset_contract.contract_id),
        asset_parse_receipt_count=len(parse_receipts),
        real_asset_parsed=real_asset_parsed,
        kinematic_validators_present=kinematic_report.status == "ok_contract_only",
        joint_vector_validation_receipt_count=len(joint_vector_receipts),
        balance_geometry_report_count=len(balance_geometry_reports),
        whole_body_replay_row_count=len(replay_rows),
        phase35_no_gpu_no_hardware_prepared=local_prepared,
        denied_gates=_denied_gates(),
        closed_local_surfaces=[
            "robot_asset_intake_contract",
            "asset_parse_receipts_or_unavailable_receipt",
            "kinematic_consistency_validator",
            "joint_vector_limit_validation_receipts",
            "balance_geometry_reports",
            "whole_body_replay_rows",
        ],
        remaining_blockers=list(PHASE35_BIPEDAL_READINESS_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return (
        audit,
        asset_contract,
        parse_receipts,
        kinematic_report,
        joint_vector_receipts,
        balance_geometry_reports,
        replay_rows,
    )


def save_phase35_bipedal_readiness_audit(
    output_dir: str | Path,
    *,
    audit: Phase35BipedalReadinessAudit,
    asset_contract: HumanoidRobotAssetContract,
    parse_receipts: list[RobotAssetParseReceipt],
    kinematic_report: KinematicConsistencyReport,
    joint_vector_receipts: list[JointVectorValidationReceipt],
    balance_geometry_reports: list[BalanceGeometryReport],
    replay_rows: list[WholeBodyReplayRow],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "audit_path": output / "phase35_bipedal_readiness_audit_v1.json",
        "asset_contract_path": output / "humanoid_robot_asset_contract_v1.json",
        "asset_parse_receipts_path": output / "robot_asset_parse_receipts_v1.jsonl",
        "kinematic_report_path": output / "kinematic_consistency_report_v1.json",
        "joint_vector_receipts_path": output
        / "joint_vector_validation_receipts_v1.jsonl",
        "balance_geometry_reports_path": output / "balance_geometry_reports_v1.jsonl",
        "whole_body_replay_rows_path": output / "whole_body_replay_rows_v1.jsonl",
    }
    _write_json(paths["audit_path"], audit.to_dict())
    _write_json(paths["asset_contract_path"], asset_contract.to_dict())
    _write_jsonl(
        paths["asset_parse_receipts_path"],
        [receipt.to_dict() for receipt in parse_receipts],
    )
    _write_json(paths["kinematic_report_path"], kinematic_report.to_dict())
    _write_jsonl(
        paths["joint_vector_receipts_path"],
        [receipt.to_dict() for receipt in joint_vector_receipts],
    )
    _write_jsonl(
        paths["balance_geometry_reports_path"],
        [report.to_dict() for report in balance_geometry_reports],
    )
    _write_jsonl(
        paths["whole_body_replay_rows_path"],
        [row.to_dict() for row in replay_rows],
    )
    return {key: str(path) for key, path in paths.items()}


def load_phase35_bipedal_readiness_audit(
    path: str | Path,
) -> Phase35BipedalReadinessAudit:
    return Phase35BipedalReadinessAudit.from_dict(_load_json(path))


def load_humanoid_robot_asset_contract(
    path: str | Path,
) -> HumanoidRobotAssetContract:
    return HumanoidRobotAssetContract.from_dict(_load_json(path))


def load_robot_asset_parse_receipts(
    path: str | Path,
) -> list[RobotAssetParseReceipt]:
    return [RobotAssetParseReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_kinematic_consistency_report(
    path: str | Path,
) -> KinematicConsistencyReport:
    return KinematicConsistencyReport.from_dict(_load_json(path))


def load_joint_vector_validation_receipts(
    path: str | Path,
) -> list[JointVectorValidationReceipt]:
    return [JointVectorValidationReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_balance_geometry_reports(path: str | Path) -> list[BalanceGeometryReport]:
    return [BalanceGeometryReport.from_dict(row) for row in _load_jsonl(path)]


def load_whole_body_replay_rows(path: str | Path) -> list[WholeBodyReplayRow]:
    return [WholeBodyReplayRow.from_dict(row) for row in _load_jsonl(path)]
