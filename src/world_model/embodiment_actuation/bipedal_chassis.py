"""Canonical bipedal chassis surfaces for Phase 3.5.

This module moves the local humanoid target beyond gripper/hand-only action
surfaces by making bipedal chassis structure explicit: floating-base posture,
limb coordinate frames, joint-limit envelopes, whole-body observation/action
schemas, and balance receipts. It does not claim Unitree runtime, hardware
calibration, safe control limits, training, promotion, or live policy control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .common import mapping, safe_float, stable_id, strings
from .morphology import (
    G1_VARIANT_JOINTS,
    G1MorphologyProfile,
    MorphologyJointSpec,
    build_g1_morphology_profile,
)

BIPEDAL_CHASSIS_SCAFFOLD_REPORT_VERSION = "bipedal_chassis_scaffold_report_v1"
HUMANOID_CHASSIS_PROFILE_VERSION = "humanoid_chassis_profile_v1"
LIMB_COORDINATE_FRAME_VERSION = "limb_coordinate_frame_v1"
HUMANOID_FRAME_TREE_VERSION = "humanoid_frame_tree_v1"
JOINT_LIMIT_ENVELOPE_VERSION = "joint_limit_envelope_v1"
WHOLE_BODY_OBSERVATION_SCHEMA_VERSION = "whole_body_observation_schema_v1"
WHOLE_BODY_ACTION_SCHEMA_VERSION = "whole_body_action_schema_v1"
BIPEDAL_SUPPORT_STATE_VERSION = "bipedal_support_state_v1"
BALANCE_ENVELOPE_RECEIPT_VERSION = "balance_envelope_receipt_v1"

DENIED_BIPEDAL_CHASSIS_AUTHORITIES = (
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

BIPEDAL_CHASSIS_REMAINING_BLOCKERS = (
    "urdf_or_sim_asset_parse_not_run",
    "hardware_joint_limit_validation_missing",
    "calibrated_limb_transforms_missing",
    "measured_imu_contact_balance_streams_missing",
    "unitree_sim_runtime_evidence_missing",
    "promotion_grade_balance_benchmark_missing",
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_BIPEDAL_CHASSIS_AUTHORITIES}


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
class HumanoidChassisProfile:
    """Canonical chassis profile for a bipedal humanoid variant."""

    chassis_id: str
    robot_family: str
    variant: str
    posture_tag: str
    morphology_profile_id: str
    controlled_joint_count: int
    floating_base_dof: int
    minimum_total_dof: int
    joint_names: list[str] = field(default_factory=list)
    limb_groups: dict[str, list[str]] = field(default_factory=dict)
    root_frame_id: str = "world"
    base_frame_id: str = "pelvis"
    imu_frame_ids: list[str] = field(default_factory=list)
    perception_frame_ids: list[str] = field(default_factory=list)
    truth_class: str = "oss_pattern_not_hardware_calibrated"
    source_refs: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "bipedal_chassis_profile_only"
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    version: str = HUMANOID_CHASSIS_PROFILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "chassis_id": self.chassis_id,
            "version": self.version,
            "robot_family": self.robot_family,
            "variant": self.variant,
            "posture_tag": self.posture_tag,
            "morphology_profile_id": self.morphology_profile_id,
            "controlled_joint_count": int(self.controlled_joint_count),
            "floating_base_dof": int(self.floating_base_dof),
            "minimum_total_dof": int(self.minimum_total_dof),
            "joint_names": strings(self.joint_names),
            "limb_groups": {
                str(key): strings(value) for key, value in self.limb_groups.items()
            },
            "root_frame_id": self.root_frame_id,
            "base_frame_id": self.base_frame_id,
            "imu_frame_ids": strings(self.imu_frame_ids),
            "perception_frame_ids": strings(self.perception_frame_ids),
            "truth_class": self.truth_class,
            "source_refs": mapping(self.source_refs),
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
            "denied_gates": dict(self.denied_gates),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidChassisProfile":
        return cls(
            chassis_id=str(payload.get("chassis_id", "")),
            robot_family=str(payload.get("robot_family", "unknown")),
            variant=str(payload.get("variant", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            morphology_profile_id=str(payload.get("morphology_profile_id", "")),
            controlled_joint_count=int(payload.get("controlled_joint_count", 0) or 0),
            floating_base_dof=int(payload.get("floating_base_dof", 0) or 0),
            minimum_total_dof=int(payload.get("minimum_total_dof", 0) or 0),
            joint_names=strings(payload.get("joint_names")),
            limb_groups={
                str(key): strings(value)
                for key, value in dict(payload.get("limb_groups", {}) or {}).items()
            },
            root_frame_id=str(payload.get("root_frame_id", "world")),
            base_frame_id=str(payload.get("base_frame_id", "pelvis")),
            imu_frame_ids=strings(payload.get("imu_frame_ids")),
            perception_frame_ids=strings(payload.get("perception_frame_ids")),
            truth_class=str(
                payload.get("truth_class", "oss_pattern_not_hardware_calibrated")
            ),
            source_refs=mapping(payload.get("source_refs")),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "bipedal_chassis_profile_only")
            ),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(
                        payload.get("denied_gates", {}) or {}
                    ).items()
                },
            },
            version=str(payload.get("version", HUMANOID_CHASSIS_PROFILE_VERSION)),
        )


@dataclass(frozen=True)
class LimbCoordinateFrame:
    """One node in the humanoid limb/body frame tree."""

    frame_id: str
    parent_frame_id: str
    frame_role: str
    side: str = "midline"
    linked_joint_names: list[str] = field(default_factory=list)
    transform_ref: str = "calibration_transform_ref_required"
    transform_truth_class: str = "frame_contract_only"
    calibration_status: str = "uncalibrated"
    missing_evidence: list[str] = field(default_factory=list)
    version: str = LIMB_COORDINATE_FRAME_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "version": self.version,
            "parent_frame_id": self.parent_frame_id,
            "frame_role": self.frame_role,
            "side": self.side,
            "linked_joint_names": strings(self.linked_joint_names),
            "transform_ref": self.transform_ref,
            "transform_truth_class": self.transform_truth_class,
            "calibration_status": self.calibration_status,
            "missing_evidence": strings(self.missing_evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LimbCoordinateFrame":
        return cls(
            frame_id=str(payload.get("frame_id", "")),
            parent_frame_id=str(payload.get("parent_frame_id", "")),
            frame_role=str(payload.get("frame_role", "")),
            side=str(payload.get("side", "midline")),
            linked_joint_names=strings(payload.get("linked_joint_names")),
            transform_ref=str(
                payload.get("transform_ref", "calibration_transform_ref_required")
            ),
            transform_truth_class=str(
                payload.get("transform_truth_class", "frame_contract_only")
            ),
            calibration_status=str(payload.get("calibration_status", "uncalibrated")),
            missing_evidence=strings(payload.get("missing_evidence")),
            version=str(payload.get("version", LIMB_COORDINATE_FRAME_VERSION)),
        )


@dataclass(frozen=True)
class HumanoidFrameTree:
    """Validated local frame tree for bipedal whole-body state."""

    tree_id: str
    chassis_id: str
    root_frame_id: str
    frame_count: int
    frame_ids: list[str] = field(default_factory=list)
    orphan_frame_ids: list[str] = field(default_factory=list)
    cycle_detected: bool = False
    missing_calibration_frame_ids: list[str] = field(default_factory=list)
    status: str = "contract_only"
    version: str = HUMANOID_FRAME_TREE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "root_frame_id": self.root_frame_id,
            "frame_count": int(self.frame_count),
            "frame_ids": strings(self.frame_ids),
            "orphan_frame_ids": strings(self.orphan_frame_ids),
            "cycle_detected": bool(self.cycle_detected),
            "missing_calibration_frame_ids": strings(
                self.missing_calibration_frame_ids
            ),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidFrameTree":
        return cls(
            tree_id=str(payload.get("tree_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            root_frame_id=str(payload.get("root_frame_id", "world")),
            frame_count=int(payload.get("frame_count", 0) or 0),
            frame_ids=strings(payload.get("frame_ids")),
            orphan_frame_ids=strings(payload.get("orphan_frame_ids")),
            cycle_detected=bool(payload.get("cycle_detected", False)),
            missing_calibration_frame_ids=strings(
                payload.get("missing_calibration_frame_ids")
            ),
            status=str(payload.get("status", "contract_only")),
            version=str(payload.get("version", HUMANOID_FRAME_TREE_VERSION)),
        )


@dataclass(frozen=True)
class JointLimitEnvelope:
    """Per-joint planning limit envelope, not a hardware safety claim."""

    envelope_id: str
    chassis_id: str
    joint_name: str
    joint_group: str
    lower_rad: float
    upper_rad: float
    velocity_limit_rad_s: Optional[float] = None
    effort_limit_nm: Optional[float] = None
    hardware_limit_verified: bool = False
    source_class: str = "local_planning_envelope_not_hardware_calibrated"
    violation_policy: str = "emit_receipt_only"
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "joint_limit_contract_only"
    version: str = JOINT_LIMIT_ENVELOPE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "joint_name": self.joint_name,
            "joint_group": self.joint_group,
            "lower_rad": safe_float(self.lower_rad),
            "upper_rad": safe_float(self.upper_rad),
            "velocity_limit_rad_s": self.velocity_limit_rad_s,
            "effort_limit_nm": self.effort_limit_nm,
            "hardware_limit_verified": bool(self.hardware_limit_verified),
            "source_class": self.source_class,
            "violation_policy": self.violation_policy,
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JointLimitEnvelope":
        return cls(
            envelope_id=str(payload.get("envelope_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            joint_name=str(payload.get("joint_name", "")),
            joint_group=str(payload.get("joint_group", "unknown")),
            lower_rad=float(payload.get("lower_rad", 0.0) or 0.0),
            upper_rad=float(payload.get("upper_rad", 0.0) or 0.0),
            velocity_limit_rad_s=(
                None
                if payload.get("velocity_limit_rad_s") is None
                else safe_float(payload.get("velocity_limit_rad_s"))
            ),
            effort_limit_nm=(
                None
                if payload.get("effort_limit_nm") is None
                else safe_float(payload.get("effort_limit_nm"))
            ),
            hardware_limit_verified=bool(payload.get("hardware_limit_verified", False)),
            source_class=str(
                payload.get(
                    "source_class", "local_planning_envelope_not_hardware_calibrated"
                )
            ),
            violation_policy=str(payload.get("violation_policy", "emit_receipt_only")),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "joint_limit_contract_only")
            ),
            version=str(payload.get("version", JOINT_LIMIT_ENVELOPE_VERSION)),
        )


@dataclass(frozen=True)
class WholeBodyObservationSchema:
    schema_id: str
    chassis_id: str
    posture_tag: str
    channel_groups: dict[str, list[str]] = field(default_factory=dict)
    frame_tree_ref: str = ""
    joint_limit_envelope_ref: str = ""
    replay_training_awareness: list[str] = field(default_factory=list)
    authority_class: str = "whole_body_observation_schema_only"
    version: str = WHOLE_BODY_OBSERVATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "posture_tag": self.posture_tag,
            "channel_groups": {
                str(key): strings(value) for key, value in self.channel_groups.items()
            },
            "frame_tree_ref": self.frame_tree_ref,
            "joint_limit_envelope_ref": self.joint_limit_envelope_ref,
            "replay_training_awareness": strings(self.replay_training_awareness),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WholeBodyObservationSchema":
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            channel_groups={
                str(key): strings(value)
                for key, value in dict(payload.get("channel_groups", {}) or {}).items()
            },
            frame_tree_ref=str(payload.get("frame_tree_ref", "")),
            joint_limit_envelope_ref=str(payload.get("joint_limit_envelope_ref", "")),
            replay_training_awareness=strings(payload.get("replay_training_awareness")),
            authority_class=str(
                payload.get("authority_class", "whole_body_observation_schema_only")
            ),
            version=str(payload.get("version", WHOLE_BODY_OBSERVATION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class WholeBodyActionSchema:
    schema_id: str
    chassis_id: str
    posture_tag: str
    action_dimension: int
    action_channels: list[str] = field(default_factory=list)
    horizon_steps: int = 1
    support_phase_constraints: list[str] = field(default_factory=list)
    joint_limit_envelope_ref: str = ""
    fallback_envelope_ref: str = "stable_base_fallback_ref_required"
    normalized: bool = True
    authority_class: str = "whole_body_action_schema_no_live_authority"
    version: str = WHOLE_BODY_ACTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "posture_tag": self.posture_tag,
            "action_dimension": int(self.action_dimension),
            "action_channels": strings(self.action_channels),
            "horizon_steps": int(self.horizon_steps),
            "support_phase_constraints": strings(self.support_phase_constraints),
            "joint_limit_envelope_ref": self.joint_limit_envelope_ref,
            "fallback_envelope_ref": self.fallback_envelope_ref,
            "normalized": bool(self.normalized),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WholeBodyActionSchema":
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            action_dimension=int(payload.get("action_dimension", 0) or 0),
            action_channels=strings(payload.get("action_channels")),
            horizon_steps=int(payload.get("horizon_steps", 1) or 1),
            support_phase_constraints=strings(payload.get("support_phase_constraints")),
            joint_limit_envelope_ref=str(payload.get("joint_limit_envelope_ref", "")),
            fallback_envelope_ref=str(
                payload.get(
                    "fallback_envelope_ref", "stable_base_fallback_ref_required"
                )
            ),
            normalized=bool(payload.get("normalized", True)),
            authority_class=str(
                payload.get(
                    "authority_class", "whole_body_action_schema_no_live_authority"
                )
            ),
            version=str(payload.get("version", WHOLE_BODY_ACTION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class BipedalSupportState:
    support_state_id: str
    chassis_id: str
    support_phase: str
    support_contact_frames: list[str] = field(default_factory=list)
    swing_contact_frames: list[str] = field(default_factory=list)
    support_polygon_vertices: list[dict[str, float]] = field(default_factory=list)
    com_projection_xy: dict[str, float] = field(default_factory=dict)
    zmp_xy: dict[str, float] = field(default_factory=dict)
    cop_xy: dict[str, float] = field(default_factory=dict)
    balance_margin_m: Optional[float] = None
    slip_risk: Optional[float] = None
    fall_risk: Optional[float] = None
    truth_class: str = "schema_slot_not_measured"
    missing_evidence: list[str] = field(default_factory=list)
    version: str = BIPEDAL_SUPPORT_STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_state_id": self.support_state_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "support_phase": self.support_phase,
            "support_contact_frames": strings(self.support_contact_frames),
            "swing_contact_frames": strings(self.swing_contact_frames),
            "support_polygon_vertices": [
                {str(k): safe_float(v) for k, v in vertex.items()}
                for vertex in self.support_polygon_vertices
            ],
            "com_projection_xy": {
                str(k): safe_float(v) for k, v in self.com_projection_xy.items()
            },
            "zmp_xy": {str(k): safe_float(v) for k, v in self.zmp_xy.items()},
            "cop_xy": {str(k): safe_float(v) for k, v in self.cop_xy.items()},
            "balance_margin_m": self.balance_margin_m,
            "slip_risk": self.slip_risk,
            "fall_risk": self.fall_risk,
            "truth_class": self.truth_class,
            "missing_evidence": strings(self.missing_evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BipedalSupportState":
        return cls(
            support_state_id=str(payload.get("support_state_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            support_phase=str(payload.get("support_phase", "")),
            support_contact_frames=strings(payload.get("support_contact_frames")),
            swing_contact_frames=strings(payload.get("swing_contact_frames")),
            support_polygon_vertices=[
                {str(k): safe_float(v) for k, v in dict(vertex).items()}
                for vertex in list(payload.get("support_polygon_vertices", []) or [])
            ],
            com_projection_xy={
                str(k): safe_float(v)
                for k, v in dict(payload.get("com_projection_xy", {}) or {}).items()
            },
            zmp_xy={
                str(k): safe_float(v)
                for k, v in dict(payload.get("zmp_xy", {}) or {}).items()
            },
            cop_xy={
                str(k): safe_float(v)
                for k, v in dict(payload.get("cop_xy", {}) or {}).items()
            },
            balance_margin_m=(
                None
                if payload.get("balance_margin_m") is None
                else safe_float(payload.get("balance_margin_m"))
            ),
            slip_risk=(
                None
                if payload.get("slip_risk") is None
                else safe_float(payload.get("slip_risk"))
            ),
            fall_risk=(
                None
                if payload.get("fall_risk") is None
                else safe_float(payload.get("fall_risk"))
            ),
            truth_class=str(payload.get("truth_class", "schema_slot_not_measured")),
            missing_evidence=strings(payload.get("missing_evidence")),
            version=str(payload.get("version", BIPEDAL_SUPPORT_STATE_VERSION)),
        )


@dataclass(frozen=True)
class BalanceEnvelopeReceipt:
    receipt_id: str
    chassis_id: str
    support_state_id: str
    status: str
    balance_margin_class: str
    observational_only: bool = True
    promotion_eligible: bool = False
    live_policy_control: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "balance_envelope_receipt_only"
    version: str = BALANCE_ENVELOPE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "support_state_id": self.support_state_id,
            "status": self.status,
            "balance_margin_class": self.balance_margin_class,
            "observational_only": bool(self.observational_only),
            "promotion_eligible": bool(self.promotion_eligible),
            "live_policy_control": bool(self.live_policy_control),
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BalanceEnvelopeReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            support_state_id=str(payload.get("support_state_id", "")),
            status=str(payload.get("status", "")),
            balance_margin_class=str(payload.get("balance_margin_class", "unknown")),
            observational_only=bool(payload.get("observational_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "balance_envelope_receipt_only")
            ),
            version=str(payload.get("version", BALANCE_ENVELOPE_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class BipedalChassisScaffoldReport:
    report_id: str
    status: str
    chassis_id: str
    morphology_profile_id: str
    controlled_joint_count: int
    frame_count: int
    joint_limit_envelope_count: int
    support_state_count: int
    balance_receipt_count: int
    canonical_bipedal_chassis_present: bool
    limb_frame_tree_present: bool
    joint_limit_envelope_present: bool
    whole_body_observation_schema_present: bool
    whole_body_action_schema_present: bool
    balance_envelope_present: bool
    local_structural_scaffold_complete: bool
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
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = BIPEDAL_CHASSIS_SCAFFOLD_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "chassis_id": self.chassis_id,
            "morphology_profile_id": self.morphology_profile_id,
            "controlled_joint_count": int(self.controlled_joint_count),
            "frame_count": int(self.frame_count),
            "joint_limit_envelope_count": int(self.joint_limit_envelope_count),
            "support_state_count": int(self.support_state_count),
            "balance_receipt_count": int(self.balance_receipt_count),
            "canonical_bipedal_chassis_present": bool(
                self.canonical_bipedal_chassis_present
            ),
            "limb_frame_tree_present": bool(self.limb_frame_tree_present),
            "joint_limit_envelope_present": bool(self.joint_limit_envelope_present),
            "whole_body_observation_schema_present": bool(
                self.whole_body_observation_schema_present
            ),
            "whole_body_action_schema_present": bool(
                self.whole_body_action_schema_present
            ),
            "balance_envelope_present": bool(self.balance_envelope_present),
            "local_structural_scaffold_complete": bool(
                self.local_structural_scaffold_complete
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
            "remaining_blockers": strings(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BipedalChassisScaffoldReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "blocked")),
            chassis_id=str(payload.get("chassis_id", "")),
            morphology_profile_id=str(payload.get("morphology_profile_id", "")),
            controlled_joint_count=int(payload.get("controlled_joint_count", 0) or 0),
            frame_count=int(payload.get("frame_count", 0) or 0),
            joint_limit_envelope_count=int(
                payload.get("joint_limit_envelope_count", 0) or 0
            ),
            support_state_count=int(payload.get("support_state_count", 0) or 0),
            balance_receipt_count=int(payload.get("balance_receipt_count", 0) or 0),
            canonical_bipedal_chassis_present=bool(
                payload.get("canonical_bipedal_chassis_present", False)
            ),
            limb_frame_tree_present=bool(payload.get("limb_frame_tree_present", False)),
            joint_limit_envelope_present=bool(
                payload.get("joint_limit_envelope_present", False)
            ),
            whole_body_observation_schema_present=bool(
                payload.get("whole_body_observation_schema_present", False)
            ),
            whole_body_action_schema_present=bool(
                payload.get("whole_body_action_schema_present", False)
            ),
            balance_envelope_present=bool(
                payload.get("balance_envelope_present", False)
            ),
            local_structural_scaffold_complete=bool(
                payload.get("local_structural_scaffold_complete", False)
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
                    for key, value in dict(
                        payload.get("denied_gates", {}) or {}
                    ).items()
                },
            },
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", BIPEDAL_CHASSIS_SCAFFOLD_REPORT_VERSION)
            ),
        )


def _joint_limb_groups(joint_names: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "left_leg": [],
        "right_leg": [],
        "waist": [],
        "left_arm": [],
        "right_arm": [],
        "left_hand": [],
        "right_hand": [],
    }
    for joint in joint_names:
        if joint.startswith("left_") and any(
            token in joint for token in ("hip", "knee", "ankle")
        ):
            groups["left_leg"].append(joint)
        elif joint.startswith("right_") and any(
            token in joint for token in ("hip", "knee", "ankle")
        ):
            groups["right_leg"].append(joint)
        elif joint.startswith("waist_"):
            groups["waist"].append(joint)
        elif joint.startswith("left_hand"):
            groups["left_hand"].append(joint)
        elif joint.startswith("right_hand"):
            groups["right_hand"].append(joint)
        elif joint.startswith("left_"):
            groups["left_arm"].append(joint)
        elif joint.startswith("right_"):
            groups["right_arm"].append(joint)
    return {key: value for key, value in groups.items() if value}


def _planning_limits(joint: MorphologyJointSpec) -> tuple[float, float, float, float]:
    name = joint.joint_name
    if joint.lower_rad is not None and joint.upper_rad is not None:
        lower = float(joint.lower_rad)
        upper = float(joint.upper_rad)
    elif "knee" in name:
        lower, upper = -0.1, 2.6
    elif "ankle_roll" in name:
        lower, upper = -0.6, 0.6
    elif "ankle_pitch" in name:
        lower, upper = -1.0, 0.8
    elif "hip" in name:
        lower, upper = -1.8, 1.8
    elif "waist" in name:
        lower, upper = -1.2, 1.2
    elif "shoulder" in name:
        lower, upper = -2.5, 2.5
    elif "elbow" in name:
        lower, upper = -0.2, 2.3
    elif "wrist" in name:
        lower, upper = -1.8, 1.8
    elif "hand" in name or "thumb" in name or "index" in name or "middle" in name:
        lower, upper = 0.0, 1.6
    else:
        lower, upper = -1.0, 1.0
    velocity = float(joint.velocity_limit) if joint.velocity_limit else 6.0
    effort = float(joint.effort_limit) if joint.effort_limit else 0.0
    return lower, upper, velocity, effort


def _build_frame_specs(joint_names: list[str]) -> list[LimbCoordinateFrame]:
    missing = ["calibrated_transform_missing"]
    frames = [
        LimbCoordinateFrame("world", "", "world", missing_evidence=[]),
        LimbCoordinateFrame(
            "pelvis", "world", "floating_base", missing_evidence=missing
        ),
        LimbCoordinateFrame("torso", "pelvis", "torso", missing_evidence=missing),
        LimbCoordinateFrame("head", "torso", "head", missing_evidence=missing),
        LimbCoordinateFrame(
            "imu_pelvis",
            "pelvis",
            "imu",
            missing_evidence=["imu_extrinsics_missing"],
        ),
        LimbCoordinateFrame(
            "head_camera",
            "head",
            "egocentric_camera",
            missing_evidence=["camera_extrinsics_missing"],
        ),
    ]
    limb_specs = [
        ("left", "hip", "pelvis", ["hip_yaw", "hip_roll", "hip_pitch"]),
        ("left", "knee", "left_hip", ["knee"]),
        ("left", "ankle", "left_knee", ["ankle_pitch", "ankle_roll"]),
        ("left", "foot", "left_ankle", []),
        ("right", "hip", "pelvis", ["hip_yaw", "hip_roll", "hip_pitch"]),
        ("right", "knee", "right_hip", ["knee"]),
        ("right", "ankle", "right_knee", ["ankle_pitch", "ankle_roll"]),
        ("right", "foot", "right_ankle", []),
        (
            "left",
            "shoulder",
            "torso",
            ["shoulder_pitch", "shoulder_roll", "shoulder_yaw"],
        ),
        ("left", "elbow", "left_shoulder", ["elbow"]),
        ("left", "wrist", "left_elbow", ["wrist_roll", "wrist_pitch", "wrist_yaw"]),
        ("left", "hand", "left_wrist", ["hand", "thumb", "index", "middle"]),
        (
            "right",
            "shoulder",
            "torso",
            ["shoulder_pitch", "shoulder_roll", "shoulder_yaw"],
        ),
        ("right", "elbow", "right_shoulder", ["elbow"]),
        ("right", "wrist", "right_elbow", ["wrist_roll", "wrist_pitch", "wrist_yaw"]),
        ("right", "hand", "right_wrist", ["hand", "thumb", "index", "middle"]),
    ]
    for side, role, parent, tokens in limb_specs:
        frame_id = f"{side}_{role}"
        linked = [
            joint
            for joint in joint_names
            if joint.startswith(f"{side}_") and any(token in joint for token in tokens)
        ]
        frames.append(
            LimbCoordinateFrame(
                frame_id=frame_id,
                parent_frame_id=parent,
                frame_role=role,
                side=side,
                linked_joint_names=linked,
                missing_evidence=missing,
            )
        )
    return frames


def _validate_frame_tree(
    *,
    chassis_id: str,
    frames: list[LimbCoordinateFrame],
    root_frame_id: str,
) -> HumanoidFrameTree:
    frame_ids = [frame.frame_id for frame in frames]
    frame_id_set = set(frame_ids)
    orphans = [
        frame.frame_id
        for frame in frames
        if frame.parent_frame_id and frame.parent_frame_id not in frame_id_set
    ]
    parent_by_child = {frame.frame_id: frame.parent_frame_id for frame in frames}
    cycle_detected = False
    for frame in frames:
        seen: set[str] = set()
        current = frame.frame_id
        while current:
            if current in seen:
                cycle_detected = True
                break
            seen.add(current)
            current = parent_by_child.get(current, "")
    missing_calibration = [
        frame.frame_id
        for frame in frames
        if frame.frame_id != root_frame_id and frame.calibration_status != "calibrated"
    ]
    status = "ok_contract_only" if not orphans and not cycle_detected else "blocked"
    return HumanoidFrameTree(
        tree_id=stable_id(
            "humanoid_frame_tree",
            {"chassis_id": chassis_id, "frame_ids": frame_ids},
        ),
        chassis_id=chassis_id,
        root_frame_id=root_frame_id,
        frame_count=len(frames),
        frame_ids=frame_ids,
        orphan_frame_ids=orphans,
        cycle_detected=cycle_detected,
        missing_calibration_frame_ids=missing_calibration,
        status=status,
    )


def _build_joint_limit_envelopes(
    chassis_id: str,
    profile: G1MorphologyProfile,
) -> list[JointLimitEnvelope]:
    envelopes: list[JointLimitEnvelope] = []
    for joint in profile.joint_specs:
        lower, upper, velocity, effort = _planning_limits(joint)
        envelopes.append(
            JointLimitEnvelope(
                envelope_id=stable_id(
                    "joint_limit_envelope",
                    {"chassis_id": chassis_id, "joint_name": joint.joint_name},
                ),
                chassis_id=chassis_id,
                joint_name=joint.joint_name,
                joint_group=joint.group,
                lower_rad=lower,
                upper_rad=upper,
                velocity_limit_rad_s=velocity,
                effort_limit_nm=effort,
                hardware_limit_verified=False,
                missing_evidence=["hardware_joint_limit_validation_missing"],
            )
        )
    return envelopes


def _support_states(chassis_id: str) -> list[BipedalSupportState]:
    missing = [
        "measured_foot_contact_stream_missing",
        "measured_com_zmp_cop_missing",
        "balance_benchmark_evidence_missing",
    ]
    return [
        BipedalSupportState(
            support_state_id=f"{chassis_id}_double_support_slot",
            chassis_id=chassis_id,
            support_phase="double_support",
            support_contact_frames=["left_foot", "right_foot"],
            swing_contact_frames=[],
            support_polygon_vertices=[
                {"x": -0.10, "y": 0.08},
                {"x": 0.10, "y": 0.08},
                {"x": 0.10, "y": -0.08},
                {"x": -0.10, "y": -0.08},
            ],
            com_projection_xy={"x": 0.0, "y": 0.0},
            zmp_xy={},
            cop_xy={},
            balance_margin_m=None,
            slip_risk=None,
            fall_risk=None,
            missing_evidence=missing,
        ),
        BipedalSupportState(
            support_state_id=f"{chassis_id}_left_support_slot",
            chassis_id=chassis_id,
            support_phase="left_single_support",
            support_contact_frames=["left_foot"],
            swing_contact_frames=["right_foot"],
            support_polygon_vertices=[
                {"x": -0.10, "y": 0.08},
                {"x": 0.10, "y": 0.08},
                {"x": 0.10, "y": -0.02},
                {"x": -0.10, "y": -0.02},
            ],
            com_projection_xy={},
            zmp_xy={},
            cop_xy={},
            missing_evidence=missing,
        ),
        BipedalSupportState(
            support_state_id=f"{chassis_id}_right_support_slot",
            chassis_id=chassis_id,
            support_phase="right_single_support",
            support_contact_frames=["right_foot"],
            swing_contact_frames=["left_foot"],
            support_polygon_vertices=[
                {"x": -0.10, "y": 0.02},
                {"x": 0.10, "y": 0.02},
                {"x": 0.10, "y": -0.08},
                {"x": -0.10, "y": -0.08},
            ],
            com_projection_xy={},
            zmp_xy={},
            cop_xy={},
            missing_evidence=missing,
        ),
    ]


def build_bipedal_chassis_scaffold(
    *,
    variant: str = "g1_29dof",
    artifact_refs: Optional[Mapping[str, Any]] = None,
    source_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    BipedalChassisScaffoldReport,
    HumanoidChassisProfile,
    HumanoidFrameTree,
    list[LimbCoordinateFrame],
    list[JointLimitEnvelope],
    WholeBodyObservationSchema,
    WholeBodyActionSchema,
    list[BipedalSupportState],
    list[BalanceEnvelopeReceipt],
]:
    """Build the local bipedal chassis scaffold for a G1/R1-class target."""

    profile = build_g1_morphology_profile(
        variant=variant,
        source_refs=source_refs,
    )
    joint_names = list(G1_VARIANT_JOINTS.get(variant, profile.joint_names()))
    chassis_id = stable_id(
        "humanoid_chassis",
        {
            "variant": variant,
            "joint_names": joint_names,
            "morphology_profile_id": profile.profile_id,
        },
    )
    chassis = HumanoidChassisProfile(
        chassis_id=chassis_id,
        robot_family="unitree_g1_r1_class",
        variant=variant,
        posture_tag="bipedal_whole_body",
        morphology_profile_id=profile.profile_id,
        controlled_joint_count=len(joint_names),
        floating_base_dof=6,
        minimum_total_dof=6 + len(joint_names),
        joint_names=joint_names,
        limb_groups=_joint_limb_groups(joint_names),
        imu_frame_ids=["imu_pelvis"],
        perception_frame_ids=["head_camera"],
        source_refs=profile.source_refs,
        missing_evidence=list(BIPEDAL_CHASSIS_REMAINING_BLOCKERS),
    )
    frames = _build_frame_specs(joint_names)
    frame_tree = _validate_frame_tree(
        chassis_id=chassis_id,
        frames=frames,
        root_frame_id=chassis.root_frame_id,
    )
    joint_limits = _build_joint_limit_envelopes(chassis_id, profile)
    observation_schema = WholeBodyObservationSchema(
        schema_id=stable_id(
            "whole_body_observation_schema",
            {"chassis_id": chassis_id, "variant": variant},
        ),
        chassis_id=chassis_id,
        posture_tag="bipedal_whole_body",
        channel_groups={
            "floating_base": [
                "base_position_xyz",
                "base_orientation_quat",
                "base_linear_velocity",
                "base_angular_velocity",
            ],
            "joint_state": [
                "joint_position",
                "joint_velocity",
                "joint_effort",
                "joint_temperature_slot",
            ],
            "imu": ["orientation", "angular_rate", "linear_acceleration"],
            "contact": [
                "left_foot_contact",
                "right_foot_contact",
                "hand_contact_slots",
                "contact_normal_slots",
            ],
            "balance": [
                "support_phase",
                "support_polygon",
                "com_projection",
                "zmp_slot",
                "cop_slot",
                "balance_margin",
            ],
            "egocentric_perception": [
                "head_camera_ref",
                "depth_ref",
                "calibration_ref",
                "body_relative_scene_ref",
            ],
            "resource_timing": [
                "compute_placement",
                "latency_ms",
                "battery_reserve",
                "thermal_headroom",
                "comms_qos",
            ],
        },
        frame_tree_ref=frame_tree.tree_id,
        joint_limit_envelope_ref=f"{chassis_id}_joint_limit_envelopes_v1",
        replay_training_awareness=[
            "posture_tag_required",
            "event_spine_ref_required",
            "governance_trace_ref_required",
            "hardware_calibration_truth_required_before_promotion",
        ],
    )
    action_schema = WholeBodyActionSchema(
        schema_id=stable_id(
            "whole_body_action_schema",
            {"chassis_id": chassis_id, "variant": variant, "dim": len(joint_names)},
        ),
        chassis_id=chassis_id,
        posture_tag="bipedal_whole_body",
        action_dimension=len(joint_names),
        action_channels=joint_names,
        horizon_steps=4,
        support_phase_constraints=[
            "do_not_command_swing_foot_as_support_without_support_phase_receipt",
            "joint_limit_envelope_must_be_checked_before_runtime_authority",
            "stable_base_fallback_required_when_balance_evidence_missing",
        ],
        joint_limit_envelope_ref=f"{chassis_id}_joint_limit_envelopes_v1",
    )
    support_states = _support_states(chassis_id)
    balance_receipts = [
        BalanceEnvelopeReceipt(
            receipt_id=stable_id(
                "balance_envelope_receipt",
                {
                    "chassis_id": chassis_id,
                    "support_state_id": support.support_state_id,
                },
            ),
            chassis_id=chassis_id,
            support_state_id=support.support_state_id,
            status="schema_slot_only",
            balance_margin_class="not_measured",
            missing_evidence=list(BIPEDAL_CHASSIS_REMAINING_BLOCKERS),
        )
        for support in support_states
    ]
    complete = (
        chassis.controlled_joint_count >= 21
        and frame_tree.status == "ok_contract_only"
        and len(joint_limits) == chassis.controlled_joint_count
        and bool(observation_schema.channel_groups)
        and action_schema.action_dimension == chassis.controlled_joint_count
        and len(support_states) >= 3
        and len(balance_receipts) == len(support_states)
    )
    report_payload = {
        "chassis_id": chassis_id,
        "morphology_profile_id": profile.profile_id,
        "controlled_joint_count": chassis.controlled_joint_count,
        "frame_count": frame_tree.frame_count,
        "joint_limit_envelope_count": len(joint_limits),
        "support_state_count": len(support_states),
    }
    report = BipedalChassisScaffoldReport(
        report_id=stable_id("bipedal_chassis_scaffold", report_payload),
        status="ok" if complete else "blocked",
        chassis_id=chassis_id,
        morphology_profile_id=profile.profile_id,
        controlled_joint_count=chassis.controlled_joint_count,
        frame_count=frame_tree.frame_count,
        joint_limit_envelope_count=len(joint_limits),
        support_state_count=len(support_states),
        balance_receipt_count=len(balance_receipts),
        canonical_bipedal_chassis_present=True,
        limb_frame_tree_present=frame_tree.status == "ok_contract_only",
        joint_limit_envelope_present=len(joint_limits)
        == chassis.controlled_joint_count,
        whole_body_observation_schema_present=bool(observation_schema.channel_groups),
        whole_body_action_schema_present=(
            action_schema.action_dimension == chassis.controlled_joint_count
        ),
        balance_envelope_present=len(balance_receipts) == len(support_states),
        local_structural_scaffold_complete=complete,
        remaining_blockers=list(BIPEDAL_CHASSIS_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return (
        report,
        chassis,
        frame_tree,
        frames,
        joint_limits,
        observation_schema,
        action_schema,
        support_states,
        balance_receipts,
    )


def save_bipedal_chassis_scaffold(
    output_dir: str | Path,
    *,
    report: BipedalChassisScaffoldReport,
    chassis: HumanoidChassisProfile,
    frame_tree: HumanoidFrameTree,
    frames: list[LimbCoordinateFrame],
    joint_limits: list[JointLimitEnvelope],
    observation_schema: WholeBodyObservationSchema,
    action_schema: WholeBodyActionSchema,
    support_states: list[BipedalSupportState],
    balance_receipts: list[BalanceEnvelopeReceipt],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "bipedal_chassis_scaffold_report_v1.json",
        "chassis_profile_path": output / "humanoid_chassis_profile_v1.json",
        "frame_tree_path": output / "humanoid_frame_tree_v1.json",
        "frames_path": output / "limb_coordinate_frames_v1.jsonl",
        "joint_limits_path": output / "joint_limit_envelopes_v1.jsonl",
        "observation_schema_path": output / "whole_body_observation_schema_v1.json",
        "action_schema_path": output / "whole_body_action_schema_v1.json",
        "support_states_path": output / "bipedal_support_states_v1.jsonl",
        "balance_receipts_path": output / "balance_envelope_receipts_v1.jsonl",
    }
    _write_json(paths["report_path"], report.to_dict())
    _write_json(paths["chassis_profile_path"], chassis.to_dict())
    _write_json(paths["frame_tree_path"], frame_tree.to_dict())
    _write_jsonl(paths["frames_path"], [frame.to_dict() for frame in frames])
    _write_jsonl(paths["joint_limits_path"], [item.to_dict() for item in joint_limits])
    _write_json(paths["observation_schema_path"], observation_schema.to_dict())
    _write_json(paths["action_schema_path"], action_schema.to_dict())
    _write_jsonl(
        paths["support_states_path"], [item.to_dict() for item in support_states]
    )
    _write_jsonl(
        paths["balance_receipts_path"], [item.to_dict() for item in balance_receipts]
    )
    return {key: str(path) for key, path in paths.items()}


def load_bipedal_chassis_scaffold_report(
    path: str | Path,
) -> BipedalChassisScaffoldReport:
    return BipedalChassisScaffoldReport.from_dict(_load_json(path))


def load_humanoid_chassis_profile(path: str | Path) -> HumanoidChassisProfile:
    return HumanoidChassisProfile.from_dict(_load_json(path))


def load_humanoid_frame_tree(path: str | Path) -> HumanoidFrameTree:
    return HumanoidFrameTree.from_dict(_load_json(path))


def load_limb_coordinate_frames(path: str | Path) -> list[LimbCoordinateFrame]:
    return [LimbCoordinateFrame.from_dict(row) for row in _load_jsonl(path)]


def load_joint_limit_envelopes(path: str | Path) -> list[JointLimitEnvelope]:
    return [JointLimitEnvelope.from_dict(row) for row in _load_jsonl(path)]


def load_whole_body_observation_schema(
    path: str | Path,
) -> WholeBodyObservationSchema:
    return WholeBodyObservationSchema.from_dict(_load_json(path))


def load_whole_body_action_schema(path: str | Path) -> WholeBodyActionSchema:
    return WholeBodyActionSchema.from_dict(_load_json(path))


def load_bipedal_support_states(path: str | Path) -> list[BipedalSupportState]:
    return [BipedalSupportState.from_dict(row) for row in _load_jsonl(path)]


def load_balance_envelope_receipts(path: str | Path) -> list[BalanceEnvelopeReceipt]:
    return [BalanceEnvelopeReceipt.from_dict(row) for row in _load_jsonl(path)]
