"""Phase 4 downstream controller scaffold for dry-run humanoid dispatch.

This module creates a local controller primitive that can sit below WM action
proposals without acquiring live authority. It borrows interface shape from
Unitree ROS2 / SDK2 and G1Pilot-style upper-body fallback control, but it does
not vendor OSS code, publish ROS2/DDS messages, write Unitree commands, execute
hardware or sim, train weights, or promote a controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from src.world_model.embodiment_actuation import (
    HumanoidChassisProfile,
    JointLimitEnvelope,
)
from src.world_model.embodiment_actuation.bipedal_readiness import (
    Phase35BipedalReadinessAudit,
    WholeBodyReplayRow,
)
from src.world_model.humanoid_readiness.common import (
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase4 import (
    Phase4DeploymentEnablerSweepReport,
)

PHASE4_DOWNSTREAM_CONTROLLER_REPORT_VERSION = (
    "phase4_downstream_controller_scaffold_report_v1"
)
CONTROLLER_BRIDGE_TARGET_VERSION = "phase4_controller_bridge_target_v1"
CONTROLLER_MODE_SPEC_VERSION = "phase4_controller_mode_spec_v1"
DOWNSTREAM_CONTROLLER_PROPOSAL_VERSION = "phase4_downstream_controller_proposal_v1"
LOW_LEVEL_COMMAND_FRAME_VERSION = "phase4_low_level_command_frame_v1"
CONTROLLER_INVOCATION_VERSION = "phase4_controller_invocation_v1"
CONTROLLER_SAFETY_RECEIPT_VERSION = "phase4_controller_safety_receipt_v1"
CONTROLLER_RECEIPT_VERSION = "phase4_controller_receipt_v1"

DENIED_DOWNSTREAM_CONTROLLER_AUTHORITIES = (
    "hardware_dispatch_enabled",
    "ros2_publish_attempted",
    "unitree_sdk2_write_enabled",
    "unitree_ros2_lowcmd_publish_enabled",
    "g1pilot_runtime_invoked",
    "live_policy_control",
    "training_executed",
    "weights_written",
    "provider_executed",
    "hardware_executed",
    "unitree_sim_runtime_executed",
    "reward_math_mutation",
    "promotion_eligible",
)

PHASE4_DOWNSTREAM_CONTROLLER_BLOCKERS = (
    "unitree_ros2_or_sdk2_runtime_not_installed_or_verified",
    "g1pilot_or_equivalent_runtime_not_vendored_or_verified",
    "real_robot_description_and_calibration_missing",
    "live_low_state_stream_missing",
    "actual_lowcmd_or_sport_request_interface_not_validated",
    "measured_control_loop_timing_and_jitter_missing",
    "physical_safety_calibration_missing",
    "operator_estop_and_recovery_path_not_verified",
    "hardware_or_honest_sim_runtime_evidence_missing",
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_DOWNSTREAM_CONTROLLER_AUTHORITIES}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _float_mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, float]:
    return {str(key): _safe_float(value) for key, value in dict(payload or {}).items()}


@dataclass(frozen=True)
class ControllerBridgeTarget:
    target_id: str
    target_name: str
    source_project: str
    source_url: str
    source_license: str
    runtime_target: str
    transport_profile: str
    command_family: str
    publish_topic: str = ""
    subscribe_topics: list[str] = field(default_factory=list)
    message_type_refs: list[str] = field(default_factory=list)
    authority_scope: str = "dry_run_contract_only"
    external_dependency_status: str = "external_not_vendored"
    vendored_code_included: bool = False
    hardware_dispatch_enabled: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    authority_class: str = "controller_bridge_target_contract_only"
    version: str = CONTROLLER_BRIDGE_TARGET_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "version": self.version,
            "target_name": self.target_name,
            "source_project": self.source_project,
            "source_url": self.source_url,
            "source_license": self.source_license,
            "runtime_target": self.runtime_target,
            "transport_profile": self.transport_profile,
            "command_family": self.command_family,
            "publish_topic": self.publish_topic,
            "subscribe_topics": list(self.subscribe_topics),
            "message_type_refs": list(self.message_type_refs),
            "authority_scope": self.authority_scope,
            "external_dependency_status": self.external_dependency_status,
            "vendored_code_included": bool(self.vendored_code_included),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerBridgeTarget":
        return cls(
            target_id=str(payload.get("target_id", "")),
            target_name=str(payload.get("target_name", "")),
            source_project=str(payload.get("source_project", "")),
            source_url=str(payload.get("source_url", "")),
            source_license=str(payload.get("source_license", "unknown")),
            runtime_target=str(payload.get("runtime_target", "")),
            transport_profile=str(payload.get("transport_profile", "")),
            command_family=str(payload.get("command_family", "")),
            publish_topic=str(payload.get("publish_topic", "")),
            subscribe_topics=strings(payload.get("subscribe_topics")),
            message_type_refs=strings(payload.get("message_type_refs")),
            authority_scope=str(
                payload.get("authority_scope", "dry_run_contract_only")
            ),
            external_dependency_status=str(
                payload.get("external_dependency_status", "external_not_vendored")
            ),
            vendored_code_included=bool(payload.get("vendored_code_included", False)),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            authority_class=str(
                payload.get(
                    "authority_class", "controller_bridge_target_contract_only"
                )
            ),
            version=str(payload.get("version", CONTROLLER_BRIDGE_TARGET_VERSION)),
        )


@dataclass(frozen=True)
class ControllerModeSpec:
    mode_id: str
    mode_name: str
    posture_tag: str
    bridge_target_id: str
    command_kind: str
    planned_rate_hz: float
    placement_class: str
    safety_gates: list[str] = field(default_factory=list)
    fallback_mode: str = ""
    oss_inspiration_refs: list[str] = field(default_factory=list)
    training_aware: bool = True
    dry_run_only: bool = True
    live_authority_allowed: bool = False
    promotion_eligible: bool = False
    authority_class: str = "controller_mode_contract_only"
    version: str = CONTROLLER_MODE_SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "version": self.version,
            "mode_name": self.mode_name,
            "posture_tag": self.posture_tag,
            "bridge_target_id": self.bridge_target_id,
            "command_kind": self.command_kind,
            "planned_rate_hz": _safe_float(self.planned_rate_hz),
            "placement_class": self.placement_class,
            "safety_gates": list(self.safety_gates),
            "fallback_mode": self.fallback_mode,
            "oss_inspiration_refs": list(self.oss_inspiration_refs),
            "training_aware": bool(self.training_aware),
            "dry_run_only": bool(self.dry_run_only),
            "live_authority_allowed": bool(self.live_authority_allowed),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerModeSpec":
        return cls(
            mode_id=str(payload.get("mode_id", "")),
            mode_name=str(payload.get("mode_name", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            bridge_target_id=str(payload.get("bridge_target_id", "")),
            command_kind=str(payload.get("command_kind", "")),
            planned_rate_hz=_safe_float(payload.get("planned_rate_hz")),
            placement_class=str(payload.get("placement_class", "")),
            safety_gates=strings(payload.get("safety_gates")),
            fallback_mode=str(payload.get("fallback_mode", "")),
            oss_inspiration_refs=strings(payload.get("oss_inspiration_refs")),
            training_aware=bool(payload.get("training_aware", True)),
            dry_run_only=bool(payload.get("dry_run_only", True)),
            live_authority_allowed=bool(payload.get("live_authority_allowed", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get("authority_class", "controller_mode_contract_only")
            ),
            version=str(payload.get("version", CONTROLLER_MODE_SPEC_VERSION)),
        )


@dataclass(frozen=True)
class DownstreamControllerProposal:
    proposal_id: str
    proposal_name: str
    mode_name: str
    posture_tag: str
    source_loop: str
    support_phase: str
    bridge_target_id: str
    requested_joint_positions: dict[str, float] = field(default_factory=dict)
    requested_joint_velocities: dict[str, float] = field(default_factory=dict)
    requested_cartesian_targets: dict[str, Any] = field(default_factory=dict)
    requested_command_payload: dict[str, Any] = field(default_factory=dict)
    source_replay_row_id: str = ""
    observation_schema_ref: str = ""
    action_schema_ref: str = ""
    operator_override_required: bool = True
    e_stop_requested: bool = False
    dry_run_only: bool = True
    hardware_dispatch_allowed: bool = False
    authority_class: str = "downstream_controller_proposal_only"
    version: str = DOWNSTREAM_CONTROLLER_PROPOSAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "version": self.version,
            "proposal_name": self.proposal_name,
            "mode_name": self.mode_name,
            "posture_tag": self.posture_tag,
            "source_loop": self.source_loop,
            "support_phase": self.support_phase,
            "bridge_target_id": self.bridge_target_id,
            "requested_joint_positions": dict(self.requested_joint_positions),
            "requested_joint_velocities": dict(self.requested_joint_velocities),
            "requested_cartesian_targets": mapping(self.requested_cartesian_targets),
            "requested_command_payload": mapping(self.requested_command_payload),
            "source_replay_row_id": self.source_replay_row_id,
            "observation_schema_ref": self.observation_schema_ref,
            "action_schema_ref": self.action_schema_ref,
            "operator_override_required": bool(self.operator_override_required),
            "e_stop_requested": bool(self.e_stop_requested),
            "dry_run_only": bool(self.dry_run_only),
            "hardware_dispatch_allowed": bool(self.hardware_dispatch_allowed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DownstreamControllerProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", "")),
            proposal_name=str(payload.get("proposal_name", "")),
            mode_name=str(payload.get("mode_name", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            source_loop=str(payload.get("source_loop", "")),
            support_phase=str(payload.get("support_phase", "")),
            bridge_target_id=str(payload.get("bridge_target_id", "")),
            requested_joint_positions=_float_mapping(
                payload.get("requested_joint_positions")
            ),
            requested_joint_velocities=_float_mapping(
                payload.get("requested_joint_velocities")
            ),
            requested_cartesian_targets=mapping(
                payload.get("requested_cartesian_targets")
            ),
            requested_command_payload=mapping(
                payload.get("requested_command_payload")
            ),
            source_replay_row_id=str(payload.get("source_replay_row_id", "")),
            observation_schema_ref=str(payload.get("observation_schema_ref", "")),
            action_schema_ref=str(payload.get("action_schema_ref", "")),
            operator_override_required=bool(
                payload.get("operator_override_required", True)
            ),
            e_stop_requested=bool(payload.get("e_stop_requested", False)),
            dry_run_only=bool(payload.get("dry_run_only", True)),
            hardware_dispatch_allowed=bool(
                payload.get("hardware_dispatch_allowed", False)
            ),
            authority_class=str(
                payload.get(
                    "authority_class", "downstream_controller_proposal_only"
                )
            ),
            version=str(
                payload.get("version", DOWNSTREAM_CONTROLLER_PROPOSAL_VERSION)
            ),
        )


@dataclass(frozen=True)
class LowLevelCommandFrame:
    frame_id: str
    proposal_id: str
    mode_name: str
    bridge_target_id: str
    command_kind: str
    planned_rate_hz: float
    channel_names: list[str] = field(default_factory=list)
    target_joint_positions: dict[str, float] = field(default_factory=dict)
    target_joint_velocities: dict[str, float] = field(default_factory=dict)
    target_kp: dict[str, float] = field(default_factory=dict)
    target_kd: dict[str, float] = field(default_factory=dict)
    feedforward_torque: dict[str, float] = field(default_factory=dict)
    command_payload: dict[str, Any] = field(default_factory=dict)
    bridge_topic: str = ""
    ros2_message_type: str = ""
    unitree_command_family: str = ""
    clamp_applied: bool = False
    clamped_joint_names: list[str] = field(default_factory=list)
    dry_run_only: bool = True
    publish_attempted: bool = False
    hardware_dispatch_enabled: bool = False
    authority_class: str = "low_level_command_frame_dry_run_only"
    version: str = LOW_LEVEL_COMMAND_FRAME_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "mode_name": self.mode_name,
            "bridge_target_id": self.bridge_target_id,
            "command_kind": self.command_kind,
            "planned_rate_hz": _safe_float(self.planned_rate_hz),
            "channel_names": list(self.channel_names),
            "target_joint_positions": dict(self.target_joint_positions),
            "target_joint_velocities": dict(self.target_joint_velocities),
            "target_kp": dict(self.target_kp),
            "target_kd": dict(self.target_kd),
            "feedforward_torque": dict(self.feedforward_torque),
            "command_payload": mapping(self.command_payload),
            "bridge_topic": self.bridge_topic,
            "ros2_message_type": self.ros2_message_type,
            "unitree_command_family": self.unitree_command_family,
            "clamp_applied": bool(self.clamp_applied),
            "clamped_joint_names": list(self.clamped_joint_names),
            "dry_run_only": bool(self.dry_run_only),
            "publish_attempted": bool(self.publish_attempted),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LowLevelCommandFrame":
        return cls(
            frame_id=str(payload.get("frame_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            mode_name=str(payload.get("mode_name", "")),
            bridge_target_id=str(payload.get("bridge_target_id", "")),
            command_kind=str(payload.get("command_kind", "")),
            planned_rate_hz=_safe_float(payload.get("planned_rate_hz")),
            channel_names=strings(payload.get("channel_names")),
            target_joint_positions=_float_mapping(
                payload.get("target_joint_positions")
            ),
            target_joint_velocities=_float_mapping(
                payload.get("target_joint_velocities")
            ),
            target_kp=_float_mapping(payload.get("target_kp")),
            target_kd=_float_mapping(payload.get("target_kd")),
            feedforward_torque=_float_mapping(payload.get("feedforward_torque")),
            command_payload=mapping(payload.get("command_payload")),
            bridge_topic=str(payload.get("bridge_topic", "")),
            ros2_message_type=str(payload.get("ros2_message_type", "")),
            unitree_command_family=str(payload.get("unitree_command_family", "")),
            clamp_applied=bool(payload.get("clamp_applied", False)),
            clamped_joint_names=strings(payload.get("clamped_joint_names")),
            dry_run_only=bool(payload.get("dry_run_only", True)),
            publish_attempted=bool(payload.get("publish_attempted", False)),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            authority_class=str(
                payload.get("authority_class", "low_level_command_frame_dry_run_only")
            ),
            version=str(payload.get("version", LOW_LEVEL_COMMAND_FRAME_VERSION)),
        )


@dataclass(frozen=True)
class ControllerSafetyReceipt:
    receipt_id: str
    proposal_id: str
    command_frame_id: str
    status: str
    joint_limit_clamp_applied: bool
    clamped_joint_names: list[str] = field(default_factory=list)
    rate_limit_checked: bool = True
    stale_data_vetoed: bool = True
    support_phase_verified: bool = False
    support_phase_constraint_satisfied: bool = False
    operator_override_required: bool = True
    e_stop_vetoed: bool = False
    live_safety_calibration_present: bool = False
    hardware_dispatch_allowed: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "controller_safety_receipt_veto_only"
    version: str = CONTROLLER_SAFETY_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "command_frame_id": self.command_frame_id,
            "status": self.status,
            "joint_limit_clamp_applied": bool(self.joint_limit_clamp_applied),
            "clamped_joint_names": list(self.clamped_joint_names),
            "rate_limit_checked": bool(self.rate_limit_checked),
            "stale_data_vetoed": bool(self.stale_data_vetoed),
            "support_phase_verified": bool(self.support_phase_verified),
            "support_phase_constraint_satisfied": bool(
                self.support_phase_constraint_satisfied
            ),
            "operator_override_required": bool(self.operator_override_required),
            "e_stop_vetoed": bool(self.e_stop_vetoed),
            "live_safety_calibration_present": bool(
                self.live_safety_calibration_present
            ),
            "hardware_dispatch_allowed": bool(self.hardware_dispatch_allowed),
            "missing_evidence": list(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerSafetyReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            command_frame_id=str(payload.get("command_frame_id", "")),
            status=str(payload.get("status", "")),
            joint_limit_clamp_applied=bool(
                payload.get("joint_limit_clamp_applied", False)
            ),
            clamped_joint_names=strings(payload.get("clamped_joint_names")),
            rate_limit_checked=bool(payload.get("rate_limit_checked", True)),
            stale_data_vetoed=bool(payload.get("stale_data_vetoed", True)),
            support_phase_verified=bool(payload.get("support_phase_verified", False)),
            support_phase_constraint_satisfied=bool(
                payload.get("support_phase_constraint_satisfied", False)
            ),
            operator_override_required=bool(
                payload.get("operator_override_required", True)
            ),
            e_stop_vetoed=bool(payload.get("e_stop_vetoed", False)),
            live_safety_calibration_present=bool(
                payload.get("live_safety_calibration_present", False)
            ),
            hardware_dispatch_allowed=bool(
                payload.get("hardware_dispatch_allowed", False)
            ),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "controller_safety_receipt_veto_only")
            ),
            version=str(payload.get("version", CONTROLLER_SAFETY_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class ControllerInvocation:
    invocation_id: str
    proposal_id: str
    command_frame_id: str
    safety_receipt_id: str
    bridge_target_id: str
    mode_name: str
    dispatch_status: str
    dispatch_denial_reasons: list[str] = field(default_factory=list)
    planned_rate_hz: float = 0.0
    placement_class: str = "companion_or_local_dry_run"
    publish_attempted: bool = False
    hardware_dispatch_enabled: bool = False
    ros2_publish_enabled: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    dry_run_only: bool = True
    authority_class: str = "controller_invocation_dispatch_denied"
    version: str = CONTROLLER_INVOCATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "command_frame_id": self.command_frame_id,
            "safety_receipt_id": self.safety_receipt_id,
            "bridge_target_id": self.bridge_target_id,
            "mode_name": self.mode_name,
            "dispatch_status": self.dispatch_status,
            "dispatch_denial_reasons": list(self.dispatch_denial_reasons),
            "planned_rate_hz": _safe_float(self.planned_rate_hz),
            "placement_class": self.placement_class,
            "publish_attempted": bool(self.publish_attempted),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "ros2_publish_enabled": bool(self.ros2_publish_enabled),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "dry_run_only": bool(self.dry_run_only),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerInvocation":
        return cls(
            invocation_id=str(payload.get("invocation_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            command_frame_id=str(payload.get("command_frame_id", "")),
            safety_receipt_id=str(payload.get("safety_receipt_id", "")),
            bridge_target_id=str(payload.get("bridge_target_id", "")),
            mode_name=str(payload.get("mode_name", "")),
            dispatch_status=str(payload.get("dispatch_status", "")),
            dispatch_denial_reasons=strings(payload.get("dispatch_denial_reasons")),
            planned_rate_hz=_safe_float(payload.get("planned_rate_hz")),
            placement_class=str(
                payload.get("placement_class", "companion_or_local_dry_run")
            ),
            publish_attempted=bool(payload.get("publish_attempted", False)),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            ros2_publish_enabled=bool(payload.get("ros2_publish_enabled", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            g1pilot_runtime_invoked=bool(payload.get("g1pilot_runtime_invoked", False)),
            dry_run_only=bool(payload.get("dry_run_only", True)),
            authority_class=str(
                payload.get("authority_class", "controller_invocation_dispatch_denied")
            ),
            version=str(payload.get("version", CONTROLLER_INVOCATION_VERSION)),
        )


@dataclass(frozen=True)
class ControllerReceipt:
    receipt_id: str
    invocation_id: str
    proposal_id: str
    command_frame_id: str
    safety_receipt_id: str
    status: str
    command_frame_emitted: bool
    safety_receipt_emitted: bool
    replay_export_ready: bool
    training_aware: bool
    hardware_dispatch_enabled: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    live_policy_control: bool = False
    promotion_eligible: bool = False
    authority_class: str = "controller_receipt_observational_only"
    version: str = CONTROLLER_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "invocation_id": self.invocation_id,
            "proposal_id": self.proposal_id,
            "command_frame_id": self.command_frame_id,
            "safety_receipt_id": self.safety_receipt_id,
            "status": self.status,
            "command_frame_emitted": bool(self.command_frame_emitted),
            "safety_receipt_emitted": bool(self.safety_receipt_emitted),
            "replay_export_ready": bool(self.replay_export_ready),
            "training_aware": bool(self.training_aware),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "live_policy_control": bool(self.live_policy_control),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            invocation_id=str(payload.get("invocation_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            command_frame_id=str(payload.get("command_frame_id", "")),
            safety_receipt_id=str(payload.get("safety_receipt_id", "")),
            status=str(payload.get("status", "")),
            command_frame_emitted=bool(payload.get("command_frame_emitted", False)),
            safety_receipt_emitted=bool(payload.get("safety_receipt_emitted", False)),
            replay_export_ready=bool(payload.get("replay_export_ready", False)),
            training_aware=bool(payload.get("training_aware", False)),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get("authority_class", "controller_receipt_observational_only")
            ),
            version=str(payload.get("version", CONTROLLER_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class Phase4DownstreamControllerScaffoldReport:
    report_id: str
    phase4_report_id: str
    phase35_bipedal_readiness_audit_id: str
    chassis_id: str
    status: str
    bridge_target_count: int
    mode_count: int
    proposal_count: int
    command_frame_count: int
    safety_receipt_count: int
    invocation_count: int
    controller_receipt_count: int
    local_downstream_controller_scaffold_complete: bool
    unitree_bridge_contract_present: bool
    g1pilot_fallback_contract_present: bool
    dry_run_controller_present: bool
    hardware_dispatch_enabled: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    live_policy_control: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    key_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE4_DOWNSTREAM_CONTROLLER_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase4_report_id": self.phase4_report_id,
            "phase35_bipedal_readiness_audit_id": (
                self.phase35_bipedal_readiness_audit_id
            ),
            "chassis_id": self.chassis_id,
            "status": self.status,
            "bridge_target_count": int(self.bridge_target_count),
            "mode_count": int(self.mode_count),
            "proposal_count": int(self.proposal_count),
            "command_frame_count": int(self.command_frame_count),
            "safety_receipt_count": int(self.safety_receipt_count),
            "invocation_count": int(self.invocation_count),
            "controller_receipt_count": int(self.controller_receipt_count),
            "local_downstream_controller_scaffold_complete": bool(
                self.local_downstream_controller_scaffold_complete
            ),
            "unitree_bridge_contract_present": bool(
                self.unitree_bridge_contract_present
            ),
            "g1pilot_fallback_contract_present": bool(
                self.g1pilot_fallback_contract_present
            ),
            "dry_run_controller_present": bool(self.dry_run_controller_present),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "live_policy_control": bool(self.live_policy_control),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": dict(self.denied_gates),
            "key_blockers": list(self.key_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase4DownstreamControllerScaffoldReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase4_report_id=str(payload.get("phase4_report_id", "")),
            phase35_bipedal_readiness_audit_id=str(
                payload.get("phase35_bipedal_readiness_audit_id", "")
            ),
            chassis_id=str(payload.get("chassis_id", "")),
            status=str(payload.get("status", "blocked")),
            bridge_target_count=int(payload.get("bridge_target_count", 0) or 0),
            mode_count=int(payload.get("mode_count", 0) or 0),
            proposal_count=int(payload.get("proposal_count", 0) or 0),
            command_frame_count=int(payload.get("command_frame_count", 0) or 0),
            safety_receipt_count=int(payload.get("safety_receipt_count", 0) or 0),
            invocation_count=int(payload.get("invocation_count", 0) or 0),
            controller_receipt_count=int(
                payload.get("controller_receipt_count", 0) or 0
            ),
            local_downstream_controller_scaffold_complete=bool(
                payload.get("local_downstream_controller_scaffold_complete", False)
            ),
            unitree_bridge_contract_present=bool(
                payload.get("unitree_bridge_contract_present", False)
            ),
            g1pilot_fallback_contract_present=bool(
                payload.get("g1pilot_fallback_contract_present", False)
            ),
            dry_run_controller_present=bool(
                payload.get("dry_run_controller_present", False)
            ),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            g1pilot_runtime_invoked=bool(payload.get("g1pilot_runtime_invoked", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(payload.get("denied_gates", {}) or {}).items()
                },
            },
            key_blockers=strings(payload.get("key_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE4_DOWNSTREAM_CONTROLLER_REPORT_VERSION)
            ),
        )


def build_controller_bridge_targets() -> list[ControllerBridgeTarget]:
    specs = [
        {
            "target_name": "unitree_ros2_lowcmd_joint_pd",
            "source_project": "unitreerobotics/unitree_ros2",
            "source_url": "https://github.com/unitreerobotics/unitree_ros2",
            "source_license": "BSD-3-Clause",
            "runtime_target": "Unitree SDK2 / ROS2 DDS low-level command lane",
            "transport_profile": "ros2_dds_cyclonedds_external",
            "command_family": "low_level_joint_pd",
            "publish_topic": "/lowcmd",
            "subscribe_topics": ["/lowstate", "/wirelesscontroller"],
            "message_type_refs": ["LowCmd", "LowState"],
        },
        {
            "target_name": "unitree_ros2_sport_request_fallback",
            "source_project": "unitreerobotics/unitree_ros2",
            "source_url": "https://github.com/unitreerobotics/unitree_ros2",
            "source_license": "BSD-3-Clause",
            "runtime_target": "Unitree SDK2 / ROS2 high-level sport request lane",
            "transport_profile": "ros2_dds_cyclonedds_external",
            "command_family": "sport_request_degraded_mode",
            "publish_topic": "/api/sport/request",
            "subscribe_topics": ["/sportmodestate", "/wirelesscontroller"],
            "message_type_refs": ["unitree_api::msg::Request"],
        },
        {
            "target_name": "g1pilot_upper_body_joint_fallback",
            "source_project": "hucebot/g1pilot",
            "source_url": "https://github.com/hucebot/g1pilot",
            "source_license": "BSD-3-Clause",
            "runtime_target": "G1Pilot-style upper-body joint control fallback",
            "transport_profile": "ros2_humble_external",
            "command_family": "upper_body_joint_tracking",
            "publish_topic": "g1pilot_joint_command_ref_required",
            "subscribe_topics": ["imu", "odometry", "motor_feedback"],
            "message_type_refs": ["joint_controller_contract"],
        },
        {
            "target_name": "g1pilot_cartesian_upper_body_fallback",
            "source_project": "hucebot/g1pilot",
            "source_url": "https://github.com/hucebot/g1pilot",
            "source_license": "BSD-3-Clause",
            "runtime_target": "G1Pilot-style Cartesian end-effector fallback",
            "transport_profile": "ros2_humble_external",
            "command_family": "cartesian_upper_body_tracking",
            "publish_topic": "g1pilot_cartesian_command_ref_required",
            "subscribe_topics": ["imu", "odometry", "motor_feedback"],
            "message_type_refs": ["cartesian_controller_contract"],
        },
        {
            "target_name": "offline_wbc_reference_stack",
            "source_project": "OCS2/TSID/Crocoddyl design references",
            "source_url": "https://github.com/leggedrobotics/ocs2",
            "source_license": "BSD-3-Clause reference noted; dependency review required",
            "runtime_target": "future whole-body MPC / inverse-dynamics reference",
            "transport_profile": "offline_reference_not_runtime_bridge",
            "command_family": "whole_body_optimal_control_reference",
            "publish_topic": "",
            "subscribe_topics": [],
            "message_type_refs": ["urdf_model", "constraints", "self_collision"],
        },
    ]
    return [
        ControllerBridgeTarget(
            target_id=stable_id("controller_bridge_target", spec),
            **spec,
        )
        for spec in specs
    ]


def build_controller_mode_specs(
    bridge_targets: list[ControllerBridgeTarget],
) -> list[ControllerModeSpec]:
    target_by_name = {target.target_name: target for target in bridge_targets}
    common_gates = [
        "joint_limit_clamp",
        "rate_limit",
        "stale_data_watchdog",
        "support_phase_constraint",
        "operator_override",
        "e_stop_veto",
    ]
    specs = [
        (
            "hold_pose",
            "bipedal_whole_body",
            "unitree_ros2_lowcmd_joint_pd",
            "joint_pd_hold",
            100.0,
            "robot_local_or_companion_dry_run",
            "stable_base_fallback",
        ),
        (
            "joint_pd_tracking",
            "bipedal_whole_body",
            "unitree_ros2_lowcmd_joint_pd",
            "joint_pd_tracking",
            200.0,
            "robot_local_or_companion_dry_run",
            "hold_pose",
        ),
        (
            "cartesian_upper_body_tracking",
            "stable_base_mobile_manipulator",
            "g1pilot_cartesian_upper_body_fallback",
            "cartesian_pose_reference",
            50.0,
            "companion_dry_run",
            "hold_pose",
        ),
        (
            "stable_base_fallback",
            "stable_base_mobile_manipulator",
            "unitree_ros2_sport_request_fallback",
            "sport_request_stub",
            20.0,
            "robot_builtin_loco_controller_contract",
            "operator_teleop_pass_through",
        ),
        (
            "operator_teleop_pass_through",
            "stable_base_mobile_manipulator",
            "g1pilot_upper_body_joint_fallback",
            "upper_body_joint_reference",
            50.0,
            "operator_companion_dry_run",
            "e_stop_veto",
        ),
        (
            "e_stop_veto",
            "unknown",
            "unitree_ros2_lowcmd_joint_pd",
            "veto_only",
            1000.0,
            "robot_local_safety_contract",
            "",
        ),
    ]
    modes: list[ControllerModeSpec] = []
    for mode_name, posture, target_name, command_kind, rate, placement, fallback in specs:
        target = target_by_name[target_name]
        modes.append(
            ControllerModeSpec(
                mode_id=stable_id(
                    "controller_mode",
                    {"mode_name": mode_name, "bridge_target_id": target.target_id},
                ),
                mode_name=mode_name,
                posture_tag=posture,
                bridge_target_id=target.target_id,
                command_kind=command_kind,
                planned_rate_hz=rate,
                placement_class=placement,
                safety_gates=common_gates,
                fallback_mode=fallback,
                oss_inspiration_refs=[target.source_project, target.source_url],
            )
        )
    return modes


def _neutral_joint_positions(
    joint_limits: list[JointLimitEnvelope],
    joint_names: Optional[list[str]] = None,
) -> dict[str, float]:
    wanted = set(joint_names or [limit.joint_name for limit in joint_limits])
    return {
        limit.joint_name: (limit.lower_rad + limit.upper_rad) / 2.0
        for limit in joint_limits
        if limit.joint_name in wanted
    }


def _upper_body_joint_names(chassis: HumanoidChassisProfile) -> list[str]:
    groups = chassis.limb_groups
    names: list[str] = []
    for group in ("waist", "left_arm", "right_arm", "left_hand", "right_hand"):
        names.extend(groups.get(group, []))
    return names


def _first_replay_row(replay_rows: list[WholeBodyReplayRow]) -> WholeBodyReplayRow:
    if not replay_rows:
        return WholeBodyReplayRow(
            row_id="missing_replay_row",
            chassis_id="missing_chassis",
            posture_tag="unknown",
            support_state_id="missing_support",
            balance_receipt_id="missing_balance",
            observation_schema_ref="missing_observation_schema",
            action_schema_ref="missing_action_schema",
        )
    return replay_rows[0]


def build_downstream_controller_proposals(
    *,
    chassis: HumanoidChassisProfile,
    joint_limits: list[JointLimitEnvelope],
    modes: list[ControllerModeSpec],
    replay_rows: list[WholeBodyReplayRow],
) -> list[DownstreamControllerProposal]:
    mode_by_name = {mode.mode_name: mode for mode in modes}
    replay = _first_replay_row(replay_rows)
    neutral = _neutral_joint_positions(joint_limits)
    upper_body = _upper_body_joint_names(chassis)
    upper_neutral = _neutral_joint_positions(joint_limits, upper_body)
    violating = dict(neutral)
    if joint_limits:
        violating[joint_limits[0].joint_name] = joint_limits[0].upper_rad + 0.75
    specs = [
        {
            "proposal_name": "hold_pose_neutral_dry_run",
            "mode_name": "hold_pose",
            "support_phase": "double_support",
            "requested_joint_positions": neutral,
            "requested_command_payload": {"reason": "fallback_hold_pose"},
        },
        {
            "proposal_name": "joint_pd_tracking_clamp_probe",
            "mode_name": "joint_pd_tracking",
            "support_phase": "double_support",
            "requested_joint_positions": violating,
            "requested_command_payload": {
                "reason": "synthetic_limit_probe",
                "expect_clamp_receipt": True,
            },
        },
        {
            "proposal_name": "cartesian_upper_body_reach_dry_run",
            "mode_name": "cartesian_upper_body_tracking",
            "support_phase": "stable_base_required",
            "requested_joint_positions": upper_neutral,
            "requested_cartesian_targets": {
                "left_hand": {"frame": "pelvis", "x": 0.25, "y": 0.22, "z": 0.85},
                "right_hand": {"frame": "pelvis", "x": 0.25, "y": -0.22, "z": 0.85},
            },
            "requested_command_payload": {"fallback_posture": "stable_base"},
        },
        {
            "proposal_name": "stable_base_fallback_request_dry_run",
            "mode_name": "stable_base_fallback",
            "support_phase": "degraded_mode",
            "requested_command_payload": {
                "desired_velocity_x_mps": 0.0,
                "desired_yaw_rate_rad_s": 0.0,
                "fallback_reason": "bipedal_authority_not_promoted",
            },
        },
        {
            "proposal_name": "operator_teleop_pass_through_dry_run",
            "mode_name": "operator_teleop_pass_through",
            "support_phase": "operator_supervised",
            "requested_joint_positions": upper_neutral,
            "requested_command_payload": {
                "operator_session_ref": "teleop_session_ref_required"
            },
        },
        {
            "proposal_name": "e_stop_veto_dry_run",
            "mode_name": "e_stop_veto",
            "support_phase": "any",
            "requested_command_payload": {"veto": True, "reason": "dry_run_estop"},
            "e_stop_requested": True,
        },
    ]
    proposals: list[DownstreamControllerProposal] = []
    for spec in specs:
        mode = mode_by_name[str(spec["mode_name"])]
        payload = {
            "proposal_name": spec["proposal_name"],
            "mode_name": mode.mode_name,
            "chassis_id": chassis.chassis_id,
            "support_phase": spec["support_phase"],
        }
        proposals.append(
            DownstreamControllerProposal(
                proposal_id=stable_id("controller_proposal", payload),
                proposal_name=str(spec["proposal_name"]),
                mode_name=mode.mode_name,
                posture_tag=mode.posture_tag,
                source_loop="wm_slow_loop_shadow_proposal",
                support_phase=str(spec["support_phase"]),
                bridge_target_id=mode.bridge_target_id,
                requested_joint_positions=dict(
                    spec.get("requested_joint_positions", {})
                ),
                requested_joint_velocities={
                    key: 0.0 for key in dict(spec.get("requested_joint_positions", {}))
                },
                requested_cartesian_targets=mapping(
                    spec.get("requested_cartesian_targets")
                ),
                requested_command_payload=mapping(
                    spec.get("requested_command_payload")
                ),
                source_replay_row_id=replay.row_id,
                observation_schema_ref=replay.observation_schema_ref,
                action_schema_ref=replay.action_schema_ref,
                e_stop_requested=bool(spec.get("e_stop_requested", False)),
            )
        )
    return proposals


def _clamp_positions(
    positions: Mapping[str, float],
    joint_limits: list[JointLimitEnvelope],
) -> tuple[dict[str, float], list[str]]:
    limits_by_name = {limit.joint_name: limit for limit in joint_limits}
    clamped: dict[str, float] = {}
    clamped_names: list[str] = []
    for joint_name, value in positions.items():
        limit = limits_by_name.get(joint_name)
        if limit is None:
            clamped[joint_name] = _safe_float(value)
            continue
        raw = _safe_float(value)
        safe = min(max(raw, limit.lower_rad), limit.upper_rad)
        clamped[joint_name] = safe
        if safe != raw:
            clamped_names.append(joint_name)
    return clamped, clamped_names


def build_low_level_command_frames(
    *,
    proposals: list[DownstreamControllerProposal],
    modes: list[ControllerModeSpec],
    bridge_targets: list[ControllerBridgeTarget],
    joint_limits: list[JointLimitEnvelope],
) -> list[LowLevelCommandFrame]:
    mode_by_name = {mode.mode_name: mode for mode in modes}
    target_by_id = {target.target_id: target for target in bridge_targets}
    frames: list[LowLevelCommandFrame] = []
    for proposal in proposals:
        mode = mode_by_name[proposal.mode_name]
        target = target_by_id[proposal.bridge_target_id]
        clamped_positions, clamped_names = _clamp_positions(
            proposal.requested_joint_positions,
            joint_limits,
        )
        kp = {joint: 20.0 for joint in clamped_positions}
        kd = {joint: 1.0 for joint in clamped_positions}
        command_payload = {
            **mapping(proposal.requested_command_payload),
            "cartesian_targets": mapping(proposal.requested_cartesian_targets),
            "hardware_dispatch": "denied_dry_run_only",
        }
        frame_payload = {
            "proposal_id": proposal.proposal_id,
            "mode_name": mode.mode_name,
            "command_kind": mode.command_kind,
            "clamped_names": clamped_names,
        }
        frames.append(
            LowLevelCommandFrame(
                frame_id=stable_id("low_level_command_frame", frame_payload),
                proposal_id=proposal.proposal_id,
                mode_name=mode.mode_name,
                bridge_target_id=target.target_id,
                command_kind=mode.command_kind,
                planned_rate_hz=mode.planned_rate_hz,
                channel_names=list(clamped_positions),
                target_joint_positions=clamped_positions,
                target_joint_velocities=proposal.requested_joint_velocities,
                target_kp=kp,
                target_kd=kd,
                feedforward_torque={joint: 0.0 for joint in clamped_positions},
                command_payload=command_payload,
                bridge_topic=target.publish_topic,
                ros2_message_type=";".join(target.message_type_refs),
                unitree_command_family=target.command_family,
                clamp_applied=bool(clamped_names),
                clamped_joint_names=clamped_names,
            )
        )
    return frames


def build_controller_safety_receipts(
    *,
    proposals: list[DownstreamControllerProposal],
    frames: list[LowLevelCommandFrame],
) -> list[ControllerSafetyReceipt]:
    frame_by_proposal = {frame.proposal_id: frame for frame in frames}
    receipts: list[ControllerSafetyReceipt] = []
    for proposal in proposals:
        frame = frame_by_proposal[proposal.proposal_id]
        status = "dry_run_stale_data_veto"
        if frame.clamp_applied:
            status = "dry_run_clamped_stale_data_veto"
        if proposal.e_stop_requested:
            status = "dry_run_e_stop_veto"
        receipt_payload = {
            "proposal_id": proposal.proposal_id,
            "frame_id": frame.frame_id,
            "status": status,
        }
        receipts.append(
            ControllerSafetyReceipt(
                receipt_id=stable_id("controller_safety_receipt", receipt_payload),
                proposal_id=proposal.proposal_id,
                command_frame_id=frame.frame_id,
                status=status,
                joint_limit_clamp_applied=frame.clamp_applied,
                clamped_joint_names=frame.clamped_joint_names,
                stale_data_vetoed=True,
                support_phase_verified=False,
                support_phase_constraint_satisfied=False,
                operator_override_required=proposal.operator_override_required,
                e_stop_vetoed=proposal.e_stop_requested,
                live_safety_calibration_present=False,
                hardware_dispatch_allowed=False,
                missing_evidence=list(PHASE4_DOWNSTREAM_CONTROLLER_BLOCKERS),
            )
        )
    return receipts


def build_controller_invocations(
    *,
    proposals: list[DownstreamControllerProposal],
    frames: list[LowLevelCommandFrame],
    safety_receipts: list[ControllerSafetyReceipt],
    modes: list[ControllerModeSpec],
) -> list[ControllerInvocation]:
    frame_by_proposal = {frame.proposal_id: frame for frame in frames}
    safety_by_proposal = {
        receipt.proposal_id: receipt for receipt in safety_receipts
    }
    mode_by_name = {mode.mode_name: mode for mode in modes}
    invocations: list[ControllerInvocation] = []
    for proposal in proposals:
        frame = frame_by_proposal[proposal.proposal_id]
        safety = safety_by_proposal[proposal.proposal_id]
        mode = mode_by_name[proposal.mode_name]
        denial_reasons = [
            "dry_run_only",
            "hardware_interface_missing",
            "live_low_state_stream_missing",
            "timing_jitter_evidence_missing",
            "physical_safety_calibration_missing",
            "operator_estop_not_verified",
        ]
        if frame.clamp_applied:
            denial_reasons.append("joint_limit_clamp_applied")
        if proposal.e_stop_requested:
            denial_reasons.append("e_stop_veto_requested")
        invocation_payload = {
            "proposal_id": proposal.proposal_id,
            "frame_id": frame.frame_id,
            "safety_receipt_id": safety.receipt_id,
        }
        invocations.append(
            ControllerInvocation(
                invocation_id=stable_id("controller_invocation", invocation_payload),
                proposal_id=proposal.proposal_id,
                command_frame_id=frame.frame_id,
                safety_receipt_id=safety.receipt_id,
                bridge_target_id=proposal.bridge_target_id,
                mode_name=proposal.mode_name,
                dispatch_status="dispatch_denied_dry_run",
                dispatch_denial_reasons=denial_reasons,
                planned_rate_hz=mode.planned_rate_hz,
                placement_class=mode.placement_class,
            )
        )
    return invocations


def build_controller_receipts(
    invocations: list[ControllerInvocation],
) -> list[ControllerReceipt]:
    receipts: list[ControllerReceipt] = []
    for invocation in invocations:
        payload = {
            "invocation_id": invocation.invocation_id,
            "frame_id": invocation.command_frame_id,
            "status": invocation.dispatch_status,
        }
        receipts.append(
            ControllerReceipt(
                receipt_id=stable_id("controller_receipt", payload),
                invocation_id=invocation.invocation_id,
                proposal_id=invocation.proposal_id,
                command_frame_id=invocation.command_frame_id,
                safety_receipt_id=invocation.safety_receipt_id,
                status="receipt_emitted_dispatch_denied",
                command_frame_emitted=True,
                safety_receipt_emitted=True,
                replay_export_ready=True,
                training_aware=True,
            )
        )
    return receipts


def build_phase4_downstream_controller_scaffold(
    *,
    phase4_report: Phase4DeploymentEnablerSweepReport,
    phase35_readiness_audit: Phase35BipedalReadinessAudit,
    chassis: HumanoidChassisProfile,
    joint_limits: list[JointLimitEnvelope],
    replay_rows: list[WholeBodyReplayRow],
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    Phase4DownstreamControllerScaffoldReport,
    list[ControllerBridgeTarget],
    list[ControllerModeSpec],
    list[DownstreamControllerProposal],
    list[LowLevelCommandFrame],
    list[ControllerSafetyReceipt],
    list[ControllerInvocation],
    list[ControllerReceipt],
]:
    bridge_targets = build_controller_bridge_targets()
    modes = build_controller_mode_specs(bridge_targets)
    proposals = build_downstream_controller_proposals(
        chassis=chassis,
        joint_limits=joint_limits,
        modes=modes,
        replay_rows=replay_rows,
    )
    command_frames = build_low_level_command_frames(
        proposals=proposals,
        modes=modes,
        bridge_targets=bridge_targets,
        joint_limits=joint_limits,
    )
    safety_receipts = build_controller_safety_receipts(
        proposals=proposals,
        frames=command_frames,
    )
    invocations = build_controller_invocations(
        proposals=proposals,
        frames=command_frames,
        safety_receipts=safety_receipts,
        modes=modes,
    )
    controller_receipts = build_controller_receipts(invocations)
    unitree_present = any("unitree_ros2" in target.target_name for target in bridge_targets)
    g1pilot_present = any("g1pilot" in target.target_name for target in bridge_targets)
    dry_run_present = (
        len(command_frames) == len(proposals)
        and len(safety_receipts) == len(proposals)
        and len(invocations) == len(proposals)
        and len(controller_receipts) == len(proposals)
        and not any(invocation.publish_attempted for invocation in invocations)
        and not any(invocation.hardware_dispatch_enabled for invocation in invocations)
    )
    complete = (
        phase4_report.local_non_hardware_scaffold_complete
        and phase35_readiness_audit.phase35_no_gpu_no_hardware_prepared
        and chassis.controlled_joint_count >= 21
        and len(joint_limits) == chassis.controlled_joint_count
        and unitree_present
        and g1pilot_present
        and dry_run_present
    )
    report_payload = {
        "phase4_report_id": phase4_report.report_id,
        "phase35_readiness_audit_id": phase35_readiness_audit.audit_id,
        "chassis_id": chassis.chassis_id,
        "proposal_count": len(proposals),
    }
    report = Phase4DownstreamControllerScaffoldReport(
        report_id=stable_id("phase4_downstream_controller", report_payload),
        phase4_report_id=phase4_report.report_id,
        phase35_bipedal_readiness_audit_id=phase35_readiness_audit.audit_id,
        chassis_id=chassis.chassis_id,
        status="ok" if complete else "blocked",
        bridge_target_count=len(bridge_targets),
        mode_count=len(modes),
        proposal_count=len(proposals),
        command_frame_count=len(command_frames),
        safety_receipt_count=len(safety_receipts),
        invocation_count=len(invocations),
        controller_receipt_count=len(controller_receipts),
        local_downstream_controller_scaffold_complete=complete,
        unitree_bridge_contract_present=unitree_present,
        g1pilot_fallback_contract_present=g1pilot_present,
        dry_run_controller_present=dry_run_present,
        denied_gates=_denied_gates(),
        key_blockers=list(PHASE4_DOWNSTREAM_CONTROLLER_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return (
        report,
        bridge_targets,
        modes,
        proposals,
        command_frames,
        safety_receipts,
        invocations,
        controller_receipts,
    )


def save_phase4_downstream_controller_scaffold(
    output_dir: str | Path,
    *,
    report: Phase4DownstreamControllerScaffoldReport,
    bridge_targets: list[ControllerBridgeTarget],
    modes: list[ControllerModeSpec],
    proposals: list[DownstreamControllerProposal],
    command_frames: list[LowLevelCommandFrame],
    safety_receipts: list[ControllerSafetyReceipt],
    invocations: list[ControllerInvocation],
    controller_receipts: list[ControllerReceipt],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase4_downstream_controller_scaffold_report_v1.json",
        "bridge_targets_path": output / "controller_bridge_targets_v1.jsonl",
        "modes_path": output / "controller_mode_specs_v1.jsonl",
        "proposals_path": output / "downstream_controller_proposals_v1.jsonl",
        "command_frames_path": output / "low_level_command_frames_v1.jsonl",
        "safety_receipts_path": output / "controller_safety_receipts_v1.jsonl",
        "invocations_path": output / "controller_invocations_v1.jsonl",
        "controller_receipts_path": output / "controller_receipts_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(paths["bridge_targets_path"], [item.to_dict() for item in bridge_targets])
    write_jsonl(paths["modes_path"], [item.to_dict() for item in modes])
    write_jsonl(paths["proposals_path"], [item.to_dict() for item in proposals])
    write_jsonl(paths["command_frames_path"], [item.to_dict() for item in command_frames])
    write_jsonl(
        paths["safety_receipts_path"],
        [item.to_dict() for item in safety_receipts],
    )
    write_jsonl(paths["invocations_path"], [item.to_dict() for item in invocations])
    write_jsonl(
        paths["controller_receipts_path"],
        [item.to_dict() for item in controller_receipts],
    )
    return {key: str(path) for key, path in paths.items()}


def load_phase4_downstream_controller_scaffold_report(
    path: str | Path,
) -> Phase4DownstreamControllerScaffoldReport:
    return Phase4DownstreamControllerScaffoldReport.from_dict(load_json(path))


def load_controller_bridge_targets(path: str | Path) -> list[ControllerBridgeTarget]:
    return [ControllerBridgeTarget.from_dict(row) for row in load_jsonl(path)]


def load_controller_mode_specs(path: str | Path) -> list[ControllerModeSpec]:
    return [ControllerModeSpec.from_dict(row) for row in load_jsonl(path)]


def load_downstream_controller_proposals(
    path: str | Path,
) -> list[DownstreamControllerProposal]:
    return [DownstreamControllerProposal.from_dict(row) for row in load_jsonl(path)]


def load_low_level_command_frames(path: str | Path) -> list[LowLevelCommandFrame]:
    return [LowLevelCommandFrame.from_dict(row) for row in load_jsonl(path)]


def load_controller_safety_receipts(
    path: str | Path,
) -> list[ControllerSafetyReceipt]:
    return [ControllerSafetyReceipt.from_dict(row) for row in load_jsonl(path)]


def load_controller_invocations(path: str | Path) -> list[ControllerInvocation]:
    return [ControllerInvocation.from_dict(row) for row in load_jsonl(path)]


def load_controller_receipts(path: str | Path) -> list[ControllerReceipt]:
    return [ControllerReceipt.from_dict(row) for row in load_jsonl(path)]
