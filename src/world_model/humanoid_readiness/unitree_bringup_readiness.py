"""Phase 4 Unitree/G1 bring-up readiness receipts.

This module converts the remaining non-GPU, non-hardware Unitree/G1 bring-up
blockers into typed local receipts. It inventories local OSS/runtime roots,
parses available robot assets for canonical joint alignment, emits stream and
command-interface contracts, runs a local-only timing probe, and records safety
and operator recovery slots.

It does not invoke Unitree SDK2, publish ROS2/DDS messages, run G1Pilot, execute
sim or hardware, train weights, mutate rewards, or promote authority.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from src.world_model.embodiment_actuation import (
    HumanoidChassisProfile,
    JointLimitEnvelope,
)
from src.world_model.humanoid_readiness.common import (
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.downstream_controller import (
    LowLevelCommandFrame,
    Phase4DownstreamControllerScaffoldReport,
)

PHASE4_UNITREE_BRINGUP_REPORT_VERSION = (
    "phase4_unitree_bringup_readiness_report_v1"
)
UNITREE_BRINGUP_BLOCK_RECEIPT_VERSION = "unitree_bringup_block_receipt_v1"
UNITREE_DEPENDENCY_TARGET_VERSION = "unitree_dependency_target_v1"
UNITREE_ASSET_CALIBRATION_RECEIPT_VERSION = "unitree_asset_calibration_receipt_v1"
UNITREE_STREAM_CONTRACT_VERSION = "unitree_stream_contract_v1"
UNITREE_COMMAND_CONFORMANCE_RECEIPT_VERSION = (
    "unitree_command_conformance_receipt_v1"
)
UNITREE_TIMING_JITTER_PROBE_RECEIPT_VERSION = (
    "unitree_timing_jitter_probe_receipt_v1"
)
UNITREE_SAFETY_PREFLIGHT_RECEIPT_VERSION = "unitree_safety_preflight_receipt_v1"
UNITREE_OPERATOR_RECOVERY_RUNBOOK_VERSION = "unitree_operator_recovery_runbook_v1"
UNITREE_SIM_HARDWARE_EVIDENCE_LEDGER_VERSION = (
    "unitree_sim_hardware_evidence_ledger_v1"
)

UNITREE_BRINGUP_BLOCK_KEYS = (
    "runtime_dependency_manifest",
    "g1pilot_or_fallback_review",
    "robot_asset_calibration_intake",
    "live_stream_interface_contracts",
    "command_interface_conformance",
    "timing_jitter_probe",
    "physical_safety_preflight",
    "operator_estop_recovery_runbook",
    "sim_hardware_evidence_ledger",
)

DENIED_UNITREE_BRINGUP_AUTHORITIES = (
    "hardware_dispatch_enabled",
    "ros2_publish_attempted",
    "unitree_sdk2_write_enabled",
    "unitree_ros2_lowcmd_publish_enabled",
    "unitree_sport_request_publish_enabled",
    "g1pilot_runtime_invoked",
    "unitree_mujoco_runtime_executed",
    "unitree_rl_gym_runtime_executed",
    "unitree_sim_isaaclab_runtime_executed",
    "honest_sim_executed",
    "hardware_executed",
    "live_policy_control",
    "training_executed",
    "weights_written",
    "provider_executed",
    "reward_math_mutation",
    "promotion_eligible",
)

UNITREE_BRINGUP_REMAINING_BLOCKERS = (
    "unitree_ros2_or_sdk2_runtime_build_and_interface_verification_missing",
    "g1pilot_or_equivalent_runtime_review_and_pin_missing",
    "hardware_calibration_and_joint_limit_certification_missing",
    "live_lowstate_imu_contact_wireless_estop_streams_missing",
    "actual_lowcmd_sport_request_or_upper_body_write_path_not_validated",
    "dds_or_on_robot_control_loop_timing_jitter_missing",
    "physical_safety_calibration_and_demote_rollback_tests_missing",
    "operator_teleop_estop_recovery_drills_missing",
    "honest_sim_or_hardware_runtime_evidence_missing",
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_UNITREE_BRINGUP_AUTHORITIES}


def default_unitree_local_roots(home: str | Path | None = None) -> dict[str, str]:
    root = Path(home) if home is not None else Path.home()
    code = root / "code"
    return {
        "unitree_sdk2": str(code / "unitree_sdk2"),
        "unitree_models": str(code / "unitree_models"),
        "unitree_rl_gym": str(code / "unitree_rl_gym"),
        "unitree_sim_isaaclab": str(code / "unitree_sim_isaaclab"),
        "unitree_il_lerobot": str(code / "unitree_IL_lerobot"),
        "g1pilot": str(code / "g1pilot"),
        "unitree_ros2": str(code / "unitree_ros2"),
        "unitree_mujoco": str(code / "unitree_mujoco"),
    }


@dataclass(frozen=True)
class UnitreeDependencyTarget:
    target_id: str
    local_root_key: str
    source_project: str
    source_url: str
    source_license: str
    local_root_path: str
    expected_markers: list[str] = field(default_factory=list)
    matched_markers: list[str] = field(default_factory=list)
    missing_markers: list[str] = field(default_factory=list)
    exists: bool = False
    verified_local_layout: bool = False
    pinned_commit: str = ""
    runtime_invoked: bool = False
    vendored_code_included: bool = False
    status: str = "missing_local_root"
    authority_class: str = "dependency_inventory_only"
    version: str = UNITREE_DEPENDENCY_TARGET_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "version": self.version,
            "local_root_key": self.local_root_key,
            "source_project": self.source_project,
            "source_url": self.source_url,
            "source_license": self.source_license,
            "local_root_path": self.local_root_path,
            "expected_markers": strings(self.expected_markers),
            "matched_markers": strings(self.matched_markers),
            "missing_markers": strings(self.missing_markers),
            "exists": bool(self.exists),
            "verified_local_layout": bool(self.verified_local_layout),
            "pinned_commit": self.pinned_commit,
            "runtime_invoked": bool(self.runtime_invoked),
            "vendored_code_included": bool(self.vendored_code_included),
            "status": self.status,
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeDependencyTarget":
        return cls(
            target_id=str(payload.get("target_id", "")),
            local_root_key=str(payload.get("local_root_key", "")),
            source_project=str(payload.get("source_project", "")),
            source_url=str(payload.get("source_url", "")),
            source_license=str(payload.get("source_license", "unknown")),
            local_root_path=str(payload.get("local_root_path", "")),
            expected_markers=strings(payload.get("expected_markers")),
            matched_markers=strings(payload.get("matched_markers")),
            missing_markers=strings(payload.get("missing_markers")),
            exists=bool(payload.get("exists", False)),
            verified_local_layout=bool(payload.get("verified_local_layout", False)),
            pinned_commit=str(payload.get("pinned_commit", "")),
            runtime_invoked=bool(payload.get("runtime_invoked", False)),
            vendored_code_included=bool(payload.get("vendored_code_included", False)),
            status=str(payload.get("status", "missing_local_root")),
            authority_class=str(
                payload.get("authority_class", "dependency_inventory_only")
            ),
            version=str(payload.get("version", UNITREE_DEPENDENCY_TARGET_VERSION)),
        )


@dataclass(frozen=True)
class UnitreeAssetCalibrationReceipt:
    receipt_id: str
    chassis_id: str
    asset_path: str
    asset_source_root_key: str
    asset_format: str
    status: str
    canonical_controlled_joint_count: int
    asset_joint_count: int
    controlled_joint_subset_aligned: bool
    missing_controlled_joint_names: list[str] = field(default_factory=list)
    extra_asset_joint_names: list[str] = field(default_factory=list)
    parsed_joint_names: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    asset_parsed: bool = False
    hardware_calibrated_limits: bool = False
    calibration_sidecar_present: bool = False
    truth_class: str = "local_asset_parse_not_hardware_calibrated"
    authority_class: str = "asset_calibration_intake_receipt_only"
    version: str = UNITREE_ASSET_CALIBRATION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "asset_path": self.asset_path,
            "asset_source_root_key": self.asset_source_root_key,
            "asset_format": self.asset_format,
            "status": self.status,
            "canonical_controlled_joint_count": int(
                self.canonical_controlled_joint_count
            ),
            "asset_joint_count": int(self.asset_joint_count),
            "controlled_joint_subset_aligned": bool(
                self.controlled_joint_subset_aligned
            ),
            "missing_controlled_joint_names": strings(
                self.missing_controlled_joint_names
            ),
            "extra_asset_joint_names": strings(self.extra_asset_joint_names),
            "parsed_joint_names": strings(self.parsed_joint_names),
            "parse_errors": strings(self.parse_errors),
            "asset_parsed": bool(self.asset_parsed),
            "hardware_calibrated_limits": bool(self.hardware_calibrated_limits),
            "calibration_sidecar_present": bool(self.calibration_sidecar_present),
            "truth_class": self.truth_class,
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnitreeAssetCalibrationReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            asset_path=str(payload.get("asset_path", "")),
            asset_source_root_key=str(payload.get("asset_source_root_key", "")),
            asset_format=str(payload.get("asset_format", "")),
            status=str(payload.get("status", "")),
            canonical_controlled_joint_count=int(
                payload.get("canonical_controlled_joint_count", 0) or 0
            ),
            asset_joint_count=int(payload.get("asset_joint_count", 0) or 0),
            controlled_joint_subset_aligned=bool(
                payload.get("controlled_joint_subset_aligned", False)
            ),
            missing_controlled_joint_names=strings(
                payload.get("missing_controlled_joint_names")
            ),
            extra_asset_joint_names=strings(payload.get("extra_asset_joint_names")),
            parsed_joint_names=strings(payload.get("parsed_joint_names")),
            parse_errors=strings(payload.get("parse_errors")),
            asset_parsed=bool(payload.get("asset_parsed", False)),
            hardware_calibrated_limits=bool(
                payload.get("hardware_calibrated_limits", False)
            ),
            calibration_sidecar_present=bool(
                payload.get("calibration_sidecar_present", False)
            ),
            truth_class=str(
                payload.get(
                    "truth_class", "local_asset_parse_not_hardware_calibrated"
                )
            ),
            authority_class=str(
                payload.get(
                    "authority_class", "asset_calibration_intake_receipt_only"
                )
            ),
            version=str(
                payload.get("version", UNITREE_ASSET_CALIBRATION_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class UnitreeStreamContract:
    contract_id: str
    stream_key: str
    direction: str
    transport_profile: str
    expected_topic_or_channel: str
    schema_refs: list[str] = field(default_factory=list)
    timing_expectation: str = "runtime_measurement_required"
    replay_slot_ref: str = ""
    mock_receiver_ready: bool = True
    live_stream_observed: bool = False
    hardware_executed: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "stream_contract_only"
    version: str = UNITREE_STREAM_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "stream_key": self.stream_key,
            "direction": self.direction,
            "transport_profile": self.transport_profile,
            "expected_topic_or_channel": self.expected_topic_or_channel,
            "schema_refs": strings(self.schema_refs),
            "timing_expectation": self.timing_expectation,
            "replay_slot_ref": self.replay_slot_ref,
            "mock_receiver_ready": bool(self.mock_receiver_ready),
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeStreamContract":
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            stream_key=str(payload.get("stream_key", "")),
            direction=str(payload.get("direction", "")),
            transport_profile=str(payload.get("transport_profile", "")),
            expected_topic_or_channel=str(
                payload.get("expected_topic_or_channel", "")
            ),
            schema_refs=strings(payload.get("schema_refs")),
            timing_expectation=str(
                payload.get("timing_expectation", "runtime_measurement_required")
            ),
            replay_slot_ref=str(payload.get("replay_slot_ref", "")),
            mock_receiver_ready=bool(payload.get("mock_receiver_ready", True)),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(payload.get("authority_class", "stream_contract_only")),
            version=str(payload.get("version", UNITREE_STREAM_CONTRACT_VERSION)),
        )


@dataclass(frozen=True)
class UnitreeCommandConformanceReceipt:
    receipt_id: str
    command_family: str
    bridge_topic: str
    frame_count: int
    channel_count: int
    dry_run_frames_available: bool
    joint_limit_clamp_path_present: bool
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    hardware_dispatch_enabled: bool = False
    conformance_status: str = "dry_run_contract_only"
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "command_conformance_receipt_only"
    version: str = UNITREE_COMMAND_CONFORMANCE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "command_family": self.command_family,
            "bridge_topic": self.bridge_topic,
            "frame_count": int(self.frame_count),
            "channel_count": int(self.channel_count),
            "dry_run_frames_available": bool(self.dry_run_frames_available),
            "joint_limit_clamp_path_present": bool(
                self.joint_limit_clamp_path_present
            ),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "conformance_status": self.conformance_status,
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnitreeCommandConformanceReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            command_family=str(payload.get("command_family", "")),
            bridge_topic=str(payload.get("bridge_topic", "")),
            frame_count=int(payload.get("frame_count", 0) or 0),
            channel_count=int(payload.get("channel_count", 0) or 0),
            dry_run_frames_available=bool(
                payload.get("dry_run_frames_available", False)
            ),
            joint_limit_clamp_path_present=bool(
                payload.get("joint_limit_clamp_path_present", False)
            ),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            g1pilot_runtime_invoked=bool(payload.get("g1pilot_runtime_invoked", False)),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            conformance_status=str(
                payload.get("conformance_status", "dry_run_contract_only")
            ),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "command_conformance_receipt_only")
            ),
            version=str(
                payload.get("version", UNITREE_COMMAND_CONFORMANCE_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class UnitreeTimingJitterProbeReceipt:
    receipt_id: str
    probe_kind: str
    requested_iterations: int
    observed_iterations: int
    local_perf_counter_probe_executed: bool
    min_step_s: float
    max_step_s: float
    mean_step_s: float
    max_jitter_s: float
    dds_measured: bool = False
    hardware_measured: bool = False
    sim_runtime_measured: bool = False
    control_loop_rate_claimed: bool = False
    authority_class: str = "local_timing_probe_not_runtime_evidence"
    version: str = UNITREE_TIMING_JITTER_PROBE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "probe_kind": self.probe_kind,
            "requested_iterations": int(self.requested_iterations),
            "observed_iterations": int(self.observed_iterations),
            "local_perf_counter_probe_executed": bool(
                self.local_perf_counter_probe_executed
            ),
            "min_step_s": float(self.min_step_s),
            "max_step_s": float(self.max_step_s),
            "mean_step_s": float(self.mean_step_s),
            "max_jitter_s": float(self.max_jitter_s),
            "dds_measured": bool(self.dds_measured),
            "hardware_measured": bool(self.hardware_measured),
            "sim_runtime_measured": bool(self.sim_runtime_measured),
            "control_loop_rate_claimed": bool(self.control_loop_rate_claimed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnitreeTimingJitterProbeReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            probe_kind=str(payload.get("probe_kind", "")),
            requested_iterations=int(payload.get("requested_iterations", 0) or 0),
            observed_iterations=int(payload.get("observed_iterations", 0) or 0),
            local_perf_counter_probe_executed=bool(
                payload.get("local_perf_counter_probe_executed", False)
            ),
            min_step_s=float(payload.get("min_step_s", 0.0) or 0.0),
            max_step_s=float(payload.get("max_step_s", 0.0) or 0.0),
            mean_step_s=float(payload.get("mean_step_s", 0.0) or 0.0),
            max_jitter_s=float(payload.get("max_jitter_s", 0.0) or 0.0),
            dds_measured=bool(payload.get("dds_measured", False)),
            hardware_measured=bool(payload.get("hardware_measured", False)),
            sim_runtime_measured=bool(payload.get("sim_runtime_measured", False)),
            control_loop_rate_claimed=bool(
                payload.get("control_loop_rate_claimed", False)
            ),
            authority_class=str(
                payload.get(
                    "authority_class", "local_timing_probe_not_runtime_evidence"
                )
            ),
            version=str(
                payload.get("version", UNITREE_TIMING_JITTER_PROBE_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class UnitreeSafetyPreflightReceipt:
    receipt_id: str
    gate_key: str
    status: str
    local_check_present: bool
    runtime_check_executed: bool = False
    hardware_calibrated: bool = False
    dispatch_veto_default: bool = True
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "safety_preflight_contract_only"
    version: str = UNITREE_SAFETY_PREFLIGHT_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "gate_key": self.gate_key,
            "status": self.status,
            "local_check_present": bool(self.local_check_present),
            "runtime_check_executed": bool(self.runtime_check_executed),
            "hardware_calibrated": bool(self.hardware_calibrated),
            "dispatch_veto_default": bool(self.dispatch_veto_default),
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeSafetyPreflightReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            gate_key=str(payload.get("gate_key", "")),
            status=str(payload.get("status", "")),
            local_check_present=bool(payload.get("local_check_present", False)),
            runtime_check_executed=bool(payload.get("runtime_check_executed", False)),
            hardware_calibrated=bool(payload.get("hardware_calibrated", False)),
            dispatch_veto_default=bool(payload.get("dispatch_veto_default", True)),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "safety_preflight_contract_only")
            ),
            version=str(payload.get("version", UNITREE_SAFETY_PREFLIGHT_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class UnitreeOperatorRecoveryRunbook:
    runbook_id: str
    scenario_key: str
    trigger_conditions: list[str] = field(default_factory=list)
    required_operator_action: list[str] = field(default_factory=list)
    recovery_posture: str = "stable_base_mobile_manipulator"
    replay_slots: list[str] = field(default_factory=list)
    local_runbook_present: bool = True
    drill_executed: bool = False
    hardware_executed: bool = False
    promotion_eligible: bool = False
    authority_class: str = "operator_recovery_runbook_only"
    version: str = UNITREE_OPERATOR_RECOVERY_RUNBOOK_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "version": self.version,
            "scenario_key": self.scenario_key,
            "trigger_conditions": strings(self.trigger_conditions),
            "required_operator_action": strings(self.required_operator_action),
            "recovery_posture": self.recovery_posture,
            "replay_slots": strings(self.replay_slots),
            "local_runbook_present": bool(self.local_runbook_present),
            "drill_executed": bool(self.drill_executed),
            "hardware_executed": bool(self.hardware_executed),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeOperatorRecoveryRunbook":
        return cls(
            runbook_id=str(payload.get("runbook_id", "")),
            scenario_key=str(payload.get("scenario_key", "")),
            trigger_conditions=strings(payload.get("trigger_conditions")),
            required_operator_action=strings(payload.get("required_operator_action")),
            recovery_posture=str(
                payload.get("recovery_posture", "stable_base_mobile_manipulator")
            ),
            replay_slots=strings(payload.get("replay_slots")),
            local_runbook_present=bool(payload.get("local_runbook_present", True)),
            drill_executed=bool(payload.get("drill_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get("authority_class", "operator_recovery_runbook_only")
            ),
            version=str(payload.get("version", UNITREE_OPERATOR_RECOVERY_RUNBOOK_VERSION)),
        )


@dataclass(frozen=True)
class UnitreeSimHardwareEvidenceLedger:
    ledger_id: str
    dependency_target_ids: list[str] = field(default_factory=list)
    local_roots_present: list[str] = field(default_factory=list)
    local_roots_missing: list[str] = field(default_factory=list)
    sim_runtime_candidates: list[str] = field(default_factory=list)
    hardware_runtime_candidates: list[str] = field(default_factory=list)
    honest_sim_executed: bool = False
    hardware_executed: bool = False
    provider_executed: bool = False
    evidence_status: str = "local_inventory_only"
    missing_evidence: list[str] = field(default_factory=list)
    authority_class: str = "sim_hardware_evidence_ledger_only"
    version: str = UNITREE_SIM_HARDWARE_EVIDENCE_LEDGER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "version": self.version,
            "dependency_target_ids": strings(self.dependency_target_ids),
            "local_roots_present": strings(self.local_roots_present),
            "local_roots_missing": strings(self.local_roots_missing),
            "sim_runtime_candidates": strings(self.sim_runtime_candidates),
            "hardware_runtime_candidates": strings(self.hardware_runtime_candidates),
            "honest_sim_executed": bool(self.honest_sim_executed),
            "hardware_executed": bool(self.hardware_executed),
            "provider_executed": bool(self.provider_executed),
            "evidence_status": self.evidence_status,
            "missing_evidence": strings(self.missing_evidence),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnitreeSimHardwareEvidenceLedger":
        return cls(
            ledger_id=str(payload.get("ledger_id", "")),
            dependency_target_ids=strings(payload.get("dependency_target_ids")),
            local_roots_present=strings(payload.get("local_roots_present")),
            local_roots_missing=strings(payload.get("local_roots_missing")),
            sim_runtime_candidates=strings(payload.get("sim_runtime_candidates")),
            hardware_runtime_candidates=strings(
                payload.get("hardware_runtime_candidates")
            ),
            honest_sim_executed=bool(payload.get("honest_sim_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            evidence_status=str(payload.get("evidence_status", "local_inventory_only")),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_class=str(
                payload.get("authority_class", "sim_hardware_evidence_ledger_only")
            ),
            version=str(
                payload.get("version", UNITREE_SIM_HARDWARE_EVIDENCE_LEDGER_VERSION)
            ),
        )


@dataclass(frozen=True)
class UnitreeBringupBlockReceipt:
    receipt_id: str
    block_key: str
    status: str
    local_prepared: bool
    external_blocked: bool
    evidence_refs: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    denied_authority: list[str] = field(default_factory=list)
    authority_class: str = "unitree_bringup_block_receipt_only"
    version: str = UNITREE_BRINGUP_BLOCK_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "block_key": self.block_key,
            "status": self.status,
            "local_prepared": bool(self.local_prepared),
            "external_blocked": bool(self.external_blocked),
            "evidence_refs": strings(self.evidence_refs),
            "missing_evidence": strings(self.missing_evidence),
            "denied_authority": strings(self.denied_authority),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeBringupBlockReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            block_key=str(payload.get("block_key", "")),
            status=str(payload.get("status", "")),
            local_prepared=bool(payload.get("local_prepared", False)),
            external_blocked=bool(payload.get("external_blocked", False)),
            evidence_refs=strings(payload.get("evidence_refs")),
            missing_evidence=strings(payload.get("missing_evidence")),
            denied_authority=strings(payload.get("denied_authority")),
            authority_class=str(
                payload.get(
                    "authority_class", "unitree_bringup_block_receipt_only"
                )
            ),
            version=str(payload.get("version", UNITREE_BRINGUP_BLOCK_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class Phase4UnitreeBringupReadinessReport:
    report_id: str
    phase4_downstream_controller_report_id: str
    chassis_id: str
    status: str
    block_count: int
    dependency_target_count: int
    dependency_verified_count: int
    asset_calibration_receipt_count: int
    stream_contract_count: int
    command_conformance_receipt_count: int
    timing_jitter_probe_count: int
    safety_preflight_receipt_count: int
    operator_recovery_runbook_count: int
    evidence_ledger_count: int
    all_block_receipts_emitted: bool
    dependency_discovery_complete: bool
    asset_joint_subset_aligned: bool
    stream_contracts_present: bool
    command_conformance_dry_run_ready: bool
    local_timing_probe_present: bool
    physical_safety_preflight_present: bool
    operator_recovery_runbook_present: bool
    honest_sim_or_hardware_evidence_present: bool
    local_pre_purchase_prepared: bool
    hardware_dispatch_enabled: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    honest_sim_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    remaining_key_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE4_UNITREE_BRINGUP_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase4_downstream_controller_report_id": (
                self.phase4_downstream_controller_report_id
            ),
            "chassis_id": self.chassis_id,
            "status": self.status,
            "block_count": int(self.block_count),
            "dependency_target_count": int(self.dependency_target_count),
            "dependency_verified_count": int(self.dependency_verified_count),
            "asset_calibration_receipt_count": int(
                self.asset_calibration_receipt_count
            ),
            "stream_contract_count": int(self.stream_contract_count),
            "command_conformance_receipt_count": int(
                self.command_conformance_receipt_count
            ),
            "timing_jitter_probe_count": int(self.timing_jitter_probe_count),
            "safety_preflight_receipt_count": int(
                self.safety_preflight_receipt_count
            ),
            "operator_recovery_runbook_count": int(
                self.operator_recovery_runbook_count
            ),
            "evidence_ledger_count": int(self.evidence_ledger_count),
            "all_block_receipts_emitted": bool(self.all_block_receipts_emitted),
            "dependency_discovery_complete": bool(
                self.dependency_discovery_complete
            ),
            "asset_joint_subset_aligned": bool(self.asset_joint_subset_aligned),
            "stream_contracts_present": bool(self.stream_contracts_present),
            "command_conformance_dry_run_ready": bool(
                self.command_conformance_dry_run_ready
            ),
            "local_timing_probe_present": bool(self.local_timing_probe_present),
            "physical_safety_preflight_present": bool(
                self.physical_safety_preflight_present
            ),
            "operator_recovery_runbook_present": bool(
                self.operator_recovery_runbook_present
            ),
            "honest_sim_or_hardware_evidence_present": bool(
                self.honest_sim_or_hardware_evidence_present
            ),
            "local_pre_purchase_prepared": bool(self.local_pre_purchase_prepared),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "honest_sim_executed": bool(self.honest_sim_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": dict(self.denied_gates),
            "remaining_key_blockers": strings(self.remaining_key_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase4UnitreeBringupReadinessReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase4_downstream_controller_report_id=str(
                payload.get("phase4_downstream_controller_report_id", "")
            ),
            chassis_id=str(payload.get("chassis_id", "")),
            status=str(payload.get("status", "blocked")),
            block_count=int(payload.get("block_count", 0) or 0),
            dependency_target_count=int(
                payload.get("dependency_target_count", 0) or 0
            ),
            dependency_verified_count=int(
                payload.get("dependency_verified_count", 0) or 0
            ),
            asset_calibration_receipt_count=int(
                payload.get("asset_calibration_receipt_count", 0) or 0
            ),
            stream_contract_count=int(payload.get("stream_contract_count", 0) or 0),
            command_conformance_receipt_count=int(
                payload.get("command_conformance_receipt_count", 0) or 0
            ),
            timing_jitter_probe_count=int(
                payload.get("timing_jitter_probe_count", 0) or 0
            ),
            safety_preflight_receipt_count=int(
                payload.get("safety_preflight_receipt_count", 0) or 0
            ),
            operator_recovery_runbook_count=int(
                payload.get("operator_recovery_runbook_count", 0) or 0
            ),
            evidence_ledger_count=int(payload.get("evidence_ledger_count", 0) or 0),
            all_block_receipts_emitted=bool(
                payload.get("all_block_receipts_emitted", False)
            ),
            dependency_discovery_complete=bool(
                payload.get("dependency_discovery_complete", False)
            ),
            asset_joint_subset_aligned=bool(
                payload.get("asset_joint_subset_aligned", False)
            ),
            stream_contracts_present=bool(
                payload.get("stream_contracts_present", False)
            ),
            command_conformance_dry_run_ready=bool(
                payload.get("command_conformance_dry_run_ready", False)
            ),
            local_timing_probe_present=bool(
                payload.get("local_timing_probe_present", False)
            ),
            physical_safety_preflight_present=bool(
                payload.get("physical_safety_preflight_present", False)
            ),
            operator_recovery_runbook_present=bool(
                payload.get("operator_recovery_runbook_present", False)
            ),
            honest_sim_or_hardware_evidence_present=bool(
                payload.get("honest_sim_or_hardware_evidence_present", False)
            ),
            local_pre_purchase_prepared=bool(
                payload.get("local_pre_purchase_prepared", False)
            ),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            g1pilot_runtime_invoked=bool(payload.get("g1pilot_runtime_invoked", False)),
            honest_sim_executed=bool(payload.get("honest_sim_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(payload.get("denied_gates", {}) or {}).items()
                },
            },
            remaining_key_blockers=strings(payload.get("remaining_key_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(payload.get("version", PHASE4_UNITREE_BRINGUP_REPORT_VERSION)),
        )


def _dependency_specs() -> list[dict[str, Any]]:
    return [
        {
            "local_root_key": "unitree_sdk2",
            "source_project": "unitreerobotics/unitree_sdk2",
            "source_url": "https://github.com/unitreerobotics/unitree_sdk2",
            "source_license": "BSD-3-Clause",
            "expected_markers": ["CMakeLists.txt", "include/unitree", "lib"],
        },
        {
            "local_root_key": "unitree_ros2",
            "source_project": "unitreerobotics/unitree_ros2",
            "source_url": "https://github.com/unitreerobotics/unitree_ros2",
            "source_license": "BSD-3-Clause",
            "expected_markers": [
                "example",
                "cyclonedds_ws/src/unitree/unitree_api",
                "cyclonedds_ws/src/unitree/unitree_go",
            ],
        },
        {
            "local_root_key": "g1pilot",
            "source_project": "hucebot/g1pilot",
            "source_url": "https://github.com/hucebot/g1pilot",
            "source_license": "BSD-3-Clause",
            "expected_markers": ["README.md", "g1pilot"],
        },
        {
            "local_root_key": "unitree_models",
            "source_project": "unitreerobotics/unitree_model",
            "source_url": "https://github.com/unitreerobotics/unitree_model",
            "source_license": "BSD-3-Clause",
            "expected_markers": ["G1/29dof/usd", "README.md"],
        },
        {
            "local_root_key": "unitree_rl_gym",
            "source_project": "unitreerobotics/unitree_rl_gym",
            "source_url": "https://github.com/unitreerobotics/unitree_rl_gym",
            "source_license": "BSD-3-Clause",
            "expected_markers": [
                "resources/robots/g1_description/g1_29dof.urdf",
                "legged_gym",
                "deploy",
            ],
        },
        {
            "local_root_key": "unitree_mujoco",
            "source_project": "unitreerobotics/unitree_mujoco",
            "source_url": "https://github.com/unitreerobotics/unitree_mujoco",
            "source_license": "BSD-3-Clause",
            "expected_markers": [
                "readme.md",
                "simulate",
                "simulate_python",
                "unitree_robots/g1",
            ],
        },
        {
            "local_root_key": "unitree_sim_isaaclab",
            "source_project": "unitree_sim_isaaclab_local_or_fork",
            "source_url": "https://github.com/unitreerobotics/unitree_sim_isaaclab",
            "source_license": "external_dependency_review_required",
            "expected_markers": ["tasks/g1_tasks", "layeredcontrol", "tools"],
        },
        {
            "local_root_key": "unitree_il_lerobot",
            "source_project": "unitree_IL_lerobot_local_or_fork",
            "source_url": "https://huggingface.co/docs/lerobot/unitree_g1",
            "source_license": "external_dependency_review_required",
            "expected_markers": [
                "unitree_lerobot/eval_robot",
                "unitree_lerobot/utils",
                "pyproject.toml",
            ],
        },
    ]


def _read_git_head(root: Path) -> str:
    git = root / ".git"
    head = git / "HEAD"
    if not head.exists():
        return ""
    try:
        head_text = head.read_text(encoding="utf-8").strip()
        if head_text.startswith("ref: "):
            ref_path = git / head_text.split(" ", 1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed = git / "packed-refs"
            if packed.exists():
                ref_name = head_text.split(" ", 1)[1]
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(ref_name):
                        return line.split(" ", 1)[0]
            return ""
        return head_text
    except OSError:
        return ""


def build_unitree_dependency_targets(
    local_roots: Mapping[str, str | Path] | None = None,
) -> list[UnitreeDependencyTarget]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    targets: list[UnitreeDependencyTarget] = []
    for spec in _dependency_specs():
        root_key = str(spec["local_root_key"])
        root = Path(roots.get(root_key, ""))
        markers = strings(spec.get("expected_markers"))
        matched = [marker for marker in markers if (root / marker).exists()]
        missing = [marker for marker in markers if marker not in matched]
        exists = root.exists()
        verified = exists and not missing
        status = "verified_local_layout" if verified else "partial_or_missing_layout"
        if not exists:
            status = "missing_local_root"
        payload = {
            "local_root_key": root_key,
            "source_project": spec["source_project"],
            "local_root_path": str(root),
            "verified": verified,
        }
        targets.append(
            UnitreeDependencyTarget(
                target_id=stable_id("unitree_dependency", payload),
                local_root_key=root_key,
                source_project=str(spec["source_project"]),
                source_url=str(spec["source_url"]),
                source_license=str(spec["source_license"]),
                local_root_path=str(root),
                expected_markers=markers,
                matched_markers=matched,
                missing_markers=missing,
                exists=exists,
                verified_local_layout=verified,
                pinned_commit=_read_git_head(root) if exists else "",
                status=status,
            )
        )
    return targets


def _parse_urdf_joints(asset_path: Path) -> tuple[list[str], list[str]]:
    try:
        root = ET.parse(asset_path).getroot()
    except Exception as exc:
        return [], [str(exc)]
    names: list[str] = []
    for element in root.findall(".//joint"):
        name = element.attrib.get("name")
        if name:
            names.append(name)
    return names, []


def _candidate_asset_paths(
    local_roots: Mapping[str, str | Path],
    asset_paths: Optional[list[str | Path]] = None,
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for path in asset_paths or []:
        candidates.append((Path(path), "explicit_asset_path"))
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    known = [
        (
            Path(roots.get("unitree_rl_gym", ""))
            / "resources/robots/g1_description/g1_29dof.urdf",
            "unitree_rl_gym",
        ),
        (
            Path(roots.get("unitree_models", "")) / "G1/29dof/usd",
            "unitree_models",
        ),
    ]
    for path, root_key in known:
        candidates.append((path, root_key))
    for root_key in ("unitree_rl_gym", "unitree_models", "unitree_mujoco"):
        root = Path(roots.get(root_key, ""))
        if root.exists():
            candidates.extend((path, root_key) for path in root.rglob("*g1*.urdf"))
    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, root_key in candidates:
        resolved = str(path)
        if resolved not in seen:
            unique.append((path, root_key))
            seen.add(resolved)
    return unique


def build_unitree_asset_calibration_receipts(
    *,
    chassis: HumanoidChassisProfile,
    local_roots: Mapping[str, str | Path] | None = None,
    asset_paths: Optional[list[str | Path]] = None,
) -> list[UnitreeAssetCalibrationReceipt]:
    candidates = _candidate_asset_paths(dict(local_roots or {}), asset_paths)
    selected_path = Path("")
    selected_root_key = ""
    parsed_names: list[str] = []
    parse_errors: list[str] = []
    asset_format = "unknown"
    for path, root_key in candidates:
        if path.is_dir():
            urdfs = sorted(path.rglob("*.urdf"))
            if urdfs:
                path = urdfs[0]
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() == ".urdf":
            parsed_names, parse_errors = _parse_urdf_joints(path)
            asset_format = "urdf"
        else:
            parsed_names = []
            parse_errors = ["asset_format_not_locally_parseable"]
            asset_format = path.suffix.lower().lstrip(".") or "unknown"
        selected_path = path
        selected_root_key = root_key
        if parsed_names or parse_errors:
            break
    canonical = set(chassis.joint_names)
    parsed = set(parsed_names)
    missing = sorted(canonical - parsed)
    extra = sorted(parsed - canonical)
    subset_aligned = bool(parsed_names) and not missing
    if not str(selected_path):
        status = "asset_missing"
    elif parse_errors:
        status = "asset_parse_error"
    elif subset_aligned and extra:
        status = "controlled_joint_subset_aligned_with_extra_asset_joints"
    elif subset_aligned:
        status = "aligned"
    else:
        status = "missing_controlled_joints"
    payload = {
        "chassis_id": chassis.chassis_id,
        "asset_path": str(selected_path),
        "status": status,
        "subset_aligned": subset_aligned,
    }
    return [
        UnitreeAssetCalibrationReceipt(
            receipt_id=stable_id("unitree_asset_calibration", payload),
            chassis_id=chassis.chassis_id,
            asset_path=str(selected_path),
            asset_source_root_key=selected_root_key,
            asset_format=asset_format,
            status=status,
            canonical_controlled_joint_count=len(chassis.joint_names),
            asset_joint_count=len(parsed_names),
            controlled_joint_subset_aligned=subset_aligned,
            missing_controlled_joint_names=missing,
            extra_asset_joint_names=extra,
            parsed_joint_names=parsed_names,
            parse_errors=parse_errors,
            asset_parsed=bool(parsed_names) and not parse_errors,
        )
    ]


def build_unitree_stream_contracts() -> list[UnitreeStreamContract]:
    specs = [
        (
            "lowstate",
            "receive",
            "ros2_dds_or_sdk2",
            "/lowstate",
            ["LowState", "motor_state", "imu_state"],
            "100-500hz_runtime_measurement_required",
        ),
        (
            "imu",
            "receive",
            "ros2_dds_or_sdk2",
            "/lowstate.imu",
            ["imu_quaternion", "gyro", "accelerometer"],
            "100-500hz_runtime_measurement_required",
        ),
        (
            "wireless_estop",
            "receive",
            "ros2_dds_or_sdk2",
            "/wirelesscontroller",
            ["buttons", "estop", "operator_intent"],
            "operator_event_latency_measurement_required",
        ),
        (
            "lowcmd",
            "send",
            "ros2_dds_or_sdk2",
            "/lowcmd",
            ["LowCmd", "joint_pd_targets"],
            "200hz_write_path_validation_required",
        ),
        (
            "sport_request",
            "send",
            "ros2_dds_or_sdk2",
            "/api/sport/request",
            ["unitree_api.msg.Request"],
            "20-50hz_degraded_mode_validation_required",
        ),
        (
            "replay_export",
            "record",
            "local_artifact",
            "whole_body_replay_row",
            ["observation_schema", "action_schema", "safety_receipts"],
            "post_run_export_validation_required",
        ),
    ]
    contracts: list[UnitreeStreamContract] = []
    for stream_key, direction, profile, channel, refs, timing in specs:
        payload = {"stream_key": stream_key, "direction": direction, "channel": channel}
        contracts.append(
            UnitreeStreamContract(
                contract_id=stable_id("unitree_stream_contract", payload),
                stream_key=stream_key,
                direction=direction,
                transport_profile=profile,
                expected_topic_or_channel=channel,
                schema_refs=list(refs),
                timing_expectation=timing,
                replay_slot_ref=f"{stream_key}_replay_slot_ref_required",
                missing_evidence=[
                    "live_stream_capture",
                    "timestamp_sync",
                    "dds_qos_trace",
                    "hardware_or_honest_sim_trace",
                ],
            )
        )
    return contracts


def build_unitree_command_conformance_receipts(
    command_frames: list[LowLevelCommandFrame],
) -> list[UnitreeCommandConformanceReceipt]:
    frames_by_family: dict[str, list[LowLevelCommandFrame]] = {}
    for frame in command_frames:
        frames_by_family.setdefault(frame.unitree_command_family, []).append(frame)
    receipts: list[UnitreeCommandConformanceReceipt] = []
    for family, frames in sorted(frames_by_family.items()):
        topics = sorted({frame.bridge_topic for frame in frames if frame.bridge_topic})
        channel_count = sum(len(frame.channel_names) for frame in frames)
        clamp_present = any(frame.clamp_applied for frame in frames)
        payload = {
            "family": family,
            "frame_count": len(frames),
            "channel_count": channel_count,
        }
        receipts.append(
            UnitreeCommandConformanceReceipt(
                receipt_id=stable_id("unitree_command_conformance", payload),
                command_family=family,
                bridge_topic=";".join(topics),
                frame_count=len(frames),
                channel_count=channel_count,
                dry_run_frames_available=bool(frames),
                joint_limit_clamp_path_present=clamp_present,
                conformance_status="dry_run_frames_shape_checked",
                missing_evidence=[
                    "actual_ros2_or_sdk2_write_path",
                    "hardware_or_honest_sim_echo",
                    "timing_jitter_trace",
                    "safety_supervisor_trace",
                ],
            )
        )
    return receipts


def build_unitree_timing_jitter_probe_receipts(
    requested_iterations: int = 200,
) -> list[UnitreeTimingJitterProbeReceipt]:
    iterations = max(2, int(requested_iterations))
    times: list[float] = []
    last = time.perf_counter()
    for _ in range(iterations):
        current = time.perf_counter()
        times.append(current - last)
        last = current
    steps = times[1:] or [0.0]
    mean = sum(steps) / len(steps)
    max_jitter = max(abs(step - mean) for step in steps) if steps else 0.0
    payload = {"probe_kind": "local_perf_counter", "iterations": iterations}
    return [
        UnitreeTimingJitterProbeReceipt(
            receipt_id=stable_id("unitree_timing_jitter", payload),
            probe_kind="local_perf_counter_no_runtime_io",
            requested_iterations=iterations,
            observed_iterations=len(steps),
            local_perf_counter_probe_executed=True,
            min_step_s=min(steps),
            max_step_s=max(steps),
            mean_step_s=mean,
            max_jitter_s=max_jitter,
        )
    ]


def build_unitree_safety_preflight_receipts(
    *,
    joint_limits: list[JointLimitEnvelope],
    command_receipts: list[UnitreeCommandConformanceReceipt],
) -> list[UnitreeSafetyPreflightReceipt]:
    clamp_present = any(receipt.joint_limit_clamp_path_present for receipt in command_receipts)
    specs = [
        (
            "joint_limit_clamp",
            bool(joint_limits) and clamp_present,
            ["hardware_calibrated_joint_limits", "actual_actuator_limit_echo"],
        ),
        (
            "stale_data_watchdog",
            True,
            ["live_lowstate_age_trace", "dds_qos_drop_trace"],
        ),
        (
            "estop_veto",
            True,
            ["wireless_or_physical_estop_stream", "operator_drill_trace"],
        ),
        (
            "self_collision_and_fall_guard",
            True,
            ["collision_model", "fall_recovery_sim_or_hardware_trace"],
        ),
        (
            "stable_base_demote_rollback",
            True,
            ["safe_demote_drill", "rollback_trace", "operator_handoff_trace"],
        ),
    ]
    receipts: list[UnitreeSafetyPreflightReceipt] = []
    for gate_key, present, missing in specs:
        payload = {"gate_key": gate_key, "present": present}
        receipts.append(
            UnitreeSafetyPreflightReceipt(
                receipt_id=stable_id("unitree_safety_preflight", payload),
                gate_key=gate_key,
                status="local_check_present_dispatch_veto_default"
                if present
                else "local_check_incomplete_dispatch_veto_default",
                local_check_present=present,
                missing_evidence=list(missing),
            )
        )
    return receipts


def build_unitree_operator_recovery_runbooks() -> list[UnitreeOperatorRecoveryRunbook]:
    specs = [
        (
            "estop_pressed",
            ["operator_or_safety_estop", "command_stream_vetoed"],
            ["hold_position_or_disable_per_unitree_runbook", "record_estop_receipt"],
            "unknown",
        ),
        (
            "lowstate_stale",
            ["lowstate_age_exceeds_budget", "dds_freshness_lost"],
            ["deny_dispatch", "request_operator_handoff", "export_stale_trace"],
            "stable_base_mobile_manipulator",
        ),
        (
            "balance_margin_low",
            ["support_polygon_margin_low", "fall_guard_warning"],
            ["demote_to_stable_base_fallback", "reduce_command_envelope"],
            "stable_base_mobile_manipulator",
        ),
        (
            "teleop_takeover",
            ["operator_requests_authority", "autonomy_confidence_low"],
            ["freeze_wm_commands", "record_teleop_trace", "resume_only_after_gate"],
            "stable_base_mobile_manipulator",
        ),
    ]
    runbooks: list[UnitreeOperatorRecoveryRunbook] = []
    for scenario, triggers, actions, posture in specs:
        payload = {"scenario": scenario, "posture": posture}
        runbooks.append(
            UnitreeOperatorRecoveryRunbook(
                runbook_id=stable_id("unitree_operator_recovery", payload),
                scenario_key=scenario,
                trigger_conditions=list(triggers),
                required_operator_action=list(actions),
                recovery_posture=posture,
                replay_slots=[
                    f"{scenario}_operator_trace_ref",
                    f"{scenario}_recovery_receipt_ref",
                ],
            )
        )
    return runbooks


def build_unitree_sim_hardware_evidence_ledgers(
    dependency_targets: list[UnitreeDependencyTarget],
) -> list[UnitreeSimHardwareEvidenceLedger]:
    present = [target.local_root_key for target in dependency_targets if target.exists]
    missing = [target.local_root_key for target in dependency_targets if not target.exists]
    sim_candidates = [
        key
        for key in present
        if key in {"unitree_rl_gym", "unitree_mujoco", "unitree_sim_isaaclab"}
    ]
    hardware_candidates = [
        key for key in present if key in {"unitree_sdk2", "unitree_ros2"}
    ]
    payload = {
        "present": present,
        "missing": missing,
        "sim_candidates": sim_candidates,
        "hardware_candidates": hardware_candidates,
    }
    return [
        UnitreeSimHardwareEvidenceLedger(
            ledger_id=stable_id("unitree_evidence_ledger", payload),
            dependency_target_ids=[target.target_id for target in dependency_targets],
            local_roots_present=present,
            local_roots_missing=missing,
            sim_runtime_candidates=sim_candidates,
            hardware_runtime_candidates=hardware_candidates,
            missing_evidence=[
                "successful_sim_launch_trace",
                "ros2_or_sdk2_loopback_trace",
                "hardware_lowstate_capture",
                "operator_recovery_drill_trace",
            ],
        )
    ]


def build_unitree_bringup_block_receipts(
    *,
    dependency_targets: list[UnitreeDependencyTarget],
    asset_receipts: list[UnitreeAssetCalibrationReceipt],
    stream_contracts: list[UnitreeStreamContract],
    command_receipts: list[UnitreeCommandConformanceReceipt],
    timing_receipts: list[UnitreeTimingJitterProbeReceipt],
    safety_receipts: list[UnitreeSafetyPreflightReceipt],
    operator_runbooks: list[UnitreeOperatorRecoveryRunbook],
    evidence_ledgers: list[UnitreeSimHardwareEvidenceLedger],
) -> list[UnitreeBringupBlockReceipt]:
    dependency_ids = [target.target_id for target in dependency_targets]
    asset_ids = [receipt.receipt_id for receipt in asset_receipts]
    stream_ids = [contract.contract_id for contract in stream_contracts]
    command_ids = [receipt.receipt_id for receipt in command_receipts]
    timing_ids = [receipt.receipt_id for receipt in timing_receipts]
    safety_ids = [receipt.receipt_id for receipt in safety_receipts]
    runbook_ids = [runbook.runbook_id for runbook in operator_runbooks]
    ledger_ids = [ledger.ledger_id for ledger in evidence_ledgers]
    g1pilot_present = any(
        target.local_root_key == "g1pilot" and target.exists
        for target in dependency_targets
    )
    block_specs = [
        (
            "runtime_dependency_manifest",
            bool(dependency_targets),
            dependency_ids,
            [
                "dependency_build_verification",
                "runtime_import_or_launch_smoke",
                "license_pin_review",
            ],
        ),
        (
            "g1pilot_or_fallback_review",
            True,
            dependency_ids + command_ids,
            []
            if g1pilot_present
            else ["g1pilot_runtime_absent_or_equivalent_not_vendored"],
        ),
        (
            "robot_asset_calibration_intake",
            bool(asset_receipts)
            and any(receipt.controlled_joint_subset_aligned for receipt in asset_receipts),
            asset_ids,
            ["hardware_calibration_sidecars", "certified_joint_safety_limits"],
        ),
        (
            "live_stream_interface_contracts",
            bool(stream_contracts),
            stream_ids,
            ["live_lowstate_imu_contact_wireless_estop_capture"],
        ),
        (
            "command_interface_conformance",
            bool(command_receipts)
            and any(receipt.dry_run_frames_available for receipt in command_receipts),
            command_ids,
            ["actual_ros2_sdk2_or_g1pilot_write_validation"],
        ),
        (
            "timing_jitter_probe",
            bool(timing_receipts)
            and all(receipt.local_perf_counter_probe_executed for receipt in timing_receipts),
            timing_ids,
            ["dds_timing_jitter", "hardware_control_loop_timing"],
        ),
        (
            "physical_safety_preflight",
            bool(safety_receipts)
            and all(receipt.dispatch_veto_default for receipt in safety_receipts),
            safety_ids,
            ["physical_safety_calibration", "safe_demote_rollback_drills"],
        ),
        (
            "operator_estop_recovery_runbook",
            bool(operator_runbooks)
            and all(runbook.local_runbook_present for runbook in operator_runbooks),
            runbook_ids,
            ["operator_teleop_estop_recovery_drill_traces"],
        ),
        (
            "sim_hardware_evidence_ledger",
            bool(evidence_ledgers),
            ledger_ids,
            ["honest_sim_runtime_trace", "hardware_runtime_trace"],
        ),
    ]
    receipts: list[UnitreeBringupBlockReceipt] = []
    for block_key, local_prepared, evidence_refs, missing in block_specs:
        payload = {"block_key": block_key, "local_prepared": local_prepared}
        receipts.append(
            UnitreeBringupBlockReceipt(
                receipt_id=stable_id("unitree_bringup_block", payload),
                block_key=block_key,
                status="local_prepared_external_evidence_blocked"
                if local_prepared
                else "local_incomplete_external_evidence_blocked",
                local_prepared=local_prepared,
                external_blocked=True,
                evidence_refs=list(evidence_refs),
                missing_evidence=list(missing),
                denied_authority=list(DENIED_UNITREE_BRINGUP_AUTHORITIES),
            )
        )
    return receipts


def build_phase4_unitree_bringup_readiness(
    *,
    phase4_downstream_controller_report: Phase4DownstreamControllerScaffoldReport,
    chassis: HumanoidChassisProfile,
    joint_limits: list[JointLimitEnvelope],
    command_frames: list[LowLevelCommandFrame],
    local_roots: Mapping[str, str | Path] | None = None,
    asset_paths: Optional[list[str | Path]] = None,
    timing_iterations: int = 200,
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    Phase4UnitreeBringupReadinessReport,
    list[UnitreeBringupBlockReceipt],
    list[UnitreeDependencyTarget],
    list[UnitreeAssetCalibrationReceipt],
    list[UnitreeStreamContract],
    list[UnitreeCommandConformanceReceipt],
    list[UnitreeTimingJitterProbeReceipt],
    list[UnitreeSafetyPreflightReceipt],
    list[UnitreeOperatorRecoveryRunbook],
    list[UnitreeSimHardwareEvidenceLedger],
]:
    dependency_targets = build_unitree_dependency_targets(local_roots)
    asset_receipts = build_unitree_asset_calibration_receipts(
        chassis=chassis,
        local_roots=local_roots,
        asset_paths=asset_paths,
    )
    stream_contracts = build_unitree_stream_contracts()
    command_receipts = build_unitree_command_conformance_receipts(command_frames)
    timing_receipts = build_unitree_timing_jitter_probe_receipts(timing_iterations)
    safety_receipts = build_unitree_safety_preflight_receipts(
        joint_limits=joint_limits,
        command_receipts=command_receipts,
    )
    operator_runbooks = build_unitree_operator_recovery_runbooks()
    evidence_ledgers = build_unitree_sim_hardware_evidence_ledgers(dependency_targets)
    block_receipts = build_unitree_bringup_block_receipts(
        dependency_targets=dependency_targets,
        asset_receipts=asset_receipts,
        stream_contracts=stream_contracts,
        command_receipts=command_receipts,
        timing_receipts=timing_receipts,
        safety_receipts=safety_receipts,
        operator_runbooks=operator_runbooks,
        evidence_ledgers=evidence_ledgers,
    )
    all_blocks = {receipt.block_key for receipt in block_receipts} == set(
        UNITREE_BRINGUP_BLOCK_KEYS
    )
    dependency_discovery = len(dependency_targets) == len(_dependency_specs())
    asset_aligned = any(
        receipt.controlled_joint_subset_aligned for receipt in asset_receipts
    )
    command_ready = (
        phase4_downstream_controller_report.local_downstream_controller_scaffold_complete
        and bool(command_receipts)
        and all(receipt.dry_run_frames_available for receipt in command_receipts)
        and not any(receipt.ros2_publish_attempted for receipt in command_receipts)
        and not any(receipt.unitree_sdk2_write_enabled for receipt in command_receipts)
    )
    local_prepared = (
        all_blocks
        and all(receipt.local_prepared for receipt in block_receipts)
        and dependency_discovery
        and asset_aligned
        and bool(stream_contracts)
        and command_ready
        and bool(timing_receipts)
        and bool(safety_receipts)
        and bool(operator_runbooks)
        and bool(evidence_ledgers)
    )
    report_payload = {
        "phase4_downstream_controller_report_id": (
            phase4_downstream_controller_report.report_id
        ),
        "chassis_id": chassis.chassis_id,
        "local_prepared": local_prepared,
        "block_count": len(block_receipts),
    }
    report = Phase4UnitreeBringupReadinessReport(
        report_id=stable_id("phase4_unitree_bringup", report_payload),
        phase4_downstream_controller_report_id=(
            phase4_downstream_controller_report.report_id
        ),
        chassis_id=chassis.chassis_id,
        status="ok" if local_prepared else "blocked",
        block_count=len(block_receipts),
        dependency_target_count=len(dependency_targets),
        dependency_verified_count=sum(
            1 for target in dependency_targets if target.verified_local_layout
        ),
        asset_calibration_receipt_count=len(asset_receipts),
        stream_contract_count=len(stream_contracts),
        command_conformance_receipt_count=len(command_receipts),
        timing_jitter_probe_count=len(timing_receipts),
        safety_preflight_receipt_count=len(safety_receipts),
        operator_recovery_runbook_count=len(operator_runbooks),
        evidence_ledger_count=len(evidence_ledgers),
        all_block_receipts_emitted=all_blocks,
        dependency_discovery_complete=dependency_discovery,
        asset_joint_subset_aligned=asset_aligned,
        stream_contracts_present=bool(stream_contracts),
        command_conformance_dry_run_ready=command_ready,
        local_timing_probe_present=bool(timing_receipts),
        physical_safety_preflight_present=bool(safety_receipts),
        operator_recovery_runbook_present=bool(operator_runbooks),
        honest_sim_or_hardware_evidence_present=False,
        local_pre_purchase_prepared=local_prepared,
        denied_gates=_denied_gates(),
        remaining_key_blockers=list(UNITREE_BRINGUP_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return (
        report,
        block_receipts,
        dependency_targets,
        asset_receipts,
        stream_contracts,
        command_receipts,
        timing_receipts,
        safety_receipts,
        operator_runbooks,
        evidence_ledgers,
    )


def save_phase4_unitree_bringup_readiness(
    output_dir: str | Path,
    *,
    report: Phase4UnitreeBringupReadinessReport,
    block_receipts: list[UnitreeBringupBlockReceipt],
    dependency_targets: list[UnitreeDependencyTarget],
    asset_receipts: list[UnitreeAssetCalibrationReceipt],
    stream_contracts: list[UnitreeStreamContract],
    command_receipts: list[UnitreeCommandConformanceReceipt],
    timing_receipts: list[UnitreeTimingJitterProbeReceipt],
    safety_receipts: list[UnitreeSafetyPreflightReceipt],
    operator_runbooks: list[UnitreeOperatorRecoveryRunbook],
    evidence_ledgers: list[UnitreeSimHardwareEvidenceLedger],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase4_unitree_bringup_readiness_report_v1.json",
        "block_receipts_path": output / "unitree_bringup_block_receipts_v1.jsonl",
        "dependency_targets_path": output / "unitree_dependency_targets_v1.jsonl",
        "asset_receipts_path": output / "unitree_asset_calibration_receipts_v1.jsonl",
        "stream_contracts_path": output / "unitree_stream_contracts_v1.jsonl",
        "command_receipts_path": output
        / "unitree_command_conformance_receipts_v1.jsonl",
        "timing_receipts_path": output
        / "unitree_timing_jitter_probe_receipts_v1.jsonl",
        "safety_receipts_path": output / "unitree_safety_preflight_receipts_v1.jsonl",
        "operator_runbooks_path": output
        / "unitree_operator_recovery_runbooks_v1.jsonl",
        "evidence_ledgers_path": output
        / "unitree_sim_hardware_evidence_ledgers_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(
        paths["block_receipts_path"], [item.to_dict() for item in block_receipts]
    )
    write_jsonl(
        paths["dependency_targets_path"], [item.to_dict() for item in dependency_targets]
    )
    write_jsonl(
        paths["asset_receipts_path"], [item.to_dict() for item in asset_receipts]
    )
    write_jsonl(
        paths["stream_contracts_path"], [item.to_dict() for item in stream_contracts]
    )
    write_jsonl(
        paths["command_receipts_path"], [item.to_dict() for item in command_receipts]
    )
    write_jsonl(
        paths["timing_receipts_path"], [item.to_dict() for item in timing_receipts]
    )
    write_jsonl(
        paths["safety_receipts_path"], [item.to_dict() for item in safety_receipts]
    )
    write_jsonl(
        paths["operator_runbooks_path"], [item.to_dict() for item in operator_runbooks]
    )
    write_jsonl(
        paths["evidence_ledgers_path"], [item.to_dict() for item in evidence_ledgers]
    )
    return {key: str(path) for key, path in paths.items()}


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_phase4_unitree_bringup_readiness_report(
    path: str | Path,
) -> Phase4UnitreeBringupReadinessReport:
    return Phase4UnitreeBringupReadinessReport.from_dict(_load_json(path))


def load_unitree_bringup_block_receipts(
    path: str | Path,
) -> list[UnitreeBringupBlockReceipt]:
    return [UnitreeBringupBlockReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_dependency_targets(path: str | Path) -> list[UnitreeDependencyTarget]:
    return [UnitreeDependencyTarget.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_asset_calibration_receipts(
    path: str | Path,
) -> list[UnitreeAssetCalibrationReceipt]:
    return [UnitreeAssetCalibrationReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_stream_contracts(path: str | Path) -> list[UnitreeStreamContract]:
    return [UnitreeStreamContract.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_command_conformance_receipts(
    path: str | Path,
) -> list[UnitreeCommandConformanceReceipt]:
    return [
        UnitreeCommandConformanceReceipt.from_dict(row) for row in _load_jsonl(path)
    ]


def load_unitree_timing_jitter_probe_receipts(
    path: str | Path,
) -> list[UnitreeTimingJitterProbeReceipt]:
    return [UnitreeTimingJitterProbeReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_safety_preflight_receipts(
    path: str | Path,
) -> list[UnitreeSafetyPreflightReceipt]:
    return [UnitreeSafetyPreflightReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_operator_recovery_runbooks(
    path: str | Path,
) -> list[UnitreeOperatorRecoveryRunbook]:
    return [UnitreeOperatorRecoveryRunbook.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_sim_hardware_evidence_ledgers(
    path: str | Path,
) -> list[UnitreeSimHardwareEvidenceLedger]:
    return [
        UnitreeSimHardwareEvidenceLedger.from_dict(row) for row in _load_jsonl(path)
    ]
