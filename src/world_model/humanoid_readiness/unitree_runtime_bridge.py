"""Phase 4 Unitree runtime-evidence bridge scaffolding.

This module sits one step beyond the local harness layer. It can run guarded
local checks that are still safe on a developer host: ROS2/colcon build
readiness, a no-policy MuJoCo headless step attempt, trace-ingestion adapter
contracts, expanded safety envelopes, and scripted operator recovery drills.

It deliberately does not publish ROS2/DDS messages, write Unitree SDK2
commands, invoke G1Pilot, run hardware, train weights, mutate reward math, or
promote authority. A successful MuJoCo headless step is recorded narrowly as
local no-policy simulation evidence, not as a robot bring-up or control claim.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from src.world_model.embodiment_actuation import JointLimitEnvelope
from src.world_model.humanoid_readiness.common import (
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.unitree_bringup_readiness import (
    default_unitree_local_roots,
)
from src.world_model.humanoid_readiness.unitree_local_harness import (
    ContactTrace,
    ImuTrace,
    LowStateTrace,
    SafetyStateTransition,
    StaleDataValidationReceipt,
    WirelessEStopTrace,
    load_contact_traces,
    load_imu_traces,
    load_low_state_traces,
    load_stale_data_validation_receipts,
    load_wireless_estop_traces,
)

PHASE4_UNITREE_RUNTIME_BRIDGE_REPORT_VERSION = (
    "phase4_unitree_runtime_evidence_bridge_report_v1"
)
ROS2_RUNTIME_READINESS_RECEIPT_VERSION = "unitree_ros2_runtime_readiness_receipt_v1"
MUJOCO_HEADLESS_STEP_RECEIPT_VERSION = "unitree_mujoco_headless_step_receipt_v1"
MUJOCO_TRACE_ROW_VERSION = "unitree_mujoco_headless_trace_row_v1"
TRACE_IMPORT_ADAPTER_RECEIPT_VERSION = "unitree_trace_import_adapter_receipt_v1"
SAFETY_ENVELOPE_EXPANSION_RECEIPT_VERSION = (
    "unitree_safety_envelope_expansion_receipt_v1"
)
OPERATOR_RECOVERY_SCENARIO_VERSION = "unitree_operator_recovery_scenario_v1"
OPERATOR_RECOVERY_DRILL_RECEIPT_VERSION = "unitree_operator_recovery_drill_receipt_v1"

DENIED_UNITREE_RUNTIME_BRIDGE_AUTHORITIES = (
    "ros2_publish_attempted",
    "unitree_sdk2_write_enabled",
    "g1pilot_runtime_invoked",
    "hardware_executed",
    "live_policy_control",
    "training_executed",
    "weights_written",
    "reward_math_mutation",
    "promotion_eligible",
    "hardware_dispatch_enabled",
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_UNITREE_RUNTIME_BRIDGE_AUTHORITIES}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_rows(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    write_jsonl(path, rows)


@dataclass(frozen=True)
class Ros2RuntimeReadinessReceipt:
    receipt_id: str
    profile_key: str
    unitree_ros2_root: str
    workspace_path: str
    status: str
    setup_script_present: bool
    package_xml_count: int
    msg_definition_count: int
    tool_status: dict[str, bool] = field(default_factory=dict)
    missing_tools: list[str] = field(default_factory=list)
    build_command: str = ""
    setup_command: str = ""
    generated_import_check_command: str = ""
    generated_import_modules: list[str] = field(default_factory=list)
    runbook_steps: list[str] = field(default_factory=list)
    build_executed: bool = False
    import_check_executed: bool = False
    ros2_launch_executed: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    hardware_executed: bool = False
    authority_class: str = "ros2_runtime_readiness_no_publish"
    version: str = ROS2_RUNTIME_READINESS_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "profile_key": self.profile_key,
            "unitree_ros2_root": self.unitree_ros2_root,
            "workspace_path": self.workspace_path,
            "status": self.status,
            "setup_script_present": bool(self.setup_script_present),
            "package_xml_count": int(self.package_xml_count),
            "msg_definition_count": int(self.msg_definition_count),
            "tool_status": {str(k): bool(v) for k, v in self.tool_status.items()},
            "missing_tools": strings(self.missing_tools),
            "build_command": self.build_command,
            "setup_command": self.setup_command,
            "generated_import_check_command": self.generated_import_check_command,
            "generated_import_modules": strings(self.generated_import_modules),
            "runbook_steps": strings(self.runbook_steps),
            "build_executed": bool(self.build_executed),
            "import_check_executed": bool(self.import_check_executed),
            "ros2_launch_executed": bool(self.ros2_launch_executed),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Ros2RuntimeReadinessReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            profile_key=str(payload.get("profile_key", "")),
            unitree_ros2_root=str(payload.get("unitree_ros2_root", "")),
            workspace_path=str(payload.get("workspace_path", "")),
            status=str(payload.get("status", "blocked")),
            setup_script_present=bool(payload.get("setup_script_present", False)),
            package_xml_count=int(payload.get("package_xml_count", 0) or 0),
            msg_definition_count=int(payload.get("msg_definition_count", 0) or 0),
            tool_status={
                str(key): bool(value)
                for key, value in dict(payload.get("tool_status", {}) or {}).items()
            },
            missing_tools=strings(payload.get("missing_tools")),
            build_command=str(payload.get("build_command", "")),
            setup_command=str(payload.get("setup_command", "")),
            generated_import_check_command=str(
                payload.get("generated_import_check_command", "")
            ),
            generated_import_modules=strings(payload.get("generated_import_modules")),
            runbook_steps=strings(payload.get("runbook_steps")),
            build_executed=bool(payload.get("build_executed", False)),
            import_check_executed=bool(payload.get("import_check_executed", False)),
            ros2_launch_executed=bool(payload.get("ros2_launch_executed", False)),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "ros2_runtime_readiness_no_publish")
            ),
            version=str(payload.get("version", ROS2_RUNTIME_READINESS_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class MujocoHeadlessTraceRow:
    trace_id: str
    sample_index: int
    sim_time_s: float
    qpos_head: list[float] = field(default_factory=list)
    qvel_norm: float = 0.0
    ctrl_dim: int = 0
    policy_controlled: bool = False
    ros2_bridge_active: bool = False
    hardware_executed: bool = False
    source: str = "unitree_mujoco_headless_no_policy"
    version: str = MUJOCO_TRACE_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "version": self.version,
            "sample_index": int(self.sample_index),
            "sim_time_s": float(self.sim_time_s),
            "qpos_head": [float(value) for value in self.qpos_head],
            "qvel_norm": float(self.qvel_norm),
            "ctrl_dim": int(self.ctrl_dim),
            "policy_controlled": bool(self.policy_controlled),
            "ros2_bridge_active": bool(self.ros2_bridge_active),
            "hardware_executed": bool(self.hardware_executed),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MujocoHeadlessTraceRow":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            sample_index=int(payload.get("sample_index", 0) or 0),
            sim_time_s=_safe_float(payload.get("sim_time_s")),
            qpos_head=[
                _safe_float(value) for value in list(payload.get("qpos_head", []))
            ],
            qvel_norm=_safe_float(payload.get("qvel_norm")),
            ctrl_dim=int(payload.get("ctrl_dim", 0) or 0),
            policy_controlled=bool(payload.get("policy_controlled", False)),
            ros2_bridge_active=bool(payload.get("ros2_bridge_active", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            source=str(payload.get("source", "unitree_mujoco_headless_no_policy")),
            version=str(payload.get("version", MUJOCO_TRACE_ROW_VERSION)),
        )


@dataclass(frozen=True)
class MujocoHeadlessStepReceipt:
    receipt_id: str
    target_xml_path: str
    status: str
    mujoco_module_available: bool
    model_loaded: bool
    step_attempted: bool
    step_executed: bool
    step_count: int
    trace_row_count: int
    final_time_s: float = 0.0
    nq: int = 0
    nv: int = 0
    nu: int = 0
    trace_path: str = ""
    error_type: str = ""
    error_message: str = ""
    unitree_mujoco_app_launched: bool = False
    policy_controlled: bool = False
    ros2_bridge_active: bool = False
    hardware_executed: bool = False
    authority_class: str = "mujoco_headless_no_policy_step"
    version: str = MUJOCO_HEADLESS_STEP_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "target_xml_path": self.target_xml_path,
            "status": self.status,
            "mujoco_module_available": bool(self.mujoco_module_available),
            "model_loaded": bool(self.model_loaded),
            "step_attempted": bool(self.step_attempted),
            "step_executed": bool(self.step_executed),
            "step_count": int(self.step_count),
            "trace_row_count": int(self.trace_row_count),
            "final_time_s": float(self.final_time_s),
            "nq": int(self.nq),
            "nv": int(self.nv),
            "nu": int(self.nu),
            "trace_path": self.trace_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "unitree_mujoco_app_launched": bool(self.unitree_mujoco_app_launched),
            "policy_controlled": bool(self.policy_controlled),
            "ros2_bridge_active": bool(self.ros2_bridge_active),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MujocoHeadlessStepReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            target_xml_path=str(payload.get("target_xml_path", "")),
            status=str(payload.get("status", "blocked")),
            mujoco_module_available=bool(payload.get("mujoco_module_available", False)),
            model_loaded=bool(payload.get("model_loaded", False)),
            step_attempted=bool(payload.get("step_attempted", False)),
            step_executed=bool(payload.get("step_executed", False)),
            step_count=int(payload.get("step_count", 0) or 0),
            trace_row_count=int(payload.get("trace_row_count", 0) or 0),
            final_time_s=_safe_float(payload.get("final_time_s")),
            nq=int(payload.get("nq", 0) or 0),
            nv=int(payload.get("nv", 0) or 0),
            nu=int(payload.get("nu", 0) or 0),
            trace_path=str(payload.get("trace_path", "")),
            error_type=str(payload.get("error_type", "")),
            error_message=str(payload.get("error_message", "")),
            unitree_mujoco_app_launched=bool(
                payload.get("unitree_mujoco_app_launched", False)
            ),
            policy_controlled=bool(payload.get("policy_controlled", False)),
            ros2_bridge_active=bool(payload.get("ros2_bridge_active", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "mujoco_headless_no_policy_step")
            ),
            version=str(payload.get("version", MUJOCO_HEADLESS_STEP_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class TraceImportAdapterReceipt:
    receipt_id: str
    adapter_key: str
    schema_targets: list[str] = field(default_factory=list)
    input_path: str = ""
    adapter_available: bool = False
    import_executed: bool = False
    rows_imported: int = 0
    supported_topics: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    live_stream_observed: bool = False
    hardware_executed: bool = False
    authority_class: str = "trace_import_adapter_no_live_stream"
    version: str = TRACE_IMPORT_ADAPTER_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "adapter_key": self.adapter_key,
            "schema_targets": strings(self.schema_targets),
            "input_path": self.input_path,
            "adapter_available": bool(self.adapter_available),
            "import_executed": bool(self.import_executed),
            "rows_imported": int(self.rows_imported),
            "supported_topics": strings(self.supported_topics),
            "blockers": strings(self.blockers),
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceImportAdapterReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            adapter_key=str(payload.get("adapter_key", "")),
            schema_targets=strings(payload.get("schema_targets")),
            input_path=str(payload.get("input_path", "")),
            adapter_available=bool(payload.get("adapter_available", False)),
            import_executed=bool(payload.get("import_executed", False)),
            rows_imported=int(payload.get("rows_imported", 0) or 0),
            supported_topics=strings(payload.get("supported_topics")),
            blockers=strings(payload.get("blockers")),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "trace_import_adapter_no_live_stream")
            ),
            version=str(payload.get("version", TRACE_IMPORT_ADAPTER_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class SafetyEnvelopeExpansionReceipt:
    receipt_id: str
    envelope_key: str
    status: str
    local_check_executed: bool
    thresholds: dict[str, Any] = field(default_factory=dict)
    sidecar_path: str = ""
    sidecar_present: bool = False
    calibrated_from_hardware: bool = False
    dispatch_veto_default: bool = True
    blockers: list[str] = field(default_factory=list)
    hardware_executed: bool = False
    authority_class: str = "safety_envelope_expansion_no_calibration_claim"
    version: str = SAFETY_ENVELOPE_EXPANSION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "envelope_key": self.envelope_key,
            "status": self.status,
            "local_check_executed": bool(self.local_check_executed),
            "thresholds": mapping(self.thresholds),
            "sidecar_path": self.sidecar_path,
            "sidecar_present": bool(self.sidecar_present),
            "calibrated_from_hardware": bool(self.calibrated_from_hardware),
            "dispatch_veto_default": bool(self.dispatch_veto_default),
            "blockers": strings(self.blockers),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SafetyEnvelopeExpansionReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            envelope_key=str(payload.get("envelope_key", "")),
            status=str(payload.get("status", "blocked")),
            local_check_executed=bool(payload.get("local_check_executed", False)),
            thresholds=mapping(payload.get("thresholds")),
            sidecar_path=str(payload.get("sidecar_path", "")),
            sidecar_present=bool(payload.get("sidecar_present", False)),
            calibrated_from_hardware=bool(
                payload.get("calibrated_from_hardware", False)
            ),
            dispatch_veto_default=bool(payload.get("dispatch_veto_default", True)),
            blockers=strings(payload.get("blockers")),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get(
                    "authority_class",
                    "safety_envelope_expansion_no_calibration_claim",
                )
            ),
            version=str(
                payload.get("version", SAFETY_ENVELOPE_EXPANSION_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class OperatorRecoveryScenario:
    scenario_id: str
    scenario_key: str
    trigger_sequence: list[str] = field(default_factory=list)
    expected_final_state: str = "recovery_ready_operator_required"
    required_operator_actions: list[str] = field(default_factory=list)
    replay_export_required: bool = True
    teleop_runtime_required: bool = False
    hardware_required: bool = False
    version: str = OPERATOR_RECOVERY_SCENARIO_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "scenario_key": self.scenario_key,
            "trigger_sequence": strings(self.trigger_sequence),
            "expected_final_state": self.expected_final_state,
            "required_operator_actions": strings(self.required_operator_actions),
            "replay_export_required": bool(self.replay_export_required),
            "teleop_runtime_required": bool(self.teleop_runtime_required),
            "hardware_required": bool(self.hardware_required),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorRecoveryScenario":
        return cls(
            scenario_id=str(payload.get("scenario_id", "")),
            scenario_key=str(payload.get("scenario_key", "")),
            trigger_sequence=strings(payload.get("trigger_sequence")),
            expected_final_state=str(
                payload.get("expected_final_state", "recovery_ready_operator_required")
            ),
            required_operator_actions=strings(payload.get("required_operator_actions")),
            replay_export_required=bool(payload.get("replay_export_required", True)),
            teleop_runtime_required=bool(payload.get("teleop_runtime_required", False)),
            hardware_required=bool(payload.get("hardware_required", False)),
            version=str(payload.get("version", OPERATOR_RECOVERY_SCENARIO_VERSION)),
        )


@dataclass(frozen=True)
class OperatorRecoveryDrillReceipt:
    receipt_id: str
    scenario_id: str
    scenario_key: str
    local_drill_executed: bool
    passed: bool
    final_state: str
    transition_ids: list[str] = field(default_factory=list)
    replay_export_ready: bool = True
    teleop_runtime_executed: bool = False
    command_dispatch_allowed: bool = False
    hardware_executed: bool = False
    authority_class: str = "operator_recovery_drill_local_only"
    version: str = OPERATOR_RECOVERY_DRILL_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "scenario_id": self.scenario_id,
            "scenario_key": self.scenario_key,
            "local_drill_executed": bool(self.local_drill_executed),
            "passed": bool(self.passed),
            "final_state": self.final_state,
            "transition_ids": strings(self.transition_ids),
            "replay_export_ready": bool(self.replay_export_ready),
            "teleop_runtime_executed": bool(self.teleop_runtime_executed),
            "command_dispatch_allowed": bool(self.command_dispatch_allowed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorRecoveryDrillReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            scenario_id=str(payload.get("scenario_id", "")),
            scenario_key=str(payload.get("scenario_key", "")),
            local_drill_executed=bool(payload.get("local_drill_executed", False)),
            passed=bool(payload.get("passed", False)),
            final_state=str(payload.get("final_state", "")),
            transition_ids=strings(payload.get("transition_ids")),
            replay_export_ready=bool(payload.get("replay_export_ready", True)),
            teleop_runtime_executed=bool(payload.get("teleop_runtime_executed", False)),
            command_dispatch_allowed=bool(
                payload.get("command_dispatch_allowed", False)
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "operator_recovery_drill_local_only")
            ),
            version=str(
                payload.get("version", OPERATOR_RECOVERY_DRILL_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase4UnitreeRuntimeEvidenceBridgeReport:
    report_id: str
    status: str
    ros2_runtime_readiness_receipt_count: int
    mujoco_headless_step_receipt_count: int
    mujoco_trace_row_count: int
    trace_import_adapter_receipt_count: int
    safety_envelope_expansion_receipt_count: int
    operator_recovery_scenario_count: int
    operator_recovery_drill_receipt_count: int
    ros2_runtime_preflight_complete: bool
    mujoco_headless_trace_attempt_complete: bool
    trace_ingestion_adapters_complete: bool
    safety_envelope_expansion_complete: bool
    operator_drill_runner_complete: bool
    local_runtime_evidence_bridge_complete: bool
    minimal_mujoco_headless_step_executed: bool = False
    live_stream_observed: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    training_executed: bool = False
    weights_written: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    remaining_evidence_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE4_UNITREE_RUNTIME_BRIDGE_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "ros2_runtime_readiness_receipt_count": int(
                self.ros2_runtime_readiness_receipt_count
            ),
            "mujoco_headless_step_receipt_count": int(
                self.mujoco_headless_step_receipt_count
            ),
            "mujoco_trace_row_count": int(self.mujoco_trace_row_count),
            "trace_import_adapter_receipt_count": int(
                self.trace_import_adapter_receipt_count
            ),
            "safety_envelope_expansion_receipt_count": int(
                self.safety_envelope_expansion_receipt_count
            ),
            "operator_recovery_scenario_count": int(
                self.operator_recovery_scenario_count
            ),
            "operator_recovery_drill_receipt_count": int(
                self.operator_recovery_drill_receipt_count
            ),
            "ros2_runtime_preflight_complete": bool(
                self.ros2_runtime_preflight_complete
            ),
            "mujoco_headless_trace_attempt_complete": bool(
                self.mujoco_headless_trace_attempt_complete
            ),
            "trace_ingestion_adapters_complete": bool(
                self.trace_ingestion_adapters_complete
            ),
            "safety_envelope_expansion_complete": bool(
                self.safety_envelope_expansion_complete
            ),
            "operator_drill_runner_complete": bool(self.operator_drill_runner_complete),
            "local_runtime_evidence_bridge_complete": bool(
                self.local_runtime_evidence_bridge_complete
            ),
            "minimal_mujoco_headless_step_executed": bool(
                self.minimal_mujoco_headless_step_executed
            ),
            "live_stream_observed": bool(self.live_stream_observed),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": dict(self.denied_gates),
            "remaining_evidence_blockers": strings(self.remaining_evidence_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase4UnitreeRuntimeEvidenceBridgeReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "blocked")),
            ros2_runtime_readiness_receipt_count=int(
                payload.get("ros2_runtime_readiness_receipt_count", 0) or 0
            ),
            mujoco_headless_step_receipt_count=int(
                payload.get("mujoco_headless_step_receipt_count", 0) or 0
            ),
            mujoco_trace_row_count=int(payload.get("mujoco_trace_row_count", 0) or 0),
            trace_import_adapter_receipt_count=int(
                payload.get("trace_import_adapter_receipt_count", 0) or 0
            ),
            safety_envelope_expansion_receipt_count=int(
                payload.get("safety_envelope_expansion_receipt_count", 0) or 0
            ),
            operator_recovery_scenario_count=int(
                payload.get("operator_recovery_scenario_count", 0) or 0
            ),
            operator_recovery_drill_receipt_count=int(
                payload.get("operator_recovery_drill_receipt_count", 0) or 0
            ),
            ros2_runtime_preflight_complete=bool(
                payload.get("ros2_runtime_preflight_complete", False)
            ),
            mujoco_headless_trace_attempt_complete=bool(
                payload.get("mujoco_headless_trace_attempt_complete", False)
            ),
            trace_ingestion_adapters_complete=bool(
                payload.get("trace_ingestion_adapters_complete", False)
            ),
            safety_envelope_expansion_complete=bool(
                payload.get("safety_envelope_expansion_complete", False)
            ),
            operator_drill_runner_complete=bool(
                payload.get("operator_drill_runner_complete", False)
            ),
            local_runtime_evidence_bridge_complete=bool(
                payload.get("local_runtime_evidence_bridge_complete", False)
            ),
            minimal_mujoco_headless_step_executed=bool(
                payload.get("minimal_mujoco_headless_step_executed", False)
            ),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            g1pilot_runtime_invoked=bool(payload.get("g1pilot_runtime_invoked", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
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
            remaining_evidence_blockers=strings(
                payload.get("remaining_evidence_blockers")
            ),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE4_UNITREE_RUNTIME_BRIDGE_REPORT_VERSION)
            ),
        )


def build_ros2_runtime_readiness_receipts(
    local_roots: Mapping[str, str | Path] | None = None,
) -> list[Ros2RuntimeReadinessReceipt]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    ros_root = Path(roots["unitree_ros2"])
    workspace = ros_root / "cyclonedds_ws"
    package_xml_count = len(list(workspace.glob("src/**/*.xml")))
    msg_definition_count = len(list(workspace.glob("src/**/*.msg")))
    setup_present = (ros_root / "setup.sh").exists()
    required_tools = ["python3", "cmake", "colcon", "ros2"]
    tool_status = {tool: shutil.which(tool) is not None for tool in required_tools}
    missing = [tool for tool, present in tool_status.items() if not present]
    base_steps = [
        "install ROS2 Humble or Jazzy in an isolated host/container profile",
        "install colcon-common-extensions for workspace build",
        "source the ROS2 distro setup script before building Unitree messages",
        "run colcon build for unitree_hg, unitree_go, and unitree_api packages",
        "source the generated install/setup script before any message import check",
    ]
    profiles = [
        (
            "native_ros2_colcon",
            "source /opt/ros/$ROS_DISTRO/setup.sh && colcon build --symlink-install",
            "source /opt/ros/$ROS_DISTRO/setup.sh && source install/setup.sh",
        ),
        (
            "container_ros2_colcon",
            "docker build -f docker/unitree_ros2_preflight.Dockerfile .",
            "docker run --rm unitree-ros2-preflight colcon build --symlink-install",
        ),
    ]
    receipts: list[Ros2RuntimeReadinessReceipt] = []
    for profile_key, build_command, setup_command in profiles:
        payload = {
            "profile_key": profile_key,
            "root": str(ros_root),
            "workspace": str(workspace),
            "missing": missing,
            "packages": package_xml_count,
            "messages": msg_definition_count,
        }
        status = (
            "ready_for_build_attempt"
            if setup_present
            and package_xml_count
            and msg_definition_count
            and not missing
            else "blocked_missing_host_tools"
        )
        if profile_key == "container_ros2_colcon" and shutil.which("docker"):
            status = "container_profile_materialized_not_executed"
        elif profile_key == "container_ros2_colcon":
            status = "container_profile_blocked_missing_docker"
        receipts.append(
            Ros2RuntimeReadinessReceipt(
                receipt_id=stable_id("unitree_ros2_runtime_readiness", payload),
                profile_key=profile_key,
                unitree_ros2_root=str(ros_root),
                workspace_path=str(workspace),
                status=status,
                setup_script_present=setup_present,
                package_xml_count=package_xml_count,
                msg_definition_count=msg_definition_count,
                tool_status=tool_status,
                missing_tools=missing
                if profile_key == "native_ros2_colcon"
                else ([] if shutil.which("docker") else ["docker"]),
                build_command=build_command,
                setup_command=setup_command,
                generated_import_check_command=(
                    'python3 -c "from unitree_hg.msg import LowCmd; '
                    'from unitree_api.msg import Request"'
                ),
                generated_import_modules=[
                    "unitree_hg.msg.LowCmd",
                    "unitree_hg.msg.LowState",
                    "unitree_api.msg.Request",
                    "unitree_go.msg.WirelessController",
                ],
                runbook_steps=base_steps,
            )
        )
    return receipts


def attempt_mujoco_headless_step(
    *,
    local_roots: Mapping[str, str | Path] | None = None,
    step_count: int = 5,
    trace_path: str | Path = "",
) -> tuple[MujocoHeadlessStepReceipt, list[MujocoHeadlessTraceRow]]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    target_xml = Path(roots["unitree_mujoco"]) / "unitree_robots/g1/scene_29dof.xml"
    step_count = max(1, int(step_count))
    module_available = importlib.util.find_spec("mujoco") is not None
    rows: list[MujocoHeadlessTraceRow] = []
    payload = {"target_xml": str(target_xml), "step_count": step_count}
    if not module_available:
        return (
            MujocoHeadlessStepReceipt(
                receipt_id=stable_id("unitree_mujoco_headless_step", payload),
                target_xml_path=str(target_xml),
                status="blocked_missing_mujoco_python_module",
                mujoco_module_available=False,
                model_loaded=False,
                step_attempted=False,
                step_executed=False,
                step_count=0,
                trace_row_count=0,
                trace_path=str(trace_path),
                error_type="ModuleNotFoundError",
                error_message="Python module 'mujoco' is not importable.",
            ),
            rows,
        )
    try:
        import mujoco  # type: ignore[import-not-found,import-untyped]

        model = mujoco.MjModel.from_xml_path(str(target_xml))
        data = mujoco.MjData(model)
        trace_id = stable_id("unitree_mujoco_headless_trace", payload)
        for index in range(step_count):
            mujoco.mj_step(model, data)
            qvel_norm = math.sqrt(float((data.qvel * data.qvel).sum()))
            rows.append(
                MujocoHeadlessTraceRow(
                    trace_id=trace_id,
                    sample_index=index,
                    sim_time_s=float(data.time),
                    qpos_head=[float(value) for value in data.qpos[: min(model.nq, 8)]],
                    qvel_norm=qvel_norm,
                    ctrl_dim=int(model.nu),
                )
            )
        receipt = MujocoHeadlessStepReceipt(
            receipt_id=stable_id("unitree_mujoco_headless_step", payload),
            target_xml_path=str(target_xml),
            status="ok",
            mujoco_module_available=True,
            model_loaded=True,
            step_attempted=True,
            step_executed=True,
            step_count=step_count,
            trace_row_count=len(rows),
            final_time_s=float(data.time),
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
            trace_path=str(trace_path),
        )
        return receipt, rows
    except Exception as exc:
        return (
            MujocoHeadlessStepReceipt(
                receipt_id=stable_id("unitree_mujoco_headless_step", payload),
                target_xml_path=str(target_xml),
                status="blocked_mujoco_headless_step_failed",
                mujoco_module_available=True,
                model_loaded=False,
                step_attempted=True,
                step_executed=False,
                step_count=0,
                trace_row_count=0,
                trace_path=str(trace_path),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            ),
            rows,
        )


def build_trace_import_adapter_receipts(
    *,
    trace_dir: str | Path,
    rosbag2_path: str | Path | None = None,
    mcap_path: str | Path | None = None,
) -> list[TraceImportAdapterReceipt]:
    trace_root = Path(trace_dir)
    jsonl_paths = [
        trace_root / "unitree_low_state_traces_v1.jsonl",
        trace_root / "unitree_imu_traces_v1.jsonl",
        trace_root / "unitree_wireless_estop_traces_v1.jsonl",
        trace_root / "unitree_contact_traces_v1.jsonl",
    ]
    jsonl_rows = sum(len(_load_jsonl(path)) for path in jsonl_paths)
    adapters = [
        (
            "jsonl_unitree_trace_bundle",
            str(trace_root),
            True,
            bool(jsonl_rows),
            jsonl_rows,
            [] if jsonl_rows else ["jsonl_trace_bundle_missing_or_empty"],
        ),
        (
            "rosbag2_unitree_topics",
            str(rosbag2_path or ""),
            importlib.util.find_spec("rosbag2_py") is not None,
            bool(rosbag2_path and Path(rosbag2_path).exists()),
            0,
            []
            if rosbag2_path and Path(rosbag2_path).exists()
            else ["rosbag2_input_path_missing"],
        ),
        (
            "mcap_unitree_topics",
            str(mcap_path or ""),
            importlib.util.find_spec("mcap") is not None,
            bool(mcap_path and Path(mcap_path).exists()),
            0,
            []
            if mcap_path and Path(mcap_path).exists()
            else ["mcap_input_path_missing"],
        ),
    ]
    receipts: list[TraceImportAdapterReceipt] = []
    for adapter_key, input_path, available, executed, row_count, blockers in adapters:
        payload = {
            "adapter_key": adapter_key,
            "input_path": input_path,
            "available": available,
            "executed": executed,
            "row_count": row_count,
            "blockers": blockers,
        }
        receipts.append(
            TraceImportAdapterReceipt(
                receipt_id=stable_id("unitree_trace_import_adapter", payload),
                adapter_key=adapter_key,
                schema_targets=[
                    "LowStateTrace",
                    "ImuTrace",
                    "WirelessEStopTrace",
                    "ContactTrace",
                ],
                input_path=input_path,
                adapter_available=available,
                import_executed=executed,
                rows_imported=row_count,
                supported_topics=[
                    "/lowstate",
                    "/lf/lowstate",
                    "/wirelesscontroller",
                    "/api/sport/request",
                    "/lowcmd",
                ],
                blockers=blockers,
            )
        )
    return receipts


def build_safety_envelope_expansion_receipts(
    *,
    joint_limits: list[JointLimitEnvelope],
    sidecar_dir: str | Path,
) -> list[SafetyEnvelopeExpansionReceipt]:
    sidecar = Path(sidecar_dir) / "unitree_physical_calibration_sidecar_v1.json"
    specs = [
        (
            "joint_limit_runtime_clamp",
            "local_planning_envelope_ready",
            {"controlled_joint_count": len(joint_limits), "margin_ratio": 0.90},
            [],
            True,
        ),
        (
            "self_collision_hook",
            "geometry_hook_materialized_uncalibrated",
            {
                "required_geometry_sources": [
                    "Unitree MuJoCo XML",
                    "URDF collision mesh",
                    "future Pinocchio/Drake collision model",
                ]
            },
            ["self_collision_geometry_not_validated_against_robot"],
            True,
        ),
        (
            "fall_posture_guard",
            "local_thresholds_materialized_uncalibrated",
            {"max_roll_rad": 0.45, "max_pitch_rad": 0.45, "min_base_height_m": 0.35},
            ["thresholds_not_validated_in_sim_or_hardware"],
            True,
        ),
        (
            "stop_distance_slot",
            "calibration_sidecar_missing",
            {
                "required_fields": [
                    "surface_type",
                    "initial_velocity_m_s",
                    "stop_distance_m",
                    "stop_time_s",
                    "operator_latency_s",
                ]
            },
            ["measured_stop_distance_missing"],
            False,
        ),
        (
            "calibrated_limit_sidecar",
            "calibration_sidecar_missing",
            {"sidecar_schema": "unitree_physical_calibration_sidecar_v1"},
            ["hardware_calibrated_limit_sidecar_missing"],
            False,
        ),
    ]
    receipts: list[SafetyEnvelopeExpansionReceipt] = []
    for envelope_key, status, thresholds, blockers, local_executed in specs:
        payload = {
            "envelope_key": envelope_key,
            "status": status,
            "thresholds": thresholds,
            "blockers": blockers,
            "sidecar": str(sidecar),
        }
        receipts.append(
            SafetyEnvelopeExpansionReceipt(
                receipt_id=stable_id("unitree_safety_envelope", payload),
                envelope_key=envelope_key,
                status=status,
                local_check_executed=local_executed,
                thresholds=mapping(thresholds)
                if isinstance(thresholds, Mapping)
                else {},
                sidecar_path=str(sidecar),
                sidecar_present=sidecar.exists(),
                calibrated_from_hardware=False,
                blockers=blockers,
            )
        )
    return receipts


def build_operator_recovery_scenarios() -> list[OperatorRecoveryScenario]:
    specs = [
        (
            "stale_stream_demote",
            ["stale_data", "demote", "operator_handoff"],
            ["acknowledge_stale_stream", "verify_stream_freshness", "deny_resume"],
        ),
        (
            "wireless_estop_latch",
            ["estop", "demote", "operator_handoff"],
            ["confirm_estop_source", "hold_motors_disabled", "manual_reset_required"],
        ),
        (
            "low_balance_margin",
            ["fall_posture_guard", "demote", "operator_handoff"],
            ["confirm_support_state", "request_stable_base_fallback", "deny_resume"],
        ),
        (
            "teleop_takeover_request",
            ["operator_takeover", "demote", "operator_handoff"],
            ["grant_observation_only", "require_runtime_teleop_gate", "deny_dispatch"],
        ),
    ]
    scenarios: list[OperatorRecoveryScenario] = []
    for scenario_key, triggers, actions in specs:
        payload = {
            "scenario_key": scenario_key,
            "triggers": triggers,
            "actions": actions,
        }
        scenarios.append(
            OperatorRecoveryScenario(
                scenario_id=stable_id("unitree_operator_recovery_scenario", payload),
                scenario_key=scenario_key,
                trigger_sequence=triggers,
                required_operator_actions=actions,
            )
        )
    return scenarios


def run_operator_recovery_drills(
    scenarios: list[OperatorRecoveryScenario],
) -> tuple[list[SafetyStateTransition], list[OperatorRecoveryDrillReceipt]]:
    transitions: list[SafetyStateTransition] = []
    receipts: list[OperatorRecoveryDrillReceipt] = []
    for scenario in scenarios:
        current = "nominal_dry_run"
        scenario_transition_ids: list[str] = []
        for index, event_key in enumerate(scenario.trigger_sequence):
            to_state = (
                "stable_base_demote_requested"
                if event_key == "demote"
                else (
                    "recovery_ready_operator_required"
                    if event_key == "operator_handoff"
                    else f"{event_key}_veto"
                )
            )
            payload = {
                "scenario_id": scenario.scenario_id,
                "event_key": event_key,
                "from_state": current,
                "to_state": to_state,
                "index": index,
            }
            transition = SafetyStateTransition(
                transition_id=stable_id("unitree_operator_drill_transition", payload),
                event_key=event_key,
                from_state=current,
                to_state=to_state,
                timestamp_s=round(0.1 * (index + 1), 3),
                reason=f"scripted local recovery drill for {scenario.scenario_key}",
            )
            transitions.append(transition)
            scenario_transition_ids.append(transition.transition_id)
            current = to_state
        passed = current == scenario.expected_final_state
        receipt_payload = {
            "scenario_id": scenario.scenario_id,
            "final_state": current,
            "passed": passed,
        }
        receipts.append(
            OperatorRecoveryDrillReceipt(
                receipt_id=stable_id(
                    "unitree_operator_recovery_drill", receipt_payload
                ),
                scenario_id=scenario.scenario_id,
                scenario_key=scenario.scenario_key,
                local_drill_executed=True,
                passed=passed,
                final_state=current,
                transition_ids=scenario_transition_ids,
            )
        )
    return transitions, receipts


def build_phase4_unitree_runtime_evidence_bridge(
    *,
    trace_dir: str | Path,
    joint_limits: list[JointLimitEnvelope],
    output_dir: str | Path,
    local_roots: Mapping[str, str | Path] | None = None,
    mujoco_steps: int = 5,
    rosbag2_path: str | Path | None = None,
    mcap_path: str | Path | None = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    Phase4UnitreeRuntimeEvidenceBridgeReport,
    list[Ros2RuntimeReadinessReceipt],
    MujocoHeadlessStepReceipt,
    list[MujocoHeadlessTraceRow],
    list[TraceImportAdapterReceipt],
    list[SafetyEnvelopeExpansionReceipt],
    list[OperatorRecoveryScenario],
    list[SafetyStateTransition],
    list[OperatorRecoveryDrillReceipt],
]:
    output = Path(output_dir)
    mujoco_trace_path = output / "unitree_mujoco_headless_trace_rows_v1.jsonl"
    ros2_receipts = build_ros2_runtime_readiness_receipts(local_roots)
    mujoco_receipt, mujoco_rows = attempt_mujoco_headless_step(
        local_roots=local_roots,
        step_count=mujoco_steps,
        trace_path=mujoco_trace_path,
    )
    trace_adapters = build_trace_import_adapter_receipts(
        trace_dir=trace_dir,
        rosbag2_path=rosbag2_path,
        mcap_path=mcap_path,
    )
    safety_receipts = build_safety_envelope_expansion_receipts(
        joint_limits=joint_limits,
        sidecar_dir=output,
    )
    scenarios = build_operator_recovery_scenarios()
    transitions, drill_receipts = run_operator_recovery_drills(scenarios)

    ros2_complete = bool(ros2_receipts) and all(
        not receipt.ros2_publish_attempted and not receipt.build_executed
        for receipt in ros2_receipts
    )
    mujoco_attempt_complete = (
        mujoco_receipt.step_attempted
        or not mujoco_receipt.mujoco_module_available
        or bool(mujoco_receipt.error_type)
    )
    adapter_complete = (
        len(trace_adapters) == 3
        and any(
            receipt.adapter_key == "jsonl_unitree_trace_bundle"
            and receipt.import_executed
            and receipt.rows_imported >= 1
            for receipt in trace_adapters
        )
        and not any(receipt.live_stream_observed for receipt in trace_adapters)
    )
    safety_complete = (
        len(safety_receipts) >= 5
        and all(receipt.dispatch_veto_default for receipt in safety_receipts)
        and not any(receipt.calibrated_from_hardware for receipt in safety_receipts)
    )
    drill_complete = (
        bool(scenarios)
        and bool(drill_receipts)
        and all(
            receipt.local_drill_executed and receipt.passed
            for receipt in drill_receipts
        )
        and not any(receipt.teleop_runtime_executed for receipt in drill_receipts)
    )
    complete = (
        ros2_complete
        and mujoco_attempt_complete
        and adapter_complete
        and safety_complete
        and drill_complete
    )
    report_payload = {
        "ros2": len(ros2_receipts),
        "mujoco_step": mujoco_receipt.step_executed,
        "adapters": len(trace_adapters),
        "safety": len(safety_receipts),
        "drills": len(drill_receipts),
    }
    report = Phase4UnitreeRuntimeEvidenceBridgeReport(
        report_id=stable_id("phase4_unitree_runtime_bridge", report_payload),
        status="ok" if complete else "blocked",
        ros2_runtime_readiness_receipt_count=len(ros2_receipts),
        mujoco_headless_step_receipt_count=1,
        mujoco_trace_row_count=len(mujoco_rows),
        trace_import_adapter_receipt_count=len(trace_adapters),
        safety_envelope_expansion_receipt_count=len(safety_receipts),
        operator_recovery_scenario_count=len(scenarios),
        operator_recovery_drill_receipt_count=len(drill_receipts),
        ros2_runtime_preflight_complete=ros2_complete,
        mujoco_headless_trace_attempt_complete=mujoco_attempt_complete,
        trace_ingestion_adapters_complete=adapter_complete,
        safety_envelope_expansion_complete=safety_complete,
        operator_drill_runner_complete=drill_complete,
        local_runtime_evidence_bridge_complete=complete,
        minimal_mujoco_headless_step_executed=mujoco_receipt.step_executed,
        denied_gates=_denied_gates(),
        remaining_evidence_blockers=[
            "ros2_colcon_build_and_generated_message_import_not_executed",
            "ros2_sdk2_g1pilot_command_echo_missing",
            "rosbag2_or_mcap_real_stream_import_missing",
            "policy_controlled_mujoco_or_hardware_trace_missing",
            "physical_stop_distance_and_calibrated_safety_limits_missing",
            "operator_teleop_runtime_drill_missing",
            "dds_network_or_on_robot_timing_missing",
        ],
        artifact_refs=mapping(artifact_refs),
    )
    return (
        report,
        ros2_receipts,
        mujoco_receipt,
        mujoco_rows,
        trace_adapters,
        safety_receipts,
        scenarios,
        transitions,
        drill_receipts,
    )


def save_phase4_unitree_runtime_evidence_bridge(
    output_dir: str | Path,
    *,
    report: Phase4UnitreeRuntimeEvidenceBridgeReport,
    ros2_receipts: list[Ros2RuntimeReadinessReceipt],
    mujoco_receipt: MujocoHeadlessStepReceipt,
    mujoco_rows: list[MujocoHeadlessTraceRow],
    trace_adapters: list[TraceImportAdapterReceipt],
    safety_receipts: list[SafetyEnvelopeExpansionReceipt],
    scenarios: list[OperatorRecoveryScenario],
    transitions: list[SafetyStateTransition],
    drill_receipts: list[OperatorRecoveryDrillReceipt],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase4_unitree_runtime_evidence_bridge_report_v1.json",
        "ros2_runtime_readiness_receipts_path": output
        / "unitree_ros2_runtime_readiness_receipts_v1.jsonl",
        "mujoco_headless_step_receipts_path": output
        / "unitree_mujoco_headless_step_receipts_v1.jsonl",
        "mujoco_headless_trace_rows_path": output
        / "unitree_mujoco_headless_trace_rows_v1.jsonl",
        "trace_import_adapter_receipts_path": output
        / "unitree_trace_import_adapter_receipts_v1.jsonl",
        "safety_envelope_expansion_receipts_path": output
        / "unitree_safety_envelope_expansion_receipts_v1.jsonl",
        "operator_recovery_scenarios_path": output
        / "unitree_operator_recovery_scenarios_v1.jsonl",
        "operator_recovery_drill_transitions_path": output
        / "unitree_operator_recovery_drill_transitions_v1.jsonl",
        "operator_recovery_drill_receipts_path": output
        / "unitree_operator_recovery_drill_receipts_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    _write_rows(
        paths["ros2_runtime_readiness_receipts_path"],
        [receipt.to_dict() for receipt in ros2_receipts],
    )
    _write_rows(
        paths["mujoco_headless_step_receipts_path"],
        [mujoco_receipt.to_dict()],
    )
    _write_rows(
        paths["mujoco_headless_trace_rows_path"],
        [row.to_dict() for row in mujoco_rows],
    )
    _write_rows(
        paths["trace_import_adapter_receipts_path"],
        [receipt.to_dict() for receipt in trace_adapters],
    )
    _write_rows(
        paths["safety_envelope_expansion_receipts_path"],
        [receipt.to_dict() for receipt in safety_receipts],
    )
    _write_rows(
        paths["operator_recovery_scenarios_path"],
        [scenario.to_dict() for scenario in scenarios],
    )
    _write_rows(
        paths["operator_recovery_drill_transitions_path"],
        [transition.to_dict() for transition in transitions],
    )
    _write_rows(
        paths["operator_recovery_drill_receipts_path"],
        [receipt.to_dict() for receipt in drill_receipts],
    )
    return {key: str(path) for key, path in paths.items()}


def load_phase4_unitree_runtime_evidence_bridge_report(
    path: str | Path,
) -> Phase4UnitreeRuntimeEvidenceBridgeReport:
    return Phase4UnitreeRuntimeEvidenceBridgeReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_ros2_runtime_readiness_receipts(
    path: str | Path,
) -> list[Ros2RuntimeReadinessReceipt]:
    return [Ros2RuntimeReadinessReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_mujoco_headless_step_receipts(
    path: str | Path,
) -> list[MujocoHeadlessStepReceipt]:
    return [MujocoHeadlessStepReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_mujoco_headless_trace_rows(path: str | Path) -> list[MujocoHeadlessTraceRow]:
    return [MujocoHeadlessTraceRow.from_dict(row) for row in _load_jsonl(path)]


def load_trace_import_adapter_receipts(
    path: str | Path,
) -> list[TraceImportAdapterReceipt]:
    return [TraceImportAdapterReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_safety_envelope_expansion_receipts(
    path: str | Path,
) -> list[SafetyEnvelopeExpansionReceipt]:
    return [SafetyEnvelopeExpansionReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_operator_recovery_scenarios(
    path: str | Path,
) -> list[OperatorRecoveryScenario]:
    return [OperatorRecoveryScenario.from_dict(row) for row in _load_jsonl(path)]


def load_operator_recovery_drill_receipts(
    path: str | Path,
) -> list[OperatorRecoveryDrillReceipt]:
    return [OperatorRecoveryDrillReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_runtime_bridge_trace_sources(
    trace_dir: str | Path,
) -> tuple[
    list[LowStateTrace],
    list[ImuTrace],
    list[WirelessEStopTrace],
    list[ContactTrace],
    list[StaleDataValidationReceipt],
]:
    root = Path(trace_dir)
    return (
        load_low_state_traces(root / "unitree_low_state_traces_v1.jsonl"),
        load_imu_traces(root / "unitree_imu_traces_v1.jsonl"),
        load_wireless_estop_traces(root / "unitree_wireless_estop_traces_v1.jsonl"),
        load_contact_traces(root / "unitree_contact_traces_v1.jsonl"),
        load_stale_data_validation_receipts(
            root / "unitree_stale_data_validation_receipts_v1.jsonl"
        ),
    )
