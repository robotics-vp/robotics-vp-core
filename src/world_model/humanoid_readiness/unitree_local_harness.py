"""Runnable local Unitree/G1 harnesses for Phase 4 readiness.

The harnesses in this module deliberately exercise contracts without pretending
to operate a robot: synthetic trace streams, no-publish command shape checks,
mock producer/consumer timing, safety/recovery state transitions, and runtime
preflight receipts for local Unitree roots.

No ROS2/DDS publishing, Unitree SDK2 writes, G1Pilot invocation, MuJoCo launch,
hardware execution, training, reward mutation, or promotion happens here.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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
)
from src.world_model.humanoid_readiness.unitree_bringup_readiness import (
    DENIED_UNITREE_BRINGUP_AUTHORITIES,
    default_unitree_local_roots,
)

PHASE4_UNITREE_LOCAL_HARNESS_REPORT_VERSION = (
    "phase4_unitree_local_harness_report_v1"
)
LOW_STATE_TRACE_VERSION = "unitree_low_state_trace_v1"
IMU_TRACE_VERSION = "unitree_imu_trace_v1"
WIRELESS_ESTOP_TRACE_VERSION = "unitree_wireless_estop_trace_v1"
CONTACT_TRACE_VERSION = "unitree_contact_trace_v1"
TRACE_REPLAY_RECEIPT_VERSION = "unitree_trace_replay_receipt_v1"
MOCK_RECEIVER_RECEIPT_VERSION = "unitree_mock_receiver_receipt_v1"
STALE_DATA_VALIDATION_RECEIPT_VERSION = "unitree_stale_data_validation_receipt_v1"
ROS_MESSAGE_DEFINITION_VERSION = "unitree_ros_message_definition_v1"
COMMAND_SHAPE_VALIDATION_RECEIPT_VERSION = (
    "unitree_command_shape_validation_receipt_v1"
)
MOCK_TIMING_RUN_RECEIPT_VERSION = "unitree_mock_timing_run_receipt_v1"
WATCHDOG_DEMOTION_RECEIPT_VERSION = "unitree_watchdog_demotion_receipt_v1"
SAFETY_STATE_TRANSITION_VERSION = "unitree_safety_state_transition_v1"
SYNTHETIC_SAFETY_DRILL_RECEIPT_VERSION = "unitree_synthetic_safety_drill_receipt_v1"
RUNTIME_PREFLIGHT_RECEIPT_VERSION = "unitree_runtime_preflight_receipt_v1"

DENIED_UNITREE_LOCAL_HARNESS_AUTHORITIES = tuple(
    dict.fromkeys(
        (
            *DENIED_UNITREE_BRINGUP_AUTHORITIES,
            "rosbag_replay_claimed",
            "dds_runtime_observed",
            "real_lowstate_observed",
            "real_estop_observed",
            "mujoco_launch_executed",
            "ros2_launch_executed",
            "g1pilot_launch_executed",
        )
    )
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_UNITREE_LOCAL_HARNESS_AUTHORITIES}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl(path, rows)


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class LowStateTrace:
    trace_id: str
    sample_index: int
    timestamp_s: float
    tick: int
    joint_positions: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    motor_count: int = 0
    source: str = "synthetic_local_trace"
    live_stream_observed: bool = False
    hardware_executed: bool = False
    version: str = LOW_STATE_TRACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "version": self.version,
            "sample_index": int(self.sample_index),
            "timestamp_s": float(self.timestamp_s),
            "tick": int(self.tick),
            "joint_positions": dict(self.joint_positions),
            "joint_velocities": dict(self.joint_velocities),
            "motor_count": int(self.motor_count),
            "source": self.source,
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LowStateTrace":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            sample_index=int(payload.get("sample_index", 0) or 0),
            timestamp_s=_safe_float(payload.get("timestamp_s")),
            tick=int(payload.get("tick", 0) or 0),
            joint_positions={
                str(key): _safe_float(value)
                for key, value in dict(payload.get("joint_positions", {}) or {}).items()
            },
            joint_velocities={
                str(key): _safe_float(value)
                for key, value in dict(payload.get("joint_velocities", {}) or {}).items()
            },
            motor_count=int(payload.get("motor_count", 0) or 0),
            source=str(payload.get("source", "synthetic_local_trace")),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            version=str(payload.get("version", LOW_STATE_TRACE_VERSION)),
        )


@dataclass(frozen=True)
class ImuTrace:
    trace_id: str
    sample_index: int
    timestamp_s: float
    quaternion_xyzw: list[float] = field(default_factory=list)
    gyro_rad_s: list[float] = field(default_factory=list)
    accel_m_s2: list[float] = field(default_factory=list)
    source: str = "synthetic_local_trace"
    live_stream_observed: bool = False
    hardware_executed: bool = False
    version: str = IMU_TRACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "version": self.version,
            "sample_index": int(self.sample_index),
            "timestamp_s": float(self.timestamp_s),
            "quaternion_xyzw": [float(value) for value in self.quaternion_xyzw],
            "gyro_rad_s": [float(value) for value in self.gyro_rad_s],
            "accel_m_s2": [float(value) for value in self.accel_m_s2],
            "source": self.source,
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImuTrace":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            sample_index=int(payload.get("sample_index", 0) or 0),
            timestamp_s=_safe_float(payload.get("timestamp_s")),
            quaternion_xyzw=[
                _safe_float(value) for value in list(payload.get("quaternion_xyzw", []))
            ],
            gyro_rad_s=[_safe_float(value) for value in list(payload.get("gyro_rad_s", []))],
            accel_m_s2=[
                _safe_float(value) for value in list(payload.get("accel_m_s2", []))
            ],
            source=str(payload.get("source", "synthetic_local_trace")),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            version=str(payload.get("version", IMU_TRACE_VERSION)),
        )


@dataclass(frozen=True)
class WirelessEStopTrace:
    trace_id: str
    sample_index: int
    timestamp_s: float
    keys_bitmask: int
    estop_pressed: bool
    operator_takeover_requested: bool = False
    source: str = "synthetic_local_trace"
    live_stream_observed: bool = False
    hardware_executed: bool = False
    version: str = WIRELESS_ESTOP_TRACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "version": self.version,
            "sample_index": int(self.sample_index),
            "timestamp_s": float(self.timestamp_s),
            "keys_bitmask": int(self.keys_bitmask),
            "estop_pressed": bool(self.estop_pressed),
            "operator_takeover_requested": bool(self.operator_takeover_requested),
            "source": self.source,
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WirelessEStopTrace":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            sample_index=int(payload.get("sample_index", 0) or 0),
            timestamp_s=_safe_float(payload.get("timestamp_s")),
            keys_bitmask=int(payload.get("keys_bitmask", 0) or 0),
            estop_pressed=bool(payload.get("estop_pressed", False)),
            operator_takeover_requested=bool(
                payload.get("operator_takeover_requested", False)
            ),
            source=str(payload.get("source", "synthetic_local_trace")),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            version=str(payload.get("version", WIRELESS_ESTOP_TRACE_VERSION)),
        )


@dataclass(frozen=True)
class ContactTrace:
    trace_id: str
    sample_index: int
    timestamp_s: float
    contact_states: dict[str, bool] = field(default_factory=dict)
    normal_forces_n: dict[str, float] = field(default_factory=dict)
    source: str = "synthetic_local_trace"
    live_stream_observed: bool = False
    hardware_executed: bool = False
    version: str = CONTACT_TRACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "version": self.version,
            "sample_index": int(self.sample_index),
            "timestamp_s": float(self.timestamp_s),
            "contact_states": {
                str(key): bool(value) for key, value in self.contact_states.items()
            },
            "normal_forces_n": dict(self.normal_forces_n),
            "source": self.source,
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContactTrace":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            sample_index=int(payload.get("sample_index", 0) or 0),
            timestamp_s=_safe_float(payload.get("timestamp_s")),
            contact_states={
                str(key): bool(value)
                for key, value in dict(payload.get("contact_states", {}) or {}).items()
            },
            normal_forces_n={
                str(key): _safe_float(value)
                for key, value in dict(payload.get("normal_forces_n", {}) or {}).items()
            },
            source=str(payload.get("source", "synthetic_local_trace")),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            version=str(payload.get("version", CONTACT_TRACE_VERSION)),
        )


@dataclass(frozen=True)
class TraceReplayReceipt:
    receipt_id: str
    trace_kind: str
    row_count: int
    export_path: str
    jsonl_export_ready: bool
    jsonl_import_verified: bool
    imported_row_count: int
    rosbag_import_ready: bool = False
    rosbag_import_executed: bool = False
    live_stream_observed: bool = False
    hardware_executed: bool = False
    authority_class: str = "trace_replay_receipt_local_only"
    version: str = TRACE_REPLAY_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "trace_kind": self.trace_kind,
            "row_count": int(self.row_count),
            "export_path": self.export_path,
            "jsonl_export_ready": bool(self.jsonl_export_ready),
            "jsonl_import_verified": bool(self.jsonl_import_verified),
            "imported_row_count": int(self.imported_row_count),
            "rosbag_import_ready": bool(self.rosbag_import_ready),
            "rosbag_import_executed": bool(self.rosbag_import_executed),
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceReplayReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            trace_kind=str(payload.get("trace_kind", "")),
            row_count=int(payload.get("row_count", 0) or 0),
            export_path=str(payload.get("export_path", "")),
            jsonl_export_ready=bool(payload.get("jsonl_export_ready", False)),
            jsonl_import_verified=bool(payload.get("jsonl_import_verified", False)),
            imported_row_count=int(payload.get("imported_row_count", 0) or 0),
            rosbag_import_ready=bool(payload.get("rosbag_import_ready", False)),
            rosbag_import_executed=bool(payload.get("rosbag_import_executed", False)),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "trace_replay_receipt_local_only")
            ),
            version=str(payload.get("version", TRACE_REPLAY_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class MockReceiverReceipt:
    receipt_id: str
    stream_key: str
    sample_count: int
    receiver_executed: bool
    stale_event_count: int
    first_timestamp_s: float
    last_timestamp_s: float
    mock_only: bool = True
    live_stream_observed: bool = False
    hardware_executed: bool = False
    authority_class: str = "mock_receiver_local_only"
    version: str = MOCK_RECEIVER_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "stream_key": self.stream_key,
            "sample_count": int(self.sample_count),
            "receiver_executed": bool(self.receiver_executed),
            "stale_event_count": int(self.stale_event_count),
            "first_timestamp_s": float(self.first_timestamp_s),
            "last_timestamp_s": float(self.last_timestamp_s),
            "mock_only": bool(self.mock_only),
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MockReceiverReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            stream_key=str(payload.get("stream_key", "")),
            sample_count=int(payload.get("sample_count", 0) or 0),
            receiver_executed=bool(payload.get("receiver_executed", False)),
            stale_event_count=int(payload.get("stale_event_count", 0) or 0),
            first_timestamp_s=_safe_float(payload.get("first_timestamp_s")),
            last_timestamp_s=_safe_float(payload.get("last_timestamp_s")),
            mock_only=bool(payload.get("mock_only", True)),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "mock_receiver_local_only")
            ),
            version=str(payload.get("version", MOCK_RECEIVER_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class StaleDataValidationReceipt:
    receipt_id: str
    stream_key: str
    max_allowed_gap_s: float
    observed_max_gap_s: float
    stale_event_count: int
    stale_sample_indices: list[int] = field(default_factory=list)
    validation_executed: bool = True
    stale_data_veto_required: bool = False
    live_stream_observed: bool = False
    hardware_executed: bool = False
    authority_class: str = "stale_data_validation_local_only"
    version: str = STALE_DATA_VALIDATION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "stream_key": self.stream_key,
            "max_allowed_gap_s": float(self.max_allowed_gap_s),
            "observed_max_gap_s": float(self.observed_max_gap_s),
            "stale_event_count": int(self.stale_event_count),
            "stale_sample_indices": [int(value) for value in self.stale_sample_indices],
            "validation_executed": bool(self.validation_executed),
            "stale_data_veto_required": bool(self.stale_data_veto_required),
            "live_stream_observed": bool(self.live_stream_observed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaleDataValidationReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            stream_key=str(payload.get("stream_key", "")),
            max_allowed_gap_s=_safe_float(payload.get("max_allowed_gap_s")),
            observed_max_gap_s=_safe_float(payload.get("observed_max_gap_s")),
            stale_event_count=int(payload.get("stale_event_count", 0) or 0),
            stale_sample_indices=[
                int(value) for value in list(payload.get("stale_sample_indices", []))
            ],
            validation_executed=bool(payload.get("validation_executed", True)),
            stale_data_veto_required=bool(
                payload.get("stale_data_veto_required", False)
            ),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "stale_data_validation_local_only")
            ),
            version=str(payload.get("version", STALE_DATA_VALIDATION_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class RosMessageDefinition:
    definition_id: str
    message_name: str
    package_name: str
    source_path: str
    fields: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    parsed: bool = False
    version: str = ROS_MESSAGE_DEFINITION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "version": self.version,
            "message_name": self.message_name,
            "package_name": self.package_name,
            "source_path": self.source_path,
            "fields": [mapping(field) for field in self.fields],
            "parse_errors": strings(self.parse_errors),
            "parsed": bool(self.parsed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RosMessageDefinition":
        return cls(
            definition_id=str(payload.get("definition_id", "")),
            message_name=str(payload.get("message_name", "")),
            package_name=str(payload.get("package_name", "")),
            source_path=str(payload.get("source_path", "")),
            fields=[mapping(field) for field in list(payload.get("fields", []))],
            parse_errors=strings(payload.get("parse_errors")),
            parsed=bool(payload.get("parsed", False)),
            version=str(payload.get("version", ROS_MESSAGE_DEFINITION_VERSION)),
        )


@dataclass(frozen=True)
class CommandShapeValidationReceipt:
    receipt_id: str
    command_family: str
    target_message_name: str
    message_definition_id: str
    frame_count: int
    required_fields_present: bool
    array_capacity_sufficient: bool
    no_publish_serialization_ready: bool
    validated_frame_ids: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    dry_run_payload_shape: dict[str, Any] = field(default_factory=dict)
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    hardware_dispatch_enabled: bool = False
    authority_class: str = "command_shape_validation_no_publish"
    version: str = COMMAND_SHAPE_VALIDATION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "command_family": self.command_family,
            "target_message_name": self.target_message_name,
            "message_definition_id": self.message_definition_id,
            "frame_count": int(self.frame_count),
            "required_fields_present": bool(self.required_fields_present),
            "array_capacity_sufficient": bool(self.array_capacity_sufficient),
            "no_publish_serialization_ready": bool(
                self.no_publish_serialization_ready
            ),
            "validated_frame_ids": strings(self.validated_frame_ids),
            "missing_fields": strings(self.missing_fields),
            "dry_run_payload_shape": mapping(self.dry_run_payload_shape),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "hardware_dispatch_enabled": bool(self.hardware_dispatch_enabled),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandShapeValidationReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            command_family=str(payload.get("command_family", "")),
            target_message_name=str(payload.get("target_message_name", "")),
            message_definition_id=str(payload.get("message_definition_id", "")),
            frame_count=int(payload.get("frame_count", 0) or 0),
            required_fields_present=bool(
                payload.get("required_fields_present", False)
            ),
            array_capacity_sufficient=bool(
                payload.get("array_capacity_sufficient", False)
            ),
            no_publish_serialization_ready=bool(
                payload.get("no_publish_serialization_ready", False)
            ),
            validated_frame_ids=strings(payload.get("validated_frame_ids")),
            missing_fields=strings(payload.get("missing_fields")),
            dry_run_payload_shape=mapping(payload.get("dry_run_payload_shape")),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            hardware_dispatch_enabled=bool(
                payload.get("hardware_dispatch_enabled", False)
            ),
            authority_class=str(
                payload.get("authority_class", "command_shape_validation_no_publish")
            ),
            version=str(
                payload.get("version", COMMAND_SHAPE_VALIDATION_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class MockTimingRunReceipt:
    receipt_id: str
    target_hz: float
    iterations: int
    producer_event_count: int
    consumer_event_count: int
    mean_latency_s: float
    max_latency_s: float
    mean_step_s: float
    max_jitter_s: float
    jitter_histogram: dict[str, int] = field(default_factory=dict)
    stale_event_count: int = 0
    local_loop_executed: bool = True
    dds_runtime_observed: bool = False
    hardware_executed: bool = False
    authority_class: str = "mock_timing_run_local_only"
    version: str = MOCK_TIMING_RUN_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "target_hz": float(self.target_hz),
            "iterations": int(self.iterations),
            "producer_event_count": int(self.producer_event_count),
            "consumer_event_count": int(self.consumer_event_count),
            "mean_latency_s": float(self.mean_latency_s),
            "max_latency_s": float(self.max_latency_s),
            "mean_step_s": float(self.mean_step_s),
            "max_jitter_s": float(self.max_jitter_s),
            "jitter_histogram": {
                str(key): int(value) for key, value in self.jitter_histogram.items()
            },
            "stale_event_count": int(self.stale_event_count),
            "local_loop_executed": bool(self.local_loop_executed),
            "dds_runtime_observed": bool(self.dds_runtime_observed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MockTimingRunReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            target_hz=_safe_float(payload.get("target_hz")),
            iterations=int(payload.get("iterations", 0) or 0),
            producer_event_count=int(payload.get("producer_event_count", 0) or 0),
            consumer_event_count=int(payload.get("consumer_event_count", 0) or 0),
            mean_latency_s=_safe_float(payload.get("mean_latency_s")),
            max_latency_s=_safe_float(payload.get("max_latency_s")),
            mean_step_s=_safe_float(payload.get("mean_step_s")),
            max_jitter_s=_safe_float(payload.get("max_jitter_s")),
            jitter_histogram={
                str(key): int(value)
                for key, value in dict(payload.get("jitter_histogram", {}) or {}).items()
            },
            stale_event_count=int(payload.get("stale_event_count", 0) or 0),
            local_loop_executed=bool(payload.get("local_loop_executed", True)),
            dds_runtime_observed=bool(payload.get("dds_runtime_observed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "mock_timing_run_local_only")
            ),
            version=str(payload.get("version", MOCK_TIMING_RUN_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class WatchdogDemotionReceipt:
    receipt_id: str
    trigger_source: str
    stale_event_count: int
    estop_seen: bool
    demotion_requested: bool
    demotion_posture: str
    command_dispatch_allowed: bool = False
    live_policy_control: bool = False
    hardware_executed: bool = False
    authority_class: str = "watchdog_demotion_receipt_local_only"
    version: str = WATCHDOG_DEMOTION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "trigger_source": self.trigger_source,
            "stale_event_count": int(self.stale_event_count),
            "estop_seen": bool(self.estop_seen),
            "demotion_requested": bool(self.demotion_requested),
            "demotion_posture": self.demotion_posture,
            "command_dispatch_allowed": bool(self.command_dispatch_allowed),
            "live_policy_control": bool(self.live_policy_control),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WatchdogDemotionReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            trigger_source=str(payload.get("trigger_source", "")),
            stale_event_count=int(payload.get("stale_event_count", 0) or 0),
            estop_seen=bool(payload.get("estop_seen", False)),
            demotion_requested=bool(payload.get("demotion_requested", False)),
            demotion_posture=str(
                payload.get("demotion_posture", "stable_base_mobile_manipulator")
            ),
            command_dispatch_allowed=bool(
                payload.get("command_dispatch_allowed", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "watchdog_demotion_receipt_local_only")
            ),
            version=str(payload.get("version", WATCHDOG_DEMOTION_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class SafetyStateTransition:
    transition_id: str
    event_key: str
    from_state: str
    to_state: str
    timestamp_s: float
    reason: str
    command_dispatch_allowed: bool = False
    hardware_executed: bool = False
    authority_class: str = "safety_state_transition_local_only"
    version: str = SAFETY_STATE_TRANSITION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "version": self.version,
            "event_key": self.event_key,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp_s": float(self.timestamp_s),
            "reason": self.reason,
            "command_dispatch_allowed": bool(self.command_dispatch_allowed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SafetyStateTransition":
        return cls(
            transition_id=str(payload.get("transition_id", "")),
            event_key=str(payload.get("event_key", "")),
            from_state=str(payload.get("from_state", "")),
            to_state=str(payload.get("to_state", "")),
            timestamp_s=_safe_float(payload.get("timestamp_s")),
            reason=str(payload.get("reason", "")),
            command_dispatch_allowed=bool(
                payload.get("command_dispatch_allowed", False)
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "safety_state_transition_local_only")
            ),
            version=str(payload.get("version", SAFETY_STATE_TRANSITION_VERSION)),
        )


@dataclass(frozen=True)
class SyntheticSafetyDrillReceipt:
    receipt_id: str
    drill_key: str
    transition_ids: list[str] = field(default_factory=list)
    drill_executed_locally: bool = True
    estop_latched: bool = False
    stale_data_vetoed: bool = False
    joint_clamp_observed: bool = False
    stable_base_demote_requested: bool = False
    recovery_state_reached: bool = False
    teleop_runtime_executed: bool = False
    hardware_executed: bool = False
    promotion_eligible: bool = False
    authority_class: str = "synthetic_safety_drill_local_only"
    version: str = SYNTHETIC_SAFETY_DRILL_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "drill_key": self.drill_key,
            "transition_ids": strings(self.transition_ids),
            "drill_executed_locally": bool(self.drill_executed_locally),
            "estop_latched": bool(self.estop_latched),
            "stale_data_vetoed": bool(self.stale_data_vetoed),
            "joint_clamp_observed": bool(self.joint_clamp_observed),
            "stable_base_demote_requested": bool(self.stable_base_demote_requested),
            "recovery_state_reached": bool(self.recovery_state_reached),
            "teleop_runtime_executed": bool(self.teleop_runtime_executed),
            "hardware_executed": bool(self.hardware_executed),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticSafetyDrillReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            drill_key=str(payload.get("drill_key", "")),
            transition_ids=strings(payload.get("transition_ids")),
            drill_executed_locally=bool(
                payload.get("drill_executed_locally", True)
            ),
            estop_latched=bool(payload.get("estop_latched", False)),
            stale_data_vetoed=bool(payload.get("stale_data_vetoed", False)),
            joint_clamp_observed=bool(payload.get("joint_clamp_observed", False)),
            stable_base_demote_requested=bool(
                payload.get("stable_base_demote_requested", False)
            ),
            recovery_state_reached=bool(payload.get("recovery_state_reached", False)),
            teleop_runtime_executed=bool(
                payload.get("teleop_runtime_executed", False)
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get("authority_class", "synthetic_safety_drill_local_only")
            ),
            version=str(payload.get("version", SYNTHETIC_SAFETY_DRILL_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class RuntimePreflightReceipt:
    receipt_id: str
    target_key: str
    preflight_kind: str
    root_path: str
    status: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    launch_request: str = ""
    build_command: str = ""
    import_available: bool = False
    build_executed: bool = False
    launch_executed: bool = False
    runtime_executed: bool = False
    hardware_executed: bool = False
    authority_class: str = "runtime_preflight_receipt_no_launch"
    version: str = RUNTIME_PREFLIGHT_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "target_key": self.target_key,
            "preflight_kind": self.preflight_kind,
            "root_path": self.root_path,
            "status": self.status,
            "checks_passed": strings(self.checks_passed),
            "checks_failed": strings(self.checks_failed),
            "launch_request": self.launch_request,
            "build_command": self.build_command,
            "import_available": bool(self.import_available),
            "build_executed": bool(self.build_executed),
            "launch_executed": bool(self.launch_executed),
            "runtime_executed": bool(self.runtime_executed),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePreflightReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            target_key=str(payload.get("target_key", "")),
            preflight_kind=str(payload.get("preflight_kind", "")),
            root_path=str(payload.get("root_path", "")),
            status=str(payload.get("status", "")),
            checks_passed=strings(payload.get("checks_passed")),
            checks_failed=strings(payload.get("checks_failed")),
            launch_request=str(payload.get("launch_request", "")),
            build_command=str(payload.get("build_command", "")),
            import_available=bool(payload.get("import_available", False)),
            build_executed=bool(payload.get("build_executed", False)),
            launch_executed=bool(payload.get("launch_executed", False)),
            runtime_executed=bool(payload.get("runtime_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get("authority_class", "runtime_preflight_receipt_no_launch")
            ),
            version=str(payload.get("version", RUNTIME_PREFLIGHT_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class Phase4UnitreeLocalHarnessReport:
    report_id: str
    chassis_id: str
    status: str
    low_state_trace_count: int
    imu_trace_count: int
    wireless_estop_trace_count: int
    contact_trace_count: int
    trace_replay_receipt_count: int
    mock_receiver_receipt_count: int
    stale_validation_receipt_count: int
    ros_message_definition_count: int
    command_shape_validation_receipt_count: int
    mock_timing_run_receipt_count: int
    watchdog_demotion_receipt_count: int
    safety_transition_count: int
    synthetic_safety_drill_receipt_count: int
    runtime_preflight_receipt_count: int
    trace_stream_harness_complete: bool
    command_shape_harness_complete: bool
    mock_timing_watchdog_harness_complete: bool
    safety_recovery_harness_complete: bool
    runtime_preflight_harness_complete: bool
    local_harnesses_complete: bool
    live_stream_observed: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    mujoco_launch_executed: bool = False
    ros2_launch_executed: bool = False
    hardware_executed: bool = False
    training_executed: bool = False
    weights_written: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    remaining_evidence_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE4_UNITREE_LOCAL_HARNESS_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "chassis_id": self.chassis_id,
            "status": self.status,
            "low_state_trace_count": int(self.low_state_trace_count),
            "imu_trace_count": int(self.imu_trace_count),
            "wireless_estop_trace_count": int(self.wireless_estop_trace_count),
            "contact_trace_count": int(self.contact_trace_count),
            "trace_replay_receipt_count": int(self.trace_replay_receipt_count),
            "mock_receiver_receipt_count": int(self.mock_receiver_receipt_count),
            "stale_validation_receipt_count": int(self.stale_validation_receipt_count),
            "ros_message_definition_count": int(self.ros_message_definition_count),
            "command_shape_validation_receipt_count": int(
                self.command_shape_validation_receipt_count
            ),
            "mock_timing_run_receipt_count": int(
                self.mock_timing_run_receipt_count
            ),
            "watchdog_demotion_receipt_count": int(
                self.watchdog_demotion_receipt_count
            ),
            "safety_transition_count": int(self.safety_transition_count),
            "synthetic_safety_drill_receipt_count": int(
                self.synthetic_safety_drill_receipt_count
            ),
            "runtime_preflight_receipt_count": int(
                self.runtime_preflight_receipt_count
            ),
            "trace_stream_harness_complete": bool(
                self.trace_stream_harness_complete
            ),
            "command_shape_harness_complete": bool(
                self.command_shape_harness_complete
            ),
            "mock_timing_watchdog_harness_complete": bool(
                self.mock_timing_watchdog_harness_complete
            ),
            "safety_recovery_harness_complete": bool(
                self.safety_recovery_harness_complete
            ),
            "runtime_preflight_harness_complete": bool(
                self.runtime_preflight_harness_complete
            ),
            "local_harnesses_complete": bool(self.local_harnesses_complete),
            "live_stream_observed": bool(self.live_stream_observed),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "mujoco_launch_executed": bool(self.mujoco_launch_executed),
            "ros2_launch_executed": bool(self.ros2_launch_executed),
            "hardware_executed": bool(self.hardware_executed),
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
    ) -> "Phase4UnitreeLocalHarnessReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            chassis_id=str(payload.get("chassis_id", "")),
            status=str(payload.get("status", "blocked")),
            low_state_trace_count=int(payload.get("low_state_trace_count", 0) or 0),
            imu_trace_count=int(payload.get("imu_trace_count", 0) or 0),
            wireless_estop_trace_count=int(
                payload.get("wireless_estop_trace_count", 0) or 0
            ),
            contact_trace_count=int(payload.get("contact_trace_count", 0) or 0),
            trace_replay_receipt_count=int(
                payload.get("trace_replay_receipt_count", 0) or 0
            ),
            mock_receiver_receipt_count=int(
                payload.get("mock_receiver_receipt_count", 0) or 0
            ),
            stale_validation_receipt_count=int(
                payload.get("stale_validation_receipt_count", 0) or 0
            ),
            ros_message_definition_count=int(
                payload.get("ros_message_definition_count", 0) or 0
            ),
            command_shape_validation_receipt_count=int(
                payload.get("command_shape_validation_receipt_count", 0) or 0
            ),
            mock_timing_run_receipt_count=int(
                payload.get("mock_timing_run_receipt_count", 0) or 0
            ),
            watchdog_demotion_receipt_count=int(
                payload.get("watchdog_demotion_receipt_count", 0) or 0
            ),
            safety_transition_count=int(payload.get("safety_transition_count", 0) or 0),
            synthetic_safety_drill_receipt_count=int(
                payload.get("synthetic_safety_drill_receipt_count", 0) or 0
            ),
            runtime_preflight_receipt_count=int(
                payload.get("runtime_preflight_receipt_count", 0) or 0
            ),
            trace_stream_harness_complete=bool(
                payload.get("trace_stream_harness_complete", False)
            ),
            command_shape_harness_complete=bool(
                payload.get("command_shape_harness_complete", False)
            ),
            mock_timing_watchdog_harness_complete=bool(
                payload.get("mock_timing_watchdog_harness_complete", False)
            ),
            safety_recovery_harness_complete=bool(
                payload.get("safety_recovery_harness_complete", False)
            ),
            runtime_preflight_harness_complete=bool(
                payload.get("runtime_preflight_harness_complete", False)
            ),
            local_harnesses_complete=bool(
                payload.get("local_harnesses_complete", False)
            ),
            live_stream_observed=bool(payload.get("live_stream_observed", False)),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            g1pilot_runtime_invoked=bool(payload.get("g1pilot_runtime_invoked", False)),
            mujoco_launch_executed=bool(payload.get("mujoco_launch_executed", False)),
            ros2_launch_executed=bool(payload.get("ros2_launch_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(payload.get("denied_gates", {}) or {}).items()
                },
            },
            remaining_evidence_blockers=strings(
                payload.get("remaining_evidence_blockers")
            ),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE4_UNITREE_LOCAL_HARNESS_REPORT_VERSION)
            ),
        )


def build_synthetic_trace_streams(
    *,
    chassis: HumanoidChassisProfile,
    sample_count: int = 12,
    dt_s: float = 0.02,
) -> tuple[
    list[LowStateTrace],
    list[ImuTrace],
    list[WirelessEStopTrace],
    list[ContactTrace],
]:
    sample_count = max(4, int(sample_count))
    joint_names = chassis.joint_names
    trace_payload = {
        "chassis_id": chassis.chassis_id,
        "sample_count": sample_count,
        "dt_s": dt_s,
    }
    trace_id = stable_id("unitree_synthetic_trace", trace_payload)
    low_state: list[LowStateTrace] = []
    imu: list[ImuTrace] = []
    wireless: list[WirelessEStopTrace] = []
    contacts: list[ContactTrace] = []
    timestamp = 0.0
    stale_gap_index = sample_count // 2
    for index in range(sample_count):
        if index == stale_gap_index:
            timestamp += dt_s * 8.0
        else:
            timestamp += dt_s
        joint_positions = {
            joint: round(0.001 * (index + offset), 6)
            for offset, joint in enumerate(joint_names)
        }
        joint_velocities = {joint: 0.0 for joint in joint_names}
        low_state.append(
            LowStateTrace(
                trace_id=trace_id,
                sample_index=index,
                timestamp_s=round(timestamp, 6),
                tick=index,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                motor_count=len(joint_names),
            )
        )
        imu.append(
            ImuTrace(
                trace_id=trace_id,
                sample_index=index,
                timestamp_s=round(timestamp, 6),
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                gyro_rad_s=[0.0, 0.0, 0.001 * index],
                accel_m_s2=[0.0, 0.0, 9.81],
            )
        )
        estop = index == sample_count - 2
        wireless.append(
            WirelessEStopTrace(
                trace_id=trace_id,
                sample_index=index,
                timestamp_s=round(timestamp, 6),
                keys_bitmask=1 if estop else 0,
                estop_pressed=estop,
                operator_takeover_requested=index == sample_count - 3,
            )
        )
        left_contact = index % 3 != 1
        right_contact = index % 3 != 2
        contacts.append(
            ContactTrace(
                trace_id=trace_id,
                sample_index=index,
                timestamp_s=round(timestamp, 6),
                contact_states={
                    "left_foot": left_contact,
                    "right_foot": right_contact,
                },
                normal_forces_n={
                    "left_foot": 180.0 if left_contact else 0.0,
                    "right_foot": 175.0 if right_contact else 0.0,
                },
            )
        )
    return low_state, imu, wireless, contacts


def _timestamps(rows: Iterable[Any]) -> list[float]:
    return [float(getattr(row, "timestamp_s", 0.0)) for row in rows]


def _validate_stale(
    *,
    stream_key: str,
    rows: list[Any],
    max_allowed_gap_s: float,
) -> StaleDataValidationReceipt:
    times = _timestamps(rows)
    gaps = [max(0.0, times[index] - times[index - 1]) for index in range(1, len(times))]
    stale_indices = [
        index + 1 for index, gap in enumerate(gaps) if gap > max_allowed_gap_s
    ]
    observed_max = max(gaps) if gaps else 0.0
    payload = {
        "stream_key": stream_key,
        "observed_max_gap_s": observed_max,
        "stale_indices": stale_indices,
    }
    return StaleDataValidationReceipt(
        receipt_id=stable_id("unitree_stale_validation", payload),
        stream_key=stream_key,
        max_allowed_gap_s=max_allowed_gap_s,
        observed_max_gap_s=observed_max,
        stale_event_count=len(stale_indices),
        stale_sample_indices=stale_indices,
        stale_data_veto_required=bool(stale_indices),
    )


def build_trace_harness_receipts(
    *,
    low_state: list[LowStateTrace],
    imu: list[ImuTrace],
    wireless: list[WirelessEStopTrace],
    contacts: list[ContactTrace],
    trace_paths: Mapping[str, str | Path],
    max_allowed_gap_s: float = 0.1,
) -> tuple[
    list[TraceReplayReceipt],
    list[MockReceiverReceipt],
    list[StaleDataValidationReceipt],
]:
    stream_rows: dict[str, list[Any]] = {
        "low_state": list(low_state),
        "imu": list(imu),
        "wireless_estop": list(wireless),
        "contact": list(contacts),
    }
    stale_receipts = [
        _validate_stale(
            stream_key=stream_key,
            rows=rows,
            max_allowed_gap_s=max_allowed_gap_s,
        )
        for stream_key, rows in stream_rows.items()
    ]
    stale_by_key = {receipt.stream_key: receipt for receipt in stale_receipts}
    replay_receipts: list[TraceReplayReceipt] = []
    receiver_receipts: list[MockReceiverReceipt] = []
    for stream_key, rows in stream_rows.items():
        path = Path(trace_paths[stream_key])
        imported_rows = _load_rows(path) if path.exists() else []
        payload = {
            "stream_key": stream_key,
            "path": str(path),
            "row_count": len(rows),
            "imported_row_count": len(imported_rows),
        }
        replay_receipts.append(
            TraceReplayReceipt(
                receipt_id=stable_id("unitree_trace_replay", payload),
                trace_kind=stream_key,
                row_count=len(rows),
                export_path=str(path),
                jsonl_export_ready=path.exists(),
                jsonl_import_verified=len(imported_rows) == len(rows),
                imported_row_count=len(imported_rows),
                rosbag_import_ready=True,
                rosbag_import_executed=False,
            )
        )
        times = _timestamps(rows)
        stale = stale_by_key[stream_key]
        receiver_payload = {
            "stream_key": stream_key,
            "sample_count": len(rows),
            "stale_event_count": stale.stale_event_count,
        }
        receiver_receipts.append(
            MockReceiverReceipt(
                receipt_id=stable_id("unitree_mock_receiver", receiver_payload),
                stream_key=stream_key,
                sample_count=len(rows),
                receiver_executed=True,
                stale_event_count=stale.stale_event_count,
                first_timestamp_s=times[0] if times else 0.0,
                last_timestamp_s=times[-1] if times else 0.0,
            )
        )
    return replay_receipts, receiver_receipts, stale_receipts


def _parse_msg_field(raw: str) -> Optional[dict[str, Any]]:
    line = raw.split("#", 1)[0].strip()
    if not line or "=" in line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    field_type = parts[0]
    field_name = parts[1]
    is_array = "[" in field_type and "]" in field_type
    array_len: Optional[int] = None
    base_type = field_type
    if is_array:
        base_type = field_type.split("[", 1)[0]
        size = field_type.split("[", 1)[1].split("]", 1)[0]
        if size:
            try:
                array_len = int(size)
            except ValueError:
                array_len = None
    return {
        "field_type": field_type,
        "base_type": base_type,
        "field_name": field_name,
        "is_array": is_array,
        "array_len": array_len,
    }


def parse_ros_message_definition(
    *,
    path: str | Path,
    package_name: str,
    message_name: str,
) -> RosMessageDefinition:
    target = Path(path)
    fields: list[dict[str, Any]] = []
    errors: list[str] = []
    if not target.exists():
        errors.append("message_definition_missing")
    else:
        try:
            for line in target.read_text(encoding="utf-8").splitlines():
                field = _parse_msg_field(line)
                if field is not None:
                    fields.append(field)
        except OSError as exc:
            errors.append(str(exc))
    payload = {
        "package_name": package_name,
        "message_name": message_name,
        "path": str(target),
        "field_count": len(fields),
        "errors": errors,
    }
    return RosMessageDefinition(
        definition_id=stable_id("unitree_ros_msg", payload),
        message_name=message_name,
        package_name=package_name,
        source_path=str(target),
        fields=fields,
        parse_errors=errors,
        parsed=bool(fields) and not errors,
    )


def build_ros_message_definitions(
    local_roots: Mapping[str, str | Path] | None = None,
) -> list[RosMessageDefinition]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    ros_root = Path(roots["unitree_ros2"]) / "cyclonedds_ws/src/unitree"
    specs = [
        ("unitree_hg", "LowCmd", ros_root / "unitree_hg/msg/LowCmd.msg"),
        ("unitree_hg", "MotorCmd", ros_root / "unitree_hg/msg/MotorCmd.msg"),
        ("unitree_hg", "LowState", ros_root / "unitree_hg/msg/LowState.msg"),
        ("unitree_hg", "IMUState", ros_root / "unitree_hg/msg/IMUState.msg"),
        ("unitree_api", "Request", ros_root / "unitree_api/msg/Request.msg"),
        (
            "unitree_api",
            "RequestHeader",
            ros_root / "unitree_api/msg/RequestHeader.msg",
        ),
        (
            "unitree_go",
            "WirelessController",
            ros_root / "unitree_go/msg/WirelessController.msg",
        ),
    ]
    return [
        parse_ros_message_definition(
            path=path,
            package_name=package,
            message_name=message,
        )
        for package, message, path in specs
    ]


def _fields_by_name(definition: RosMessageDefinition) -> dict[str, dict[str, Any]]:
    return {str(field["field_name"]): mapping(field) for field in definition.fields}


def _message_definition(
    definitions: list[RosMessageDefinition],
    message_name: str,
    package_name: str = "",
) -> RosMessageDefinition:
    for definition in definitions:
        if definition.message_name == message_name and (
            not package_name or definition.package_name == package_name
        ):
            return definition
    return RosMessageDefinition(
        definition_id="missing_definition",
        message_name=message_name,
        package_name=package_name,
        source_path="",
        parse_errors=["definition_not_found"],
        parsed=False,
    )


def build_command_shape_validation_receipts(
    *,
    command_frames: list[LowLevelCommandFrame],
    definitions: list[RosMessageDefinition],
) -> list[CommandShapeValidationReceipt]:
    lowcmd = _message_definition(definitions, "LowCmd", "unitree_hg")
    request = _message_definition(definitions, "Request", "unitree_api")
    lowcmd_fields = _fields_by_name(lowcmd)
    request_fields = _fields_by_name(request)
    lowcmd_frames = [
        frame
        for frame in command_frames
        if frame.unitree_command_family == "low_level_joint_pd"
    ]
    sport_frames = [
        frame
        for frame in command_frames
        if frame.unitree_command_family == "sport_request_degraded_mode"
    ]
    lowcmd_required = ["mode_pr", "mode_machine", "motor_cmd", "reserve", "crc"]
    lowcmd_missing = [field for field in lowcmd_required if field not in lowcmd_fields]
    motor_capacity = int(lowcmd_fields.get("motor_cmd", {}).get("array_len") or 0)
    max_channels = max((len(frame.channel_names) for frame in lowcmd_frames), default=0)
    lowcmd_capacity_ok = bool(motor_capacity) and max_channels <= motor_capacity
    lowcmd_payload = {
        "mode_pr": "dry_run_required",
        "mode_machine": "dry_run_required",
        "motor_cmd_capacity": motor_capacity,
        "max_frame_channels": max_channels,
        "crc": "not_computed_no_publish",
    }
    request_required = ["header", "parameter", "binary"]
    request_missing = [field for field in request_required if field not in request_fields]
    request_payload = {
        "header": "dry_run_header_required",
        "parameter": "json_command_payload_no_publish",
        "binary": "empty_binary_no_publish",
    }
    specs = [
        (
            "low_level_joint_pd",
            "unitree_hg/LowCmd",
            lowcmd,
            lowcmd_frames,
            lowcmd_missing,
            lowcmd_capacity_ok,
            lowcmd_payload,
        ),
        (
            "sport_request_degraded_mode",
            "unitree_api/Request",
            request,
            sport_frames,
            request_missing,
            True,
            request_payload,
        ),
    ]
    receipts: list[CommandShapeValidationReceipt] = []
    for (
        command_family,
        target_message,
        definition,
        frames,
        missing,
        capacity_ok,
        payload_shape,
    ) in specs:
        receipt_payload = {
            "command_family": command_family,
            "target_message": target_message,
            "frame_ids": [frame.frame_id for frame in frames],
            "missing": missing,
            "capacity_ok": capacity_ok,
        }
        receipts.append(
            CommandShapeValidationReceipt(
                receipt_id=stable_id("unitree_command_shape", receipt_payload),
                command_family=command_family,
                target_message_name=target_message,
                message_definition_id=definition.definition_id,
                frame_count=len(frames),
                required_fields_present=definition.parsed and not missing,
                array_capacity_sufficient=capacity_ok,
                no_publish_serialization_ready=(
                    bool(frames) and definition.parsed and not missing and capacity_ok
                ),
                validated_frame_ids=[frame.frame_id for frame in frames],
                missing_fields=missing,
                dry_run_payload_shape=payload_shape,
            )
        )
    return receipts


def _histogram(values: list[float]) -> dict[str, int]:
    buckets = {"lt_1ms": 0, "lt_5ms": 0, "lt_20ms": 0, "gte_20ms": 0}
    for value in values:
        if value < 0.001:
            buckets["lt_1ms"] += 1
        elif value < 0.005:
            buckets["lt_5ms"] += 1
        elif value < 0.02:
            buckets["lt_20ms"] += 1
        else:
            buckets["gte_20ms"] += 1
    return buckets


def run_mock_timing_harness(
    *,
    target_hz: float = 50.0,
    iterations: int = 32,
    injected_stale_events: int = 1,
) -> tuple[list[MockTimingRunReceipt], list[WatchdogDemotionReceipt]]:
    iterations = max(4, int(iterations))
    period = 1.0 / max(target_hz, 1.0)
    producer_times: list[float] = []
    consumer_times: list[float] = []
    for index in range(iterations):
        now = time.perf_counter()
        producer_times.append(now)
        if index == iterations // 2 and injected_stale_events:
            time.sleep(min(period * 1.5, 0.02))
        else:
            time.sleep(0)
        consumer_times.append(time.perf_counter())
    latencies = [
        max(0.0, consumer - producer)
        for producer, consumer in zip(producer_times, consumer_times)
    ]
    steps = [
        max(0.0, producer_times[index] - producer_times[index - 1])
        for index in range(1, len(producer_times))
    ]
    mean_step = sum(steps) / len(steps) if steps else 0.0
    jitters = [abs(step - period) for step in steps]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    max_jitter = max(jitters) if jitters else 0.0
    stale_count = max(0, int(injected_stale_events))
    timing_payload = {
        "target_hz": target_hz,
        "iterations": iterations,
        "stale_count": stale_count,
    }
    timing = MockTimingRunReceipt(
        receipt_id=stable_id("unitree_mock_timing", timing_payload),
        target_hz=target_hz,
        iterations=iterations,
        producer_event_count=len(producer_times),
        consumer_event_count=len(consumer_times),
        mean_latency_s=mean_latency,
        max_latency_s=max_latency,
        mean_step_s=mean_step,
        max_jitter_s=max_jitter,
        jitter_histogram=_histogram(jitters),
        stale_event_count=stale_count,
    )
    watchdog_payload = {
        "trigger_source": "mock_timing_loop",
        "stale_count": stale_count,
    }
    watchdog = WatchdogDemotionReceipt(
        receipt_id=stable_id("unitree_watchdog", watchdog_payload),
        trigger_source="mock_timing_loop",
        stale_event_count=stale_count,
        estop_seen=False,
        demotion_requested=stale_count > 0,
        demotion_posture="stable_base_mobile_manipulator",
    )
    return [timing], [watchdog]


def run_safety_recovery_harness(
    *,
    stale_receipts: list[StaleDataValidationReceipt],
    wireless_traces: list[WirelessEStopTrace],
    command_frames: list[LowLevelCommandFrame],
    joint_limits: list[JointLimitEnvelope],
) -> tuple[list[SafetyStateTransition], list[SyntheticSafetyDrillReceipt]]:
    transitions: list[SafetyStateTransition] = []
    current = "nominal_dry_run"

    def transition(event_key: str, to_state: str, reason: str, timestamp_s: float) -> None:
        nonlocal current
        payload = {
            "event_key": event_key,
            "from_state": current,
            "to_state": to_state,
            "reason": reason,
            "timestamp_s": timestamp_s,
        }
        transitions.append(
            SafetyStateTransition(
                transition_id=stable_id("unitree_safety_transition", payload),
                event_key=event_key,
                from_state=current,
                to_state=to_state,
                timestamp_s=timestamp_s,
                reason=reason,
            )
        )
        current = to_state

    stale_seen = any(receipt.stale_event_count for receipt in stale_receipts)
    estop_seen = any(trace.estop_pressed for trace in wireless_traces)
    clamp_seen = any(frame.clamp_applied for frame in command_frames)
    if stale_seen:
        transition("stale_data", "stale_data_veto", "mock low-state gap exceeded", 0.1)
    if clamp_seen and joint_limits:
        transition(
            "joint_clamp",
            "joint_limit_veto",
            "dry-run command frame crossed local planning envelope",
            0.2,
        )
    if estop_seen:
        transition("estop", "estop_latched", "synthetic wireless e-stop pressed", 0.3)
    if stale_seen or estop_seen or clamp_seen:
        transition(
            "demote",
            "stable_base_demote_requested",
            "dispatch denied and stable-base fallback requested",
            0.4,
        )
        transition(
            "operator_handoff",
            "recovery_ready_operator_required",
            "operator review required before any resume",
            0.5,
        )
    drill_payload = {
        "transition_ids": [transition.transition_id for transition in transitions],
        "stale_seen": stale_seen,
        "estop_seen": estop_seen,
        "clamp_seen": clamp_seen,
    }
    drill = SyntheticSafetyDrillReceipt(
        receipt_id=stable_id("unitree_safety_drill", drill_payload),
        drill_key="stale_clamp_estop_demote_recovery",
        transition_ids=[transition.transition_id for transition in transitions],
        estop_latched=estop_seen,
        stale_data_vetoed=stale_seen,
        joint_clamp_observed=clamp_seen,
        stable_base_demote_requested=bool(transitions),
        recovery_state_reached=bool(
            transitions and transitions[-1].to_state == "recovery_ready_operator_required"
        ),
    )
    return transitions, [drill]


def _check_paths(root: Path, markers: list[str]) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    for marker in markers:
        if (root / marker).exists():
            passed.append(marker)
        else:
            failed.append(marker)
    return passed, failed


def _receipt_status(failed: list[str], soft_failures_allowed: bool = False) -> str:
    if not failed:
        return "ok"
    if soft_failures_allowed:
        return "partial_preflight_blocked_by_host_dependency"
    return "blocked"


def build_runtime_preflight_receipts(
    local_roots: Mapping[str, str | Path] | None = None,
) -> list[RuntimePreflightReceipt]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    receipts: list[RuntimePreflightReceipt] = []

    def add_receipt(
        *,
        target_key: str,
        preflight_kind: str,
        checks_passed: list[str],
        checks_failed: list[str],
        launch_request: str = "",
        build_command: str = "",
        import_available: bool = False,
        soft_failures_allowed: bool = False,
    ) -> None:
        payload = {
            "target_key": target_key,
            "preflight_kind": preflight_kind,
            "passed": checks_passed,
            "failed": checks_failed,
            "launch_request": launch_request,
            "build_command": build_command,
        }
        receipts.append(
            RuntimePreflightReceipt(
                receipt_id=stable_id("unitree_preflight", payload),
                target_key=target_key,
                preflight_kind=preflight_kind,
                root_path=str(roots.get(target_key, "")),
                status=_receipt_status(checks_failed, soft_failures_allowed),
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                launch_request=launch_request,
                build_command=build_command,
                import_available=import_available,
            )
        )

    ros_root = Path(roots["unitree_ros2"])
    passed, failed = _check_paths(
        ros_root,
        [
            "setup.sh",
            "cyclonedds_ws/src/unitree/unitree_hg/msg/LowCmd.msg",
            "cyclonedds_ws/src/unitree/unitree_hg/msg/LowState.msg",
            "cyclonedds_ws/src/unitree/unitree_api/msg/Request.msg",
        ],
    )
    add_receipt(
        target_key="unitree_ros2",
        preflight_kind="source_layout_and_message_files",
        checks_passed=passed,
        checks_failed=failed,
    )
    tool_passed = [tool for tool in ("cmake", "python3") if shutil.which(tool)]
    tool_failed = [tool for tool in ("colcon", "ros2") if not shutil.which(tool)]
    add_receipt(
        target_key="unitree_ros2",
        preflight_kind="build_tool_availability",
        checks_passed=tool_passed,
        checks_failed=tool_failed,
        build_command="colcon build --packages-select unitree_hg unitree_api",
        soft_failures_allowed=True,
    )
    add_receipt(
        target_key="unitree_ros2",
        preflight_kind="launch_request_materialized",
        checks_passed=["launch_request_written"],
        checks_failed=[],
        launch_request=(
            "source /Users/amarmurray/code/unitree_ros2/setup.sh && "
            "ros2 topic echo /lowstate"
        ),
    )

    mujoco_root = Path(roots["unitree_mujoco"])
    passed, failed = _check_paths(
        mujoco_root,
        [
            "simulate_python/unitree_mujoco.py",
            "simulate_python/unitree_sdk2py_bridge.py",
            "unitree_robots/g1/scene_29dof.xml",
            "unitree_robots/g1/g1_29dof.xml",
        ],
    )
    xml_failed = list(failed)
    for xml_marker in ("unitree_robots/g1/scene_29dof.xml", "unitree_robots/g1/g1_29dof.xml"):
        xml_path = mujoco_root / xml_marker
        if xml_path.exists():
            try:
                ET.parse(xml_path)
            except Exception:
                xml_failed.append(f"{xml_marker}:xml_parse_failed")
    add_receipt(
        target_key="unitree_mujoco",
        preflight_kind="source_layout_and_xml_parse",
        checks_passed=passed,
        checks_failed=xml_failed,
    )
    mujoco_import = importlib.util.find_spec("mujoco") is not None
    add_receipt(
        target_key="unitree_mujoco",
        preflight_kind="python_import_availability",
        checks_passed=["mujoco_python_module"] if mujoco_import else [],
        checks_failed=[] if mujoco_import else ["mujoco_python_module_missing"],
        launch_request=(
            "python /Users/amarmurray/code/unitree_mujoco/simulate_python/"
            "unitree_mujoco.py"
        ),
        import_available=mujoco_import,
        soft_failures_allowed=True,
    )

    g1pilot_root = Path(roots["g1pilot"])
    passed, failed = _check_paths(
        g1pilot_root,
        [
            "package.xml",
            "launch/bringup_launcher.launch.py",
            "launch/teleoperation_launcher.launch.py",
            "g1pilot",
            "description_files/urdf",
        ],
    )
    add_receipt(
        target_key="g1pilot",
        preflight_kind="source_layout",
        checks_passed=passed,
        checks_failed=failed,
    )
    add_receipt(
        target_key="g1pilot",
        preflight_kind="launch_request_materialized",
        checks_passed=["launch_request_written"],
        checks_failed=[],
        launch_request=(
            "ros2 launch g1pilot bringup_launcher.launch.py "
            "# not executed by local harness"
        ),
        soft_failures_allowed=True,
    )
    return receipts


def build_phase4_unitree_local_harnesses(
    *,
    chassis: HumanoidChassisProfile,
    joint_limits: list[JointLimitEnvelope],
    command_frames: list[LowLevelCommandFrame],
    local_roots: Mapping[str, str | Path] | None = None,
    sample_count: int = 12,
    timing_iterations: int = 32,
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    Phase4UnitreeLocalHarnessReport,
    list[LowStateTrace],
    list[ImuTrace],
    list[WirelessEStopTrace],
    list[ContactTrace],
    list[TraceReplayReceipt],
    list[MockReceiverReceipt],
    list[StaleDataValidationReceipt],
    list[RosMessageDefinition],
    list[CommandShapeValidationReceipt],
    list[MockTimingRunReceipt],
    list[WatchdogDemotionReceipt],
    list[SafetyStateTransition],
    list[SyntheticSafetyDrillReceipt],
    list[RuntimePreflightReceipt],
]:
    low_state, imu, wireless, contacts = build_synthetic_trace_streams(
        chassis=chassis,
        sample_count=sample_count,
    )
    definitions = build_ros_message_definitions(local_roots)
    command_shape = build_command_shape_validation_receipts(
        command_frames=command_frames,
        definitions=definitions,
    )
    timing_receipts, watchdog_receipts = run_mock_timing_harness(
        iterations=timing_iterations,
    )
    stale_receipts = [
        _validate_stale(stream_key="low_state", rows=low_state, max_allowed_gap_s=0.1),
        _validate_stale(stream_key="imu", rows=imu, max_allowed_gap_s=0.1),
        _validate_stale(
            stream_key="wireless_estop",
            rows=wireless,
            max_allowed_gap_s=0.1,
        ),
        _validate_stale(stream_key="contact", rows=contacts, max_allowed_gap_s=0.1),
    ]
    transitions, drills = run_safety_recovery_harness(
        stale_receipts=stale_receipts,
        wireless_traces=wireless,
        command_frames=command_frames,
        joint_limits=joint_limits,
    )
    preflights = build_runtime_preflight_receipts(local_roots)
    # Replay and receiver receipts need the actual export paths, so the CLI
    # rebuilds them after saving. Use empty placeholders here for direct callers.
    replay_receipts: list[TraceReplayReceipt] = []
    receiver_receipts: list[MockReceiverReceipt] = []

    trace_complete = (
        bool(low_state)
        and bool(imu)
        and bool(wireless)
        and bool(contacts)
        and all(receipt.validation_executed for receipt in stale_receipts)
    )
    command_complete = (
        bool(definitions)
        and all(definition.parsed for definition in definitions)
        and bool(command_shape)
        and all(receipt.no_publish_serialization_ready for receipt in command_shape)
        and not any(receipt.ros2_publish_attempted for receipt in command_shape)
    )
    timing_complete = (
        bool(timing_receipts)
        and all(receipt.local_loop_executed for receipt in timing_receipts)
        and bool(watchdog_receipts)
        and all(not receipt.dds_runtime_observed for receipt in timing_receipts)
    )
    safety_complete = (
        bool(transitions)
        and bool(drills)
        and all(receipt.drill_executed_locally for receipt in drills)
        and not any(receipt.hardware_executed for receipt in drills)
    )
    preflight_complete = (
        bool(preflights)
        and any(receipt.target_key == "unitree_ros2" for receipt in preflights)
        and any(receipt.target_key == "unitree_mujoco" for receipt in preflights)
        and not any(receipt.launch_executed for receipt in preflights)
    )
    complete = (
        trace_complete
        and command_complete
        and timing_complete
        and safety_complete
        and preflight_complete
    )
    report_payload = {
        "chassis_id": chassis.chassis_id,
        "trace_count": len(low_state) + len(imu) + len(wireless) + len(contacts),
        "command_receipts": len(command_shape),
        "preflights": len(preflights),
    }
    report = Phase4UnitreeLocalHarnessReport(
        report_id=stable_id("phase4_unitree_local_harness", report_payload),
        chassis_id=chassis.chassis_id,
        status="ok" if complete else "blocked",
        low_state_trace_count=len(low_state),
        imu_trace_count=len(imu),
        wireless_estop_trace_count=len(wireless),
        contact_trace_count=len(contacts),
        trace_replay_receipt_count=len(replay_receipts),
        mock_receiver_receipt_count=len(receiver_receipts),
        stale_validation_receipt_count=len(stale_receipts),
        ros_message_definition_count=len(definitions),
        command_shape_validation_receipt_count=len(command_shape),
        mock_timing_run_receipt_count=len(timing_receipts),
        watchdog_demotion_receipt_count=len(watchdog_receipts),
        safety_transition_count=len(transitions),
        synthetic_safety_drill_receipt_count=len(drills),
        runtime_preflight_receipt_count=len(preflights),
        trace_stream_harness_complete=trace_complete,
        command_shape_harness_complete=command_complete,
        mock_timing_watchdog_harness_complete=timing_complete,
        safety_recovery_harness_complete=safety_complete,
        runtime_preflight_harness_complete=preflight_complete,
        local_harnesses_complete=complete,
        denied_gates=_denied_gates(),
        remaining_evidence_blockers=[
            "real_lowstate_imu_contact_wireless_estop_streams_missing",
            "real_ros2_sdk2_or_g1pilot_command_echo_missing",
            "dds_network_or_on_robot_timing_missing",
            "physical_safety_calibration_and_stop_distance_missing",
            "operator_teleop_recovery_drill_runtime_missing",
            "honest_mujoco_isaac_or_hardware_execution_missing",
        ],
        artifact_refs=mapping(artifact_refs),
    )
    return (
        report,
        low_state,
        imu,
        wireless,
        contacts,
        replay_receipts,
        receiver_receipts,
        stale_receipts,
        definitions,
        command_shape,
        timing_receipts,
        watchdog_receipts,
        transitions,
        drills,
        preflights,
    )


def save_phase4_unitree_local_harnesses(
    output_dir: str | Path,
    *,
    report: Phase4UnitreeLocalHarnessReport,
    low_state: list[LowStateTrace],
    imu: list[ImuTrace],
    wireless: list[WirelessEStopTrace],
    contacts: list[ContactTrace],
    replay_receipts: list[TraceReplayReceipt],
    receiver_receipts: list[MockReceiverReceipt],
    stale_receipts: list[StaleDataValidationReceipt],
    definitions: list[RosMessageDefinition],
    command_shape: list[CommandShapeValidationReceipt],
    timing_receipts: list[MockTimingRunReceipt],
    watchdog_receipts: list[WatchdogDemotionReceipt],
    transitions: list[SafetyStateTransition],
    drills: list[SyntheticSafetyDrillReceipt],
    preflights: list[RuntimePreflightReceipt],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase4_unitree_local_harness_report_v1.json",
        "low_state_traces_path": output / "unitree_low_state_traces_v1.jsonl",
        "imu_traces_path": output / "unitree_imu_traces_v1.jsonl",
        "wireless_estop_traces_path": output
        / "unitree_wireless_estop_traces_v1.jsonl",
        "contact_traces_path": output / "unitree_contact_traces_v1.jsonl",
        "trace_replay_receipts_path": output
        / "unitree_trace_replay_receipts_v1.jsonl",
        "mock_receiver_receipts_path": output
        / "unitree_mock_receiver_receipts_v1.jsonl",
        "stale_validation_receipts_path": output
        / "unitree_stale_data_validation_receipts_v1.jsonl",
        "ros_message_definitions_path": output
        / "unitree_ros_message_definitions_v1.jsonl",
        "command_shape_receipts_path": output
        / "unitree_command_shape_validation_receipts_v1.jsonl",
        "mock_timing_receipts_path": output
        / "unitree_mock_timing_run_receipts_v1.jsonl",
        "watchdog_demotion_receipts_path": output
        / "unitree_watchdog_demotion_receipts_v1.jsonl",
        "safety_transitions_path": output
        / "unitree_safety_state_transitions_v1.jsonl",
        "synthetic_safety_drills_path": output
        / "unitree_synthetic_safety_drill_receipts_v1.jsonl",
        "runtime_preflight_receipts_path": output
        / "unitree_runtime_preflight_receipts_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    _write_rows(paths["low_state_traces_path"], [row.to_dict() for row in low_state])
    _write_rows(paths["imu_traces_path"], [row.to_dict() for row in imu])
    _write_rows(
        paths["wireless_estop_traces_path"], [row.to_dict() for row in wireless]
    )
    _write_rows(paths["contact_traces_path"], [row.to_dict() for row in contacts])
    _write_rows(
        paths["trace_replay_receipts_path"],
        [receipt.to_dict() for receipt in replay_receipts],
    )
    _write_rows(
        paths["mock_receiver_receipts_path"],
        [receipt.to_dict() for receipt in receiver_receipts],
    )
    _write_rows(
        paths["stale_validation_receipts_path"],
        [receipt.to_dict() for receipt in stale_receipts],
    )
    _write_rows(
        paths["ros_message_definitions_path"],
        [definition.to_dict() for definition in definitions],
    )
    _write_rows(
        paths["command_shape_receipts_path"],
        [receipt.to_dict() for receipt in command_shape],
    )
    _write_rows(
        paths["mock_timing_receipts_path"],
        [receipt.to_dict() for receipt in timing_receipts],
    )
    _write_rows(
        paths["watchdog_demotion_receipts_path"],
        [receipt.to_dict() for receipt in watchdog_receipts],
    )
    _write_rows(
        paths["safety_transitions_path"],
        [transition.to_dict() for transition in transitions],
    )
    _write_rows(
        paths["synthetic_safety_drills_path"],
        [receipt.to_dict() for receipt in drills],
    )
    _write_rows(
        paths["runtime_preflight_receipts_path"],
        [receipt.to_dict() for receipt in preflights],
    )
    return {key: str(path) for key, path in paths.items()}


def load_phase4_unitree_local_harness_report(
    path: str | Path,
) -> Phase4UnitreeLocalHarnessReport:
    return Phase4UnitreeLocalHarnessReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_low_state_traces(path: str | Path) -> list[LowStateTrace]:
    return [LowStateTrace.from_dict(row) for row in _load_rows(path)]


def load_imu_traces(path: str | Path) -> list[ImuTrace]:
    return [ImuTrace.from_dict(row) for row in _load_rows(path)]


def load_wireless_estop_traces(path: str | Path) -> list[WirelessEStopTrace]:
    return [WirelessEStopTrace.from_dict(row) for row in _load_rows(path)]


def load_contact_traces(path: str | Path) -> list[ContactTrace]:
    return [ContactTrace.from_dict(row) for row in _load_rows(path)]


def load_trace_replay_receipts(path: str | Path) -> list[TraceReplayReceipt]:
    return [TraceReplayReceipt.from_dict(row) for row in _load_rows(path)]


def load_mock_receiver_receipts(path: str | Path) -> list[MockReceiverReceipt]:
    return [MockReceiverReceipt.from_dict(row) for row in _load_rows(path)]


def load_stale_data_validation_receipts(
    path: str | Path,
) -> list[StaleDataValidationReceipt]:
    return [StaleDataValidationReceipt.from_dict(row) for row in _load_rows(path)]


def load_ros_message_definitions(path: str | Path) -> list[RosMessageDefinition]:
    return [RosMessageDefinition.from_dict(row) for row in _load_rows(path)]


def load_command_shape_validation_receipts(
    path: str | Path,
) -> list[CommandShapeValidationReceipt]:
    return [CommandShapeValidationReceipt.from_dict(row) for row in _load_rows(path)]


def load_mock_timing_run_receipts(path: str | Path) -> list[MockTimingRunReceipt]:
    return [MockTimingRunReceipt.from_dict(row) for row in _load_rows(path)]


def load_watchdog_demotion_receipts(path: str | Path) -> list[WatchdogDemotionReceipt]:
    return [WatchdogDemotionReceipt.from_dict(row) for row in _load_rows(path)]


def load_safety_state_transitions(path: str | Path) -> list[SafetyStateTransition]:
    return [SafetyStateTransition.from_dict(row) for row in _load_rows(path)]


def load_synthetic_safety_drill_receipts(
    path: str | Path,
) -> list[SyntheticSafetyDrillReceipt]:
    return [SyntheticSafetyDrillReceipt.from_dict(row) for row in _load_rows(path)]


def load_runtime_preflight_receipts(path: str | Path) -> list[RuntimePreflightReceipt]:
    return [RuntimePreflightReceipt.from_dict(row) for row in _load_rows(path)]
