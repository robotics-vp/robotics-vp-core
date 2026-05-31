"""CPU/non-GPU August-gap execution receipts for Unitree Phase 4.

This module stitches together the locally executable Unitree/G1 work that can
be burned down before GPU/provider/hardware access exists. It records ROS2 /
SDK2 build and message-validation receipts, then joins the existing trace,
dry-run, watchdog, safety, MuJoCo, event-spine, replay, and lower-WM ingestion
surfaces without publishing commands or claiming hardware/runtime proof.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord, ReplayWindowRecord
from src.runtime.event_spine import (
    DecisionLedgerEntry,
    RuntimeEvent,
    decision_ledger_sidecar_payload,
    event_spine_sidecar_payload,
)
from src.world_model.humanoid_readiness.common import (
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.unitree_blocker_probes import (
    _compile_probe,
    load_unitree_blocker_stress_probe_receipts,
    load_unitree_mujoco_model_stress_receipts,
)
from src.world_model.humanoid_readiness.unitree_bringup_readiness import (
    default_unitree_local_roots,
)
from src.world_model.humanoid_readiness.unitree_local_harness import (
    build_ros_message_definitions,
    load_command_shape_validation_receipts,
    load_contact_traces,
    load_imu_traces,
    load_low_state_traces,
    load_mock_timing_run_receipts,
    load_safety_state_transitions,
    load_stale_data_validation_receipts,
    load_synthetic_safety_drill_receipts,
    load_trace_replay_receipts,
    load_watchdog_demotion_receipts,
    load_wireless_estop_traces,
)
from src.world_model.humanoid_readiness.unitree_runtime_bridge import (
    load_mujoco_headless_step_receipts,
    load_mujoco_headless_trace_rows,
    load_operator_recovery_drill_receipts,
    load_ros2_runtime_readiness_receipts,
    load_safety_envelope_expansion_receipts,
    load_trace_import_adapter_receipts,
)

CPU_AUGUST_GAP_EXECUTION_REPORT_VERSION = "cpu_august_gap_execution_report_v1"
UNITREE_ROS2_SDK2_BUILD_MESSAGE_VALIDATION_RECEIPT_VERSION = (
    "unitree_ros2_sdk2_build_message_validation_receipt_v1"
)
UNITREE_EVENT_REPLAY_JOIN_ROW_VERSION = "unitree_event_replay_join_row_v1"
UNITREE_LOWER_WM_INGESTION_ROW_VERSION = "unitree_lower_wm_ingestion_row_v1"

DENIED_CPU_AUGUST_GAP_AUTHORITIES = (
    "ros2_publish_attempted",
    "unitree_sdk2_write_enabled",
    "g1pilot_runtime_invoked",
    "hardware_executed",
    "live_policy_control",
    "training_executed",
    "weights_written",
    "provider_executed",
    "gpu_training_executed",
    "reward_math_mutation",
    "phase7_authority_granted",
    "promotion_eligible",
)

ROS2_GENERATED_IMPORT_TARGETS = {
    "unitree_hg.msg": ("LowCmd", "LowState"),
    "unitree_api.msg": ("Request",),
    "unitree_go.msg": ("WirelessController",),
}


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_CPU_AUGUST_GAP_AUTHORITIES}


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    width = max(len(vector) for vector in vectors)
    means: list[float] = []
    for index in range(width):
        values = [float(vector[index]) for vector in vectors if index < len(vector)]
        means.append(_mean(values))
    return means


def _run_shell(command: str, *, cwd: Path, timeout_s: float) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "attempted": True,
            "returncode": None,
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "command timed out")[-2000:],
            "timed_out": True,
        }
    return {
        "attempted": True,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
        "timed_out": False,
    }


def _ros_setup_script() -> str:
    opt_ros = Path("/opt/ros")
    distro = ""
    if opt_ros.exists():
        candidates = sorted(path for path in opt_ros.iterdir() if path.is_dir())
        if candidates:
            distro = candidates[0].name
    if not distro:
        return ""
    for name in ("setup.bash", "setup.sh"):
        candidate = opt_ros / distro / name
        if candidate.exists():
            return str(candidate)
    return ""


def _generated_import_status() -> tuple[dict[str, bool], dict[str, str]]:
    status: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for module_name, symbols in ROS2_GENERATED_IMPORT_TARGETS.items():
        try:
            module = importlib.import_module(module_name)
            missing_symbols = [
                symbol for symbol in symbols if not hasattr(module, symbol)
            ]
            key = f"{module_name}:{','.join(symbols)}"
            status[key] = not missing_symbols
            if missing_symbols:
                errors[key] = "missing_symbols:" + ",".join(missing_symbols)
        except Exception as exc:
            key = f"{module_name}:{','.join(symbols)}"
            status[key] = False
            errors[key] = f"{type(exc).__name__}:{exc}"
    return status, errors


@dataclass(frozen=True)
class UnitreeRos2Sdk2BuildMessageValidationReceipt:
    receipt_id: str
    validation_key: str
    status: str
    succeeded: bool
    local_probe_executed: bool
    target_path: str = ""
    command: str = ""
    tool_status: dict[str, bool] = field(default_factory=dict)
    package_xml_count: int = 0
    msg_definition_count: int = 0
    parsed_message_count: int = 0
    generated_import_status: dict[str, bool] = field(default_factory=dict)
    build_attempted: bool = False
    build_returncode: Optional[int] = None
    import_check_attempted: bool = False
    import_check_succeeded: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    hardware_executed: bool = False
    authority_class: str = "build_message_validation_no_publish_no_write"
    version: str = UNITREE_ROS2_SDK2_BUILD_MESSAGE_VALIDATION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "validation_key": self.validation_key,
            "status": self.status,
            "succeeded": bool(self.succeeded),
            "local_probe_executed": bool(self.local_probe_executed),
            "target_path": self.target_path,
            "command": self.command,
            "tool_status": {str(k): bool(v) for k, v in self.tool_status.items()},
            "package_xml_count": int(self.package_xml_count),
            "msg_definition_count": int(self.msg_definition_count),
            "parsed_message_count": int(self.parsed_message_count),
            "generated_import_status": {
                str(k): bool(v) for k, v in self.generated_import_status.items()
            },
            "build_attempted": bool(self.build_attempted),
            "build_returncode": self.build_returncode,
            "import_check_attempted": bool(self.import_check_attempted),
            "import_check_succeeded": bool(self.import_check_succeeded),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "blockers": strings(self.blockers),
            "metadata": mapping(self.metadata),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnitreeRos2Sdk2BuildMessageValidationReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            validation_key=str(payload.get("validation_key", "")),
            status=str(payload.get("status", "blocked")),
            succeeded=bool(payload.get("succeeded", False)),
            local_probe_executed=bool(payload.get("local_probe_executed", False)),
            target_path=str(payload.get("target_path", "")),
            command=str(payload.get("command", "")),
            tool_status={
                str(key): bool(value)
                for key, value in dict(payload.get("tool_status", {}) or {}).items()
            },
            package_xml_count=int(payload.get("package_xml_count", 0) or 0),
            msg_definition_count=int(payload.get("msg_definition_count", 0) or 0),
            parsed_message_count=int(payload.get("parsed_message_count", 0) or 0),
            generated_import_status={
                str(key): bool(value)
                for key, value in dict(
                    payload.get("generated_import_status", {}) or {}
                ).items()
            },
            build_attempted=bool(payload.get("build_attempted", False)),
            build_returncode=payload.get("build_returncode"),
            import_check_attempted=bool(payload.get("import_check_attempted", False)),
            import_check_succeeded=bool(payload.get("import_check_succeeded", False)),
            stdout_tail=str(payload.get("stdout_tail", "")),
            stderr_tail=str(payload.get("stderr_tail", "")),
            blockers=strings(payload.get("blockers")),
            metadata=mapping(payload.get("metadata")),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get(
                    "authority_class",
                    "build_message_validation_no_publish_no_write",
                )
            ),
            version=str(
                payload.get(
                    "version",
                    UNITREE_ROS2_SDK2_BUILD_MESSAGE_VALIDATION_RECEIPT_VERSION,
                )
            ),
        )


@dataclass(frozen=True)
class UnitreeEventReplayJoinRow:
    join_id: str
    join_key: str
    source_receipt_count: int
    event_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    replay_episode_ids: list[str] = field(default_factory=list)
    replay_step_count: int = 0
    join_status: str = "blocked"
    blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    authority_class: str = "event_replay_join_shadow_only"
    promotion_eligible: bool = False
    version: str = UNITREE_EVENT_REPLAY_JOIN_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_id": self.join_id,
            "version": self.version,
            "join_key": self.join_key,
            "source_receipt_count": int(self.source_receipt_count),
            "event_ids": strings(self.event_ids),
            "decision_ids": strings(self.decision_ids),
            "replay_episode_ids": strings(self.replay_episode_ids),
            "replay_step_count": int(self.replay_step_count),
            "join_status": self.join_status,
            "blockers": strings(self.blockers),
            "artifact_refs": mapping(self.artifact_refs),
            "authority_class": self.authority_class,
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeEventReplayJoinRow":
        return cls(
            join_id=str(payload.get("join_id", "")),
            join_key=str(payload.get("join_key", "")),
            source_receipt_count=int(payload.get("source_receipt_count", 0) or 0),
            event_ids=strings(payload.get("event_ids")),
            decision_ids=strings(payload.get("decision_ids")),
            replay_episode_ids=strings(payload.get("replay_episode_ids")),
            replay_step_count=int(payload.get("replay_step_count", 0) or 0),
            join_status=str(payload.get("join_status", "blocked")),
            blockers=strings(payload.get("blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            authority_class=str(
                payload.get("authority_class", "event_replay_join_shadow_only")
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", UNITREE_EVENT_REPLAY_JOIN_ROW_VERSION)),
        )


@dataclass(frozen=True)
class UnitreeLowerWMIngestionRow:
    ingestion_row_id: str
    wm_key: str
    canonical_surface: str
    ingestion_status: str
    source_receipt_count: int
    source_trace_count: int
    event_spine_ref: str = ""
    replay_ref: str = ""
    source_artifact_refs: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready_for_economic_shadow: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    authority_class: str = "lower_wm_ingestion_shadow_only"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = UNITREE_LOWER_WM_INGESTION_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion_row_id": self.ingestion_row_id,
            "version": self.version,
            "wm_key": self.wm_key,
            "canonical_surface": self.canonical_surface,
            "ingestion_status": self.ingestion_status,
            "source_receipt_count": int(self.source_receipt_count),
            "source_trace_count": int(self.source_trace_count),
            "event_spine_ref": self.event_spine_ref,
            "replay_ref": self.replay_ref,
            "source_artifact_refs": mapping(self.source_artifact_refs),
            "blockers": strings(self.blockers),
            "ready_for_economic_shadow": bool(self.ready_for_economic_shadow),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeLowerWMIngestionRow":
        return cls(
            ingestion_row_id=str(payload.get("ingestion_row_id", "")),
            wm_key=str(payload.get("wm_key", "")),
            canonical_surface=str(payload.get("canonical_surface", "")),
            ingestion_status=str(payload.get("ingestion_status", "blocked")),
            source_receipt_count=int(payload.get("source_receipt_count", 0) or 0),
            source_trace_count=int(payload.get("source_trace_count", 0) or 0),
            event_spine_ref=str(payload.get("event_spine_ref", "")),
            replay_ref=str(payload.get("replay_ref", "")),
            source_artifact_refs=mapping(payload.get("source_artifact_refs")),
            blockers=strings(payload.get("blockers")),
            ready_for_economic_shadow=bool(
                payload.get("ready_for_economic_shadow", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get("authority_class", "lower_wm_ingestion_shadow_only")
            ),
            metadata=mapping(payload.get("metadata")),
            version=str(payload.get("version", UNITREE_LOWER_WM_INGESTION_ROW_VERSION)),
        )


@dataclass(frozen=True)
class CpuAugustGapExecutionReport:
    report_id: str
    status: str
    ros2_sdk2_build_message_validation_complete: bool
    trace_import_complete: bool
    command_dry_run_complete: bool
    timing_watchdog_complete: bool
    safety_recovery_complete: bool
    cpu_mujoco_probe_complete: bool
    event_spine_replay_joins_complete: bool
    lower_wm_ingestion_complete: bool
    cpu_august_gap_tranche_complete: bool
    validation_receipt_count: int
    event_count: int
    decision_count: int
    replay_episode_count: int
    replay_step_count: int
    replay_window_count: int
    event_replay_join_row_count: int
    lower_wm_ingestion_row_count: int
    ros2_build_attempted: bool = False
    ros2_build_succeeded: bool = False
    generated_message_import_succeeded: bool = False
    sdk2_header_compile_succeeded: bool = False
    sdk2_cmake_build_attempted: bool = False
    sdk2_cmake_build_succeeded: bool = False
    minimal_mujoco_headless_step_executed: bool = False
    g1_mujoco_model_stress_succeeded: bool = False
    live_stream_observed: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    gpu_training_executed: bool = False
    reward_math_mutation: bool = False
    phase7_authority_granted: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    remaining_evidence_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = CPU_AUGUST_GAP_EXECUTION_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "ros2_sdk2_build_message_validation_complete": bool(
                self.ros2_sdk2_build_message_validation_complete
            ),
            "trace_import_complete": bool(self.trace_import_complete),
            "command_dry_run_complete": bool(self.command_dry_run_complete),
            "timing_watchdog_complete": bool(self.timing_watchdog_complete),
            "safety_recovery_complete": bool(self.safety_recovery_complete),
            "cpu_mujoco_probe_complete": bool(self.cpu_mujoco_probe_complete),
            "event_spine_replay_joins_complete": bool(
                self.event_spine_replay_joins_complete
            ),
            "lower_wm_ingestion_complete": bool(self.lower_wm_ingestion_complete),
            "cpu_august_gap_tranche_complete": bool(
                self.cpu_august_gap_tranche_complete
            ),
            "validation_receipt_count": int(self.validation_receipt_count),
            "event_count": int(self.event_count),
            "decision_count": int(self.decision_count),
            "replay_episode_count": int(self.replay_episode_count),
            "replay_step_count": int(self.replay_step_count),
            "replay_window_count": int(self.replay_window_count),
            "event_replay_join_row_count": int(self.event_replay_join_row_count),
            "lower_wm_ingestion_row_count": int(self.lower_wm_ingestion_row_count),
            "ros2_build_attempted": bool(self.ros2_build_attempted),
            "ros2_build_succeeded": bool(self.ros2_build_succeeded),
            "generated_message_import_succeeded": bool(
                self.generated_message_import_succeeded
            ),
            "sdk2_header_compile_succeeded": bool(
                self.sdk2_header_compile_succeeded
            ),
            "sdk2_cmake_build_attempted": bool(self.sdk2_cmake_build_attempted),
            "sdk2_cmake_build_succeeded": bool(self.sdk2_cmake_build_succeeded),
            "minimal_mujoco_headless_step_executed": bool(
                self.minimal_mujoco_headless_step_executed
            ),
            "g1_mujoco_model_stress_succeeded": bool(
                self.g1_mujoco_model_stress_succeeded
            ),
            "live_stream_observed": bool(self.live_stream_observed),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "gpu_training_executed": bool(self.gpu_training_executed),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": dict(self.denied_gates),
            "remaining_evidence_blockers": strings(self.remaining_evidence_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CpuAugustGapExecutionReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "blocked")),
            ros2_sdk2_build_message_validation_complete=bool(
                payload.get("ros2_sdk2_build_message_validation_complete", False)
            ),
            trace_import_complete=bool(payload.get("trace_import_complete", False)),
            command_dry_run_complete=bool(
                payload.get("command_dry_run_complete", False)
            ),
            timing_watchdog_complete=bool(
                payload.get("timing_watchdog_complete", False)
            ),
            safety_recovery_complete=bool(
                payload.get("safety_recovery_complete", False)
            ),
            cpu_mujoco_probe_complete=bool(
                payload.get("cpu_mujoco_probe_complete", False)
            ),
            event_spine_replay_joins_complete=bool(
                payload.get("event_spine_replay_joins_complete", False)
            ),
            lower_wm_ingestion_complete=bool(
                payload.get("lower_wm_ingestion_complete", False)
            ),
            cpu_august_gap_tranche_complete=bool(
                payload.get("cpu_august_gap_tranche_complete", False)
            ),
            validation_receipt_count=int(
                payload.get("validation_receipt_count", 0) or 0
            ),
            event_count=int(payload.get("event_count", 0) or 0),
            decision_count=int(payload.get("decision_count", 0) or 0),
            replay_episode_count=int(payload.get("replay_episode_count", 0) or 0),
            replay_step_count=int(payload.get("replay_step_count", 0) or 0),
            replay_window_count=int(payload.get("replay_window_count", 0) or 0),
            event_replay_join_row_count=int(
                payload.get("event_replay_join_row_count", 0) or 0
            ),
            lower_wm_ingestion_row_count=int(
                payload.get("lower_wm_ingestion_row_count", 0) or 0
            ),
            ros2_build_attempted=bool(payload.get("ros2_build_attempted", False)),
            ros2_build_succeeded=bool(payload.get("ros2_build_succeeded", False)),
            generated_message_import_succeeded=bool(
                payload.get("generated_message_import_succeeded", False)
            ),
            sdk2_header_compile_succeeded=bool(
                payload.get("sdk2_header_compile_succeeded", False)
            ),
            sdk2_cmake_build_attempted=bool(
                payload.get("sdk2_cmake_build_attempted", False)
            ),
            sdk2_cmake_build_succeeded=bool(
                payload.get("sdk2_cmake_build_succeeded", False)
            ),
            minimal_mujoco_headless_step_executed=bool(
                payload.get("minimal_mujoco_headless_step_executed", False)
            ),
            g1_mujoco_model_stress_succeeded=bool(
                payload.get("g1_mujoco_model_stress_succeeded", False)
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
            provider_executed=bool(payload.get("provider_executed", False)),
            gpu_training_executed=bool(
                payload.get("gpu_training_executed", False)
            ),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
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
                payload.get("version", CPU_AUGUST_GAP_EXECUTION_REPORT_VERSION)
            ),
        )


def build_unitree_ros2_sdk2_build_message_validation_receipts(
    *,
    local_roots: Mapping[str, str | Path] | None = None,
    scratch_dir: str | Path | None = None,
    allow_build_attempt: bool = True,
    build_timeout_s: float = 120.0,
) -> list[UnitreeRos2Sdk2BuildMessageValidationReceipt]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    ros_root = Path(roots["unitree_ros2"])
    workspace = ros_root / "cyclonedds_ws"
    sdk2_root = Path(roots["unitree_sdk2"])
    scratch = Path(scratch_dir) if scratch_dir else Path("artifacts/unitree_build_probe")

    package_xml_count = len(list((workspace / "src").glob("**/package.xml")))
    msg_definition_count = len(list((workspace / "src").glob("**/*.msg")))
    definitions = build_ros_message_definitions(roots)
    parsed_count = sum(1 for definition in definitions if definition.parsed)
    expected_messages = {definition.message_name for definition in definitions}
    static_succeeded = (
        {"LowCmd", "LowState", "IMUState", "Request", "WirelessController"}
        <= expected_messages
        and parsed_count >= 5
    )
    receipts: list[UnitreeRos2Sdk2BuildMessageValidationReceipt] = []
    static_payload = {
        "validation_key": "ros2_static_message_definition_validation",
        "ros_root": str(ros_root),
        "parsed_count": parsed_count,
        "msg_count": msg_definition_count,
    }
    receipts.append(
        UnitreeRos2Sdk2BuildMessageValidationReceipt(
            receipt_id=stable_id("unitree_ros2_sdk2_validation", static_payload),
            validation_key="ros2_static_message_definition_validation",
            status="ok_static_message_definitions_parsed"
            if static_succeeded
            else "blocked_message_definition_parse_gap",
            succeeded=static_succeeded,
            local_probe_executed=True,
            target_path=str(workspace / "src"),
            package_xml_count=package_xml_count,
            msg_definition_count=msg_definition_count,
            parsed_message_count=parsed_count,
            blockers=[]
            if static_succeeded
            else ["unitree_ros2_message_definitions_missing_or_unparsed"],
            metadata={
                "parsed_messages": [
                    definition.message_name
                    for definition in definitions
                    if definition.parsed
                ],
                "parse_errors": {
                    definition.message_name: definition.parse_errors
                    for definition in definitions
                    if definition.parse_errors
                },
            },
        )
    )

    import_status, import_errors = _generated_import_status()
    import_succeeded = bool(import_status) and all(import_status.values())
    import_payload = {
        "validation_key": "ros2_generated_message_import_validation",
        "import_status": import_status,
    }
    receipts.append(
        UnitreeRos2Sdk2BuildMessageValidationReceipt(
            receipt_id=stable_id("unitree_ros2_sdk2_validation", import_payload),
            validation_key="ros2_generated_message_import_validation",
            status="ok_generated_message_imports"
            if import_succeeded
            else "blocked_generated_message_imports_missing",
            succeeded=import_succeeded,
            local_probe_executed=True,
            target_path=str(workspace / "install"),
            command=(
                "python3 -c \"from unitree_hg.msg import LowCmd, LowState; "
                "from unitree_api.msg import Request; "
                "from unitree_go.msg import WirelessController\""
            ),
            package_xml_count=package_xml_count,
            msg_definition_count=msg_definition_count,
            parsed_message_count=parsed_count,
            generated_import_status=import_status,
            import_check_attempted=True,
            import_check_succeeded=import_succeeded,
            blockers=[] if import_succeeded else ["generated_unitree_messages_not_importable"],
            metadata={"import_errors": import_errors},
        )
    )

    required_tools = ("python3", "cmake", "colcon", "ros2")
    tool_status = {tool: shutil.which(tool) is not None for tool in required_tools}
    missing_tools = [tool for tool, present in tool_status.items() if not present]
    ros_setup = _ros_setup_script()
    ready_for_build = (
        workspace.exists()
        and bool(ros_setup)
        and package_xml_count > 0
        and msg_definition_count > 0
        and not missing_tools
    )
    build_command = ""
    build_result = {
        "attempted": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }
    if ready_for_build:
        build_command = (
            f"source {ros_setup} && colcon build --symlink-install "
            "--packages-select unitree_hg unitree_go unitree_api"
        )
    if allow_build_attempt and ready_for_build:
        build_result = _run_shell(
            build_command,
            cwd=workspace,
            timeout_s=build_timeout_s,
        )
    ros2_build_succeeded = build_result.get("returncode") == 0
    if ros2_build_succeeded:
        ros2_status = "ok_ros2_colcon_build_completed"
        ros2_blockers: list[str] = []
    elif build_result.get("attempted"):
        ros2_status = "blocked_ros2_colcon_build_failed"
        ros2_blockers = ["ros2_colcon_build_failed"]
    elif not allow_build_attempt:
        ros2_status = "not_executed_build_probe_disabled"
        ros2_blockers = ["build_probe_disabled"]
    else:
        ros2_status = "blocked_missing_ros2_colcon_build_runtime"
        ros2_blockers = [
            *missing_tools,
            *(["/opt/ros setup script"] if not ros_setup else []),
        ]
    ros2_payload = {
        "validation_key": "ros2_colcon_build_validation",
        "workspace": str(workspace),
        "tool_status": tool_status,
        "ros_setup": ros_setup,
        "attempted": build_result.get("attempted"),
        "returncode": build_result.get("returncode"),
    }
    receipts.append(
        UnitreeRos2Sdk2BuildMessageValidationReceipt(
            receipt_id=stable_id("unitree_ros2_sdk2_validation", ros2_payload),
            validation_key="ros2_colcon_build_validation",
            status=ros2_status,
            succeeded=ros2_build_succeeded,
            local_probe_executed=True,
            target_path=str(workspace),
            command=build_command
            or "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build",
            tool_status=tool_status,
            package_xml_count=package_xml_count,
            msg_definition_count=msg_definition_count,
            parsed_message_count=parsed_count,
            build_attempted=bool(build_result.get("attempted")),
            build_returncode=_optional_int(build_result.get("returncode")),
            stdout_tail=str(build_result.get("stdout", "")),
            stderr_tail=str(build_result.get("stderr", "")),
            blockers=ros2_blockers,
            metadata={
                "ros_setup_script": ros_setup,
                "timed_out": build_result.get("timed_out"),
            },
        )
    )

    sdk2_header_result = _compile_probe(
        source=(
            "#include <unitree/robot/channel/channel_factory.hpp>\n"
            "#include <unitree/robot/channel/channel_publisher.hpp>\n"
            "#include <unitree/idl/go2/LowCmd_.hpp>\n"
            "#include <unitree/idl/hg/LowCmd_.hpp>\n"
            "#include <dds/dds.h>\n"
            "int main() { return 0; }\n"
        ),
        include_dirs=[sdk2_root / "include", sdk2_root / "thirdparty/include"],
        output_name="unitree_sdk2_header_probe.o",
        timeout_s=10.0,
    )
    sdk2_header_succeeded = sdk2_header_result.get("returncode") == 0
    sdk2_header_payload = {
        "validation_key": "unitree_sdk2_header_compile_validation",
        "sdk2_root": str(sdk2_root),
        "returncode": sdk2_header_result.get("returncode"),
    }
    receipts.append(
        UnitreeRos2Sdk2BuildMessageValidationReceipt(
            receipt_id=stable_id("unitree_ros2_sdk2_validation", sdk2_header_payload),
            validation_key="unitree_sdk2_header_compile_validation",
            status="ok_sdk2_header_compile_only"
            if sdk2_header_succeeded
            else "blocked_sdk2_header_compile_failed",
            succeeded=sdk2_header_succeeded,
            local_probe_executed=bool(sdk2_header_result.get("attempted")),
            target_path=str(sdk2_root),
            command="c++ -std=c++17 -c Unitree SDK2 header probe",
            tool_status={"c++": shutil.which("c++") is not None},
            build_attempted=bool(sdk2_header_result.get("attempted")),
            build_returncode=_optional_int(sdk2_header_result.get("returncode")),
            stdout_tail=str(sdk2_header_result.get("stdout", "")),
            stderr_tail=str(sdk2_header_result.get("stderr", "")),
            blockers=[]
            if sdk2_header_succeeded
            else ["unitree_sdk2_header_compile_failed"],
            metadata={"compile_result": mapping(sdk2_header_result)},
        )
    )

    cmake_status = {"cmake": shutil.which("cmake") is not None}
    cmake_lists = sdk2_root / "CMakeLists.txt"
    linux_host = platform.system().lower() == "linux"
    sdk2_build_dir = scratch / "unitree_sdk2_cmake_build"
    sdk2_ready = linux_host and cmake_status["cmake"] and cmake_lists.exists()
    sdk2_command = (
        f"cmake -S {sdk2_root} -B {sdk2_build_dir} && "
        f"cmake --build {sdk2_build_dir} --parallel 2"
    )
    sdk2_result = {
        "attempted": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }
    if allow_build_attempt and sdk2_ready:
        sdk2_build_dir.parent.mkdir(parents=True, exist_ok=True)
        sdk2_result = _run_shell(
            sdk2_command,
            cwd=sdk2_root,
            timeout_s=build_timeout_s,
        )
    sdk2_build_succeeded = sdk2_result.get("returncode") == 0
    if sdk2_build_succeeded:
        sdk2_status = "ok_sdk2_cmake_build_completed"
        sdk2_blockers: list[str] = []
    elif sdk2_result.get("attempted"):
        sdk2_status = "blocked_sdk2_cmake_build_failed"
        sdk2_blockers = ["unitree_sdk2_cmake_build_failed"]
    elif not allow_build_attempt:
        sdk2_status = "not_executed_build_probe_disabled"
        sdk2_blockers = ["build_probe_disabled"]
    else:
        sdk2_status = "blocked_sdk2_cmake_build_requires_linux_runtime"
        sdk2_blockers = [
            *(["linux_host_required"] if not linux_host else []),
            *(["cmake"] if not cmake_status["cmake"] else []),
            *(["unitree_sdk2_CMakeLists_missing"] if not cmake_lists.exists() else []),
        ]
    sdk2_payload = {
        "validation_key": "unitree_sdk2_cmake_build_validation",
        "sdk2_root": str(sdk2_root),
        "linux_host": linux_host,
        "attempted": sdk2_result.get("attempted"),
        "returncode": sdk2_result.get("returncode"),
    }
    receipts.append(
        UnitreeRos2Sdk2BuildMessageValidationReceipt(
            receipt_id=stable_id("unitree_ros2_sdk2_validation", sdk2_payload),
            validation_key="unitree_sdk2_cmake_build_validation",
            status=sdk2_status,
            succeeded=sdk2_build_succeeded,
            local_probe_executed=True,
            target_path=str(sdk2_root),
            command=sdk2_command,
            tool_status=cmake_status,
            build_attempted=bool(sdk2_result.get("attempted")),
            build_returncode=_optional_int(sdk2_result.get("returncode")),
            stdout_tail=str(sdk2_result.get("stdout", "")),
            stderr_tail=str(sdk2_result.get("stderr", "")),
            blockers=sdk2_blockers,
            metadata={
                "platform": platform.system(),
                "cmake_lists_present": cmake_lists.exists(),
                "build_dir": str(sdk2_build_dir),
                "timed_out": sdk2_result.get("timed_out"),
            },
        )
    )
    return receipts


def _validation_receipts_by_key(
    receipts: Iterable[UnitreeRos2Sdk2BuildMessageValidationReceipt],
) -> dict[str, UnitreeRos2Sdk2BuildMessageValidationReceipt]:
    return {receipt.validation_key: receipt for receipt in receipts}


def _episode_replay_rows(
    *,
    low_state: Sequence[Any],
    imu: Sequence[Any],
    wireless: Sequence[Any],
    contacts: Sequence[Any],
    event_ids: Sequence[str],
    decision_ids: Sequence[str],
    artifact_refs: Mapping[str, Any],
) -> tuple[list[ReplayEpisodeRecord], list[ReplayStepRecord], list[ReplayWindowRecord]]:
    if not low_state:
        return [], [], []
    run_id = "cpu_august_gap_unitree_trace"
    episode_id = "unitree_cpu_august_gap_episode_000"
    timestamp = "2026-05-30T00:00:00+00:00"
    imu_by_index = {row.sample_index: row for row in imu}
    wireless_by_index = {row.sample_index: row for row in wireless}
    contact_by_index = {row.sample_index: row for row in contacts}
    steps: list[ReplayStepRecord] = []
    for index, state in enumerate(low_state):
        imu_row = imu_by_index.get(state.sample_index)
        wireless_row = wireless_by_index.get(state.sample_index)
        contact_row = contact_by_index.get(state.sample_index)
        joint_values = [
            float(value)
            for _, value in sorted(state.joint_positions.items())[:8]
        ]
        velocity_values = [
            float(value)
            for _, value in sorted(state.joint_velocities.items())[:8]
        ]
        gyro = list(getattr(imu_row, "gyro_rad_s", []) or [])[:3]
        accel = list(getattr(imu_row, "accel_m_s2", []) or [])[:3]
        estop = 1.0 if getattr(wireless_row, "estop_pressed", False) else 0.0
        contact_count = float(
            sum(1 for value in getattr(contact_row, "contact_states", {}).values() if value)
        )
        obs_vector = [
            *joint_values,
            *velocity_values,
            *[float(value) for value in gyro],
            *[float(value) for value in accel],
            estop,
            contact_count,
        ]
        action_vector = [0.0 for _ in joint_values[:8]]
        constraint_flags = [
            {
                "constraint": "no_live_dispatch",
                "satisfied": True,
                "source": "cpu_august_gap_tranche",
            },
            {
                "constraint": "estop_pressed",
                "satisfied": not bool(estop),
                "source": "wireless_estop_trace",
            },
        ]
        steps.append(
            ReplayStepRecord(
                run_id=run_id,
                episode_id=episode_id,
                step_idx=index,
                obs={
                    "low_state_trace_id": state.trace_id,
                    "sample_index": state.sample_index,
                    "tick": state.tick,
                    "motor_count": state.motor_count,
                    "estop_pressed": bool(estop),
                    "contact_count": int(contact_count),
                },
                obs_vector=obs_vector,
                action={
                    "dispatch": "denied",
                    "command_family": "dry_run_no_publish",
                    "source": "cpu_august_gap_tranche",
                },
                action_vector=action_vector,
                reward=0.0,
                reward_decomposition={
                    "shadow_only": 0.0,
                    "hardware_reward_unavailable": True,
                },
                done=index == len(low_state) - 1,
                task_id="unitree_g1_cpu_august_gap_validation",
                env_id="local_no_hardware_no_ros_publish",
                condition_vector={
                    "source_domain": "unitree_cpu_august_gap_trace",
                    "hardware_executed": False,
                    "promotion_eligible": False,
                },
                condition_vector_values=[estop, contact_count],
                skill_mode="safety_first_shadow_trace",
                objective_tensor_summary={
                    "objective_profile_id": "cpu_august_gap_shadow",
                    "promotion_eligible": False,
                },
                objective_tensor_ref=None,
                econ_tensor_summary={
                    "authority_class": "economic_shadow_join_only",
                    "reward_math_mutation": False,
                },
                econ_tensor_ref=None,
                constraint_flags=constraint_flags,
                pricing_tick_ref=None,
                ledger_event_ref=decision_ids[0] if decision_ids else None,
                source_domain="unitree_cpu_august_gap_trace",
                seed=0,
                timestamp=timestamp,
                metadata={
                    "event_refs": list(event_ids),
                    "decision_refs": list(decision_ids),
                    "artifact_refs": mapping(artifact_refs),
                    "source_adapter": "cpu_august_gap_unitree_trace_replay",
                },
                provenance={
                    "component": "humanoid_readiness.cpu_august_gap",
                    "input_trace_id": state.trace_id,
                },
            )
        )
    episode = ReplayEpisodeRecord(
        run_id=run_id,
        episode_id=episode_id,
        task_id="unitree_g1_cpu_august_gap_validation",
        env_id="local_no_hardware_no_ros_publish",
        source_domain="unitree_cpu_august_gap_trace",
        seed=0,
        status="shadow_trace_only",
        started_at=timestamp,
        ended_at=timestamp,
        total_steps=len(steps),
        total_reward=sum(step.reward for step in steps),
        skill_mode="safety_first_shadow_trace",
        condition_vector={
            "source_domain": "unitree_cpu_august_gap_trace",
            "hardware_executed": False,
            "promotion_eligible": False,
        },
        condition_vector_values=[0.0, float(len(steps))],
        objective_tensor_summary={
            "objective_profile_id": "cpu_august_gap_shadow",
            "promotion_eligible": False,
        },
        objective_tensor_ref=None,
        econ_tensor_summary={
            "authority_class": "economic_shadow_join_only",
            "reward_math_mutation": False,
        },
        econ_tensor_ref=None,
        pricing_summary={"pricing_live": False, "shadow_join_only": True},
        pricing_tick_refs=[],
        constraint_flags=[
            {
                "constraint": "no_live_dispatch",
                "satisfied": True,
                "source": "cpu_august_gap_tranche",
            }
        ],
        regal_summary={"promotion_eligible": False, "authority_granted": False},
        datapack_summary={
            "row_count": len(steps),
            "source": "unitree_cpu_august_gap_trace",
        },
        ledger_event_ids=list(decision_ids),
        metadata={
            "event_refs": list(event_ids),
            "decision_refs": list(decision_ids),
            "artifact_refs": mapping(artifact_refs),
        },
        provenance={"component": "humanoid_readiness.cpu_august_gap"},
    )
    window = ReplayWindowRecord(
        run_id=run_id,
        episode_id=episode_id,
        window_id=f"{episode_id}:0000_{len(steps) - 1:04d}",
        start_step=0,
        end_step=len(steps) - 1,
        task_id=episode.task_id,
        env_id=episode.env_id,
        source_domain=episode.source_domain,
        seed=0,
        timestamp=timestamp,
        reward_sum=sum(step.reward for step in steps),
        obs_vector_mean=_mean_vector([step.obs_vector for step in steps]),
        action_vector_mean=_mean_vector([step.action_vector for step in steps]),
        condition_vector=dict(episode.condition_vector),
        condition_vector_values=list(episode.condition_vector_values),
        skill_mode=episode.skill_mode,
        objective_tensor_summary=dict(episode.objective_tensor_summary),
        econ_tensor_summary=dict(episode.econ_tensor_summary),
        pricing_summary=dict(episode.pricing_summary),
        constraint_flags=list(episode.constraint_flags),
        metadata={
            "event_refs": list(event_ids),
            "decision_refs": list(decision_ids),
            "source_adapter": "cpu_august_gap_unitree_trace_replay",
        },
        provenance=dict(episode.provenance),
    )
    return [episode], steps, [window]


def _add_event(
    events: list[RuntimeEvent],
    *,
    kind: str,
    sequence_idx: int,
    receipt_refs: Sequence[str],
    artifact_refs: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> RuntimeEvent:
    event = RuntimeEvent.from_components(
        run_id="cpu_august_gap_unitree",
        episode_id="unitree_cpu_august_gap_episode_000",
        timestamp="2026-05-30T00:00:00+00:00",
        event_kind=kind,
        sequence_idx=sequence_idx,
        scope={"scope_kind": "cpu_august_gap_tranche", "robot_family": "unitree_g1"},
        runtime_packet_id=None,
        contract_id="unitree_cpu_august_gap_contract_v1",
        receipt_label_refs=list(receipt_refs),
        artifact_refs=artifact_refs,
        provenance={"component": "humanoid_readiness.cpu_august_gap"},
        metadata=metadata,
    )
    events.append(event)
    return event


def _add_decision(
    decisions: list[DecisionLedgerEntry],
    *,
    kind: str,
    outcome: str,
    sequence_idx: int,
    source_event_ids: Sequence[str],
    receipt_refs: Sequence[str],
    reasons: Sequence[str],
    artifact_refs: Mapping[str, Any],
) -> DecisionLedgerEntry:
    decision = DecisionLedgerEntry.from_components(
        run_id="cpu_august_gap_unitree",
        episode_id="unitree_cpu_august_gap_episode_000",
        timestamp="2026-05-30T00:00:00+00:00",
        decision_kind=kind,
        outcome=outcome,
        sequence_idx=sequence_idx,
        scope={"scope_kind": "cpu_august_gap_tranche", "robot_family": "unitree_g1"},
        reasons=list(reasons),
        source_event_ids=list(source_event_ids),
        runtime_packet_id=None,
        contract_id="unitree_cpu_august_gap_contract_v1",
        receipt_label_refs=list(receipt_refs),
        artifact_refs=artifact_refs,
        provenance={"component": "humanoid_readiness.cpu_august_gap"},
    )
    decisions.append(decision)
    return decision


def build_cpu_august_gap_event_replay_lower_wm_surfaces(
    *,
    validation_receipts: Sequence[UnitreeRos2Sdk2BuildMessageValidationReceipt],
    phase4_unitree_local_harness_dir: str | Path,
    phase4_unitree_runtime_bridge_dir: str | Path,
    phase4_unitree_blocker_stress_probe_dir: str | Path,
    artifact_refs: Mapping[str, Any],
) -> tuple[
    list[RuntimeEvent],
    list[DecisionLedgerEntry],
    list[ReplayEpisodeRecord],
    list[ReplayStepRecord],
    list[ReplayWindowRecord],
    list[UnitreeEventReplayJoinRow],
    list[UnitreeLowerWMIngestionRow],
]:
    harness = Path(phase4_unitree_local_harness_dir)
    runtime = Path(phase4_unitree_runtime_bridge_dir)
    blockers = Path(phase4_unitree_blocker_stress_probe_dir)

    low_state = load_low_state_traces(harness / "unitree_low_state_traces_v1.jsonl")
    imu = load_imu_traces(harness / "unitree_imu_traces_v1.jsonl")
    wireless = load_wireless_estop_traces(
        harness / "unitree_wireless_estop_traces_v1.jsonl"
    )
    contacts = load_contact_traces(harness / "unitree_contact_traces_v1.jsonl")
    trace_replay = load_trace_replay_receipts(
        harness / "unitree_trace_replay_receipts_v1.jsonl"
    )
    stale = load_stale_data_validation_receipts(
        harness / "unitree_stale_data_validation_receipts_v1.jsonl"
    )
    command_shapes = load_command_shape_validation_receipts(
        harness / "unitree_command_shape_validation_receipts_v1.jsonl"
    )
    timing = load_mock_timing_run_receipts(
        harness / "unitree_mock_timing_run_receipts_v1.jsonl"
    )
    watchdog = load_watchdog_demotion_receipts(
        harness / "unitree_watchdog_demotion_receipts_v1.jsonl"
    )
    safety_transitions = load_safety_state_transitions(
        harness / "unitree_safety_state_transitions_v1.jsonl"
    )
    safety_drills = load_synthetic_safety_drill_receipts(
        harness / "unitree_synthetic_safety_drill_receipts_v1.jsonl"
    )
    ros2_runtime = load_ros2_runtime_readiness_receipts(
        runtime / "unitree_ros2_runtime_readiness_receipts_v1.jsonl"
    )
    trace_adapters = load_trace_import_adapter_receipts(
        runtime / "unitree_trace_import_adapter_receipts_v1.jsonl"
    )
    mujoco_steps = load_mujoco_headless_step_receipts(
        runtime / "unitree_mujoco_headless_step_receipts_v1.jsonl"
    )
    mujoco_rows = load_mujoco_headless_trace_rows(
        runtime / "unitree_mujoco_headless_trace_rows_v1.jsonl"
    )
    safety_envelopes = load_safety_envelope_expansion_receipts(
        runtime / "unitree_safety_envelope_expansion_receipts_v1.jsonl"
    )
    operator_drills = load_operator_recovery_drill_receipts(
        runtime / "unitree_operator_recovery_drill_receipts_v1.jsonl"
    )
    blocker_receipts = load_unitree_blocker_stress_probe_receipts(
        blockers / "unitree_blocker_stress_probe_receipts_v1.jsonl"
    )
    mujoco_model_stress = load_unitree_mujoco_model_stress_receipts(
        blockers / "unitree_mujoco_model_stress_receipts_v1.jsonl"
    )

    events: list[RuntimeEvent] = []
    decisions: list[DecisionLedgerEntry] = []
    sequence = 0
    for receipt in validation_receipts:
        event = _add_event(
            events,
            kind=f"{receipt.validation_key}_recorded",
            sequence_idx=sequence,
            receipt_refs=[receipt.receipt_id],
            artifact_refs=artifact_refs,
            metadata={
                "status": receipt.status,
                "succeeded": receipt.succeeded,
                "blockers": list(receipt.blockers),
            },
        )
        sequence += 1
        if not receipt.succeeded:
            _add_decision(
                decisions,
                kind="unitree_runtime_dispatch_gate",
                outcome="dispatch_denied",
                sequence_idx=len(decisions),
                source_event_ids=[event.event_id],
                receipt_refs=[receipt.receipt_id],
                reasons=receipt.blockers or [receipt.status],
                artifact_refs=artifact_refs,
            )
    event = _add_event(
        events,
        kind="unitree_trace_import_bundle_recorded",
        sequence_idx=sequence,
        receipt_refs=[receipt.receipt_id for receipt in trace_replay],
        artifact_refs=artifact_refs,
        metadata={
            "low_state_trace_count": len(low_state),
            "imu_trace_count": len(imu),
            "contact_trace_count": len(contacts),
            "wireless_estop_trace_count": len(wireless),
        },
    )
    sequence += 1
    _add_decision(
        decisions,
        kind="unitree_trace_replay_admission",
        outcome="shadow_replay_admitted",
        sequence_idx=len(decisions),
        source_event_ids=[event.event_id],
        receipt_refs=[receipt.receipt_id for receipt in trace_replay],
        reasons=["jsonl_trace_bundle_imported", "live_stream_unavailable"],
        artifact_refs=artifact_refs,
    )
    event = _add_event(
        events,
        kind="unitree_command_dry_run_recorded",
        sequence_idx=sequence,
        receipt_refs=[receipt.receipt_id for receipt in command_shapes],
        artifact_refs=artifact_refs,
        metadata={"command_shape_receipt_count": len(command_shapes)},
    )
    sequence += 1
    _add_decision(
        decisions,
        kind="unitree_command_publication_gate",
        outcome="no_publish_dry_run_only",
        sequence_idx=len(decisions),
        source_event_ids=[event.event_id],
        receipt_refs=[receipt.receipt_id for receipt in command_shapes],
        reasons=["no_ros2_publish", "no_unitree_sdk2_write", "shape_validation_only"],
        artifact_refs=artifact_refs,
    )
    event = _add_event(
        events,
        kind="unitree_timing_watchdog_recorded",
        sequence_idx=sequence,
        receipt_refs=[
            *[receipt.receipt_id for receipt in timing],
            *[receipt.receipt_id for receipt in watchdog],
            *[receipt.receipt_id for receipt in stale],
        ],
        artifact_refs=artifact_refs,
        metadata={
            "timing_receipt_count": len(timing),
            "watchdog_receipt_count": len(watchdog),
            "stale_validation_receipt_count": len(stale),
        },
    )
    sequence += 1
    _add_decision(
        decisions,
        kind="unitree_watchdog_dispatch_gate",
        outcome="dispatch_denied",
        sequence_idx=len(decisions),
        source_event_ids=[event.event_id],
        receipt_refs=[receipt.receipt_id for receipt in watchdog],
        reasons=["stale_stream_demotion", "dds_network_timing_unavailable"],
        artifact_refs=artifact_refs,
    )
    _add_event(
        events,
        kind="unitree_safety_recovery_recorded",
        sequence_idx=sequence,
        receipt_refs=[
            *[transition.transition_id for transition in safety_transitions],
            *[receipt.receipt_id for receipt in safety_drills],
            *[receipt.receipt_id for receipt in safety_envelopes],
            *[receipt.receipt_id for receipt in operator_drills],
        ],
        artifact_refs=artifact_refs,
        metadata={
            "safety_transition_count": len(safety_transitions),
            "safety_drill_count": len(safety_drills),
            "safety_envelope_count": len(safety_envelopes),
            "operator_drill_count": len(operator_drills),
        },
    )
    sequence += 1
    _add_event(
        events,
        kind="unitree_cpu_mujoco_probe_recorded",
        sequence_idx=sequence,
        receipt_refs=[
            *[receipt.receipt_id for receipt in mujoco_steps],
            *[receipt.receipt_id for receipt in mujoco_model_stress],
        ],
        artifact_refs=artifact_refs,
        metadata={
            "mujoco_headless_trace_row_count": len(mujoco_rows),
            "mujoco_model_stress_receipt_count": len(mujoco_model_stress),
        },
    )
    sequence += 1
    _add_event(
        events,
        kind="unitree_runtime_blocker_probe_recorded",
        sequence_idx=sequence,
        receipt_refs=[
            *[receipt.receipt_id for receipt in ros2_runtime],
            *[receipt.receipt_id for receipt in trace_adapters],
            *[receipt.receipt_id for receipt in blocker_receipts],
        ],
        artifact_refs=artifact_refs,
        metadata={
            "ros2_runtime_receipt_count": len(ros2_runtime),
            "trace_adapter_receipt_count": len(trace_adapters),
            "blocker_probe_receipt_count": len(blocker_receipts),
        },
    )

    event_ids = [event.event_id for event in events]
    decision_ids = [decision.decision_id for decision in decisions]
    episodes, steps, windows = _episode_replay_rows(
        low_state=low_state,
        imu=imu,
        wireless=wireless,
        contacts=contacts,
        event_ids=event_ids,
        decision_ids=decision_ids,
        artifact_refs=artifact_refs,
    )
    replay_episode_ids = [episode.episode_id for episode in episodes]
    join_specs: list[tuple[str, int, list[str]]] = [
        ("build_message_validation", len(validation_receipts), []),
        ("trace_import", len(trace_replay) + len(trace_adapters), []),
        ("command_dry_run", len(command_shapes), []),
        ("timing_watchdog", len(timing) + len(watchdog) + len(stale), []),
        (
            "safety_recovery",
            len(safety_transitions)
            + len(safety_drills)
            + len(safety_envelopes)
            + len(operator_drills),
            [],
        ),
        ("cpu_mujoco_probe", len(mujoco_steps) + len(mujoco_model_stress), []),
    ]
    join_rows = [
        UnitreeEventReplayJoinRow(
            join_id=stable_id(
                "unitree_event_replay_join",
                {"join_key": key, "receipt_count": count, "events": event_ids},
            ),
            join_key=key,
            source_receipt_count=count,
            event_ids=event_ids,
            decision_ids=decision_ids,
            replay_episode_ids=replay_episode_ids,
            replay_step_count=len(steps),
            join_status="ok_shadow_join" if count and steps else "blocked_no_rows",
            blockers=blockers
            if count and steps
            else ["missing_source_receipts_or_replay"],
            artifact_refs=mapping(artifact_refs),
        )
        for key, count, blockers in join_specs
    ]

    embodiment_receipts = (
        len(low_state)
        + len(imu)
        + len(contacts)
        + len(command_shapes)
        + len(safety_drills)
        + len(safety_envelopes)
    )
    sim_receipts = len(mujoco_steps) + len(mujoco_rows) + len(mujoco_model_stress)
    economic_receipts = len(events) + len(decisions) + len(join_rows)
    event_ref = str(artifact_refs.get("event_spine_path", ""))
    replay_ref = str(artifact_refs.get("replay_steps_path", ""))
    ingestion_specs: list[tuple[str, str, str, int, int, list[str], bool]] = [
        (
            "embodiment_actuation",
            "unitree_phase4_trace_command_safety_receipts",
            "ok_shadow_lower_wm_ingestion",
            embodiment_receipts,
            len(low_state) + len(imu) + len(contacts),
            ["live_stream_or_hardware_evidence_missing"],
            True,
        ),
        (
            "sim_synth_physics",
            "unitree_cpu_mujoco_no_policy_probe_receipts",
            "ok_cpu_no_policy_probe_ingestion"
            if sim_receipts
            else "blocked_no_mujoco_probe",
            sim_receipts,
            len(mujoco_rows),
            ["ros2_bridge_policy_control_or_hardware_trace_missing"],
            bool(sim_receipts),
        ),
        (
            "perception_grounding",
            "unitree_visual_stream_absence_receipt",
            "blocked_no_visual_or_calibrated_scene_stream",
            0,
            0,
            ["real_or_sim_visual_stream_missing", "camera_calibration_missing"],
            True,
        ),
        (
            "economic_world_model",
            "unitree_event_replay_lower_wm_shadow_join",
            "ok_shadow_economic_ingestion",
            economic_receipts,
            len(steps),
            ["promotion_grade_outcome_receipts_missing"],
            True,
        ),
    ]
    ingestion_rows = [
        UnitreeLowerWMIngestionRow(
            ingestion_row_id=stable_id(
                "unitree_lower_wm_ingestion",
                {
                    "wm_key": wm_key,
                    "canonical_surface": surface,
                    "receipt_count": receipt_count,
                },
            ),
            wm_key=wm_key,
            canonical_surface=surface,
            ingestion_status=status,
            source_receipt_count=receipt_count,
            source_trace_count=trace_count,
            event_spine_ref=event_ref,
            replay_ref=replay_ref,
            source_artifact_refs=mapping(artifact_refs),
            blockers=blockers,
            ready_for_economic_shadow=ready,
            ready_for_training=False,
            promotion_eligible=False,
            metadata={
                "boundary": "shadow ingestion only; no GPU/provider/hardware proof",
            },
        )
        for (
            wm_key,
            surface,
            status,
            receipt_count,
            trace_count,
            blockers,
            ready,
        ) in ingestion_specs
    ]
    return events, decisions, episodes, steps, windows, join_rows, ingestion_rows


def build_cpu_august_gap_execution_report(
    *,
    validation_receipts: Sequence[UnitreeRos2Sdk2BuildMessageValidationReceipt],
    phase4_unitree_local_harness_report: Mapping[str, Any],
    phase4_unitree_runtime_bridge_report: Mapping[str, Any],
    phase4_unitree_blocker_stress_probe_report: Mapping[str, Any],
    events: Sequence[RuntimeEvent],
    decisions: Sequence[DecisionLedgerEntry],
    episodes: Sequence[ReplayEpisodeRecord],
    steps: Sequence[ReplayStepRecord],
    windows: Sequence[ReplayWindowRecord],
    join_rows: Sequence[UnitreeEventReplayJoinRow],
    ingestion_rows: Sequence[UnitreeLowerWMIngestionRow],
    artifact_refs: Mapping[str, Any],
) -> CpuAugustGapExecutionReport:
    by_key = _validation_receipts_by_key(validation_receipts)
    static_messages_ok = by_key.get(
        "ros2_static_message_definition_validation"
    ) and by_key["ros2_static_message_definition_validation"].succeeded
    build_validation_receipted = all(
        key in by_key
        for key in (
            "ros2_static_message_definition_validation",
            "ros2_generated_message_import_validation",
            "ros2_colcon_build_validation",
            "unitree_sdk2_header_compile_validation",
            "unitree_sdk2_cmake_build_validation",
        )
    )
    ros2_sdk2_complete = bool(static_messages_ok and build_validation_receipted)
    trace_import_complete = bool(
        phase4_unitree_local_harness_report.get("trace_stream_harness_complete")
        and phase4_unitree_runtime_bridge_report.get("trace_ingestion_adapters_complete")
    )
    command_dry_run_complete = bool(
        phase4_unitree_local_harness_report.get("command_shape_harness_complete")
    )
    timing_watchdog_complete = bool(
        phase4_unitree_local_harness_report.get("mock_timing_watchdog_harness_complete")
    )
    safety_recovery_complete = bool(
        phase4_unitree_local_harness_report.get("safety_recovery_harness_complete")
        and phase4_unitree_runtime_bridge_report.get("safety_envelope_expansion_complete")
        and phase4_unitree_runtime_bridge_report.get("operator_drill_runner_complete")
    )
    cpu_mujoco_probe_complete = bool(
        phase4_unitree_runtime_bridge_report.get("mujoco_headless_trace_attempt_complete")
        and phase4_unitree_blocker_stress_probe_report.get(
            "all_local_probe_attempts_complete"
        )
    )
    event_replay_complete = bool(events and steps and join_rows) and all(
        row.join_status == "ok_shadow_join" for row in join_rows
    )
    lower_wm_complete = bool(ingestion_rows) and all(
        row.ready_for_economic_shadow and not row.promotion_eligible
        for row in ingestion_rows
    )
    complete = all(
        (
            ros2_sdk2_complete,
            trace_import_complete,
            command_dry_run_complete,
            timing_watchdog_complete,
            safety_recovery_complete,
            cpu_mujoco_probe_complete,
            event_replay_complete,
            lower_wm_complete,
        )
    )
    ros2_build = by_key.get("ros2_colcon_build_validation")
    generated_import = by_key.get("ros2_generated_message_import_validation")
    sdk2_header = by_key.get("unitree_sdk2_header_compile_validation")
    sdk2_cmake = by_key.get("unitree_sdk2_cmake_build_validation")
    blocker_set = {
        "ros2_or_colcon_runtime_missing_for_generated_build_import"
        if not (ros2_build and ros2_build.succeeded)
        else "",
        "unitree_sdk2_linux_build_or_header_compile_missing"
        if not (sdk2_header and sdk2_header.succeeded)
        else "",
        "rosbag2_or_mcap_real_stream_import_missing",
        "live_lowstate_imu_contact_wireless_estop_streams_missing",
        "ros2_sdk2_g1pilot_command_echo_missing",
        "dds_network_or_on_robot_timing_missing",
        "physical_stop_distance_and_calibrated_safety_limits_missing",
        "operator_teleop_runtime_drill_missing",
        "policy_controlled_mujoco_or_hardware_trace_missing",
        "gpu_provider_training_and_promotion_benchmarks_missing",
    }
    remaining = sorted(item for item in blocker_set if item)
    payload = {
        "validation_receipt_count": len(validation_receipts),
        "event_count": len(events),
        "decision_count": len(decisions),
        "replay_step_count": len(steps),
        "ingestion_rows": len(ingestion_rows),
    }
    return CpuAugustGapExecutionReport(
        report_id=stable_id("cpu_august_gap_execution", payload),
        status="ok" if complete else "blocked",
        ros2_sdk2_build_message_validation_complete=ros2_sdk2_complete,
        trace_import_complete=trace_import_complete,
        command_dry_run_complete=command_dry_run_complete,
        timing_watchdog_complete=timing_watchdog_complete,
        safety_recovery_complete=safety_recovery_complete,
        cpu_mujoco_probe_complete=cpu_mujoco_probe_complete,
        event_spine_replay_joins_complete=event_replay_complete,
        lower_wm_ingestion_complete=lower_wm_complete,
        cpu_august_gap_tranche_complete=complete,
        validation_receipt_count=len(validation_receipts),
        event_count=len(events),
        decision_count=len(decisions),
        replay_episode_count=len(episodes),
        replay_step_count=len(steps),
        replay_window_count=len(windows),
        event_replay_join_row_count=len(join_rows),
        lower_wm_ingestion_row_count=len(ingestion_rows),
        ros2_build_attempted=bool(ros2_build and ros2_build.build_attempted),
        ros2_build_succeeded=bool(ros2_build and ros2_build.succeeded),
        generated_message_import_succeeded=bool(
            generated_import and generated_import.succeeded
        ),
        sdk2_header_compile_succeeded=bool(sdk2_header and sdk2_header.succeeded),
        sdk2_cmake_build_attempted=bool(sdk2_cmake and sdk2_cmake.build_attempted),
        sdk2_cmake_build_succeeded=bool(sdk2_cmake and sdk2_cmake.succeeded),
        minimal_mujoco_headless_step_executed=bool(
            phase4_unitree_runtime_bridge_report.get(
                "minimal_mujoco_headless_step_executed", False
            )
        ),
        g1_mujoco_model_stress_succeeded=bool(
            phase4_unitree_blocker_stress_probe_report.get(
                "g1_mujoco_model_stress_succeeded", False
            )
        ),
        denied_gates=_denied_gates(),
        remaining_evidence_blockers=remaining,
        artifact_refs=mapping(artifact_refs),
    )


def save_cpu_august_gap_execution(
    output_dir: str | Path,
    *,
    report: CpuAugustGapExecutionReport,
    validation_receipts: Sequence[UnitreeRos2Sdk2BuildMessageValidationReceipt],
    events: Sequence[RuntimeEvent],
    decisions: Sequence[DecisionLedgerEntry],
    episodes: Sequence[ReplayEpisodeRecord],
    steps: Sequence[ReplayStepRecord],
    windows: Sequence[ReplayWindowRecord],
    join_rows: Sequence[UnitreeEventReplayJoinRow],
    ingestion_rows: Sequence[UnitreeLowerWMIngestionRow],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "cpu_august_gap_execution_report_v1.json",
        "validation_receipts_path": output
        / "unitree_ros2_sdk2_build_message_validation_receipts_v1.jsonl",
        "event_spine_path": output / "event_spine.json",
        "decision_ledger_path": output / "decision_ledger.json",
        "replay_episodes_path": output / "unitree_replay_episodes_v1.jsonl",
        "replay_steps_path": output / "unitree_replay_steps_v1.jsonl",
        "replay_windows_path": output / "unitree_replay_windows_v1.jsonl",
        "event_replay_join_rows_path": output
        / "unitree_event_replay_join_rows_v1.jsonl",
        "lower_wm_ingestion_rows_path": output
        / "unitree_lower_wm_ingestion_rows_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(
        paths["validation_receipts_path"],
        [receipt.to_dict() for receipt in validation_receipts],
    )
    write_json(
        paths["event_spine_path"],
        event_spine_sidecar_payload(
            run_id="cpu_august_gap_unitree",
            events=events,
        ),
    )
    write_json(
        paths["decision_ledger_path"],
        decision_ledger_sidecar_payload(
            run_id="cpu_august_gap_unitree",
            decisions=decisions,
        ),
    )
    write_jsonl(paths["replay_episodes_path"], [row.to_dict() for row in episodes])
    write_jsonl(paths["replay_steps_path"], [row.to_dict() for row in steps])
    write_jsonl(paths["replay_windows_path"], [row.to_dict() for row in windows])
    write_jsonl(
        paths["event_replay_join_rows_path"],
        [row.to_dict() for row in join_rows],
    )
    write_jsonl(
        paths["lower_wm_ingestion_rows_path"],
        [row.to_dict() for row in ingestion_rows],
    )
    return {key: str(path) for key, path in paths.items()}


def load_cpu_august_gap_execution_report(
    path: str | Path,
) -> CpuAugustGapExecutionReport:
    return CpuAugustGapExecutionReport.from_dict(_load_json(path))


def load_unitree_ros2_sdk2_build_message_validation_receipts(
    path: str | Path,
) -> list[UnitreeRos2Sdk2BuildMessageValidationReceipt]:
    return [
        UnitreeRos2Sdk2BuildMessageValidationReceipt.from_dict(row)
        for row in _load_jsonl(path)
    ]


def load_unitree_event_replay_join_rows(
    path: str | Path,
) -> list[UnitreeEventReplayJoinRow]:
    return [UnitreeEventReplayJoinRow.from_dict(row) for row in _load_jsonl(path)]


def load_unitree_lower_wm_ingestion_rows(
    path: str | Path,
) -> list[UnitreeLowerWMIngestionRow]:
    return [UnitreeLowerWMIngestionRow.from_dict(row) for row in _load_jsonl(path)]


__all__ = [
    "CPU_AUGUST_GAP_EXECUTION_REPORT_VERSION",
    "DENIED_CPU_AUGUST_GAP_AUTHORITIES",
    "UNITREE_EVENT_REPLAY_JOIN_ROW_VERSION",
    "UNITREE_LOWER_WM_INGESTION_ROW_VERSION",
    "UNITREE_ROS2_SDK2_BUILD_MESSAGE_VALIDATION_RECEIPT_VERSION",
    "CpuAugustGapExecutionReport",
    "UnitreeEventReplayJoinRow",
    "UnitreeLowerWMIngestionRow",
    "UnitreeRos2Sdk2BuildMessageValidationReceipt",
    "build_cpu_august_gap_event_replay_lower_wm_surfaces",
    "build_cpu_august_gap_execution_report",
    "build_unitree_ros2_sdk2_build_message_validation_receipts",
    "load_cpu_august_gap_execution_report",
    "load_unitree_event_replay_join_rows",
    "load_unitree_lower_wm_ingestion_rows",
    "load_unitree_ros2_sdk2_build_message_validation_receipts",
    "save_cpu_august_gap_execution",
]
