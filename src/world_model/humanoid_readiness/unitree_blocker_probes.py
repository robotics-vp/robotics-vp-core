"""Phase 4 Unitree blocker stress probes.

These probes press on the remaining Unitree/G1 Phase 4 blockers without
crossing into live control. They inspect local runtime roots, static package
surfaces, imports, headers, and no-policy MuJoCo model stepping. Successful
checks become receipts; blocked checks stay explicit blockers.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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

PHASE4_UNITREE_BLOCKER_STRESS_REPORT_VERSION = (
    "phase4_unitree_blocker_stress_probe_report_v1"
)
UNITREE_BLOCKER_STRESS_PROBE_RECEIPT_VERSION = "unitree_blocker_stress_probe_receipt_v1"
UNITREE_MUJOCO_MODEL_STRESS_RECEIPT_VERSION = "unitree_mujoco_model_stress_receipt_v1"

DENIED_UNITREE_BLOCKER_STRESS_AUTHORITIES = (
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
    "unitree_sim_runtime_claimed",
    "policy_controlled_mujoco_claimed",
)

BLOCKER_PROBE_KEYS = (
    "host_ros2_colcon_toolchain",
    "python_runtime_imports",
    "unitree_ros2_static_message_surface",
    "g1pilot_static_launch_surface",
    "g1pilot_runtime_dependency_surface",
    "cyclonedds_header_compile",
    "unitree_sdk2_header_compile",
    "unitree_mujoco_g1_model_stress",
    "unitree_rl_gym_policy_asset_visibility",
    "unitree_isaaclab_static_task_surface",
    "unitree_lerobot_static_adapter_surface",
    "rosbag2_mcap_import_modules",
    "physical_calibration_sidecar",
    "operator_teleop_runtime_surface",
)


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_UNITREE_BLOCKER_STRESS_AUTHORITIES}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl(path, rows)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class UnitreeBlockerStressProbeReceipt:
    receipt_id: str
    blocker_key: str
    probe_key: str
    status: str
    succeeded: bool
    local_probe_executed: bool
    evidence_class: str
    target_path: str = ""
    command_or_import: str = ""
    observed: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    follow_up_work_unlocked: list[str] = field(default_factory=list)
    external_requirement: str = ""
    build_executed: bool = False
    runtime_invoked: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    g1pilot_runtime_invoked: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    training_executed: bool = False
    weights_written: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    authority_class: str = "unitree_blocker_stress_probe_no_live_authority"
    version: str = UNITREE_BLOCKER_STRESS_PROBE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "blocker_key": self.blocker_key,
            "probe_key": self.probe_key,
            "status": self.status,
            "succeeded": bool(self.succeeded),
            "local_probe_executed": bool(self.local_probe_executed),
            "evidence_class": self.evidence_class,
            "target_path": self.target_path,
            "command_or_import": self.command_or_import,
            "observed": mapping(self.observed),
            "missing": strings(self.missing),
            "blockers": strings(self.blockers),
            "follow_up_work_unlocked": strings(self.follow_up_work_unlocked),
            "external_requirement": self.external_requirement,
            "build_executed": bool(self.build_executed),
            "runtime_invoked": bool(self.runtime_invoked),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "g1pilot_runtime_invoked": bool(self.g1pilot_runtime_invoked),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnitreeBlockerStressProbeReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            blocker_key=str(payload.get("blocker_key", "")),
            probe_key=str(payload.get("probe_key", "")),
            status=str(payload.get("status", "blocked")),
            succeeded=bool(payload.get("succeeded", False)),
            local_probe_executed=bool(payload.get("local_probe_executed", False)),
            evidence_class=str(payload.get("evidence_class", "")),
            target_path=str(payload.get("target_path", "")),
            command_or_import=str(payload.get("command_or_import", "")),
            observed=mapping(payload.get("observed")),
            missing=strings(payload.get("missing")),
            blockers=strings(payload.get("blockers")),
            follow_up_work_unlocked=strings(payload.get("follow_up_work_unlocked")),
            external_requirement=str(payload.get("external_requirement", "")),
            build_executed=bool(payload.get("build_executed", False)),
            runtime_invoked=bool(payload.get("runtime_invoked", False)),
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
            authority_class=str(
                payload.get(
                    "authority_class",
                    "unitree_blocker_stress_probe_no_live_authority",
                )
            ),
            version=str(
                payload.get("version", UNITREE_BLOCKER_STRESS_PROBE_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class UnitreeMujocoModelStressReceipt:
    receipt_id: str
    model_key: str
    xml_path: str
    status: str
    mujoco_module_available: bool
    model_loaded: bool
    step_attempted: bool
    step_executed: bool
    stress_step_count: int
    final_time_s: float
    wall_time_s: float
    nq: int = 0
    nv: int = 0
    nu: int = 0
    njnt: int = 0
    nsensor: int = 0
    nbody: int = 0
    ngeom: int = 0
    timestep_s: float = 0.0
    qvel_norm: float = 0.0
    joint_name_head: list[str] = field(default_factory=list)
    actuator_name_head: list[str] = field(default_factory=list)
    sensor_name_head: list[str] = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""
    policy_controlled: bool = False
    ros2_bridge_active: bool = False
    hardware_executed: bool = False
    authority_class: str = "mujoco_model_stress_no_policy_no_bridge"
    version: str = UNITREE_MUJOCO_MODEL_STRESS_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "model_key": self.model_key,
            "xml_path": self.xml_path,
            "status": self.status,
            "mujoco_module_available": bool(self.mujoco_module_available),
            "model_loaded": bool(self.model_loaded),
            "step_attempted": bool(self.step_attempted),
            "step_executed": bool(self.step_executed),
            "stress_step_count": int(self.stress_step_count),
            "final_time_s": float(self.final_time_s),
            "wall_time_s": float(self.wall_time_s),
            "nq": int(self.nq),
            "nv": int(self.nv),
            "nu": int(self.nu),
            "njnt": int(self.njnt),
            "nsensor": int(self.nsensor),
            "nbody": int(self.nbody),
            "ngeom": int(self.ngeom),
            "timestep_s": float(self.timestep_s),
            "qvel_norm": float(self.qvel_norm),
            "joint_name_head": strings(self.joint_name_head),
            "actuator_name_head": strings(self.actuator_name_head),
            "sensor_name_head": strings(self.sensor_name_head),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "policy_controlled": bool(self.policy_controlled),
            "ros2_bridge_active": bool(self.ros2_bridge_active),
            "hardware_executed": bool(self.hardware_executed),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitreeMujocoModelStressReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            model_key=str(payload.get("model_key", "")),
            xml_path=str(payload.get("xml_path", "")),
            status=str(payload.get("status", "blocked")),
            mujoco_module_available=bool(payload.get("mujoco_module_available", False)),
            model_loaded=bool(payload.get("model_loaded", False)),
            step_attempted=bool(payload.get("step_attempted", False)),
            step_executed=bool(payload.get("step_executed", False)),
            stress_step_count=int(payload.get("stress_step_count", 0) or 0),
            final_time_s=_safe_float(payload.get("final_time_s")),
            wall_time_s=_safe_float(payload.get("wall_time_s")),
            nq=_safe_int(payload.get("nq")),
            nv=_safe_int(payload.get("nv")),
            nu=_safe_int(payload.get("nu")),
            njnt=_safe_int(payload.get("njnt")),
            nsensor=_safe_int(payload.get("nsensor")),
            nbody=_safe_int(payload.get("nbody")),
            ngeom=_safe_int(payload.get("ngeom")),
            timestep_s=_safe_float(payload.get("timestep_s")),
            qvel_norm=_safe_float(payload.get("qvel_norm")),
            joint_name_head=strings(payload.get("joint_name_head")),
            actuator_name_head=strings(payload.get("actuator_name_head")),
            sensor_name_head=strings(payload.get("sensor_name_head")),
            error_type=str(payload.get("error_type", "")),
            error_message=str(payload.get("error_message", "")),
            policy_controlled=bool(payload.get("policy_controlled", False)),
            ros2_bridge_active=bool(payload.get("ros2_bridge_active", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "mujoco_model_stress_no_policy_no_bridge"
                )
            ),
            version=str(
                payload.get("version", UNITREE_MUJOCO_MODEL_STRESS_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase4UnitreeBlockerStressProbeReport:
    report_id: str
    status: str
    probe_receipt_count: int
    succeeded_probe_count: int
    blocked_probe_count: int
    mujoco_model_stress_receipt_count: int
    mujoco_model_stress_success_count: int
    all_local_probe_attempts_complete: bool
    local_phase4_probe_expansion_complete: bool
    g1_mujoco_model_stress_succeeded: bool
    g1pilot_static_surface_succeeded: bool
    cyclonedds_header_compile_succeeded: bool
    unitree_sdk2_header_compile_succeeded: bool
    ros2_runtime_available: bool
    trace_import_modules_available: bool
    policy_checkpoint_visible: bool
    isaaclab_task_surface_visible: bool
    lerobot_adapter_surface_visible: bool
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
    unlocked_local_followups: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE4_UNITREE_BLOCKER_STRESS_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "probe_receipt_count": int(self.probe_receipt_count),
            "succeeded_probe_count": int(self.succeeded_probe_count),
            "blocked_probe_count": int(self.blocked_probe_count),
            "mujoco_model_stress_receipt_count": int(
                self.mujoco_model_stress_receipt_count
            ),
            "mujoco_model_stress_success_count": int(
                self.mujoco_model_stress_success_count
            ),
            "all_local_probe_attempts_complete": bool(
                self.all_local_probe_attempts_complete
            ),
            "local_phase4_probe_expansion_complete": bool(
                self.local_phase4_probe_expansion_complete
            ),
            "g1_mujoco_model_stress_succeeded": bool(
                self.g1_mujoco_model_stress_succeeded
            ),
            "g1pilot_static_surface_succeeded": bool(
                self.g1pilot_static_surface_succeeded
            ),
            "cyclonedds_header_compile_succeeded": bool(
                self.cyclonedds_header_compile_succeeded
            ),
            "unitree_sdk2_header_compile_succeeded": bool(
                self.unitree_sdk2_header_compile_succeeded
            ),
            "ros2_runtime_available": bool(self.ros2_runtime_available),
            "trace_import_modules_available": bool(self.trace_import_modules_available),
            "policy_checkpoint_visible": bool(self.policy_checkpoint_visible),
            "isaaclab_task_surface_visible": bool(self.isaaclab_task_surface_visible),
            "lerobot_adapter_surface_visible": bool(
                self.lerobot_adapter_surface_visible
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
            "unlocked_local_followups": strings(self.unlocked_local_followups),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase4UnitreeBlockerStressProbeReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "blocked")),
            probe_receipt_count=int(payload.get("probe_receipt_count", 0) or 0),
            succeeded_probe_count=int(payload.get("succeeded_probe_count", 0) or 0),
            blocked_probe_count=int(payload.get("blocked_probe_count", 0) or 0),
            mujoco_model_stress_receipt_count=int(
                payload.get("mujoco_model_stress_receipt_count", 0) or 0
            ),
            mujoco_model_stress_success_count=int(
                payload.get("mujoco_model_stress_success_count", 0) or 0
            ),
            all_local_probe_attempts_complete=bool(
                payload.get("all_local_probe_attempts_complete", False)
            ),
            local_phase4_probe_expansion_complete=bool(
                payload.get("local_phase4_probe_expansion_complete", False)
            ),
            g1_mujoco_model_stress_succeeded=bool(
                payload.get("g1_mujoco_model_stress_succeeded", False)
            ),
            g1pilot_static_surface_succeeded=bool(
                payload.get("g1pilot_static_surface_succeeded", False)
            ),
            cyclonedds_header_compile_succeeded=bool(
                payload.get("cyclonedds_header_compile_succeeded", False)
            ),
            unitree_sdk2_header_compile_succeeded=bool(
                payload.get("unitree_sdk2_header_compile_succeeded", False)
            ),
            ros2_runtime_available=bool(payload.get("ros2_runtime_available", False)),
            trace_import_modules_available=bool(
                payload.get("trace_import_modules_available", False)
            ),
            policy_checkpoint_visible=bool(
                payload.get("policy_checkpoint_visible", False)
            ),
            isaaclab_task_surface_visible=bool(
                payload.get("isaaclab_task_surface_visible", False)
            ),
            lerobot_adapter_surface_visible=bool(
                payload.get("lerobot_adapter_surface_visible", False)
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
            unlocked_local_followups=strings(payload.get("unlocked_local_followups")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE4_UNITREE_BLOCKER_STRESS_REPORT_VERSION)
            ),
        )


def _receipt(
    *,
    blocker_key: str,
    probe_key: str,
    status: str,
    succeeded: bool,
    local_probe_executed: bool,
    evidence_class: str,
    target_path: str = "",
    command_or_import: str = "",
    observed: Mapping[str, Any] | None = None,
    missing: list[str] | None = None,
    blockers: list[str] | None = None,
    follow_up_work_unlocked: list[str] | None = None,
    external_requirement: str = "",
) -> UnitreeBlockerStressProbeReceipt:
    payload = {
        "blocker_key": blocker_key,
        "probe_key": probe_key,
        "status": status,
        "target_path": target_path,
        "observed": mapping(observed),
    }
    return UnitreeBlockerStressProbeReceipt(
        receipt_id=stable_id("unitree_blocker_stress_probe", payload),
        blocker_key=blocker_key,
        probe_key=probe_key,
        status=status,
        succeeded=succeeded,
        local_probe_executed=local_probe_executed,
        evidence_class=evidence_class,
        target_path=target_path,
        command_or_import=command_or_import,
        observed=mapping(observed),
        missing=strings(missing),
        blockers=strings(blockers),
        follow_up_work_unlocked=strings(follow_up_work_unlocked),
        external_requirement=external_requirement,
    )


def _compile_probe(
    *,
    source: str,
    include_dirs: list[Path],
    output_name: str,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    compiler = shutil.which("c++")
    if not compiler:
        return {
            "attempted": False,
            "returncode": None,
            "stdout": "",
            "stderr": "c++ compiler not available",
        }
    with tempfile.TemporaryDirectory(prefix="unitree_probe_") as tmp:
        obj = Path(tmp) / output_name
        command = [
            compiler,
            "-std=c++17",
            *[item for inc in include_dirs for item in ("-I", str(inc))],
            "-x",
            "c++",
            "-c",
            "-o",
            str(obj),
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                input=source,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "attempted": True,
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "compile timed out",
            }
    return {
        "attempted": True,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }


def _g1_mujoco_xmls(root: Path) -> list[Path]:
    g1 = root / "unitree_robots/g1"
    return [
        g1 / "scene_29dof.xml",
        g1 / "g1_29dof.xml",
        g1 / "scene_23dof.xml",
        g1 / "g1_23dof.xml",
        g1 / "scene.xml",
    ]


def build_mujoco_model_stress_receipts(
    *,
    local_roots: Mapping[str, str | Path] | None = None,
    stress_steps: int = 100,
) -> list[UnitreeMujocoModelStressReceipt]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    mujoco_root = Path(roots["unitree_mujoco"])
    module_available = _module_available("mujoco")
    steps = max(1, int(stress_steps))
    receipts: list[UnitreeMujocoModelStressReceipt] = []
    for xml in _g1_mujoco_xmls(mujoco_root):
        payload = {"xml_path": str(xml), "steps": steps}
        receipt_id = stable_id("unitree_mujoco_model_stress", payload)
        if not module_available:
            receipts.append(
                UnitreeMujocoModelStressReceipt(
                    receipt_id=receipt_id,
                    model_key=xml.name,
                    xml_path=str(xml),
                    status="blocked_missing_mujoco_python_module",
                    mujoco_module_available=False,
                    model_loaded=False,
                    step_attempted=False,
                    step_executed=False,
                    stress_step_count=0,
                    final_time_s=0.0,
                    wall_time_s=0.0,
                    error_type="ModuleNotFoundError",
                    error_message="Python module 'mujoco' is not importable.",
                )
            )
            continue
        try:
            import mujoco  # type: ignore[import-not-found,import-untyped]

            started = time.perf_counter()
            model = mujoco.MjModel.from_xml_path(str(xml))
            data = mujoco.MjData(model)
            for _ in range(steps):
                mujoco.mj_step(model, data)
            elapsed = time.perf_counter() - started
            joint_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) or ""
                for idx in range(model.njnt)
            ]
            actuator_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or ""
                for idx in range(model.nu)
            ]
            sensor_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, idx) or ""
                for idx in range(model.nsensor)
            ]
            qvel_norm = float((data.qvel * data.qvel).sum()) ** 0.5
            receipts.append(
                UnitreeMujocoModelStressReceipt(
                    receipt_id=receipt_id,
                    model_key=xml.name,
                    xml_path=str(xml),
                    status="ok",
                    mujoco_module_available=True,
                    model_loaded=True,
                    step_attempted=True,
                    step_executed=True,
                    stress_step_count=steps,
                    final_time_s=float(data.time),
                    wall_time_s=elapsed,
                    nq=int(model.nq),
                    nv=int(model.nv),
                    nu=int(model.nu),
                    njnt=int(model.njnt),
                    nsensor=int(model.nsensor),
                    nbody=int(model.nbody),
                    ngeom=int(model.ngeom),
                    timestep_s=float(model.opt.timestep),
                    qvel_norm=qvel_norm,
                    joint_name_head=joint_names[:10],
                    actuator_name_head=actuator_names[:10],
                    sensor_name_head=sensor_names[:10],
                )
            )
        except Exception as exc:
            receipts.append(
                UnitreeMujocoModelStressReceipt(
                    receipt_id=receipt_id,
                    model_key=xml.name,
                    xml_path=str(xml),
                    status="blocked_mujoco_model_stress_failed",
                    mujoco_module_available=True,
                    model_loaded=False,
                    step_attempted=True,
                    step_executed=False,
                    stress_step_count=0,
                    final_time_s=0.0,
                    wall_time_s=0.0,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
            )
    return receipts


def _parse_python_surfaces(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("g1pilot/**/*.py")) + sorted(root.glob("launch/*.py"))
    setup = root / "setup.py"
    if setup.exists():
        files.append(setup)
    syntax_errors: list[str] = []
    import_roots: set[str] = set()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            syntax_errors.append(f"{path.relative_to(root)}:{type(exc).__name__}:{exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_roots.add(node.module.split(".")[0])
    stdlib_or_local = {
        "__future__",
        "array",
        "copy",
        "enum",
        "functools",
        "glob",
        "g1pilot",
        "heapq",
        "json",
        "launch",
        "logging",
        "math",
        "os",
        "setuptools",
        "subprocess",
        "sys",
        "threading",
        "time",
        "typing",
    }
    external = sorted(import_roots - stdlib_or_local)
    return {
        "python_file_count": len(files),
        "syntax_error_count": len(syntax_errors),
        "syntax_errors": syntax_errors,
        "external_import_roots": external,
        "launch_file_count": len(list(root.glob("launch/*.launch.py"))),
        "teleop_launch_present": (
            root / "launch/teleoperation_launcher.launch.py"
        ).exists(),
        "bringup_launch_present": (root / "launch/bringup_launcher.launch.py").exists(),
        "setup_py_present": setup.exists(),
    }


def build_blocker_stress_probe_receipts(
    *,
    local_roots: Mapping[str, str | Path] | None = None,
) -> list[UnitreeBlockerStressProbeReceipt]:
    roots = {**default_unitree_local_roots(), **dict(local_roots or {})}
    ros2_root = Path(roots["unitree_ros2"])
    g1pilot_root = Path(roots["g1pilot"])
    sdk2_root = Path(roots["unitree_sdk2"])
    rl_gym_root = Path(roots["unitree_rl_gym"])
    isaaclab_root = Path(roots["unitree_sim_isaaclab"])
    lerobot_root = Path(roots["unitree_il_lerobot"])
    receipts: list[UnitreeBlockerStressProbeReceipt] = []

    required_tools = ["cmake", "colcon", "ros2", "docker"]
    tool_status = {tool: shutil.which(tool) is not None for tool in required_tools}
    missing_tools = [tool for tool, present in tool_status.items() if not present]
    opt_ros_present = Path("/opt/ros").exists()
    receipts.append(
        _receipt(
            blocker_key="ros2_colcon_build_and_generated_message_import_not_executed",
            probe_key="host_ros2_colcon_toolchain",
            status="ok"
            if not missing_tools and opt_ros_present
            else "blocked_missing_host_runtime_tools",
            succeeded=not missing_tools and opt_ros_present,
            local_probe_executed=True,
            evidence_class="host_toolchain_probe",
            target_path="/opt/ros",
            command_or_import="command -v cmake colcon ros2 docker",
            observed={"tool_status": tool_status, "opt_ros_present": opt_ros_present},
            missing=[*missing_tools, *(["/opt/ros"] if not opt_ros_present else [])],
            blockers=["ros2_runtime_not_installed"]
            if missing_tools or not opt_ros_present
            else [],
            external_requirement="Install/source ROS2 and colcon before generated message imports.",
        )
    )

    modules = [
        "rclpy",
        "rosbag2_py",
        "mcap",
        "cyclonedds",
        "mujoco",
        "unitree_hg",
        "unitree_go",
        "unitree_api",
        "unitree_sdk2py",
        "pinocchio",
        "hppfcl",
    ]
    module_status = {name: _module_available(name) for name in modules}
    missing_modules = [name for name, present in module_status.items() if not present]
    receipts.append(
        _receipt(
            blocker_key="runtime_python_imports_missing",
            probe_key="python_runtime_imports",
            status="partial_mujoco_only"
            if module_status.get("mujoco")
            else "blocked_missing_runtime_modules",
            succeeded=all(module_status.values()),
            local_probe_executed=True,
            evidence_class="python_import_probe",
            command_or_import="importlib.util.find_spec",
            observed={"module_status": module_status},
            missing=missing_modules,
            blockers=missing_modules,
            follow_up_work_unlocked=["mujoco_model_stress_receipts"]
            if module_status.get("mujoco")
            else [],
            external_requirement="Install ROS2/G1Pilot/runtime Python dependencies for live import checks.",
        )
    )

    workspace = ros2_root / "cyclonedds_ws/src"
    package_xml_count = len(list(workspace.glob("**/package.xml")))
    msg_count = len(list(workspace.glob("**/*.msg")))
    cmake_count = len(list(workspace.glob("**/CMakeLists.txt")))
    receipts.append(
        _receipt(
            blocker_key="ros2_colcon_build_and_generated_message_import_not_executed",
            probe_key="unitree_ros2_static_message_surface",
            status="ok_static_message_surface_visible"
            if package_xml_count >= 3 and msg_count >= 1
            else "blocked_static_message_surface_missing",
            succeeded=package_xml_count >= 3 and msg_count >= 1,
            local_probe_executed=True,
            evidence_class="static_source_surface_probe",
            target_path=str(workspace),
            observed={
                "package_xml_count": package_xml_count,
                "msg_definition_count": msg_count,
                "cmake_file_count": cmake_count,
                "unitree_hg_present": (workspace / "unitree/unitree_hg").exists(),
                "unitree_go_present": (workspace / "unitree/unitree_go").exists(),
                "unitree_api_present": (workspace / "unitree/unitree_api").exists(),
            },
            blockers=["generated_message_import_requires_ros2_colcon_build"],
            follow_up_work_unlocked=["generated_message_import_check_plan"],
            external_requirement="Run colcon build under ROS2, then import generated messages.",
        )
    )

    g1pilot_surface = (
        _parse_python_surfaces(g1pilot_root) if g1pilot_root.exists() else {}
    )
    g1pilot_static_ok = bool(g1pilot_surface) and not g1pilot_surface.get(
        "syntax_error_count", 1
    )
    receipts.append(
        _receipt(
            blocker_key="ros2_sdk2_g1pilot_command_echo_missing",
            probe_key="g1pilot_static_launch_surface",
            status="ok_static_launch_surface_visible"
            if g1pilot_static_ok
            else "blocked_static_g1pilot_surface_missing",
            succeeded=g1pilot_static_ok,
            local_probe_executed=True,
            evidence_class="g1pilot_static_parse_probe",
            target_path=str(g1pilot_root),
            observed=g1pilot_surface,
            blockers=[
                "g1pilot_runtime_dependencies_missing",
                "command_echo_not_executed",
            ],
            follow_up_work_unlocked=[
                "g1pilot_launch_surface_receipts",
                "teleop_runtime_dependency_receipts",
            ],
            external_requirement="Install ROS2/G1Pilot deps and run launch/import checks before command echo.",
        )
    )

    external_imports = strings(g1pilot_surface.get("external_import_roots"))
    runtime_missing = [
        name for name in external_imports if name and not _module_available(name)
    ]
    receipts.append(
        _receipt(
            blocker_key="operator_teleop_runtime_drill_missing",
            probe_key="g1pilot_runtime_dependency_surface",
            status="ok_runtime_imports_available"
            if external_imports and not runtime_missing
            else "blocked_missing_g1pilot_runtime_imports",
            succeeded=bool(external_imports) and not runtime_missing,
            local_probe_executed=True,
            evidence_class="runtime_dependency_probe_no_launch",
            target_path=str(g1pilot_root),
            command_or_import="importlib.util.find_spec(g1pilot external import roots)",
            observed={"external_import_roots": external_imports},
            missing=runtime_missing,
            blockers=runtime_missing,
            external_requirement="Install G1Pilot ROS2, Unitree, IK, GUI, and teleop dependencies.",
        )
    )

    dds_result = _compile_probe(
        source=(
            "#include <dds/dds.h>\n"
            "#include <dds/version.h>\n"
            "#ifndef DDS_VERSION\n#error DDS_VERSION missing\n#endif\n"
            "int main() { return DDS_VERSION_MAJOR; }\n"
        ),
        include_dirs=[sdk2_root / "thirdparty/include"],
        output_name="cyclonedds_header_probe.o",
    )
    dds_ok = dds_result.get("returncode") == 0
    receipts.append(
        _receipt(
            blocker_key="dds_network_or_on_robot_timing_missing",
            probe_key="cyclonedds_header_compile",
            status="ok_header_compile_only"
            if dds_ok
            else "blocked_header_compile_failed",
            succeeded=dds_ok,
            local_probe_executed=bool(dds_result.get("attempted")),
            evidence_class="compile_only_header_probe_no_network",
            target_path=str(sdk2_root / "thirdparty/include"),
            command_or_import="c++ -std=c++17 -c CycloneDDS header probe",
            observed=dds_result,
            blockers=[] if dds_ok else ["cyclonedds_header_compile_failed"],
            follow_up_work_unlocked=["dds_compile_surface_receipt"] if dds_ok else [],
            external_requirement="Run DDS network timing against ROS2/SDK2 runtime or robot/sim.",
        )
    )

    sdk2_result = _compile_probe(
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
    )
    sdk2_ok = sdk2_result.get("returncode") == 0
    receipts.append(
        _receipt(
            blocker_key="ros2_sdk2_g1pilot_command_echo_missing",
            probe_key="unitree_sdk2_header_compile",
            status="ok_sdk2_header_compile_only"
            if sdk2_ok
            else "blocked_sdk2_header_compile_failed",
            succeeded=sdk2_ok,
            local_probe_executed=bool(sdk2_result.get("attempted")),
            evidence_class="compile_only_sdk2_header_probe_no_write",
            target_path=str(sdk2_root),
            command_or_import="c++ -std=c++17 -c Unitree SDK2 header probe",
            observed=sdk2_result,
            blockers=[] if sdk2_ok else ["unitree_sdk2_header_compile_failed"],
            external_requirement="Build SDK2 in supported Linux/ROS2 runtime before command echo.",
        )
    )

    policy_path = rl_gym_root / "deploy/pre_train/g1/motion.pt"
    g1_urdfs = list((rl_gym_root / "resources/robots/g1_description").glob("*.urdf"))
    g1_xmls = list((rl_gym_root / "resources/robots/g1_description").glob("*.xml"))
    deploy_configs = list((rl_gym_root / "deploy").glob("**/configs/g1.yaml"))
    receipts.append(
        _receipt(
            blocker_key="policy_controlled_mujoco_or_hardware_trace_missing",
            probe_key="unitree_rl_gym_policy_asset_visibility",
            status="ok_static_policy_assets_visible"
            if policy_path.exists() and g1_urdfs
            else "blocked_policy_assets_missing",
            succeeded=policy_path.exists() and bool(g1_urdfs),
            local_probe_executed=True,
            evidence_class="static_policy_asset_visibility_no_execution",
            target_path=str(rl_gym_root),
            observed={
                "g1_policy_checkpoint_present": policy_path.exists(),
                "g1_policy_checkpoint_path": str(policy_path),
                "g1_urdf_count": len(g1_urdfs),
                "g1_xml_count": len(g1_xmls),
                "deploy_g1_config_count": len(deploy_configs),
            },
            blockers=["policy_not_loaded_or_executed", "no_policy_controlled_trace"],
            follow_up_work_unlocked=["policy_asset_manifest_receipts"],
            external_requirement="Run policy-controlled sim on a proper Unitree/Isaac/MuJoCo runtime host.",
        )
    )

    g1_tasks = (
        list((isaaclab_root / "tasks/g1_tasks").glob("*"))
        if isaaclab_root.exists()
        else []
    )
    isaac_runtime_modules = {
        "isaaclab": _module_available("isaaclab"),
        "omni": _module_available("omni"),
    }
    receipts.append(
        _receipt(
            blocker_key="policy_controlled_mujoco_or_hardware_trace_missing",
            probe_key="unitree_isaaclab_static_task_surface",
            status="ok_static_isaaclab_tasks_visible"
            if g1_tasks
            else "blocked_isaaclab_task_surface_missing",
            succeeded=bool(g1_tasks),
            local_probe_executed=True,
            evidence_class="static_isaaclab_task_surface_no_runtime",
            target_path=str(isaaclab_root),
            observed={
                "g1_task_count": len(g1_tasks),
                "runtime_module_status": isaac_runtime_modules,
            },
            blockers=[
                name for name, present in isaac_runtime_modules.items() if not present
            ],
            follow_up_work_unlocked=["isaaclab_task_manifest_receipts"]
            if g1_tasks
            else [],
            external_requirement="Install/run Isaac Lab provider before sim runtime evidence.",
        )
    )

    lerobot_surfaces = [
        lerobot_root / "unitree_lerobot/eval_robot/eval_g1.py",
        lerobot_root / "unitree_lerobot/eval_robot/eval_g1_sim.py",
        lerobot_root / "unitree_lerobot/utils/convert_unitree_json_to_lerobot.py",
        lerobot_root / "unitree_lerobot/utils/convert_unitree_json_to_h5.py",
    ]
    lerobot_present = [path for path in lerobot_surfaces if path.exists()]
    receipts.append(
        _receipt(
            blocker_key="rosbag2_or_mcap_real_stream_import_missing",
            probe_key="unitree_lerobot_static_adapter_surface",
            status="ok_static_lerobot_adapter_surface_visible"
            if len(lerobot_present) == len(lerobot_surfaces)
            else "blocked_lerobot_adapter_surface_missing",
            succeeded=len(lerobot_present) == len(lerobot_surfaces),
            local_probe_executed=True,
            evidence_class="static_lerobot_adapter_surface_no_robot",
            target_path=str(lerobot_root),
            observed={
                "expected_surface_count": len(lerobot_surfaces),
                "present_surface_count": len(lerobot_present),
                "present_surfaces": [str(path) for path in lerobot_present],
            },
            blockers=["real_unitree_dataset_or_stream_missing"],
            follow_up_work_unlocked=["json_h5_lerobot_trace_conversion_receipts"],
            external_requirement="Collect real or honest-sim Unitree traces before dataset import evidence.",
        )
    )

    trace_modules = {
        "rosbag2_py": _module_available("rosbag2_py"),
        "mcap": _module_available("mcap"),
    }
    receipts.append(
        _receipt(
            blocker_key="rosbag2_or_mcap_real_stream_import_missing",
            probe_key="rosbag2_mcap_import_modules",
            status="ok_trace_import_modules_available"
            if all(trace_modules.values())
            else "blocked_missing_trace_import_modules",
            succeeded=all(trace_modules.values()),
            local_probe_executed=True,
            evidence_class="trace_import_module_probe",
            command_or_import="importlib.util.find_spec(rosbag2_py, mcap)",
            observed={"module_status": trace_modules},
            missing=[name for name, present in trace_modules.items() if not present],
            blockers=[name for name, present in trace_modules.items() if not present],
            external_requirement="Install rosbag2_py/MCAP and provide real stream files.",
        )
    )

    calibration_sidecars = [
        Path("artifacts/economic_world_model/phase4_unitree_runtime_evidence_bridge")
        / "unitree_physical_calibration_sidecar_v1.json",
        Path("artifacts/economic_world_model/phase4_unitree_bringup_readiness")
        / "unitree_physical_calibration_sidecar_v1.json",
    ]
    present_sidecars = [path for path in calibration_sidecars if path.exists()]
    receipts.append(
        _receipt(
            blocker_key="physical_stop_distance_and_calibrated_safety_limits_missing",
            probe_key="physical_calibration_sidecar",
            status="ok_calibration_sidecar_present"
            if present_sidecars
            else "blocked_calibration_sidecar_missing",
            succeeded=bool(present_sidecars),
            local_probe_executed=True,
            evidence_class="calibration_sidecar_presence_probe",
            observed={
                "expected_sidecar_paths": [str(path) for path in calibration_sidecars],
                "present_sidecar_paths": [str(path) for path in present_sidecars],
            },
            blockers=[]
            if present_sidecars
            else ["physical_calibration_sidecar_missing"],
            external_requirement="Measure stop distance, limits, and e-stop behavior on robot or honest sim.",
        )
    )

    receipts.append(
        _receipt(
            blocker_key="operator_teleop_runtime_drill_missing",
            probe_key="operator_teleop_runtime_surface",
            status="ok_static_teleop_surface_visible"
            if bool(g1pilot_surface.get("teleop_launch_present"))
            else "blocked_teleop_surface_missing",
            succeeded=bool(g1pilot_surface.get("teleop_launch_present")),
            local_probe_executed=True,
            evidence_class="static_teleop_surface_probe_no_launch",
            target_path=str(g1pilot_root / "launch/teleoperation_launcher.launch.py"),
            observed={
                "teleop_launch_present": bool(
                    g1pilot_surface.get("teleop_launch_present")
                ),
                "runtime_missing_imports": runtime_missing,
            },
            blockers=["teleop_launch_not_executed", *runtime_missing],
            follow_up_work_unlocked=["teleop_launch_preflight_receipts"],
            external_requirement="Install ROS2/G1Pilot deps and run operator drills against sim or hardware.",
        )
    )
    return receipts


def build_phase4_unitree_blocker_stress_probes(
    *,
    local_roots: Mapping[str, str | Path] | None = None,
    stress_steps: int = 100,
    artifact_refs: Optional[Mapping[str, Any]] = None,
) -> tuple[
    Phase4UnitreeBlockerStressProbeReport,
    list[UnitreeBlockerStressProbeReceipt],
    list[UnitreeMujocoModelStressReceipt],
]:
    probe_receipts = build_blocker_stress_probe_receipts(local_roots=local_roots)
    mujoco_receipts = build_mujoco_model_stress_receipts(
        local_roots=local_roots,
        stress_steps=stress_steps,
    )
    mujoco_success_count = sum(
        1 for receipt in mujoco_receipts if receipt.step_executed
    )
    probe_receipts.append(
        _receipt(
            blocker_key="policy_controlled_mujoco_or_hardware_trace_missing",
            probe_key="unitree_mujoco_g1_model_stress",
            status="ok_no_policy_model_stress"
            if mujoco_success_count
            else "blocked_no_mujoco_model_stress_success",
            succeeded=bool(mujoco_success_count),
            local_probe_executed=True,
            evidence_class="mujoco_multi_model_stress_no_policy_no_bridge",
            observed={
                "model_receipt_count": len(mujoco_receipts),
                "model_success_count": mujoco_success_count,
                "stress_steps": int(stress_steps),
            },
            blockers=[
                "policy_not_loaded_or_executed",
                "ros2_bridge_not_active",
                "hardware_not_executed",
            ],
            follow_up_work_unlocked=["mujoco_multi_model_stress_receipts"]
            if mujoco_success_count
            else [],
            external_requirement=(
                "Run policy-controlled MuJoCo/Unitree sim or hardware traces "
                "before claiming controlled runtime evidence."
            ),
        )
    )
    succeeded = [receipt for receipt in probe_receipts if receipt.succeeded]
    blocked = [receipt for receipt in probe_receipts if not receipt.succeeded]
    g1pilot_static_ok = any(
        receipt.probe_key == "g1pilot_static_launch_surface" and receipt.succeeded
        for receipt in probe_receipts
    )
    cyclonedds_ok = any(
        receipt.probe_key == "cyclonedds_header_compile" and receipt.succeeded
        for receipt in probe_receipts
    )
    sdk2_ok = any(
        receipt.probe_key == "unitree_sdk2_header_compile" and receipt.succeeded
        for receipt in probe_receipts
    )
    ros2_available = any(
        receipt.probe_key == "host_ros2_colcon_toolchain" and receipt.succeeded
        for receipt in probe_receipts
    )
    trace_modules_available = any(
        receipt.probe_key == "rosbag2_mcap_import_modules" and receipt.succeeded
        for receipt in probe_receipts
    )
    policy_visible = any(
        receipt.probe_key == "unitree_rl_gym_policy_asset_visibility"
        and receipt.succeeded
        for receipt in probe_receipts
    )
    isaac_visible = any(
        receipt.probe_key == "unitree_isaaclab_static_task_surface"
        and receipt.succeeded
        for receipt in probe_receipts
    )
    lerobot_visible = any(
        receipt.probe_key == "unitree_lerobot_static_adapter_surface"
        and receipt.succeeded
        for receipt in probe_receipts
    )
    all_attempts_complete = (
        len(probe_receipts) == len(BLOCKER_PROBE_KEYS)
        and all(receipt.local_probe_executed for receipt in probe_receipts)
        and bool(mujoco_receipts)
        and all(
            receipt.step_attempted or not receipt.mujoco_module_available
            for receipt in mujoco_receipts
        )
    )
    unlocked = sorted(
        {item for receipt in probe_receipts for item in receipt.follow_up_work_unlocked}
    )
    remaining = [
        "ros2_colcon_build_and_generated_message_import_not_executed",
        "ros2_sdk2_g1pilot_command_echo_missing",
        "rosbag2_or_mcap_real_stream_import_missing",
        "policy_controlled_mujoco_or_hardware_trace_missing",
        "physical_stop_distance_and_calibrated_safety_limits_missing",
        "operator_teleop_runtime_drill_missing",
        "dds_network_or_on_robot_timing_missing",
    ]
    payload = {
        "probe_receipts": len(probe_receipts),
        "mujoco_success_count": mujoco_success_count,
        "succeeded": len(succeeded),
        "blocked": len(blocked),
        "stress_steps": stress_steps,
    }
    report = Phase4UnitreeBlockerStressProbeReport(
        report_id=stable_id("phase4_unitree_blocker_stress_probe", payload),
        status="ok" if all_attempts_complete else "blocked",
        probe_receipt_count=len(probe_receipts),
        succeeded_probe_count=len(succeeded),
        blocked_probe_count=len(blocked),
        mujoco_model_stress_receipt_count=len(mujoco_receipts),
        mujoco_model_stress_success_count=mujoco_success_count,
        all_local_probe_attempts_complete=all_attempts_complete,
        local_phase4_probe_expansion_complete=all_attempts_complete,
        g1_mujoco_model_stress_succeeded=mujoco_success_count > 0,
        g1pilot_static_surface_succeeded=g1pilot_static_ok,
        cyclonedds_header_compile_succeeded=cyclonedds_ok,
        unitree_sdk2_header_compile_succeeded=sdk2_ok,
        ros2_runtime_available=ros2_available,
        trace_import_modules_available=trace_modules_available,
        policy_checkpoint_visible=policy_visible,
        isaaclab_task_surface_visible=isaac_visible,
        lerobot_adapter_surface_visible=lerobot_visible,
        denied_gates=_denied_gates(),
        remaining_evidence_blockers=remaining,
        unlocked_local_followups=unlocked,
        artifact_refs=mapping(artifact_refs),
    )
    return report, probe_receipts, mujoco_receipts


def save_phase4_unitree_blocker_stress_probes(
    output_dir: str | Path,
    *,
    report: Phase4UnitreeBlockerStressProbeReport,
    probe_receipts: list[UnitreeBlockerStressProbeReceipt],
    mujoco_receipts: list[UnitreeMujocoModelStressReceipt],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase4_unitree_blocker_stress_probe_report_v1.json",
        "probe_receipts_path": output
        / "unitree_blocker_stress_probe_receipts_v1.jsonl",
        "mujoco_model_stress_receipts_path": output
        / "unitree_mujoco_model_stress_receipts_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    _write_rows(
        paths["probe_receipts_path"],
        [receipt.to_dict() for receipt in probe_receipts],
    )
    _write_rows(
        paths["mujoco_model_stress_receipts_path"],
        [receipt.to_dict() for receipt in mujoco_receipts],
    )
    return {key: str(path) for key, path in paths.items()}


def load_phase4_unitree_blocker_stress_probe_report(
    path: str | Path,
) -> Phase4UnitreeBlockerStressProbeReport:
    return Phase4UnitreeBlockerStressProbeReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_unitree_blocker_stress_probe_receipts(
    path: str | Path,
) -> list[UnitreeBlockerStressProbeReceipt]:
    return [
        UnitreeBlockerStressProbeReceipt.from_dict(row) for row in _load_jsonl(path)
    ]


def load_unitree_mujoco_model_stress_receipts(
    path: str | Path,
) -> list[UnitreeMujocoModelStressReceipt]:
    return [UnitreeMujocoModelStressReceipt.from_dict(row) for row in _load_jsonl(path)]
