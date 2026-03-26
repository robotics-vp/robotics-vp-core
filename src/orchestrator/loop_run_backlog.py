"""Typed loop-run backlog and host precondition evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence

from src.evidence.benchmark_gating import build_benchmark_gate_report
from src.evidence.grounded_data_host import collect_grounded_data_host_capabilities
from src.evidence.preconditions import ExecutionPreconditionsReport, build_execution_preconditions
from src.utils.json_safe import to_json_safe


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOOP_RUN_BACKLOG_PATH = REPO_ROOT / "scripts" / "LOOP_RUN_BACKLOG.json"


def _string_list(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def collect_host_capabilities() -> Dict[str, Any]:
    def has_module(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    openvla_model_ref = os.environ.get("OPENVLA_MODEL_NAME") or os.environ.get("OPENVLA_MODEL") or ""
    openvla_model_path_ready = bool(openvla_model_ref and Path(openvla_model_ref).exists())
    return {
        **collect_grounded_data_host_capabilities(repo_root=REPO_ROOT),
        "torch_available": has_module("torch"),
        "mujoco_available": has_module("mujoco"),
        "transformers_available": has_module("transformers"),
        "timm_available": has_module("timm"),
        "imageio_available": has_module("imageio"),
        "openvla_model_ref_present": bool(openvla_model_ref),
        "openvla_model_path_ready": openvla_model_path_ready,
        "droid_dataset_present": bool(
            (os.environ.get("DROID_DATASET_ROOT") and Path(os.environ["DROID_DATASET_ROOT"]).exists())
            or (REPO_ROOT / "data" / "external" / "droid").exists()
        ),
        "workcell_bootstrap_script_present": (REPO_ROOT / "scripts" / "bootstrap_semantic_workcell_loop.py").exists(),
    }


@dataclass(frozen=True)
class LoopRunBacklogItem:
    loop_run_id: str
    title: str
    command: str
    cwd: str = "."
    priority: str = "P1"
    owner: str = "codex"
    auto_trigger: bool = False
    notes: str = ""
    required_capabilities: Dict[str, bool] = field(default_factory=dict)
    required_python_modules: list[str] = field(default_factory=list)
    required_commands: list[str] = field(default_factory=list)
    required_paths: list[str] = field(default_factory=list)
    required_globs: list[str] = field(default_factory=list)
    required_env_vars: list[str] = field(default_factory=list)
    required_env_path_vars: list[str] = field(default_factory=list)
    required_internal_data: list[str] = field(default_factory=list)
    required_external_data: list[str] = field(default_factory=list)
    benchmark_gate: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoopRunBacklogItem":
        return cls(
            loop_run_id=str(payload.get("loop_run_id", "")),
            title=str(payload.get("title", "")),
            command=str(payload.get("command", "")),
            cwd=str(payload.get("cwd", ".")),
            priority=str(payload.get("priority", "P1")),
            owner=str(payload.get("owner", "codex")),
            auto_trigger=bool(payload.get("auto_trigger", False)),
            notes=str(payload.get("notes", "")),
            required_capabilities=_mapping(payload.get("required_capabilities")),
            required_python_modules=_string_list(payload.get("required_python_modules")),
            required_commands=_string_list(payload.get("required_commands")),
            required_paths=_string_list(payload.get("required_paths")),
            required_globs=_string_list(payload.get("required_globs")),
            required_env_vars=_string_list(payload.get("required_env_vars")),
            required_env_path_vars=_string_list(payload.get("required_env_path_vars")),
            required_internal_data=_string_list(payload.get("required_internal_data")),
            required_external_data=_string_list(payload.get("required_external_data")),
            benchmark_gate=_mapping(payload.get("benchmark_gate")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_run_id": self.loop_run_id,
            "title": self.title,
            "command": self.command,
            "cwd": self.cwd,
            "priority": self.priority,
            "owner": self.owner,
            "auto_trigger": self.auto_trigger,
            "notes": self.notes,
            "required_capabilities": dict(self.required_capabilities),
            "required_python_modules": list(self.required_python_modules),
            "required_commands": list(self.required_commands),
            "required_paths": list(self.required_paths),
            "required_globs": list(self.required_globs),
            "required_env_vars": list(self.required_env_vars),
            "required_env_path_vars": list(self.required_env_path_vars),
            "required_internal_data": list(self.required_internal_data),
            "required_external_data": list(self.required_external_data),
            "benchmark_gate": dict(self.benchmark_gate),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class LoopRunAssessment:
    item: LoopRunBacklogItem
    readiness: ExecutionPreconditionsReport
    benchmark_gate: Optional[ExecutionPreconditionsReport] = None
    pending_requirements: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        benchmark_ready = True if self.benchmark_gate is None else bool(self.benchmark_gate.ready)
        return bool(self.readiness.ready and benchmark_ready)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_run_id": self.item.loop_run_id,
            "title": self.item.title,
            "command": self.item.command,
            "cwd": self.item.cwd,
            "priority": self.item.priority,
            "auto_trigger": self.item.auto_trigger,
            "ready": self.ready,
            "pending_requirements": list(self.pending_requirements),
            "readiness": self.readiness.to_dict(),
            "benchmark_gate": self.benchmark_gate.to_dict() if self.benchmark_gate is not None else None,
            "required_internal_data": list(self.item.required_internal_data),
            "required_external_data": list(self.item.required_external_data),
            "notes": self.item.notes,
        }


def load_loop_run_backlog(path: Optional[Path] = None) -> list[LoopRunBacklogItem]:
    backlog_path = path or DEFAULT_LOOP_RUN_BACKLOG_PATH
    if not backlog_path.exists():
        return []
    payload = json.loads(backlog_path.read_text(encoding="utf-8"))
    return [
        LoopRunBacklogItem.from_dict(row)
        for row in list(payload.get("backlog", []) or [])
    ]


def _env_path_refs(item: LoopRunBacklogItem) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    for key in item.required_env_path_vars:
        value = os.environ.get(key, "")
        if value and Path(value).exists():
            refs[f"env_path::{key}"] = value
        else:
            refs[f"env_path::{key}"] = ""
    for rel_path in item.required_paths:
        path = Path(rel_path)
        if not path.is_absolute():
            path = REPO_ROOT / rel_path
        refs[f"path::{rel_path}"] = str(path) if path.exists() else ""
    for pattern in item.required_globs:
        matches = sorted(str(path) for path in REPO_ROOT.glob(pattern))
        refs[f"glob::{pattern}"] = matches
    return refs


def _module_and_command_signals(item: LoopRunBacklogItem, host_capabilities: Mapping[str, Any]) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}
    for name, expected in sorted(dict(item.required_capabilities).items()):
        signals[str(name)] = bool(host_capabilities.get(str(name), False)) == bool(expected)
    for module in item.required_python_modules:
        signals[f"python_module::{module}"] = importlib.util.find_spec(module) is not None
    for command in item.required_commands:
        signals[f"command::{command}"] = shutil.which(command) is not None
    for env_var in item.required_env_vars:
        signals[f"env_var::{env_var}"] = bool(os.environ.get(env_var, "").strip())
    return signals


def _host_metadata(host_capabilities: Mapping[str, Any]) -> Dict[str, Any]:
    scene_tracks_backend = "real" if all(
        bool(host_capabilities.get(key, False))
        for key in (
            "gpu_available",
            "opencv_available",
            "sam3d_objects_repo_available",
            "sam3d_body_repo_available",
            "sam3d_objects_checkpoint_available",
            "sam3d_body_checkpoint_available",
        )
    ) else "unavailable"
    teacher_backend = "real" if all(
        bool(host_capabilities.get(key, False))
        for key in ("gpu_available", "transformers_available", "openvla_model_ref_present")
    ) and bool(host_capabilities.get("openvla_model_path_ready", False)) else "unavailable"
    vision_backbone = "real" if bool(host_capabilities.get("timm_available", False) or host_capabilities.get("transformers_available", False)) else "unavailable"
    return {
        "scene_tracks_backend": scene_tracks_backend,
        "semantic_memory_grounded": scene_tracks_backend == "real",
        "semantic_grounding_non_heuristic": scene_tracks_backend == "real",
        "openvla_backend_selected": teacher_backend,
        "vision_backbone_selected": vision_backbone,
    }


def evaluate_loop_run_item(
    item: LoopRunBacklogItem,
    *,
    host_capabilities: Optional[Mapping[str, Any]] = None,
) -> LoopRunAssessment:
    host = dict(host_capabilities or collect_host_capabilities())
    readiness = build_execution_preconditions(
        subject_id=item.loop_run_id,
        subject_kind="loop_run",
        artifact_refs=_env_path_refs(item),
        required_artifact_refs=[
            f"path::{path}"
            for path in item.required_paths
        ] + [
            f"env_path::{env_var}"
            for env_var in item.required_env_path_vars
        ] + [
            f"glob::{pattern}"
            for pattern in item.required_globs
        ],
        signal_values=_module_and_command_signals(item, host),
        required_boolean_signals={
            key: True
            for key in (
                [f"python_module::{module}" for module in item.required_python_modules]
                + [f"command::{command}" for command in item.required_commands]
                + [f"env_var::{env_var}" for env_var in item.required_env_vars]
                + list(str(key) for key in item.required_capabilities.keys())
            )
        },
        metadata={"host_capabilities": host},
    )

    benchmark_gate = None
    if item.benchmark_gate:
        benchmark_gate = build_benchmark_gate_report(
            subject_id=item.loop_run_id,
            subject_kind="loop_run_benchmark_gate",
            metadata=_host_metadata(host),
            require_real_scene_tracks=bool(item.benchmark_gate.get("require_real_scene_tracks", True)),
            require_teacher_runtime=bool(item.benchmark_gate.get("require_teacher_runtime", False)),
            require_vision_backbone=bool(item.benchmark_gate.get("require_vision_backbone", False)),
        )

    pending = list(readiness.blocking_preconditions)
    if benchmark_gate is not None:
        pending.extend(list(benchmark_gate.blocking_preconditions))
    return LoopRunAssessment(
        item=item,
        readiness=readiness,
        benchmark_gate=benchmark_gate,
        pending_requirements=sorted(set(pending)),
    )


def evaluate_loop_run_backlog(
    *,
    backlog_items: Optional[Sequence[LoopRunBacklogItem]] = None,
    backlog_path: Optional[Path] = None,
    host_capabilities: Optional[Mapping[str, Any]] = None,
) -> list[LoopRunAssessment]:
    items = list(backlog_items or load_loop_run_backlog(backlog_path))
    host = dict(host_capabilities or collect_host_capabilities())
    return [evaluate_loop_run_item(item, host_capabilities=host) for item in items]


__all__ = [
    "DEFAULT_LOOP_RUN_BACKLOG_PATH",
    "LoopRunAssessment",
    "LoopRunBacklogItem",
    "collect_host_capabilities",
    "evaluate_loop_run_backlog",
    "evaluate_loop_run_item",
    "load_loop_run_backlog",
]
