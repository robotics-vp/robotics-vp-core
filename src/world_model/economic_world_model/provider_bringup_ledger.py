"""Cross-WM provider bring-up readiness ledger.

This module compiles provider/runtime backlog surfaces into typed, launch-blocked
ledger rows. It is a local planning and receipt surface only: it does not
download weights, launch RunPod, execute providers, train models, write
checkpoints, operate hardware, or make promotion claims.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

PROVIDER_BRINGUP_LEDGER_ENTRY_VERSION = "provider_bringup_ledger_entry_v1"
PROVIDER_BRINGUP_LEDGER_REPORT_VERSION = "provider_bringup_ledger_report_v1"

DENIED_PROVIDER_BRINGUP_AUTHORITIES = (
    "provider_executed",
    "gpu_executed",
    "runpod_launched",
    "weights_downloaded",
    "weights_written",
    "training_executed",
    "hardware_executed",
    "ros2_publish_attempted",
    "unitree_sdk2_write_enabled",
    "live_policy_control",
    "reward_math_mutation",
    "promotion_eligible",
)

REQUIRED_PROVIDER_FAMILIES = (
    "sam_sam3d",
    "dino_siglip",
    "vjepa2",
    "openvla",
    "isaac_unitree",
    "holosoma",
)

_TEMPLATE_GUARD = (
    "echo 'TEMPLATE_ONLY_PROVIDER_LEDGER: replace guard with a real approved "
    "provider command before launch' && false"
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_PROVIDER_BRINGUP_AUTHORITIES}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _backlog_by_id(path: str | Path) -> dict[str, Mapping[str, Any]]:
    target = Path(path)
    if not target.exists():
        return {}
    payload = _load_json(target)
    return {
        str(row.get("id", "")): _mapping(row)
        for row in list(payload.get("backlog", []) or [])
        if isinstance(row, Mapping) and row.get("id")
    }


@dataclass(frozen=True)
class ProviderLedgerSpec:
    provider_key: str
    provider_family: str
    provider_label: str
    owner_wm: str
    subsystem: str
    run_class: str
    pod_class: str
    runpod_profile: str
    unavailable_posture: str
    source_backlog_ids: list[str]
    surface_roles: list[str]
    source_files: list[str]
    command_templates: list[str]
    local_verification_commands: list[str]
    expected_receipts: list[str]
    artifact_paths: list[str]
    blocker_codes: list[str]
    required_prerequisites: list[str]
    notes: str = ""


@dataclass(frozen=True)
class ProviderBringupLedgerEntry:
    ledger_entry_id: str
    provider_key: str
    provider_family: str
    provider_label: str
    owner_wm: str
    subsystem: str
    run_class: str
    pod_class: str
    runpod_profile: str
    status: str
    unavailable_posture: str
    launch_allowed: bool
    provider_bringup_ready: bool
    local_verification_available: bool
    command_templates: list[str] = field(default_factory=list)
    local_verification_commands: list[str] = field(default_factory=list)
    expected_receipts: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    blocker_codes: list[str] = field(default_factory=list)
    required_prerequisites: list[str] = field(default_factory=list)
    missing_prerequisites: list[str] = field(default_factory=list)
    source_backlog_ids: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    surface_roles: list[str] = field(default_factory=list)
    source_backlog_rows: list[dict[str, Any]] = field(default_factory=list)
    manifest_stub: dict[str, Any] = field(default_factory=dict)
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    provider_executed: bool = False
    gpu_executed: bool = False
    runpod_launched: bool = False
    weights_downloaded: bool = False
    weights_written: bool = False
    training_executed: bool = False
    hardware_executed: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PROVIDER_BRINGUP_LEDGER_ENTRY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_entry_id": self.ledger_entry_id,
            "version": self.version,
            "provider_key": self.provider_key,
            "provider_family": self.provider_family,
            "provider_label": self.provider_label,
            "owner_wm": self.owner_wm,
            "subsystem": self.subsystem,
            "run_class": self.run_class,
            "pod_class": self.pod_class,
            "runpod_profile": self.runpod_profile,
            "status": self.status,
            "unavailable_posture": self.unavailable_posture,
            "launch_allowed": bool(self.launch_allowed),
            "provider_bringup_ready": bool(self.provider_bringup_ready),
            "local_verification_available": bool(self.local_verification_available),
            "command_templates": list(self.command_templates),
            "local_verification_commands": list(self.local_verification_commands),
            "expected_receipts": list(self.expected_receipts),
            "artifact_paths": list(self.artifact_paths),
            "blocker_codes": list(self.blocker_codes),
            "required_prerequisites": list(self.required_prerequisites),
            "missing_prerequisites": list(self.missing_prerequisites),
            "source_backlog_ids": list(self.source_backlog_ids),
            "source_files": list(self.source_files),
            "surface_roles": list(self.surface_roles),
            "source_backlog_rows": [_mapping(row) for row in self.source_backlog_rows],
            "manifest_stub": _mapping(self.manifest_stub),
            "denied_gates": dict(self.denied_gates),
            "provider_executed": bool(self.provider_executed),
            "gpu_executed": bool(self.gpu_executed),
            "runpod_launched": bool(self.runpod_launched),
            "weights_downloaded": bool(self.weights_downloaded),
            "weights_written": bool(self.weights_written),
            "training_executed": bool(self.training_executed),
            "hardware_executed": bool(self.hardware_executed),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderBringupLedgerEntry":
        return cls(
            ledger_entry_id=str(payload.get("ledger_entry_id", "")),
            provider_key=str(payload.get("provider_key", "")),
            provider_family=str(payload.get("provider_family", "")),
            provider_label=str(payload.get("provider_label", "")),
            owner_wm=str(payload.get("owner_wm", "")),
            subsystem=str(payload.get("subsystem", "")),
            run_class=str(payload.get("run_class", "provider")),
            pod_class=str(payload.get("pod_class", "provider")),
            runpod_profile=str(payload.get("runpod_profile", "")),
            status=str(payload.get("status", "blocked")),
            unavailable_posture=str(payload.get("unavailable_posture", "")),
            launch_allowed=bool(payload.get("launch_allowed", False)),
            provider_bringup_ready=bool(payload.get("provider_bringup_ready", False)),
            local_verification_available=bool(
                payload.get("local_verification_available", False)
            ),
            command_templates=_strings(payload.get("command_templates")),
            local_verification_commands=_strings(
                payload.get("local_verification_commands")
            ),
            expected_receipts=_strings(payload.get("expected_receipts")),
            artifact_paths=_strings(payload.get("artifact_paths")),
            blocker_codes=_strings(payload.get("blocker_codes")),
            required_prerequisites=_strings(payload.get("required_prerequisites")),
            missing_prerequisites=_strings(payload.get("missing_prerequisites")),
            source_backlog_ids=_strings(payload.get("source_backlog_ids")),
            source_files=_strings(payload.get("source_files")),
            surface_roles=_strings(payload.get("surface_roles")),
            source_backlog_rows=[
                _mapping(row)
                for row in list(payload.get("source_backlog_rows", []) or [])
                if isinstance(row, Mapping)
            ],
            manifest_stub=_mapping(payload.get("manifest_stub")),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(payload.get("denied_gates", {}) or {}).items()
                },
            },
            provider_executed=bool(payload.get("provider_executed", False)),
            gpu_executed=bool(payload.get("gpu_executed", False)),
            runpod_launched=bool(payload.get("runpod_launched", False)),
            weights_downloaded=bool(payload.get("weights_downloaded", False)),
            weights_written=bool(payload.get("weights_written", False)),
            training_executed=bool(payload.get("training_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", PROVIDER_BRINGUP_LEDGER_ENTRY_VERSION)),
        )


@dataclass(frozen=True)
class ProviderBringupLedgerReport:
    report_id: str
    status: str
    entry_count: int
    required_family_count: int
    covered_required_family_count: int
    launch_allowed_count: int
    provider_bringup_ready_count: int
    local_verification_available_count: int
    runpod_template_count: int
    missing_prerequisite_count: int
    all_entries_fail_closed: bool
    provider_executed: bool = False
    gpu_executed: bool = False
    runpod_launched: bool = False
    weights_downloaded: bool = False
    weights_written: bool = False
    training_executed: bool = False
    hardware_executed: bool = False
    ros2_publish_attempted: bool = False
    unitree_sdk2_write_enabled: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    covered_required_families: list[str] = field(default_factory=list)
    missing_required_families: list[str] = field(default_factory=list)
    blocker_codes: list[str] = field(default_factory=list)
    prerequisite_status: dict[str, bool] = field(default_factory=dict)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PROVIDER_BRINGUP_LEDGER_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "entry_count": int(self.entry_count),
            "required_family_count": int(self.required_family_count),
            "covered_required_family_count": int(self.covered_required_family_count),
            "launch_allowed_count": int(self.launch_allowed_count),
            "provider_bringup_ready_count": int(self.provider_bringup_ready_count),
            "local_verification_available_count": int(
                self.local_verification_available_count
            ),
            "runpod_template_count": int(self.runpod_template_count),
            "missing_prerequisite_count": int(self.missing_prerequisite_count),
            "all_entries_fail_closed": bool(self.all_entries_fail_closed),
            "provider_executed": bool(self.provider_executed),
            "gpu_executed": bool(self.gpu_executed),
            "runpod_launched": bool(self.runpod_launched),
            "weights_downloaded": bool(self.weights_downloaded),
            "weights_written": bool(self.weights_written),
            "training_executed": bool(self.training_executed),
            "hardware_executed": bool(self.hardware_executed),
            "ros2_publish_attempted": bool(self.ros2_publish_attempted),
            "unitree_sdk2_write_enabled": bool(self.unitree_sdk2_write_enabled),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": dict(self.denied_gates),
            "covered_required_families": list(self.covered_required_families),
            "missing_required_families": list(self.missing_required_families),
            "blocker_codes": list(self.blocker_codes),
            "prerequisite_status": dict(self.prerequisite_status),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderBringupLedgerReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "blocked")),
            entry_count=int(payload.get("entry_count", 0) or 0),
            required_family_count=int(payload.get("required_family_count", 0) or 0),
            covered_required_family_count=int(
                payload.get("covered_required_family_count", 0) or 0
            ),
            launch_allowed_count=int(payload.get("launch_allowed_count", 0) or 0),
            provider_bringup_ready_count=int(
                payload.get("provider_bringup_ready_count", 0) or 0
            ),
            local_verification_available_count=int(
                payload.get("local_verification_available_count", 0) or 0
            ),
            runpod_template_count=int(payload.get("runpod_template_count", 0) or 0),
            missing_prerequisite_count=int(
                payload.get("missing_prerequisite_count", 0) or 0
            ),
            all_entries_fail_closed=bool(payload.get("all_entries_fail_closed", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            gpu_executed=bool(payload.get("gpu_executed", False)),
            runpod_launched=bool(payload.get("runpod_launched", False)),
            weights_downloaded=bool(payload.get("weights_downloaded", False)),
            weights_written=bool(payload.get("weights_written", False)),
            training_executed=bool(payload.get("training_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            ros2_publish_attempted=bool(payload.get("ros2_publish_attempted", False)),
            unitree_sdk2_write_enabled=bool(
                payload.get("unitree_sdk2_write_enabled", False)
            ),
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
            covered_required_families=_strings(
                payload.get("covered_required_families")
            ),
            missing_required_families=_strings(payload.get("missing_required_families")),
            blocker_codes=_strings(payload.get("blocker_codes")),
            prerequisite_status={
                str(key): bool(value)
                for key, value in dict(payload.get("prerequisite_status", {}) or {}).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", PROVIDER_BRINGUP_LEDGER_REPORT_VERSION)),
        )


def _provider_specs() -> list[ProviderLedgerSpec]:
    return [
        ProviderLedgerSpec(
            provider_key="sam_sam3d_scene_ir",
            provider_family="sam_sam3d",
            provider_label="SAM / SAM3D scene grounding",
            owner_wm="perception_grounding",
            subsystem="scene_ir_tracker",
            run_class="provider",
            pod_class="provider",
            runpod_profile="provider",
            unavailable_posture="stub_or_zero_inference_only_until_sam3d_weights_runtime_and_receipts_exist",
            source_backlog_ids=[],
            surface_roles=["lower-WM producer", "provider/hardware adapter", "receipt substrate"],
            source_files=[
                "src/vision/scene_ir_tracker/",
                "third_party/sam3d_objects_wrapper.py",
                "third_party/sam3d_body_wrapper.py",
                "scripts/run_scene_ir_eval.py",
            ],
            command_templates=[
                "python3 scripts/run_scene_ir_eval.py --input-dir {input_dir} --output artifacts/economic_world_model/provider_runs/{run_id}/scene_ir_eval.md --manifest {manifest}",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 scripts/run_scene_ir_eval.py --help",
                "python3 -m pytest -q tests/vision/scene_ir_tracker/test_fallback_behavior.py tests/vision/scene_ir_tracker/test_upstream_integration.py",
            ],
            expected_receipts=[
                "scene_tracks_provider_truth_v1",
                "sam3d_provider_invocation_receipt_v1",
                "sam3d_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/scene_tracks_provider_truth_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/sam3d_unavailable_receipt_v1.json",
            ],
            blocker_codes=[
                "sam3d_weights_missing",
                "sam3d_runtime_not_verified",
                "gpu_provider_execution_not_run",
            ],
            required_prerequisites=[
                "CUDA-capable provider host",
                "SAM3D weights/runtime mounted",
                "scene input manifest",
            ],
        ),
        ProviderLedgerSpec(
            provider_key="dino_siglip_vision_backbone",
            provider_family="dino_siglip",
            provider_label="DINOv2 / SigLIP vision backbone",
            owner_wm="perception_grounding",
            subsystem="vision_backbone",
            run_class="provider",
            pod_class="provider",
            runpod_profile="provider",
            unavailable_posture="deterministic_stub_latents_only_until_real_backbone_truth_receipts_exist",
            source_backlog_ids=["vision_backbone_stub_replacement"],
            surface_roles=["lower-WM producer", "trainer/runtime lane", "receipt substrate"],
            source_files=[
                "src/vision/backbone_stub.py",
                "scripts/train_vision_backbone.py",
                "src/datasets/vision_dataset.py",
            ],
            command_templates=[
                "python3 scripts/train_vision_backbone.py --provider dino_siglip --emit-provider-truth --output-dir artifacts/economic_world_model/provider_runs/{run_id}/vision_backbone",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 -m pytest -q tests/test_perception_grounding_compiler.py tests/test_perception_seam_training.py",
            ],
            expected_receipts=[
                "vision_backbone_provider_truth_v1",
                "vision_backbone_latent_export_receipt_v1",
                "vision_backbone_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/vision_backbone_provider_truth_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/vision_backbone_unavailable_receipt_v1.json",
            ],
            blocker_codes=[
                "vision_backbone_weights_missing",
                "real_frame_corpus_missing",
                "gpu_provider_execution_not_run",
            ],
            required_prerequisites=[
                "DINOv2 or SigLIP weights cached",
                "real frame corpus",
                "CUDA-capable provider host",
            ],
        ),
        ProviderLedgerSpec(
            provider_key="vjepa2_sim_synth_predictive",
            provider_family="vjepa2",
            provider_label="V-JEPA2 sim/synth predictive provider",
            owner_wm="sim_synth_physics",
            subsystem="predictive_state",
            run_class="provider",
            pod_class="provider",
            runpod_profile="provider",
            unavailable_posture="predictive_provider_unavailable_until_upstream_runtime_weights_and_receipts_exist",
            source_backlog_ids=["vjepa2_sim_synth_predictive_provider"],
            surface_roles=["lower-WM producer", "provider/hardware adapter", "trainer/runtime lane"],
            source_files=[
                "src/world_model/sim_synth_physics/compiler.py",
                "src/world_model/sim_synth_physics/runtime.py",
                "src/orchestrator/coverage_loop.py",
            ],
            command_templates=[
                "python3 scripts/economic_world_model/run_vjepa2_sim_synth_provider.py --output-dir artifacts/economic_world_model/provider_runs/{run_id}/vjepa2_sim_synth",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_runtime_work_orders.py",
            ],
            expected_receipts=[
                "vjepa2_provider_truth_v1",
                "sim_synth_predictive_state_receipt_v1",
                "vjepa2_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/vjepa2_provider_truth_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/sim_synth_predictive_state_receipt_v1.json",
            ],
            blocker_codes=[
                "vjepa2_runtime_missing",
                "vjepa2_weights_missing",
                "gpu_provider_execution_not_run",
            ],
            required_prerequisites=[
                "facebookresearch/vjepa2 runtime",
                "local or cached V-JEPA2 weights",
                "CUDA-capable provider host",
            ],
        ),
        ProviderLedgerSpec(
            provider_key="vjepa2_perception_temporal",
            provider_family="vjepa2",
            provider_label="V-JEPA2 perception temporal grounding",
            owner_wm="perception_grounding",
            subsystem="temporal_grounding",
            run_class="provider",
            pod_class="provider",
            runpod_profile="provider",
            unavailable_posture="temporal_provider_unavailable_until_scene_persistence_receipts_exist",
            source_backlog_ids=["vjepa2_perception_temporal_grounding_provider"],
            surface_roles=["lower-WM producer", "receipt substrate", "trainer/runtime lane"],
            source_files=[
                "src/vision/backbone_stub.py",
                "src/vision/dataset_builder.py",
                "scripts/train_vision_backbone.py",
            ],
            command_templates=[
                "python3 scripts/economic_world_model/run_vjepa2_perception_provider.py --output-dir artifacts/economic_world_model/provider_runs/{run_id}/vjepa2_perception",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 -m pytest -q tests/test_dataset_bridges.py tests/test_perception_grounding_compiler.py",
            ],
            expected_receipts=[
                "vjepa2_temporal_provider_truth_v1",
                "scene_persistence_receipt_v1",
                "vjepa2_temporal_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/vjepa2_temporal_provider_truth_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/scene_persistence_receipt_v1.json",
            ],
            blocker_codes=[
                "vjepa2_runtime_missing",
                "vjepa2_weights_missing",
                "temporal_benchmark_missing",
                "gpu_provider_execution_not_run",
            ],
            required_prerequisites=[
                "facebookresearch/vjepa2 runtime",
                "temporal frame corpus",
                "CUDA-capable provider host",
            ],
        ),
        ProviderLedgerSpec(
            provider_key="openvla_semantic_teacher",
            provider_family="openvla",
            provider_label="OpenVLA semantic teacher",
            owner_wm="perception_grounding",
            subsystem="semantic_vla",
            run_class="provider",
            pod_class="provider",
            runpod_profile="provider",
            unavailable_posture="external_vla_advisory_unavailable_until_teacher_runtime_truth_exists",
            source_backlog_ids=["semantic_vla_placeholder_replacement"],
            surface_roles=["lower-WM producer", "provider/hardware adapter", "receipt substrate"],
            source_files=[
                "src/vla/semantic_vla.py",
                "src/vla/recap_dataset_builder.py",
                "scripts/train_vla_recap_offline.py",
                "scripts/run_vla_on_episode.py",
            ],
            command_templates=[
                "python3 scripts/run_vla_on_episode.py --episode {episode_json} --output artifacts/economic_world_model/provider_runs/{run_id}/openvla_teacher_trace_v1.json",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 -m pytest -q tests/test_semantic_runtime_learning.py tests/test_vla_semantic_evidence.py tests/test_vla_backend_policy.py",
            ],
            expected_receipts=[
                "openvla_teacher_trace_v1",
                "vla_provider_truth_v1",
                "openvla_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/openvla_teacher_trace_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/vla_provider_truth_v1.json",
            ],
            blocker_codes=[
                "openvla_weights_missing",
                "teacher_labeled_corpus_missing",
                "provider_runtime_not_verified",
            ],
            required_prerequisites=[
                "OpenVLA model path mounted",
                "teacher-labeled corpus",
                "CUDA-capable provider host",
            ],
        ),
        ProviderLedgerSpec(
            provider_key="isaac_unitree_runtime",
            provider_family="isaac_unitree",
            provider_label="Isaac / Unitree runtime execution",
            owner_wm="sim_synth_physics_and_embodiment_actuation",
            subsystem="isaac_unitree_backend",
            run_class="loop",
            pod_class="loop",
            runpod_profile="loop",
            unavailable_posture="shadow_contract_only_until_isaac_unitree_runtime_manifest_and_receipts_exist",
            source_backlog_ids=["isaac_unitree_backend_execution"],
            surface_roles=["lower-WM producer", "provider/hardware adapter", "curriculum/regression source"],
            source_files=[
                "src/envs/physics/isaac_backend.py",
                "src/world_model/sim_synth_physics/adapters/backend_isaac.py",
                "src/world_model/sim_synth_physics/backend_bindings.py",
                "src/world_model/humanoid_readiness/",
            ],
            command_templates=[
                "python3 scripts/run_isaac_unitree_executable_adapter.py --backend isaac --output-dir artifacts/economic_world_model/provider_runs/{run_id}/isaac_unitree",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 -m pytest -q tests/test_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_pack.py tests/test_humanoid_phase4_unitree_runtime_evidence_bridge.py",
            ],
            expected_receipts=[
                "isaac_unitree_runtime_manifest_v1",
                "backend_execution_receipt_v1",
                "unitree_runtime_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/isaac_unitree_runtime_manifest_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/unitree_runtime_unavailable_receipt_v1.json",
            ],
            blocker_codes=[
                "isaac_runtime_missing",
                "unitree_assets_or_policy_missing",
                "hardware_or_honest_sim_not_executed",
            ],
            required_prerequisites=[
                "Linux host with NVIDIA GPU",
                "Isaac Lab/Sim runtime",
                "Unitree assets and policies",
                "ROS2/SDK2 bridge runtime for command echo",
            ],
        ),
        ProviderLedgerSpec(
            provider_key="holosoma_runtime",
            provider_family="holosoma",
            provider_label="Holosoma whole-body runtime",
            owner_wm="embodiment_actuation",
            subsystem="holosoma_backend",
            run_class="loop",
            pod_class="loop",
            runpod_profile="loop",
            unavailable_posture="shadow_work_order_only_until_holosoma_runtime_and_motion_data_exist",
            source_backlog_ids=["holosoma_runtime_execution"],
            surface_roles=["lower-WM producer", "provider/hardware adapter", "trainer/runtime lane"],
            source_files=[
                "src/motor_backend/holosoma_backend.py",
                "src/world_model/sim_synth_physics/adapters/backend_holosoma.py",
                "src/world_model/sim_synth_physics/shadow_execution.py",
            ],
            command_templates=[
                "python3 scripts/economic_world_model/run_holosoma_runtime_provider.py --output-dir artifacts/economic_world_model/provider_runs/{run_id}/holosoma",
                _TEMPLATE_GUARD,
            ],
            local_verification_commands=[
                "python3 -m pytest -q tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_physics_world_model.py",
            ],
            expected_receipts=[
                "holosoma_runtime_receipt_v1",
                "whole_body_rollout_receipt_v1",
                "holosoma_unavailable_receipt_v1",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/holosoma_runtime_receipt_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/holosoma_unavailable_receipt_v1.json",
            ],
            blocker_codes=[
                "holosoma_runtime_missing",
                "motion_data_or_policy_missing",
                "gpu_provider_execution_not_run",
            ],
            required_prerequisites=[
                "Holosoma runtime installed",
                "whole-body motion data",
                "runtime policy/checkpoint",
                "CUDA-capable provider host",
            ],
        ),
    ]


def _prerequisite_status() -> dict[str, bool]:
    return {
        "runpodctl_on_path": shutil.which("runpodctl") is not None,
        "RUNPOD_API_KEY_set": bool(os.environ.get("RUNPOD_API_KEY")),
        "RUNPOD_VOLUME_ID_set": bool(os.environ.get("RUNPOD_VOLUME_ID")),
        "cuda_visible_devices_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
    }


def _manifest_stub(entry: ProviderLedgerSpec) -> dict[str, Any]:
    digest = sha256_json(
        {
            "provider_key": entry.provider_key,
            "version": PROVIDER_BRINGUP_LEDGER_ENTRY_VERSION,
        }
    )[:6]
    return {
        "run_id": f"runpod-19700101-000000-{digest}",
        "mode": "runpod",
        "pod_class": entry.pod_class,
        "run_class": entry.run_class,
        "epistemic_status": "provider_readiness_template_only",
        "commit_sha": "template_commit_sha_required_at_launch",
        "branch": "template_branch_required_at_launch",
        "task": f"[TEMPLATE ONLY] {entry.provider_label}",
        "wm": entry.owner_wm,
        "subsystem": entry.subsystem,
        "blocker": ",".join(entry.blocker_codes),
        "commands": list(entry.command_templates),
        "artifact_paths": list(entry.artifact_paths),
        "status": "pending",
        "pod_id": None,
        "started_at": None,
        "finished_at": None,
        "cost_snapshot": None,
        "provider_executed": False,
        "promotion_eligible": False,
        "rollback_notes": "Template only. Do not launch until guard commands are replaced and a real manifest is recorded.",
    }


def _build_entry(
    *,
    spec: ProviderLedgerSpec,
    backlog_rows: Mapping[str, Mapping[str, Any]],
) -> ProviderBringupLedgerEntry:
    source_rows = [_mapping(backlog_rows[key]) for key in spec.source_backlog_ids if key in backlog_rows]
    blockers = sorted({*spec.blocker_codes, "provider_execution_not_run"})
    missing = list(spec.required_prerequisites)
    payload = {
        "provider_key": spec.provider_key,
        "owner_wm": spec.owner_wm,
        "runpod_profile": spec.runpod_profile,
        "version": PROVIDER_BRINGUP_LEDGER_ENTRY_VERSION,
    }
    return ProviderBringupLedgerEntry(
        ledger_entry_id=_stable_id("provider_bringup_ledger_entry", payload),
        provider_key=spec.provider_key,
        provider_family=spec.provider_family,
        provider_label=spec.provider_label,
        owner_wm=spec.owner_wm,
        subsystem=spec.subsystem,
        run_class=spec.run_class,
        pod_class=spec.pod_class,
        runpod_profile=spec.runpod_profile,
        status="blocked_template_only",
        unavailable_posture=spec.unavailable_posture,
        launch_allowed=False,
        provider_bringup_ready=False,
        local_verification_available=bool(spec.local_verification_commands),
        command_templates=list(spec.command_templates),
        local_verification_commands=list(spec.local_verification_commands),
        expected_receipts=list(spec.expected_receipts),
        artifact_paths=list(spec.artifact_paths),
        blocker_codes=blockers,
        required_prerequisites=list(spec.required_prerequisites),
        missing_prerequisites=missing,
        source_backlog_ids=list(spec.source_backlog_ids),
        source_files=list(spec.source_files),
        surface_roles=list(spec.surface_roles),
        source_backlog_rows=source_rows,
        manifest_stub=_manifest_stub(spec),
        denied_gates=_denied_gates(),
        metadata={
            "template_only": True,
            "notes": spec.notes,
            "source_backlog_row_count": len(source_rows),
        },
    )


def build_provider_bringup_ledger(
    *,
    foundation_backlog_path: str | Path = "scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json",
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[ProviderBringupLedgerReport, list[ProviderBringupLedgerEntry]]:
    backlog_rows = _backlog_by_id(foundation_backlog_path)
    entries = [
        _build_entry(spec=spec, backlog_rows=backlog_rows)
        for spec in _provider_specs()
    ]
    families = sorted({entry.provider_family for entry in entries})
    missing_families = sorted(set(REQUIRED_PROVIDER_FAMILIES) - set(families))
    blocker_codes = sorted({code for entry in entries for code in entry.blocker_codes})
    prerequisite_status = _prerequisite_status()
    all_fail_closed = (
        bool(entries)
        and not any(entry.launch_allowed for entry in entries)
        and not any(entry.provider_bringup_ready for entry in entries)
        and not any(entry.provider_executed for entry in entries)
        and not any(entry.promotion_eligible for entry in entries)
        and all(not any(entry.denied_gates.values()) for entry in entries)
    )
    payload = {
        "entry_ids": [entry.ledger_entry_id for entry in entries],
        "families": families,
        "version": PROVIDER_BRINGUP_LEDGER_REPORT_VERSION,
    }
    report = ProviderBringupLedgerReport(
        report_id=_stable_id("provider_bringup_ledger_report", payload),
        status="ok" if all_fail_closed and not missing_families else "blocked",
        entry_count=len(entries),
        required_family_count=len(REQUIRED_PROVIDER_FAMILIES),
        covered_required_family_count=len(set(REQUIRED_PROVIDER_FAMILIES) & set(families)),
        launch_allowed_count=sum(1 for entry in entries if entry.launch_allowed),
        provider_bringup_ready_count=sum(
            1 for entry in entries if entry.provider_bringup_ready
        ),
        local_verification_available_count=sum(
            1 for entry in entries if entry.local_verification_available
        ),
        runpod_template_count=len(entries),
        missing_prerequisite_count=sum(
            len(entry.missing_prerequisites) for entry in entries
        ),
        all_entries_fail_closed=all_fail_closed,
        denied_gates=_denied_gates(),
        covered_required_families=families,
        missing_required_families=missing_families,
        blocker_codes=blocker_codes,
        prerequisite_status=prerequisite_status,
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "source_foundation_backlog_path": str(foundation_backlog_path),
            "foundation_backlog_row_count": len(backlog_rows),
            **_mapping(metadata),
        },
    )
    return report, entries


def save_provider_bringup_ledger(
    output_dir: str | Path,
    *,
    report: ProviderBringupLedgerReport,
    entries: list[ProviderBringupLedgerEntry],
) -> dict[str, str]:
    output = Path(output_dir)
    manifest_dir = output / "manifest_templates"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_path": output / "provider_bringup_ledger_report_v1.json",
        "entries_path": output / "provider_bringup_ledger_entries_v1.jsonl",
        "manifest_template_dir": manifest_dir,
    }
    paths["report_path"].write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["entries_path"].write_text(
        "\n".join(json.dumps(entry.to_dict(), sort_keys=True) for entry in entries)
        + "\n",
        encoding="utf-8",
    )
    for entry in entries:
        (manifest_dir / f"{entry.provider_key}.manifest_template.json").write_text(
            json.dumps(entry.manifest_stub, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return {key: str(path) for key, path in paths.items()}


def build_and_save_provider_bringup_ledger(
    *,
    output_dir: str | Path,
    foundation_backlog_path: str | Path = "scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    refs = {
        "report_path": str(output / "provider_bringup_ledger_report_v1.json"),
        "entries_path": str(output / "provider_bringup_ledger_entries_v1.jsonl"),
        "manifest_template_dir": str(output / "manifest_templates"),
    }
    report, entries = build_provider_bringup_ledger(
        foundation_backlog_path=foundation_backlog_path,
        artifact_refs=refs,
        metadata=metadata,
    )
    saved = save_provider_bringup_ledger(output, report=report, entries=entries)
    payload = report.to_dict()
    payload["entries"] = [entry.to_dict() for entry in entries]
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved}
    return payload


def load_provider_bringup_ledger_report(
    path: str | Path,
) -> ProviderBringupLedgerReport:
    return ProviderBringupLedgerReport.from_dict(_load_json(path))


def load_provider_bringup_ledger_entries(
    path: str | Path,
) -> list[ProviderBringupLedgerEntry]:
    target = Path(path)
    return [
        ProviderBringupLedgerEntry.from_dict(json.loads(line))
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_provider_bringup_ledger(
    *,
    report: ProviderBringupLedgerReport,
    entries: list[ProviderBringupLedgerEntry],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    families = {entry.provider_family for entry in entries}
    for family in REQUIRED_PROVIDER_FAMILIES:
        if family not in families:
            errors.append(f"missing_required_family:{family}")
    for entry in entries:
        if entry.launch_allowed:
            errors.append(f"launch_allowed:{entry.provider_key}")
        if entry.provider_bringup_ready:
            errors.append(f"provider_bringup_ready:{entry.provider_key}")
        if entry.provider_executed or entry.runpod_launched:
            errors.append(f"execution_claimed:{entry.provider_key}")
        if entry.promotion_eligible:
            errors.append(f"promotion_eligible:{entry.provider_key}")
        if any(entry.denied_gates.values()):
            errors.append(f"denied_gate_true:{entry.provider_key}")
        if not entry.expected_receipts:
            errors.append(f"missing_expected_receipts:{entry.provider_key}")
        if not entry.command_templates or not any(
            "TEMPLATE_ONLY_PROVIDER_LEDGER" in command
            for command in entry.command_templates
        ):
            errors.append(f"missing_template_guard:{entry.provider_key}")
        if not entry.unavailable_posture:
            errors.append(f"missing_unavailable_posture:{entry.provider_key}")
        if not entry.surface_roles:
            warnings.append(f"missing_surface_roles:{entry.provider_key}")
    if report.launch_allowed_count:
        errors.append("report_launch_allowed_count_nonzero")
    if not report.all_entries_fail_closed:
        errors.append("report_not_fail_closed")
    return {
        "status": "ok" if not errors else "blocked",
        "safe_for_template_storage": not errors,
        "safe_for_launch": False,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "covered_required_families": sorted(families),
        "missing_required_families": sorted(set(REQUIRED_PROVIDER_FAMILIES) - families),
    }
