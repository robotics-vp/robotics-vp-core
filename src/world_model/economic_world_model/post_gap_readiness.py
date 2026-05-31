"""Post-gap readiness manifests for Economic WM GPU/data/hardware bring-up.

This module covers the CPU-capable planning work that should exist before a
GPU, provider, large external corpus, or humanoid hardware window opens. It
emits typed plans and fail-closed receipts only; it does not download external
datasets, launch providers, run GPU training, or grant Phase 7 authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

POST_GAP_READINESS_REPORT_VERSION = "economic_wm_post_gap_readiness_report_v1"
GPU_DAY_ONE_RUNBOOK_VERSION = "gpu_day_one_runbook_v1"
EXTERNAL_DATASET_CORPUS_PLAN_VERSION = "external_dataset_corpus_plan_v1"
CORPUS_PREP_ARTIFACT_PLAN_VERSION = "corpus_prep_artifact_plan_v1"
BENCHMARK_GATE_SPEC_VERSION = "post_gap_benchmark_gate_spec_v1"
PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION = "provider_runtime_packaging_spec_v1"
PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION = (
    "perception_embodiment_replay_loop_spec_v1"
)
G1_R1_PURCHASE_READINESS_VERSION = "g1_r1_purchase_readiness_v1"
EVIDENCE_HYGIENE_SPEC_VERSION = "post_gap_evidence_hygiene_spec_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class GPUDayOneRunbook:
    """One guarded runbook for the first GPU/provider windows."""

    runbook_id: str
    name: str
    plane: str
    pod_class: str
    horizon: str
    status: str
    launch_allowed: bool
    provider_bringup_ready: bool
    gpu_training_ready: bool
    estimated_cost_usd: dict[str, float] = field(default_factory=dict)
    estimated_wallclock: str = ""
    provider_bringup_commands: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    failure_receipts: list[str] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    artifact_storage_paths: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = GPU_DAY_ONE_RUNBOOK_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "version": self.version,
            "name": self.name,
            "plane": self.plane,
            "pod_class": self.pod_class,
            "horizon": self.horizon,
            "status": self.status,
            "launch_allowed": bool(self.launch_allowed),
            "provider_bringup_ready": bool(self.provider_bringup_ready),
            "gpu_training_ready": bool(self.gpu_training_ready),
            "estimated_cost_usd": dict(self.estimated_cost_usd),
            "estimated_wallclock": self.estimated_wallclock,
            "provider_bringup_commands": list(self.provider_bringup_commands),
            "verification_commands": list(self.verification_commands),
            "expected_artifacts": list(self.expected_artifacts),
            "failure_receipts": list(self.failure_receipts),
            "checkpoint_paths": list(self.checkpoint_paths),
            "artifact_storage_paths": list(self.artifact_storage_paths),
            "stop_conditions": list(self.stop_conditions),
            "blocked_by": list(self.blocked_by),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GPUDayOneRunbook":
        return cls(
            runbook_id=str(payload.get("runbook_id", "")),
            name=str(payload.get("name", "")),
            plane=str(payload.get("plane", "")),
            pod_class=str(payload.get("pod_class", "")),
            horizon=str(payload.get("horizon", "")),
            status=str(payload.get("status", "")),
            launch_allowed=bool(payload.get("launch_allowed", False)),
            provider_bringup_ready=bool(payload.get("provider_bringup_ready", False)),
            gpu_training_ready=bool(payload.get("gpu_training_ready", False)),
            estimated_cost_usd={
                str(key): float(value)
                for key, value in dict(payload.get("estimated_cost_usd", {}) or {}).items()
            },
            estimated_wallclock=str(payload.get("estimated_wallclock", "")),
            provider_bringup_commands=_strings(
                payload.get("provider_bringup_commands")
            ),
            verification_commands=_strings(payload.get("verification_commands")),
            expected_artifacts=_strings(payload.get("expected_artifacts")),
            failure_receipts=_strings(payload.get("failure_receipts")),
            checkpoint_paths=_strings(payload.get("checkpoint_paths")),
            artifact_storage_paths=_strings(payload.get("artifact_storage_paths")),
            stop_conditions=_strings(payload.get("stop_conditions")),
            blocked_by=_strings(payload.get("blocked_by")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", GPU_DAY_ONE_RUNBOOK_VERSION)),
        )


@dataclass(frozen=True)
class ExternalDatasetCorpusPlan:
    """External or local corpus candidate mapped into repo-native contracts."""

    dataset_id: str
    name: str
    source_kind: str
    priority: str
    bring_in_status: str
    source_url: str
    access_method: str
    expected_scale: str
    modalities: list[str] = field(default_factory=list)
    embodiment_fit: list[str] = field(default_factory=list)
    repo_schema_targets: list[str] = field(default_factory=list)
    normalization_steps: list[str] = field(default_factory=list)
    split_manifest_plan: list[str] = field(default_factory=list)
    replay_indexer_plan: list[str] = field(default_factory=list)
    data_quality_receipt_plan: list[str] = field(default_factory=list)
    label_gap_ledger_plan: list[str] = field(default_factory=list)
    governance_label_spec: list[str] = field(default_factory=list)
    transport_meta_node_plan: list[str] = field(default_factory=list)
    import_blockers: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    download_executed: bool = False
    ready_for_training: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_DATASET_CORPUS_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "name": self.name,
            "source_kind": self.source_kind,
            "priority": self.priority,
            "bring_in_status": self.bring_in_status,
            "source_url": self.source_url,
            "access_method": self.access_method,
            "expected_scale": self.expected_scale,
            "modalities": list(self.modalities),
            "embodiment_fit": list(self.embodiment_fit),
            "repo_schema_targets": list(self.repo_schema_targets),
            "normalization_steps": list(self.normalization_steps),
            "split_manifest_plan": list(self.split_manifest_plan),
            "replay_indexer_plan": list(self.replay_indexer_plan),
            "data_quality_receipt_plan": list(self.data_quality_receipt_plan),
            "label_gap_ledger_plan": list(self.label_gap_ledger_plan),
            "governance_label_spec": list(self.governance_label_spec),
            "transport_meta_node_plan": list(self.transport_meta_node_plan),
            "import_blockers": list(self.import_blockers),
            "risk_notes": list(self.risk_notes),
            "download_executed": bool(self.download_executed),
            "ready_for_training": bool(self.ready_for_training),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalDatasetCorpusPlan":
        return cls(
            dataset_id=str(payload.get("dataset_id", "")),
            name=str(payload.get("name", "")),
            source_kind=str(payload.get("source_kind", "")),
            priority=str(payload.get("priority", "")),
            bring_in_status=str(payload.get("bring_in_status", "")),
            source_url=str(payload.get("source_url", "")),
            access_method=str(payload.get("access_method", "")),
            expected_scale=str(payload.get("expected_scale", "")),
            modalities=_strings(payload.get("modalities")),
            embodiment_fit=_strings(payload.get("embodiment_fit")),
            repo_schema_targets=_strings(payload.get("repo_schema_targets")),
            normalization_steps=_strings(payload.get("normalization_steps")),
            split_manifest_plan=_strings(payload.get("split_manifest_plan")),
            replay_indexer_plan=_strings(payload.get("replay_indexer_plan")),
            data_quality_receipt_plan=_strings(
                payload.get("data_quality_receipt_plan")
            ),
            label_gap_ledger_plan=_strings(payload.get("label_gap_ledger_plan")),
            governance_label_spec=_strings(payload.get("governance_label_spec")),
            transport_meta_node_plan=_strings(payload.get("transport_meta_node_plan")),
            import_blockers=_strings(payload.get("import_blockers")),
            risk_notes=_strings(payload.get("risk_notes")),
            download_executed=bool(payload.get("download_executed", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", EXTERNAL_DATASET_CORPUS_PLAN_VERSION)),
        )


@dataclass(frozen=True)
class CorpusPrepArtifactPlan:
    """Planned artifact that turns a corpus candidate into repo-native rows."""

    prep_id: str
    dataset_id: str
    artifact_kind: str
    status: str
    output_template: str
    command_template: str
    required_fields: list[str] = field(default_factory=list)
    acceptance_checks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    launch_allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = CORPUS_PREP_ARTIFACT_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "prep_id": self.prep_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "output_template": self.output_template,
            "command_template": self.command_template,
            "required_fields": list(self.required_fields),
            "acceptance_checks": list(self.acceptance_checks),
            "blockers": list(self.blockers),
            "launch_allowed": bool(self.launch_allowed),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusPrepArtifactPlan":
        return cls(
            prep_id=str(payload.get("prep_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            artifact_kind=str(payload.get("artifact_kind", "")),
            status=str(payload.get("status", "")),
            output_template=str(payload.get("output_template", "")),
            command_template=str(payload.get("command_template", "")),
            required_fields=_strings(payload.get("required_fields")),
            acceptance_checks=_strings(payload.get("acceptance_checks")),
            blockers=_strings(payload.get("blockers")),
            launch_allowed=bool(payload.get("launch_allowed", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", CORPUS_PREP_ARTIFACT_PLAN_VERSION)),
        )


@dataclass(frozen=True)
class BenchmarkGateSpec:
    """Fail-closed benchmark gate that must pass before model promotion."""

    gate_id: str
    gate_key: str
    surface: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    required_artifacts: list[str] = field(default_factory=list)
    fail_closed_reasons: list[str] = field(default_factory=list)
    promotion_gate: bool = True
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = BENCHMARK_GATE_SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "version": self.version,
            "gate_key": self.gate_key,
            "surface": self.surface,
            "status": self.status,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "required_artifacts": list(self.required_artifacts),
            "fail_closed_reasons": list(self.fail_closed_reasons),
            "promotion_gate": bool(self.promotion_gate),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkGateSpec":
        return cls(
            gate_id=str(payload.get("gate_id", "")),
            gate_key=str(payload.get("gate_key", "")),
            surface=str(payload.get("surface", "")),
            status=str(payload.get("status", "")),
            metrics={
                str(key): float(value)
                for key, value in dict(payload.get("metrics", {}) or {}).items()
            },
            thresholds={
                str(key): float(value)
                for key, value in dict(payload.get("thresholds", {}) or {}).items()
            },
            required_artifacts=_strings(payload.get("required_artifacts")),
            fail_closed_reasons=_strings(payload.get("fail_closed_reasons")),
            promotion_gate=bool(payload.get("promotion_gate", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", BENCHMARK_GATE_SPEC_VERSION)),
        )


@dataclass(frozen=True)
class ReadinessSpec:
    """Generic typed readiness row for non-dataset post-gap lanes."""

    spec_id: str
    version: str
    lane: str
    key: str
    title: str
    status: str
    ready: bool
    launch_allowed: bool
    required_artifacts: list[str] = field(default_factory=list)
    commands_or_steps: list[str] = field(default_factory=list)
    receipts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "version": self.version,
            "lane": self.lane,
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "ready": bool(self.ready),
            "launch_allowed": bool(self.launch_allowed),
            "required_artifacts": list(self.required_artifacts),
            "commands_or_steps": list(self.commands_or_steps),
            "receipts": list(self.receipts),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReadinessSpec":
        return cls(
            spec_id=str(payload.get("spec_id", "")),
            version=str(payload.get("version", "")),
            lane=str(payload.get("lane", "")),
            key=str(payload.get("key", "")),
            title=str(payload.get("title", "")),
            status=str(payload.get("status", "")),
            ready=bool(payload.get("ready", False)),
            launch_allowed=bool(payload.get("launch_allowed", False)),
            required_artifacts=_strings(payload.get("required_artifacts")),
            commands_or_steps=_strings(payload.get("commands_or_steps")),
            receipts=_strings(payload.get("receipts")),
            blockers=_strings(payload.get("blockers")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class PostGapReadinessReport:
    """Aggregate report tying every post-gap lane to artifacts."""

    report_id: str
    status: str
    all_post_gap_items_manifested: bool
    gpu_day_one_runbook_count: int
    external_dataset_count: int
    corpus_prep_artifact_count: int
    benchmark_gate_count: int
    provider_runtime_packaging_count: int
    replay_loop_count: int
    g1_r1_purchase_readiness_count: int
    evidence_hygiene_count: int
    launch_authority_granted: bool
    provider_executed: bool
    gpu_training_executed: bool
    external_download_executed: bool
    phase7_constraint_honored: bool
    promotion_eligible: bool
    ready_for_august_gpu_window: bool
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = POST_GAP_READINESS_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "all_post_gap_items_manifested": bool(self.all_post_gap_items_manifested),
            "gpu_day_one_runbook_count": int(self.gpu_day_one_runbook_count),
            "external_dataset_count": int(self.external_dataset_count),
            "corpus_prep_artifact_count": int(self.corpus_prep_artifact_count),
            "benchmark_gate_count": int(self.benchmark_gate_count),
            "provider_runtime_packaging_count": int(
                self.provider_runtime_packaging_count
            ),
            "replay_loop_count": int(self.replay_loop_count),
            "g1_r1_purchase_readiness_count": int(
                self.g1_r1_purchase_readiness_count
            ),
            "evidence_hygiene_count": int(self.evidence_hygiene_count),
            "launch_authority_granted": bool(self.launch_authority_granted),
            "provider_executed": bool(self.provider_executed),
            "gpu_training_executed": bool(self.gpu_training_executed),
            "external_download_executed": bool(self.external_download_executed),
            "phase7_constraint_honored": bool(self.phase7_constraint_honored),
            "promotion_eligible": bool(self.promotion_eligible),
            "ready_for_august_gpu_window": bool(self.ready_for_august_gpu_window),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PostGapReadinessReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "")),
            all_post_gap_items_manifested=bool(
                payload.get("all_post_gap_items_manifested", False)
            ),
            gpu_day_one_runbook_count=int(
                payload.get("gpu_day_one_runbook_count", 0) or 0
            ),
            external_dataset_count=int(payload.get("external_dataset_count", 0) or 0),
            corpus_prep_artifact_count=int(
                payload.get("corpus_prep_artifact_count", 0) or 0
            ),
            benchmark_gate_count=int(payload.get("benchmark_gate_count", 0) or 0),
            provider_runtime_packaging_count=int(
                payload.get("provider_runtime_packaging_count", 0) or 0
            ),
            replay_loop_count=int(payload.get("replay_loop_count", 0) or 0),
            g1_r1_purchase_readiness_count=int(
                payload.get("g1_r1_purchase_readiness_count", 0) or 0
            ),
            evidence_hygiene_count=int(payload.get("evidence_hygiene_count", 0) or 0),
            launch_authority_granted=bool(
                payload.get("launch_authority_granted", False)
            ),
            provider_executed=bool(payload.get("provider_executed", False)),
            gpu_training_executed=bool(payload.get("gpu_training_executed", False)),
            external_download_executed=bool(
                payload.get("external_download_executed", False)
            ),
            phase7_constraint_honored=bool(
                payload.get("phase7_constraint_honored", False)
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            ready_for_august_gpu_window=bool(
                payload.get("ready_for_august_gpu_window", False)
            ),
            remaining_blockers=_strings(payload.get("remaining_blockers")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", POST_GAP_READINESS_REPORT_VERSION)),
        )


def _runbook(
    *,
    name: str,
    plane: str,
    pod_class: str,
    horizon: str,
    estimated_cost_usd: Mapping[str, float],
    estimated_wallclock: str,
    provider_bringup_commands: Sequence[str],
    verification_commands: Sequence[str],
    expected_artifacts: Sequence[str],
    failure_receipts: Sequence[str],
    checkpoint_paths: Sequence[str],
    artifact_storage_paths: Sequence[str],
    stop_conditions: Sequence[str],
    blocked_by: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> GPUDayOneRunbook:
    payload = {
        "name": name,
        "plane": plane,
        "pod_class": pod_class,
        "horizon": horizon,
    }
    return GPUDayOneRunbook(
        runbook_id=_stable_id("gpu_day_one_runbook", payload),
        name=name,
        plane=plane,
        pod_class=pod_class,
        horizon=horizon,
        status="template_ready_launch_blocked",
        launch_allowed=False,
        provider_bringup_ready=False,
        gpu_training_ready=False,
        estimated_cost_usd=dict(estimated_cost_usd),
        estimated_wallclock=estimated_wallclock,
        provider_bringup_commands=list(provider_bringup_commands),
        verification_commands=list(verification_commands),
        expected_artifacts=list(expected_artifacts),
        failure_receipts=list(failure_receipts),
        checkpoint_paths=list(checkpoint_paths),
        artifact_storage_paths=list(artifact_storage_paths),
        stop_conditions=list(stop_conditions),
        blocked_by=list(blocked_by),
        metadata={
            "template_only": True,
            "replace_guard_before_launch": True,
            **dict(metadata or {}),
        },
    )


def build_gpu_day_one_runbooks() -> list[GPUDayOneRunbook]:
    """Build guarded first-hour, first-day, and weekend GPU runbooks."""

    manifest_template = ".agent/runs/{run_id}/manifest.json"
    return [
        _runbook(
            name="RunPod provider proof-of-life, first hour",
            plane="runpod",
            pod_class="provider",
            horizon="first_1_hour",
            estimated_cost_usd={"min": 3.0, "max": 25.0},
            estimated_wallclock="45-60 minutes",
            provider_bringup_commands=[
                "python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook",
                "python3 scripts/economic_world_model/validate_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook_validation --runbook artifacts/economic_world_model/economic_wm_provider_runbook/economic_wm_provider_runbook_v1.json --manifest-template-dir artifacts/economic_world_model/economic_wm_provider_runbook/manifest_templates",
                "echo TEMPLATE_ONLY_PROVIDER_COMMAND && false",
            ],
            verification_commands=[
                "python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model -q",
                "python3 -m pytest -q tests/test_economic_wm_provider_runbook.py tests/test_economic_wm_provider_runbook_validation.py",
            ],
            expected_artifacts=[
                manifest_template,
                "artifacts/economic_world_model/provider_runs/{run_id}/provider_invocation_receipt_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/external_provider_truth_v1.json",
            ],
            failure_receipts=[
                "artifacts/economic_world_model/provider_runs/{run_id}/provider_unavailable_receipt_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/dependency_failure_receipt_v1.json",
            ],
            checkpoint_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/checkpoints/"
            ],
            artifact_storage_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/"
            ],
            stop_conditions=[
                "provider command requires replacing TEMPLATE_ONLY guard",
                "provider truth receipt is missing",
                "manifest status is not completed_or_failed_explicitly",
                "cost exceeds configured first-hour cap",
            ],
            blocked_by=[
                "provider credentials/runtime not mounted",
                "RunPod image not selected",
                "real non-stub provider command not approved",
            ],
        ),
        _runbook(
            name="RunPod Economic WM GPU shape run, first 8 hours",
            plane="runpod",
            pod_class="train",
            horizon="first_8_hours",
            estimated_cost_usd={"min": 20.0, "max": 180.0},
            estimated_wallclock="2-8 hours",
            provider_bringup_commands=[
                "python3 scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py --output-dir artifacts/economic_world_model/economic_wm_neural_architecture_manifest",
                "python3 scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py --output-dir artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl --no-compile-missing-refs",
                "echo TEMPLATE_ONLY_GPU_TRAINING_COMMAND && false",
            ],
            verification_commands=[
                "python3 -m pytest -q tests/test_economic_wm_lower_wm_consumption.py tests/test_economic_wm_neural_architecture_manifest.py",
            ],
            expected_artifacts=[
                manifest_template,
                "artifacts/economic_world_model/provider_runs/{run_id}/training_runtime_manifest_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/gpu_runtime_receipt_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/checkpoint_manifest_v1.json",
            ],
            failure_receipts=[
                "artifacts/economic_world_model/provider_runs/{run_id}/gpu_unavailable_receipt_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/oom_or_timeout_receipt_v1.json",
            ],
            checkpoint_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/checkpoints/economic_wm_shape/"
            ],
            artifact_storage_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/logs/",
                "artifacts/economic_world_model/provider_runs/{run_id}/metrics/",
            ],
            stop_conditions=[
                "lower-WM refs missing from corpus rows",
                "training manifest omits dataset digest",
                "checkpoint manifest not written atomically",
                "loss becomes NaN or benchmark gate is absent",
            ],
            blocked_by=[
                "GPU training script still guard-only",
                "large corpus not downloaded/normalized",
                "promotion benchmark evidence missing",
            ],
        ),
        _runbook(
            name="RunPod weekend corpus and benchmark candidate",
            plane="runpod",
            pod_class="train",
            horizon="first_weekend",
            estimated_cost_usd={"min": 80.0, "max": 650.0},
            estimated_wallclock="24-48 hours",
            provider_bringup_commands=[
                "python3 scripts/economic_world_model/materialize_economic_wm_training_rows.py --output-dir artifacts/economic_world_model/economic_wm_training_rows --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json",
                "python3 scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py --output-dir artifacts/economic_world_model/economic_wm_shadow_allocation_eval --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl",
                "echo TEMPLATE_ONLY_WEEKEND_BENCHMARK_COMMAND && false",
            ],
            verification_commands=[
                "python3 -m pytest -q tests/test_economic_wm_training_rows.py tests/test_economic_wm_shadow_allocation_eval.py",
            ],
            expected_artifacts=[
                manifest_template,
                "artifacts/economic_world_model/provider_runs/{run_id}/benchmark_gate_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/promotion_metric_report_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/benchmark_evidence_bundle_v1.json",
            ],
            failure_receipts=[
                "artifacts/economic_world_model/provider_runs/{run_id}/benchmark_not_ready_receipt_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/corpus_quality_failure_receipt_v1.json",
            ],
            checkpoint_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/checkpoints/economic_wm_weekend/"
            ],
            artifact_storage_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/benchmarks/",
                "artifacts/economic_world_model/provider_runs/{run_id}/receipts/",
            ],
            stop_conditions=[
                "benchmark gate fails or is computed from fixture-only rows",
                "quality receipts reject more than configured corpus fraction",
                "provider/runtime truth is missing",
                "estimated spend exceeds weekend cap",
            ],
            blocked_by=[
                "external corpus import not materialized",
                "promotion-grade benchmark thresholds not met",
                "GPU/provider execution not yet authorized",
            ],
        ),
        _runbook(
            name="Local Linux runtime preflight",
            plane="local_linux",
            pod_class="refactor",
            horizon="first_1_hour",
            estimated_cost_usd={"min": 0.0, "max": 0.0},
            estimated_wallclock="30-60 minutes",
            provider_bringup_commands=[
                "python3 scripts/economic_world_model/run_cpu_august_gap_tranche.py --sample-count 12 --timing-iterations 32 --mujoco-steps 5 --stress-steps 100",
                "python3 scripts/scan_phase1_runtime_layouts.py --output-path artifacts/economic_world_model/post_gap_readiness/local_linux_runtime_scan.json",
            ],
            verification_commands=[
                "python3 -m pytest -q tests/test_cpu_august_gap_tranche.py tests/test_scan_phase1_runtime_layouts.py",
            ],
            expected_artifacts=[
                manifest_template,
                "artifacts/economic_world_model/cpu_august_gap_execution/cpu_august_gap_execution_report_v1.json",
                "artifacts/economic_world_model/post_gap_readiness/local_linux_runtime_scan.json",
            ],
            failure_receipts=[
                "artifacts/economic_world_model/post_gap_readiness/local_linux_unavailable_receipt_v1.json"
            ],
            checkpoint_paths=[],
            artifact_storage_paths=[
                "artifacts/economic_world_model/post_gap_readiness/"
            ],
            stop_conditions=[
                "ROS2/colcon unavailable",
                "Unitree SDK2 build still blocked",
                "MuJoCo probe fails",
            ],
            blocked_by=["Linux host and ROS2/Unitree runtime not verified here"],
        ),
        _runbook(
            name="Codex cloud code-only readiness audit",
            plane="codex_cloud",
            pod_class="code",
            horizon="first_1_hour",
            estimated_cost_usd={"min": 0.0, "max": 30.0},
            estimated_wallclock="30-60 minutes",
            provider_bringup_commands=[
                "bash scripts/economic_world_model/run_nightly_codex_task.sh --queue-only",
                "python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md",
            ],
            verification_commands=[
                "python3 -m compileall src scripts/economic_world_model -q"
            ],
            expected_artifacts=[
                manifest_template,
                "artifacts/economic_world_model/nightly_audit_summary.json",
                "artifacts/economic_world_model/nightly_audit_summary.md",
            ],
            failure_receipts=[
                "artifacts/economic_world_model/post_gap_readiness/codex_cloud_unavailable_receipt_v1.json"
            ],
            checkpoint_paths=[],
            artifact_storage_paths=[
                "artifacts/economic_world_model/post_gap_readiness/"
            ],
            stop_conditions=[
                "Codex credentials absent",
                "audit recommends no safe task",
                "cloud runner cannot access repo state",
            ],
            blocked_by=["CODEX_API_KEY or app automation may be unavailable"],
        ),
    ]


def build_external_dataset_corpus_plans() -> list[ExternalDatasetCorpusPlan]:
    """Identify concrete external corpora to bring in through staged manifests."""

    common_targets = [
        "ReplayEpisodeRecord",
        "ReplayStepRecord",
        "ReplayWindowRecord",
        "EconomicWMReplayFeatureRow",
        "EconomicWMCanonicalConsumptionRow",
    ]
    common_quality = [
        "episode_count_nonzero",
        "action_schema_present_or_explicitly_perception_only",
        "timestamp_or_step_index_monotonic",
        "license_and_terms_reviewed",
        "source_digest_recorded",
    ]
    common_governance = [
        "false_veto_candidate",
        "false_allow_candidate",
        "human_override_required",
        "privacy_or_consent_review_required",
        "unsafe_contact_or_workspace_event",
    ]
    return [
        ExternalDatasetCorpusPlan(
            dataset_id="open_x_embodiment_oxe",
            name="Open X-Embodiment / RT-X data mixture",
            source_kind="external_public_robot_manipulation",
            priority="P0",
            bring_in_status="planned_manifest_only",
            source_url="https://robotics-transformer-x.github.io/",
            access_method="project data links and per-dataset citations",
            expected_scale="1M+ real robot trajectories across 22 embodiments; 60 pooled datasets",
            modalities=["rgb", "robot_action", "language", "multi_embodiment"],
            embodiment_fit=["broad_manipulation", "mobile_manipulator_fallback"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "preserve source dataset identity before any mixture-level row",
                "map gripper-frame 7D action fields into action_adapter_v2 metadata",
                "emit one datapack composition row per source dataset and skill family",
            ],
            split_manifest_plan=[
                "stratify by source dataset, embodiment, task verb, and scene family",
                "reserve cross-embodiment holdout split before any training run",
            ],
            replay_indexer_plan=[
                "index by source_dataset_id, embodiment_id, task_text, episode_id, step_idx"
            ],
            data_quality_receipt_plan=common_quality,
            label_gap_ledger_plan=[
                "missing force/contact state",
                "ambiguous language instruction",
                "action-frame mismatch",
            ],
            governance_label_spec=common_governance,
            transport_meta_node_plan=[
                "emit WM transport exporter rows grouped by source embodiment"
            ],
            import_blockers=[
                "large download/storage planning",
                "per-source license/citation review",
                "RLDS/TFDS conversion runtime",
            ],
            risk_notes=[
                "mixture is useful for breadth but cannot be treated as native Unitree control truth"
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="droid",
            name="DROID in-the-wild robot manipulation dataset",
            source_kind="external_public_robot_manipulation",
            priority="P0",
            bring_in_status="planned_manifest_only",
            source_url="https://droid-dataset.github.io/",
            access_method="official dataset/Hugging Face releases",
            expected_scale="76k demonstration trajectories / about 350 hours across hundreds of scenes",
            modalities=["rgb", "stereo", "robot_state", "action", "language"],
            embodiment_fit=["franka_tabletop", "real_world_scene_generalization"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "preserve camera calibration version in source_refs",
                "map successful episode language annotations into task_id and objective summary",
                "emit scene diversity and OOD tags into datapack composition rows",
            ],
            split_manifest_plan=[
                "hold out scenes and object instances, not only random episodes",
                "separate successful-language-annotated episodes from unlabeled/failed rows",
            ],
            replay_indexer_plan=[
                "index by collector/site, scene_id, task_id, camera_id, episode_id, step_idx"
            ],
            data_quality_receipt_plan=[
                *common_quality,
                "camera_calibration_version_recorded",
                "success_annotation_present_or_blocked",
            ],
            label_gap_ledger_plan=[
                "missing failure-mode labels",
                "OOD distractor labels",
                "calibration revision mismatch",
            ],
            governance_label_spec=common_governance,
            transport_meta_node_plan=[
                "create train/eval rows for allocation under scene and task diversity"
            ],
            import_blockers=[
                "large download/storage planning",
                "dataset terms review",
                "calibration/version normalization",
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="bridgedata_v2",
            name="BridgeData V2",
            source_kind="external_public_robot_manipulation",
            priority="P0",
            bring_in_status="planned_manifest_only",
            source_url="https://bridgedata-v2.github.io/",
            access_method="project page / public release",
            expected_scale="53,896 trajectories demonstrating 13 skills across 24 environments",
            modalities=["rgb", "robot_action", "goal_image", "language"],
            embodiment_fit=["kitchen_tabletop", "drawer_and_object_manipulation"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "map WidowX action/state fields into embodiment-specific source_refs",
                "tag kitchen environments and skill verbs for economic allocation rows",
            ],
            split_manifest_plan=[
                "hold out environments and goal-object combinations",
                "build small CPU fixture split before full import",
            ],
            replay_indexer_plan=[
                "index by environment_id, task_text, object_family, episode_id, step_idx"
            ],
            data_quality_receipt_plan=common_quality,
            label_gap_ledger_plan=[
                "non-Unitree embodiment gap",
                "goal image versus language target mismatch",
            ],
            governance_label_spec=common_governance,
            transport_meta_node_plan=[
                "map environment holdout outcomes into Economic WM shadow allocation rows"
            ],
            import_blockers=[
                "storage planning",
                "source format adapter verification",
                "license/citation review",
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="lerobot_hub_curated",
            name="Hugging Face LeRobot curated robotics datasets",
            source_kind="external_public_robotics_format",
            priority="P0",
            bring_in_status="planned_manifest_only",
            source_url="https://huggingface.co/lerobot",
            access_method="Hugging Face Hub / LeRobotDataset",
            expected_scale="variable; standardized Parquet plus MP4 datasets",
            modalities=["parquet_rows", "mp4_video", "robot_action", "task_metadata"],
            embodiment_fit=["small_ci_fixtures", "format_adapter_validation"],
            repo_schema_targets=[
                "ReplayEpisodeRecord",
                "ReplayStepRecord",
                "lerobot_bridge",
                "EconomicWMReplayFeatureRow",
            ],
            normalization_steps=[
                "select tiny permissive subsets first for CI fixture imports",
                "preserve info.json, stats.json, tasks.parquet, and episodes metadata",
                "round-trip through src.dataset_bridges.lerobot_bridge",
            ],
            split_manifest_plan=[
                "train/eval split from episode metadata with deterministic seed",
                "fixture split must stay small enough for CI",
            ],
            replay_indexer_plan=[
                "index by repo_id, task_index, episode_index, frame_index"
            ],
            data_quality_receipt_plan=[
                *common_quality,
                "LeRobot info/stats/task metadata present",
            ],
            label_gap_ledger_plan=[
                "community dataset license ambiguity",
                "missing task semantics",
            ],
            governance_label_spec=common_governance,
            transport_meta_node_plan=[
                "use as adapter proof before large OXE/DROID downloads"
            ],
            import_blockers=[
                "dataset-by-dataset license review",
                "avoid silently training on arbitrary community uploads",
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="robomind_v2",
            name="RoboMIND multi-embodiment manipulation dataset",
            source_kind="external_public_robot_manipulation",
            priority="P1",
            bring_in_status="planned_manifest_only",
            source_url="https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND",
            access_method="Hugging Face / ModelScope collection",
            expected_scale="107k real-world trajectories across 479 tasks and multiple embodiments",
            modalities=["rgb", "robot_action", "language", "multi_embodiment"],
            embodiment_fit=["humanoid_relevance", "dual_arm_and_single_arm_transfer"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "split Tien Kung humanoid rows from Franka/UR/AgileX rows",
                "tag task categories and object classes as datapack composition fields",
            ],
            split_manifest_plan=[
                "hold out task categories and object classes",
                "create humanoid-only shadow eval slice",
            ],
            replay_indexer_plan=[
                "index by embodiment, task_category, object_class, trajectory_id, step_idx"
            ],
            data_quality_receipt_plan=common_quality,
            label_gap_ledger_plan=[
                "source terms and access path review",
                "humanoid embodiment does not equal Unitree G1/R1 control truth",
            ],
            governance_label_spec=common_governance,
            transport_meta_node_plan=[
                "use as humanoid-transfer corpus only after source review"
            ],
            import_blockers=[
                "license/source review",
                "download/storage planning",
                "format adapter not yet executed",
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="rh20t",
            name="RH20T contact-rich multimodal manipulation dataset",
            source_kind="external_public_robot_manipulation",
            priority="P1",
            bring_in_status="planned_manifest_only_sensitive",
            source_url="https://rh20t.github.io/",
            access_method="official download links and parsing API",
            expected_scale="110k+ contact-rich sequences; resized release still multi-TB",
            modalities=[
                "rgb",
                "rgbd",
                "force",
                "audio",
                "robot_action",
                "human_demonstration_video",
            ],
            embodiment_fit=["contact_rich_manipulation", "force_audio_quality_checks"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "keep human demonstration video isolated from robot-action rows",
                "emit force/audio availability into lower-WM canonical refs",
                "require privacy-sensitive source handling receipts",
            ],
            split_manifest_plan=[
                "split by task family, camera configuration, and modality completeness"
            ],
            replay_indexer_plan=[
                "index by task_id, config_id, modality_set, sequence_id, step_idx"
            ],
            data_quality_receipt_plan=[
                *common_quality,
                "privacy_sensitive_media_handling_recorded",
                "force_audio_sync_quality_recorded",
            ],
            label_gap_ledger_plan=[
                "face/voice privacy handling",
                "compressed depth quality gap",
                "human demo versus robot action alignment",
            ],
            governance_label_spec=[
                *common_governance,
                "privacy_sensitive_media_do_not_export",
            ],
            transport_meta_node_plan=[
                "use force/audio availability as lower-WM maturity signal"
            ],
            import_blockers=[
                "multi-TB storage",
                "privacy/consent policy review",
                "download bandwidth",
            ],
            risk_notes=[
                "official page warns about volunteer faces/voices; keep raw media gated"
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="ego4d_ego_exo4d",
            name="Ego4D / Ego-Exo4D perception-only video corpora",
            source_kind="external_public_perception_video",
            priority="P2",
            bring_in_status="planned_manifest_only_perception_only",
            source_url="https://ego4d-data.org/docs/",
            access_method="official dataset portal",
            expected_scale="thousands of hours of egocentric/multiview skilled-activity video",
            modalities=["egocentric_video", "audio", "narration", "annotations"],
            embodiment_fit=["perception_grounding", "human_object_interaction_priors"],
            repo_schema_targets=[
                "PerceptionGroundingWorldState",
                "ReplayEpisodeRecord",
                "EconomicWMCanonicalConsumptionRow",
            ],
            normalization_steps=[
                "never synthesize robot action truth from human-only video",
                "emit perception packet rows and block embodiment action fields",
            ],
            split_manifest_plan=[
                "split by activity, environment, and camera viewpoint"
            ],
            replay_indexer_plan=[
                "index by video_uid, clip_id, timestamp, annotation_type"
            ],
            data_quality_receipt_plan=[
                "video_clip_decodable",
                "annotation_window_present",
                "robot_action_absent_and_explicit",
                "terms_reviewed",
            ],
            label_gap_ledger_plan=[
                "no robot action",
                "human hand-object priors only",
                "domain gap to humanoid egocentric cameras",
            ],
            governance_label_spec=[
                "privacy_or_consent_review_required",
                "perception_only_no_control_authority",
            ],
            transport_meta_node_plan=[
                "perception grounding replay rows only; no control labels"
            ],
            import_blockers=[
                "access approval/storage",
                "perception-only action-denial gate",
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="agibot_world_watchlist",
            name="AgiBot World humanoid/manipulation dataset watchlist",
            source_kind="external_public_robot_manipulation_watchlist",
            priority="P2",
            bring_in_status="watchlist_source_review_required",
            source_url="https://www.agibot.com/article/231/detail/54.html",
            access_method="official release pages / mirrored dataset portals",
            expected_scale="large structured robot interaction release; exact import slice TBD",
            modalities=["robot_video", "robot_action", "humanoid_or_mobile_manipulation"],
            embodiment_fit=["humanoid_relevance_watchlist"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "review official access terms before any download",
                "identify whether humanoid rows expose action/state fields usable for replay",
            ],
            split_manifest_plan=[
                "do not split until official schema and terms are reviewed"
            ],
            replay_indexer_plan=["pending source schema review"],
            data_quality_receipt_plan=[
                "official_source_verified",
                "license_and_terms_reviewed",
                "schema_sample_decoded",
            ],
            label_gap_ledger_plan=[
                "access/source ambiguity",
                "benchmark comparability unknown",
            ],
            governance_label_spec=common_governance,
            transport_meta_node_plan=[
                "watchlist only until sample rows decode into repo-native schema"
            ],
            import_blockers=[
                "official schema and terms review",
                "sample decode not executed",
            ],
        ),
        ExternalDatasetCorpusPlan(
            dataset_id="local_robotics_vp_artifacts",
            name="Local robotics-vp-core replay and receipt artifacts",
            source_kind="local_repo_artifacts",
            priority="P0",
            bring_in_status="local_manifest_ready",
            source_url="artifacts/",
            access_method="repo-local artifact paths",
            expected_scale="small local scaffold and receipt corpus",
            modalities=["event_spine", "decision_ledger", "replay_rows", "receipts"],
            embodiment_fit=["unitree_local_harness", "economic_shadow_eval"],
            repo_schema_targets=common_targets,
            normalization_steps=[
                "preserve artifact path and digest",
                "join cpu_august_gap rows into Economic WM lower-WM ingestion",
            ],
            split_manifest_plan=[
                "fixture split only; not training-grade",
                "reserve all hardware/provider-missing rows as shadow eval",
            ],
            replay_indexer_plan=[
                "index by artifact family, event_id, decision_id, episode_id, step_idx"
            ],
            data_quality_receipt_plan=[
                "artifact_exists_or_unavailable_receipt",
                "version_present",
                "promotion_eligible_false",
            ],
            label_gap_ledger_plan=[
                "hardware proof missing",
                "provider proof missing",
                "GPU training proof missing",
            ],
            governance_label_spec=[
                "false_allow_on_promotion_claim",
                "false_veto_on_available_local_receipt",
            ],
            transport_meta_node_plan=[
                "transport/meta-node fixture rows from existing artifacts only"
            ],
            import_blockers=["not a large external corpus"],
            download_executed=False,
        ),
    ]


def build_corpus_prep_artifact_plans(
    dataset_plans: Sequence[ExternalDatasetCorpusPlan],
) -> list[CorpusPrepArtifactPlan]:
    """Build split, index, quality, label-gap, and training-corpus plans."""

    artifact_specs = [
        (
            "train_eval_split_manifest",
            "split_manifest.json",
            "python3 scripts/economic_world_model/compile_post_gap_readiness.py --output-dir {output_dir}",
            [
                "dataset_id",
                "split_id",
                "source_episode_ids",
                "train_eval_holdout_policy",
            ],
            ["split_has_no_source_leakage", "split_seed_recorded"],
        ),
        (
            "replay_indexer",
            "replay_index.jsonl",
            "python3 scripts/build_shadow_replay_dataset.py --output-dir {output_dir}/replay_dataset",
            ["dataset_id", "episode_id", "step_idx", "source_ref", "schema_version"],
            ["episode_step_keys_unique", "source_digest_recorded"],
        ),
        (
            "data_quality_receipt",
            "data_quality_receipts.jsonl",
            "python3 scripts/economic_world_model/compile_post_gap_readiness.py --output-dir {output_dir}",
            ["dataset_id", "quality_key", "status", "blockers"],
            ["failed_checks_emit_blockers", "no_training_ready_without_quality"],
        ),
        (
            "label_gap_ledger",
            "label_gap_ledger.jsonl",
            "python3 scripts/economic_world_model/compile_post_gap_readiness.py --output-dir {output_dir}",
            ["dataset_id", "gap_key", "severity", "downstream_effect"],
            ["every_gap_has_owner_or_blocker", "privacy_gaps_are_high_severity"],
        ),
        (
            "false_veto_false_allow_governance_labels",
            "governance_label_specs.jsonl",
            "python3 scripts/economic_world_model/compile_post_gap_readiness.py --output-dir {output_dir}",
            ["dataset_id", "label_key", "positive_definition", "negative_definition"],
            ["false_allow_and_false_veto_labels_present"],
        ),
        (
            "transport_meta_node_training_corpus",
            "transport_meta_node_rows.jsonl",
            "python3 scripts/economic_world_model/compile_post_gap_readiness.py --output-dir {output_dir}",
            ["dataset_id", "source_wm", "target_wm", "event_ref", "outcome_ref"],
            ["rows_are_shadow_only", "phase7_authority_not_granted"],
        ),
    ]
    rows: list[CorpusPrepArtifactPlan] = []
    for dataset in dataset_plans:
        for artifact_kind, output_name, command, required, checks in artifact_specs:
            payload = {
                "dataset_id": dataset.dataset_id,
                "artifact_kind": artifact_kind,
            }
            status = (
                "local_fixture_plan_ready"
                if dataset.source_kind == "local_repo_artifacts"
                else "planned_external_import_blocked"
            )
            rows.append(
                CorpusPrepArtifactPlan(
                    prep_id=_stable_id("corpus_prep", payload),
                    dataset_id=dataset.dataset_id,
                    artifact_kind=artifact_kind,
                    status=status,
                    output_template=(
                        f"artifacts/economic_world_model/post_gap_readiness/"
                        f"corpus/{dataset.dataset_id}/{output_name}"
                    ),
                    command_template=command,
                    required_fields=list(required),
                    acceptance_checks=list(checks),
                    blockers=list(dataset.import_blockers),
                    launch_allowed=False,
                    metadata={
                        "download_executed": dataset.download_executed,
                        "ready_for_training": dataset.ready_for_training,
                    },
                )
            )
    return rows


def build_benchmark_gate_specs() -> list[BenchmarkGateSpec]:
    """Build fail-closed benchmark gates before model work."""

    specs = [
        (
            "transport_eval_acceptance",
            "wm_transport",
            {"roundtrip_loss_max": 0.05, "receiver_actionability_min": 0.8},
            [
                "artifacts/economic_world_model/phase6_transport_eval/transport_eval_v1.json"
            ],
        ),
        (
            "perception_replay_consistency",
            "perception_grounding",
            {
                "packet_reconstruction_f1_min": 0.75,
                "temporal_consistency_min": 0.8,
            },
            [
                "artifacts/economic_world_model/post_gap_readiness/perception_replay_consistency_v1.json"
            ],
        ),
        (
            "command_timing_safety_benchmark",
            "embodiment_actuation",
            {"watchdog_timeout_violation_max": 0.0, "command_echo_match_min": 0.99},
            [
                "artifacts/economic_world_model/cpu_august_gap_execution/cpu_august_gap_execution_report_v1.json"
            ],
        ),
        (
            "economic_allocation_shadow_benchmark",
            "economic_world_model",
            {"counterfactual_regret_max": 0.05, "allocation_trace_coverage_min": 0.9},
            [
                "artifacts/economic_world_model/economic_wm_shadow_allocation_eval/economic_wm_shadow_allocation_eval_v1.json"
            ],
        ),
        (
            "phase7_governance_outcome_scoring",
            "phase7_shadow_only",
            {"false_allow_rate_max": 0.0, "false_veto_audit_recall_min": 0.95},
            [
                "artifacts/economic_world_model/phase7_meta_governance_eval/phase7_meta_governance_eval_v1.json"
            ],
        ),
        (
            "promotion_gate_fail_closed",
            "promotion",
            {"provider_truth_required": 1.0, "hardware_or_honest_sim_required": 1.0},
            [
                "artifacts/economic_world_model/provider_runs/{run_id}/benchmark_evidence_bundle_v1.json",
                "artifacts/economic_world_model/provider_runs/{run_id}/promotion_metric_report_v1.json",
            ],
        ),
    ]
    rows: list[BenchmarkGateSpec] = []
    for key, surface, thresholds, required_artifacts in specs:
        rows.append(
            BenchmarkGateSpec(
                gate_id=_stable_id("benchmark_gate", {"gate_key": key}),
                gate_key=key,
                surface=surface,
                status="fail_closed_missing_evidence",
                metrics={metric: 0.0 for metric in thresholds},
                thresholds=dict(thresholds),
                required_artifacts=list(required_artifacts),
                fail_closed_reasons=[
                    "benchmark_not_executed",
                    "provider_or_hardware_truth_missing",
                    "promotion_grade_evidence_missing",
                ],
                promotion_gate=True,
                promotion_eligible=False,
                metadata={
                    "advisory_only": True,
                    "phase7_new_concepts_added": False,
                },
            )
        )
    return rows


def _spec(
    *,
    version: str,
    lane: str,
    key: str,
    title: str,
    status: str,
    ready: bool,
    required_artifacts: Sequence[str],
    commands_or_steps: Sequence[str],
    receipts: Sequence[str],
    blockers: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReadinessSpec:
    return ReadinessSpec(
        spec_id=_stable_id("post_gap_spec", {"lane": lane, "key": key}),
        version=version,
        lane=lane,
        key=key,
        title=title,
        status=status,
        ready=ready,
        launch_allowed=False,
        required_artifacts=list(required_artifacts),
        commands_or_steps=list(commands_or_steps),
        receipts=list(receipts),
        blockers=list(blockers),
        metadata={
            "planning_only": True,
            "launch_authority_granted": False,
            **dict(metadata or {}),
        },
    )


def build_provider_runtime_packaging_specs() -> list[ReadinessSpec]:
    return [
        _spec(
            version=PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION,
            lane="provider_runtime_packaging",
            key="docker_devcontainer_linux_setup",
            title="Docker/devcontainer and local Linux setup notes",
            status="template_ready",
            ready=True,
            required_artifacts=[
                "Dockerfile.runpod",
                "docs/agent_ergonomics/run_manifest_schema.md",
            ],
            commands_or_steps=[
                "document ROS2/Unitree SDK2/Isaac/Holosoma dependency variants",
                "record Linux host assumptions and CUDA driver requirements",
            ],
            receipts=["linux_setup_notes_receipt_v1"],
            blockers=["actual Linux provider host not executed in this pass"],
        ),
        _spec(
            version=PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION,
            lane="provider_runtime_packaging",
            key="host_scanners",
            title="Host scanners for runtime roots",
            status="repo_scanners_present",
            ready=True,
            required_artifacts=["scripts/scan_phase1_runtime_layouts.py"],
            commands_or_steps=[
                "python3 scripts/scan_phase1_runtime_layouts.py --output-path artifacts/economic_world_model/post_gap_readiness/runtime_scan.json"
            ],
            receipts=["runtime_host_scan_receipt_v1"],
            blockers=["scanner output must be refreshed per host"],
        ),
        _spec(
            version=PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION,
            lane="provider_runtime_packaging",
            key="dependency_matrix",
            title="Dependency matrix",
            status="manifested",
            ready=True,
            required_artifacts=[
                "requirements-dev.txt",
                "requirements-holosoma-smoke.txt",
                "scripts/TRAINING_MIGRATION_BACKLOG.json",
            ],
            commands_or_steps=[
                "record package/runtime availability per plane before launch"
            ],
            receipts=["dependency_matrix_receipt_v1"],
            blockers=["provider-specific images still need live package lock"],
        ),
        _spec(
            version=PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION,
            lane="provider_runtime_packaging",
            key="path_root_discovery",
            title="Path and root discovery",
            status="manifested",
            ready=True,
            required_artifacts=[
                "src/world_model/sim_synth_physics/local_runtime_discovery.py"
            ],
            commands_or_steps=[
                "resolve Unitree, Isaac, Holosoma, dataset, checkpoint, and artifact roots"
            ],
            receipts=["runtime_root_discovery_receipt_v1"],
            blockers=["real host roots must be mounted"],
        ),
        _spec(
            version=PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION,
            lane="provider_runtime_packaging",
            key="artifact_pack_contracts",
            title="Artifact pack contracts",
            status="manifested",
            ready=True,
            required_artifacts=[
                "docs/agent_ergonomics/run_manifest_schema.md",
                "artifacts/economic_world_model/economic_wm_provider_runbook/manifest_templates/",
            ],
            commands_or_steps=[
                "write .agent/runs/{run_id}/manifest.json for every remote run"
            ],
            receipts=["artifact_pack_contract_receipt_v1"],
            blockers=["remote run not executed"],
        ),
        _spec(
            version=PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION,
            lane="provider_runtime_packaging",
            key="unavailable_receipts_and_noop_wrappers",
            title="Unavailable receipts and CPU no-op wrappers",
            status="manifested_fail_closed",
            ready=True,
            required_artifacts=[
                "src/world_model/economic_world_model/provider_runbook_validation.py"
            ],
            commands_or_steps=[
                "reject template-only launch",
                "emit unavailable receipt when runtime/GPU/asset missing",
            ],
            receipts=["provider_unavailable_receipt_v1", "cpu_noop_receipt_v1"],
            blockers=["no real provider invocation in this pass"],
        ),
    ]


def build_perception_embodiment_replay_loop_specs() -> list[ReadinessSpec]:
    return [
        _spec(
            version=PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION,
            lane="perception_embodiment_replay_loop",
            key="canonical_replay_fixtures",
            title="Canonical replay fixtures",
            status="manifested",
            ready=True,
            required_artifacts=["src/replay/schema.py", "scripts/build_shadow_replay_dataset.py"],
            commands_or_steps=[
                "build ReplayEpisodeRecord/ReplayStepRecord/ReplayWindowRecord views"
            ],
            receipts=["canonical_replay_fixture_receipt_v1"],
            blockers=["large external replay rows not imported yet"],
        ),
        _spec(
            version=PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION,
            lane="perception_embodiment_replay_loop",
            key="cpu_light_inference",
            title="CPU/light inference where practical",
            status="planned_blocked_by_source_media",
            ready=False,
            required_artifacts=[
                "src/world_model/perception_grounding/compiler.py",
                "src/dataset_bridges/lerobot_perception_adapter.py",
            ],
            commands_or_steps=[
                "run only light CPU packet extraction on small decoded clips",
                "block V-JEPA/VLA/foundation inference without GPU/provider proof",
            ],
            receipts=["cpu_light_perception_receipt_v1"],
            blockers=["stored image/video corpus not imported in this pass"],
        ),
        _spec(
            version=PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION,
            lane="perception_embodiment_replay_loop",
            key="perception_packets_from_media",
            title="Perception packets from stored images/video",
            status="planned_blocked_by_media",
            ready=False,
            required_artifacts=["src/world_model/perception_grounding/state.py"],
            commands_or_steps=[
                "emit packet refs only after source clips decode and terms are reviewed"
            ],
            receipts=["perception_packet_receipt_v1"],
            blockers=["external media not downloaded/decoded"],
        ),
        _spec(
            version=PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION,
            lane="perception_embodiment_replay_loop",
            key="map_packets_into_embodiment_and_economic_receipts",
            title="Map perception packets into embodiment/economic receipts",
            status="manifested_shadow_only",
            ready=True,
            required_artifacts=[
                "src/world_model/economic_world_model/lower_wm_consumption.py",
                "src/world_model/humanoid_readiness/cpu_august_gap.py",
            ],
            commands_or_steps=[
                "join perception refs into lower-WM consumption rows",
                "keep rows shadow-only until runtime truth exists",
            ],
            receipts=["lower_wm_consumption_receipt_v1"],
            blockers=["real/sim visual stream proof missing"],
        ),
        _spec(
            version=PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION,
            lane="perception_embodiment_replay_loop",
            key="event_spine_from_recorded_data",
            title="Event-spine proof from recorded data",
            status="local_unitree_event_spine_manifested",
            ready=True,
            required_artifacts=[
                "artifacts/economic_world_model/cpu_august_gap_execution/event_spine.json",
                "artifacts/economic_world_model/cpu_august_gap_execution/decision_ledger.json",
            ],
            commands_or_steps=[
                "use existing local event/decision rows as fixture proof",
                "require external corpus event spine after import",
            ],
            receipts=["event_spine_replay_join_receipt_v1"],
            blockers=["external corpus event spine not materialized"],
        ),
    ]


def build_g1_r1_purchase_readiness_specs() -> list[ReadinessSpec]:
    """Build fail-closed G1/R1 purchase and bring-up readiness criteria."""

    return [
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="variant_decision_criteria",
            title="Exact robot variant decision criteria",
            status="manifested_requires_vendor_confirmation",
            ready=True,
            required_artifacts=["docs/economic_world_model/post_gap_readiness.md"],
            commands_or_steps=[
                "Prefer G1 EDU/G1-D when secondary development, Jetson Orin, depth/LiDAR, longer battery, and dexterous-hand options are required.",
                "Treat R1 EDU as lower-cost experimentation only if secondary development and camera/compute options are contractually confirmed.",
                "Do not buy non-developer R1 AIR/R1 for SDK/control work unless vendor confirms full joint/sensor API access.",
            ],
            receipts=["robot_variant_decision_receipt_v1"],
            blockers=["vendor quote and developer interface confirmation missing"],
            metadata={
                "researched_date": "2026-05-30",
                "g1_source": "https://www.unitree.com/g1/",
                "r1_source": "https://www.unitree.com/mobile/R1/",
            },
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="workspace_safety_plan",
            title="Workspace safety plan",
            status="manifested_do_not_run_until_satisfied",
            ready=True,
            required_artifacts=["workspace_safety_plan_v1.md"],
            commands_or_steps=[
                "mark exclusion zone with fall radius plus operator buffer",
                "clear tripping hazards and fragile objects",
                "use stable-base/mobile-manipulator fallback for degraded tasks",
            ],
            receipts=["workspace_safety_receipt_v1"],
            blockers=["physical workspace not inspected"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="estop_and_recovery_plan",
            title="E-stop and recovery plan",
            status="manifested_do_not_run_until_satisfied",
            ready=True,
            required_artifacts=["estop_recovery_plan_v1.md"],
            commands_or_steps=[
                "verify manual controller stop path",
                "verify network stop path",
                "rehearse fall/recovery without payload",
            ],
            receipts=["estop_recovery_receipt_v1"],
            blockers=["robot hardware not present"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="network_dds_plan",
            title="Network/DDS plan",
            status="manifested_requires_linux_ros2_host",
            ready=True,
            required_artifacts=["network_dds_plan_v1.md"],
            commands_or_steps=[
                "dedicated robot VLAN or isolated WiFi/LAN",
                "ROS_DOMAIN_ID and DDS participant policy recorded",
                "deny publish/write until echo and watchdog receipts pass",
            ],
            receipts=["network_dds_preflight_receipt_v1"],
            blockers=["ROS2/Unitree runtime host not verified"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="companion_compute_assumptions",
            title="Companion compute assumptions",
            status="manifested_requires_budget",
            ready=True,
            required_artifacts=["companion_compute_contract_v1.json"],
            commands_or_steps=[
                "Jetson Orin developer module when vendor-supported",
                "external Linux workstation for ROS2/SDK2 logging",
                "GPU workstation/RunPod for training, not on-robot training",
            ],
            receipts=["companion_compute_receipt_v1"],
            blockers=["purchase budget and vendor configuration missing"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="camera_sensor_mounting_plan",
            title="Camera/sensor mounting plan",
            status="manifested_requires_hardware",
            ready=True,
            required_artifacts=["sensor_mounting_plan_v1.md"],
            commands_or_steps=[
                "record stock depth/LiDAR/binocular availability by variant",
                "reserve egocentric and exocentric calibration targets",
                "do not attach payloads until arm load and balance constraints are verified",
            ],
            receipts=["sensor_mounting_receipt_v1"],
            blockers=["variant not purchased and payload limits not tested"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="storage_logging_plan",
            title="Storage/logging plan",
            status="manifested",
            ready=True,
            required_artifacts=["storage_logging_plan_v1.md"],
            commands_or_steps=[
                "log rosbag2/MCAP, event spine, command echo, watchdog, safety, and video refs",
                "write digests before moving artifacts into training corpora",
            ],
            receipts=["storage_logging_receipt_v1"],
            blockers=["logging host not provisioned"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="calibration_checklist",
            title="Calibration checklist",
            status="manifested_do_not_run_until_satisfied",
            ready=True,
            required_artifacts=["calibration_checklist_v1.md"],
            commands_or_steps=[
                "camera intrinsics/extrinsics",
                "IMU orientation sanity",
                "joint zero offsets",
                "foot contact and support polygon checks",
            ],
            receipts=["calibration_receipt_v1"],
            blockers=["hardware and calibration target unavailable"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="first_week_bringup_runbook",
            title="First-week bring-up runbook",
            status="manifested",
            ready=True,
            required_artifacts=["first_week_bringup_runbook_v1.md"],
            commands_or_steps=[
                "Day 1: inventory, vendor firmware, no-write ROS2 echo",
                "Day 2: static low-state/IMU/camera logging",
                "Day 3: command echo with robot disabled or supported sim",
                "Day 4: constrained stand/walk vendor demos only",
                "Day 5: replay import and Economic WM lower-WM ingestion",
            ],
            receipts=["first_week_bringup_receipt_v1"],
            blockers=["robot not purchased"],
        ),
        _spec(
            version=G1_R1_PURCHASE_READINESS_VERSION,
            lane="g1_r1_purchase_readiness",
            key="do_not_run_until_safety_gates",
            title="Do-not-run-until safety gates",
            status="manifested_fail_closed",
            ready=True,
            required_artifacts=["do_not_run_until_safety_gates_v1.json"],
            commands_or_steps=[
                "no autonomous locomotion until e-stop, recovery, logging, calibration, and vendor safety docs are complete",
                "no publish/write until no-write echo, watchdog, and command-shape receipts pass",
            ],
            receipts=["safety_gate_receipt_v1"],
            blockers=["all gates intentionally fail without hardware receipts"],
        ),
    ]


def build_evidence_hygiene_specs() -> list[ReadinessSpec]:
    return [
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="nightly_audit_hardening",
            title="Nightly audit hardening",
            status="repo_path_present",
            ready=True,
            required_artifacts=["scripts/economic_world_model/nightly_audit.py"],
            commands_or_steps=[
                "python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md"
            ],
            receipts=["nightly_audit_receipt_v1"],
            blockers=["automation credentials may be absent"],
        ),
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="artifact_retention",
            title="Artifact retention",
            status="manifested",
            ready=True,
            required_artifacts=["artifact_retention_policy_v1.md"],
            commands_or_steps=[
                "keep manifests and small receipts in Git-adjacent artifacts",
                "store large datasets/checkpoints outside Git with digest refs",
            ],
            receipts=["artifact_retention_receipt_v1"],
            blockers=["external object store not configured"],
        ),
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="run_manifests",
            title="Run manifests",
            status="schema_present",
            ready=True,
            required_artifacts=["docs/agent_ergonomics/run_manifest_schema.md"],
            commands_or_steps=[
                "write .agent/runs/{run_id}/manifest.json for Codex cloud and RunPod runs"
            ],
            receipts=["run_manifest_receipt_v1"],
            blockers=["no remote run executed in this pass"],
        ),
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="stale_artifact_detection",
            title="Stale artifact detection",
            status="manifested",
            ready=True,
            required_artifacts=["stale_artifact_detection_report_v1.json"],
            commands_or_steps=[
                "compare source digests, commit sha, artifact timestamps, and schema versions"
            ],
            receipts=["stale_artifact_detection_receipt_v1"],
            blockers=["retention backend not configured"],
        ),
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="claim_vs_evidence_checker",
            title="Claim-vs-evidence checker",
            status="manifested_fail_closed",
            ready=True,
            required_artifacts=["claim_vs_evidence_report_v1.json"],
            commands_or_steps=[
                "reject GPU/provider/hardware/promotion claims unless matching receipt exists"
            ],
            receipts=["claim_vs_evidence_receipt_v1"],
            blockers=["checker is planning-only in this pass"],
        ),
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="ci_focused_suites",
            title="CI focused suites for roadmap surfaces",
            status="manifested",
            ready=True,
            required_artifacts=[
                "tests/test_cpu_august_gap_tranche.py",
                "tests/test_economic_wm_provider_runbook.py",
                "tests/test_economic_wm_lower_wm_consumption.py",
            ],
            commands_or_steps=[
                "python3 -m pytest -q tests/test_cpu_august_gap_tranche.py tests/test_economic_wm_provider_runbook.py tests/test_economic_wm_provider_runbook_validation.py"
            ],
            receipts=["focused_ci_suite_receipt_v1"],
            blockers=["CI workflow expansion not executed in this pass"],
        ),
        _spec(
            version=EVIDENCE_HYGIENE_SPEC_VERSION,
            lane="automation_evidence_hygiene",
            key="readiness_state_report_generator",
            title="Readiness-state report generator",
            status="implemented_this_pass",
            ready=True,
            required_artifacts=[
                "scripts/economic_world_model/compile_post_gap_readiness.py"
            ],
            commands_or_steps=[
                "python3 scripts/economic_world_model/compile_post_gap_readiness.py --output-dir artifacts/economic_world_model/post_gap_readiness"
            ],
            receipts=["post_gap_readiness_report_v1"],
            blockers=[],
        ),
    ]


def build_post_gap_readiness_bundle() -> dict[str, Any]:
    runbooks = build_gpu_day_one_runbooks()
    datasets = build_external_dataset_corpus_plans()
    corpus_prep = build_corpus_prep_artifact_plans(datasets)
    benchmark_gates = build_benchmark_gate_specs()
    provider_packaging = build_provider_runtime_packaging_specs()
    replay_loop = build_perception_embodiment_replay_loop_specs()
    purchase_readiness = build_g1_r1_purchase_readiness_specs()
    evidence_hygiene = build_evidence_hygiene_specs()

    all_manifested = all(
        [
            len(runbooks) >= 5,
            len(datasets) >= 8,
            len(corpus_prep) >= len(datasets) * 6,
            len(benchmark_gates) >= 6,
            len(provider_packaging) >= 6,
            len(replay_loop) >= 5,
            len(purchase_readiness) >= 10,
            len(evidence_hygiene) >= 7,
        ]
    )
    remaining_blockers = [
        "external dataset download and license/storage review not executed",
        "RunPod/cloud/local Linux provider windows not executed",
        "GPU training and promotion-grade benchmarks not executed",
        "G1/R1 vendor quote, purchase, and physical safety inspection missing",
        "real/sim visual stream calibration missing",
    ]
    report_payload = {
        "runbooks": [row.to_dict() for row in runbooks],
        "datasets": [row.to_dict() for row in datasets],
        "corpus_prep": [row.to_dict() for row in corpus_prep],
        "benchmark_gates": [row.to_dict() for row in benchmark_gates],
        "provider_packaging": [row.to_dict() for row in provider_packaging],
        "replay_loop": [row.to_dict() for row in replay_loop],
        "purchase_readiness": [row.to_dict() for row in purchase_readiness],
        "evidence_hygiene": [row.to_dict() for row in evidence_hygiene],
    }
    report = PostGapReadinessReport(
        report_id=_stable_id("post_gap_readiness_report", report_payload),
        status="ok_planning_complete_launch_blocked",
        all_post_gap_items_manifested=all_manifested,
        gpu_day_one_runbook_count=len(runbooks),
        external_dataset_count=len(datasets),
        corpus_prep_artifact_count=len(corpus_prep),
        benchmark_gate_count=len(benchmark_gates),
        provider_runtime_packaging_count=len(provider_packaging),
        replay_loop_count=len(replay_loop),
        g1_r1_purchase_readiness_count=len(purchase_readiness),
        evidence_hygiene_count=len(evidence_hygiene),
        launch_authority_granted=False,
        provider_executed=False,
        gpu_training_executed=False,
        external_download_executed=False,
        phase7_constraint_honored=True,
        promotion_eligible=False,
        ready_for_august_gpu_window=all_manifested,
        remaining_blockers=remaining_blockers,
        artifact_refs={},
        metadata={
            "ad_hoc_note": "2026-05-25-cpu-capable-august-gap-items",
            "phase7_new_concepts_added": False,
            "dataset_sources_identified": [dataset.dataset_id for dataset in datasets],
        },
    )
    return {
        "report": report,
        "runbooks": runbooks,
        "datasets": datasets,
        "corpus_prep": corpus_prep,
        "benchmark_gates": benchmark_gates,
        "provider_packaging": provider_packaging,
        "replay_loop": replay_loop,
        "purchase_readiness": purchase_readiness,
        "evidence_hygiene": evidence_hygiene,
    }


def _write_markdown(path: Path, bundle: Mapping[str, Any]) -> None:
    report: PostGapReadinessReport = bundle["report"]
    datasets: Sequence[ExternalDatasetCorpusPlan] = bundle["datasets"]
    runbooks: Sequence[GPUDayOneRunbook] = bundle["runbooks"]
    gates: Sequence[BenchmarkGateSpec] = bundle["benchmark_gates"]
    lines = [
        "# Economic WM Post-Gap Readiness",
        "",
        "[ad-hoc note]",
        "",
        f"- Report ID: `{report.report_id}`",
        f"- Status: `{report.status}`",
        f"- All post-gap items manifested: `{str(report.all_post_gap_items_manifested).lower()}`",
        f"- Ready for August GPU window: `{str(report.ready_for_august_gpu_window).lower()}`",
        f"- Launch authority granted: `{str(report.launch_authority_granted).lower()}`",
        f"- External downloads executed: `{str(report.external_download_executed).lower()}`",
        f"- Provider executed: `{str(report.provider_executed).lower()}`",
        f"- GPU training executed: `{str(report.gpu_training_executed).lower()}`",
        f"- Promotion eligible: `{str(report.promotion_eligible).lower()}`",
        f"- Phase 7 constraint honored: `{str(report.phase7_constraint_honored).lower()}`",
        "",
        "## Counts",
        "",
        f"- GPU day-one runbooks: `{report.gpu_day_one_runbook_count}`",
        f"- external/local dataset plans: `{report.external_dataset_count}`",
        f"- corpus prep artifact plans: `{report.corpus_prep_artifact_count}`",
        f"- benchmark gates: `{report.benchmark_gate_count}`",
        f"- provider/runtime packaging specs: `{report.provider_runtime_packaging_count}`",
        f"- replay loop specs: `{report.replay_loop_count}`",
        f"- G1/R1 purchase readiness specs: `{report.g1_r1_purchase_readiness_count}`",
        f"- evidence hygiene specs: `{report.evidence_hygiene_count}`",
        "",
        "## External Datasets To Bring In",
        "",
    ]
    for dataset in datasets:
        lines.extend(
            [
                f"### `{dataset.dataset_id}`",
                f"- name: {dataset.name}",
                f"- priority: `{dataset.priority}`",
                f"- status: `{dataset.bring_in_status}`",
                f"- source: {dataset.source_url}",
                f"- expected scale: {dataset.expected_scale}",
                f"- schema targets: {', '.join(dataset.repo_schema_targets)}",
                f"- import blockers: {', '.join(dataset.import_blockers) or 'none'}",
            ]
        )
    lines.extend(["", "## GPU Day-One Runbooks", ""])
    for runbook in runbooks:
        lines.extend(
            [
                f"### `{runbook.runbook_id}`",
                f"- name: {runbook.name}",
                f"- plane: `{runbook.plane}`",
                f"- pod class: `{runbook.pod_class}`",
                f"- horizon: `{runbook.horizon}`",
                f"- launch allowed: `{str(runbook.launch_allowed).lower()}`",
                f"- stop conditions: {', '.join(runbook.stop_conditions)}",
            ]
        )
    lines.extend(["", "## Benchmark Gates", ""])
    for gate in gates:
        lines.extend(
            [
                f"- `{gate.gate_key}` on `{gate.surface}`: `{gate.status}`; promotion eligible `{str(gate.promotion_eligible).lower()}`",
            ]
        )
    lines.extend(["", "## Remaining Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in report.remaining_blockers)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report is a planning and receipt surface. It does not download external datasets, run providers, run GPU training, purchase hardware, grant promotion, or expand Phase 7 concepts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_post_gap_readiness_bundle(
    *,
    output_dir: str | Path,
    bundle: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write all post-gap readiness artifacts and return the report payload."""

    resolved_bundle = dict(bundle or build_post_gap_readiness_bundle())
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "post_gap_readiness_report_v1.json"
    markdown_path = output_root / "post_gap_readiness_v1.md"
    runbooks_path = output_root / "gpu_day_one_runbooks_v1.jsonl"
    datasets_path = output_root / "external_dataset_corpus_plan_v1.jsonl"
    corpus_prep_path = output_root / "corpus_prep_artifact_plans_v1.jsonl"
    benchmark_path = output_root / "benchmark_gate_specs_v1.jsonl"
    provider_packaging_path = output_root / "provider_runtime_packaging_specs_v1.jsonl"
    replay_loop_path = output_root / "perception_embodiment_replay_loop_specs_v1.jsonl"
    purchase_path = output_root / "g1_r1_purchase_readiness_v1.jsonl"
    hygiene_path = output_root / "evidence_hygiene_specs_v1.jsonl"

    artifact_refs = {
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "gpu_day_one_runbooks_path": str(runbooks_path),
        "external_dataset_corpus_plan_path": str(datasets_path),
        "corpus_prep_artifact_plans_path": str(corpus_prep_path),
        "benchmark_gate_specs_path": str(benchmark_path),
        "provider_runtime_packaging_specs_path": str(provider_packaging_path),
        "perception_embodiment_replay_loop_specs_path": str(replay_loop_path),
        "g1_r1_purchase_readiness_path": str(purchase_path),
        "evidence_hygiene_specs_path": str(hygiene_path),
    }
    report: PostGapReadinessReport = resolved_bundle["report"]
    report = PostGapReadinessReport.from_dict(
        {**report.to_dict(), "artifact_refs": artifact_refs}
    )
    resolved_bundle["report"] = report

    _jsonl(runbooks_path, [row.to_dict() for row in resolved_bundle["runbooks"]])
    _jsonl(datasets_path, [row.to_dict() for row in resolved_bundle["datasets"]])
    _jsonl(corpus_prep_path, [row.to_dict() for row in resolved_bundle["corpus_prep"]])
    _jsonl(
        benchmark_path,
        [row.to_dict() for row in resolved_bundle["benchmark_gates"]],
    )
    _jsonl(
        provider_packaging_path,
        [row.to_dict() for row in resolved_bundle["provider_packaging"]],
    )
    _jsonl(replay_loop_path, [row.to_dict() for row in resolved_bundle["replay_loop"]])
    _jsonl(
        purchase_path,
        [row.to_dict() for row in resolved_bundle["purchase_readiness"]],
    )
    _jsonl(
        hygiene_path,
        [row.to_dict() for row in resolved_bundle["evidence_hygiene"]],
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, resolved_bundle)
    return report.to_dict()


def load_post_gap_readiness_report(path: str | Path) -> PostGapReadinessReport:
    return PostGapReadinessReport.from_dict(_load_json(path))


def load_gpu_day_one_runbooks(path: str | Path) -> list[GPUDayOneRunbook]:
    return [GPUDayOneRunbook.from_dict(row) for row in _load_jsonl(path)]


def load_external_dataset_corpus_plans(
    path: str | Path,
) -> list[ExternalDatasetCorpusPlan]:
    return [ExternalDatasetCorpusPlan.from_dict(row) for row in _load_jsonl(path)]


def load_corpus_prep_artifact_plans(path: str | Path) -> list[CorpusPrepArtifactPlan]:
    return [CorpusPrepArtifactPlan.from_dict(row) for row in _load_jsonl(path)]


def load_benchmark_gate_specs(path: str | Path) -> list[BenchmarkGateSpec]:
    return [BenchmarkGateSpec.from_dict(row) for row in _load_jsonl(path)]


def load_readiness_specs(path: str | Path) -> list[ReadinessSpec]:
    return [ReadinessSpec.from_dict(row) for row in _load_jsonl(path)]
