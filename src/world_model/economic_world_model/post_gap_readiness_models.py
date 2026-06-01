"""Typed post-gap readiness models for Economic WM bring-up."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

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



__all__ = [
    "BENCHMARK_GATE_SPEC_VERSION",
    "CORPUS_PREP_ARTIFACT_PLAN_VERSION",
    "EVIDENCE_HYGIENE_SPEC_VERSION",
    "EXTERNAL_DATASET_CORPUS_PLAN_VERSION",
    "G1_R1_PURCHASE_READINESS_VERSION",
    "GPU_DAY_ONE_RUNBOOK_VERSION",
    "PERCEPTION_EMBODIMENT_REPLAY_LOOP_SPEC_VERSION",
    "POST_GAP_READINESS_REPORT_VERSION",
    "PROVIDER_RUNTIME_PACKAGING_SPEC_VERSION",
    "BenchmarkGateSpec",
    "CorpusPrepArtifactPlan",
    "ExternalDatasetCorpusPlan",
    "GPUDayOneRunbook",
    "PostGapReadinessReport",
    "ReadinessSpec",
]
