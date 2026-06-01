"""Typed receipt models for external robotics corpus imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from src.utils.json_safe import to_json_safe

EXTERNAL_LEROBOT_CORPUS_IMPORT_REPORT_VERSION = (
    "external_lerobot_corpus_import_report_v1"
)
EXTERNAL_CORPUS_QUALITY_RECEIPT_VERSION = "external_corpus_quality_receipt_v1"
EXTERNAL_CORPUS_LABEL_GAP_LEDGER_VERSION = "external_corpus_label_gap_ledger_v1"
EXTERNAL_CORPUS_GOVERNANCE_LABEL_SPEC_VERSION = (
    "external_corpus_governance_label_spec_v1"
)
EXTERNAL_CORPUS_SPLIT_MANIFEST_VERSION = "external_corpus_split_manifest_v1"
EXTERNAL_CORPUS_REPLAY_INDEX_ROW_VERSION = "external_corpus_replay_index_row_v1"
ECONOMIC_WM_EXTERNAL_CORPUS_INGESTION_ROW_VERSION = (
    "economic_wm_external_corpus_ingestion_row_v1"
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


@dataclass(frozen=True)
class ExternalCorpusQualityReceipt:
    receipt_id: str
    dataset_id: str
    check_key: str
    status: str
    passed: bool
    measured_value: Any = None
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_CORPUS_QUALITY_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "check_key": self.check_key,
            "status": self.status,
            "passed": bool(self.passed),
            "measured_value": to_json_safe(self.measured_value),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class ExternalCorpusLabelGapLedgerEntry:
    gap_id: str
    dataset_id: str
    gap_key: str
    severity: str
    downstream_effect: str
    mitigation: str
    blocks_training: bool
    blocks_promotion: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_CORPUS_LABEL_GAP_LEDGER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "gap_key": self.gap_key,
            "severity": self.severity,
            "downstream_effect": self.downstream_effect,
            "mitigation": self.mitigation,
            "blocks_training": bool(self.blocks_training),
            "blocks_promotion": bool(self.blocks_promotion),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class ExternalCorpusGovernanceLabelSpec:
    label_id: str
    dataset_id: str
    label_key: str
    positive_definition: str
    negative_definition: str
    use_for_training: bool
    authority_class: str
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_CORPUS_GOVERNANCE_LABEL_SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "label_key": self.label_key,
            "positive_definition": self.positive_definition,
            "negative_definition": self.negative_definition,
            "use_for_training": bool(self.use_for_training),
            "authority_class": self.authority_class,
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class ExternalCorpusSplitManifest:
    split_id: str
    dataset_id: str
    train_episode_ids: list[str]
    eval_episode_ids: list[str]
    holdout_policy: str
    ready_for_training: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_CORPUS_SPLIT_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "train_episode_ids": list(self.train_episode_ids),
            "eval_episode_ids": list(self.eval_episode_ids),
            "holdout_policy": self.holdout_policy,
            "ready_for_training": bool(self.ready_for_training),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class ExternalCorpusReplayIndexRow:
    index_id: str
    dataset_id: str
    episode_id: str
    step_idx: int
    task_id: str
    source_domain: str
    replay_step_record_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_CORPUS_REPLAY_INDEX_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_id": self.index_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "step_idx": int(self.step_idx),
            "task_id": self.task_id,
            "source_domain": self.source_domain,
            "replay_step_record_id": self.replay_step_record_id,
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class EconomicWMExternalCorpusIngestionRow:
    ingestion_id: str
    dataset_id: str
    corpus_surface: str
    status: str
    episode_count: int
    step_count: int
    replay_dataset_dir: str
    split_manifest_ref: str
    replay_index_ref: str
    data_quality_ref: str
    label_gap_ref: str
    governance_label_ref: str
    ready_for_shadow_eval: bool
    ready_for_training: bool
    promotion_eligible: bool
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_EXTERNAL_CORPUS_INGESTION_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion_id": self.ingestion_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "corpus_surface": self.corpus_surface,
            "status": self.status,
            "episode_count": int(self.episode_count),
            "step_count": int(self.step_count),
            "replay_dataset_dir": self.replay_dataset_dir,
            "split_manifest_ref": self.split_manifest_ref,
            "replay_index_ref": self.replay_index_ref,
            "data_quality_ref": self.data_quality_ref,
            "label_gap_ref": self.label_gap_ref,
            "governance_label_ref": self.governance_label_ref,
            "ready_for_shadow_eval": bool(self.ready_for_shadow_eval),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class ExternalLerobotCorpusImportReport:
    report_id: str
    dataset_id: str
    status: str
    source_root: str
    download_executed: bool
    files_downloaded_count: int
    video_files_downloaded_count: int
    source_total_bytes: int
    video_total_bytes: int
    selected_episode_count: int
    selected_step_count: int
    replay_episode_count: int
    replay_step_count: int
    quality_receipt_count: int
    quality_passed_count: int
    label_gap_count: int
    governance_label_count: int
    ingestion_row_count: int
    ready_for_shadow_eval: bool
    ready_for_training: bool
    provider_executed: bool
    gpu_training_executed: bool
    unitree_hardware_truth: bool
    promotion_eligible: bool
    phase7_authority_granted: bool
    image_video_modalities_imported: bool = False
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EXTERNAL_LEROBOT_CORPUS_IMPORT_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "source_root": self.source_root,
            "download_executed": bool(self.download_executed),
            "files_downloaded_count": int(self.files_downloaded_count),
            "video_files_downloaded_count": int(self.video_files_downloaded_count),
            "source_total_bytes": int(self.source_total_bytes),
            "video_total_bytes": int(self.video_total_bytes),
            "selected_episode_count": int(self.selected_episode_count),
            "selected_step_count": int(self.selected_step_count),
            "replay_episode_count": int(self.replay_episode_count),
            "replay_step_count": int(self.replay_step_count),
            "quality_receipt_count": int(self.quality_receipt_count),
            "quality_passed_count": int(self.quality_passed_count),
            "label_gap_count": int(self.label_gap_count),
            "governance_label_count": int(self.governance_label_count),
            "ingestion_row_count": int(self.ingestion_row_count),
            "ready_for_shadow_eval": bool(self.ready_for_shadow_eval),
            "ready_for_training": bool(self.ready_for_training),
            "provider_executed": bool(self.provider_executed),
            "gpu_training_executed": bool(self.gpu_training_executed),
            "unitree_hardware_truth": bool(self.unitree_hardware_truth),
            "promotion_eligible": bool(self.promotion_eligible),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "image_video_modalities_imported": bool(
                self.image_video_modalities_imported
            ),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ExternalLerobotCorpusImportReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            status=str(payload.get("status", "")),
            source_root=str(payload.get("source_root", "")),
            download_executed=bool(payload.get("download_executed", False)),
            files_downloaded_count=int(payload.get("files_downloaded_count", 0) or 0),
            video_files_downloaded_count=int(
                payload.get("video_files_downloaded_count", 0) or 0
            ),
            source_total_bytes=int(payload.get("source_total_bytes", 0) or 0),
            video_total_bytes=int(payload.get("video_total_bytes", 0) or 0),
            selected_episode_count=int(payload.get("selected_episode_count", 0) or 0),
            selected_step_count=int(payload.get("selected_step_count", 0) or 0),
            replay_episode_count=int(payload.get("replay_episode_count", 0) or 0),
            replay_step_count=int(payload.get("replay_step_count", 0) or 0),
            quality_receipt_count=int(payload.get("quality_receipt_count", 0) or 0),
            quality_passed_count=int(payload.get("quality_passed_count", 0) or 0),
            label_gap_count=int(payload.get("label_gap_count", 0) or 0),
            governance_label_count=int(payload.get("governance_label_count", 0) or 0),
            ingestion_row_count=int(payload.get("ingestion_row_count", 0) or 0),
            ready_for_shadow_eval=bool(payload.get("ready_for_shadow_eval", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            gpu_training_executed=bool(payload.get("gpu_training_executed", False)),
            unitree_hardware_truth=bool(payload.get("unitree_hardware_truth", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            image_video_modalities_imported=bool(
                payload.get("image_video_modalities_imported", False)
            ),
            remaining_blockers=_strings(payload.get("remaining_blockers")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", EXTERNAL_LEROBOT_CORPUS_IMPORT_REPORT_VERSION
                )
            ),
        )



__all__ = [
    "ECONOMIC_WM_EXTERNAL_CORPUS_INGESTION_ROW_VERSION",
    "EXTERNAL_CORPUS_GOVERNANCE_LABEL_SPEC_VERSION",
    "EXTERNAL_CORPUS_LABEL_GAP_LEDGER_VERSION",
    "EXTERNAL_CORPUS_QUALITY_RECEIPT_VERSION",
    "EXTERNAL_CORPUS_REPLAY_INDEX_ROW_VERSION",
    "EXTERNAL_CORPUS_SPLIT_MANIFEST_VERSION",
    "EXTERNAL_LEROBOT_CORPUS_IMPORT_REPORT_VERSION",
    "EconomicWMExternalCorpusIngestionRow",
    "ExternalCorpusGovernanceLabelSpec",
    "ExternalCorpusLabelGapLedgerEntry",
    "ExternalCorpusQualityReceipt",
    "ExternalCorpusReplayIndexRow",
    "ExternalCorpusSplitManifest",
    "ExternalLerobotCorpusImportReport",
]
