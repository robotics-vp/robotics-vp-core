"""External robotics corpus import proof for Economic WM readiness.

This module performs the first small, real external-corpus import step. It is
intentionally narrow: download or consume a LeRobot-format Parquet slice,
convert selected rows into repo-native replay records, and emit split, index,
quality, label-gap, governance, and Economic WM shadow-ingestion artifacts.

It does not train on the imported corpus, claim Unitree hardware truth, run a
provider, or grant promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.replay.dataset import ReplayDatasetBuilder
from src.utils.config_digest import sha256_json
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

DEFAULT_LEROBOT_FILES = [
    ".gitattributes",
    "README.md",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
]


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_mapping(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repo_dir(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace(":", "_")


def _download_hf_dataset_file(repo_id: str, file_path: str, destination: Path) -> None:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def download_lerobot_minimal_files(
    *,
    repo_id: str,
    download_root: str | Path,
    files: Sequence[str] = DEFAULT_LEROBOT_FILES,
) -> tuple[Path, list[dict[str, Any]]]:
    """Download the minimal LeRobot files needed for a CPU import proof."""

    dataset_root = Path(download_root) / _safe_repo_dir(repo_id)
    receipts: list[dict[str, Any]] = []
    for file_path in files:
        destination = dataset_root / file_path
        if not destination.exists():
            _download_hf_dataset_file(repo_id, file_path, destination)
        receipts.append(
            {
                "path": str(destination),
                "source_path": file_path,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    return dataset_root, receipts


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - exercised only when missing.
        raise RuntimeError(
            "pyarrow is required for LeRobot Parquet import; install pyarrow first"
        ) from exc
    return [_mapping(row) for row in pq.read_table(path).to_pylist()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_map(tasks_rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row in tasks_rows:
        try:
            index = int(row.get("task_index", 0) or 0)
        except Exception:
            index = 0
        tasks[index] = str(row.get("task", "unknown_task") or "unknown_task")
    return tasks


def _float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    else:
        return []
    floats: list[float] = []
    for item in values:
        try:
            floats.append(float(item))
        except Exception:
            continue
    return floats


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _selected_episode_indexes(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    max_episodes: int,
) -> list[int]:
    indexes: list[int] = []
    for row in sorted(episode_rows, key=lambda item: int(item.get("episode_index", 0))):
        try:
            indexes.append(_int_value(row.get("episode_index"), 0))
        except Exception:
            continue
        if len(indexes) >= max_episodes:
            break
    return indexes


def _lerobot_rows_from_parquet_rows(
    *,
    repo_id: str,
    data_rows: Sequence[Mapping[str, Any]],
    task_by_index: Mapping[int, str],
    selected_episode_indexes: Sequence[int],
    max_steps_per_episode: int,
    source_file_receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = set(int(index) for index in selected_episode_indexes)
    by_episode: dict[int, list[Mapping[str, Any]]] = {index: [] for index in selected}
    for row in data_rows:
        episode_index = _int_value(row.get("episode_index"), -1)
        if episode_index in selected and len(by_episode[episode_index]) < max_steps_per_episode:
            by_episode[episode_index].append(row)
    rows: list[dict[str, Any]] = []
    source_digests = {
        str(receipt["source_path"]): str(receipt["sha256"])
        for receipt in source_file_receipts
    }
    for episode_index, episode_rows in sorted(by_episode.items()):
        ordered = sorted(episode_rows, key=lambda row: int(row.get("frame_index", 0)))
        for local_frame_index, row in enumerate(ordered):
            task_index = _int_value(row.get("task_index"), 0)
            task = str(task_by_index.get(task_index, "unknown_task"))
            frame_index = _int_value(row.get("frame_index"), local_frame_index)
            done = bool(row.get("next.done", False))
            if local_frame_index == len(ordered) - 1:
                done = done or len(ordered) < max_steps_per_episode
            rows.append(
                {
                    "episode_id": f"{_safe_repo_dir(repo_id)}_episode_{episode_index:06d}",
                    "frame_index": frame_index,
                    "timestamp": f"{float(row.get('timestamp', 0.0) or 0.0):.6f}",
                    "observation": {
                        "observation.state": _float_list(
                            row.get("observation.state")
                        ),
                        "observation.environment_state": _float_list(
                            row.get("observation.environment_state")
                        ),
                    },
                    "action": {"action": _float_list(row.get("action"))},
                    "reward": float(row.get("next.reward", 0.0) or 0.0),
                    "done": done,
                    "task": task,
                    "environment": repo_id,
                    "source_domain": "external_lerobot_parquet",
                    "metadata": {
                        "run_id": f"external_lerobot_{_safe_repo_dir(repo_id)}",
                        "seed": 0,
                        "skill_mode": "external_corpus_shadow_import",
                        "task_index": task_index,
                        "source_dataset_id": repo_id,
                        "source_episode_index": episode_index,
                        "source_frame_index": frame_index,
                        "source_row_index": _int_value(row.get("index"), 0),
                        "next_success": bool(row.get("next.success", False)),
                        "benchmark_gate": {
                            "ready": False,
                            "reason": "external_slice_import_not_benchmark_grade",
                        },
                        "future_training_signals": {
                            "external_corpus_imported": True,
                            "training_ready": False,
                            "promotion_eligible": False,
                            "unitree_hardware_truth": False,
                            "provider_truth": False,
                        },
                        "internal_sidecars": {
                            "external_dataset_id": repo_id,
                            "source_file_digests": source_digests,
                            "label_gap_ledger_ref": "label_gap_ledger.jsonl",
                            "data_quality_receipts_ref": "data_quality_receipts.jsonl",
                        },
                    },
                }
            )
    return rows


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
    source_total_bytes: int
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
            "source_total_bytes": int(self.source_total_bytes),
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
            source_total_bytes=int(payload.get("source_total_bytes", 0) or 0),
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
            remaining_blockers=_strings(payload.get("remaining_blockers")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", EXTERNAL_LEROBOT_CORPUS_IMPORT_REPORT_VERSION
                )
            ),
        )


def _quality_receipts(
    *,
    dataset_id: str,
    source_file_receipts: Sequence[Mapping[str, Any]],
    tasks_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    replay_episode_count: int,
    replay_step_count: int,
    selected_episode_ids: Sequence[str],
) -> list[ExternalCorpusQualityReceipt]:
    def receipt(
        key: str,
        passed: bool,
        measured_value: Any,
        blockers: Optional[Sequence[str]] = None,
    ) -> ExternalCorpusQualityReceipt:
        status = "ok" if passed else "blocked"
        return ExternalCorpusQualityReceipt(
            receipt_id=_stable_id(
                "external_corpus_quality",
                {
                    "dataset_id": dataset_id,
                    "check_key": key,
                    "measured_value": measured_value,
                },
            ),
            dataset_id=dataset_id,
            check_key=key,
            status=status,
            passed=passed,
            measured_value=measured_value,
            blockers=list(blockers or ([] if passed else [key])),
        )

    monotonic = True
    by_episode: dict[str, list[float]] = {}
    for row in selected_rows:
        episode_id = str(row["episode_id"])
        by_episode.setdefault(episode_id, []).append(float(row["timestamp"]))
    for timestamps in by_episode.values():
        if timestamps != sorted(timestamps):
            monotonic = False
            break
    return [
        receipt("source_files_downloaded_or_present", bool(source_file_receipts), len(source_file_receipts)),
        receipt("source_file_digests_recorded", all("sha256" in row for row in source_file_receipts), len(source_file_receipts)),
        receipt("task_metadata_present", bool(tasks_rows), len(tasks_rows)),
        receipt("episode_metadata_present", bool(episode_rows), len(episode_rows)),
        receipt("selected_episode_count_nonzero", replay_episode_count > 0, replay_episode_count),
        receipt("selected_step_count_nonzero", replay_step_count > 0, replay_step_count),
        receipt("action_schema_present", all(bool(row.get("action", {}).get("action")) for row in selected_rows), replay_step_count),
        receipt("observation_schema_present", all(bool(row.get("observation", {}).get("observation.state")) for row in selected_rows), replay_step_count),
        receipt("timestamp_monotonic_per_episode", monotonic, selected_episode_ids),
        receipt("train_eval_split_possible", len(selected_episode_ids) >= 2, len(selected_episode_ids)),
        receipt("promotion_gate_fail_closed", True, False),
    ]


def _label_gaps(dataset_id: str) -> list[ExternalCorpusLabelGapLedgerEntry]:
    specs = [
        (
            "not_unitree_hardware_truth",
            "high",
            "Rows are useful for schema and shadow evaluation but cannot prove Unitree control.",
            "Require Unitree G1/R1 or honest sim runtime receipts before control claims.",
            True,
            True,
        ),
        (
            "no_provider_or_gpu_training_truth",
            "high",
            "Rows have not been used in provider-backed or GPU training.",
            "Require provider and GPU runtime manifests before training claims.",
            True,
            True,
        ),
        (
            "no_force_contact_or_estop_state",
            "medium",
            "Safety and recovery labels are absent from this keypoint-only corpus.",
            "Join with hardware/sim safety receipts before safety benchmark promotion.",
            False,
            True,
        ),
        (
            "no_image_video_modalities_in_selected_slice",
            "medium",
            "The selected proof slice validates Parquet rows, not perception pixels.",
            "Follow with LeRobot image/video slice only after this row path proves stable.",
            False,
            True,
        ),
        (
            "non_bipedal_task_domain",
            "medium",
            "PushT tabletop/keypoint behavior is not bipedal whole-body locomotion.",
            "Use as external corpus adapter proof, not humanoid embodiment truth.",
            False,
            True,
        ),
    ]
    return [
        ExternalCorpusLabelGapLedgerEntry(
            gap_id=_stable_id(
                "external_corpus_label_gap",
                {"dataset_id": dataset_id, "gap_key": gap_key},
            ),
            dataset_id=dataset_id,
            gap_key=gap_key,
            severity=severity,
            downstream_effect=effect,
            mitigation=mitigation,
            blocks_training=blocks_training,
            blocks_promotion=blocks_promotion,
        )
        for gap_key, severity, effect, mitigation, blocks_training, blocks_promotion in specs
    ]


def _governance_labels(dataset_id: str) -> list[ExternalCorpusGovernanceLabelSpec]:
    specs = [
        (
            "false_allow_hardware_truth",
            "A claim says the imported external rows prove Unitree hardware or real control.",
            "A claim states the rows are external corpus shadow/import proof only.",
        ),
        (
            "false_allow_training_or_promotion",
            "A claim treats imported rows as training-complete or promotion-grade evidence.",
            "A claim requires GPU/provider/benchmark receipts before training or promotion.",
        ),
        (
            "false_veto_valid_external_row",
            "A valid decoded external row is discarded despite passing source, schema, and digest checks.",
            "A row is rejected because it lacks required schema, digest, or source metadata.",
        ),
        (
            "false_veto_shadow_eval_eligible_corpus",
            "A corpus with fail-closed promotion posture is blocked from shadow evaluation.",
            "A corpus is blocked from live training or promotion because hard evidence is missing.",
        ),
    ]
    return [
        ExternalCorpusGovernanceLabelSpec(
            label_id=_stable_id(
                "external_corpus_governance_label",
                {"dataset_id": dataset_id, "label_key": label_key},
            ),
            dataset_id=dataset_id,
            label_key=label_key,
            positive_definition=positive,
            negative_definition=negative,
            use_for_training=False,
            authority_class="shadow_label_spec_only",
        )
        for label_key, positive, negative in specs
    ]


def _split_manifest(
    *,
    dataset_id: str,
    episode_ids: Sequence[str],
) -> ExternalCorpusSplitManifest:
    ordered = sorted(set(episode_ids))
    eval_count = 1 if len(ordered) > 1 else 0
    eval_ids = ordered[-eval_count:] if eval_count else []
    train_ids = ordered[:-eval_count] if eval_count else ordered
    return ExternalCorpusSplitManifest(
        split_id=_stable_id("external_corpus_split", {"dataset_id": dataset_id, "episode_ids": ordered}),
        dataset_id=dataset_id,
        train_episode_ids=train_ids,
        eval_episode_ids=eval_ids,
        holdout_policy="deterministic_last_episode_eval_fixture_only",
        ready_for_training=False,
        metadata={
            "fixture_import": True,
            "external_download_executed": True,
            "training_blocked_until_quality_and_benchmark_scale": True,
        },
    )


def import_lerobot_corpus_slice(
    *,
    repo_id: str,
    output_dir: str | Path,
    source_root: Optional[str | Path] = None,
    download: bool = True,
    max_episodes: int = 2,
    max_steps_per_episode: int = 200,
) -> dict[str, Any]:
    """Import a small LeRobot Parquet slice into repo-native artifacts."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    download_root = output_root / "downloads"
    if source_root is None:
        resolved_source_root, source_file_receipts = download_lerobot_minimal_files(
            repo_id=repo_id,
            download_root=download_root,
        )
        download_executed = download
    else:
        resolved_source_root = Path(source_root)
        source_file_receipts = [
            {
                "path": str(resolved_source_root / file_path),
                "source_path": file_path,
                "size_bytes": (resolved_source_root / file_path).stat().st_size,
                "sha256": _sha256_file(resolved_source_root / file_path),
            }
            for file_path in DEFAULT_LEROBOT_FILES
            if (resolved_source_root / file_path).exists()
        ]
        download_executed = False
    info = _read_json(resolved_source_root / "meta/info.json")
    tasks_rows = _read_parquet(resolved_source_root / "meta/tasks.parquet")
    episode_rows = _read_parquet(
        resolved_source_root / "meta/episodes/chunk-000/file-000.parquet"
    )
    data_rows = _read_parquet(resolved_source_root / "data/chunk-000/file-000.parquet")
    selected_indexes = _selected_episode_indexes(
        episode_rows,
        max_episodes=max_episodes,
    )
    task_by_index = _task_map(tasks_rows)
    lerobot_rows = _lerobot_rows_from_parquet_rows(
        repo_id=repo_id,
        data_rows=data_rows,
        task_by_index=task_by_index,
        selected_episode_indexes=selected_indexes,
        max_steps_per_episode=max_steps_per_episode,
        source_file_receipts=source_file_receipts,
    )
    rows_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in lerobot_rows:
        rows_by_episode.setdefault(str(row["episode_id"]), []).append(row)

    replay_builder = ReplayDatasetBuilder()
    for rows in rows_by_episode.values():
        replay_builder.add_lerobot_rows(rows)
    replay_dataset_dir = output_root / "replay_dataset"
    replay_bundle = replay_builder.write(replay_dataset_dir)

    episode_ids = [episode.episode_id for episode in replay_bundle.episodes]
    split_manifest = _split_manifest(dataset_id=repo_id, episode_ids=episode_ids)
    replay_index = [
        ExternalCorpusReplayIndexRow(
            index_id=_stable_id(
                "external_corpus_replay_index",
                {
                    "dataset_id": repo_id,
                    "episode_id": step.episode_id,
                    "step_idx": step.step_idx,
                },
            ),
            dataset_id=repo_id,
            episode_id=step.episode_id,
            step_idx=step.step_idx,
            task_id=step.task_id,
            source_domain=step.source_domain,
            replay_step_record_id=step.record_id,
            metadata={
                "source_episode_index": step.metadata.get("source_episode_index"),
                "source_frame_index": step.metadata.get("source_frame_index"),
            },
        )
        for step in replay_bundle.steps
    ]
    quality_receipts = _quality_receipts(
        dataset_id=repo_id,
        source_file_receipts=source_file_receipts,
        tasks_rows=tasks_rows,
        episode_rows=episode_rows,
        selected_rows=lerobot_rows,
        replay_episode_count=len(replay_bundle.episodes),
        replay_step_count=len(replay_bundle.steps),
        selected_episode_ids=episode_ids,
    )
    label_gaps = _label_gaps(repo_id)
    governance_labels = _governance_labels(repo_id)

    rows_path = output_root / "external_lerobot_rows.jsonl"
    split_path = output_root / "train_eval_split_manifest.json"
    index_path = output_root / "replay_index.jsonl"
    quality_path = output_root / "data_quality_receipts.jsonl"
    gap_path = output_root / "label_gap_ledger.jsonl"
    governance_path = output_root / "governance_label_specs.jsonl"
    ingestion_path = output_root / "economic_wm_external_corpus_ingestion_rows.jsonl"
    report_path = output_root / "external_lerobot_corpus_import_report_v1.json"

    ingestion_rows = [
        EconomicWMExternalCorpusIngestionRow(
            ingestion_id=_stable_id(
                "economic_wm_external_corpus_ingestion",
                {"dataset_id": repo_id, "episode_count": len(replay_bundle.episodes)},
            ),
            dataset_id=repo_id,
            corpus_surface="external_lerobot_parquet_shadow_import",
            status="ok_shadow_ingestion_training_blocked",
            episode_count=len(replay_bundle.episodes),
            step_count=len(replay_bundle.steps),
            replay_dataset_dir=str(replay_dataset_dir),
            split_manifest_ref=str(split_path),
            replay_index_ref=str(index_path),
            data_quality_ref=str(quality_path),
            label_gap_ref=str(gap_path),
            governance_label_ref=str(governance_path),
            ready_for_shadow_eval=True,
            ready_for_training=False,
            promotion_eligible=False,
            blockers=[
                "external_slice_not_training_scale",
                "not_unitree_hardware_truth",
                "gpu_provider_training_not_run",
                "promotion_benchmark_not_run",
            ],
            metadata={
                "source_info": info,
                "selected_episode_indexes": selected_indexes,
            },
        )
    ]
    _write_jsonl(rows_path, lerobot_rows)
    _write_json(split_path, split_manifest.to_dict())
    _write_jsonl(index_path, [row.to_dict() for row in replay_index])
    _write_jsonl(quality_path, [row.to_dict() for row in quality_receipts])
    _write_jsonl(gap_path, [row.to_dict() for row in label_gaps])
    _write_jsonl(governance_path, [row.to_dict() for row in governance_labels])
    _write_jsonl(ingestion_path, [row.to_dict() for row in ingestion_rows])

    source_total_bytes = sum(int(row.get("size_bytes", 0) or 0) for row in source_file_receipts)
    quality_passed_count = sum(1 for receipt in quality_receipts if receipt.passed)
    artifact_refs = {
        "report_path": str(report_path),
        "source_root": str(resolved_source_root),
        "external_lerobot_rows_path": str(rows_path),
        "replay_dataset_dir": str(replay_dataset_dir),
        "replay_manifest_path": str(replay_dataset_dir / "manifest.json"),
        "train_eval_split_manifest_path": str(split_path),
        "replay_index_path": str(index_path),
        "data_quality_receipts_path": str(quality_path),
        "label_gap_ledger_path": str(gap_path),
        "governance_label_specs_path": str(governance_path),
        "economic_wm_external_corpus_ingestion_rows_path": str(ingestion_path),
    }
    report = ExternalLerobotCorpusImportReport(
        report_id=_stable_id(
            "external_lerobot_corpus_import_report",
            {
                "dataset_id": repo_id,
                "episodes": episode_ids,
                "step_count": len(replay_bundle.steps),
            },
        ),
        dataset_id=repo_id,
        status="ok_external_corpus_slice_imported_shadow_only",
        source_root=str(resolved_source_root),
        download_executed=download_executed,
        files_downloaded_count=len(source_file_receipts),
        source_total_bytes=source_total_bytes,
        selected_episode_count=len(rows_by_episode),
        selected_step_count=len(lerobot_rows),
        replay_episode_count=len(replay_bundle.episodes),
        replay_step_count=len(replay_bundle.steps),
        quality_receipt_count=len(quality_receipts),
        quality_passed_count=quality_passed_count,
        label_gap_count=len(label_gaps),
        governance_label_count=len(governance_labels),
        ingestion_row_count=len(ingestion_rows),
        ready_for_shadow_eval=True,
        ready_for_training=False,
        provider_executed=False,
        gpu_training_executed=False,
        unitree_hardware_truth=False,
        promotion_eligible=False,
        phase7_authority_granted=False,
        remaining_blockers=[
            "external_slice_not_training_scale",
            "not_unitree_hardware_truth",
            "gpu_provider_training_not_run",
            "promotion_benchmark_not_run",
            "image_video_perception_modalities_not_imported_in_this_slice",
        ],
        artifact_refs=artifact_refs,
        metadata={
            "source_info": info,
            "selected_episode_indexes": selected_indexes,
            "task_by_index": dict(task_by_index),
            "source_file_receipts": list(source_file_receipts),
        },
    )
    _write_json(report_path, report.to_dict())
    return report.to_dict()


def load_external_lerobot_corpus_import_report(
    path: str | Path,
) -> ExternalLerobotCorpusImportReport:
    return ExternalLerobotCorpusImportReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_external_corpus_quality_receipts(
    path: str | Path,
) -> list[ExternalCorpusQualityReceipt]:
    return [
        ExternalCorpusQualityReceipt(
            receipt_id=str(row.get("receipt_id", "")),
            dataset_id=str(row.get("dataset_id", "")),
            check_key=str(row.get("check_key", "")),
            status=str(row.get("status", "")),
            passed=bool(row.get("passed", False)),
            measured_value=row.get("measured_value"),
            blockers=_strings(row.get("blockers")),
            metadata=_mapping(row.get("metadata")),
            version=str(row.get("version", EXTERNAL_CORPUS_QUALITY_RECEIPT_VERSION)),
        )
        for row in _load_jsonl(path)
    ]


__all__ = [
    "EconomicWMExternalCorpusIngestionRow",
    "ExternalCorpusGovernanceLabelSpec",
    "ExternalCorpusLabelGapLedgerEntry",
    "ExternalCorpusQualityReceipt",
    "ExternalCorpusReplayIndexRow",
    "ExternalCorpusSplitManifest",
    "ExternalLerobotCorpusImportReport",
    "download_lerobot_minimal_files",
    "import_lerobot_corpus_slice",
    "load_external_corpus_quality_receipts",
    "load_external_lerobot_corpus_import_report",
]
