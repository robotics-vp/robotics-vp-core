"""Unified training-runtime manifest helpers for regal-aware jobs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.contracts.schemas import TrajectoryAuditV1
from src.replay.compatibility import CompatibilityCheckResult
from src.replay.dataset import ReplayDatasetBundle
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord
from src.utils.config_digest import sha256_file, sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit


TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION = "training_runtime_manifest_v1"


@dataclass(frozen=True)
class TrainingRuntimeManifest:
    """Standardized manifest emitted by canonical regal-aware training jobs."""

    schema_version: str
    run_id: str
    training_kind: str
    status: str
    seed: int
    plan_id: str
    plan_sha: str
    started_at: str
    ended_at: str
    config_path: Optional[str]
    config_digest: str
    replay_dataset_dir: Optional[str]
    replay_manifest_digest: Optional[str]
    replay_dataset_summary: Dict[str, Any]
    objective_profile_snapshot: Dict[str, Any]
    promotion_policy_snapshot: Dict[str, Any]
    source_domain_coverage: Dict[str, Any]
    receipt_label_coverage: Dict[str, Any]
    artifact_paths: Dict[str, str]
    inferential_learnability_summary: Dict[str, Any] = field(default_factory=dict)
    inferential_work_order_summary: Dict[str, Any] = field(default_factory=dict)
    checkpoint_registry_path: Optional[str] = None
    checkpoint_registry_digest: Optional[str] = None
    promotion_evidence_path: Optional[str] = None
    promotion_evidence_digest: Optional[str] = None
    promotion_ledger_path: Optional[str] = None
    promotion_ledger_digest: Optional[str] = None
    budget_settlement_path: Optional[str] = None
    budget_settlement_digest: Optional[str] = None
    budget_settlement_live: bool = False
    artifact_schema_compatibility: list[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "training_kind": self.training_kind,
            "status": self.status,
            "seed": int(self.seed),
            "plan_id": self.plan_id,
            "plan_sha": self.plan_sha,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "config_path": self.config_path,
            "config_digest": self.config_digest,
            "replay_dataset_dir": self.replay_dataset_dir,
            "replay_manifest_digest": self.replay_manifest_digest,
            "replay_dataset_summary": dict(self.replay_dataset_summary),
            "objective_profile_snapshot": dict(self.objective_profile_snapshot),
            "promotion_policy_snapshot": dict(self.promotion_policy_snapshot),
            "source_domain_coverage": dict(self.source_domain_coverage),
            "receipt_label_coverage": dict(self.receipt_label_coverage),
            "inferential_learnability_summary": dict(self.inferential_learnability_summary),
            "inferential_work_order_summary": dict(self.inferential_work_order_summary),
            "artifact_paths": dict(self.artifact_paths),
            "checkpoint_registry_path": self.checkpoint_registry_path,
            "checkpoint_registry_digest": self.checkpoint_registry_digest,
            "promotion_evidence_path": self.promotion_evidence_path,
            "promotion_evidence_digest": self.promotion_evidence_digest,
            "promotion_ledger_path": self.promotion_ledger_path,
            "promotion_ledger_digest": self.promotion_ledger_digest,
            "budget_settlement_path": self.budget_settlement_path,
            "budget_settlement_digest": self.budget_settlement_digest,
            "budget_settlement_live": bool(self.budget_settlement_live),
            "artifact_schema_compatibility": [dict(row) for row in self.artifact_schema_compatibility],
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingRuntimeManifest":
        return cls(
            schema_version=str(payload.get("schema_version", TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION)),
            run_id=str(payload.get("run_id", "")),
            training_kind=str(payload.get("training_kind", "")),
            status=str(payload.get("status", "unknown")),
            seed=int(payload.get("seed", 0)),
            plan_id=str(payload.get("plan_id", "")),
            plan_sha=str(payload.get("plan_sha", "")),
            started_at=str(payload.get("started_at", "")),
            ended_at=str(payload.get("ended_at", "")),
            config_path=payload.get("config_path"),
            config_digest=str(payload.get("config_digest", "")),
            replay_dataset_dir=payload.get("replay_dataset_dir"),
            replay_manifest_digest=payload.get("replay_manifest_digest"),
            replay_dataset_summary=dict(payload.get("replay_dataset_summary", {}) or {}),
            objective_profile_snapshot=dict(payload.get("objective_profile_snapshot", {}) or {}),
            promotion_policy_snapshot=dict(payload.get("promotion_policy_snapshot", {}) or {}),
            source_domain_coverage=dict(payload.get("source_domain_coverage", {}) or {}),
            receipt_label_coverage=dict(payload.get("receipt_label_coverage", {}) or {}),
            inferential_learnability_summary=dict(
                payload.get("inferential_learnability_summary", {}) or {}
            ),
            inferential_work_order_summary=dict(
                payload.get("inferential_work_order_summary", {}) or {}
            ),
            artifact_paths=dict(payload.get("artifact_paths", {}) or {}),
            checkpoint_registry_path=payload.get("checkpoint_registry_path"),
            checkpoint_registry_digest=payload.get("checkpoint_registry_digest"),
            promotion_evidence_path=payload.get("promotion_evidence_path"),
            promotion_evidence_digest=payload.get("promotion_evidence_digest"),
            promotion_ledger_path=payload.get("promotion_ledger_path"),
            promotion_ledger_digest=payload.get("promotion_ledger_digest"),
            budget_settlement_path=payload.get("budget_settlement_path"),
            budget_settlement_digest=payload.get("budget_settlement_digest"),
            budget_settlement_live=bool(payload.get("budget_settlement_live", False)),
            artifact_schema_compatibility=[
                dict(row)
                for row in list(payload.get("artifact_schema_compatibility", []) or [])
            ],
            failure_reason=payload.get("failure_reason"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def write_training_runtime_manifest(
    path: str | Path,
    manifest: TrainingRuntimeManifest,
) -> str:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sha256_file(output_path)


def load_training_runtime_manifest(path: str | Path) -> TrainingRuntimeManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingRuntimeManifest.from_dict(payload)


def check_training_runtime_manifest_compatibility(
    manifest: TrainingRuntimeManifest,
    *,
    expected_schema_version: str = TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION,
) -> CompatibilityCheckResult:
    reasons: list[str] = []
    if manifest.schema_version != expected_schema_version:
        reasons.append("schema_version_mismatch")
    required_paths = [
        value
        for value in [
            manifest.checkpoint_registry_path,
            manifest.promotion_evidence_path,
            manifest.promotion_ledger_path,
            manifest.budget_settlement_path,
            *manifest.artifact_paths.values(),
        ]
        if value
    ]
    missing_paths = [path for path in required_paths if not Path(path).exists()]
    if missing_paths:
        reasons.append("artifact_path_missing")
    return CompatibilityCheckResult(
        compatible=not reasons,
        subject="training_runtime_manifest",
        expected_version=expected_schema_version,
        found_version=manifest.schema_version,
        reasons=reasons or ["compatible"],
        metadata={
            "run_id": manifest.run_id,
            "training_kind": manifest.training_kind,
            "missing_paths": missing_paths,
        },
    )


def build_replay_dataset_summary(dataset: ReplayDatasetBundle) -> Dict[str, Any]:
    return dataset.to_summary()


def build_source_domain_coverage(dataset: ReplayDatasetBundle) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    tasks: Dict[str, int] = {}
    for episode in dataset.episodes:
        counts[episode.source_domain] = counts.get(episode.source_domain, 0) + 1
        tasks[episode.task_id] = tasks.get(episode.task_id, 0) + 1
    total = max(1, len(dataset.episodes))
    return {
        "total_episodes": len(dataset.episodes),
        "source_domain_counts": dict(sorted(counts.items())),
        "source_domain_fractions": {
            key: round(value / float(total), 6)
            for key, value in sorted(counts.items())
        },
        "task_counts": dict(sorted(tasks.items())),
    }


def build_replay_trajectory_audits(dataset: ReplayDatasetBundle) -> list[TrajectoryAuditV1]:
    steps_by_episode: Dict[str, list[ReplayStepRecord]] = {}
    episodes_by_id: Dict[str, ReplayEpisodeRecord] = {
        episode.episode_id: episode for episode in dataset.episodes
    }
    for step in dataset.steps:
        steps_by_episode.setdefault(step.episode_id, []).append(step)

    audits: list[TrajectoryAuditV1] = []
    for episode_id, rows in sorted(steps_by_episode.items()):
        rows = sorted(rows, key=lambda row: row.step_idx)
        events: list[str] = []
        penetrations: list[float] = []
        velocities: list[list[float]] = []
        reward_components: Dict[str, list[float]] = {}
        for row in rows:
            events.extend([str(flag.get("flag", "constraint_flag")) for flag in row.constraint_flags])
            penetrations.append(float(row.metadata.get("task_info", {}).get("constraint_error", 0.0) or 0.0))
            velocity_value = float(row.metadata.get("task_info", {}).get("tool_velocity", 0.0) or 0.0)
            velocities.append([velocity_value])
            for key, value in dict(row.reward_decomposition).items():
                try:
                    reward_components.setdefault(str(key), []).append(float(value))
                except (TypeError, ValueError):
                    continue
        episode = episodes_by_id.get(episode_id)
        audits.append(
            create_trajectory_audit(
                episode_id=episode_id,
                num_steps=len(rows),
                actions=[list(map(float, row.action_vector)) for row in rows],
                rewards=[float(row.reward) for row in rows],
                reward_components=reward_components,
                events=events,
                penetrations=penetrations,
                velocities=velocities,
                scene_tracks_sha=(
                    str(episode.provenance.get("scene_tracks_sha"))
                    if episode is not None and episode.provenance.get("scene_tracks_sha")
                    else None
                ),
            )
        )
    return audits


def build_training_runtime_summary_markdown(
    manifest: TrainingRuntimeManifest,
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [
        f"# Training Runtime Summary: {manifest.training_kind}",
        "",
        f"- Run ID: {manifest.run_id}",
        f"- Status: {manifest.status}",
        f"- Seed: {manifest.seed}",
        f"- Plan: {manifest.plan_id}",
        f"- Config digest: {manifest.config_digest}",
        f"- Replay manifest digest: {manifest.replay_manifest_digest or 'n/a'}",
        f"- Receipt coverage labels: {manifest.receipt_label_coverage.get('total_labels', 0)}",
        f"- Budget settlement live: {'yes' if manifest.budget_settlement_live else 'no'}",
        "",
        "## Inferential",
        f"- Learnability contracts: {manifest.inferential_learnability_summary.get('contract_count', 0)}",
        f"- Benchmark receipt-backed: {manifest.inferential_learnability_summary.get('benchmark_receipt_backed_count', 0)}",
        f"- Inferential work orders: {manifest.inferential_work_order_summary.get('work_orders', 0)}",
        "",
        "## Source Coverage",
    ]
    for source_domain, count in sorted(
        dict(manifest.source_domain_coverage.get("source_domain_counts", {})).items()
    ):
        lines.append(f"- {source_domain}: {count}")
    lines.extend(["", "## Artifacts"])
    for artifact_id, path in sorted(manifest.artifact_paths.items()):
        lines.append(f"- {artifact_id}: {path}")
    if manifest.promotion_ledger_path:
        lines.append(f"- promotion_ledger_path: {manifest.promotion_ledger_path}")
    if manifest.budget_settlement_path:
        lines.append(f"- budget_settlement_path: {manifest.budget_settlement_path}")
    if checkpoint_rows:
        lines.extend(["", "## Checkpoints"])
        for row in checkpoint_rows:
            lines.append(
                f"- {row.get('checkpoint_id')}: {row.get('model_version')} @ {row.get('path')}"
            )
    if manifest.failure_reason:
        lines.extend(["", "## Failure", f"- {manifest.failure_reason}"])
    return "\n".join(lines) + "\n"


__all__ = [
    "TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION",
    "TrainingRuntimeManifest",
    "write_training_runtime_manifest",
    "load_training_runtime_manifest",
    "check_training_runtime_manifest_compatibility",
    "build_replay_dataset_summary",
    "build_source_domain_coverage",
    "build_replay_trajectory_audits",
    "build_training_runtime_summary_markdown",
]
