"""Deterministic checkpoint registry for regal-aware training jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.replay.compatibility import CompatibilityCheckResult
from src.utils.config_digest import sha256_file, sha256_json


CHECKPOINT_REGISTRY_SCHEMA_VERSION = "training_checkpoint_registry_v1"


@dataclass(frozen=True)
class CheckpointRecord:
    """Single registered checkpoint emitted by a training job."""

    checkpoint_id: str
    model_family: str
    model_version: str
    path: str
    file_name: str
    artifact_digest: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    is_best: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "path": self.path,
            "file_name": self.file_name,
            "artifact_digest": self.artifact_digest,
            "step": self.step,
            "epoch": self.epoch,
            "is_best": bool(self.is_best),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointRecord":
        return cls(
            checkpoint_id=str(payload.get("checkpoint_id", "")),
            model_family=str(payload.get("model_family", "")),
            model_version=str(payload.get("model_version", "")),
            path=str(payload.get("path", "")),
            file_name=str(payload.get("file_name", "")),
            artifact_digest=str(payload.get("artifact_digest", "")),
            step=int(payload["step"]) if payload.get("step") is not None else None,
            epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
            is_best=bool(payload.get("is_best", False)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class CheckpointRegistry:
    """Stable registry for all checkpoints produced in a training run."""

    schema_version: str
    run_id: str
    training_kind: str
    created_at: str
    checkpoints: list[CheckpointRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def registry_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "training_kind": self.training_kind,
            "created_at": self.created_at,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointRegistry":
        return cls(
            schema_version=str(payload.get("schema_version", CHECKPOINT_REGISTRY_SCHEMA_VERSION)),
            run_id=str(payload.get("run_id", "")),
            training_kind=str(payload.get("training_kind", "")),
            created_at=str(payload.get("created_at", "")),
            checkpoints=[
                CheckpointRecord.from_dict(row)
                for row in list(payload.get("checkpoints", []) or [])
            ],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def build_checkpoint_record(
    *,
    checkpoint_id: str,
    model_family: str,
    model_version: str,
    path: str | Path,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    is_best: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CheckpointRecord:
    checkpoint_path = Path(path)
    artifact_digest = (
        sha256_file(checkpoint_path)
        if checkpoint_path.exists() and checkpoint_path.is_file()
        else sha256_json(
            {
                "checkpoint_id": checkpoint_id,
                "model_family": model_family,
                "model_version": model_version,
                "path": str(checkpoint_path),
                "metadata": dict(metadata or {}),
            }
        )
    )
    return CheckpointRecord(
        checkpoint_id=str(checkpoint_id),
        model_family=str(model_family),
        model_version=str(model_version),
        path=str(checkpoint_path),
        file_name=checkpoint_path.name,
        artifact_digest=artifact_digest,
        step=step,
        epoch=epoch,
        is_best=bool(is_best),
        metadata=dict(metadata or {}),
    )


def create_checkpoint_registry(
    *,
    run_id: str,
    training_kind: str,
    checkpoints: Sequence[CheckpointRecord],
    metadata: Optional[Mapping[str, Any]] = None,
) -> CheckpointRegistry:
    return CheckpointRegistry(
        schema_version=CHECKPOINT_REGISTRY_SCHEMA_VERSION,
        run_id=str(run_id),
        training_kind=str(training_kind),
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoints=sorted(
            list(checkpoints),
            key=lambda checkpoint: (
                checkpoint.model_family,
                checkpoint.model_version,
                checkpoint.file_name,
                checkpoint.checkpoint_id,
            ),
        ),
        metadata=dict(metadata or {}),
    )


def write_checkpoint_registry(path: str | Path, registry: CheckpointRegistry) -> str:
    output_path = Path(path)
    output_path.write_text(
        __import__("json").dumps(registry.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sha256_file(output_path)


def load_checkpoint_registry(path: str | Path) -> CheckpointRegistry:
    payload = __import__("json").loads(Path(path).read_text(encoding="utf-8"))
    return CheckpointRegistry.from_dict(payload)


def check_checkpoint_registry_compatibility(
    registry: CheckpointRegistry,
    *,
    expected_schema_version: str = CHECKPOINT_REGISTRY_SCHEMA_VERSION,
) -> CompatibilityCheckResult:
    reasons: list[str] = []
    if registry.schema_version != expected_schema_version:
        reasons.append("schema_version_mismatch")
    missing_files = [checkpoint.path for checkpoint in registry.checkpoints if not Path(checkpoint.path).exists()]
    if missing_files:
        reasons.append("checkpoint_file_missing")
    return CompatibilityCheckResult(
        compatible=not reasons,
        subject="training_checkpoint_registry",
        expected_version=expected_schema_version,
        found_version=registry.schema_version,
        reasons=reasons or ["compatible"],
        metadata={
            "run_id": registry.run_id,
            "training_kind": registry.training_kind,
            "checkpoint_count": len(registry.checkpoints),
            "missing_files": missing_files,
        },
    )


__all__ = [
    "CHECKPOINT_REGISTRY_SCHEMA_VERSION",
    "CheckpointRecord",
    "CheckpointRegistry",
    "build_checkpoint_record",
    "create_checkpoint_registry",
    "write_checkpoint_registry",
    "load_checkpoint_registry",
    "check_checkpoint_registry_compatibility",
]
