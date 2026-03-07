"""Artifact schema/version compatibility helpers for replay and shadow artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.replay.schema import ReplayDatasetManifest
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class CompatibilityCheckResult:
    """Deterministic compatibility check output."""

    compatible: bool
    subject: str
    expected_version: str
    found_version: str
    reasons: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": bool(self.compatible),
            "subject": self.subject,
            "expected_version": self.expected_version,
            "found_version": self.found_version,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


def check_replay_manifest_compatibility(
    manifest: ReplayDatasetManifest,
    *,
    expected_schema_version: str,
    expected_source_adapters: Sequence[str] | None = None,
) -> CompatibilityCheckResult:
    reasons: list[str] = []
    if manifest.schema_version != expected_schema_version:
        reasons.append("schema_version_mismatch")
    expected_adapters = sorted(str(value) for value in (expected_source_adapters or []))
    if expected_adapters and sorted(manifest.source_adapters) != expected_adapters:
        reasons.append("source_adapter_set_mismatch")
    compatible = not reasons
    return CompatibilityCheckResult(
        compatible=compatible,
        subject="replay_dataset_manifest",
        expected_version=expected_schema_version,
        found_version=manifest.schema_version,
        reasons=reasons or ["compatible"],
        metadata={
            "manifest_hash": manifest.manifest_hash,
            "dataset_digest": manifest.dataset_digest,
            "source_adapters": list(manifest.source_adapters),
        },
    )


def build_artifact_schema_fingerprint(artifacts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a stable fingerprint for a set of schema/version-bearing artifacts."""

    normalized = {}
    for artifact_id, payload in sorted(artifacts.items()):
        normalized[str(artifact_id)] = {
            "schema_version": str(payload.get("schema_version", "")),
            "config_digest": str(payload.get("config_digest", "")),
            "dataset_digest": str(payload.get("dataset_digest", "")),
        }
    return {
        "artifacts": normalized,
        "schema_fingerprint": sha256_json(normalized),
    }


def check_artifact_schema_versions(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    required_versions: Mapping[str, str],
) -> list[CompatibilityCheckResult]:
    results: list[CompatibilityCheckResult] = []
    for artifact_id, expected_version in sorted(required_versions.items()):
        payload = dict(artifacts.get(artifact_id, {}) or {})
        found_version = str(payload.get("schema_version", ""))
        reasons = []
        if artifact_id not in artifacts:
            reasons.append("artifact_missing")
        if found_version != str(expected_version):
            reasons.append("schema_version_mismatch")
        results.append(
            CompatibilityCheckResult(
                compatible=not reasons,
                subject=str(artifact_id),
                expected_version=str(expected_version),
                found_version=found_version,
                reasons=reasons or ["compatible"],
                metadata={"artifact_digest": sha256_json(payload)},
            )
        )
    return results


def required_artifacts_present(root_dir: str | Path, required_files: Iterable[str]) -> CompatibilityCheckResult:
    root = Path(root_dir)
    missing = [str(name) for name in required_files if not (root / name).exists()]
    return CompatibilityCheckResult(
        compatible=not missing,
        subject="shadow_artifact_bundle",
        expected_version="present",
        found_version="present" if not missing else "missing",
        reasons=missing or ["compatible"],
        metadata={"root_dir": str(root)},
    )
