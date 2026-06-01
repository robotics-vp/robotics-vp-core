"""Evidence hygiene checks for Economic WM artifacts.

The checker is intentionally conservative: GPU, provider, hardware, promotion,
and launch claims must either stay false or carry concrete artifact evidence.
It also verifies artifact references and local retention bounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

EVIDENCE_HYGIENE_REPORT_VERSION = "economic_wm_evidence_hygiene_report_v1"
CLAIM_VS_EVIDENCE_RECEIPT_VERSION = "claim_vs_evidence_receipt_v1"
STALE_ARTIFACT_RECEIPT_VERSION = "stale_artifact_receipt_v1"
ARTIFACT_RETENTION_RECEIPT_VERSION = "artifact_retention_receipt_v1"

DEFAULT_MAX_LOCAL_ARTIFACT_BYTES = 50_000_000
DEFAULT_ADVISORY_LOCAL_ARTIFACT_BYTES = 10_000_000
ALLOWED_EMPTY_JSONL_NAMES = {"video_file_receipts.jsonl"}
EXCLUDED_ARTIFACT_DIR_NAMES = {
    "evidence_hygiene",
    "gpu_run_hygiene",
    "wm_surface_hygiene",
}

RISKY_CLAIM_EVIDENCE_KEYS: dict[str, tuple[str, ...]] = {
    "provider_executed": (
        "provider_receipt_path",
        "provider_runtime_manifest_path",
        "provider_run_manifest_path",
    ),
    "gpu_training_executed": (
        "gpu_run_manifest_path",
        "training_receipt_path",
        "training_manifest_path",
    ),
    "unitree_hardware_truth": (
        "hardware_receipt_path",
        "unitree_hardware_receipt_path",
        "unitree_trace_receipt_path",
    ),
    "promotion_eligible": (
        "benchmark_gate_receipt_path",
        "promotion_gate_receipt_path",
        "promotion_evidence_path",
    ),
    "phase7_authority_granted": (
        "phase7_authority_receipt_path",
        "lower_wm_evidence_receipt_path",
    ),
    "launch_authority_granted": (
        "launch_authority_receipt_path",
        "remote_run_manifest_path",
    ),
    "ready_for_training": (
        "training_manifest_path",
        "training_data_quality_receipt_path",
        "benchmark_gate_receipt_path",
    ),
}

LOCAL_REFERENCE_KEYS = {
    "artifact_ref",
    "artifact_refs",
    "artifact_path",
    "artifact_paths",
    "report_path",
    "manifest_path",
    "manifest_ref",
}
REFERENCE_SCHEME_PREFIXES = (
    "http://",
    "https://",
    "s3://",
    "gs://",
    "hf://",
    "wandb://",
    "intrinsics://",
    "extrinsics://",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_mapping(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ClaimEvidenceReceipt:
    receipt_id: str
    artifact_path: str
    claim_key: str
    claim_value: bool
    status: str
    passed: bool
    evidence_keys_required: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = CLAIM_VS_EVIDENCE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "claim_key": self.claim_key,
            "claim_value": bool(self.claim_value),
            "status": self.status,
            "passed": bool(self.passed),
            "evidence_keys_required": list(self.evidence_keys_required),
            "evidence_refs": list(self.evidence_refs),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class StaleArtifactReceipt:
    receipt_id: str
    artifact_path: str
    check_key: str
    status: str
    passed: bool
    referenced_path: str = ""
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = STALE_ARTIFACT_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "check_key": self.check_key,
            "status": self.status,
            "passed": bool(self.passed),
            "referenced_path": self.referenced_path,
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactRetentionReceipt:
    receipt_id: str
    artifact_path: str
    status: str
    passed: bool
    size_bytes: int
    retention_tier: str
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = ARTIFACT_RETENTION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "status": self.status,
            "passed": bool(self.passed),
            "size_bytes": int(self.size_bytes),
            "retention_tier": self.retention_tier,
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class EvidenceHygieneReport:
    report_id: str
    artifact_root: str
    status: str
    scanned_file_count: int
    claim_receipt_count: int
    stale_receipt_count: int
    retention_receipt_count: int
    blocking_issue_count: int
    advisory_issue_count: int
    provider_gpu_hardware_claims_blocked: bool
    artifact_refs_resolved: bool
    retention_policy_passed: bool
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EVIDENCE_HYGIENE_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "artifact_root": self.artifact_root,
            "status": self.status,
            "scanned_file_count": int(self.scanned_file_count),
            "claim_receipt_count": int(self.claim_receipt_count),
            "stale_receipt_count": int(self.stale_receipt_count),
            "retention_receipt_count": int(self.retention_receipt_count),
            "blocking_issue_count": int(self.blocking_issue_count),
            "advisory_issue_count": int(self.advisory_issue_count),
            "provider_gpu_hardware_claims_blocked": bool(
                self.provider_gpu_hardware_claims_blocked
            ),
            "artifact_refs_resolved": bool(self.artifact_refs_resolved),
            "retention_policy_passed": bool(self.retention_policy_passed),
            "output_paths": _mapping(self.output_paths),
            "metadata": _mapping(self.metadata),
        }


def _artifact_files(artifact_root: Path) -> list[Path]:
    suffixes = {".json", ".jsonl"}
    return sorted(
        path
        for path in artifact_root.rglob("*")
        if path.suffix in suffixes
        and not any(part in EXCLUDED_ARTIFACT_DIR_NAMES for part in path.parts)
    )


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [_mapping(payload)] if isinstance(payload, Mapping) else []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, Mapping):
                rows.append(_mapping(row))
    return rows


def _walk_mappings(
    payload: Any,
    *,
    prefix: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(payload, Mapping):
        yield prefix, payload
        for key, value in payload.items():
            yield from _walk_mappings(value, prefix=prefix + (str(key),))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk_mappings(value, prefix=prefix + (str(index),))


def _walk_values(
    payload: Any,
    *,
    prefix: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            current = prefix + (key_text,)
            yield current, key_text, value
            yield from _walk_values(value, prefix=current)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk_values(value, prefix=prefix + (str(index),))


def _evidence_refs_for_claim(
    payload: Mapping[str, Any],
    claim_key: str,
    evidence_keys: Sequence[str],
) -> list[str]:
    refs: list[str] = []
    for _prefix, key, value in _walk_values(payload):
        if key not in evidence_keys:
            continue
        if isinstance(value, str) and value:
            refs.append(value)
        elif isinstance(value, list):
            refs.extend(str(item) for item in value if str(item))
    artifact_refs = payload.get("artifact_refs")
    if isinstance(artifact_refs, Mapping):
        for key in evidence_keys:
            value = artifact_refs.get(key)
            if isinstance(value, str) and value:
                refs.append(value)
    if claim_key == "ready_for_training":
        refs.extend(_strings(payload.get("training_receipts")))
    return sorted(set(refs))


def _claim_receipts_for_payload(
    *,
    artifact_path: Path,
    payload: Mapping[str, Any],
    row_index: int,
) -> list[ClaimEvidenceReceipt]:
    receipts: list[ClaimEvidenceReceipt] = []
    for prefix, mapping in _walk_mappings(payload):
        for claim_key, evidence_keys in RISKY_CLAIM_EVIDENCE_KEYS.items():
            if claim_key not in mapping:
                continue
            claim_value = bool(mapping.get(claim_key))
            evidence_refs = _evidence_refs_for_claim(mapping, claim_key, evidence_keys)
            passed = (not claim_value) or bool(evidence_refs)
            blockers = [] if passed else [f"{claim_key}_missing_required_evidence"]
            status = "ok_false_or_evidenced" if passed else "blocked_missing_evidence"
            receipts.append(
                ClaimEvidenceReceipt(
                    receipt_id=_stable_id(
                        "claim_vs_evidence",
                        {
                            "path": str(artifact_path),
                            "row_index": row_index,
                            "prefix": prefix,
                            "claim_key": claim_key,
                            "claim_value": claim_value,
                        },
                    ),
                    artifact_path=str(artifact_path),
                    claim_key=claim_key,
                    claim_value=claim_value,
                    status=status,
                    passed=passed,
                    evidence_keys_required=list(evidence_keys),
                    evidence_refs=evidence_refs,
                    blockers=blockers,
                    metadata={"payload_prefix": list(prefix), "row_index": row_index},
                )
            )
    return receipts


def _looks_like_local_reference(value: str) -> bool:
    if not value or "{" in value or "}" in value:
        return False
    lowered = value.lower()
    if lowered.startswith(REFERENCE_SCHEME_PREFIXES):
        return False
    return (
        value.startswith("/")
        or value.startswith("./")
        or value.startswith("artifacts/")
        or value.endswith((".json", ".jsonl", ".md", ".txt", ".parquet"))
    )


def _resolve_reference(
    *,
    artifact_root: Path,
    artifact_path: Path,
    value: str,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    artifact_relative = artifact_path.parent / path
    if artifact_relative.exists():
        return artifact_relative
    return artifact_root / path


def _stale_receipts_for_payload(
    *,
    artifact_root: Path,
    artifact_path: Path,
    payload: Mapping[str, Any],
    row_index: int,
) -> list[StaleArtifactReceipt]:
    receipts: list[StaleArtifactReceipt] = []
    for prefix, key, value in _walk_values(payload):
        candidates: list[str] = []
        if key == "artifact_refs" and len(prefix) == 1 and isinstance(value, Mapping):
            candidates = [
                str(item)
                for item in value.values()
                if isinstance(item, str) and _looks_like_local_reference(item)
            ]
        elif len(prefix) == 1 and key in LOCAL_REFERENCE_KEYS and isinstance(value, str):
            candidates = [value]
        elif len(prefix) == 1 and key in LOCAL_REFERENCE_KEYS and isinstance(value, list):
            candidates = [str(item) for item in value if isinstance(item, str)]
        for candidate in candidates:
            if not _looks_like_local_reference(candidate):
                continue
            resolved = _resolve_reference(
                artifact_root=artifact_root,
                artifact_path=artifact_path,
                value=candidate,
            )
            external_source_ref = candidate.startswith("/tmp/")
            passed = resolved.exists() or external_source_ref
            status = "ok_reference_exists"
            if external_source_ref and not resolved.exists():
                status = "ok_external_source_ref_not_retained"
            elif not passed:
                status = "blocked_missing_ref"
            receipts.append(
                StaleArtifactReceipt(
                    receipt_id=_stable_id(
                        "stale_artifact",
                        {
                            "path": str(artifact_path),
                            "row_index": row_index,
                            "prefix": prefix,
                            "candidate": candidate,
                        },
                    ),
                    artifact_path=str(artifact_path),
                    check_key="local_artifact_reference_exists",
                    status=status,
                    passed=passed,
                    referenced_path=str(resolved),
                    blockers=[] if passed else ["missing_local_artifact_reference"],
                    metadata={
                        "payload_prefix": list(prefix),
                        "row_index": row_index,
                        "raw_reference": candidate,
                        "external_source_ref": external_source_ref,
                    },
                )
            )
    return receipts


def _retention_receipt_for_file(
    *,
    path: Path,
    max_local_artifact_bytes: int,
    advisory_local_artifact_bytes: int,
) -> ArtifactRetentionReceipt:
    size_bytes = path.stat().st_size
    if size_bytes > max_local_artifact_bytes:
        passed = False
        tier = "external_storage_required"
        blockers = ["local_artifact_exceeds_retention_policy"]
    elif size_bytes > advisory_local_artifact_bytes:
        passed = True
        tier = "large_local_artifact_advisory"
        blockers = []
    else:
        passed = True
        tier = "local_receipt_artifact"
        blockers = []
    return ArtifactRetentionReceipt(
        receipt_id=_stable_id(
            "artifact_retention",
            {"path": str(path), "size_bytes": size_bytes, "tier": tier},
        ),
        artifact_path=str(path),
        status="ok_retention_policy" if passed else "blocked_retention_policy",
        passed=passed,
        size_bytes=size_bytes,
        retention_tier=tier,
        blockers=blockers,
        metadata={
            "max_local_artifact_bytes": max_local_artifact_bytes,
            "advisory_local_artifact_bytes": advisory_local_artifact_bytes,
        },
    )


def run_economic_wm_evidence_hygiene(
    *,
    artifact_root: str | Path,
    output_dir: str | Path,
    max_local_artifact_bytes: int = DEFAULT_MAX_LOCAL_ARTIFACT_BYTES,
    advisory_local_artifact_bytes: int = DEFAULT_ADVISORY_LOCAL_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Scan Economic WM artifacts for unsafe claims, stale refs, and size drift."""

    root = Path(artifact_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = _artifact_files(root)

    claim_receipts: list[ClaimEvidenceReceipt] = []
    stale_receipts: list[StaleArtifactReceipt] = []
    retention_receipts: list[ArtifactRetentionReceipt] = []
    parse_failures: list[StaleArtifactReceipt] = []
    for path in files:
        retention_receipts.append(
            _retention_receipt_for_file(
                path=path,
                max_local_artifact_bytes=max_local_artifact_bytes,
                advisory_local_artifact_bytes=advisory_local_artifact_bytes,
            )
        )
        try:
            payloads = _load_payloads(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            parse_failures.append(
                StaleArtifactReceipt(
                    receipt_id=_stable_id(
                        "stale_artifact_parse",
                        {"path": str(path), "error": str(exc)},
                    ),
                    artifact_path=str(path),
                    check_key="artifact_json_parseable",
                    status="blocked_unparseable",
                    passed=False,
                    blockers=["artifact_json_unparseable"],
                    metadata={"error": str(exc)},
                )
            )
            continue
        if (
            path.suffix == ".jsonl"
            and not payloads
            and path.name not in ALLOWED_EMPTY_JSONL_NAMES
        ):
            parse_failures.append(
                StaleArtifactReceipt(
                    receipt_id=_stable_id("empty_jsonl", {"path": str(path)}),
                    artifact_path=str(path),
                    check_key="artifact_jsonl_nonempty",
                    status="blocked_empty_jsonl",
                    passed=False,
                    blockers=["artifact_jsonl_empty"],
                )
            )
        for row_index, payload in enumerate(payloads):
            claim_receipts.extend(
                _claim_receipts_for_payload(
                    artifact_path=path,
                    payload=payload,
                    row_index=row_index,
                )
            )
            stale_receipts.extend(
                _stale_receipts_for_payload(
                    artifact_root=root,
                    artifact_path=path,
                    payload=payload,
                    row_index=row_index,
                )
            )
    stale_receipts.extend(parse_failures)

    blocking_issue_count = (
        sum(1 for receipt in claim_receipts if not receipt.passed)
        + sum(1 for receipt in stale_receipts if not receipt.passed)
        + sum(1 for receipt in retention_receipts if not receipt.passed)
    )
    advisory_issue_count = sum(
        1
        for receipt in retention_receipts
        if receipt.retention_tier == "large_local_artifact_advisory"
    )
    output_paths = {
        "claim_vs_evidence_receipts_path": str(
            out / "claim_vs_evidence_receipts_v1.jsonl"
        ),
        "stale_artifact_receipts_path": str(out / "stale_artifact_receipts_v1.jsonl"),
        "artifact_retention_receipts_path": str(
            out / "artifact_retention_receipts_v1.jsonl"
        ),
        "report_path": str(out / "evidence_hygiene_report_v1.json"),
    }
    report = EvidenceHygieneReport(
        report_id=_stable_id(
            "economic_wm_evidence_hygiene_report",
            {
                "artifact_root": str(root),
                "files": [str(path) for path in files],
                "blocking_issue_count": blocking_issue_count,
            },
        ),
        artifact_root=str(root),
        status="ok_evidence_hygiene_passed"
        if blocking_issue_count == 0
        else "blocked_evidence_hygiene_failed",
        scanned_file_count=len(files),
        claim_receipt_count=len(claim_receipts),
        stale_receipt_count=len(stale_receipts),
        retention_receipt_count=len(retention_receipts),
        blocking_issue_count=blocking_issue_count,
        advisory_issue_count=advisory_issue_count,
        provider_gpu_hardware_claims_blocked=all(
            receipt.passed for receipt in claim_receipts
        ),
        artifact_refs_resolved=all(receipt.passed for receipt in stale_receipts),
        retention_policy_passed=all(receipt.passed for receipt in retention_receipts),
        output_paths=output_paths,
        metadata={
            "risky_claim_keys": sorted(RISKY_CLAIM_EVIDENCE_KEYS),
            "max_local_artifact_bytes": max_local_artifact_bytes,
            "advisory_local_artifact_bytes": advisory_local_artifact_bytes,
        },
    )
    _write_jsonl(
        Path(output_paths["claim_vs_evidence_receipts_path"]),
        [receipt.to_dict() for receipt in claim_receipts],
    )
    _write_jsonl(
        Path(output_paths["stale_artifact_receipts_path"]),
        [receipt.to_dict() for receipt in stale_receipts],
    )
    _write_jsonl(
        Path(output_paths["artifact_retention_receipts_path"]),
        [receipt.to_dict() for receipt in retention_receipts],
    )
    _write_json(Path(output_paths["report_path"]), report.to_dict())
    return report.to_dict()


def load_evidence_hygiene_report(path: str | Path) -> EvidenceHygieneReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvidenceHygieneReport(
        report_id=str(payload.get("report_id", "")),
        artifact_root=str(payload.get("artifact_root", "")),
        status=str(payload.get("status", "")),
        scanned_file_count=int(payload.get("scanned_file_count", 0) or 0),
        claim_receipt_count=int(payload.get("claim_receipt_count", 0) or 0),
        stale_receipt_count=int(payload.get("stale_receipt_count", 0) or 0),
        retention_receipt_count=int(payload.get("retention_receipt_count", 0) or 0),
        blocking_issue_count=int(payload.get("blocking_issue_count", 0) or 0),
        advisory_issue_count=int(payload.get("advisory_issue_count", 0) or 0),
        provider_gpu_hardware_claims_blocked=bool(
            payload.get("provider_gpu_hardware_claims_blocked", False)
        ),
        artifact_refs_resolved=bool(payload.get("artifact_refs_resolved", False)),
        retention_policy_passed=bool(payload.get("retention_policy_passed", False)),
        output_paths={
            str(key): str(value)
            for key, value in dict(payload.get("output_paths", {}) or {}).items()
        },
        metadata=_mapping(payload.get("metadata")),
        version=str(payload.get("version", EVIDENCE_HYGIENE_REPORT_VERSION)),
    )


__all__ = [
    "ArtifactRetentionReceipt",
    "ClaimEvidenceReceipt",
    "EvidenceHygieneReport",
    "StaleArtifactReceipt",
    "load_evidence_hygiene_report",
    "run_economic_wm_evidence_hygiene",
]
