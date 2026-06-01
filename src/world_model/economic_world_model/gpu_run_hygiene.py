"""GPU/provider run-manifest hygiene checks.

This module validates manifest-shaped GPU/provider/loop/training runs before
they are treated as launchable evidence. It does not launch providers or run
training; it emits receipts about whether a manifest is structurally safe to
queue and later interpret.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

GPU_RUN_HYGIENE_REPORT_VERSION = "gpu_run_hygiene_report_v1"
GPU_RUN_HYGIENE_RECEIPT_VERSION = "gpu_run_hygiene_receipt_v1"

ALLOWED_MODES = {"local", "codex_cloud", "runpod"}
ALLOWED_RUN_CLASSES = {"loop", "provider", "train", "refactor"}
ALLOWED_POD_CLASSES = {"loop", "provider", "train", "refactor"}
ALLOWED_EPISTEMIC_STATUS = {
    "smoke",
    "proof_of_life",
    "benchmark_candidate",
    "promotion_candidate",
    "deployment_candidate",
}
ALLOWED_STATUS = {"pending", "running", "completed", "failed"}
REQUIRED_FIELDS = {
    "run_id",
    "mode",
    "commit_sha",
    "branch",
    "task",
    "run_class",
    "epistemic_status",
    "config_paths",
    "seeds",
    "image",
    "template",
    "commands",
    "artifact_paths",
    "status",
    "rollback_notes",
    "replay_notes",
}
PROTECTED_PATH_FRAGMENTS = {
    "checkpoints/stable_world_model.pt",
    "src/controllers/synthetic_weight_controller.py",
}
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*="),
    re.compile(r"(?i)--(api[_-]?key|token|password|secret)(=|\s+)"),
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


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


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


@dataclass(frozen=True)
class GPURunHygieneReceipt:
    receipt_id: str
    manifest_path: str
    check_key: str
    status: str
    passed: bool
    severity: str = "blocking"
    measured_value: Any = None
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = GPU_RUN_HYGIENE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "manifest_path": self.manifest_path,
            "check_key": self.check_key,
            "status": self.status,
            "passed": bool(self.passed),
            "severity": self.severity,
            "measured_value": to_json_safe(self.measured_value),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class GPURunHygieneReport:
    report_id: str
    status: str
    manifest_count: int
    receipt_count: int
    blocking_issue_count: int
    advisory_issue_count: int
    safe_to_queue_count: int
    unsafe_to_queue_count: int
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = GPU_RUN_HYGIENE_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "manifest_count": int(self.manifest_count),
            "receipt_count": int(self.receipt_count),
            "blocking_issue_count": int(self.blocking_issue_count),
            "advisory_issue_count": int(self.advisory_issue_count),
            "safe_to_queue_count": int(self.safe_to_queue_count),
            "unsafe_to_queue_count": int(self.unsafe_to_queue_count),
            "output_paths": _mapping(self.output_paths),
            "metadata": _mapping(self.metadata),
        }


def _receipt(
    *,
    manifest_path: Path,
    check_key: str,
    passed: bool,
    measured_value: Any = None,
    severity: str = "blocking",
    blocker: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> GPURunHygieneReceipt:
    return GPURunHygieneReceipt(
        receipt_id=_stable_id(
            "gpu_run_hygiene",
            {
                "manifest_path": str(manifest_path),
                "check_key": check_key,
                "measured_value": measured_value,
            },
        ),
        manifest_path=str(manifest_path),
        check_key=check_key,
        status="ok" if passed else "blocked",
        passed=passed,
        severity=severity,
        measured_value=measured_value,
        blockers=[] if passed else [blocker or check_key],
        metadata=dict(metadata or {}),
    )


def _missing_required_fields(payload: Mapping[str, Any]) -> list[str]:
    missing = [field for field in sorted(REQUIRED_FIELDS) if field not in payload]
    if payload.get("mode") == "runpod" and "pod_class" not in payload:
        missing.append("pod_class")
    return missing


def _generic_checkpoint_paths(paths: Sequence[str]) -> list[str]:
    generic = []
    for path in paths:
        text = str(path).rstrip("/")
        if text in {"checkpoints", "checkpoints/"} or text.startswith("checkpoints/"):
            generic.append(str(path))
    return generic


def _protected_path_hits(values: Sequence[str]) -> list[str]:
    hits: list[str] = []
    for value in values:
        for fragment in PROTECTED_PATH_FRAGMENTS:
            if fragment in value:
                hits.append(value)
    return hits


def _secret_hits(commands: Sequence[str]) -> list[str]:
    hits: list[str] = []
    for command in commands:
        if any(pattern.search(command) for pattern in SECRET_PATTERNS):
            hits.append(command)
    return hits


def _comparison_artifact_present(paths: Sequence[str]) -> bool:
    return any(
        "comparison" in path or "benchmark" in path or "promotion" in path
        for path in paths
    )


def validate_gpu_run_manifest_payload(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path = "<memory>",
) -> list[GPURunHygieneReceipt]:
    """Return hygiene receipts for one run manifest payload."""

    path = Path(manifest_path)
    receipts: list[GPURunHygieneReceipt] = []
    missing = _missing_required_fields(payload)
    receipts.append(
        _receipt(
            manifest_path=path,
            check_key="required_fields_present",
            passed=not missing,
            measured_value=missing,
            blocker="missing_required_manifest_fields",
        )
    )

    mode = str(payload.get("mode", ""))
    run_class = str(payload.get("run_class", ""))
    pod_class = payload.get("pod_class")
    epistemic_status = str(payload.get("epistemic_status", ""))
    status = str(payload.get("status", ""))
    commands = _strings(payload.get("commands"))
    artifact_paths = _strings(payload.get("artifact_paths"))
    config_paths = _strings(payload.get("config_paths"))
    dependency_chain = _strings(payload.get("dependency_chain"))
    seeds = payload.get("seeds")

    receipts.extend(
        [
            _receipt(
                manifest_path=path,
                check_key="mode_valid",
                passed=mode in ALLOWED_MODES,
                measured_value=mode,
                blocker="invalid_execution_mode",
            ),
            _receipt(
                manifest_path=path,
                check_key="run_class_valid",
                passed=run_class in ALLOWED_RUN_CLASSES,
                measured_value=run_class,
                blocker="invalid_or_missing_run_class",
            ),
            _receipt(
                manifest_path=path,
                check_key="epistemic_status_valid",
                passed=epistemic_status in ALLOWED_EPISTEMIC_STATUS,
                measured_value=epistemic_status,
                blocker="invalid_or_missing_epistemic_status",
            ),
            _receipt(
                manifest_path=path,
                check_key="status_valid",
                passed=status in ALLOWED_STATUS,
                measured_value=status,
                blocker="invalid_run_status",
            ),
        ]
    )
    if mode == "runpod":
        receipts.append(
            _receipt(
                manifest_path=path,
                check_key="pod_class_valid_for_runpod",
                passed=str(pod_class) in ALLOWED_POD_CLASSES,
                measured_value=pod_class,
                blocker="invalid_or_missing_pod_class",
            )
        )

    receipts.extend(
        [
            _receipt(
                manifest_path=path,
                check_key="commands_present",
                passed=bool(commands),
                measured_value=len(commands),
                blocker="commands_missing",
            ),
            _receipt(
                manifest_path=path,
                check_key="artifact_paths_present",
                passed=bool(artifact_paths),
                measured_value=len(artifact_paths),
                blocker="artifact_paths_missing",
            ),
            _receipt(
                manifest_path=path,
                check_key="config_paths_present",
                passed=bool(config_paths) or run_class in {"refactor", "loop"},
                measured_value=len(config_paths),
                blocker="config_paths_missing",
            ),
            _receipt(
                manifest_path=path,
                check_key="dependency_chain_present",
                passed=bool(dependency_chain) or run_class == "refactor",
                measured_value=len(dependency_chain),
                blocker="dependency_chain_missing",
            ),
            _receipt(
                manifest_path=path,
                check_key="seeds_declared",
                passed=isinstance(seeds, list),
                measured_value=seeds,
                blocker="seeds_not_declared_as_list",
            ),
        ]
    )

    generic_checkpoint_paths = _generic_checkpoint_paths(artifact_paths)
    receipts.append(
        _receipt(
            manifest_path=path,
            check_key="no_generic_checkpoint_sink",
            passed=not generic_checkpoint_paths,
            measured_value=generic_checkpoint_paths,
            blocker="generic_checkpoint_sink_forbidden",
        )
    )
    protected_hits = _protected_path_hits([*commands, *artifact_paths, *config_paths])
    receipts.append(
        _receipt(
            manifest_path=path,
            check_key="protected_baseline_paths_untouched",
            passed=not protected_hits,
            measured_value=protected_hits,
            blocker="protected_baseline_path_referenced",
        )
    )
    secret_hits = _secret_hits(commands)
    receipts.append(
        _receipt(
            manifest_path=path,
            check_key="commands_do_not_inline_secrets",
            passed=not secret_hits,
            measured_value=len(secret_hits),
            blocker="command_contains_inline_secret",
        )
    )

    if status == "pending":
        runtime_fields_empty = all(
            payload.get(field) in (None, "", [], {})
            for field in ("pod_id", "started_at", "finished_at")
        )
        receipts.append(
            _receipt(
                manifest_path=path,
                check_key="pending_manifest_has_no_runtime_truth",
                passed=runtime_fields_empty,
                measured_value={
                    "pod_id": payload.get("pod_id"),
                    "started_at": payload.get("started_at"),
                    "finished_at": payload.get("finished_at"),
                },
                blocker="pending_manifest_contains_runtime_truth",
            )
        )
    if status == "completed":
        completed_fields_present = bool(
            payload.get("started_at")
            and payload.get("finished_at")
            and payload.get("cost_snapshot")
        )
        receipts.append(
            _receipt(
                manifest_path=path,
                check_key="completed_manifest_has_runtime_receipts",
                passed=completed_fields_present,
                measured_value={
                    "started_at": payload.get("started_at"),
                    "finished_at": payload.get("finished_at"),
                    "cost_snapshot_present": bool(payload.get("cost_snapshot")),
                },
                blocker="completed_manifest_missing_runtime_receipts",
            )
        )

    high_status = epistemic_status in {"promotion_candidate", "deployment_candidate"}
    receipts.append(
        _receipt(
            manifest_path=path,
            check_key="high_epistemic_status_has_comparison_artifact",
            passed=(not high_status) or _comparison_artifact_present(artifact_paths),
            measured_value={"epistemic_status": epistemic_status},
            blocker="promotion_or_deployment_candidate_missing_comparison_artifact",
        )
    )
    receipts.append(
        _receipt(
            manifest_path=path,
            check_key="rollback_and_replay_notes_present",
            passed=bool(str(payload.get("rollback_notes", "")))
            and bool(str(payload.get("replay_notes", ""))),
            measured_value={
                "rollback_notes": bool(str(payload.get("rollback_notes", ""))),
                "replay_notes": bool(str(payload.get("replay_notes", ""))),
            },
            blocker="rollback_or_replay_notes_missing",
        )
    )
    return receipts


def validate_gpu_run_manifest_file(path: str | Path) -> list[GPURunHygieneReceipt]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return [
            _receipt(
                manifest_path=manifest_path,
                check_key="manifest_json_object",
                passed=False,
                measured_value=type(payload).__name__,
                blocker="manifest_payload_not_object",
            )
        ]
    return validate_gpu_run_manifest_payload(payload, manifest_path=manifest_path)


def run_gpu_run_hygiene(
    *,
    manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate one or more GPU/provider/loop/training manifests."""

    out = Path(output_dir)
    all_receipts: list[GPURunHygieneReceipt] = []
    safe_to_queue_count = 0
    for path in manifest_paths:
        receipts = validate_gpu_run_manifest_file(path)
        all_receipts.extend(receipts)
        if all(receipt.passed for receipt in receipts if receipt.severity == "blocking"):
            safe_to_queue_count += 1

    blocking_issue_count = sum(
        1
        for receipt in all_receipts
        if not receipt.passed and receipt.severity == "blocking"
    )
    advisory_issue_count = sum(
        1
        for receipt in all_receipts
        if not receipt.passed and receipt.severity != "blocking"
    )
    output_paths = {
        "receipts_path": str(out / "gpu_run_hygiene_receipts_v1.jsonl"),
        "report_path": str(out / "gpu_run_hygiene_report_v1.json"),
    }
    report = GPURunHygieneReport(
        report_id=_stable_id(
            "gpu_run_hygiene_report",
            {
                "manifest_paths": [str(path) for path in manifest_paths],
                "blocking_issue_count": blocking_issue_count,
            },
        ),
        status="ok_gpu_run_hygiene_passed"
        if blocking_issue_count == 0
        else "blocked_gpu_run_hygiene_failed",
        manifest_count=len(manifest_paths),
        receipt_count=len(all_receipts),
        blocking_issue_count=blocking_issue_count,
        advisory_issue_count=advisory_issue_count,
        safe_to_queue_count=safe_to_queue_count,
        unsafe_to_queue_count=len(manifest_paths) - safe_to_queue_count,
        output_paths=output_paths,
        metadata={
            "allowed_modes": sorted(ALLOWED_MODES),
            "allowed_run_classes": sorted(ALLOWED_RUN_CLASSES),
            "allowed_epistemic_status": sorted(ALLOWED_EPISTEMIC_STATUS),
        },
    )
    _write_jsonl(
        Path(output_paths["receipts_path"]),
        [receipt.to_dict() for receipt in all_receipts],
    )
    _write_json(Path(output_paths["report_path"]), report.to_dict())
    return report.to_dict()


__all__ = [
    "GPURunHygieneReceipt",
    "GPURunHygieneReport",
    "run_gpu_run_hygiene",
    "validate_gpu_run_manifest_file",
    "validate_gpu_run_manifest_payload",
]
