"""Repository-wide WM surface hygiene sweep.

The sweep is intentionally receipt-oriented. It checks for launch/training
claim drift, protected baseline edits, required hygiene tooling, and GPU run
manifest readiness without treating known local scaffolding as execution proof.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.gpu_run_hygiene import (
    validate_gpu_run_manifest_file,
)

WM_SURFACE_HYGIENE_REPORT_VERSION = "wm_surface_hygiene_report_v1"
WM_SURFACE_HYGIENE_RECEIPT_VERSION = "wm_surface_hygiene_receipt_v1"

DEFAULT_TARGET_ROOTS = (
    "src/world_model",
    "src/runtime",
    "src/evidence",
    "src/embodiment",
    "scripts/economic_world_model",
    "docs/economic_world_model",
    ".github/workflows",
    "configs/runpod/examples",
)
PROTECTED_PATHS = {
    "checkpoints/stable_world_model.pt",
    "src/controllers/synthetic_weight_controller.py",
}
REQUIRED_HYGIENE_PATHS = {
    "docs/agent_ergonomics/run_manifest_schema.md",
    "src/world_model/economic_world_model/evidence_hygiene.py",
    "src/world_model/economic_world_model/gpu_run_hygiene.py",
    "scripts/economic_world_model/check_evidence_hygiene.py",
    "scripts/economic_world_model/check_gpu_run_hygiene.py",
    ".github/workflows/economic-world-model-focused.yml",
}
RISKY_TRUE_CLAIM_PATTERN = re.compile(
    r"(?P<key>gpu_training_executed|provider_executed|unitree_hardware_truth|"
    r"promotion_eligible|phase7_authority_granted|launch_authority_granted|"
    r"ready_for_training)\s*[:=]\s*(?P<value>true|True)\b"
)
TODO_COMMENT_PATTERN = re.compile(r"(?i)(#|<!--|//)\s*(TODO|FIXME|XXX)\b")


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


@dataclass(frozen=True)
class WMSurfaceHygieneReceipt:
    receipt_id: str
    check_key: str
    status: str
    passed: bool
    severity: str = "blocking"
    measured_value: Any = None
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = WM_SURFACE_HYGIENE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "check_key": self.check_key,
            "status": self.status,
            "passed": bool(self.passed),
            "severity": self.severity,
            "measured_value": to_json_safe(self.measured_value),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class WMSurfaceHygieneReport:
    report_id: str
    status: str
    scanned_file_count: int
    python_file_count: int
    doc_file_count: int
    manifest_file_count: int
    receipt_count: int
    blocking_issue_count: int
    advisory_issue_count: int
    risky_true_claim_count: int
    protected_change_count: int
    oversized_python_file_count: int
    todo_marker_count: int
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = WM_SURFACE_HYGIENE_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "scanned_file_count": int(self.scanned_file_count),
            "python_file_count": int(self.python_file_count),
            "doc_file_count": int(self.doc_file_count),
            "manifest_file_count": int(self.manifest_file_count),
            "receipt_count": int(self.receipt_count),
            "blocking_issue_count": int(self.blocking_issue_count),
            "advisory_issue_count": int(self.advisory_issue_count),
            "risky_true_claim_count": int(self.risky_true_claim_count),
            "protected_change_count": int(self.protected_change_count),
            "oversized_python_file_count": int(self.oversized_python_file_count),
            "todo_marker_count": int(self.todo_marker_count),
            "output_paths": _mapping(self.output_paths),
            "metadata": _mapping(self.metadata),
        }


def _receipt(
    *,
    check_key: str,
    passed: bool,
    measured_value: Any = None,
    severity: str = "blocking",
    blocker: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMSurfaceHygieneReceipt:
    return WMSurfaceHygieneReceipt(
        receipt_id=_stable_id(
            "wm_surface_hygiene",
            {
                "check_key": check_key,
                "measured_value": measured_value,
                "severity": severity,
            },
        ),
        check_key=check_key,
        status="ok" if passed else "blocked",
        passed=passed,
        severity=severity,
        measured_value=measured_value,
        blockers=[] if passed else [blocker or check_key],
        metadata=dict(metadata or {}),
    )


def _scan_files(repo_root: Path, target_roots: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for root in target_roots:
        path = repo_root / root
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        files.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and "__pycache__" not in candidate.parts
            and candidate.suffix in {".py", ".md", ".json", ".jsonl", ".yml", ".yaml"}
        )
    return sorted(set(files))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _risky_true_claims(files: Sequence[Path]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in files:
        if path.suffix not in {".py", ".md", ".json", ".jsonl", ".yml", ".yaml"}:
            continue
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            stripped = line.strip()
            structured_claim = stripped.startswith(('"', "'", "{", "-", "`")) or bool(
                re.match(r"^[A-Za-z_]+\s*[:=]", stripped)
            )
            if not structured_claim:
                continue
            match = RISKY_TRUE_CLAIM_PATTERN.search(line)
            if match:
                hits.append(
                    {
                        "path": str(path),
                        "line_number": line_number,
                        "claim_key": match.group("key"),
                        "line": line.strip(),
                    }
                )
    return hits


def _todo_markers(files: Sequence[Path]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in files:
        if path.suffix not in {".py", ".md"}:
            continue
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            if TODO_COMMENT_PATTERN.search(line):
                hits.append({"path": str(path), "line_number": line_number})
    return hits


def _oversized_python_files(files: Sequence[Path], threshold: int) -> list[dict[str, Any]]:
    oversized: list[dict[str, Any]] = []
    for path in files:
        if path.suffix != ".py":
            continue
        line_count = len(_read_text(path).splitlines())
        if line_count > threshold:
            oversized.append({"path": str(path), "line_count": line_count})
    return oversized


def _run_manifest_example_receipts(repo_root: Path) -> list[WMSurfaceHygieneReceipt]:
    example_paths = sorted((repo_root / "configs/runpod/examples").glob("*.json"))
    receipts: list[WMSurfaceHygieneReceipt] = []
    for path in example_paths:
        manifest_receipts = validate_gpu_run_manifest_file(path)
        failing = [
            receipt.to_dict()
            for receipt in manifest_receipts
            if not receipt.passed and receipt.severity == "blocking"
        ]
        receipts.append(
            _receipt(
                check_key=f"gpu_run_manifest_example_hygiene::{path.name}",
                passed=not failing,
                measured_value={"receipt_count": len(manifest_receipts)},
                blocker="gpu_run_manifest_example_failed_hygiene",
                metadata={"failing_receipts": failing},
            )
        )
    return receipts


def run_wm_surface_hygiene(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    changed_paths: Optional[Sequence[str]] = None,
    target_roots: Sequence[str] = DEFAULT_TARGET_ROOTS,
    large_python_line_threshold: int = 2_000,
) -> dict[str, Any]:
    """Run a receipt-emitting sweep over WM code, docs, scripts, and manifests."""

    root = Path(repo_root)
    out = Path(output_dir)
    files = _scan_files(root, target_roots)
    python_files = [path for path in files if path.suffix == ".py"]
    doc_files = [path for path in files if path.suffix == ".md"]
    manifest_files = [path for path in files if path.suffix in {".json", ".jsonl"}]

    missing_roots = [path for path in target_roots if not (root / path).exists()]
    missing_hygiene_paths = [
        path for path in sorted(REQUIRED_HYGIENE_PATHS) if not (root / path).exists()
    ]
    changed = [str(path) for path in list(changed_paths or [])]
    protected_changes = [
        path
        for path in changed
        if path in PROTECTED_PATHS or any(path.startswith(f"{item}/") for item in PROTECTED_PATHS)
    ]
    risky_claims = _risky_true_claims(files)
    todo_markers = _todo_markers(files)
    oversized = _oversized_python_files(files, large_python_line_threshold)

    receipts: list[WMSurfaceHygieneReceipt] = [
        _receipt(
            check_key="target_roots_present",
            passed=not missing_roots,
            measured_value=missing_roots,
            blocker="wm_surface_target_root_missing",
        ),
        _receipt(
            check_key="required_hygiene_tools_present",
            passed=not missing_hygiene_paths,
            measured_value=missing_hygiene_paths,
            blocker="required_hygiene_tool_missing",
        ),
        _receipt(
            check_key="protected_baseline_paths_not_modified",
            passed=not protected_changes,
            measured_value=protected_changes,
            blocker="protected_baseline_path_modified",
        ),
        _receipt(
            check_key="risky_true_claims_absent",
            passed=not risky_claims,
            measured_value=risky_claims[:20],
            blocker="risky_execution_claim_true_without_sweep_context",
            metadata={"total": len(risky_claims)},
        ),
        _receipt(
            check_key="oversized_python_files_inventory",
            passed=True,
            measured_value=oversized[:20],
            severity="info",
            metadata={"total": len(oversized), "threshold": large_python_line_threshold},
        ),
        _receipt(
            check_key="todo_marker_inventory",
            passed=True,
            measured_value=todo_markers[:20],
            severity="info",
            metadata={"total": len(todo_markers)},
        ),
    ]
    receipts.extend(_run_manifest_example_receipts(root))

    blocking_issue_count = sum(
        1 for receipt in receipts if not receipt.passed and receipt.severity == "blocking"
    )
    advisory_issue_count = sum(
        1 for receipt in receipts if not receipt.passed and receipt.severity != "blocking"
    )
    output_paths = {
        "receipts_path": str(out / "wm_surface_hygiene_receipts_v1.jsonl"),
        "report_path": str(out / "wm_surface_hygiene_report_v1.json"),
    }
    report = WMSurfaceHygieneReport(
        report_id=_stable_id(
            "wm_surface_hygiene_report",
            {
                "target_roots": list(target_roots),
                "scanned_file_count": len(files),
                "blocking_issue_count": blocking_issue_count,
            },
        ),
        status="ok_wm_surface_hygiene_passed"
        if blocking_issue_count == 0
        else "blocked_wm_surface_hygiene_failed",
        scanned_file_count=len(files),
        python_file_count=len(python_files),
        doc_file_count=len(doc_files),
        manifest_file_count=len(manifest_files),
        receipt_count=len(receipts),
        blocking_issue_count=blocking_issue_count,
        advisory_issue_count=advisory_issue_count,
        risky_true_claim_count=len(risky_claims),
        protected_change_count=len(protected_changes),
        oversized_python_file_count=len(oversized),
        todo_marker_count=len(todo_markers),
        output_paths=output_paths,
        metadata={
            "target_roots": list(target_roots),
            "required_hygiene_paths": sorted(REQUIRED_HYGIENE_PATHS),
            "protected_paths": sorted(PROTECTED_PATHS),
        },
    )
    _write_jsonl(
        Path(output_paths["receipts_path"]),
        [receipt.to_dict() for receipt in receipts],
    )
    _write_json(Path(output_paths["report_path"]), report.to_dict())
    return report.to_dict()


__all__ = [
    "WMSurfaceHygieneReceipt",
    "WMSurfaceHygieneReport",
    "run_wm_surface_hygiene",
]
