"""Validation reports for Economic WM provider runbook templates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.provider_runbook import (
    EconomicWMProviderRunTemplate,
    EconomicWMProviderRunbook,
    load_economic_wm_provider_runbook,
)

ECONOMIC_WM_PROVIDER_RUNBOOK_VALIDATION_VERSION = (
    "economic_wm_provider_runbook_validation_v1"
)

_REQUIRED_TEMPLATE_KEYS = {
    "non_stub_teacher_runtime_invocation",
    "provider_runtime_truth_receipts",
    "promotion_grade_benchmark_evidence",
    "gpu_training_runtime_receipt",
    "replay_row_linkage_integrity",
}


@dataclass(frozen=True)
class EconomicWMProviderRunbookValidationReport:
    """Validation result for template-only provider runbooks."""

    validation_id: str
    runbook_id: str
    status: str
    safe_for_template_storage: bool
    safe_for_launch: bool = False
    error_count: int = 0
    warning_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_template_ids: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_PROVIDER_RUNBOOK_VALIDATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "version": self.version,
            "runbook_id": self.runbook_id,
            "status": self.status,
            "safe_for_template_storage": bool(self.safe_for_template_storage),
            "safe_for_launch": bool(self.safe_for_launch),
            "error_count": int(self.error_count),
            "warning_count": int(self.warning_count),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checked_template_ids": list(self.checked_template_ids),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMProviderRunbookValidationReport":
        return cls(
            validation_id=str(payload.get("validation_id", "")),
            runbook_id=str(payload.get("runbook_id", "")),
            status=str(payload.get("status", "failed")),
            safe_for_template_storage=bool(
                payload.get("safe_for_template_storage", False)
            ),
            safe_for_launch=bool(payload.get("safe_for_launch", False)),
            error_count=int(payload.get("error_count", 0) or 0),
            warning_count=int(payload.get("warning_count", 0) or 0),
            errors=[str(item) for item in list(payload.get("errors", []) or [])],
            warnings=[str(item) for item in list(payload.get("warnings", []) or [])],
            checked_template_ids=[
                str(item)
                for item in list(payload.get("checked_template_ids", []) or [])
            ],
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_PROVIDER_RUNBOOK_VALIDATION_VERSION)
            ),
        )


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _template_lookup(
    template_payloads: Iterable[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for item in template_payloads:
        template_id = str(item.get("template_id", ""))
        if template_id:
            lookup[template_id] = item
    return lookup


def _validate_manifest_stub(
    *,
    template: EconomicWMProviderRunTemplate,
    manifest_stub: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    prefix = f"template {template.template_id} ({template.requirement_key})"
    if manifest_stub.get("status") != "pending":
        errors.append(f"{prefix}: manifest status must remain pending")
    if not str(manifest_stub.get("task", "")).startswith("[TEMPLATE ONLY]"):
        errors.append(f"{prefix}: manifest task must be prefixed with [TEMPLATE ONLY]")
    if not _is_empty(manifest_stub.get("pod_id")):
        errors.append(f"{prefix}: manifest pod_id must be empty before launch")
    if not _is_empty(manifest_stub.get("started_at")):
        errors.append(f"{prefix}: manifest started_at must be empty before launch")
    if not _is_empty(manifest_stub.get("finished_at")):
        errors.append(f"{prefix}: manifest finished_at must be empty before launch")
    if not _is_empty(manifest_stub.get("cost_snapshot")):
        errors.append(f"{prefix}: manifest cost_snapshot must be empty before launch")
    if manifest_stub.get("justified_itself") not in (None, "unclear"):
        errors.append(
            f"{prefix}: template evidence cannot justify itself before execution"
        )
    if manifest_stub.get("mode") != template.mode:
        errors.append(f"{prefix}: manifest mode does not match template mode")
    if manifest_stub.get("run_class") != template.run_class:
        errors.append(f"{prefix}: manifest run_class does not match template run_class")
    if manifest_stub.get("epistemic_status") != template.epistemic_status:
        errors.append(
            f"{prefix}: manifest epistemic_status does not match template epistemic_status"
        )
    commands = [str(item) for item in list(manifest_stub.get("commands", []) or [])]
    if not commands:
        errors.append(f"{prefix}: manifest commands are missing")
    needs_guard = template.mode == "runpod" or template.run_class in {
        "provider",
        "train",
    }
    if needs_guard and not any("TEMPLATE_ONLY" in command for command in commands):
        errors.append(f"{prefix}: external/provider/GPU template lacks guard command")
    if template.mode == "local" and any(
        "TEMPLATE_ONLY" in command for command in commands
    ):
        warnings.append(
            f"{prefix}: local verification template contains a template guard"
        )
    if not manifest_stub.get("artifact_paths"):
        errors.append(f"{prefix}: manifest artifact_paths are missing")
    if not manifest_stub.get("dependency_chain"):
        warnings.append(f"{prefix}: manifest dependency_chain is empty")
    return errors, warnings


def validate_economic_wm_provider_runbook_payload(
    payload: Mapping[str, Any],
    *,
    manifest_template_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMProviderRunbookValidationReport:
    """Validate a serialized provider runbook and any saved manifest templates."""

    runbook = EconomicWMProviderRunbook.from_dict(payload)
    embedded_templates = _template_lookup(list(payload.get("templates", []) or []))
    manifest_payloads = dict(manifest_template_payloads or {})
    errors: list[str] = []
    warnings: list[str] = []

    if runbook.authority_class != "runbook_template_only":
        errors.append("runbook authority_class must be runbook_template_only")
    if runbook.launch_allowed:
        errors.append("runbook launch_allowed must remain false")
    if runbook.provider_bringup_ready:
        errors.append(
            "runbook provider_bringup_ready must remain false before receipts"
        )
    if runbook.gpu_training_ready:
        errors.append("runbook gpu_training_ready must remain false before receipts")
    if runbook.promotion_eligible:
        errors.append("runbook promotion_eligible must remain false before benchmarks")
    if runbook.reward_math_mutation:
        errors.append("runbook reward_math_mutation must remain false")

    keys = {template.requirement_key for template in runbook.templates}
    missing_keys = sorted(_REQUIRED_TEMPLATE_KEYS - keys)
    if missing_keys:
        errors.append(f"runbook missing required template keys: {missing_keys}")

    checked_template_ids: list[str] = []
    for template in runbook.templates:
        checked_template_ids.append(template.template_id)
        prefix = f"template {template.template_id} ({template.requirement_key})"
        if template.launch_allowed:
            errors.append(f"{prefix}: launch_allowed must remain false")
        if template.promotion_eligible:
            errors.append(f"{prefix}: promotion_eligible must remain false")
        if template.mode == "runpod" and template.pod_class not in {
            "loop",
            "provider",
            "train",
            "refactor",
        }:
            errors.append(f"{prefix}: runpod template has invalid pod_class")
        if template.run_class not in {"loop", "provider", "train", "refactor"}:
            errors.append(f"{prefix}: invalid run_class")
        if template.epistemic_status not in {
            "smoke",
            "proof_of_life",
            "benchmark_candidate",
            "promotion_candidate",
            "deployment_candidate",
        }:
            errors.append(f"{prefix}: invalid epistemic_status")
        if not template.required_artifacts:
            warnings.append(f"{prefix}: no required_artifacts named")

        embedded_manifest = dict(
            embedded_templates.get(template.template_id, {}).get("manifest_stub", {})
            or {}
        )
        manifest_stub = manifest_payloads.get(template.template_id) or embedded_manifest
        if not manifest_stub:
            manifest_stub = template.to_manifest_stub()
            warnings.append(f"{prefix}: validating generated manifest stub only")
        stub_errors, stub_warnings = _validate_manifest_stub(
            template=template,
            manifest_stub=manifest_stub,
        )
        errors.extend(stub_errors)
        warnings.extend(stub_warnings)

    if not runbook.templates:
        errors.append("runbook has no templates")

    aggregate_counts = {
        "template_count": float(len(runbook.templates)),
        "error_count": float(len(errors)),
        "warning_count": float(len(warnings)),
        "required_template_key_count": float(len(_REQUIRED_TEMPLATE_KEYS)),
        "missing_required_template_key_count": float(len(missing_keys)),
        "runpod_template_count": float(
            sum(1 for template in runbook.templates if template.mode == "runpod")
        ),
        "local_template_count": float(
            sum(1 for template in runbook.templates if template.mode == "local")
        ),
    }
    report_payload = {
        "runbook_id": runbook.runbook_id,
        "status": "ok" if not errors else "failed",
        "safe_for_template_storage": not errors,
        "safe_for_launch": False,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checked_template_ids": checked_template_ids,
        "aggregate_counts": aggregate_counts,
        "artifact_refs": {
            **_mapping(runbook.artifact_refs),
            **_mapping(artifact_refs),
        },
        "metadata": {
            "boundary": "validation only; no provider/GPU/training/promotion claim",
            "validated_template_only_posture": True,
            **_mapping(metadata),
        },
        "version": ECONOMIC_WM_PROVIDER_RUNBOOK_VALIDATION_VERSION,
    }
    validation_id_payload = {
        "runbook_id": runbook.runbook_id,
        "status": report_payload["status"],
        "errors": errors,
        "warnings": warnings,
        "checked_template_ids": checked_template_ids,
        "version": ECONOMIC_WM_PROVIDER_RUNBOOK_VALIDATION_VERSION,
    }
    return EconomicWMProviderRunbookValidationReport(
        validation_id=f"ewm_provider_runbook_validation_{sha256_json(validation_id_payload)[:16]}",
        **report_payload,
    )


def validate_economic_wm_provider_runbook(
    runbook: EconomicWMProviderRunbook,
    *,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMProviderRunbookValidationReport:
    return validate_economic_wm_provider_runbook_payload(
        runbook.to_dict(), artifact_refs=artifact_refs, metadata=metadata
    )


def save_economic_wm_provider_runbook_validation_report(
    path: str | Path, report: EconomicWMProviderRunbookValidationReport
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_provider_runbook_validation_report(
    path: str | Path,
) -> EconomicWMProviderRunbookValidationReport:
    return EconomicWMProviderRunbookValidationReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def validate_economic_wm_provider_runbook_from_path(
    *,
    runbook_path: str | Path,
    output_path: str | Path,
    manifest_template_dir: Optional[str | Path] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMProviderRunbookValidationReport:
    payload = json.loads(Path(runbook_path).read_text(encoding="utf-8"))
    runbook = load_economic_wm_provider_runbook(runbook_path)
    manifest_payloads: Dict[str, Mapping[str, Any]] = {}
    manifest_dir = Path(manifest_template_dir) if manifest_template_dir else None
    if manifest_dir and manifest_dir.exists():
        for template in runbook.templates:
            manifest_path = (
                manifest_dir / f"{template.template_id}.manifest_template.json"
            )
            if manifest_path.exists():
                manifest_payloads[template.template_id] = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
    report = validate_economic_wm_provider_runbook_payload(
        payload,
        manifest_template_payloads=manifest_payloads,
        artifact_refs={
            "runbook_path": str(runbook_path),
            "validation_path": str(output_path),
            **({"manifest_template_dir": str(manifest_dir)} if manifest_dir else {}),
        },
        metadata=metadata,
    )
    save_economic_wm_provider_runbook_validation_report(output_path, report)
    return report


__all__ = [
    "ECONOMIC_WM_PROVIDER_RUNBOOK_VALIDATION_VERSION",
    "EconomicWMProviderRunbookValidationReport",
    "load_economic_wm_provider_runbook_validation_report",
    "save_economic_wm_provider_runbook_validation_report",
    "validate_economic_wm_provider_runbook",
    "validate_economic_wm_provider_runbook_from_path",
    "validate_economic_wm_provider_runbook_payload",
]
