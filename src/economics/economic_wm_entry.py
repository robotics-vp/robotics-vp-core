"""Entry preflight for starting the Economic World Model scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

ENTRY_PREFLIGHT_VERSION = "economic_wm_entry_preflight_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    return bool(payload.get(key, False))


@dataclass(frozen=True)
class EconomicWMEntryPreflightReport:
    """Readiness split for Economic WM scaffold start vs training/promotion."""

    readiness_class: str
    ready_for_scaffold: bool
    ready_for_training: bool
    scaffold_blockers: list[str] = field(default_factory=list)
    training_blockers: list[str] = field(default_factory=list)
    required_surfaces: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ENTRY_PREFLIGHT_VERSION

    @property
    def report_id(self) -> str:
        payload = self.to_dict(include_report_id=False)
        return f"ewm_entry_{sha256_json(payload)[:16]}"

    def to_dict(self, *, include_report_id: bool = True) -> Dict[str, Any]:
        payload = {
            "version": self.version,
            "readiness_class": self.readiness_class,
            "ready_for_scaffold": bool(self.ready_for_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "scaffold_blockers": list(self.scaffold_blockers),
            "training_blockers": list(self.training_blockers),
            "required_surfaces": _mapping(self.required_surfaces),
            "counts": {str(key): int(value) for key, value in self.counts.items()},
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMEntryPreflightReport":
        return cls(
            readiness_class=str(payload.get("readiness_class", "blocked")),
            ready_for_scaffold=bool(payload.get("ready_for_scaffold", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            scaffold_blockers=[
                str(item) for item in list(payload.get("scaffold_blockers", []) or [])
            ],
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            required_surfaces=_mapping(payload.get("required_surfaces")),
            counts={
                str(key): int(value)
                for key, value in dict(payload.get("counts", {}) or {}).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ENTRY_PREFLIGHT_VERSION)),
        )


def evaluate_economic_wm_entry_preflight(
    *,
    stage1_sweep_report: Mapping[str, Any],
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMEntryPreflightReport:
    """Evaluate whether Economic WM scaffold work may start honestly.

    This intentionally separates local scaffold entry from training/promotion.
    A green local sweep can make the Economic WM scaffold admissible while GPU,
    provider, and promotion-grade training remain blocked.
    """

    sweep = _mapping(stage1_sweep_report)
    scenario_reports = list(sweep.get("scenario_reports", []) or [])
    scaffold_blockers: list[str] = []

    if sweep.get("status") != "ok":
        scaffold_blockers.append("stage1_bridge_sweep_failed")
    if list(sweep.get("failures", []) or []):
        scaffold_blockers.append("stage1_bridge_sweep_has_failures")
    if int(sweep.get("scenario_count", 0) or 0) < 5:
        scaffold_blockers.append("stage1_manifest_diversity_below_minimum")
    if int(sweep.get("admission_count", 0) or 0) != int(
        sweep.get("scenario_count", 0) or 0
    ):
        scaffold_blockers.append("stage1_admission_count_mismatch")
    if int(sweep.get("rlds_episode_count", 0) or 0) != int(
        sweep.get("admission_count", 0) or 0
    ):
        scaffold_blockers.append("rlds_export_count_mismatch")
    if int(sweep.get("lerobot_row_count", 0) or 0) != int(
        sweep.get("admission_count", 0) or 0
    ):
        scaffold_blockers.append("lerobot_export_count_mismatch")
    if int(sweep.get("benchmark_ready_count", 0) or 0) < 1:
        scaffold_blockers.append("no_benchmark_ready_stage1_examples")
    if int(sweep.get("shadow_only_count", 0) or 0) < 1:
        scaffold_blockers.append("no_shadow_only_stage1_examples")
    if any(not _bool(dict(row or {}), "passed") for row in scenario_reports):
        scaffold_blockers.append("stage1_scenario_report_failed")

    required_surfaces = {
        "stage1_governed_video_supervision_refs": not scaffold_blockers,
        "canonical_replay_import": int(sweep.get("admission_count", 0) or 0) > 0,
        "rlds_bridge_truth_preserved": all(
            dict(row.get("observed", {}) or {}).get("rlds_benchmark_ready")
            == dict(row.get("observed", {}) or {}).get("benchmark_ready")
            for row in scenario_reports
            if isinstance(row, Mapping)
        ),
        "lerobot_bridge_truth_preserved": all(
            dict(row.get("observed", {}) or {}).get("lerobot_benchmark_ready")
            == dict(row.get("observed", {}) or {}).get("benchmark_ready")
            for row in scenario_reports
            if isinstance(row, Mapping)
        ),
        "benchmark_and_shadow_mix_present": int(
            sweep.get("benchmark_ready_count", 0) or 0
        )
        > 0
        and int(sweep.get("shadow_only_count", 0) or 0) > 0,
        "promotion_claim_absent": not bool(sweep.get("promotion_eligible", False)),
    }
    if not all(bool(value) for value in required_surfaces.values()):
        for key, value in required_surfaces.items():
            if not bool(value):
                scaffold_blockers.append(f"required_surface_missing::{key}")

    ready_for_scaffold = not scaffold_blockers

    training_blockers = []
    if ready_for_scaffold:
        training_blockers.extend(
            [
                "gpu_training_not_run",
                "provider_bringup_not_run",
                "non_stub_teacher_runtime_not_verified",
                "promotion_grade_benchmark_evidence_missing",
            ]
        )
    else:
        training_blockers.append("scaffold_preflight_blocked")

    ready_for_training = ready_for_scaffold and not training_blockers
    if ready_for_training:
        readiness_class = "training_ready"
    elif ready_for_scaffold:
        readiness_class = "scaffold_ready_training_blocked"
    else:
        readiness_class = "blocked"

    counts = {
        "scenario_count": int(sweep.get("scenario_count", 0) or 0),
        "admission_count": int(sweep.get("admission_count", 0) or 0),
        "rlds_episode_count": int(sweep.get("rlds_episode_count", 0) or 0),
        "lerobot_row_count": int(sweep.get("lerobot_row_count", 0) or 0),
        "benchmark_ready_count": int(sweep.get("benchmark_ready_count", 0) or 0),
        "shadow_only_count": int(sweep.get("shadow_only_count", 0) or 0),
    }
    return EconomicWMEntryPreflightReport(
        readiness_class=readiness_class,
        ready_for_scaffold=ready_for_scaffold,
        ready_for_training=ready_for_training,
        scaffold_blockers=sorted(set(scaffold_blockers)),
        training_blockers=sorted(set(training_blockers)),
        required_surfaces=required_surfaces,
        counts=counts,
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "boundary": "Economic WM scaffold entry only; no GPU/provider/training promotion claim",
            **_mapping(metadata),
        },
    )


def save_economic_wm_entry_preflight_report(
    path: str | Path,
    report: EconomicWMEntryPreflightReport,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_entry_preflight_report(
    path: str | Path,
) -> EconomicWMEntryPreflightReport:
    return EconomicWMEntryPreflightReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "ENTRY_PREFLIGHT_VERSION",
    "EconomicWMEntryPreflightReport",
    "evaluate_economic_wm_entry_preflight",
    "load_economic_wm_entry_preflight_report",
    "save_economic_wm_entry_preflight_report",
]
