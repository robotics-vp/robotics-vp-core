"""Lower-WM maturity sweep for Economic WM Phase-5.1.

The Economic WM now consumes canonical lower-WM refs. This sweep checks whether
those refs and their adjacent sidecars are merely structurally present or mature
enough to support later transport/training work. It does not promote any lower
WM, provider, or Economic WM output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.lower_wm_consumption import (
    EconomicWMCanonicalConsumptionRow,
    EconomicWMLowerWMConsumptionPreflight,
    load_economic_wm_canonical_consumption_rows,
    load_economic_wm_lower_wm_consumption_preflight,
)
from src.world_model.economic_world_model.phase5_local_prep import (
    EconomicWMDatapackCompositionRow,
    EconomicWMPhase5LocalPrepManifest,
    load_economic_wm_datapack_composition_rows,
    load_economic_wm_phase5_local_prep_manifest,
)
from src.world_model.economic_world_model.resource_surfaces import (
    EconomicWMResourceIngestionManifest,
    load_economic_wm_resource_ingestion_manifest,
)

ECONOMIC_WM_LOWER_WM_MATURITY_ROW_VERSION = "economic_wm_lower_wm_maturity_row_v1"
ECONOMIC_WM_LOWER_WM_MATURITY_SWEEP_VERSION = "economic_wm_lower_wm_maturity_sweep_v1"

MATURITY_BLOCKERS = (
    "production_scene_tracks_or_calibration_not_complete_for_all_rows",
    "non_stub_teacher_runtime_not_verified",
    "hardware_or_provider_runtime_evidence_missing",
    "promotion_grade_lower_wm_benchmark_missing",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _safe_load(path: str) -> Dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        return _load_json(candidate)
    except Exception:
        return None


@dataclass(frozen=True)
class EconomicWMLowerWMMaturityRow:
    """One lower-WM reference maturity assessment."""

    maturity_row_id: str
    source_row_id: str
    source_episode_id: str
    wm_key: str
    artifact_path: str
    observed_version: str = ""
    state_id: str = ""
    reference_status: str = "missing"
    direct_reference: bool = False
    artifact_exists: bool = False
    sidecar_scores: Dict[str, float] = field(default_factory=dict)
    maturity_score: float = 0.0
    maturity_class: str = "blocked"
    authority_class: str = "lower_wm_maturity_sweep_only"
    ready_for_phase6_contracts: bool = False
    ready_for_production: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_LOWER_WM_MATURITY_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maturity_row_id": self.maturity_row_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "wm_key": self.wm_key,
            "artifact_path": self.artifact_path,
            "observed_version": self.observed_version,
            "state_id": self.state_id,
            "reference_status": self.reference_status,
            "direct_reference": bool(self.direct_reference),
            "artifact_exists": bool(self.artifact_exists),
            "sidecar_scores": _float_dict(self.sidecar_scores),
            "maturity_score": float(self.maturity_score),
            "maturity_class": self.maturity_class,
            "authority_class": self.authority_class,
            "ready_for_phase6_contracts": bool(self.ready_for_phase6_contracts),
            "ready_for_production": bool(self.ready_for_production),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMLowerWMMaturityRow":
        return cls(
            maturity_row_id=str(payload.get("maturity_row_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            wm_key=str(payload.get("wm_key", "")),
            artifact_path=str(payload.get("artifact_path", "")),
            observed_version=str(payload.get("observed_version", "")),
            state_id=str(payload.get("state_id", "")),
            reference_status=str(payload.get("reference_status", "missing")),
            direct_reference=bool(payload.get("direct_reference", False)),
            artifact_exists=bool(payload.get("artifact_exists", False)),
            sidecar_scores=_float_dict(payload.get("sidecar_scores", {})),
            maturity_score=float(payload.get("maturity_score", 0.0)),
            maturity_class=str(payload.get("maturity_class", "blocked")),
            authority_class=str(
                payload.get("authority_class", "lower_wm_maturity_sweep_only")
            ),
            ready_for_phase6_contracts=bool(
                payload.get("ready_for_phase6_contracts", False)
            ),
            ready_for_production=bool(payload.get("ready_for_production", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_LOWER_WM_MATURITY_ROW_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMLowerWMMaturitySweep:
    """Manifest for lower-WM maturity over Economic WM canonical refs."""

    sweep_id: str
    phase5_manifest_id: str
    lower_wm_preflight_id: str
    resource_manifest_id: str
    maturity_row_count: int
    structural_ready_count: int
    production_ready_count: int
    maturity_rows_path: str
    status: str
    authority_class: str = "lower_wm_maturity_sweep_only"
    ready_for_phase6_contracts: bool = False
    ready_for_production: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_LOWER_WM_MATURITY_SWEEP_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "version": self.version,
            "phase5_manifest_id": self.phase5_manifest_id,
            "lower_wm_preflight_id": self.lower_wm_preflight_id,
            "resource_manifest_id": self.resource_manifest_id,
            "maturity_row_count": int(self.maturity_row_count),
            "structural_ready_count": int(self.structural_ready_count),
            "production_ready_count": int(self.production_ready_count),
            "maturity_rows_path": self.maturity_rows_path,
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_phase6_contracts": bool(self.ready_for_phase6_contracts),
            "ready_for_production": bool(self.ready_for_production),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMLowerWMMaturitySweep":
        return cls(
            sweep_id=str(payload.get("sweep_id", "")),
            phase5_manifest_id=str(payload.get("phase5_manifest_id", "")),
            lower_wm_preflight_id=str(payload.get("lower_wm_preflight_id", "")),
            resource_manifest_id=str(payload.get("resource_manifest_id", "")),
            maturity_row_count=int(payload.get("maturity_row_count", 0) or 0),
            structural_ready_count=int(payload.get("structural_ready_count", 0) or 0),
            production_ready_count=int(payload.get("production_ready_count", 0) or 0),
            maturity_rows_path=str(payload.get("maturity_rows_path", "")),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "lower_wm_maturity_sweep_only")
            ),
            ready_for_phase6_contracts=bool(
                payload.get("ready_for_phase6_contracts", False)
            ),
            ready_for_production=bool(payload.get("ready_for_production", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_LOWER_WM_MATURITY_SWEEP_VERSION)
            ),
        )


def _sidecar_scores(
    composition: Optional[EconomicWMDatapackCompositionRow],
) -> Dict[str, float]:
    if composition is None:
        return {}
    refs = composition.source_refs
    recon = _safe_load(str(refs.get("reconstruction_grounding_report_path", "")))
    benchmark = _safe_load(str(refs.get("benchmark_gate_path", "")))
    teacher = _safe_load(str(refs.get("teacher_trace_path", "")))
    scores = {
        "resource_receipt_present": 1.0 if composition.resource_receipt_ref else 0.0,
        "counterfactual_value_join_present": 1.0
        if composition.counterfactual_value_join_ref
        else 0.0,
    }
    if recon:
        quality = dict(recon.get("quality", {}) or {})
        metadata = dict(recon.get("metadata", {}) or {})
        scores.update(
            {
                "calibration_complete": float(quality.get("calibration_complete", 0.0)),
                "calibration_score": float(quality.get("calibration_score", 0.0)),
                "real_scene_tracks_joined": 1.0
                if metadata.get("scene_tracks_backend") == "real"
                else 0.0,
                "reconstruction_training_eligible": 1.0
                if recon.get("training_eligible")
                else 0.0,
                "reconstruction_benchmark_ready": 1.0
                if recon.get("benchmark_ready")
                else 0.0,
            }
        )
    else:
        scores.update(
            {
                "calibration_complete": 0.0,
                "calibration_score": 0.0,
                "real_scene_tracks_joined": 0.0,
                "reconstruction_training_eligible": 0.0,
                "reconstruction_benchmark_ready": 0.0,
            }
        )
    scores["benchmark_gate_ready"] = (
        1.0 if benchmark and benchmark.get("ready") else 0.0
    )
    teacher_summary = dict((teacher or {}).get("summary", {}) or {})
    scores["teacher_runtime_real"] = (
        1.0 if teacher_summary.get("teacher_confidence_mean", 0.0) else 0.0
    )
    return scores


def _maturity_class(
    score: float, scores: Mapping[str, float], blockers: list[str]
) -> str:
    if blockers and score < 0.5:
        return "blocked"
    if scores.get("calibration_complete", 0.0) < 1.0:
        return "calibration_gap"
    if scores.get("real_scene_tracks_joined", 0.0) < 1.0:
        return "scene_tracks_gap"
    if score >= 0.75:
        return "local_structural_mature"
    return "structural_partial"


def _row_from_ref(
    *,
    consumption: EconomicWMCanonicalConsumptionRow,
    composition: Optional[EconomicWMDatapackCompositionRow],
    ref_payload: Mapping[str, Any],
) -> EconomicWMLowerWMMaturityRow:
    artifact_path = str(ref_payload.get("artifact_path", ""))
    artifact_exists = bool(artifact_path and Path(artifact_path).exists())
    sidecar = _sidecar_scores(composition)
    structural_score = (
        1.0 if artifact_exists and ref_payload.get("satisfied", False) else 0.0
    )
    score = (
        0.45 * structural_score
        + 0.15 * sidecar.get("calibration_complete", 0.0)
        + 0.1 * sidecar.get("real_scene_tracks_joined", 0.0)
        + 0.1 * sidecar.get("benchmark_gate_ready", 0.0)
        + 0.1 * sidecar.get("resource_receipt_present", 0.0)
        + 0.1 * sidecar.get("counterfactual_value_join_present", 0.0)
    )
    blockers: list[str] = []
    if not artifact_exists:
        blockers.append("canonical_state_artifact_missing")
    if not ref_payload.get("satisfied", False):
        blockers.append("canonical_state_ref_not_satisfied")
    if sidecar.get("calibration_complete", 0.0) < 1.0:
        blockers.append("camera_calibration_incomplete")
    if sidecar.get("real_scene_tracks_joined", 0.0) < 1.0:
        blockers.append("real_scene_tracks_not_joined")
    if sidecar.get("teacher_runtime_real", 0.0) < 1.0:
        blockers.append("teacher_runtime_unavailable")
    maturity_class = _maturity_class(score, sidecar, blockers)
    ready_for_phase6 = artifact_exists and bool(ref_payload.get("satisfied", False))
    production_ready = (
        ready_for_phase6
        and maturity_class == "local_structural_mature"
        and not blockers
    )
    payload = {
        "source_row_id": consumption.source_row_id,
        "wm_key": ref_payload.get("wm_key", ""),
        "artifact_path": artifact_path,
        "score": score,
        "class": maturity_class,
    }
    return EconomicWMLowerWMMaturityRow(
        maturity_row_id=f"ewm_lower_wm_maturity_{sha256_json(payload)[:16]}",
        source_row_id=consumption.source_row_id,
        source_episode_id=consumption.source_episode_id,
        wm_key=str(ref_payload.get("wm_key", "")),
        artifact_path=artifact_path,
        observed_version=str(ref_payload.get("observed_version", "")),
        state_id=str(ref_payload.get("state_id", "")),
        reference_status=str(ref_payload.get("reference_status", "missing")),
        direct_reference=bool(ref_payload.get("direct_reference", False)),
        artifact_exists=artifact_exists,
        sidecar_scores=sidecar,
        maturity_score=score,
        maturity_class=maturity_class,
        ready_for_phase6_contracts=ready_for_phase6,
        ready_for_production=production_ready,
        blockers=sorted(set(blockers)),
        source_refs=_mapping(composition.source_refs if composition else {}),
        metadata={
            "boundary": "maturity sweep only; no promotion",
            "summary_only": bool(ref_payload.get("summary_only", True)),
        },
    )


def build_economic_wm_lower_wm_maturity_sweep(
    *,
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
    lower_wm_preflight: EconomicWMLowerWMConsumptionPreflight,
    consumption_rows: Iterable[EconomicWMCanonicalConsumptionRow],
    composition_rows: Iterable[EconomicWMDatapackCompositionRow],
    resource_manifest: EconomicWMResourceIngestionManifest,
    maturity_rows_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[EconomicWMLowerWMMaturitySweep, list[EconomicWMLowerWMMaturityRow]]:
    compositions_by_row = {item.source_row_id: item for item in composition_rows}
    maturity_rows: list[EconomicWMLowerWMMaturityRow] = []
    for consumption in consumption_rows:
        composition = compositions_by_row.get(consumption.source_row_id)
        for ref in consumption.canonical_refs:
            maturity_rows.append(
                _row_from_ref(
                    consumption=consumption,
                    composition=composition,
                    ref_payload=ref.to_dict(),
                )
            )
    structural_ready_count = sum(
        1 for row in maturity_rows if row.ready_for_phase6_contracts
    )
    production_ready_count = sum(1 for row in maturity_rows if row.ready_for_production)
    ready_for_phase6 = bool(maturity_rows) and structural_ready_count == len(
        maturity_rows
    )
    ready_for_production = bool(maturity_rows) and production_ready_count == len(
        maturity_rows
    )
    status = "ok" if ready_for_phase6 else "blocked"
    blockers = list(MATURITY_BLOCKERS)
    payload = {
        "phase5_manifest_id": phase5_manifest.manifest_id,
        "lower_wm_preflight_id": lower_wm_preflight.preflight_id,
        "resource_manifest_id": resource_manifest.manifest_id,
        "maturity_row_ids": [row.maturity_row_id for row in maturity_rows],
    }
    aggregate_counts = {
        "maturity_row_count": float(len(maturity_rows)),
        "structural_ready_count": float(structural_ready_count),
        "production_ready_count": float(production_ready_count),
        "direct_reference_count": float(
            sum(1 for row in maturity_rows if row.direct_reference)
        ),
        "canonical_artifact_exists_count": float(
            sum(1 for row in maturity_rows if row.artifact_exists)
        ),
        "calibration_complete_count": float(
            sum(
                1
                for row in maturity_rows
                if row.sidecar_scores.get("calibration_complete", 0.0) >= 1.0
            )
        ),
        "real_scene_tracks_joined_count": float(
            sum(
                1
                for row in maturity_rows
                if row.sidecar_scores.get("real_scene_tracks_joined", 0.0) >= 1.0
            )
        ),
        "benchmark_ready_count": float(
            sum(
                1
                for row in maturity_rows
                if row.sidecar_scores.get("benchmark_gate_ready", 0.0) >= 1.0
            )
        ),
        "teacher_runtime_real_count": float(
            sum(
                1
                for row in maturity_rows
                if row.sidecar_scores.get("teacher_runtime_real", 0.0) >= 1.0
            )
        ),
        "resource_receipt_count": float(resource_manifest.receipt_count),
    }
    return (
        EconomicWMLowerWMMaturitySweep(
            sweep_id=f"ewm_lower_wm_maturity_{sha256_json(payload)[:16]}",
            phase5_manifest_id=phase5_manifest.manifest_id,
            lower_wm_preflight_id=lower_wm_preflight.preflight_id,
            resource_manifest_id=resource_manifest.manifest_id,
            maturity_row_count=len(maturity_rows),
            structural_ready_count=structural_ready_count,
            production_ready_count=production_ready_count,
            maturity_rows_path=str(maturity_rows_path),
            status=status,
            ready_for_phase6_contracts=ready_for_phase6,
            ready_for_production=ready_for_production,
            blockers=blockers,
            aggregate_counts=aggregate_counts,
            artifact_refs={
                **_mapping(artifact_refs),
                "maturity_rows_path": str(maturity_rows_path),
            },
            metadata={
                **_mapping(metadata),
                "boundary": "lower-WM maturity sweep only",
            },
        ),
        maturity_rows,
    )


def save_economic_wm_lower_wm_maturity_sweep(
    *,
    sweep_path: str | Path,
    sweep: EconomicWMLowerWMMaturitySweep,
    maturity_rows: Iterable[EconomicWMLowerWMMaturityRow],
) -> None:
    _write_json(sweep_path, sweep.to_dict())
    _write_jsonl(sweep.maturity_rows_path, [row.to_dict() for row in maturity_rows])


def load_economic_wm_lower_wm_maturity_sweep(
    path: str | Path,
) -> EconomicWMLowerWMMaturitySweep:
    return EconomicWMLowerWMMaturitySweep.from_dict(_load_json(path))


def load_economic_wm_lower_wm_maturity_rows(
    path: str | Path,
) -> list[EconomicWMLowerWMMaturityRow]:
    return [EconomicWMLowerWMMaturityRow.from_dict(row) for row in _load_jsonl(path)]


def build_economic_wm_lower_wm_maturity_sweep_from_paths(
    *,
    phase5_prep_path: str | Path,
    lower_wm_preflight_path: str | Path,
    canonical_consumption_rows_path: str | Path,
    resource_manifest_path: str | Path,
    sweep_path: str | Path,
    maturity_rows_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMLowerWMMaturitySweep:
    phase5_manifest = load_economic_wm_phase5_local_prep_manifest(phase5_prep_path)
    lower_wm_preflight = load_economic_wm_lower_wm_consumption_preflight(
        lower_wm_preflight_path
    )
    consumption_rows = load_economic_wm_canonical_consumption_rows(
        canonical_consumption_rows_path
    )
    composition_rows = load_economic_wm_datapack_composition_rows(
        phase5_manifest.composition_rows_path
    )
    resource_manifest = load_economic_wm_resource_ingestion_manifest(
        resource_manifest_path
    )
    sweep, rows = build_economic_wm_lower_wm_maturity_sweep(
        phase5_manifest=phase5_manifest,
        lower_wm_preflight=lower_wm_preflight,
        consumption_rows=consumption_rows,
        composition_rows=composition_rows,
        resource_manifest=resource_manifest,
        maturity_rows_path=maturity_rows_path,
        artifact_refs={
            "phase5_prep_path": str(phase5_prep_path),
            "lower_wm_preflight_path": str(lower_wm_preflight_path),
            "canonical_consumption_rows_path": str(canonical_consumption_rows_path),
            "resource_manifest_path": str(resource_manifest_path),
            "sweep_path": str(sweep_path),
        },
        metadata=metadata,
    )
    save_economic_wm_lower_wm_maturity_sweep(
        sweep_path=sweep_path,
        sweep=sweep,
        maturity_rows=rows,
    )
    return sweep
