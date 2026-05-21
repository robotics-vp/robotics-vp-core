"""Shadow-only allocation evaluation for the Economic World Model scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.scaffold import (
    EconomicWMScaffoldReport,
    load_economic_wm_scaffold_report,
)
from src.world_model.economic_world_model.training_rows import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    load_economic_wm_replay_feature_rows,
    load_economic_wm_training_corpus_manifest,
)

ECONOMIC_WM_ALLOCATION_CANDIDATE_VERSION = "economic_wm_allocation_candidate_v1"
ECONOMIC_WM_SHADOW_ALLOCATION_EVAL_VERSION = "economic_wm_shadow_allocation_eval_v1"


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


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


@dataclass(frozen=True)
class EconomicWMAllocationCandidate:
    """One advisory allocation candidate in a shadow eval."""

    candidate_id: str
    label: str
    allowed: bool
    expected_value: float
    resource_request: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    denial_reasons: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_ALLOCATION_CANDIDATE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "label": self.label,
            "allowed": bool(self.allowed),
            "expected_value": float(self.expected_value),
            "resource_request": _float_dict(self.resource_request),
            "rationale": self.rationale,
            "denial_reasons": list(self.denial_reasons),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMAllocationCandidate":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            label=str(payload.get("label", "")),
            allowed=bool(payload.get("allowed", False)),
            expected_value=float(payload.get("expected_value", 0.0)),
            resource_request=_float_dict(payload.get("resource_request", {})),
            rationale=str(payload.get("rationale", "")),
            denial_reasons=[
                str(item) for item in list(payload.get("denial_reasons", []) or [])
            ],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_ALLOCATION_CANDIDATE_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMShadowAllocationEval:
    """Shadow-only allocation recommendation over local Economic WM rows."""

    eval_id: str
    scaffold_id: str
    corpus_id: str
    allocation_envelope_id: str
    recommended_candidate: str
    candidates: list[EconomicWMAllocationCandidate] = field(default_factory=list)
    row_count: int = 0
    benchmark_ready_count: int = 0
    shadow_only_count: int = 0
    authority_class: str = "shadow_eval_only"
    reward_math_mutation: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    training_blockers: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SHADOW_ALLOCATION_EVAL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "version": self.version,
            "scaffold_id": self.scaffold_id,
            "corpus_id": self.corpus_id,
            "allocation_envelope_id": self.allocation_envelope_id,
            "recommended_candidate": self.recommended_candidate,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "row_count": int(self.row_count),
            "benchmark_ready_count": int(self.benchmark_ready_count),
            "shadow_only_count": int(self.shadow_only_count),
            "authority_class": self.authority_class,
            "reward_math_mutation": bool(self.reward_math_mutation),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "training_blockers": list(self.training_blockers),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMShadowAllocationEval":
        return cls(
            eval_id=str(payload.get("eval_id", "")),
            scaffold_id=str(payload.get("scaffold_id", "")),
            corpus_id=str(payload.get("corpus_id", "")),
            allocation_envelope_id=str(payload.get("allocation_envelope_id", "")),
            recommended_candidate=str(payload.get("recommended_candidate", "")),
            candidates=[
                EconomicWMAllocationCandidate.from_dict(item)
                for item in list(payload.get("candidates", []) or [])
            ],
            row_count=int(payload.get("row_count", 0) or 0),
            benchmark_ready_count=int(payload.get("benchmark_ready_count", 0) or 0),
            shadow_only_count=int(payload.get("shadow_only_count", 0) or 0),
            authority_class=str(payload.get("authority_class", "shadow_eval_only")),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_SHADOW_ALLOCATION_EVAL_VERSION)
            ),
        )


def _candidate(
    *,
    label: str,
    allowed: bool,
    expected_value: float,
    resource_request: Mapping[str, Any],
    rationale: str,
    denial_reasons: Iterable[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMAllocationCandidate:
    payload = {
        "label": label,
        "allowed": bool(allowed),
        "expected_value": float(expected_value),
        "resource_request": _float_dict(resource_request),
        "rationale": rationale,
        "denial_reasons": sorted(set(str(item) for item in denial_reasons)),
        "metadata": _mapping(metadata),
        "version": ECONOMIC_WM_ALLOCATION_CANDIDATE_VERSION,
    }
    return EconomicWMAllocationCandidate(
        candidate_id=f"alloc_candidate_{sha256_json(payload)[:16]}",
        label=label,
        allowed=allowed,
        expected_value=float(expected_value),
        resource_request=_float_dict(resource_request),
        rationale=rationale,
        denial_reasons=payload["denial_reasons"],
        metadata=_mapping(metadata),
    )


def build_economic_wm_shadow_allocation_eval(
    *,
    scaffold_report: EconomicWMScaffoldReport,
    corpus_manifest: EconomicWMTrainingCorpusManifest,
    rows: Iterable[EconomicWMReplayFeatureRow],
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMShadowAllocationEval:
    row_items = list(rows)
    row_count = max(1, len(row_items))
    benchmark_fraction = corpus_manifest.benchmark_ready_count / row_count
    shadow_fraction = corpus_manifest.shadow_only_count / row_count
    teacher_gap = _mean(
        row.target_vector.get("teacher_runtime_gap_weight", 0.0) for row in row_items
    )
    provider_gap = _mean(
        row.target_vector.get("provider_bringup_gap_weight", 0.0) for row in row_items
    )
    reconstruction_weight = _mean(
        row.target_vector.get("reconstruction_training_weight", 0.0)
        for row in row_items
    )
    local_budget = scaffold_report.allocation_envelope.budget_envelopes.get(
        "local_scaffold_budget", 0.0
    )
    denied_actions = set(scaffold_report.allocation_envelope.denied_actions)
    training_blockers = list(scaffold_report.training_blockers)

    candidates = [
        _candidate(
            label="curate_benchmark_ready_replay",
            allowed=corpus_manifest.benchmark_ready_count > 0,
            expected_value=benchmark_fraction
            * max(0.1, reconstruction_weight)
            * max(0.1, local_budget),
            resource_request={"local_dev_budget": 0.25, "gpu_budget": 0.0},
            rationale="Use benchmark-ready rows for evaluator and trainer-contract fixtures.",
        ),
        _candidate(
            label="close_shadow_gap_replay",
            allowed=corpus_manifest.shadow_only_count > 0,
            expected_value=shadow_fraction * max(0.1, local_budget),
            resource_request={"local_dev_budget": 0.35, "gpu_budget": 0.0},
            rationale="Prioritize shadow-only rows because they encode missing calibration, passthrough, or unqualified grounding gaps.",
        ),
        _candidate(
            label="prepare_teacher_provider_evidence_contracts",
            allowed=True,
            expected_value=((teacher_gap + provider_gap) / 2.0)
            * max(0.1, local_budget),
            resource_request={"local_dev_budget": 0.3, "gpu_budget": 0.0},
            rationale="Prepare evidence contracts for later non-stub teacher/provider bring-up without claiming the bring-up ran.",
            denial_reasons=[] if provider_gap < 1.0 else ["provider_bringup_not_run"],
        ),
        _candidate(
            label="run_gpu_training",
            allowed="gpu_training" not in denied_actions and not training_blockers,
            expected_value=0.0,
            resource_request={"local_dev_budget": 0.0, "gpu_budget": 1.0},
            rationale="Training remains denied until GPU/provider/promotion evidence exists.",
            denial_reasons=training_blockers or ["gpu_training_denied_by_envelope"],
        ),
    ]
    allowed_candidates = [candidate for candidate in candidates if candidate.allowed]
    recommended = (
        max(allowed_candidates, key=lambda item: item.expected_value).label
        if allowed_candidates
        else "none"
    )
    payload = {
        "scaffold_id": scaffold_report.scaffold_id,
        "corpus_id": corpus_manifest.corpus_id,
        "allocation_envelope_id": scaffold_report.allocation_envelope.envelope_id,
        "recommended_candidate": recommended,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "row_count": len(row_items),
        "version": ECONOMIC_WM_SHADOW_ALLOCATION_EVAL_VERSION,
    }
    return EconomicWMShadowAllocationEval(
        eval_id=f"ewm_shadow_alloc_{sha256_json(payload)[:16]}",
        scaffold_id=scaffold_report.scaffold_id,
        corpus_id=corpus_manifest.corpus_id,
        allocation_envelope_id=scaffold_report.allocation_envelope.envelope_id,
        recommended_candidate=recommended,
        candidates=candidates,
        row_count=len(row_items),
        benchmark_ready_count=corpus_manifest.benchmark_ready_count,
        shadow_only_count=corpus_manifest.shadow_only_count,
        authority_class="shadow_eval_only",
        reward_math_mutation=False,
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=training_blockers,
        artifact_refs={
            "economic_wm_scaffold_report_id": scaffold_report.scaffold_id,
            "economic_wm_training_corpus_id": corpus_manifest.corpus_id,
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "shadow allocation eval only; no control authority or reward mutation",
            "training_claim": False,
            **_mapping(metadata),
        },
    )


def save_economic_wm_shadow_allocation_eval(
    path: str | Path, eval_report: EconomicWMShadowAllocationEval
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(eval_report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_shadow_allocation_eval(
    path: str | Path,
) -> EconomicWMShadowAllocationEval:
    return EconomicWMShadowAllocationEval.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def evaluate_economic_wm_shadow_allocations_from_paths(
    *,
    scaffold_report_path: str | Path,
    corpus_manifest_path: str | Path,
    rows_path: str | Path,
    output_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMShadowAllocationEval:
    scaffold_report = load_economic_wm_scaffold_report(scaffold_report_path)
    corpus_manifest = load_economic_wm_training_corpus_manifest(corpus_manifest_path)
    rows = load_economic_wm_replay_feature_rows(rows_path)
    eval_report = build_economic_wm_shadow_allocation_eval(
        scaffold_report=scaffold_report,
        corpus_manifest=corpus_manifest,
        rows=rows,
        artifact_refs={
            "scaffold_report_path": str(scaffold_report_path),
            "corpus_manifest_path": str(corpus_manifest_path),
            "rows_path": str(rows_path),
        },
        metadata=metadata,
    )
    save_economic_wm_shadow_allocation_eval(output_path, eval_report)
    return eval_report


__all__ = [
    "ECONOMIC_WM_ALLOCATION_CANDIDATE_VERSION",
    "ECONOMIC_WM_SHADOW_ALLOCATION_EVAL_VERSION",
    "EconomicWMAllocationCandidate",
    "EconomicWMShadowAllocationEval",
    "build_economic_wm_shadow_allocation_eval",
    "evaluate_economic_wm_shadow_allocations_from_paths",
    "load_economic_wm_shadow_allocation_eval",
    "save_economic_wm_shadow_allocation_eval",
]
