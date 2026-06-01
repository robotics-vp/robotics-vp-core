"""Teacher/provider evidence contracts for future Economic WM bring-up.

These contracts are local scaffold artifacts. They prepare the evidence shape
needed for provider/GPU seasons without claiming any provider was brought up or
any training was run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.allocation_eval import (
    EconomicWMShadowAllocationEval,
    load_economic_wm_shadow_allocation_eval,
)
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

ECONOMIC_WM_EVIDENCE_REQUIREMENT_VERSION = "economic_wm_evidence_requirement_v1"
ECONOMIC_WM_TEACHER_PROVIDER_CONTRACT_VERSION = (
    "economic_wm_teacher_provider_contract_v1"
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


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _fraction(rows: Iterable[EconomicWMReplayFeatureRow], feature_name: str) -> float:
    row_items = list(rows)
    if not row_items:
        return 0.0
    return sum(
        1.0 for row in row_items if row.feature_vector.get(feature_name, 0.0) > 0.0
    ) / len(row_items)


@dataclass(frozen=True)
class EconomicWMEvidenceRequirement:
    """One required evidence surface before a provider/training gate can open."""

    requirement_id: str
    requirement_key: str
    provider_family: str
    evidence_kind: str
    current_status: str
    required_artifacts: list[str] = field(default_factory=list)
    local_prep_actions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    satisfaction_score: float = 0.0
    promotion_gate: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_EVIDENCE_REQUIREMENT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "version": self.version,
            "requirement_key": self.requirement_key,
            "provider_family": self.provider_family,
            "evidence_kind": self.evidence_kind,
            "current_status": self.current_status,
            "required_artifacts": list(self.required_artifacts),
            "local_prep_actions": list(self.local_prep_actions),
            "blockers": list(self.blockers),
            "satisfaction_score": float(self.satisfaction_score),
            "promotion_gate": bool(self.promotion_gate),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMEvidenceRequirement":
        return cls(
            requirement_id=str(payload.get("requirement_id", "")),
            requirement_key=str(payload.get("requirement_key", "")),
            provider_family=str(payload.get("provider_family", "")),
            evidence_kind=str(payload.get("evidence_kind", "")),
            current_status=str(payload.get("current_status", "missing")),
            required_artifacts=[
                str(item) for item in list(payload.get("required_artifacts", []) or [])
            ],
            local_prep_actions=[
                str(item) for item in list(payload.get("local_prep_actions", []) or [])
            ],
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            satisfaction_score=float(payload.get("satisfaction_score", 0.0)),
            promotion_gate=bool(payload.get("promotion_gate", True)),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_EVIDENCE_REQUIREMENT_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMTeacherProviderContract:
    """Contract pack for future non-stub teacher/provider bring-up."""

    contract_id: str
    scaffold_id: str
    allocation_eval_id: str
    corpus_id: str
    readiness_class: str
    requirements: list[EconomicWMEvidenceRequirement] = field(default_factory=list)
    provider_bringup_ready: bool = False
    gpu_training_ready: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    authority_class: str = "evidence_contract_only"
    recommended_next_actions: list[str] = field(default_factory=list)
    training_blockers: list[str] = field(default_factory=list)
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_TEACHER_PROVIDER_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "scaffold_id": self.scaffold_id,
            "allocation_eval_id": self.allocation_eval_id,
            "corpus_id": self.corpus_id,
            "readiness_class": self.readiness_class,
            "requirements": [
                requirement.to_dict() for requirement in self.requirements
            ],
            "provider_bringup_ready": bool(self.provider_bringup_ready),
            "gpu_training_ready": bool(self.gpu_training_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "authority_class": self.authority_class,
            "recommended_next_actions": list(self.recommended_next_actions),
            "training_blockers": list(self.training_blockers),
            "aggregate_scores": _float_dict(self.aggregate_scores),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMTeacherProviderContract":
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            scaffold_id=str(payload.get("scaffold_id", "")),
            allocation_eval_id=str(payload.get("allocation_eval_id", "")),
            corpus_id=str(payload.get("corpus_id", "")),
            readiness_class=str(payload.get("readiness_class", "blocked")),
            requirements=[
                EconomicWMEvidenceRequirement.from_dict(item)
                for item in list(payload.get("requirements", []) or [])
            ],
            provider_bringup_ready=bool(payload.get("provider_bringup_ready", False)),
            gpu_training_ready=bool(payload.get("gpu_training_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            authority_class=str(
                payload.get("authority_class", "evidence_contract_only")
            ),
            recommended_next_actions=[
                str(item)
                for item in list(payload.get("recommended_next_actions", []) or [])
            ],
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            aggregate_scores=_float_dict(payload.get("aggregate_scores", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_TEACHER_PROVIDER_CONTRACT_VERSION)
            ),
        )


def _requirement(
    *,
    requirement_key: str,
    provider_family: str,
    evidence_kind: str,
    current_status: str,
    required_artifacts: Iterable[str],
    local_prep_actions: Iterable[str],
    blockers: Iterable[str],
    satisfaction_score: float,
    promotion_gate: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMEvidenceRequirement:
    blocker_list = sorted(set(str(item) for item in blockers))
    clamped_satisfaction_score = max(0.0, min(1.0, float(satisfaction_score)))
    payload = {
        "requirement_key": requirement_key,
        "provider_family": provider_family,
        "evidence_kind": evidence_kind,
        "current_status": current_status,
        "required_artifacts": list(required_artifacts),
        "local_prep_actions": list(local_prep_actions),
        "blockers": blocker_list,
        "satisfaction_score": clamped_satisfaction_score,
        "promotion_gate": bool(promotion_gate),
        "metadata": _mapping(metadata),
        "version": ECONOMIC_WM_EVIDENCE_REQUIREMENT_VERSION,
    }
    return EconomicWMEvidenceRequirement(
        requirement_id=f"ewm_req_{sha256_json(payload)[:16]}",
        requirement_key=requirement_key,
        provider_family=provider_family,
        evidence_kind=evidence_kind,
        current_status=current_status,
        required_artifacts=list(required_artifacts),
        local_prep_actions=list(local_prep_actions),
        blockers=blocker_list,
        satisfaction_score=clamped_satisfaction_score,
        promotion_gate=promotion_gate,
        metadata=_mapping(metadata),
    )


def build_economic_wm_teacher_provider_contract(
    *,
    scaffold_report: EconomicWMScaffoldReport,
    allocation_eval: EconomicWMShadowAllocationEval,
    corpus_manifest: EconomicWMTrainingCorpusManifest,
    rows: Iterable[EconomicWMReplayFeatureRow],
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMTeacherProviderContract:
    """Build the evidence contract pack recommended by the shadow allocation eval."""

    row_items = list(rows)
    row_count = len(row_items)
    teacher_contract_fraction = _fraction(
        row_items, "teacher_runtime_contract_complete"
    )
    teacher_real_fraction = _fraction(row_items, "teacher_runtime_real")
    provider_gap = _mean(
        row.target_vector.get("provider_bringup_gap_weight", 0.0) for row in row_items
    )
    benchmark_fraction = corpus_manifest.benchmark_ready_count / max(1, row_count)
    reconstruction_fraction = _fraction(row_items, "reconstruction_training_eligible")
    replay_export_flow = float(
        scaffold_report.economic_state.flow_fields.get("replay_export_flow", 0.0)
    )
    training_blockers = sorted(
        set(scaffold_report.training_blockers) | set(allocation_eval.training_blockers)
    )

    requirements = [
        _requirement(
            requirement_key="non_stub_teacher_runtime_invocation",
            provider_family="teacher_runtime",
            evidence_kind="teacher_invocation_receipt",
            current_status="blocked_external_runtime"
            if teacher_real_fraction == 0.0
            else "partially_satisfied",
            required_artifacts=[
                "teacher_adapter_contract_v1",
                "teacher_action_envelope_v1",
                "teacher_trace_v1",
                "external_provider_truth_v1",
            ],
            local_prep_actions=[
                "preserve_teacher_contract_refs_in_rows",
                "require_provider_truth_available_true_before_training",
                "record_unavailable_teacher_as_gap_not_success",
            ],
            blockers=["non_stub_teacher_runtime_not_verified"]
            if teacher_real_fraction < 1.0
            else [],
            satisfaction_score=teacher_real_fraction,
            metadata={
                "teacher_contract_fraction": teacher_contract_fraction,
                "teacher_real_fraction": teacher_real_fraction,
            },
        ),
        _requirement(
            requirement_key="provider_runtime_truth_receipts",
            provider_family="external_provider_runtime",
            evidence_kind="provider_truth_receipt",
            current_status="blocked_external_runtime"
            if provider_gap > 0.0
            else "satisfied",
            required_artifacts=[
                "external_provider_truth_v1",
                "runtime_provider_manifest_v1",
                "provider_invocation_receipt_v1",
            ],
            local_prep_actions=[
                "compile_provider_truth_requirements_from_rows",
                "deny_provider_truth_promotion_without_available_true_receipt",
                "preserve_planning_only_fallback_truth",
            ],
            blockers=["provider_bringup_not_run"] if provider_gap > 0.0 else [],
            satisfaction_score=1.0 - provider_gap,
            metadata={"provider_gap_weight_mean": provider_gap},
        ),
        _requirement(
            requirement_key="promotion_grade_benchmark_evidence",
            provider_family="benchmark_evidence",
            evidence_kind="promotion_metric_report",
            current_status="local_fixture_only",
            required_artifacts=[
                "benchmark_gate_v1",
                "promotion_metric_report_v1",
                "benchmark_evidence_bundle_v1",
            ],
            local_prep_actions=[
                "preserve_benchmark_ready_split",
                "keep_shadow_only_rows_as_negative_gap_evidence",
                "require_promotion_grade_metric_report_before_model_promotion",
            ],
            blockers=["promotion_grade_benchmark_evidence_missing"],
            satisfaction_score=benchmark_fraction,
            metadata={
                "benchmark_ready_count": corpus_manifest.benchmark_ready_count,
                "shadow_only_count": corpus_manifest.shadow_only_count,
            },
        ),
        _requirement(
            requirement_key="gpu_training_runtime_receipt",
            provider_family="gpu_training_runtime",
            evidence_kind="training_runtime_manifest",
            current_status="blocked_external_runtime",
            required_artifacts=[
                "training_runtime_manifest_v1",
                "gpu_runtime_receipt_v1",
                "checkpoint_manifest_v1",
            ],
            local_prep_actions=[
                "keep_gpu_training_denied_in_allocation_envelope",
                "prepare_manifest_fields_for_future_gpu_run",
                "require_run_manifest_before_training_claim",
            ],
            blockers=["gpu_training_not_run"],
            satisfaction_score=0.0,
        ),
        _requirement(
            requirement_key="replay_row_linkage_integrity",
            provider_family="local_replay_bridge",
            evidence_kind="row_sidecar_linkage",
            current_status="satisfied_local_scaffold"
            if replay_export_flow >= 1.0
            else "partial",
            required_artifacts=[
                "economic_wm_replay_feature_row_v1",
                "economic_wm_training_corpus_manifest_v1",
                "governed_video_stage1_bridge_export_v1",
            ],
            local_prep_actions=[
                "keep_replay_export_flow_at_one",
                "preserve_runtime_value_governance_teacher_refs",
                "fail_rows_when_sidecar_refs_disappear",
            ],
            blockers=[]
            if replay_export_flow >= 1.0
            else ["replay_export_flow_incomplete"],
            satisfaction_score=replay_export_flow,
            promotion_gate=False,
            metadata={
                "reconstruction_training_eligible_fraction": reconstruction_fraction
            },
        ),
    ]

    blocker_set = sorted(
        set(training_blockers)
        | {blocker for requirement in requirements for blocker in requirement.blockers}
    )
    provider_bringup_ready = False
    gpu_training_ready = False
    promotion_eligible = False
    aggregate_scores = {
        "teacher_contract_fraction": teacher_contract_fraction,
        "teacher_real_fraction": teacher_real_fraction,
        "provider_gap_weight_mean": provider_gap,
        "benchmark_ready_fraction": benchmark_fraction,
        "reconstruction_training_eligible_fraction": reconstruction_fraction,
        "replay_export_flow": replay_export_flow,
        "mean_requirement_satisfaction": _mean(
            requirement.satisfaction_score for requirement in requirements
        ),
    }
    recommended_next_actions = [
        "wire_contract_pack_into_future_provider_bringup_manifest",
        "add_provider_invocation_receipt_placeholders_to_gpu_run_templates",
        "keep_run_gpu_training_denied_until receipts are present".replace(" ", "_"),
    ]
    if teacher_real_fraction == 0.0:
        recommended_next_actions.insert(
            0, "prepare_non_stub_teacher_runtime_invocation_fixture"
        )
    if provider_gap > 0.0:
        recommended_next_actions.insert(
            1, "prepare_external_provider_truth_receipt_fixture"
        )

    payload = {
        "scaffold_id": scaffold_report.scaffold_id,
        "allocation_eval_id": allocation_eval.eval_id,
        "corpus_id": corpus_manifest.corpus_id,
        "requirements": [requirement.to_dict() for requirement in requirements],
        "aggregate_scores": aggregate_scores,
        "version": ECONOMIC_WM_TEACHER_PROVIDER_CONTRACT_VERSION,
    }
    return EconomicWMTeacherProviderContract(
        contract_id=f"ewm_teacher_provider_contract_{sha256_json(payload)[:16]}",
        scaffold_id=scaffold_report.scaffold_id,
        allocation_eval_id=allocation_eval.eval_id,
        corpus_id=corpus_manifest.corpus_id,
        readiness_class=scaffold_report.economic_state.regime,
        requirements=requirements,
        provider_bringup_ready=provider_bringup_ready,
        gpu_training_ready=gpu_training_ready,
        promotion_eligible=promotion_eligible,
        reward_math_mutation=False,
        authority_class="evidence_contract_only",
        recommended_next_actions=recommended_next_actions,
        training_blockers=blocker_set,
        aggregate_scores=aggregate_scores,
        artifact_refs={
            "economic_wm_scaffold_report_id": scaffold_report.scaffold_id,
            "economic_wm_shadow_allocation_eval_id": allocation_eval.eval_id,
            "economic_wm_training_corpus_id": corpus_manifest.corpus_id,
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "contract prep only; no provider bring-up, GPU training, or promotion claim",
            "recommended_candidate": allocation_eval.recommended_candidate,
            "training_claim": False,
            **_mapping(metadata),
        },
    )


def save_economic_wm_teacher_provider_contract(
    path: str | Path, contract: EconomicWMTeacherProviderContract
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_teacher_provider_contract(
    path: str | Path,
) -> EconomicWMTeacherProviderContract:
    return EconomicWMTeacherProviderContract.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def build_economic_wm_teacher_provider_contract_from_paths(
    *,
    scaffold_report_path: str | Path,
    allocation_eval_path: str | Path,
    corpus_manifest_path: str | Path,
    rows_path: str | Path,
    output_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMTeacherProviderContract:
    scaffold_report = load_economic_wm_scaffold_report(scaffold_report_path)
    allocation_eval = load_economic_wm_shadow_allocation_eval(allocation_eval_path)
    corpus_manifest = load_economic_wm_training_corpus_manifest(corpus_manifest_path)
    rows = load_economic_wm_replay_feature_rows(rows_path)
    contract = build_economic_wm_teacher_provider_contract(
        scaffold_report=scaffold_report,
        allocation_eval=allocation_eval,
        corpus_manifest=corpus_manifest,
        rows=rows,
        artifact_refs={
            "scaffold_report_path": str(scaffold_report_path),
            "allocation_eval_path": str(allocation_eval_path),
            "corpus_manifest_path": str(corpus_manifest_path),
            "rows_path": str(rows_path),
        },
        metadata=metadata,
    )
    save_economic_wm_teacher_provider_contract(output_path, contract)
    return contract


__all__ = [
    "ECONOMIC_WM_EVIDENCE_REQUIREMENT_VERSION",
    "ECONOMIC_WM_TEACHER_PROVIDER_CONTRACT_VERSION",
    "EconomicWMEvidenceRequirement",
    "EconomicWMTeacherProviderContract",
    "build_economic_wm_teacher_provider_contract",
    "build_economic_wm_teacher_provider_contract_from_paths",
    "load_economic_wm_teacher_provider_contract",
    "save_economic_wm_teacher_provider_contract",
]
