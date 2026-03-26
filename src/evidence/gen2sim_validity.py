"""Explicit gen2sim validity/value admission assessment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.evidence.benchmark_gating import (
    build_benchmark_gate_report,
    collect_benchmark_gating_signals,
)
from src.evidence.preconditions import (
    ExecutionPreconditionsReport,
    build_execution_preconditions,
)
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

GEN2SIM_OBJECTIVE_DIM = 4
GEN2SIM_FEATURE_NAMES = [
    "trust_score",
    "std_ratio_alignment",
    "plausibility_score",
    "reward_safety_score",
    "coverage_gap_contribution",
    "economic_priority",
    "branch_value_support",
    "scene_tracks_backend_real",
    "semantic_grounding_non_heuristic",
    "vision_backbone_real",
    "teacher_runtime_real",
    "benchmark_eligible",
    "objective_0",
    "objective_1",
    "objective_2",
    "objective_3",
]


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _squash_positive(value: float) -> float:
    return float(max(0.0, value) / (1.0 + max(0.0, value)))


def _std_ratio_alignment(std_ratio: Any) -> float:
    ratio = _safe_float(std_ratio, 0.0)
    if ratio <= 0.0:
        return 0.0
    return _clip01(1.0 - min(1.0, abs(ratio - 1.0) / 0.35))


def _generated_source(context: Mapping[str, Any]) -> bool:
    source_candidates = [
        context.get("source_domain"),
        context.get("source"),
        context.get("source_type"),
        context.get("subject_kind"),
    ]
    normalized = " ".join(str(item or "").lower() for item in source_candidates)
    return any(
        token in normalized
        for token in (
            "synthetic",
            "synth",
            "gen2sim",
            "generated",
            "diffusion",
            "branch",
        )
    )


def _compose_metadata(context: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = _mapping(context.get("metadata"))
    for key in (
        "scene_tracks_backend",
        "teacher_runtime_backend_selected",
        "openvla_backend_selected",
        "vision_backbone_selected",
        "semantic_grounding_mode",
        "semantic_memory_grounded",
        "source_runtime_metadata",
        "branch_gap_labels",
        "plausibility_score",
        "reward_safety_score",
        "coverage_gap_contribution",
        "economic_priority",
        "branch_value",
        "trust_score",
        "std_ratio",
    ):
        value = context.get(key)
        if value not in (None, "", []):
            metadata[key] = value
    benchmark_signals = context.get("benchmark_signals")
    if isinstance(benchmark_signals, Mapping):
        metadata.update(_mapping(benchmark_signals))
    return metadata


def _objective_vector(
    context: Mapping[str, Any],
    *,
    objective_dim: int = GEN2SIM_OBJECTIVE_DIM,
) -> list[float]:
    objective_vector = context.get("objective_vector")
    if objective_vector is not None:
        values = [_safe_float(value, 0.0) for value in list(objective_vector or [])]
    else:
        values = []
        objective_tensor_payload = context.get("objective_tensor")
        if objective_tensor_payload is not None:
            try:
                from src.objectives.tensor import ObjectiveTensor

                if isinstance(objective_tensor_payload, ObjectiveTensor):
                    values = objective_tensor_payload.mean_vector(normalize=True).tolist()
                elif isinstance(objective_tensor_payload, Mapping):
                    values = ObjectiveTensor.from_dict(objective_tensor_payload).mean_vector(
                        normalize=True
                    ).tolist()
            except Exception:
                values = []
    padded = values[:objective_dim]
    if len(padded) < objective_dim:
        padded.extend([0.0] * (objective_dim - len(padded)))
    return [_clip01(_safe_float(value, 0.0)) for value in padded]


def build_gen2sim_feature_dict(
    context: Mapping[str, Any],
    *,
    objective_dim: int = GEN2SIM_OBJECTIVE_DIM,
) -> Dict[str, float]:
    metadata = _compose_metadata(context)
    gap_labels = _mapping(context.get("gap_labels"))
    benchmark_signals = collect_benchmark_gating_signals(metadata)
    objective_vector = _objective_vector(context, objective_dim=objective_dim)
    feature_dict = {
        "trust_score": _clip01(_safe_float(context.get("trust_score"), metadata.get("trust_score", 0.0))),
        "std_ratio_alignment": _std_ratio_alignment(
            context.get("std_ratio", metadata.get("std_ratio"))
        ),
        "plausibility_score": _clip01(
            _safe_float(context.get("plausibility_score"), metadata.get("plausibility_score", 1.0))
        ),
        "reward_safety_score": _clip01(
            _safe_float(
                context.get("reward_safety_score"),
                metadata.get("reward_safety_score", 1.0),
            )
        ),
        "coverage_gap_contribution": _clip01(
            _safe_float(
                gap_labels.get("coverage_gap_contribution"),
                metadata.get("coverage_gap_contribution", 0.0),
            )
        ),
        "economic_priority": _clip01(
            _safe_float(
                gap_labels.get("economic_priority"),
                metadata.get("economic_priority", 0.0),
            )
        ),
        "branch_value_support": _squash_positive(
            _safe_float(context.get("branch_value"), metadata.get("branch_value", 0.0))
        ),
        "scene_tracks_backend_real": float(
            bool(benchmark_signals.get("scene_tracks_backend_real", False))
        ),
        "semantic_grounding_non_heuristic": float(
            bool(benchmark_signals.get("semantic_grounding_non_heuristic", False))
        ),
        "vision_backbone_real": float(bool(benchmark_signals.get("vision_backbone_real", False))),
        "teacher_runtime_real": float(bool(benchmark_signals.get("teacher_runtime_real", False))),
        "benchmark_eligible": float(bool(benchmark_signals.get("benchmark_eligible", False))),
    }
    for idx, value in enumerate(objective_vector):
        feature_dict[f"objective_{idx}"] = _clip01(_safe_float(value, 0.0))
    return feature_dict


def build_gen2sim_feature_vector(
    context: Mapping[str, Any],
    *,
    objective_dim: int = GEN2SIM_OBJECTIVE_DIM,
) -> list[float]:
    feature_dict = build_gen2sim_feature_dict(context, objective_dim=objective_dim)
    feature_names = GEN2SIM_FEATURE_NAMES[: 12 + objective_dim]
    return [_safe_float(feature_dict.get(name), 0.0) for name in feature_names]


def _coerce_helper_inference(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return _mapping(payload)
    if hasattr(payload, "to_dict"):
        try:
            return _mapping(payload.to_dict())
        except Exception:
            return {}
    if hasattr(payload, "__dict__"):
        return _mapping(vars(payload))
    return {}


def _infer_helper_payload(helper: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
    if helper is None:
        return {}
    if hasattr(helper, "infer_context"):
        return _coerce_helper_inference(helper.infer_context(context=context))
    if hasattr(helper, "predict_context"):
        return _coerce_helper_inference(helper.predict_context(context=context))
    if callable(helper):
        return _coerce_helper_inference(helper(context))
    return {}


def _blend_weight(helper_status: Mapping[str, Any], helper_inference: Mapping[str, Any]) -> float:
    promotion_stage = str(
        helper_inference.get("promotion_stage")
        or helper_status.get("promotion_stage")
        or "heuristic_fallback"
    )
    if promotion_stage == "promoted":
        return 0.25
    if promotion_stage == "shadow_candidate":
        return 0.12
    return 0.0


def _attach_helper_trace(
    prior: Gen2SimValidityAssessment,
    *,
    context: Mapping[str, Any],
    helper: Any = None,
    helper_status: Optional[Mapping[str, Any]] = None,
) -> Gen2SimValidityAssessment:
    helper_status_payload = _mapping(helper_status)
    helper_inference = _infer_helper_payload(helper, context)
    blend_weight = _blend_weight(helper_status_payload, helper_inference)

    learned_validity = _clip01(
        _safe_float(helper_inference.get("predicted_validity_score"), prior.validity_score)
    )
    learned_value_support = _clip01(
        _safe_float(
            helper_inference.get("predicted_value_support_score"),
            prior.value_support_score,
        )
    )
    validity = _clip01(
        prior.validity_score + (blend_weight * (learned_validity - prior.validity_score))
    )
    value_support = _clip01(
        prior.value_support_score
        + (blend_weight * (learned_value_support - prior.value_support_score))
    )

    reason_codes = list(prior.reason_codes)
    helper_status_name = str(helper_status_payload.get("status", "") or "")
    if helper_status_name and helper_status_name != "disabled":
        if helper_inference and blend_weight > 0.0:
            reason_codes.append("learned_helper_adjustment_applied")
        elif helper_inference:
            reason_codes.append("learned_helper_observed_only")
        else:
            reason_codes.append(f"learned_helper_{helper_status_name}")

    metadata = {
        **dict(prior.metadata),
        "helper_status": helper_status_payload,
        "helper_inference": helper_inference,
        "conditioning_features": build_gen2sim_feature_dict(context),
        "helper_blend": {
            "weight": float(blend_weight),
            "learned_validity_score": learned_validity,
            "learned_value_support_score": learned_value_support,
        },
    }
    component_scores = {
        **dict(prior.component_scores),
        "helper_blend_weight": float(blend_weight),
        "helper_predicted_validity": learned_validity,
        "helper_predicted_value_support": learned_value_support,
    }
    return _assessment_from_scores(
        subject_id=prior.subject_id,
        subject_kind=prior.subject_kind,
        benchmark_gate=ExecutionPreconditionsReport.from_dict(prior.benchmark_gate),
        execution_preconditions=ExecutionPreconditionsReport.from_dict(
            prior.execution_preconditions
        ),
        validity_score=validity,
        value_support_score=value_support,
        component_scores=component_scores,
        reason_codes=reason_codes,
        metadata=metadata,
    )


@dataclass(frozen=True)
class Gen2SimValidityAssessment:
    assessment_id: str
    subject_id: str
    subject_kind: str
    validity_score: float
    value_support_score: float
    admission_score: float
    promotion_stage: str
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    component_scores: Dict[str, float] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "gen2sim_validity_assessment_v1"

    @property
    def benchmark_gate_ready(self) -> bool:
        return bool(self.benchmark_gate.get("ready", False))

    @property
    def execution_ready(self) -> bool:
        return bool(self.execution_preconditions.get("ready", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "validity_score": float(self.validity_score),
            "value_support_score": float(self.value_support_score),
            "admission_score": float(self.admission_score),
            "promotion_stage": self.promotion_stage,
            "benchmark_gate": dict(self.benchmark_gate),
            "execution_preconditions": dict(self.execution_preconditions),
            "component_scores": {
                str(key): float(value)
                for key, value in self.component_scores.items()
            },
            "reason_codes": list(self.reason_codes),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Gen2SimValidityAssessment":
        return cls(
            assessment_id=str(payload.get("assessment_id", "")),
            subject_id=str(payload.get("subject_id", "")),
            subject_kind=str(payload.get("subject_kind", "")),
            validity_score=_clip01(_safe_float(payload.get("validity_score", 0.0))),
            value_support_score=_clip01(
                _safe_float(payload.get("value_support_score", 0.0))
            ),
            admission_score=_clip01(_safe_float(payload.get("admission_score", 0.0))),
            promotion_stage=str(payload.get("promotion_stage", "heuristic_fallback")),
            benchmark_gate=_mapping(payload.get("benchmark_gate")),
            execution_preconditions=_mapping(payload.get("execution_preconditions")),
            component_scores={
                str(key): _clip01(_safe_float(value, 0.0))
                for key, value in dict(payload.get("component_scores", {}) or {}).items()
            },
            reason_codes=[str(item) for item in list(payload.get("reason_codes", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "gen2sim_validity_assessment_v1")),
        )


def coerce_gen2sim_validity_assessment(
    payload: Gen2SimValidityAssessment | Mapping[str, Any] | None,
) -> Gen2SimValidityAssessment | None:
    if payload is None:
        return None
    if isinstance(payload, Gen2SimValidityAssessment):
        return payload
    return Gen2SimValidityAssessment.from_dict(payload)


def load_gen2sim_validity_assessment(path: str | Path) -> Gen2SimValidityAssessment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"gen2sim validity assessment is not an object: {path}")
    assessment = coerce_gen2sim_validity_assessment(payload)
    if assessment is None:
        raise ValueError(f"gen2sim validity assessment is empty: {path}")
    return assessment


def _assessment_from_scores(
    *,
    subject_id: str,
    subject_kind: str,
    benchmark_gate: ExecutionPreconditionsReport,
    execution_preconditions: ExecutionPreconditionsReport,
    validity_score: float,
    value_support_score: float,
    component_scores: Mapping[str, float],
    reason_codes: list[str],
    metadata: Mapping[str, Any],
) -> Gen2SimValidityAssessment:
    validity = _clip01(validity_score)
    value_support = _clip01(value_support_score)
    admission = _clip01(validity * (0.75 + (0.25 * value_support)))
    if benchmark_gate.ready and execution_preconditions.ready and admission >= 0.72:
        promotion_stage = "promoted"
    elif execution_preconditions.ready and admission >= 0.35:
        promotion_stage = "shadow_candidate"
    else:
        promotion_stage = "heuristic_fallback"
    payload = {
        "subject_id": subject_id,
        "subject_kind": subject_kind,
        "validity_score": validity,
        "value_support_score": value_support,
        "admission_score": admission,
        "promotion_stage": promotion_stage,
        "benchmark_gate": benchmark_gate.to_dict(),
        "execution_preconditions": execution_preconditions.to_dict(),
        "component_scores": dict(component_scores),
        "reason_codes": list(reason_codes),
        "metadata": _mapping(metadata),
        "version": "gen2sim_validity_assessment_v1",
    }
    return Gen2SimValidityAssessment(
        assessment_id=f"gen2sim_{sha256_json(payload)[:16]}",
        subject_id=subject_id,
        subject_kind=subject_kind,
        validity_score=validity,
        value_support_score=value_support,
        admission_score=admission,
        promotion_stage=promotion_stage,
        benchmark_gate=benchmark_gate.to_dict(),
        execution_preconditions=execution_preconditions.to_dict(),
        component_scores={
            str(key): _clip01(_safe_float(value, 0.0))
            for key, value in dict(component_scores).items()
        },
        reason_codes=list(reason_codes),
        metadata=_mapping(metadata),
    )


def assess_gen2sim_validity(
    *,
    subject_id: str,
    subject_kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
    trust_score: Any = None,
    std_ratio: Any = None,
    branch_value: Any = None,
    gap_labels: Optional[Mapping[str, Any]] = None,
    plausibility_score: Any = 1.0,
    reward_safety_score: Any = 1.0,
    legacy_validity_hint: Any = None,
    require_real_scene_tracks: bool = True,
    require_teacher_runtime: bool = False,
    require_vision_backbone: bool = True,
) -> Gen2SimValidityAssessment:
    metadata_payload = _mapping(metadata)
    gap_payload = _mapping(gap_labels)
    benchmark_signals = collect_benchmark_gating_signals(metadata_payload)
    benchmark_gate = build_benchmark_gate_report(
        subject_id=subject_id,
        subject_kind=subject_kind,
        metadata=metadata_payload,
        require_real_scene_tracks=require_real_scene_tracks,
        require_teacher_runtime=require_teacher_runtime,
        require_vision_backbone=require_vision_backbone,
    )

    trust = _clip01(_safe_float(trust_score, metadata_payload.get("trust_score", 0.0)))
    std_alignment = _std_ratio_alignment(
        std_ratio if std_ratio is not None else metadata_payload.get("std_ratio")
    )
    plausibility = _clip01(
        _safe_float(plausibility_score, metadata_payload.get("plausibility_score", 1.0))
    )
    reward_safety = _clip01(
        _safe_float(
            reward_safety_score,
            metadata_payload.get("reward_safety_score", 1.0),
        )
    )
    coverage_gap = _clip01(
        _safe_float(
            gap_payload.get("coverage_gap_contribution"),
            metadata_payload.get("coverage_gap_contribution", 0.0),
        )
    )
    econ_priority = _clip01(
        _safe_float(
            gap_payload.get("economic_priority"),
            metadata_payload.get("economic_priority", 0.0),
        )
    )
    branch_value_support = _squash_positive(
        _safe_float(branch_value, metadata_payload.get("branch_value", 0.0))
    )

    dynamics_score = _clip01((0.65 * trust) + (0.35 * std_alignment))
    grounding_score = _clip01(
        (0.45 if benchmark_signals.get("semantic_grounding_non_heuristic", False) else 0.0)
        + (0.25 if benchmark_signals.get("scene_tracks_backend_real", False) else 0.0)
        + (0.20 if benchmark_signals.get("vision_backbone_real", False) else 0.0)
        + (0.10 if benchmark_signals.get("teacher_runtime_real", False) else 0.0)
    )
    safety_score = _clip01((0.6 * plausibility) + (0.4 * reward_safety))
    value_support = _clip01((0.65 * branch_value_support) + (0.35 * max(coverage_gap, econ_priority)))

    execution_preconditions = build_execution_preconditions(
        subject_id=subject_id,
        subject_kind=subject_kind,
        artifact_refs={
            "source_runtime_metadata": metadata_payload.get("source_runtime_metadata"),
            "branch_gap_labels": metadata_payload.get("branch_gap_labels"),
        },
        soft_required_artifact_refs=["source_runtime_metadata", "branch_gap_labels"],
        signal_values={
            "trust_score": trust,
            "std_ratio_alignment": std_alignment,
            "plausibility_score": plausibility,
            "reward_safety_score": reward_safety,
            "value_support_score": value_support,
            "coverage_gap_contribution": coverage_gap,
            "economic_priority": econ_priority,
            **benchmark_signals,
        },
        min_signal_thresholds={
            "trust_score": 0.75,
            "std_ratio_alignment": 0.35,
            "plausibility_score": 0.45,
            "reward_safety_score": 0.55,
        },
        soft_boolean_signals={
            "semantic_grounding_non_heuristic": True,
            "benchmark_eligible": True,
        },
        metadata={"assessment_contract": "gen2sim_validity_assessment_v1"},
    )

    validity = (0.45 * dynamics_score) + (0.35 * grounding_score) + (0.20 * safety_score)
    if legacy_validity_hint is not None:
        validity = (0.85 * validity) + (0.15 * _clip01(_safe_float(legacy_validity_hint, 0.0)))
    if not execution_preconditions.ready:
        validity = min(validity, 0.45)

    reasons: list[str] = []
    if not benchmark_gate.ready:
        reasons.append("benchmark_gate_not_ready")
    if not execution_preconditions.ready:
        reasons.append("execution_preconditions_not_ready")
    if dynamics_score < 0.5:
        reasons.append("dynamics_consistency_low")
    if grounding_score < 0.5:
        reasons.append("grounding_support_low")
    if value_support < 0.2:
        reasons.append("value_support_low")
    if legacy_validity_hint is not None:
        reasons.append("legacy_validity_hint_present")
    if not reasons:
        reasons.append("gen2sim_validity_ok")

    return _assessment_from_scores(
        subject_id=subject_id,
        subject_kind=subject_kind,
        benchmark_gate=benchmark_gate,
        execution_preconditions=execution_preconditions,
        validity_score=validity,
        value_support_score=value_support,
        component_scores={
            "dynamics_score": dynamics_score,
            "grounding_score": grounding_score,
            "safety_score": safety_score,
            "branch_value_support": branch_value_support,
            "gap_value_support": max(coverage_gap, econ_priority),
        },
        reason_codes=reasons,
        metadata={
            **metadata_payload,
            "benchmark_signals": benchmark_signals,
            "coverage_gap_contribution": coverage_gap,
            "economic_priority": econ_priority,
            "trust_score": trust,
            "std_ratio_alignment": std_alignment,
        },
    )


def resolve_gen2sim_validity_assessment(
    context: Mapping[str, Any],
    *,
    subject_id: Optional[str] = None,
    subject_kind: Optional[str] = None,
    helper: Any = None,
    helper_status: Optional[Mapping[str, Any]] = None,
) -> Gen2SimValidityAssessment:
    explicit = coerce_gen2sim_validity_assessment(context.get("gen2sim_validity_assessment"))
    if explicit is not None:
        return explicit

    report_path = context.get("gen2sim_validity_report_path") or context.get(
        "gen2sim_validity_path"
    )
    if isinstance(report_path, str) and report_path.strip():
        path = Path(report_path)
        if path.exists():
            return load_gen2sim_validity_assessment(path)

    resolved_subject_id = str(
        subject_id
        or context.get("datapack_id")
        or context.get("episode_id")
        or context.get("task_id")
        or "gen2sim_candidate"
    )
    resolved_subject_kind = str(
        subject_kind
        or context.get("subject_kind")
        or context.get("source")
        or context.get("source_domain")
        or "candidate"
    )

    if not _generated_source(context) and context.get("gen2sim_validity_score") is None:
        ready_report = build_execution_preconditions(
            subject_id=resolved_subject_id,
            subject_kind=resolved_subject_kind,
            signal_values={"not_applicable": 1.0},
            min_signal_thresholds={"not_applicable": 1.0},
            metadata={"assessment_contract": "gen2sim_not_applicable_v1"},
        )
        assessment = _assessment_from_scores(
            subject_id=resolved_subject_id,
            subject_kind=resolved_subject_kind,
            benchmark_gate=ready_report,
            execution_preconditions=ready_report,
            validity_score=1.0,
            value_support_score=1.0,
            component_scores={
                "dynamics_score": 1.0,
                "grounding_score": 1.0,
                "safety_score": 1.0,
                "branch_value_support": 1.0,
                "gap_value_support": 1.0,
            },
            reason_codes=["gen2sim_not_applicable"],
            metadata={"source_generated": False},
        )
        return _attach_helper_trace(
            assessment,
            context=context,
            helper=helper,
            helper_status=helper_status,
        )

    metadata = _compose_metadata(context)
    gap_labels = _mapping(context.get("gap_labels"))
    assessment = assess_gen2sim_validity(
        subject_id=resolved_subject_id,
        subject_kind=resolved_subject_kind,
        metadata=metadata,
        trust_score=context.get("trust_score"),
        std_ratio=context.get("std_ratio"),
        branch_value=context.get("branch_value"),
        gap_labels=gap_labels,
        plausibility_score=context.get("plausibility_score", 1.0),
        reward_safety_score=context.get("reward_safety_score", 1.0),
        legacy_validity_hint=context.get("gen2sim_validity_score"),
    )
    return _attach_helper_trace(
        assessment,
        context=context,
        helper=helper,
        helper_status=helper_status,
    )


__all__ = [
    "GEN2SIM_FEATURE_NAMES",
    "GEN2SIM_OBJECTIVE_DIM",
    "Gen2SimValidityAssessment",
    "assess_gen2sim_validity",
    "build_gen2sim_feature_dict",
    "build_gen2sim_feature_vector",
    "coerce_gen2sim_validity_assessment",
    "load_gen2sim_validity_assessment",
    "resolve_gen2sim_validity_assessment",
]
