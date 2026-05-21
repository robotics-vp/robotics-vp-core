"""Deterministic Economic WM scaffold from typed lower-WM readiness receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.economics.economic_wm_entry import (
    EconomicWMEntryPreflightReport,
    evaluate_economic_wm_entry_preflight,
)
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

ECONOMIC_STATE_VERSION = "economic_state_v1"
ALLOCATION_ENVELOPE_VERSION = "allocation_envelope_v1"
ECONOMIC_WM_SCAFFOLD_REPORT_VERSION = "economic_wm_scaffold_report_v1"


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


def _report(
    payload: Mapping[str, Any] | EconomicWMEntryPreflightReport,
) -> EconomicWMEntryPreflightReport:
    if isinstance(payload, EconomicWMEntryPreflightReport):
        return payload
    if str(payload.get("version", "")) == "economic_wm_entry_preflight_v1":
        return EconomicWMEntryPreflightReport.from_dict(payload)
    return evaluate_economic_wm_entry_preflight(stage1_sweep_report=payload)


@dataclass(frozen=True)
class EconomicState:
    """First native Economic WM state artifact.

    This is not a learned latent. It is a typed state estimate over available
    lower-WM receipts so later learned models have a stable target surface.
    """

    state_id: str
    regime: str
    resource_reservoirs: Dict[str, float] = field(default_factory=dict)
    flow_fields: Dict[str, float] = field(default_factory=dict)
    dissipation_fields: Dict[str, float] = field(default_factory=dict)
    bottleneck_map: Dict[str, float] = field(default_factory=dict)
    opportunity_field: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_STATE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "version": self.version,
            "regime": self.regime,
            "resource_reservoirs": _float_dict(self.resource_reservoirs),
            "flow_fields": _float_dict(self.flow_fields),
            "dissipation_fields": _float_dict(self.dissipation_fields),
            "bottleneck_map": _float_dict(self.bottleneck_map),
            "opportunity_field": _float_dict(self.opportunity_field),
            "confidence": float(self.confidence),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicState":
        return cls(
            state_id=str(payload.get("state_id", "")),
            regime=str(payload.get("regime", "blocked")),
            resource_reservoirs=_float_dict(payload.get("resource_reservoirs", {})),
            flow_fields=_float_dict(payload.get("flow_fields", {})),
            dissipation_fields=_float_dict(payload.get("dissipation_fields", {})),
            bottleneck_map=_float_dict(payload.get("bottleneck_map", {})),
            opportunity_field=_float_dict(payload.get("opportunity_field", {})),
            confidence=float(payload.get("confidence", 0.0)),
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_STATE_VERSION)),
        )


@dataclass(frozen=True)
class AllocationEnvelope:
    """Downward scaffold-only allocation envelope from the Economic WM."""

    envelope_id: str
    readiness_class: str
    allowed_actions: list[str] = field(default_factory=list)
    denied_actions: list[str] = field(default_factory=list)
    budget_envelopes: Dict[str, float] = field(default_factory=dict)
    shaping_fields: Dict[str, float] = field(default_factory=dict)
    persistence_annotations: Dict[str, Any] = field(default_factory=dict)
    authority_class: str = "scaffold_only"
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ALLOCATION_ENVELOPE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "version": self.version,
            "readiness_class": self.readiness_class,
            "allowed_actions": list(self.allowed_actions),
            "denied_actions": list(self.denied_actions),
            "budget_envelopes": _float_dict(self.budget_envelopes),
            "shaping_fields": _float_dict(self.shaping_fields),
            "persistence_annotations": _mapping(self.persistence_annotations),
            "authority_class": self.authority_class,
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AllocationEnvelope":
        return cls(
            envelope_id=str(payload.get("envelope_id", "")),
            readiness_class=str(payload.get("readiness_class", "blocked")),
            allowed_actions=[
                str(item) for item in list(payload.get("allowed_actions", []) or [])
            ],
            denied_actions=[
                str(item) for item in list(payload.get("denied_actions", []) or [])
            ],
            budget_envelopes=_float_dict(payload.get("budget_envelopes", {})),
            shaping_fields=_float_dict(payload.get("shaping_fields", {})),
            persistence_annotations=_mapping(payload.get("persistence_annotations")),
            authority_class=str(payload.get("authority_class", "scaffold_only")),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ALLOCATION_ENVELOPE_VERSION)),
        )


@dataclass(frozen=True)
class EconomicWMScaffoldReport:
    """Receipt proving the first Economic WM scaffold artifacts were built."""

    scaffold_id: str
    economic_state: EconomicState
    allocation_envelope: AllocationEnvelope
    entry_preflight: Dict[str, Any]
    ready_for_scaffold: bool
    ready_for_training: bool
    scaffold_blockers: list[str] = field(default_factory=list)
    training_blockers: list[str] = field(default_factory=list)
    promotion_eligible: bool = False
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SCAFFOLD_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scaffold_id": self.scaffold_id,
            "version": self.version,
            "economic_state": self.economic_state.to_dict(),
            "allocation_envelope": self.allocation_envelope.to_dict(),
            "entry_preflight": _mapping(self.entry_preflight),
            "ready_for_scaffold": bool(self.ready_for_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "scaffold_blockers": list(self.scaffold_blockers),
            "training_blockers": list(self.training_blockers),
            "promotion_eligible": bool(self.promotion_eligible),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMScaffoldReport":
        return cls(
            scaffold_id=str(payload.get("scaffold_id", "")),
            economic_state=EconomicState.from_dict(
                dict(payload.get("economic_state", {}) or {})
            ),
            allocation_envelope=AllocationEnvelope.from_dict(
                dict(payload.get("allocation_envelope", {}) or {})
            ),
            entry_preflight=_mapping(payload.get("entry_preflight")),
            ready_for_scaffold=bool(payload.get("ready_for_scaffold", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            scaffold_blockers=[
                str(item) for item in list(payload.get("scaffold_blockers", []) or [])
            ],
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_SCAFFOLD_REPORT_VERSION)),
        )


def build_economic_state(
    entry_preflight: Mapping[str, Any] | EconomicWMEntryPreflightReport,
) -> EconomicState:
    report = _report(entry_preflight)
    payload = report.to_dict()
    counts = dict(payload.get("counts", {}) or {})
    scenario_count = max(1.0, float(counts.get("scenario_count", 0) or 0))
    admission_count = float(counts.get("admission_count", 0) or 0)
    admission_denominator = max(1.0, admission_count)
    benchmark_ready = float(counts.get("benchmark_ready_count", 0) or 0)
    shadow_only = float(counts.get("shadow_only_count", 0) or 0)
    training_blockers = list(payload.get("training_blockers", []) or [])
    scaffold_ready = bool(payload.get("ready_for_scaffold", False))

    resource_reservoirs = {
        "replay_datapack_inventory": admission_count,
        "benchmark_ready_inventory": benchmark_ready,
        "shadow_only_inventory": shadow_only,
        "training_gpu_budget_available": 0.0,
        "provider_runtime_capacity_available": 0.0,
    }
    flow_fields = {
        "benchmark_ready_flow": benchmark_ready / scenario_count,
        "shadow_data_flow": shadow_only / scenario_count,
        "replay_export_flow": min(
            float(counts.get("rlds_episode_count", 0) or 0),
            float(counts.get("lerobot_row_count", 0) or 0),
        )
        / admission_denominator,
    }
    dissipation_fields = {
        "grounding_uncertainty": shadow_only / scenario_count,
        "provider_friction": 1.0
        if "provider_bringup_not_run" in training_blockers
        else 0.0,
        "gpu_training_friction": 1.0
        if "gpu_training_not_run" in training_blockers
        else 0.0,
        "promotion_friction": 1.0
        if "promotion_grade_benchmark_evidence_missing" in training_blockers
        else 0.0,
    }
    bottleneck_map = {str(blocker): 1.0 for blocker in training_blockers}
    opportunity_field = {
        "economic_wm_scaffold_contracts": 1.0 if scaffold_ready else 0.0,
        "economic_replay_feature_extraction": 1.0 if scaffold_ready else 0.0,
        "collect_non_stub_teacher_runtime_evidence": 1.0
        if "non_stub_teacher_runtime_not_verified" in training_blockers
        else 0.0,
        "prepare_gpu_training_contract": 1.0
        if "gpu_training_not_run" in training_blockers
        else 0.0,
    }
    confidence = 0.75 if scaffold_ready else 0.2
    base = {
        "regime": payload.get("readiness_class", "blocked"),
        "resource_reservoirs": resource_reservoirs,
        "flow_fields": flow_fields,
        "dissipation_fields": dissipation_fields,
        "bottleneck_map": bottleneck_map,
        "opportunity_field": opportunity_field,
        "confidence": confidence,
        "preflight_report_id": payload.get("report_id", ""),
    }
    return EconomicState(
        state_id=f"econ_state_{sha256_json(base)[:16]}",
        regime=str(payload.get("readiness_class", "blocked")),
        resource_reservoirs=resource_reservoirs,
        flow_fields=flow_fields,
        dissipation_fields=dissipation_fields,
        bottleneck_map=bottleneck_map,
        opportunity_field=opportunity_field,
        confidence=confidence,
        source_refs=dict(payload.get("artifact_refs", {}) or {}),
        metadata={
            "entry_preflight_report_id": payload.get("report_id", ""),
            "ready_for_scaffold": bool(payload.get("ready_for_scaffold", False)),
            "ready_for_training": bool(payload.get("ready_for_training", False)),
        },
    )


def build_allocation_envelope(state: EconomicState) -> AllocationEnvelope:
    scaffold_ready = bool(state.metadata.get("ready_for_scaffold", False))
    allowed = (
        [
            "build_economic_wm_scaffold",
            "extract_replay_features",
            "materialize_training_rows",
            "run_shadow_allocation_evals",
        ]
        if scaffold_ready
        else ["collect_readiness_evidence"]
    )
    denied = [
        "gpu_training",
        "model_promotion",
        "reward_math_mutation",
        "stable_phase_b_rewrite",
        "external_provider_truth_promotion",
    ]
    budgets = {
        "local_scaffold_budget": 1.0 if scaffold_ready else 0.25,
        "gpu_training_budget": 0.0,
        "promotion_budget": 0.0,
    }
    shaping = {
        "prefer_benchmark_ready_replay": state.flow_fields.get(
            "benchmark_ready_flow", 0.0
        ),
        "prefer_shadow_gap_collection": state.flow_fields.get("shadow_data_flow", 0.0),
        "penalize_provider_friction": state.dissipation_fields.get(
            "provider_friction", 0.0
        ),
    }
    payload = {
        "readiness_class": state.regime,
        "state_id": state.state_id,
        "allowed_actions": allowed,
        "denied_actions": denied,
        "budget_envelopes": budgets,
        "shaping_fields": shaping,
    }
    return AllocationEnvelope(
        envelope_id=f"alloc_env_{sha256_json(payload)[:16]}",
        readiness_class=state.regime,
        allowed_actions=allowed,
        denied_actions=denied,
        budget_envelopes=budgets,
        shaping_fields=shaping,
        persistence_annotations={
            "hold_training_block_until": "gpu_provider_promotion_evidence_exists",
            "timescale": "scaffold_pretraining",
        },
        promotion_eligible=False,
        metadata={"economic_state_id": state.state_id},
    )


def build_economic_wm_scaffold_report(
    entry_preflight: Mapping[str, Any] | EconomicWMEntryPreflightReport,
) -> EconomicWMScaffoldReport:
    report = _report(entry_preflight)
    state = build_economic_state(report)
    envelope = build_allocation_envelope(state)
    payload = {
        "state_id": state.state_id,
        "envelope_id": envelope.envelope_id,
        "entry_preflight_report_id": report.report_id,
    }
    return EconomicWMScaffoldReport(
        scaffold_id=f"econ_wm_scaffold_{sha256_json(payload)[:16]}",
        economic_state=state,
        allocation_envelope=envelope,
        entry_preflight=report.to_dict(),
        ready_for_scaffold=bool(report.ready_for_scaffold),
        ready_for_training=bool(report.ready_for_training),
        scaffold_blockers=list(report.scaffold_blockers),
        training_blockers=list(report.training_blockers),
        promotion_eligible=False,
        artifact_refs={
            "entry_preflight_report_id": report.report_id,
            **dict(report.artifact_refs),
        },
        metadata={
            "authority_class": "scaffold_only",
            "reward_math_mutation": False,
            "training_claim": False,
        },
    )


def save_economic_wm_scaffold_report(
    path: str | Path,
    report: EconomicWMScaffoldReport,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_scaffold_report(path: str | Path) -> EconomicWMScaffoldReport:
    return EconomicWMScaffoldReport.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "ALLOCATION_ENVELOPE_VERSION",
    "ECONOMIC_STATE_VERSION",
    "ECONOMIC_WM_SCAFFOLD_REPORT_VERSION",
    "AllocationEnvelope",
    "EconomicState",
    "EconomicWMScaffoldReport",
    "build_allocation_envelope",
    "build_economic_state",
    "build_economic_wm_scaffold_report",
    "load_economic_wm_scaffold_report",
    "save_economic_wm_scaffold_report",
]
