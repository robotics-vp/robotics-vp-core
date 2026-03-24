"""Packet/evidence/governance supervision bundle for governed video loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.constraints.constraint_set import ConstraintSet
from src.economics.counterfactual_eval import CounterfactualEval, build_counterfactual_eval
from src.economics.functor import ObjectiveEconFunctor
from src.economics.pricing_sentinel import PricingSentinel, PricingTick, PricingTickInput
from src.economics.value_ledger import ValueLedgerReceipt
from src.economics.value_targets import ValueTargetPack, build_value_target_pack
from src.governance.trace import GovernanceTraceEntry
from src.objectives.profile import ObjectiveProfile
from src.objectives.runtime_builder import ObjectiveRuntimeBuilder, ObjectiveRuntimeRecord, SourceDomain
from src.regal.base import RegalDecision, RegalReport
from src.regal.gen_plausibility import RegalGenPlausibilityNode
from src.runtime import (
    ActionAdapterV2,
    DecisionLedgerEntry,
    ObservationAdapterV2,
    RuntimeEvent,
    runtime_packet_from_record,
)
from src.runtime.packets import RuntimePacket
from src.utils.config_digest import sha256_json
from src.world_model.governed_video_world_model import GovernedVideoHypothesis, VideoStateSnapshot


def _timestamp(timestamp: Optional[str] = None) -> str:
    return timestamp or datetime.now(timezone.utc).isoformat()


def _objective_profile_for_preset(objective_preset: str) -> ObjectiveProfile:
    weights = {"throughput": 1.0, "error": -1.0, "safety": 1.0, "energy": -1.0}
    maximize = {"throughput": True, "error": False, "safety": True, "energy": False}
    if objective_preset == "throughput":
        weights["throughput"] = 2.0
    elif objective_preset == "safety":
        weights["safety"] = 2.5
        weights["error"] = -1.5
    elif objective_preset == "energy_saver":
        weights["energy"] = -2.0
    return ObjectiveProfile.weighted_sum(weights=weights, maximize=maximize, profile_id=f"{objective_preset}_contract")


def _runtime_metrics_from_state(
    *,
    snapshot: VideoStateSnapshot,
    belief_state: Any,
    objective_preset: str,
    semantic_tags: Sequence[str],
    duration_s: float,
) -> Dict[str, float]:
    confidence = float(snapshot.state_features.get("evidence_confidence_mean", 0.0))
    disagreement = float(snapshot.state_features.get("evidence_disagreement_mean", 0.0))
    throughput_base = 4.0 if objective_preset == "throughput" else 2.0
    if objective_preset == "energy_saver":
        energy_wh = 0.4
    else:
        energy_wh = 0.9
    if "fragile" in semantic_tags or objective_preset == "safety":
        throughput_base *= 0.85
        energy_wh *= 0.9
    items_completed = max(1.0, throughput_base * (duration_s / 3600.0))
    return {
        "items_completed": float(items_completed),
        "duration_s": float(max(duration_s, 1.0)),
        "error_rate": float(max(0.0, min(1.0, disagreement * 0.8))),
        "energy_wh": float(energy_wh),
        "safety_score": float(max(0.0, min(1.0, snapshot.state_features.get("geometry_quality", confidence)))),
        "map_first_quality_score": float(snapshot.state_features.get("geometry_quality", confidence)),
        "semantic_fusion_confidence_mean": float(snapshot.state_features.get("semantic_quality", confidence)),
        "semantic_disagreement_vla_vs_map": disagreement,
    }


@dataclass(frozen=True)
class VideoBranchEvaluation:
    """Evaluation over one governed hypothesis."""

    hypothesis_id: str
    mode: str
    regal_report: Dict[str, Any]
    expected_net_value: float
    decision: str
    reasons: list[str] = field(default_factory=list)
    action: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "mode": self.mode,
            "regal_report": dict(self.regal_report),
            "expected_net_value": float(self.expected_net_value),
            "decision": self.decision,
            "reasons": list(self.reasons),
            "action": dict(self.action),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GovernedVideoSupervisionBundle:
    """All supervision artifacts derived from one governed video episode."""

    runtime_packet: RuntimePacket
    pricing_tick: PricingTick
    branch_evaluations: list[VideoBranchEvaluation]
    events: list[RuntimeEvent]
    decisions: list[DecisionLedgerEntry]
    governance_traces: list[GovernanceTraceEntry]
    counterfactual_eval: CounterfactualEval
    value_target_pack: ValueTargetPack
    value_ledger_receipt: ValueLedgerReceipt


def build_governed_video_supervision_bundle(
    *,
    run_id: str,
    video_ref: Mapping[str, Any],
    semantic_tags: Sequence[str],
    belief_state: Any,
    snapshot: VideoStateSnapshot,
    hypotheses: Sequence[GovernedVideoHypothesis],
    objective_preset: str,
    constraint_set: ConstraintSet | Mapping[str, Any],
    sidecar_refs: Optional[Mapping[str, Any]] = None,
    value_ledger_path: str | Path | None = None,
    timestamp: Optional[str] = None,
) -> GovernedVideoSupervisionBundle:
    timestamp = _timestamp(timestamp)
    constraint = constraint_set if isinstance(constraint_set, ConstraintSet) else ConstraintSet.from_runtime(
        hard_constraints=(constraint_set or {}).get("hard_bounds"),
        soft_constraints=(constraint_set or {}).get("soft_bounds"),
        geometry_priors=(constraint_set or {}).get("geometry_priors"),
        semantic_evidence={"semantic_tags": list(semantic_tags)},
        uncertainty={"semantic_disagreement": float(snapshot.state_features.get("evidence_disagreement_mean", 0.0))},
        metadata={"source": "governed_video_supervision"},
    )

    duration_s = float((video_ref.get("metadata") or {}).get("duration_s", 12.0) or 12.0)
    metrics = _runtime_metrics_from_state(
        snapshot=snapshot,
        belief_state=belief_state,
        objective_preset=objective_preset,
        semantic_tags=semantic_tags,
        duration_s=duration_s,
    )
    runtime_record = ObjectiveRuntimeRecord(
        task_id=str(video_ref.get("task_type", "video_task")),
        episode_id=str(video_ref.get("episode_id", "episode")),
        env_id="stage1_video_loop",
        world_id="governed_video_world",
        robot_id="video_demonstration",
        source_domain=SourceDomain.REAL_LAB if video_ref.get("source_type") != "simulated_reference" else SourceDomain.SYNTHETIC,
        seed=0,
        run_id=run_id,
        timestamp=timestamp,
        episode_metrics=metrics,
        telemetry={
            "trust_score": float(snapshot.state_features.get("evidence_confidence_mean", 0.0)),
            "uncertainty": float(getattr(belief_state, "uncertainty", {}).get("epistemic", 0.0)),
        },
    )
    objective_builder = ObjectiveRuntimeBuilder()
    objective_tensor = objective_builder.build(runtime_record)
    objective_profile = _objective_profile_for_preset(objective_preset)
    compile_result = objective_builder.compile_contract(
        objective_tensor,
        objective_profile,
        metadata={"objective_preset": objective_preset},
    )
    econ_tensor = ObjectiveEconFunctor(base_price_per_unit=5.0).map(
        objective_tensor,
        constraint_flags=compile_result.constraint_flags,
        uncertainty=float(getattr(belief_state, "uncertainty", {}).get("epistemic", 0.0)),
        context={"objective_preset": objective_preset},
    )

    observation_adapter = ObservationAdapterV2(
        schema_id="governed_video_observation_v2",
        proprio_fields=["geometry_quality", "semantic_quality", "evidence_disagreement_mean"],
        sensor_refs=["video_ref", "scene_tracks", "semantic_fusion", "belief_state", "teacher_trace", "reconstruction"],
        sample_hz=1.0,
        translator_ref="governed_video_loop",
        embodiment_id="video_demonstration",
    )
    action_adapter = ActionAdapterV2(
        schema_id="governed_video_action_v2",
        channel_order=["speed_scale", "clearance_bias", "camera_reframe", "smoothness_bias", "regrasp_bias"],
        control_hz=1.0,
        translator_ref="governed_video_loop",
        embodiment_id="video_demonstration",
    )
    runtime_packet = runtime_packet_from_record(
        record=runtime_record,
        contract_id=f"contract.{objective_preset}.governed_video.v2",
        objective_profile_id=objective_profile.profile_id,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint,
        observation_schema=observation_adapter,
        action_schema=action_adapter,
        semantic_evidence={
            "belief_state_id": getattr(belief_state, "belief_id", ""),
            "semantic_tags": list(semantic_tags),
            **dict(sidecar_refs or {}),
        },
        uncertainty=getattr(belief_state, "uncertainty", {}),
        metadata={"video_state_id": snapshot.state_id},
    )

    pricing_tick = PricingSentinel().emit_tick(
        PricingTickInput(
            run_id=run_id,
            episode_id=runtime_record.episode_id,
            objective_profile_id=objective_profile.profile_id,
            source_domain=str(runtime_record.source_domain),
            timestamp=timestamp,
            mode="episode",
            econ_tensor=econ_tensor,
            uncertainty=float(getattr(belief_state, "uncertainty", {}).get("epistemic", 0.0)),
            constraint_flags=constraint.constraint_flags(metrics),
            trust_score=float(snapshot.state_features.get("evidence_confidence_mean", 0.0)),
            metadata={"video_state_id": snapshot.state_id},
        )
    )

    plausibility_node = RegalGenPlausibilityNode()
    branch_evaluations: list[VideoBranchEvaluation] = []
    events: list[RuntimeEvent] = []
    decisions: list[DecisionLedgerEntry] = []
    governance_traces: list[GovernanceTraceEntry] = []

    base_value = float(pricing_tick.net_customer_rate * max(0.1, snapshot.state_features.get("evidence_confidence_mean", 0.1)))
    for idx, hypothesis in enumerate(hypotheses, start=1):
        regal_report = _evaluate_branch_regality(hypothesis, snapshot, belief_state, plausibility_node)
        risk_penalty = 8.0 if regal_report.decision == RegalDecision.BLOCK else 0.0
        expected_net_value = float(
            pricing_tick.net_customer_rate
            * max(0.05, hypothesis.scores.get("objective_fit", 0.0))
            * max(0.05, hypothesis.scores.get("plausibility", 0.0))
            - risk_penalty
        )
        decision = "render"
        reasons = list(regal_report.reason_codes)
        if regal_report.decision == RegalDecision.BLOCK:
            decision = "collect_more_data" if "semantic_disagreement_high" in reasons else "skip"
        elif hypothesis.render_intent.get("should_render") is not True:
            decision = "skip"

        branch_evaluations.append(
            VideoBranchEvaluation(
                hypothesis_id=hypothesis.hypothesis_id,
                mode=hypothesis.mode,
                regal_report=regal_report.to_dict(),
                expected_net_value=expected_net_value,
                decision=decision,
                reasons=reasons,
                action=hypothesis.action_conditioning,
                metadata={"render_priority": hypothesis.scores.get("render_priority", 0.0)},
            )
        )

        event = RuntimeEvent.from_components(
            run_id=run_id,
            episode_id=runtime_record.episode_id,
            timestamp=timestamp,
            event_kind="video_hypothesis_evaluated",
            sequence_idx=idx,
            scope={"scope_kind": "episode", "hypothesis_id": hypothesis.hypothesis_id, "mode": hypothesis.mode},
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            artifact_refs={**dict(sidecar_refs or {}), "hypothesis_id": hypothesis.hypothesis_id},
            provenance={"component": "governed_video_supervision"},
            metadata={"decision": decision, "expected_net_value": expected_net_value},
        )
        events.append(event)
        decision_entry = DecisionLedgerEntry.from_components(
            run_id=run_id,
            episode_id=runtime_record.episode_id,
            timestamp=timestamp,
            decision_kind="video_branch_decision",
            outcome=decision,
            sequence_idx=idx,
            scope={"scope_kind": "episode", "hypothesis_id": hypothesis.hypothesis_id, "mode": hypothesis.mode},
            reasons=reasons or [decision],
            source_event_ids=[event.event_id],
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            artifact_refs={**dict(sidecar_refs or {}), "hypothesis_id": hypothesis.hypothesis_id},
            provenance={"component": "governed_video_supervision"},
            metadata={"expected_net_value": expected_net_value},
        )
        decisions.append(decision_entry)
        governance_traces.append(
            GovernanceTraceEntry.from_components(
                run_id=run_id,
                episode_id=runtime_record.episode_id,
                timestamp=timestamp,
                node_id=regal_report.node_id,
                outcome=decision,
                reason_codes=reasons or [decision],
                runtime_packet_id=runtime_packet.packet_id,
                contract_id=runtime_packet.contract.contract_id,
                source_event_ids=[event.event_id],
                decision_id=decision_entry.decision_id,
                evidence_refs={
                    "belief_state_id": getattr(belief_state, "belief_id", ""),
                    **dict(sidecar_refs or {}),
                },
                rule_refs=["regal_gen_plausibility"],
                artifact_refs={**dict(sidecar_refs or {}), "hypothesis_id": hypothesis.hypothesis_id},
                provenance={"component": "governed_video_supervision"},
                metadata={"regal_report": regal_report.to_dict()},
            )
        )

    counterfactual_eval = build_counterfactual_eval(
        run_id=run_id,
        episode_id=runtime_record.episode_id,
        timestamp=timestamp,
        runtime_packet_id=runtime_packet.packet_id,
        objective_profile_id=objective_profile.profile_id,
        baseline_value=base_value,
        branch_values=[
            {
                "label": evaluation.mode,
                "expected_net_value": evaluation.expected_net_value,
                "deltas": {"decision_positive": 1.0 if evaluation.decision == "render" else 0.0},
                "action": evaluation.action,
                "artifact_refs": {"hypothesis_id": evaluation.hypothesis_id},
                "metadata": {"decision": evaluation.decision},
            }
            for evaluation in branch_evaluations
        ],
        evidence_refs={"belief_state_id": getattr(belief_state, "belief_id", "")},
        artifact_refs=dict(sidecar_refs or {}),
    )

    recommended_value = max(
        (candidate.expected_net_value for candidate in counterfactual_eval.candidates),
        default=base_value,
    )
    value_target_pack = build_value_target_pack(
        run_id=run_id,
        episode_id=runtime_record.episode_id,
        runtime_packet_id=runtime_packet.packet_id,
        base_value=base_value,
        recommended_value=recommended_value,
        disagreement=float(snapshot.state_features.get("evidence_disagreement_mean", 0.0)),
        coverage=float(snapshot.state_features.get("evidence_coverage", 0.0)),
        counterfactual_eval_id=counterfactual_eval.eval_id,
        pricing_tick_ref=pricing_tick.tick_id,
        governance_trace_ref=f"governance:{sha256_json([trace.to_dict() for trace in governance_traces])[:12]}",
        metadata={"objective_preset": objective_preset},
    )

    from src.economics.value_ledger import ValueLedger  # local import to avoid cycles

    ledger_path = Path(value_ledger_path) if value_ledger_path is not None else Path("/tmp/governed_video_value_ledger.jsonl")
    receipt = ValueLedger(ledger_path).build_receipt(
        event_type="governed_video_episode",
        run_id=run_id,
        episode_id=runtime_record.episode_id,
        objective_profile_id=objective_profile.profile_id,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        pricing_tick=pricing_tick,
        constraint_set=constraint,
        regal_decision_summary={
            "overall_status": counterfactual_eval.recommended_action,
            "branch_count": len(branch_evaluations),
        },
        datapack_id=None,
        source_domain=str(runtime_record.source_domain),
        timestamp=timestamp,
    )
    return GovernedVideoSupervisionBundle(
        runtime_packet=runtime_packet,
        pricing_tick=pricing_tick,
        branch_evaluations=branch_evaluations,
        events=events,
        decisions=decisions,
        governance_traces=governance_traces,
        counterfactual_eval=counterfactual_eval,
        value_target_pack=value_target_pack,
        value_ledger_receipt=receipt,
    )


def _evaluate_branch_regality(
    hypothesis: GovernedVideoHypothesis,
    snapshot: VideoStateSnapshot,
    belief_state: Any,
    plausibility_node: RegalGenPlausibilityNode,
) -> RegalReport:
    disagreement = float(snapshot.state_features.get("evidence_disagreement_mean", 0.0))
    if hypothesis.mode == "semantic_disambiguation":
        disagreement *= 0.8
    elif hypothesis.mode == "throughput_push":
        disagreement *= 1.1
    context = {
        "map_first_quality_score": float(snapshot.state_features.get("geometry_quality", 0.0)),
        "semantic_disagreement_vla_vs_map": disagreement,
        "vla_evidence_coverage": float(snapshot.state_features.get("evidence_coverage", 0.0)),
    }
    return plausibility_node.evaluate(context)


__all__ = [
    "GovernedVideoSupervisionBundle",
    "VideoBranchEvaluation",
    "build_governed_video_supervision_bundle",
]
