"""Coverage loop orchestrator — closes the 9-step semantic evidence cycle.

This module implements the full loop described in Section I of the
semantic-WM synth-pipeline handoff:

1. Harvest evidence counts from replay / runtime learning store
2. Build coverage graph from skill graph + env inventories + evidence
3. Rank missing edges
4. Compile simulation agenda
5. Compile gap-driven diffusion prompts
6. Emit fill-path decisions (real sim / diffusion / synth branch / blocked)
7. Write artifacts
8. Return CoverageLoopResult

The loop is designed to be called:
- As **Step 7** of ``run_pipeline_step_with_causal_order`` (automatic)
- Standalone via ``scripts/run_coverage_loop.py`` (manual)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.world_model.coverage_evidence_harvester import (
    EvidenceHarvestResult,
    harvest_evidence_counts,
)
from src.world_model.semantic_coverage_graph import (
    SemanticCoverageGraph,
)
from src.world_model.fill_outcome_store import (
    FillOutcomeRecord,
    FillOutcomeStore,
)
from src.world_model.semantic_feedback_packets import (
    SemanticCoverageFeedback,
    compile_semantic_coverage_feedback,
)
from src.world_model.semantic_wm_correction import (
    apply_semantic_wm_correction_overlay,
    compile_semantic_wm_correction_overlay,
)
from src.world_model.graph_mutation_executor import (
    GovernedGraphMutationExecutor,
    GraphMutationExecutionResult,
)
from src.world_model.feedback_topology_adapters import (
    SemanticFeedbackAdapterPackage,
    shadow_fit_feedback_adapter_package,
)
from src.hrl.skill_graph import SkillGraph
from src.envs.primitive_inventory import for_env, list_registered_env_ids
from src.orchestrator.semantic_simulation import compile_simulation_agenda
from src.orchestrator.diffusion_requests import build_diffusion_prompt_from_coverage_gaps


# ---------------------------------------------------------------------------
# Fill-path decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FillPathDecision:
    """Recommended action for filling a specific coverage gap."""

    edge_key: str  # "src -> tgt"
    fill_method: str  # "real_sim" | "diffusion" | "synthetic_branch" | "blocked"
    confidence: float
    rationale: str
    coverage_gap_score: float
    economic_priority: float
    trust_priority: float
    readiness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_key": self.edge_key,
            "fill_method": self.fill_method,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "coverage_gap_score": self.coverage_gap_score,
            "economic_priority": self.economic_priority,
            "trust_priority": self.trust_priority,
            "readiness": self.readiness,
        }


def _decide_fill_path(
    gap: Any,
    *,
    trust_threshold: float = 0.3,
    readiness_threshold: float = 0.5,
) -> FillPathDecision:
    """Choose fill method for a single coverage gap edge."""
    edge_key = f"{gap.source_id} -> {gap.target_id}"
    econ = getattr(gap, "economic_priority", 0.0)
    trust = getattr(gap, "trust_priority", 0.0)
    readiness = getattr(gap, "promotion_readiness", 0.0)
    gap_score = gap.gap_score() if callable(getattr(gap, "gap_score", None)) else 0.0
    metadata = dict(getattr(gap, "metadata", {}) or {})

    # Decision logic
    if bool(metadata.get("governance_blocked", False)):
        method = "blocked"
        rationale = "Governance trace blocked this edge; keep as meta-node review target"
        confidence = 0.95
    elif readiness < readiness_threshold:
        method = "blocked"
        rationale = f"Readiness {readiness:.2f} < {readiness_threshold}: prerequisites not met"
        confidence = 0.3
    elif trust < trust_threshold:
        # Low trust → prefer real sim (higher-fidelity data)
        method = "real_sim"
        rationale = f"Trust {trust:.2f} < {trust_threshold}: real sim preferred for high-fidelity evidence"
        confidence = 0.7
    elif econ > 0.7:
        # High economic priority → diffusion (fast, cheap generation)
        method = "diffusion"
        rationale = f"Economic priority {econ:.2f} > 0.7: diffusion for fast gap filling"
        confidence = 0.8
    elif econ > 0.3:
        # Moderate priority → synthetic branch (balanced cost/fidelity)
        method = "synthetic_branch"
        rationale = f"Moderate economic priority {econ:.2f}: synthetic branching"
        confidence = 0.6
    else:
        # Low priority → diffusion (cheapest)
        method = "diffusion"
        rationale = f"Low economic priority {econ:.2f}: diffusion with low urgency"
        confidence = 0.5

    return FillPathDecision(
        edge_key=edge_key,
        fill_method=method,
        confidence=confidence,
        rationale=rationale,
        coverage_gap_score=gap_score,
        economic_priority=econ,
        trust_priority=trust,
        readiness=readiness,
    )


# ---------------------------------------------------------------------------
# CoverageLoopResult
# ---------------------------------------------------------------------------

@dataclass
class CoverageLoopResult:
    """Output of a single coverage loop execution."""

    coverage_graph: SemanticCoverageGraph
    coverage_summary: Dict[str, Any]
    evidence_harvest: EvidenceHarvestResult
    simulation_agenda: List[Dict[str, Any]]
    diffusion_prompts: List[Dict[str, Any]]
    fill_decisions: List[Dict[str, Any]]
    feedback_summary: Dict[str, Any] = field(default_factory=dict)
    wm_validation_summary: Dict[str, Any] = field(default_factory=dict)
    trust_calibration_overlay: Dict[str, float] = field(default_factory=dict)
    econ_calibration_overlay: Dict[str, float] = field(default_factory=dict)
    graph_mutation_proposals: List[Dict[str, Any]] = field(default_factory=list)
    graph_mutation_execution: Dict[str, Any] = field(default_factory=dict)
    semantic_wm_correction_overlay: Dict[str, Any] = field(default_factory=dict)
    corrected_semantic_world_model: Optional[Dict[str, Any]] = None
    pre_evidence_snapshot: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_graph": self.coverage_graph.to_dict(),
            "coverage_summary": self.coverage_summary,
            "evidence_harvest": self.evidence_harvest.to_dict(),
            "simulation_agenda": list(self.simulation_agenda),
            "diffusion_prompts": list(self.diffusion_prompts),
            "fill_decisions": list(self.fill_decisions),
            "feedback_summary": dict(self.feedback_summary),
            "wm_validation_summary": dict(self.wm_validation_summary),
            "trust_calibration_overlay": dict(self.trust_calibration_overlay),
            "econ_calibration_overlay": dict(self.econ_calibration_overlay),
            "graph_mutation_proposals": list(self.graph_mutation_proposals),
            "graph_mutation_execution": dict(self.graph_mutation_execution),
            "semantic_wm_correction_overlay": dict(self.semantic_wm_correction_overlay),
            "corrected_semantic_world_model": dict(self.corrected_semantic_world_model or {}),
            "metadata": dict(self.metadata),
        }

    def write_artifacts(self, artifact_dir: str) -> Dict[str, str]:
        """Write all result artifacts to disk. Returns paths written."""
        out = Path(artifact_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        cg_path = out / "coverage_graph.json"
        cg_path.write_text(json.dumps(self.coverage_graph.to_dict(), indent=2))
        paths["coverage_graph"] = str(cg_path)

        summary_path = out / "coverage_summary.json"
        summary_path.write_text(json.dumps(self.coverage_summary, indent=2))
        paths["coverage_summary"] = str(summary_path)

        agenda_path = out / "simulation_agenda_v1.json"
        agenda_path.write_text(json.dumps(self.simulation_agenda, indent=2))
        paths["simulation_agenda"] = str(agenda_path)

        diff_path = out / "diffusion_prompts.json"
        diff_path.write_text(json.dumps(self.diffusion_prompts, indent=2))
        paths["diffusion_prompts"] = str(diff_path)

        fill_path = out / "fill_decisions.json"
        fill_path.write_text(json.dumps(self.fill_decisions, indent=2))
        paths["fill_decisions"] = str(fill_path)

        feedback_path = out / "coverage_feedback_summary.json"
        feedback_path.write_text(json.dumps(self.feedback_summary, indent=2))
        paths["coverage_feedback_summary"] = str(feedback_path)

        wm_validation_path = out / "wm_validation_summary.json"
        wm_validation_path.write_text(json.dumps(self.wm_validation_summary, indent=2))
        paths["wm_validation_summary"] = str(wm_validation_path)

        trust_path = out / "trust_calibration_overlay.json"
        trust_path.write_text(json.dumps(self.trust_calibration_overlay, indent=2))
        paths["trust_calibration_overlay"] = str(trust_path)

        econ_path = out / "econ_calibration_overlay.json"
        econ_path.write_text(json.dumps(self.econ_calibration_overlay, indent=2))
        paths["econ_calibration_overlay"] = str(econ_path)

        mutation_path = out / "graph_mutation_proposals.json"
        mutation_path.write_text(json.dumps(self.graph_mutation_proposals, indent=2))
        paths["graph_mutation_proposals"] = str(mutation_path)

        mutation_exec_path = out / "graph_mutation_execution.json"
        mutation_exec_path.write_text(json.dumps(self.graph_mutation_execution, indent=2))
        paths["graph_mutation_execution"] = str(mutation_exec_path)

        correction_path = out / "semantic_wm_correction_overlay.json"
        correction_path.write_text(json.dumps(self.semantic_wm_correction_overlay, indent=2))
        paths["semantic_wm_correction_overlay"] = str(correction_path)

        corrected_wm_path = out / "corrected_semantic_world_model.json"
        corrected_wm_path.write_text(json.dumps(self.corrected_semantic_world_model or {}, indent=2))
        paths["corrected_semantic_world_model"] = str(corrected_wm_path)

        evidence_path = out / "evidence_harvest.json"
        evidence_path.write_text(json.dumps(self.evidence_harvest.to_dict(), indent=2))
        paths["evidence_harvest"] = str(evidence_path)

        return paths

    def record_outcomes(
        self,
        store: FillOutcomeStore,
        post_evidence_counts: Mapping[Tuple[str, str], int],
        quality_scores: Optional[Mapping[str, float]] = None,
        wall_times: Optional[Mapping[str, float]] = None,
    ) -> List[FillOutcomeRecord]:
        """Record fill outcomes by comparing pre/post evidence.

        Call this after fill actions have been executed to close the
        feedback loop.  The resulting records become training data
        for the learned gap ranker (Phase 2) and fill-path policy
        (Phase 3).
        """
        pre_ratio = self.coverage_summary.get("coverage_ratio", 0.0)
        post_counts = dict(post_evidence_counts)
        q_scores = dict(quality_scores or {})
        w_times = dict(wall_times or {})

        # Recompute post coverage ratio
        total = self.coverage_summary.get("total_edges", 1)
        post_covered = sum(
            1 for e in self.coverage_graph.edges
            if (e.evidence_count > 0 or post_counts.get((e.source_id, e.target_id), 0) > 0)
        )
        post_ratio = post_covered / max(total, 1)
        delta = post_ratio - pre_ratio

        records: List[FillOutcomeRecord] = []
        for decision in self.fill_decisions:
            edge_key = decision.get("edge_key", "")
            method = decision.get("fill_method", "")
            if method == "blocked":
                continue

            # Parse edge_key back to (src, tgt)
            parts = edge_key.split(" -> ", 1)
            src = parts[0].strip() if len(parts) == 2 else ""
            tgt = parts[1].strip() if len(parts) == 2 else ""

            pre_count = self.pre_evidence_snapshot.get(edge_key, 0)
            post_count = post_counts.get((src, tgt), pre_count)

            record = FillOutcomeRecord(
                edge_key=edge_key,
                fill_method=method,
                gap_features={
                    "coverage_gap_score": decision.get("coverage_gap_score", 0.0),
                    "economic_priority": decision.get("economic_priority", 0.0),
                    "trust_priority": decision.get("trust_priority", 0.0),
                    "readiness": decision.get("readiness", 0.0),
                },
                pre_evidence_count=pre_count,
                post_evidence_count=post_count,
                coverage_delta=delta,
                wall_time_s=w_times.get(edge_key, 1.0),
                quality_score=q_scores.get(edge_key, 1.0),
            )
            records.append(record)

        if records:
            store.append_batch(records)

        return records


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_coverage_loop(
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    trust_state: Optional[Mapping[str, Any]] = None,
    governance_traces: Optional[Sequence[Mapping[str, Any]]] = None,
    process_reward_summaries: Optional[Sequence[Mapping[str, Any]]] = None,
    fill_outcome_records: Optional[Sequence[Any]] = None,
    coverage_outcomes: Optional[Sequence[Mapping[str, Any]]] = None,
    wm_validation_packets: Optional[Sequence[Mapping[str, Any]]] = None,
    stage2_ontology_proposals: Optional[Sequence[Any]] = None,
    backend_health_reports: Optional[Sequence[Mapping[str, Any]]] = None,
    env_names: Optional[Sequence[str]] = None,
    hrl_skills: bool = True,
    sima_sequences: Optional[Sequence[Mapping[str, Any]]] = None,
    vla_hints: Optional[Sequence[Mapping[str, Any]]] = None,
    semantic_world_model: Optional[Any] = None,
    feedback_adapter_package: Optional[Any] = None,
    shadow_fit_feedback_adapter: bool = True,
    economic_weight: float = 1.0,
    trust_weight: float = 1.0,
    readiness_weight: float = 1.0,
    sim_agenda_limit: int = 10,
    diffusion_limit: int = 10,
    gap_ranker: Any = None,
    fill_path_policy: Any = None,
    write_artifacts: bool = False,
    artifact_dir: str = "data/coverage",
) -> CoverageLoopResult:
    """Execute one iteration of the coverage evidence loop.

    This is the 9-step cycle from Section I of the handoff:

    1. Harvest evidence counts from runtime rows
    2. Build skill graph
    3. Load env inventories
    4. Build coverage graph with real evidence
    5. Rank missing edges
    6. Compile simulation agenda
    7. Compile gap-driven diffusion prompts
    8. Compute fill-path decisions
    9. (Optional) write artifacts to disk
    """
    # Step 1: Harvest evidence
    harvest = harvest_evidence_counts(
        runtime_rows,
        econ_signals=econ_signals,
        trust_state=trust_state,
        governance_traces=governance_traces,
    )

    # Step 2: Build skill graph
    skill_graph = SkillGraph.build_from_registry(
        hrl_skills=hrl_skills,
        sima_sequences=list(sima_sequences or []),
        vla_hints=list(vla_hints or []),
    )

    # Step 3: Load env inventories
    resolved_envs = list(env_names or [])
    if not resolved_envs:
        resolved_envs = list(list_registered_env_ids())
    env_inventories = []
    for env_id in resolved_envs:
        try:
            env_inventories.append(for_env(env_id))
        except KeyError:
            pass

    # Step 3.5: Compile return-path feedback before graph construction
    feedback: SemanticCoverageFeedback = compile_semantic_coverage_feedback(
        coverage_outcomes=coverage_outcomes,
        wm_validation_packets=wm_validation_packets,
        fill_outcome_records=fill_outcome_records,
        process_reward_summaries=process_reward_summaries,
        governance_traces=governance_traces,
        stage2_ontology_proposals=stage2_ontology_proposals,
        econ_signals=econ_signals,
        trust_state=trust_state,
        backend_health_reports=backend_health_reports,
    )
    correction_overlay = compile_semantic_wm_correction_overlay(
        semantic_world_model,
        wm_validation_packets,
    )
    corrected_semantic_world_model = apply_semantic_wm_correction_overlay(
        semantic_world_model,
        correction_overlay,
    )

    mutation_executor = GovernedGraphMutationExecutor()
    mutation_result: GraphMutationExecutionResult = mutation_executor.execute(
        skill_graph,
        env_inventories,
        feedback.graph_mutation_proposals,
        governance_traces=governance_traces,
    )
    skill_graph = mutation_result.skill_graph
    env_inventories = mutation_result.env_inventories

    def _merge_edge_signals(
        base: Mapping[Tuple[str, str], float],
        overlay: Mapping[Tuple[str, str], float],
    ) -> Dict[Tuple[str, str], float]:
        merged: Dict[Tuple[str, str], float] = {}
        keys = set(dict(base).keys()) | set(dict(overlay).keys())
        for key in keys:
            base_value = float(dict(base).get(key, 0.0))
            overlay_value = float(dict(overlay).get(key, base_value))
            if key in overlay:
                merged[key] = max(0.0, min(1.0, 0.55 * base_value + 0.45 * overlay_value))
            else:
                merged[key] = base_value
        return merged

    # Step 4: Build coverage graph with real evidence counts
    coverage_graph = SemanticCoverageGraph.build(
        skill_graph=skill_graph,
        env_inventories=env_inventories,
        semantic_wm=corrected_semantic_world_model or semantic_world_model,
        economic_priorities=_merge_edge_signals(
            harvest.economic_priorities,
            feedback.edge_economic_overlay,
        ),
        trust_priorities=_merge_edge_signals(
            harvest.trust_priorities,
            feedback.edge_trust_overlay,
        ),
        readiness_signals=_merge_edge_signals(
            harvest.promotion_readiness,
            feedback.edge_readiness_overlay,
        ),
        evidence_counts=harvest.evidence_counts,
        edge_metadata=feedback.edge_metadata,
    )

    adapter: Optional[SemanticFeedbackAdapterPackage]
    adapter = feedback_adapter_package if isinstance(feedback_adapter_package, SemanticFeedbackAdapterPackage) else None
    if adapter is None and isinstance(feedback_adapter_package, Mapping) and "state_dict" in feedback_adapter_package:
        try:
            adapter = SemanticFeedbackAdapterPackage.from_checkpoint(feedback_adapter_package)
        except Exception:
            adapter = None
    if adapter is None and shadow_fit_feedback_adapter:
        adapter = shadow_fit_feedback_adapter_package(coverage_graph)
    if adapter is not None:
        predictions = adapter.predict_edges(coverage_graph.edges)
        for edge, prediction in zip(coverage_graph.edges, predictions):
            edge.economic_priority = float(max(edge.economic_priority, prediction.get("economic_priority", edge.economic_priority)))
            edge.trust_priority = float(max(edge.trust_priority, prediction.get("trust_priority", edge.trust_priority)))
            if not bool(edge.metadata.get("governance_blocked", False)):
                edge.promotion_readiness = float(max(edge.promotion_readiness, prediction.get("promotion_readiness", edge.promotion_readiness)))
            edge.metadata["wm_validation_pressure"] = float(
                max(
                    float(edge.metadata.get("wm_validation_pressure", 0.0)),
                    prediction.get("wm_correction_pressure", 0.0),
                )
            )

    # Step 5: Rank missing edges (implicit in rank_gaps)
    summary = coverage_graph.coverage_summary()
    summary["feedback_loop"] = dict(feedback.feedback_summary)
    summary["wm_validation_summary"] = dict(feedback.wm_validation_summary)
    summary["trust_calibration_overlay"] = dict(feedback.trust_calibration_overlay)
    summary["econ_calibration_overlay"] = dict(feedback.econ_calibration_overlay)
    total_edges = max(summary.get("total_edges", 0), 1)
    summary["feedback_loop"]["missing_edge_fraction"] = summary.get("missing_edges", 0) / float(total_edges)
    summary["feedback_loop"]["governance_blocked_fraction"] = summary.get("governance_blocked_edges", 0) / float(total_edges)
    summary["feedback_loop"]["graph_mutation_pressure"] = float(len(feedback.graph_mutation_proposals)) / float(total_edges)
    summary["feedback_loop"]["graph_mutation_applied_count"] = int(mutation_result.metadata.get("applied_count", 0))
    summary["feedback_loop"]["graph_mutation_blocked_count"] = int(mutation_result.metadata.get("blocked_count", 0))
    summary["feedback_loop"]["wm_correction_pressure"] = float(correction_overlay.meta_node_pressure)

    # Step 6: Compile simulation agenda
    sim_agenda = compile_simulation_agenda(
        coverage_graph,
        economic_weight=economic_weight,
        trust_weight=trust_weight,
        readiness_weight=readiness_weight,
        limit=sim_agenda_limit,
    )

    # Step 7: Compile gap-driven diffusion prompts
    gap_prompts = build_diffusion_prompt_from_coverage_gaps(
        coverage_graph,
        limit=diffusion_limit,
    )
    diffusion_dicts = [p.to_dict() for p in gap_prompts]

    # Step 8: Compute fill-path decisions
    ranked_gaps = coverage_graph.rank_gaps(
        economic_weight=economic_weight,
        trust_weight=trust_weight,
        readiness_weight=readiness_weight,
        limit=sim_agenda_limit + diffusion_limit,
        gap_ranker=gap_ranker,
    )
    summary.setdefault(
        "top_missing_edges",
        [f"{gap.source_id} -> {gap.target_id}" for gap in ranked_gaps[:6]],
    )

    if fill_path_policy is not None:
        # Use learned fill-path policy
        try:
            predictions = fill_path_policy.predict_batch(ranked_gaps, coverage_graph)
            fill_decisions = []
            for gap, (method, confidence) in zip(ranked_gaps, predictions):
                edge_key = f"{gap.source_id} -> {gap.target_id}"
                econ = getattr(gap, "economic_priority", 0.0)
                trust = getattr(gap, "trust_priority", 0.0)
                readiness_val = getattr(gap, "promotion_readiness", 0.0)
                gap_score_val = gap.gap_score() if callable(getattr(gap, "gap_score", None)) else 0.0
                fill_decisions.append(FillPathDecision(
                    edge_key=edge_key,
                    fill_method=method,
                    confidence=confidence,
                    rationale=f"Learned policy: {method} (confidence={confidence:.2f})",
                    coverage_gap_score=gap_score_val,
                    economic_priority=econ,
                    trust_priority=trust,
                    readiness=readiness_val,
                ).to_dict())
        except Exception:
            # Fall back to heuristic
            fill_decisions = [_decide_fill_path(gap).to_dict() for gap in ranked_gaps]
    else:
        fill_decisions = [_decide_fill_path(gap).to_dict() for gap in ranked_gaps]

    # Snapshot pre-evidence for outcome tracking
    pre_evidence = {
        f"{g.source_id} -> {g.target_id}": g.evidence_count
        for g in ranked_gaps
    }

    result = CoverageLoopResult(
        coverage_graph=coverage_graph,
        coverage_summary=summary,
        evidence_harvest=harvest,
        simulation_agenda=sim_agenda,
        diffusion_prompts=diffusion_dicts,
        fill_decisions=fill_decisions,
        feedback_summary=feedback.feedback_summary,
        wm_validation_summary=feedback.wm_validation_summary,
        trust_calibration_overlay=feedback.trust_calibration_overlay,
        econ_calibration_overlay=feedback.econ_calibration_overlay,
        graph_mutation_proposals=[item.to_dict() for item in feedback.graph_mutation_proposals],
        graph_mutation_execution=mutation_result.to_dict(),
        semantic_wm_correction_overlay=correction_overlay.to_dict(),
        corrected_semantic_world_model=(
            corrected_semantic_world_model.to_dict()
            if getattr(corrected_semantic_world_model, "to_dict", None) is not None
            else None
        ),
        pre_evidence_snapshot=pre_evidence,
        metadata={
            "env_names": resolved_envs,
            "hrl_skills": hrl_skills,
            "economic_weight": economic_weight,
            "trust_weight": trust_weight,
            "readiness_weight": readiness_weight,
            "semantic_world_model_present": semantic_world_model is not None,
            "feedback_adapter_applied": adapter is not None,
        },
    )

    # Step 9: Write artifacts
    if write_artifacts:
        result.write_artifacts(artifact_dir)

    return result


__all__ = [
    "CoverageLoopResult",
    "FillPathDecision",
    "run_coverage_loop",
]
