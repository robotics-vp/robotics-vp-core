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

    # Decision logic
    if readiness < readiness_threshold:
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
    env_names: Optional[Sequence[str]] = None,
    hrl_skills: bool = True,
    sima_sequences: Optional[Sequence[Mapping[str, Any]]] = None,
    vla_hints: Optional[Sequence[Mapping[str, Any]]] = None,
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

    # Step 4: Build coverage graph with real evidence counts
    coverage_graph = SemanticCoverageGraph.build(
        skill_graph=skill_graph,
        env_inventories=env_inventories,
        evidence_counts=harvest.evidence_counts,
    )

    # Step 5: Rank missing edges (implicit in rank_gaps)
    summary = coverage_graph.coverage_summary()

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
        pre_evidence_snapshot=pre_evidence,
        metadata={
            "env_names": resolved_envs,
            "hrl_skills": hrl_skills,
            "economic_weight": economic_weight,
            "trust_weight": trust_weight,
            "readiness_weight": readiness_weight,
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
