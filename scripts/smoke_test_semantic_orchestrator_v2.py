#!/usr/bin/env python3
"""
Smoke test for SemanticOrchestratorV2 advisories.
"""
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.semantic.models import SemanticSnapshot, EconSlice, MetaTransformerSlice
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal, RefinementType
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal, SupervisionHints
from src.orchestrator.semantic_orchestrator_v2 import SemanticOrchestratorV2


def main():
    now = datetime.utcnow()
    hints = SupervisionHints(prioritize_for_training=True, priority_level="high", suggested_weight_multiplier=1.0, suggested_replay_frequency="standard", requires_human_review=False, safety_critical=True, curriculum_stage="mid")
    sem_tag = SemanticEnrichmentProposal(
        proposal_id="tag_orch",
        timestamp=now.timestamp(),
        video_id="vid1",
        episode_id="ep_orch",
        task="task_orch",
        fragility_tags=[],
        risk_tags=[],
        affordance_tags=[],
        efficiency_tags=[],
        novelty_tags=[],
        intervention_tags=[],
        semantic_conflicts=[],
        coherence_score=0.8,
        supervision_hints=hints,
        confidence=0.9,
        source_proposals=[],
        validation_status="passed",
    )
    snapshot = SemanticSnapshot(
        task_id="task_orch",
        ontology_proposals=[OntologyUpdateProposal(proposal_id="p1", proposal_type=ProposalType.ADD_AFFORDANCE)],
        task_refinements=[TaskGraphRefinementProposal(proposal_id="r1", refinement_type=RefinementType.SPLIT_TASK)],
        semantic_tags=[sem_tag],
        econ_slice=EconSlice(task_id="task_orch", avg_mpl_units_per_hour=80, avg_wage_parity=0.9, avg_energy_cost=0.4, avg_error_rate=0.05, frontier_episodes=["ep_orch"]),
        meta_slice=MetaTransformerSlice(task_id="task_orch", presets=["balanced"], expected_deltas={"mpl": 0.1}, backends=["pybullet"]),
        timestamp=now.timestamp(),
    )
    orch = SemanticOrchestratorV2(config={"write_to_file": False})
    advisory1 = orch.propose(snapshot)
    advisory2 = orch.propose(snapshot)
    assert advisory1.to_json() == advisory2.to_json(), "Advisory must be deterministic"
    assert advisory1.sampler_strategy_overrides.get("econ_urgency", 0) > 0
    assert advisory1.safety_emphasis >= 0.3
    print("[smoke_test_semantic_orchestrator_v2] All tests passed.")


if __name__ == "__main__":
    main()
