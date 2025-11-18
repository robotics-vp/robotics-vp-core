#!/usr/bin/env python3
"""
Smoke test for RECAP → semantic aggregator → orchestrator integration.
"""
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.semantic.aggregator import SemanticAggregator
from src.semantic.models import SemanticSnapshot, EconSlice, MetaTransformerSlice
from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot, Episode, EconVector
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal, RefinementType
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal, SupervisionHints
from src.orchestrator.semantic_orchestrator_v2 import SemanticOrchestratorV2


def _build_store(root: Path) -> OntologyStore:
    store = OntologyStore(root_dir=str(root))
    task_id = "recap_sem"
    now = datetime.utcnow()
    store.upsert_task(Task(task_id=task_id, name="RecapSem", environment_id="env", human_mpl_units_per_hour=50.0, human_wage_per_hour=15.0, default_energy_cost_per_wh=0.1))
    store.upsert_robot(Robot(robot_id="r_recap", name="RecapBot"))
    for i in range(2):
        ep_id = f"ep_recap_sem_{i}"
        store.upsert_episode(Episode(episode_id=ep_id, task_id=task_id, robot_id="r_recap", started_at=now, status="success"))
        store.upsert_econ_vector(EconVector(episode_id=ep_id, mpl_units_per_hour=60 + i * 5, wage_parity=1.0, energy_cost=0.5, damage_cost=0.05, novelty_delta=0.1, reward_scalar_sum=5.0))
    return store


def main():
    root = Path("data/ontology/recap_sem_integration")
    if root.exists():
        import shutil
        shutil.rmtree(root)
    store = _build_store(root)
    agg = SemanticAggregator(store)

    now = datetime.utcnow()
    hints = SupervisionHints(prioritize_for_training=True, priority_level="high", suggested_weight_multiplier=1.0, suggested_replay_frequency="standard", requires_human_review=False, safety_critical=False, curriculum_stage="mid")
    sem_tag = SemanticEnrichmentProposal(
        proposal_id="tag_recap_sem",
        timestamp=now.timestamp(),
        video_id="vid",
        episode_id="ep_recap_sem_0",
        task="recap_sem",
        fragility_tags=[],
        risk_tags=[],
        affordance_tags=[],
        efficiency_tags=[],
        novelty_tags=[],
        intervention_tags=[],
        semantic_conflicts=[],
        coherence_score=0.7,
        supervision_hints=hints,
        confidence=0.9,
        source_proposals=[],
        validation_status="passed",
    )
    recap_scores = {
        "ep_recap_sem_0": {"episode_id": "ep_recap_sem_0", "recap_goodness_score": 2.0},
        "ep_recap_sem_1": {"episode_id": "ep_recap_sem_1", "recap_goodness_score": -0.5},
    }
    snapshot = agg.build_snapshot(
        task_id="recap_sem",
        stage2_ontology_proposals=[OntologyUpdateProposal(proposal_id="p_recap", proposal_type=ProposalType.ADD_AFFORDANCE)],
        stage2_task_refinements=[TaskGraphRefinementProposal(proposal_id="r_recap", refinement_type=RefinementType.SPLIT_TASK)],
        stage2_tags=[sem_tag],
        meta_outputs=None,
        recap_scores=recap_scores,
    )
    d = snapshot.to_dict()
    snapshot_rt = SemanticSnapshot.from_dict(d)
    assert snapshot_rt.metadata.get("recap", {}).get("mean_goodness", 0) > 0
    orch = SemanticOrchestratorV2(config={"write_to_file": False})
    advisory = orch.propose(snapshot_rt)
    assert "recap_top" in advisory.datapack_priority_tags
    adv2 = orch.propose(snapshot_rt)
    assert advisory.to_json() == adv2.to_json()
    print("[smoke_test_recap_semantic_integration] All tests passed.")


if __name__ == "__main__":
    main()
