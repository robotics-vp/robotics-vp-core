#!/usr/bin/env python3
"""
Smoke test for SemanticAggregator building SemanticSnapshot.
"""
import shutil
import sys
from pathlib import Path
from datetime import datetime

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.semantic.aggregator import SemanticAggregator
from src.semantic.models import SemanticSnapshot
from src.ontology.models import Task, Robot, Episode, EconVector, Datapack
from src.ontology.store import OntologyStore
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal, RefinementType, RefinementPriority
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal, SupervisionHints
from src.orchestrator.meta_transformer import MetaTransformerOutputs


def main():
    root = Path("data/ontology/test_semantic_agg")
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))
    task = Task(task_id="task_sem", name="SemTask", environment_id="env", human_mpl_units_per_hour=60.0, human_wage_per_hour=18.0, default_energy_cost_per_wh=0.12)
    robot = Robot(robot_id="robot_sem", name="SemBot")
    store.upsert_task(task)
    store.upsert_robot(robot)
    now = datetime.utcnow()
    store.append_datapacks([Datapack(datapack_id="dp_sem", source_type="human_video", task_id="task_sem", modality="video", storage_uri="/tmp/dp", created_at=now)])
    store.upsert_episode(Episode(episode_id="ep_sem", task_id="task_sem", robot_id="robot_sem", started_at=now, status="success"))
    store.upsert_econ_vector(EconVector(episode_id="ep_sem", mpl_units_per_hour=90, wage_parity=1.1, energy_cost=0.5, damage_cost=0.1, novelty_delta=0.2, reward_scalar_sum=12.0))

    proposals = [OntologyUpdateProposal(proposal_id="p1", proposal_type=ProposalType.ADD_AFFORDANCE, priority=ProposalPriority.HIGH)]
    refinements = [TaskGraphRefinementProposal(proposal_id="r1", refinement_type=RefinementType.SPLIT_TASK, priority=RefinementPriority.MEDIUM)]
    hints = SupervisionHints(prioritize_for_training=True, priority_level="high", suggested_weight_multiplier=1.0, suggested_replay_frequency="standard", requires_human_review=False, safety_critical=True, curriculum_stage="mid")
    sem_tag = SemanticEnrichmentProposal(
        proposal_id="tag1",
        timestamp=now.timestamp(),
        video_id="vid1",
        episode_id="ep_sem",
        task="task_sem",
        fragility_tags=[],
        risk_tags=[],
        affordance_tags=[],
        efficiency_tags=[],
        novelty_tags=[],
        intervention_tags=[],
        semantic_conflicts=[],
        coherence_score=0.9,
        supervision_hints=hints,
        confidence=0.9,
        source_proposals=[],
        validation_status="passed",
    )
    meta_out = MetaTransformerOutputs(objective_preset="balanced", data_mix_weights={}, energy_profile_weights={}, chosen_backend="pybullet")

    agg = SemanticAggregator(store)
    snapshot = agg.build_snapshot(
        task_id="task_sem",
        stage2_ontology_proposals=proposals,
        stage2_task_refinements=refinements,
        stage2_tags=[sem_tag],
        meta_outputs=meta_out,
    )

    d = snapshot.to_dict()
    snapshot2 = SemanticSnapshot.from_dict(d)
    assert snapshot2.to_dict() == snapshot.to_dict(), "Round-trip must be deterministic"
    assert snapshot2.econ_slice.avg_mpl_units_per_hour > 0
    assert snapshot2.meta_slice.presets
    assert snapshot2.semantic_tags
    print("[smoke_test_semantic_aggregator] All tests passed.")


if __name__ == "__main__":
    main()
