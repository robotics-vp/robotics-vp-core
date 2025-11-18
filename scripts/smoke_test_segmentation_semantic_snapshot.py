#!/usr/bin/env python3
"""
Smoke test for SemanticSnapshot segmentation aggregation.
"""
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.semantic.aggregator import SemanticAggregator
from src.semantic.models import SemanticSnapshot
from src.ontology.store import OntologyStore
from src.sima2.tags.semantic_tags import (
    SemanticEnrichmentProposal,
    SegmentBoundaryTag,
    SubtaskTag,
    SupervisionHints,
)


def _build_proposal():
    hints = SupervisionHints(
        prioritize_for_training=True,
        priority_level="medium",
        suggested_weight_multiplier=1.0,
        suggested_replay_frequency="standard",
        requires_human_review=False,
        safety_critical=False,
        curriculum_stage="early",
    )
    boundaries = [
        SegmentBoundaryTag(episode_id="ep1", segment_id="ep1_seg0", timestep=0, reason="start", subtask_label="drawer"),
        SegmentBoundaryTag(episode_id="ep1", segment_id="ep1_seg0", timestep=5, reason="end", subtask_label="drawer"),
        SegmentBoundaryTag(episode_id="ep1", segment_id="ep1_seg1", timestep=6, reason="start", subtask_label="vase"),
        SegmentBoundaryTag(episode_id="ep1", segment_id="ep1_seg1", timestep=9, reason="end", subtask_label="vase"),
    ]
    subtasks = [
        SubtaskTag(episode_id="ep1", segment_id="ep1_seg0", subtask_label="drawer"),
        SubtaskTag(episode_id="ep1", segment_id="ep1_seg1", subtask_label="vase"),
    ]
    return SemanticEnrichmentProposal(
        proposal_id="prop1",
        timestamp=0.0,
        video_id="vid1",
        episode_id="ep1",
        task="drawer_vase",
        fragility_tags=[],
        risk_tags=[],
        affordance_tags=[],
        efficiency_tags=[],
        novelty_tags=[],
        intervention_tags=[],
        segment_boundary_tags=boundaries,
        subtask_tags=subtasks,
        semantic_conflicts=[],
        coherence_score=1.0,
        supervision_hints=hints,
        confidence=1.0,
        source_proposals=[],
        justification="",
        validation_status="passed",
        validation_errors=[],
    )


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OntologyStore(root_dir=tmpdir)
        aggregator = SemanticAggregator(store)
        proposal = _build_proposal()
        snapshot = aggregator.build_snapshot(
            task_id="drawer_vase",
            stage2_ontology_proposals=[],
            stage2_task_refinements=[],
            stage2_tags=[proposal],
            meta_outputs=None,
            recap_scores=None,
        )
        assert isinstance(snapshot, SemanticSnapshot)
        assert snapshot.num_segments == 2
        assert snapshot.segment_types.get("start") == 2
        assert snapshot.subtask_label_histogram.get("drawer") == 1
        as_dict = snapshot.to_dict()
        assert as_dict["segment_types"]["end"] == 2
        print("[smoke_test_segmentation_semantic_snapshot] PASS")


if __name__ == "__main__":
    main()
