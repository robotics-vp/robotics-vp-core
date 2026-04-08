from __future__ import annotations

import numpy as np
import pytest
import torch

from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics.agenda import SimulationJobSpec
from src.world_model.sim_synth_physics.gen2sim_admission import (
    assess_local_branch_corpus_gen2sim,
    build_gen2sim_admission_receipt,
    compile_gen2sim_admission_state,
)
from src.world_model.sim_synth_physics.state import SyntheticBranchPlan
from src.world_model.sim_synth_physics.synthetic_branches import (
    build_synthetic_branch_corpus_metadata,
    collect_local_synthetic_branch_records,
    compute_branch_gap_labels,
)


def _make_graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("skill:grasp_handle", "skill", "grasp_handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "locate_handle"),
        ],
        edges=[
            CoverageEdge(
                "skill:grasp_handle",
                "prim:locate_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.7,
                trust_priority=0.4,
                promotion_readiness=0.3,
            )
        ],
    )


class _DummyWorldModel:
    def rollout(self, z_init, actions_segment):
        z_dim = int(z_init.shape[-1])
        steps = int(actions_segment.shape[0]) + 1
        base = z_init.unsqueeze(0).repeat(steps, 1)
        deltas = torch.linspace(0.0, 0.2, steps, device=z_init.device).unsqueeze(1).repeat(1, z_dim)
        return base + deltas


class _DummyTrustNet:
    def __call__(self, _features):
        return torch.tensor([0.96], dtype=torch.float32)


def test_compute_branch_gap_labels_uses_coverage_graph() -> None:
    labels = compute_branch_gap_labels(_make_graph(), task_id="drawer_vase")

    assert labels["skill_edge"] == "skill:grasp_handle -> prim:locate_handle"
    assert labels["coverage_gap_contribution"] > 0.0
    assert labels["economic_priority"] == 0.7


def test_collect_local_synthetic_branch_records_returns_branch_metadata() -> None:
    np.random.seed(0)
    episodes = [
        {
            "z_sequence": torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=torch.float32),
            "actions": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
            "length": 2,
        },
        {
            "z_sequence": torch.tensor([[0.2, 0.1], [0.3, 0.2], [0.4, 0.3], [0.5, 0.4]], dtype=torch.float32),
            "actions": torch.tensor([[0.2], [0.1], [0.05]], dtype=torch.float32),
            "length": 3,
        },
    ]

    branches, stats = collect_local_synthetic_branch_records(
        episodes=episodes,
        world_model=_DummyWorldModel(),
        trust_net=_DummyTrustNet(),
        trust_mean=torch.zeros(6),
        trust_std_norm=torch.ones(6),
        real_z_std=0.1,
        horizon=2,
        branches_per_episode=1,
        min_trust=0.8,
        min_std_ratio=0.0,
        max_std_ratio=10.0,
        objective_vector=[1.0, 0.0, 0.0, 1.0],
        coverage_graph=_make_graph(),
        brick_manifest=None,
        brick_id_fn=lambda ep_idx, _manifest: ep_idx,
    )

    assert len(branches) == 1
    assert branches[0]["trust_score"] == pytest.approx(0.96)
    assert branches[0]["gap_labels"]["skill_edge"]
    assert stats["passed_all"] == 1


def test_compile_gen2sim_admission_state_blocks_when_grounding_missing() -> None:
    job = SimulationJobSpec(
        job_id="job_1",
        rank=1,
        task_family="drawer_vase",
        env_backend="isaac",
        skill_edge="a -> b",
        risk_family="collision",
        object_family="",
        objective_preset="balanced",
        data_collection_intent="validate",
        coverage_gap_score=0.8,
        economic_priority=0.9,
        trust_priority=0.4,
        readiness=0.9,
        ranking_policy="heuristic_only",
        rationale="test",
        coverage_targets={},
        expected_receipts=[],
        inferential_learnability_contract={
            "subject_id": "job_1",
            "subject_kind": "sim_synth_job",
            "inferential_replay_weight": 0.3,
            "signal_yield": {"score": 0.8},
        },
    )
    plan = SyntheticBranchPlan(
        plan_id="plan_1",
        source_job_id="job_1",
        branch_family="drawer_vase:validate",
        generation_mode="physics_probe",
        render_backend="isaac",
        admission_preconditions={
            "requires_non_heuristic_grounding": True,
            "requires_benchmark_ready": True,
            "min_readiness": 0.0,
            "min_inferential_replay_weight": 0.1,
        },
        expected_yield_score=0.7,
        inferential_learnability_contract={
            "subject_id": "plan_1",
            "subject_kind": "synthetic_branch_plan",
            "inferential_replay_weight": 0.3,
            "signal_yield": {"score": 0.8},
        },
    )

    admission = compile_gen2sim_admission_state(
        [plan],
        [job],
        benchmark_signals={"ready": False, "semantic_grounding_non_heuristic": False},
    )
    receipt = build_gen2sim_admission_receipt(admission, [plan], [job])

    assert admission.admissible_branch_ids == []
    assert admission.blocked_branch_ids == ["plan_1"]
    assert receipt.admission_id == admission.admission_id
    assert receipt.metadata["helper_promotion_stage_counts"]["heuristic_fallback"] == 1
    assert receipt.metadata["synthetic_evidence_counts"]["blocked_synthetic_branch_count"] == 1


def test_assess_local_branch_corpus_gen2sim_and_metadata_summary() -> None:
    branches = [
        {
            "trust_score": 0.95,
            "std_ratio": 1.0,
            "branch_value": 0.8,
            "gap_labels": {"coverage_gap_contribution": 0.5, "economic_priority": 0.4},
        }
    ]
    rows, summary = assess_local_branch_corpus_gen2sim(
        branches,
        corpus_name="demo_branches",
        source_runtime_metadata={},
        scene_tracks_backend="real",
        teacher_runtime_backend_selected="real",
        vision_backbone_selected="real",
        semantic_grounding_mode="non_heuristic",
        semantic_memory_grounded=True,
        gap_labels_path="gap_labels.json",
    )

    metadata = build_synthetic_branch_corpus_metadata(
        output_path="branches.npz",
        world_model_path="wm.pt",
        dataset_path="rollouts.npz",
        horizon=5,
        branches_per_episode=2,
        objective_dim=4,
        min_trust=0.9,
        min_std_ratio=0.8,
        max_std_ratio=1.2,
        stats={"total_attempted": 2, "passed_trust": 1, "passed_std": 1, "by_brick": {}},
        branches=branches,
        coverage_graph_used=True,
        coverage_graph_path="coverage_graph.json",
        source_runtime_metadata={},
        source_runtime_metadata_artifact="seed_runtime_metadata.json",
        scene_tracks_backend="real",
        teacher_runtime_backend_selected="real",
        vision_backbone_selected="real",
        semantic_grounding_mode="non_heuristic",
        semantic_memory_grounded=True,
        gap_labels_path="gap_labels.json",
        gen2sim_validity_path="gen2sim.json",
        gen2sim_summary=summary,
    )

    assert rows[0]["branch_idx"] == 0
    assert summary["count"] == 1
    assert metadata["gen2sim_validity_summary"]["count"] == 1
    assert metadata["future_training_signals"]["semantic_grounding_non_heuristic"] is True
    assert metadata["future_training_artifacts"]["source_runtime_metadata"] == "seed_runtime_metadata.json"
