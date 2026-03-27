import json
from datetime import datetime

import pytest

from src.orchestrator.orchestrator_shell_policy import (
    build_shell_policy_feature_map,
    extract_orchestrator_advisory_target,
)
from src.orchestrator.orchestrator_shell_policy_runtime import (
    resolve_orchestrator_shell_policy_helper,
)
from src.orchestrator.orchestrator_shell_policy_training import (
    build_orchestrator_shell_training_dataset,
    train_orchestrator_shell_policy_model,
)
from src.orchestrator.semantic_orchestrator_v2 import SemanticOrchestratorV2
from src.semantic.models import EconSlice, MetaTransformerSlice, SemanticSnapshot
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal, SupervisionHints
from src.sima2.task_graph_proposals import RefinementType, TaskGraphRefinementProposal


pytest.importorskip("torch")


def _ready_execution_summary():
    return {
        "report_count": 1,
        "ready_count": 1,
        "blocked_count": 0,
        "mean_readiness_score": 1.0,
        "blocking_preconditions": {},
        "satisfied_preconditions": {
            "artifact::runtime_packet_ref": 1,
        },
    }


def _build_snapshot() -> SemanticSnapshot:
    now = datetime.utcnow()
    hints = SupervisionHints(
        prioritize_for_training=True,
        priority_level="high",
        suggested_weight_multiplier=1.0,
        suggested_replay_frequency="standard",
        requires_human_review=False,
        safety_critical=True,
        curriculum_stage="mid",
    )
    semantic_tag = SemanticEnrichmentProposal(
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
    return SemanticSnapshot(
        task_id="task_orch",
        ontology_proposals=[
            OntologyUpdateProposal(
                proposal_id="p1",
                proposal_type=ProposalType.ADD_AFFORDANCE,
            )
        ],
        task_refinements=[
            TaskGraphRefinementProposal(
                proposal_id="r1",
                refinement_type=RefinementType.SPLIT_TASK,
            )
        ],
        semantic_tags=[semantic_tag],
        econ_slice=EconSlice(
            task_id="task_orch",
            avg_mpl_units_per_hour=80,
            avg_wage_parity=0.9,
            avg_energy_cost=0.4,
            avg_error_rate=0.05,
            frontier_episodes=["ep_orch"],
        ),
        meta_slice=MetaTransformerSlice(
            task_id="task_orch",
            presets=["balanced", "safety"],
            expected_deltas={"mpl": 0.1, "error": -0.02, "energy": 0.03},
            backends=["pybullet"],
        ),
        timestamp=now.timestamp(),
        metadata={
            "execution_precondition_summary": _ready_execution_summary(),
            "recap": {"mean_goodness": 0.7, "top_episodes": ["ep_orch"]},
        },
    )


def _write_package(tmp_path) -> str:
    snapshot = _build_snapshot()
    advisory = {
        "task_id": snapshot.task_id,
        "focus_objective_presets": ["safety"],
        "sampler_strategy_overrides": {
            "balanced": 0.2,
            "frontier_prioritized": 0.5,
            "econ_urgency": 0.3,
        },
        "safety_emphasis": 0.85,
        "execution_mode": "preconditioned_routing",
        "policy_source": "heuristic_fallback",
        "activation_plan": {"mode": "preconditioned_routing"},
    }
    snapshot_path = tmp_path / "snapshot.json"
    advisory_path = tmp_path / "advisory.json"
    snapshot_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    advisory_path.write_text(json.dumps(advisory, indent=2), encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "sample_id": "semantic_runtime_test",
                "artifact_refs": {
                    "semantic_snapshot_path": str(snapshot_path),
                    "orchestrator_advisory_path": str(advisory_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = build_orchestrator_shell_training_dataset(
        [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    checkpoint_path = tmp_path / "orchestrator_shell_policy.pt"
    train_orchestrator_shell_policy_model(dataset, epochs=2, hidden_dim=16, save_path=str(checkpoint_path))
    package_path = tmp_path / "orchestrator_shell_policy_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "orchestrator_shell_policy_test",
                "checkpoint_path": checkpoint_path.name,
                "benchmark_gate": {"ready": False},
                "execution_preconditions": {"ready": True},
                "promotion_stage": "shadow_candidate",
                "inference_contract": {
                    "helper_blend_policy": {
                        "shadow_candidate_helper_weight": 0.15,
                        "promoted_helper_weight": 0.4,
                        "shadow_candidate_max_safety_delta": 0.1,
                        "promoted_max_safety_delta": 0.25,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(package_path)


def test_shell_policy_feature_map_extracts_runtime_state() -> None:
    snapshot = _build_snapshot()
    feature_map = build_shell_policy_feature_map(snapshot)

    assert feature_map["avg_wage_parity"] == pytest.approx(0.9)
    assert feature_map["mean_readiness_score"] == pytest.approx(1.0)
    assert feature_map["preset_safety_available"] == pytest.approx(1.0)


def test_extract_orchestrator_advisory_target_normalizes_fields() -> None:
    target = extract_orchestrator_advisory_target(
        {
            "focus_objective_presets": ["safety", "balanced"],
            "sampler_strategy_overrides": {"balanced": 1.0, "econ_urgency": 1.0},
            "safety_emphasis": 0.75,
            "execution_mode": "preconditioned_routing",
        }
    )

    assert target["activation_label"] == pytest.approx(1.0)
    assert target["preset_distribution"]["safety"] > 0.0
    assert target["sampler_strategy_overrides"]["balanced"] == pytest.approx(0.5)


def test_semantic_orchestrator_v2_applies_shell_helper(tmp_path) -> None:
    package_path = _write_package(tmp_path)
    orchestrator = SemanticOrchestratorV2(
        {
            "write_to_file": False,
            "shell_policy_helper_mode": "auto",
            "shell_policy_package_path": package_path,
        }
    )

    advisory = orchestrator.propose(_build_snapshot())

    assert advisory.policy_source == "heuristic_plus_learned_helper"
    assert advisory.promotion_stage == "shadow_candidate"
    assert advisory.helper_trace["package_id"] == "orchestrator_shell_policy_test"
    assert advisory.metadata["shell_policy_helper_mode"] == "auto"
    assert advisory.metadata["shell_policy_helper"]["helper_weight"] == pytest.approx(0.15)


def test_required_shell_helper_enforces_benchmark_gate(tmp_path) -> None:
    package_path = _write_package(tmp_path)

    with pytest.raises(ValueError, match="benchmark-gated"):
        resolve_orchestrator_shell_policy_helper(
            helper_mode="required",
            package_path=package_path,
        )
