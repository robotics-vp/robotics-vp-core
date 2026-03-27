import json
from argparse import Namespace
from datetime import datetime

import pytest

from scripts.train_orchestrator_shell_policy import _run_training
from src.semantic.models import EconSlice, MetaTransformerSlice, SemanticSnapshot
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal, SupervisionHints
from src.sima2.task_graph_proposals import RefinementType, TaskGraphRefinementProposal
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


pytest.importorskip("torch")


def _write_runtime_export(root) -> str:
    export_dir = root / "runtime_export"
    export_dir.mkdir(parents=True, exist_ok=True)
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
    snapshot = SemanticSnapshot(
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
            "execution_precondition_summary": {
                "report_count": 1,
                "ready_count": 1,
                "blocked_count": 0,
                "mean_readiness_score": 1.0,
            },
            "recap": {"mean_goodness": 0.7, "top_episodes": ["ep_orch"]},
        },
    )
    advisory = {
        "task_id": "task_orch",
        "focus_objective_presets": ["safety"],
        "sampler_strategy_overrides": {
            "balanced": 0.2,
            "frontier_prioritized": 0.5,
            "econ_urgency": 0.3,
        },
        "datapack_priority_tags": ["risk_triage", "precondition_ready"],
        "safety_emphasis": 0.85,
        "execution_mode": "preconditioned_routing",
        "policy_source": "heuristic_fallback",
        "activation_plan": {"mode": "preconditioned_routing"},
    }
    snapshot_path = export_dir / "ep_orch_semantic_snapshot_v1.json"
    advisory_path = export_dir / "ep_orch_orchestrator_advisory_v1.json"
    snapshot_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    advisory_path.write_text(json.dumps(advisory, indent=2), encoding="utf-8")
    rows_path = export_dir / "semantic_runtime_learning_rows.jsonl"
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
    return str(export_dir)


def test_train_orchestrator_shell_policy_emits_runtime_package(tmp_path) -> None:
    export_dir = _write_runtime_export(tmp_path)
    output_dir = tmp_path / "shell_policy_training"
    args = Namespace(
        runtime_export_dir=export_dir,
        rows_json=[],
        dataset_json=None,
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        output_dir=str(output_dir),
        run_name="orchestrator_shell_policy_test",
        seed=7,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="orchestrator_shell_policy_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    package = json.loads((output_dir / "orchestrator_shell_policy_package.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (output_dir / "orchestrator_shell_policy_dataset_summary.json").read_text(encoding="utf-8")
    )

    assert manifest["training_kind"] == "orchestrator_shell_policy"
    assert manifest["artifact_paths"]["orchestrator_shell_policy_runtime_package"].endswith(
        "orchestrator_shell_policy_package.json"
    )
    assert package["promotion_stage"] == "shadow_candidate"
    assert package["inference_contract"]["target_contract"] == "semantic_orchestrator_shell_policy_v1"
    assert summary["num_examples"] == 1
    assert holder["payload"]["benchmark_gate_ready"] is False
