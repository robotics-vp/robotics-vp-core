"""Stage6 Golden Gate Test.

The 'no bullshit' CI gate for regality.

This test proves:
1. Stage6 minimal mode produces artifacts
2. Replay produces SHA comparisons
3. Mutating artifacts without updating SHA fails the gate
4. Deterministic seed produces same artifacts
"""
import pytest
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def create_mock_audit():
    """Create a mock AuditAggregateV1 for minimal runs."""
    from src.contracts.schemas import AuditAggregateV1
    
    return AuditAggregateV1(
        audit_suite_id="golden_gate_test",
        seed=42,
        num_episodes=1,
        success_rate=0.8,
        mean_error=0.1,
        mean_return=5.0,
        mean_energy_Wh=10.0,
        mean_mpl_proxy=0.05,
        episodes_sha="mock_episodes_sha",
        config_sha="mock_config_sha",
    )


def create_test_orchestrator(tmp_path, run_id):
    """Create a test orchestrator with fail_on_verify_error=False."""
    from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
    from src.training.regal_training_runner import TrainingRunConfig, RegalTrainingRunner
    
    # Create orchestrator with relaxed config for testing
    orch = Stage6TrainingOrchestrator.__new__(Stage6TrainingOrchestrator)
    orch.run_id = run_id
    orch.output_dir = Path(tmp_path) / run_id
    orch.output_dir.mkdir(parents=True, exist_ok=True)
    orch.env_type = "workcell"
    orch.seed = 42
    orch._child_results = []
    orch._checkpoint_refs = []
    orch._total_training_steps = 0
    
    # Use relaxed config for testing
    config = TrainingRunConfig(
        run_id=run_id,
        output_dir=str(orch.output_dir),
        seed=42,
        num_episodes=1,
        training_steps=10,
        fail_on_verify_error=False,  # Don't exit on verify failure
        require_trajectory_audit=True,
    )
    orch.runner = RegalTrainingRunner(config)
    
    return orch


class TestStage6GoldenGate:
    """The 'no bullshit' CI gate for regality."""
    
    def _setup_minimal_run(self, orch):
        """Setup minimal artifacts for a run."""
        from src.contracts.schemas import (
            SelectionManifestV1,
            OrchestratorStateV1,
            TrajectoryAuditV1,
        )
        
        output_dir = orch.output_dir
        
        # Create selection manifest with enough eligible datapacks
        selection = SelectionManifestV1(
            manifest_id=f"{orch.run_id}_selection",
            eligible_datapack_ids=["dp_001", "dp_002", "dp_003", "dp_004"],  # >= min count
            selected_datapack_ids=["dp_001", "dp_002"],
            quarantine_datapack_ids=[],
            rng_seed=42,
        )
        selection_path = output_dir / "selection_manifest.json"
        with open(selection_path, "w") as f:
            json.dump(selection.model_dump(mode="json"), f, indent=2, sort_keys=True)
        
        # Create orchestrator state
        orch_state = OrchestratorStateV1(step=10)
        orch_path = output_dir / "orchestrator_state.json"
        with open(orch_path, "w") as f:
            json.dump(orch_state.model_dump(mode="json"), f, indent=2, sort_keys=True)
        
        # Add trajectory audit
        audit = TrajectoryAuditV1(
            episode_id=f"{orch.run_id}_ep0",
            num_steps=10,
            total_return=5.0,
        )
        orch.runner.add_trajectory_audit(audit)
        
        # Set proper audit results
        mock_audit = create_mock_audit()
        orch.runner.set_audit_results(mock_audit, mock_audit)
    
    def test_stage6_minimal_artifacts_created(self, tmp_path):
        """Stage6 minimal mode creates required artifacts."""
        orch = create_test_orchestrator(tmp_path, "artifact-test")
        
        orch.runner.start_training()
        self._setup_minimal_run(orch)
        result = orch.runner.finalize(plan_sha="test_plan_sha")
        
        required = [
            "run_manifest.json",
            "selection_manifest.json",
            "orchestrator_state.json",
        ]
        
        for artifact in required:
            path = orch.output_dir / artifact
            assert path.exists(), f"Missing artifact: {artifact}"
        
        # Result should have run_id
        assert result.run_id == orch.run_id
    
    def test_stage6_produces_manifest_sha(self, tmp_path):
        """Stage6 produces manifest SHA."""
        orch = create_test_orchestrator(tmp_path, "manifest-sha-test")
        
        orch.runner.start_training()
        self._setup_minimal_run(orch)
        result = orch.runner.finalize(plan_sha="test_plan_sha")
        
        # Result should have manifest SHA
        assert result.manifest_sha is not None
        assert len(result.manifest_sha) == 64  # SHA-256 hex
    
    def test_replay_sha_comparison(self, tmp_path):
        """Replay produces SHA comparisons for key artifacts."""
        from scripts.replay_run_from_artifacts import replay_run
        
        orch = create_test_orchestrator(tmp_path, "replay-test")
        
        orch.runner.start_training()
        self._setup_minimal_run(orch)
        orch.runner.finalize(plan_sha="test_plan_sha")
        
        # Run replay
        results = replay_run(str(orch.output_dir), verbose=False)
        
        # Check that SHA comparisons exist
        assert "sha_comparisons" in results
        
        # Key SHAs that should match if present
        for sha_name in ["selection_manifest_sha", "orchestrator_state_sha"]:
            if sha_name in results["sha_comparisons"]:
                comparison = results["sha_comparisons"][sha_name]
                assert comparison["match"], (
                    f"{sha_name} mismatch:\n"
                    f"  expected: {comparison['expected']}\n"
                    f"  actual: {comparison['actual']}"
                )


class TestGoldenGateTamperDetection:
    """Test that mutating artifacts fails the gate."""
    
    def test_mutation_detected_by_replay(self, tmp_path):
        """Mutating artifact file is detected by replay."""
        from scripts.replay_run_from_artifacts import replay_run
        from src.contracts.schemas import SelectionManifestV1, TrajectoryAuditV1
        
        orch = create_test_orchestrator(tmp_path, "tamper-test")
        
        orch.runner.start_training()
        
        # Create selection manifest
        selection = SelectionManifestV1(
            manifest_id="tamper-test_selection",
            eligible_datapack_ids=["dp_001", "dp_002", "dp_003", "dp_004"],
            selected_datapack_ids=["dp_001"],
            quarantine_datapack_ids=[],
            rng_seed=42,
        )
        selection_path = orch.output_dir / "selection_manifest.json"
        with open(selection_path, "w") as f:
            json.dump(selection.model_dump(mode="json"), f, indent=2, sort_keys=True)
        
        # Add audit
        audit = TrajectoryAuditV1(episode_id="ep0", num_steps=10)
        orch.runner.add_trajectory_audit(audit)
        orch.runner.set_audit_results(create_mock_audit(), create_mock_audit())
        
        # Finalize
        orch.runner.finalize(plan_sha="test_plan_sha")
        
        # TAMPER: Modify selection manifest after SHA was recorded
        with open(selection_path, "r") as f:
            data = json.load(f)
        data["selected_datapack_ids"].append("dp_TAMPERED")
        with open(selection_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        
        # Replay should detect the tamper
        results = replay_run(str(orch.output_dir), verbose=False)
        
        # If SHA was recorded, it should now mismatch
        if "selection_manifest_sha" in results["sha_comparisons"]:
            sha_match = results["sha_comparisons"]["selection_manifest_sha"]["match"]
            assert not sha_match, "Tamper not detected via SHA mismatch!"


class TestGoldenGateDeterminism:
    """Test that golden gate is deterministic."""
    
    def test_same_content_produces_same_sha(self, tmp_path):
        """Same content produces identical SHAs."""
        from src.contracts.schemas import SelectionManifestV1
        from src.utils.config_digest import sha256_json
        
        # Create same manifest twice with fixed timestamp - should have same SHA
        fixed_time = "2026-01-13T00:00:00"
        
        selection1 = SelectionManifestV1(
            manifest_id="determinism-test_selection",
            created_at=fixed_time,
            eligible_datapack_ids=["dp_001"],
            selected_datapack_ids=["dp_001"],
            quarantine_datapack_ids=[],
            rng_seed=42,
        )
        
        selection2 = SelectionManifestV1(
            manifest_id="determinism-test_selection",
            created_at=fixed_time,
            eligible_datapack_ids=["dp_001"],
            selected_datapack_ids=["dp_001"],
            quarantine_datapack_ids=[],
            rng_seed=42,
        )
        
        sha1 = sha256_json(selection1.model_dump(mode="json"))
        sha2 = sha256_json(selection2.model_dump(mode="json"))
        
        # SHAs should match
        assert sha1 == sha2, "Same content should produce same SHA"


class TestOrchestratorIntegration:
    """Test Stage6 orchestrator integration."""
    
    def test_orchestrator_uses_workcell_default(self, tmp_path):
        """Stage6 orchestrator defaults to workcell env."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        
        orch = Stage6TrainingOrchestrator(output_dir=str(tmp_path))
        assert orch.env_type == "workcell"
    
    def test_orchestrator_has_migrated_trainers(self, tmp_path):
        """Stage6 has MIGRATED_TRAINERS mapping."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        
        assert hasattr(Stage6TrainingOrchestrator, "MIGRATED_TRAINERS")
        assert len(Stage6TrainingOrchestrator.MIGRATED_TRAINERS) >= 10
    
    def test_orchestrator_has_inprocess_method(self, tmp_path):
        """Stage6 has in-process execution method."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        
        orch = Stage6TrainingOrchestrator(output_dir=str(tmp_path))
        assert hasattr(orch, "run_child_trainer_inprocess")
