"""End-to-end smoke test for regality pipeline.

Exercises the full pipeline components that exist:
- RegalTrainingRunner instantiation
- Decorator and wrapper verification
- Trainer importability
- Verifier checks

This is NOT a heavy RL training test - just validates critical components.
"""
import pytest
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestRegalityEndToEndSmoke:
    """End-to-end smoke test for regality pipeline."""
    
    def test_runner_can_be_instantiated(self):
        """RegalTrainingRunner can be instantiated with config."""
        from src.training.regal_training_runner import RegalTrainingRunner, TrainingRunConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingRunConfig(
                output_dir=tmpdir,
                num_episodes=1,
                training_steps=5,
                seed=42,
                enable_regal=True,
            )
            
            runner = RegalTrainingRunner(config)
            assert runner.run_id is not None
            assert runner.output_dir.exists()
    
    def test_runner_start_training(self):
        """Runner can start training."""
        from src.training.regal_training_runner import RegalTrainingRunner, TrainingRunConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingRunConfig(
                output_dir=tmpdir,
                num_episodes=1,
                training_steps=10,
            )
            
            runner = RegalTrainingRunner(config)
            runner.start_training()
            
            # Just test that it doesn't throw
            assert True
    
    def test_runner_update_step(self):
        """Runner can track training steps."""
        from src.training.regal_training_runner import RegalTrainingRunner, TrainingRunConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingRunConfig(
                output_dir=tmpdir,
                num_episodes=1,
                training_steps=10,
            )
            
            runner = RegalTrainingRunner(config)
            runner.start_training()
            
            # Simulate training steps  
            for step in range(1, 11):
                runner.update_step(step)
            
            # Just test that it doesn't throw
            assert True
    
    def test_verify_run_importable(self):
        """verify_run can be imported and called on empty dir."""
        from src.valuation.valuation_verifier import verify_run
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report = verify_run(tmpdir)
            assert report is not None
            assert not report.all_passed  # Empty dir should fail
    
    def test_exposure_manifest_module_exists(self):
        """ExposureManifest module can be imported."""
        from src.valuation.exposure_manifest import (
            ExposureManifestV1,
        )
        
        # Just check import works
        assert ExposureManifestV1 is not None
    
    def test_run_manifest_module_exists(self):
        """RunManifest module can be imported."""
        from src.valuation.run_manifest import (
            RunManifestV1,
            create_run_manifest,
        )
        
        # Test creation
        manifest = create_run_manifest(
            run_id="smoke-test-run",
            plan_sha="abc123",
            audit_suite_id="smoke_suite",
            audit_seed=42,
            audit_config_sha="def456",
            datapack_ids=["dp_001"],
            seeds={"audit": 42},
            determinism_config={"torch": True, "numpy": True},
            baseline_weights_sha=None,
            final_weights_sha=None,
        )
        
        assert manifest.run_id == "smoke-test-run"
        assert manifest.plan_sha == "abc123"


class TestWrappedTrainersImportable:
    """Test that wrapped trainers are importable."""
    
    WRAPPED_TRAINERS = [
        "train_hydra_policy",
        "train_sac_with_ontology_logging",
        "train_offline_policy",
        "train_skill_policies",
        "train_behaviour_model",
        "train_world_model_from_datapacks",
        "train_stable_world_model",
        "train_trust_aware_world_model",
        "train_horizon_agnostic_world_model",
    ]
    
    @pytest.mark.parametrize("trainer_name", WRAPPED_TRAINERS)
    def test_trainer_importable(self, trainer_name):
        """Wrapped trainer can be imported."""
        import importlib
        
        try:
            module = importlib.import_module(f"scripts.{trainer_name}")
            assert hasattr(module, "main"), f"{trainer_name} must have main()"
        except ImportError as e:
            # Some trainers may have optional dependencies
            pytest.skip(f"Import skipped: {e}")
    
    @pytest.mark.parametrize("trainer_name", WRAPPED_TRAINERS)
    def test_trainer_has_regality_decorator(self, trainer_name):
        """Wrapped trainer uses @regal_training decorator."""
        scripts_dir = ROOT / "scripts"
        script_path = scripts_dir / f"{trainer_name}.py"
        
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        content = script_path.read_text()
        has_decorator = "@regal_training" in content
        
        assert has_decorator, f"{trainer_name} must use @regal_training"
    
    @pytest.mark.parametrize("trainer_name", WRAPPED_TRAINERS)
    def test_trainer_specifies_workcell(self, trainer_name):
        """Wrapped trainer specifies env_type='workcell'."""
        scripts_dir = ROOT / "scripts"
        script_path = scripts_dir / f"{trainer_name}.py"
        
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        content = script_path.read_text()
        has_workcell = 'env_type="workcell"' in content or "env_type='workcell'" in content
        
        assert has_workcell, f"{trainer_name} must specify env_type='workcell'"


class TestBlockingCheckIDsComplete:
    """Test BLOCKING_CHECK_IDS completeness."""
    
    def test_has_structural_checks(self):
        """Must include structural SHA checks."""
        from src.valuation.valuation_verifier import BLOCKING_CHECK_IDS
        
        structural = [
            "orchestrator_state_sha_match",
            "selection_manifest_sha_match",
        ]
        
        for check in structural:
            assert check in BLOCKING_CHECK_IDS, f"Missing: {check}"
    
    def test_has_policy_checks(self):
        """Must include Phase 10 policy checks."""
        from src.valuation.valuation_verifier import BLOCKING_CHECK_IDS
        
        policy = [
            "regality_thresholds_sha_match",
            "unknown_blocking_override_check_id",
        ]
        
        for check in policy:
            assert check in BLOCKING_CHECK_IDS, f"Missing: {check}"


class TestP0MigrationComplete:
    """Verify P0 migration is complete."""
    
    def test_no_p0_in_backlog(self):
        """P0 count must be zero."""
        import json
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        p0_count = sum(1 for item in data["backlog"] if item["priority"] == "P0")
        
        assert p0_count == 0, f"P0 count is {p0_count}, must be 0"
    
    def test_migrated_section_has_p0_scripts(self):
        """Migrated section should track P0 completions."""
        import json
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        migrated = data.get("migrated", [])
        migrated_scripts = [m["script"] for m in migrated]
        
        expected_p0 = [
            "train_behaviour_model.py",
            "train_world_model_from_datapacks.py",
            "train_stable_world_model.py",
            "train_trust_aware_world_model.py",
            "train_horizon_agnostic_world_model.py",
        ]
        
        for script in expected_p0:
            assert script in migrated_scripts, f"Missing from migrated: {script}"
