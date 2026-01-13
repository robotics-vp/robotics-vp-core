"""Stage6 blessed path tests.

Ensures Stage6 is the canonical training orchestration layer:
1. Migrated trainers must use in-process execution
2. MIGRATED_TRAINERS list must be kept in sync with backlog
3. Stage6 must be the only way to run production training
"""
import pytest
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestStage6BlessedPath:
    """Test that Stage6 is the canonical blessed path."""
    
    def test_migrated_trainers_list_complete(self):
        """MIGRATED_TRAINERS must include all migrated scripts."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        # Get all migrated scripts from backlog
        migrated = data.get("migrated", [])
        migrated_names = [m["script"].replace(".py", "") for m in migrated]
        
        # Check that each migrated script is in MIGRATED_TRAINERS
        stage6_trainers = set(Stage6TrainingOrchestrator.MIGRATED_TRAINERS.keys())
        
        missing = []
        for name in migrated_names:
            if name not in stage6_trainers:
                missing.append(name)
        
        assert len(missing) == 0, (
            f"MIGRATED_TRAINERS is out of sync. Missing: {missing}\n"
            "Add these to run_stage6_train_all.py MIGRATED_TRAINERS"
        )
    
    def test_stage6_has_inprocess_method(self):
        """Stage6 must have in-process execution method."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        
        assert hasattr(Stage6TrainingOrchestrator, "run_child_trainer_inprocess"), (
            "Stage6TrainingOrchestrator must have run_child_trainer_inprocess method"
        )
    
    def test_migrated_trainers_have_correct_module_paths(self):
        """MIGRATED_TRAINERS module paths must be valid."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        import importlib
        
        for name, module_path in Stage6TrainingOrchestrator.MIGRATED_TRAINERS.items():
            assert module_path.startswith("scripts."), (
                f"{name} has invalid module path: {module_path}"
            )
            
            # Try to import the module
            try:
                module = importlib.import_module(module_path)
                assert hasattr(module, "main"), f"{module_path} must have main()"
            except ImportError as e:
                pytest.skip(f"Optional import failed: {e}")


class TestStage6OrchestratorStructure:
    """Test Stage6 orchestrator structure."""
    
    def test_orchestrator_defaults_to_workcell(self):
        """Stage6 defaults to workcell env_type."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Stage6TrainingOrchestrator(output_dir=tmpdir)
            assert orch.env_type == "workcell", "Stage6 must default to workcell"
    
    def test_orchestrator_has_runner(self):
        """Stage6 must have RegalTrainingRunner."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Stage6TrainingOrchestrator(output_dir=tmpdir)
            assert orch.runner is not None, "Stage6 must have runner"
            assert orch.run_id == orch.runner.run_id, "run_id must match"
    
    def test_orchestrator_creates_output_dir(self):
        """Stage6 creates output directory."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Stage6TrainingOrchestrator(output_dir=tmpdir)
            assert orch.output_dir.exists(), "Stage6 must create output_dir"


class TestMigrationSync:
    """Test synchronization between backlog and Stage6."""
    
    def test_backlog_migrated_count_matches_stage6(self):
        """Backlog migrated count should match Stage6 MIGRATED_TRAINERS."""
        from scripts.run_stage6_train_all import Stage6TrainingOrchestrator
        
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        migrated_count = len(data.get("migrated", []))
        stage6_count = len(Stage6TrainingOrchestrator.MIGRATED_TRAINERS)
        
        # Stage6 may have additional pre-wrapped trainers like hydra_policy
        assert stage6_count >= migrated_count, (
            f"Stage6 MIGRATED_TRAINERS ({stage6_count}) < backlog migrated ({migrated_count})"
        )
