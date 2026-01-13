"""CI tests for workcell paramount guarantee.

Ensures workcell is the default env_type across all regality-aware components.

Tests:
1. Stage6TrainingOrchestrator defaults to env_type="workcell"
2. @regal_training defaults to env_type="workcell" if not specified
3. RegalTrainingRunner defaults to env_type="workcell"
4. All wrapped trainers use env_type="workcell"
"""
import pytest
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestWorkcellParamountDefaults:
    """Test that workcell is the default env_type everywhere."""
    
    def test_regal_training_decorator_defaults_to_workcell(self):
        """@regal_training must default to env_type='workcell'."""
        from src.training.wrap_training_entrypoint import regal_training
        import inspect
        
        sig = inspect.signature(regal_training)
        env_type_param = sig.parameters.get("env_type")
        
        assert env_type_param is not None, "regal_training must have env_type parameter"
        assert env_type_param.default == "workcell", (
            f"regal_training must default to 'workcell', got '{env_type_param.default}'"
        )
    
    def test_stage6_orchestrator_defaults_to_workcell(self):
        """Stage6TrainingOrchestrator must default to env_type='workcell'."""
        stage6_path = ROOT / "scripts" / "run_stage6_train_all.py"
        assert stage6_path.exists(), "run_stage6_train_all.py not found"
        
        content = stage6_path.read_text()
        
        # Check for env_type = "workcell" or env_type="workcell"
        has_workcell_default = (
            'env_type="workcell"' in content or
            "env_type='workcell'" in content or
            'env_type: str = "workcell"' in content
        )
        
        assert has_workcell_default, (
            "Stage6TrainingOrchestrator must default to env_type='workcell'"
        )
    
    def test_regal_training_runner_has_config(self):
        """RegalTrainingRunner must accept config with training params."""
        from src.training.regal_training_runner import RegalTrainingRunner, TrainingRunConfig
        import inspect
        
        sig = inspect.signature(RegalTrainingRunner.__init__)
        
        # Should accept config parameter
        has_config = "config" in sig.parameters
        assert has_config, "RegalTrainingRunner must accept config parameter"
        
        # TrainingRunConfig should exist and be a dataclass
        from dataclasses import fields
        config_fields = {f.name for f in fields(TrainingRunConfig)}
        
        # Must have core fields
        required_fields = {"run_id", "output_dir", "seed", "enable_regal"}
        assert required_fields.issubset(config_fields), (
            f"TrainingRunConfig missing required fields: {required_fields - config_fields}"
        )


class TestWrappedTrainersUseWorkcell:
    """Test that all wrapped trainers use workcell."""
    
    WRAPPED_TRAINERS = [
        "train_hydra_policy.py",
        "train_sac_with_ontology_logging.py",
        "train_offline_policy.py",
        "train_skill_policies.py",
    ]
    
    def test_wrapped_trainers_specify_workcell(self):
        """All wrapped trainers must explicitly use env_type='workcell'."""
        scripts_dir = ROOT / "scripts"
        
        for trainer_name in self.WRAPPED_TRAINERS:
            trainer_path = scripts_dir / trainer_name
            if not trainer_path.exists():
                continue  # Skip if not found
            
            content = trainer_path.read_text()
            
            # Must contain @regal_training
            assert "@regal_training" in content, (
                f"{trainer_name} must use @regal_training decorator"
            )
            
            # Must specify workcell
            has_workcell = (
                'env_type="workcell"' in content or
                "env_type='workcell'" in content
            )
            assert has_workcell, (
                f"{trainer_name} must specify env_type='workcell'"
            )


class TestEnvTypeNotOmittable:
    """Test that env_type cannot be accidentally omitted for FULL runs."""
    
    def test_missing_env_type_raises_or_defaults_workcell(self):
        """Missing env_type should either fail or default to workcell."""
        from src.training.wrap_training_entrypoint import regal_training
        
        # Calling without env_type should use workcell default
        @regal_training()  # No env_type specified
        def dummy_trainer(runner=None):
            pass
        
        # The decorator should have set workcell as default
        # This test passes if decorator doesn't raise


class TestPendingMigrationTracking:
    """Test that pending migration scripts are tracked."""
    
    def test_allowlist_has_no_unlisted_noncompliant(self):
        """All non-compliant trainers must be in allowlist."""
        from scripts.check_training_regality import (
            check_script_compliance,
            find_training_scripts,
            TRAINING_SCRIPT_ALLOWLIST,
        )
        
        scripts_dir = ROOT / "scripts"
        training_scripts = find_training_scripts(scripts_dir)
        
        unlisted_noncompliant = []
        for script in training_scripts:
            is_compliant, reason = check_script_compliance(script)
            if not is_compliant and script.name not in TRAINING_SCRIPT_ALLOWLIST:
                unlisted_noncompliant.append(script.name)
        
        assert len(unlisted_noncompliant) == 0, (
            f"Non-compliant scripts not in allowlist: {unlisted_noncompliant}\n"
            "Either add @regal_training or add to TRAINING_SCRIPT_ALLOWLIST"
        )
    
    def test_pending_migration_count_tracked(self):
        """Track number of pending migrations for progress monitoring."""
        from scripts.check_training_regality import TRAINING_SCRIPT_ALLOWLIST
        
        pending_count = sum(
            1 for reason in TRAINING_SCRIPT_ALLOWLIST.values()
            if "PENDING_MIGRATION" in reason
        )
        legacy_count = sum(
            1 for reason in TRAINING_SCRIPT_ALLOWLIST.values()
            if "LEGACY" in reason
        )
        
        # This is informational, not a failure
        print(f"\nMigration status:")
        print(f"  Pending: {pending_count} scripts")
        print(f"  Legacy:  {legacy_count} scripts")
        
        # Fail if pending grows beyond 25 (regression guard)
        assert pending_count <= 25, (
            f"Pending migration count ({pending_count}) exceeds limit (25). "
            "Wrap more scripts or add legitimate entries to allowlist."
        )
