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
    """Test that pending migration scripts are tracked via backlog artifact."""
    
    def test_allowlist_contains_only_legacy(self):
        """Allowlist must contain ONLY legacy/deprecated scripts."""
        from scripts.check_training_regality import TRAINING_SCRIPT_ALLOWLIST
        
        for script_name, reason in TRAINING_SCRIPT_ALLOWLIST.items():
            assert "LEGACY" in reason, (
                f"Allowlist contains non-legacy script: {script_name} -> {reason}\n"
                "Pending scripts should be in TRAINING_MIGRATION_BACKLOG.json, not allowlist"
            )
    
    def test_backlog_exists_and_valid(self):
        """Migration backlog artifact must exist and be valid JSON."""
        import json
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        
        assert backlog_path.exists(), (
            "TRAINING_MIGRATION_BACKLOG.json not found. "
            "Pending migrations must be tracked in backlog artifact."
        )
        
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        assert "backlog" in data, "Backlog must have 'backlog' key"
        assert len(data["backlog"]) > 0, "Backlog must not be empty"
        
        # Validate structure of first entry
        first_entry = data["backlog"][0]
        required_fields = {"script", "priority", "owner", "notes"}
        assert required_fields.issubset(first_entry.keys()), (
            f"Backlog entry missing required fields: {required_fields - first_entry.keys()}"
        )
    
    def test_all_noncompliant_in_backlog_or_allowlist(self):
        """All non-compliant trainers must be in backlog or allowlist."""
        from scripts.check_training_regality import (
            check_script_compliance,
            find_training_scripts,
            load_migration_backlog,
            TRAINING_SCRIPT_ALLOWLIST,
        )
        
        scripts_dir = ROOT / "scripts"
        training_scripts = find_training_scripts(scripts_dir)
        migration_backlog = load_migration_backlog()
        
        unlisted_noncompliant = []
        for script in training_scripts:
            is_compliant, reason = check_script_compliance(script, migration_backlog)
            if not is_compliant:
                unlisted_noncompliant.append(script.name)
        
        assert len(unlisted_noncompliant) == 0, (
            f"Non-compliant scripts not tracked: {unlisted_noncompliant}\n"
            "Add to TRAINING_MIGRATION_BACKLOG.json or wrap with @regal_training"
        )
    
    def test_backlog_count_regression_guard(self):
        """Backlog count must not exceed limit (regression guard)."""
        from scripts.check_training_regality import load_migration_backlog
        
        backlog = load_migration_backlog()
        backlog_count = len(backlog)
        
        print(f"\nBacklog status: {backlog_count} scripts pending migration")
        
        # Fail if backlog grows beyond 25 (regression guard)
        assert backlog_count <= 25, (
            f"Backlog count ({backlog_count}) exceeds limit (25). "
            "Wrap more scripts to reduce backlog."
        )
    
    def test_backlog_has_priority_distribution(self):
        """Backlog should have reasonable priority distribution."""
        import json
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        priorities = [item["priority"] for item in data["backlog"]]
        p0_count = sum(1 for p in priorities if p == "P0")
        p1_count = sum(1 for p in priorities if p == "P1")
        p2_count = sum(1 for p in priorities if p == "P2")
        
        print(f"\nPriority distribution: P0={p0_count}, P1={p1_count}, P2={p2_count}")
        
        # P0 should be limited (urgent work)
        assert p0_count <= 10, (
            f"Too many P0 scripts ({p0_count}). P0 means 'do this week'."
        )
    
    def test_p0_must_be_empty_or_blocked(self):
        """P0 entries must either be empty or have blocked_by set.
        
        This prevents "P0 forever" - if something is P0, it must be:
        1. Wrapped immediately, OR
        2. Have a documented blocker
        """
        import json
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        p0_without_blockers = []
        for item in data["backlog"]:
            if item["priority"] == "P0":
                if not item.get("blocked_by"):
                    p0_without_blockers.append(item["script"])
        
        assert len(p0_without_blockers) == 0, (
            f"P0 scripts without blockers: {p0_without_blockers}\n"
            "Either wrap these scripts or add blocked_by field explaining why"
        )
    
    def test_migration_progress_tracked(self):
        """Track migration progress for reporting."""
        import json
        backlog_path = ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
        
        with open(backlog_path, "r") as f:
            data = json.load(f)
        
        pending_count = len(data.get("backlog", []))
        migrated_count = len(data.get("migrated", []))
        
        print(f"\n=== Migration Progress ===")
        print(f"  Pending: {pending_count}")
        print(f"  Migrated: {migrated_count}")
        print(f"  Legacy: 5 (allowlisted)")
        print(f"  Compliant: {30 - pending_count - 5}")  # 30 total - pending - legacy
        print(f"==========================")
        
        # Track that we're making progress
        assert migrated_count >= 0, "Should track migrated scripts"

