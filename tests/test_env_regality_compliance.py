"""Environment regality compliance tests.

CI rule: Any env under src/envs/* must declare REGALITY_LEVEL.
- FULL: must implement hooks + have runner
- BASIC: must include reason string, excluded from Stage6

This prevents "cute demo envs" from polluting the training graph.
"""
import pytest
import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestRegalEnvKitProtocol:
    """Test RegalEnvKit protocol."""
    
    def test_protocol_importable(self):
        """RegalEnvKit protocol can be imported."""
        from src.training.regal_env_kit import (
            RegalEnvKit,
            RegalityLevel,
            validate_env_regality,
            is_stage6_eligible,
        )
        
        assert RegalEnvKit is not None
        assert RegalityLevel.FULL == "FULL"
        assert RegalityLevel.BASIC == "BASIC"
    
    def test_validate_env_regality_detects_missing(self):
        """validate_env_regality detects missing declarations."""
        from src.training.regal_env_kit import validate_env_regality
        
        # Fake env with no declarations
        class NoDeclarationEnv:
            pass
        
        result = validate_env_regality(NoDeclarationEnv())
        
        assert not result["valid"]
        assert "REGALITY_LEVEL" in result["missing"]
    
    def test_validate_env_regality_full_valid(self):
        """validate_env_regality validates FULL env correctly."""
        from src.training.regal_env_kit import validate_env_regality
        from src.contracts.schemas import RewardBreakdownV1, TrajectoryAuditV1
        
        class FullEnv:
            REGALITY_LEVEL = "FULL"
            env_type = "workcell"
            task_family = "dishwashing"
            safety_invariants = ["no_collision", "gripper_safe"]
            
            def compute_reward_breakdown(self, reward, info):
                return RewardBreakdownV1(
                    step_reward=reward,
                    components={},
                )
            
            def build_trajectory_audit(self, episode_id, trajectory):
                return TrajectoryAuditV1(
                    episode_id=episode_id,
                    num_steps=len(trajectory),
                )
        
        result = validate_env_regality(FullEnv())
        
        assert result["valid"]
        assert result["level"] == "FULL"
        assert len(result["missing"]) == 0
    
    def test_validate_env_regality_basic_requires_reason(self):
        """BASIC envs must have REGALITY_BASIC_REASON."""
        from src.training.regal_env_kit import validate_env_regality
        
        class BasicEnvNoReason:
            REGALITY_LEVEL = "BASIC"
            env_type = "demo"
            task_family = "toy"
            safety_invariants = []
        
        result = validate_env_regality(BasicEnvNoReason())
        
        assert not result["valid"]
        assert "REGALITY_BASIC_REASON" in result["missing"]
    
    def test_validate_env_regality_basic_valid(self):
        """BASIC env with reason is valid."""
        from src.training.regal_env_kit import validate_env_regality
        
        class BasicEnvWithReason:
            REGALITY_LEVEL = "BASIC"
            env_type = "demo"
            task_family = "toy"
            safety_invariants = []
            REGALITY_BASIC_REASON = "Demo-only, no production use"
        
        result = validate_env_regality(BasicEnvWithReason())
        
        assert result["valid"]
        assert result["level"] == "BASIC"
        assert result["reason"] == "Demo-only, no production use"
    
    def test_is_stage6_eligible_full_only(self):
        """Only FULL envs are Stage6 eligible."""
        from src.training.regal_env_kit import is_stage6_eligible
        from src.contracts.schemas import RewardBreakdownV1, TrajectoryAuditV1
        
        class FullEnv:
            REGALITY_LEVEL = "FULL"
            env_type = "workcell"
            task_family = "dishwashing"
            safety_invariants = []
            
            def compute_reward_breakdown(self, reward, info):
                return RewardBreakdownV1(step_reward=reward, components={})
            
            def build_trajectory_audit(self, episode_id, trajectory):
                return TrajectoryAuditV1(episode_id=episode_id, num_steps=0)
        
        class BasicEnv:
            REGALITY_LEVEL = "BASIC"
            env_type = "demo"
            task_family = "toy"
            safety_invariants = []
            REGALITY_BASIC_REASON = "Demo only"
        
        assert is_stage6_eligible(FullEnv())
        assert not is_stage6_eligible(BasicEnv())


class TestEnvComplianceAudit:
    """Audit existing envs for regality compliance."""
    
    def test_workcell_envs_directory_structure(self):
        """Check src/envs directory structure."""
        envs_dir = ROOT / "src" / "envs"
        assert envs_dir.exists(), "src/envs directory must exist"
    
    def test_physics_envs_exist(self):
        """Physics envs directory exists."""
        physics_dir = ROOT / "src" / "envs" / "physics"
        # May or may not exist - just check structure
        if physics_dir.exists():
            self._audit_env_files(physics_dir)
    
    def _audit_env_files(self, env_dir: Path):
        """Audit env files in a directory."""
        env_files = list(env_dir.glob("*_env.py"))
        
        for env_file in env_files:
            content = env_file.read_text()
            
            # Check for REGALITY_LEVEL declaration
            has_regality = "REGALITY_LEVEL" in content
            
            # This is advisory - we want to track coverage
            if not has_regality:
                print(f"  [ADVISORY] {env_file.name}: Missing REGALITY_LEVEL")


class TestEnvRegistryCompliance:
    """Test that any registered envs have regality declarations."""
    
    def test_env_registry_if_exists(self):
        """If env registry exists, all entries should be compliant."""
        registry_path = ROOT / "src" / "envs" / "registry.py"
        
        if not registry_path.exists():
            pytest.skip("No env registry found")
        
        # If registry exists, check it
        content = registry_path.read_text()
        
        # Look for env class definitions
        # This is a lightweight check - full validation would import and check
        assert "REGALITY_LEVEL" in content or "regal" in content.lower(), (
            "Env registry should reference regality compliance"
        )


class TestDefaultEnvType:
    """Test DEFAULT_ENV_TYPE is workcell."""
    
    def test_default_is_workcell(self):
        """DEFAULT_ENV_TYPE must be 'workcell'."""
        from src.training.regal_env_kit import DEFAULT_ENV_TYPE
        
        assert DEFAULT_ENV_TYPE == "workcell", (
            "Workcell must be the paramount env_type"
        )
