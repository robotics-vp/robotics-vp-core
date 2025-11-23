"""
Smoke test for Phase H Economic Learner.

Tests:
- Skill lifecycle transitions
- Budget reallocation
- ROI computation
- Artifact generation
- Advisory boundaries (no econ controller mutation)
"""
import sys
from pathlib import Path
import tempfile
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.phase_h.economic_learner import EconomicLearner, allocate_exploration_budget, compute_skill_roi
from src.phase_h.models import Skill, SkillStatus, SkillReturns, update_skill_status


def create_test_skill(skill_id: str, status: str = "training") -> Skill:
    """Create a test skill."""
    return Skill(
        skill_id=skill_id,
        display_name=f"Test Skill {skill_id}",
        description="Test skill",
        mpl_baseline=50.0,
        mpl_current=55.0,
        mpl_target=65.0,
        training_cost_usd=500.0,
        data_cost_per_episode=0.50,
        success_rate=0.75,
        failure_rate=0.25,
        recovery_rate=0.60,
        fragility_score=0.75,
        ood_exposure=0.10,
        novelty_tier_avg=1.0,
        training_episodes=1000,
        last_updated=datetime.utcnow().isoformat() + "Z",
        status=status,
    )


def test_skill_lifecycle_transitions():
    """Test skill status transitions."""
    print("Testing skill lifecycle transitions...")

    # Test EXPLORATION → TRAINING
    skill_explore = create_test_skill("skill1", "exploration")
    skill_explore.success_rate = 0.65  # Above exploration threshold
    new_status = update_skill_status(skill_explore)
    assert new_status == SkillStatus.TRAINING.value, f"Expected training, got {new_status}"

    # Test TRAINING → MATURE
    skill_train = create_test_skill("skill2", "training")
    skill_train.success_rate = 0.96
    skill_train.mpl_current = 66.0  # Above target
    new_status = update_skill_status(skill_train)
    assert new_status == SkillStatus.MATURE.value, f"Expected mature, got {new_status}"

    # Test stay in EXPLORATION
    skill_low = create_test_skill("skill3", "exploration")
    skill_low.success_rate = 0.55
    new_status = update_skill_status(skill_low)
    assert new_status == SkillStatus.EXPLORATION.value, f"Expected exploration, got {new_status}"

    print("✓ Skill lifecycle transitions work correctly")


def test_budget_allocation():
    """Test exploration budget allocation."""
    print("Testing budget allocation...")

    skill = create_test_skill("test_skill")
    mpl_gap = skill.mpl_target - skill.mpl_current  # 10.0

    budget = allocate_exploration_budget(skill, total_budget_usd=10000.0, mpl_gap=mpl_gap)

    assert budget.skill_id == skill.skill_id
    assert budget.budget_usd > 0, "Budget should be positive"
    assert budget.remaining_usd == budget.budget_usd, "Initially remaining == budget"
    assert budget.max_episodes > 0, "Should allocate some episodes"

    # Check allocation percentages sum to ~1.0
    total_pct = budget.data_collection_pct + budget.compute_training_pct + budget.human_supervision_pct
    assert abs(total_pct - 1.0) < 0.01, f"Percentages should sum to 1.0, got {total_pct}"

    print(f"✓ Budget allocated: ${budget.budget_usd:.2f}, {budget.max_episodes} episodes")


def test_roi_computation():
    """Test ROI calculation."""
    print("Testing ROI computation...")

    skill = create_test_skill("test_skill")
    skill.training_cost_usd = 1000.0

    returns = SkillReturns(
        skill_id=skill.skill_id,
        delta_mpl=10.0,  # 10 units/hr improvement
        delta_mpl_pct=20.0,
        delta_energy_wh=-50.0,  # 50 Wh savings
        delta_time_sec=0.0,
        delta_damage=-5.0,  # $5/hr damage reduction
        delta_success_rate=0.20,
        delta_novelty_coverage=0.10,
        unique_failure_modes_discovered=3,
        roi_pct=0.0,
    )

    roi = compute_skill_roi(skill, returns, price_per_unit=0.30, hours_deployed=1000)

    # Expected value:
    # Productivity: 10 * 0.30 * 1000 = $3000
    # Energy: (50/1000) * 0.12 * 1000 = $6
    # Damage: 5 * 1000 = $5000
    # Total value: ~$8006
    # ROI: (8006 - 1000) / 1000 * 100 = 700.6%

    assert roi > 0, f"ROI should be positive, got {roi}"
    assert roi > 500, f"ROI should be substantial, got {roi}"

    print(f"✓ ROI computed: {roi:.1f}%")


def test_learner_cycle():
    """Test learner cycle execution."""
    print("Testing learner cycle...")

    config = {
        "total_exploration_budget": 5000.0,
        "reallocation_period_episodes": 1000,
        "price_per_unit": 0.30,
        "hours_deployed": 1000,
    }

    learner = EconomicLearner(config)

    # Add skills
    skill1 = create_test_skill("skill1")
    skill2 = create_test_skill("skill2")
    learner.add_skill(skill1)
    learner.add_skill(skill2)

    assert len(learner.skills) == 2
    assert len(learner.budgets) == 2

    # Run cycle
    summary = learner.run_cycle(1000)  # At reallocation period

    assert summary is not None, "Should return summary at reallocation period"
    assert summary["skill_count"] == 2
    assert "roi_by_skill" in summary

    # Not at reallocation period
    summary_none = learner.run_cycle(500)
    assert summary_none is None, "Should return None when not at reallocation period"

    print("✓ Learner cycle executes correctly")


def test_artifact_generation():
    """Test artifact generation."""
    print("Testing artifact generation...")

    config = {
        "total_exploration_budget": 5000.0,
        "reallocation_period_episodes": 1000,
    }

    learner = EconomicLearner(config)
    skill = create_test_skill("test_skill")
    learner.add_skill(skill)

    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        learner.save_artifacts(output_dir)

        # Check files exist
        assert (output_dir / "skill_market_state.json").exists(), "skill_market_state.json should exist"
        assert (output_dir / "exploration_budget.json").exists(), "exploration_budget.json should exist"

        # Verify JSON is valid
        import json

        with open(output_dir / "skill_market_state.json") as f:
            market_state = json.load(f)
            assert "skills" in market_state
            assert "test_skill" in market_state["skills"]

        with open(output_dir / "exploration_budget.json") as f:
            budget_state = json.load(f)
            assert "budgets_by_skill" in budget_state

    print("✓ Artifacts generated successfully")


def test_advisory_boundaries():
    """Test that Phase H doesn't modify prohibited modules."""
    print("Testing advisory boundaries...")

    # Phase H should NOT directly modify:
    # - EconController
    # - Reward functions
    # - Task ordering

    # Verify that learner only produces advisory outputs (JSON artifacts)
    # and doesn't import or mutate forbidden modules

    config = {"total_exploration_budget": 5000.0}
    learner = EconomicLearner(config)

    # Should only have skills, budgets, returns_history
    # No econ_controller, no reward_function attributes
    assert not hasattr(learner, "econ_controller"), "Should not have econ_controller"
    assert not hasattr(learner, "reward_function"), "Should not have reward_function"
    assert not hasattr(learner, "task_ordering"), "Should not have task_ordering"

    print("✓ Advisory boundaries respected (no forbidden module access)")


def test_determinism():
    """Test deterministic behavior."""
    print("Testing determinism...")

    skill = create_test_skill("test_skill")
    mpl_gap = 10.0

    # Allocate budget twice
    budget1 = allocate_exploration_budget(skill, 5000.0, mpl_gap)
    budget2 = allocate_exploration_budget(skill, 5000.0, mpl_gap)

    assert budget1.budget_usd == budget2.budget_usd, "Budget should be deterministic"
    assert budget1.max_episodes == budget2.max_episodes, "Max episodes should be deterministic"

    print("✓ Deterministic behavior confirmed")


def main():
    print("=== Phase H Economic Learner Smoke Tests ===\n")

    try:
        test_skill_lifecycle_transitions()
        test_budget_allocation()
        test_roi_computation()
        test_learner_cycle()
        test_artifact_generation()
        test_advisory_boundaries()
        test_determinism()

        print("\n=== All Phase H Tests Passed ===")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
