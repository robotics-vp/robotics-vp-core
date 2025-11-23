"""
Run Phase H Economic Learner.

Generates skill portfolio artifacts:
- skill_market_state.json
- exploration_budget.json
- skill_returns.json
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.phase_h.economic_learner import EconomicLearner
from src.phase_h.models import Skill, SkillStatus


def create_example_skills() -> list:
    """Create example skills for testing."""
    return [
        Skill(
            skill_id="drawer_open_v2",
            display_name="Drawer Opening v2",
            description="Open drawers with various handle types",
            mpl_baseline=50.0,
            mpl_current=58.5,
            mpl_target=65.0,
            training_cost_usd=450.0,
            data_cost_per_episode=0.50,
            success_rate=0.87,
            failure_rate=0.13,
            recovery_rate=0.65,
            fragility_score=0.87,
            ood_exposure=0.15,
            novelty_tier_avg=1.2,
            training_episodes=900,
            last_updated=datetime.utcnow().isoformat() + "Z",
            status=SkillStatus.TRAINING.value,
        ),
        Skill(
            skill_id="dish_place_precision",
            display_name="Dish Placement (Precision)",
            description="Place dishes with high precision and low breakage",
            mpl_baseline=60.0,
            mpl_current=72.0,
            mpl_target=70.0,
            training_cost_usd=1500.0,
            data_cost_per_episode=0.50,
            success_rate=0.96,
            failure_rate=0.04,
            recovery_rate=0.85,
            fragility_score=0.96,
            ood_exposure=0.05,
            novelty_tier_avg=0.5,
            training_episodes=3000,
            last_updated=datetime.utcnow().isoformat() + "Z",
            status=SkillStatus.MATURE.value,
        ),
        Skill(
            skill_id="cup_stacking_fast",
            display_name="Fast Cup Stacking",
            description="Stack cups quickly with moderate precision",
            mpl_baseline=45.0,
            mpl_current=48.0,
            mpl_target=60.0,
            training_cost_usd=200.0,
            data_cost_per_episode=0.50,
            success_rate=0.55,
            failure_rate=0.45,
            recovery_rate=0.30,
            fragility_score=0.55,
            ood_exposure=0.25,
            novelty_tier_avg=1.8,
            training_episodes=400,
            last_updated=datetime.utcnow().isoformat() + "Z",
            status=SkillStatus.EXPLORATION.value,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Phase H Economic Learner")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase_h",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--total-budget",
        type=float,
        default=10000.0,
        help="Total exploration budget (USD)",
    )
    parser.add_argument(
        "--reallocation-period",
        type=int,
        default=1000,
        help="Episodes between budget reallocations",
    )

    args = parser.parse_args()

    print("=== Phase H Economic Learner ===\n")

    # Create learner
    config = {
        "total_exploration_budget": args.total_budget,
        "reallocation_period_episodes": args.reallocation_period,
        "price_per_unit": 0.30,  # Dishwashing example
        "hours_deployed": 1000,
    }

    learner = EconomicLearner(config)

    # Add example skills
    skills = create_example_skills()
    for skill in skills:
        learner.add_skill(skill)
        print(f"Added skill: {skill.display_name}")
        print(f"  MPL: {skill.mpl_current:.1f} / {skill.mpl_target:.1f}")
        print(f"  Success rate: {skill.success_rate:.2%}")
        print(f"  Status: {skill.status}")
        print()

    # Run a learner cycle
    print("Running learner cycle...")
    summary = learner.run_cycle(args.reallocation_period)

    if summary:
        print("\nLearner Cycle Summary:")
        print(f"  Total budget: ${summary['total_budget_usd']:,.2f}")
        print(f"  Skills: {summary['skill_count']}")
        print("\n  ROI by skill:")
        for skill_id, roi in summary['roi_by_skill'].items():
            print(f"    {skill_id}: {roi:.1f}%")

    # Save artifacts
    output_dir = Path(args.output_dir)
    learner.save_artifacts(output_dir)

    print(f"\n✓ Artifacts saved to {output_dir}/")
    print("  - skill_market_state.json")
    print("  - exploration_budget.json")
    print("  - skill_returns.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
