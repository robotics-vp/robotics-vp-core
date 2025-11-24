#!/usr/bin/env python3
"""
Phase H Controller CLI.

Runs Phase H cycle orchestrator for economic learning integration.

Usage:
    python3 run_phase_h_controller.py \
      --ontology-root data/ontology \
      --episodes-per-cycle 1000 \
      --enable-phase-h
"""
import argparse
import json
from pathlib import Path

from src.phase_h.controller import PhaseHCycleOrchestrator
from src.phase_h.economic_learner import EconomicLearner


def main():
    parser = argparse.ArgumentParser(description="Phase H Cycle Controller")
    parser.add_argument(
        "--ontology-root",
        type=str,
        required=True,
        help="Path to ontology root directory",
    )
    parser.add_argument(
        "--episodes-per-cycle",
        type=int,
        default=1000,
        help="Episodes between Phase H cycles (default: 1000)",
    )
    parser.add_argument(
        "--enable-phase-h",
        action="store_true",
        help="Enable Phase H integration",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=5000,
        help="Total episodes to simulate (default: 5000)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/phase_h",
        help="Directory for Phase H logs (default: logs/phase_h)",
    )
    parser.add_argument(
        "--total-budget",
        type=float,
        default=10000.0,
        help="Total exploration budget in USD (default: 10000.0)",
    )

    args = parser.parse_args()

    # Initialize controller
    config = {
        "ontology_root": args.ontology_root,
        "cycle_period_episodes": args.episodes_per_cycle,
        "enable_phase_h": args.enable_phase_h,
        "log_dir": args.log_dir,
    }

    print("=== Phase H Cycle Controller ===\n")
    print(f"Ontology Root: {args.ontology_root}")
    print(f"Episodes Per Cycle: {args.episodes_per_cycle}")
    print(f"Enable Phase H: {args.enable_phase_h}")
    print(f"Total Episodes: {args.total_episodes}")
    print(f"Log Directory: {args.log_dir}\n")

    orchestrator = PhaseHCycleOrchestrator(config)

    # Initialize economic learner
    learner_config = {
        "total_exploration_budget": args.total_budget,
        "reallocation_period_episodes": args.episodes_per_cycle,
        "price_per_unit": 0.30,
        "hours_deployed": 1000,
    }

    learner = EconomicLearner(learner_config)

    # Add example skills (would be loaded from real data in production)
    from src.phase_h.models import Skill
    import time

    learner.add_skill(Skill(
        skill_id="dishwashing_precision",
        display_name="Dishwashing (Precision Mode)",
        description="High-precision dish cleaning with fragility awareness",
        mpl_baseline=50.0,
        mpl_current=55.0,
        mpl_target=60.0,
        training_cost_usd=500.0,
        data_cost_per_episode=1.0,
        success_rate=0.75,
        failure_rate=0.25,
        recovery_rate=0.6,
        fragility_score=0.75,
        ood_exposure=0.3,
        novelty_tier_avg=1.5,
        training_episodes=500,
        last_updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status="training",
    ))

    learner.add_skill(Skill(
        skill_id="drawer_open_v2",
        display_name="Drawer Opening V2",
        description="Robust drawer opening with novel handle detection",
        mpl_baseline=40.0,
        mpl_current=50.0,
        mpl_target=60.0,
        training_cost_usd=300.0,
        data_cost_per_episode=0.8,
        success_rate=0.85,
        failure_rate=0.15,
        recovery_rate=0.8,
        fragility_score=0.85,
        ood_exposure=0.5,
        novelty_tier_avg=2.0,
        training_episodes=300,
        last_updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status="training",
    ))

    # Simulate training loop
    print("=== Starting Phase H Cycle Loop ===\n")

    for episode in range(args.total_episodes):
        # Run learner cycle
        learner_summary = learner.run_cycle(episode)
        if learner_summary:
            print(f"[Episode {episode}] Economic Learner Cycle:")
            print(f"  - Total Budget: ${learner_summary['total_budget_usd']:.2f}")
            print(f"  - Skills: {learner_summary['skill_count']}")
            print(f"  - ROI by Skill: {learner_summary['roi_by_skill']}")

            # Save artifacts
            ontology_root = Path(args.ontology_root)
            phase_h_dir = ontology_root / "phase_h"
            learner.save_artifacts(phase_h_dir)
            print(f"  - Artifacts saved to {phase_h_dir}\n")

        # Run controller cycle
        cycle_summary = orchestrator.run_cycle_once(episode)
        if cycle_summary:
            print(f"[Episode {episode}] Phase H Cycle {cycle_summary['cycle_count']}:")
            print(f"  - Timestamp: {cycle_summary['timestamp']}")
            print(f"  - Skill Multipliers: {cycle_summary['skill_multipliers']}")
            print(f"  - Routing Advisories:")
            for key, val in cycle_summary['routing_advisories'].items():
                print(f"    - {key}: {val}")
            print()

    # Final summary
    print("\n=== Phase H Controller Summary ===")
    controller_summary = orchestrator.get_cycle_summary()
    print(json.dumps(controller_summary, indent=2))

    print(f"\nPhase H artifacts saved to: {args.ontology_root}/phase_h/")
    print(f"Cycle logs saved to: {args.log_dir}/")


if __name__ == "__main__":
    main()
