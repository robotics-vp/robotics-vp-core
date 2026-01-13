"""
Training entrypoint wrapper for regality compliance.

Provides a decorator and helper function to wrap any training script
with RegalTrainingRunner, ensuring FULL artifact production + verify_run().

Usage:
    from src.training.wrap_training_entrypoint import wrap_training_entrypoint, regal_training

    # Decorator style
    @regal_training(env_type="workcell")
    def main(runner: RegalTrainingRunner):
        # Your training code here
        runner.record_sample("pick", datapack_id="dp_001")
        runner.add_trajectory_audit(audit)
        ...

    # Or function wrapper style
    def train():
        ...
    
    wrap_training_entrypoint(train, env_type="workcell")
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    TrainingRunResult,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json


def create_training_argparser(
    description: str = "Training script with regality compliance",
) -> argparse.ArgumentParser:
    """Create argument parser with standard training + regality flags.
    
    Returns:
        ArgumentParser with --seed, --output-dir, --env-type, --regal-level, etc.
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Standard training args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="artifacts/training", 
                        help="Output directory for artifacts")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of training episodes")
    parser.add_argument("--training-steps", type=int, default=1000,
                        help="Number of training steps")
    
    # Regality args
    parser.add_argument("--env-type", type=str, default="workcell",
                        choices=["workcell", "dishwashing", "manufacturing"],
                        help="Environment type (default: workcell)")
    parser.add_argument("--regal-level", type=str, default="FULL",
                        choices=["NONE", "DEMO", "FULL"],
                        help="Regality level (default: FULL)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification (NOT recommended)")
    parser.add_argument("--no-fail-on-verify", action="store_true",
                        help="Don't exit with error on verification failure")
    
    # Quarantine
    parser.add_argument("--quarantine-ids", type=str, nargs="*", default=[],
                        help="Datapack IDs to quarantine")
    
    return parser


def wrap_training_entrypoint(
    training_fn: Callable[[RegalTrainingRunner], None],
    *,
    env_type: str = "workcell",
    seed: int = 42,
    output_dir: str = "artifacts/training",
    num_episodes: int = 10,
    training_steps: int = 1000,
    regal_ids: Optional[List[str]] = None,
    quarantine_ids: Optional[List[str]] = None,
    plan_config: Optional[Dict[str, Any]] = None,
    fail_on_verify_error: bool = True,
) -> TrainingRunResult:
    """Wrap a training function with full regality compliance.
    
    Args:
        training_fn: Training function that receives RegalTrainingRunner
        env_type: Environment type (workcell, dishwashing, manufacturing)
        seed: Random seed
        output_dir: Output directory for artifacts
        num_episodes: Number of training episodes
        training_steps: Number of training steps
        regal_ids: List of regal IDs to evaluate (default: all)
        quarantine_ids: Datapack IDs to quarantine
        plan_config: Optional plan configuration dict (for plan_sha)
        fail_on_verify_error: Exit with error on verification failure
        
    Returns:
        TrainingRunResult with all artifact SHAs and verification result
    """
    # Create config
    config = TrainingRunConfig(
        output_dir=output_dir,
        seed=seed,
        num_episodes=num_episodes,
        training_steps=training_steps,
        quarantine_datapack_ids=quarantine_ids or [],
        regal_ids=regal_ids or ["spec_guardian", "world_coherence", "reward_integrity", "econ_data"],
        fail_on_verify_error=fail_on_verify_error,
    )
    
    # Compute plan SHA
    plan_dict = plan_config or {
        "env_type": env_type,
        "seed": seed,
        "num_episodes": num_episodes,
        "training_steps": training_steps,
    }
    plan_sha = sha256_json(plan_dict)
    
    # Run with regality
    return run_training_with_regality(
        training_fn=training_fn,
        config=config,
        plan_sha=plan_sha,
        plan_id=f"train_{env_type}",
    )


def regal_training(
    env_type: str = "workcell",
    regal_ids: Optional[List[str]] = None,
    fail_on_verify_error: bool = True,
) -> Callable:
    """Decorator for training functions with regality compliance.
    
    Usage:
        @regal_training(env_type="workcell")
        def main(runner: RegalTrainingRunner, args):
            # Training code
            ...
            
    The decorated function receives:
        - runner: RegalTrainingRunner instance
        - args: Parsed command line arguments
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(argv: Optional[List[str]] = None):
            parser = create_training_argparser(
                description=fn.__doc__ or f"Training script: {fn.__name__}"
            )
            # Allow training function to add its own args
            args, remaining = parser.parse_known_args(argv)
            
            # Override env_type if specified in decorator
            effective_env_type = args.env_type or env_type
            
            def training_with_args(runner: RegalTrainingRunner):
                fn(runner, args)
            
            # Run with regality
            result = wrap_training_entrypoint(
                training_with_args,
                env_type=effective_env_type,
                seed=args.seed,
                output_dir=args.output_dir,
                num_episodes=args.num_episodes,
                training_steps=args.training_steps,
                regal_ids=regal_ids,
                quarantine_ids=args.quarantine_ids,
                fail_on_verify_error=not args.no_fail_on_verify and fail_on_verify_error,
            )
            
            return result
        return wrapper
    return decorator


# List of training scripts that are ALLOWED to NOT use RegalTrainingRunner
# Each entry must have a reason
TRAINING_SCRIPT_ALLOWLIST: Dict[str, str] = {
    # Example:
    # "train_legacy_baseline.py": "Historical baseline, frozen for comparison",
}


def check_training_script_compliance(script_path: str) -> bool:
    """Check if a training script uses RegalTrainingRunner.
    
    For CI: Fails if script doesn't import runner and isn't in allowlist.
    
    Returns:
        True if compliant, False otherwise
    """
    path = Path(script_path)
    
    # Check allowlist
    if path.name in TRAINING_SCRIPT_ALLOWLIST:
        print(f"[ALLOWLIST] {path.name}: {TRAINING_SCRIPT_ALLOWLIST[path.name]}")
        return True
    
    # Check for runner import
    content = path.read_text()
    runner_imports = [
        "from src.training.regal_training_runner import",
        "from src.training.wrap_training_entrypoint import",
        "import src.training.regal_training_runner",
    ]
    
    has_runner = any(imp in content for imp in runner_imports)
    
    if not has_runner:
        print(f"[FAIL] {path.name}: Does not use RegalTrainingRunner")
        print(f"       Add runner import or add to TRAINING_SCRIPT_ALLOWLIST with reason")
        return False
    
    print(f"[OK] {path.name}: Uses RegalTrainingRunner")
    return True


def main():
    """CLI for checking training script compliance."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Check training script compliance")
    parser.add_argument("--scripts-dir", type=str, default="scripts",
                        help="Directory containing training scripts")
    parser.add_argument("--pattern", type=str, default="train_*.py",
                        help="Glob pattern for training scripts")
    args = parser.parse_args()
    
    scripts = glob.glob(f"{args.scripts_dir}/{args.pattern}")
    
    if not scripts:
        print(f"No scripts found matching {args.scripts_dir}/{args.pattern}")
        sys.exit(1)
    
    all_compliant = True
    for script in sorted(scripts):
        if not check_training_script_compliance(script):
            all_compliant = False
    
    if not all_compliant:
        print(f"\n[FAIL] Some training scripts are not compliant")
        sys.exit(1)
    
    print(f"\n[OK] All {len(scripts)} training scripts are compliant")


if __name__ == "__main__":
    main()
