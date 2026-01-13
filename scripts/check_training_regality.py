#!/usr/bin/env python3
"""
CI compliance checker for training script regality.

Ensures all training entrypoints (scripts/train_*.py) either:
1. Use the @regal_training decorator or RegalTrainingRunner, OR
2. Are explicitly allowlisted with a documented reason

This script is intended to be run in CI as a hard gate.
Exit code 0: All scripts compliant
Exit code 1: Non-compliant scripts found
"""
import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Allowlist: scripts that are explicitly permitted to NOT use RegalTrainingRunner
# Each entry must have a reason documented here
#
# MIGRATION STATUS:
# - Phase 10 complete: 4 trainers wrapped (hydra, sac_ontology, offline, skill)
# - Pending migration: 23 trainers (listed below with PENDING_MIGRATION tag)
# - Goal: reduce to 0 pending by end of week
#
TRAINING_SCRIPT_ALLOWLIST: Dict[str, str] = {
    # =========================================================================
    # Already compliant (use @regal_training or RegalTrainingRunner)
    # These should NOT be in the allowlist - but we leave as comment for tracking
    # =========================================================================
    # "train_hydra_policy.py": "COMPLIANT - wrapped in Phase 9",
    # "train_sac_with_ontology_logging.py": "COMPLIANT - wrapped in Phase 9",
    # "train_offline_policy.py": "COMPLIANT - wrapped in Phase 9",
    # "train_skill_policies.py": "COMPLIANT - wrapped in Phase 9",
    
    # =========================================================================
    # PENDING_MIGRATION: Scripts awaiting @regal_training wrapper
    # Priority: Wrap in order listed (most used first)
    # =========================================================================
    "train_behaviour_model.py": "PENDING_MIGRATION: High priority - behaviour cloning core",
    "train_world_model_from_datapacks.py": "PENDING_MIGRATION: High priority - world model",
    "train_stable_world_model.py": "PENDING_MIGRATION: High priority - stable world model",
    "train_trust_aware_world_model.py": "PENDING_MIGRATION: High priority - trust-aware WM",
    "train_horizon_agnostic_world_model.py": "PENDING_MIGRATION: High priority - horizon WM",
    "train_latent_diffusion.py": "PENDING_MIGRATION: Medium priority - diffusion model",
    "train_orchestration_transformer.py": "PENDING_MIGRATION: Medium priority - orch transformer",
    "train_orchestration_transformer_v1_curriculum.py": "PENDING_MIGRATION: Medium - orch curriculum",
    "train_high_level_controller.py": "PENDING_MIGRATION: Medium priority - HLC",
    "train_meta_transformer_synthetic.py": "PENDING_MIGRATION: Medium priority - meta transformer",
    "train_motion_hierarchy_node.py": "PENDING_MIGRATION: Medium priority - motion hierarchy",
    "train_aligned_encoder.py": "PENDING_MIGRATION: Medium priority - visual encoder",
    "train_vision_backbone.py": "PENDING_MIGRATION: Medium priority - vision backbone",
    "train_trust_net.py": "PENDING_MIGRATION: Medium priority - trust network",
    "train_trust_weighted_offline.py": "PENDING_MIGRATION: Medium priority - trust-weighted",
    "train_vla_recap_offline.py": "PENDING_MIGRATION: Medium priority - VLA offline",
    "train_offline_with_local_synth.py": "PENDING_MIGRATION: Medium - offline local synth",
    "train_energy_profile_policy.py": "PENDING_MIGRATION: Low priority - energy profile",
    "train_energy_response_model.py": "PENDING_MIGRATION: Low priority - energy response",
    "train_synth_lambda_controller.py": "PENDING_MIGRATION: Low priority - synth lambda",
    "train_w_econ_lattice.py": "PENDING_MIGRATION: Low priority - econ lattice",
    "train_w_econ_lattice_from_J.py": "PENDING_MIGRATION: Low priority - econ lattice J",
    "train_ggds_on_lsd_vector_scenes.py": "PENDING_MIGRATION: Low priority - GGDS/LSD",
    
    # =========================================================================
    # Legacy scripts (not for production, no migration planned)
    # =========================================================================
    "train_vision_backbone_real.py": "LEGACY: Vision-only script, no RL loop, deprecated",
    "train_sima2_segmenter.py": "LEGACY: Perception script, no RL loop, deprecated",
    "train_spatial_rnn.py": "LEGACY: Perception script, no RL loop, deprecated",
    "train_tiny_demo.py": "LEGACY: Minimal demo for onboarding, no production use",
    "train_knob_model.py": "LEGACY: Meta-training script, regality at different layer",
}

# Pattern to detect RegalTrainingRunner usage
REGAL_PATTERNS = [
    r"RegalTrainingRunner",
    r"@regal_training",
    r"from src\.training\.regal_training_runner import",
    r"from src\.training\.wrap_training_entrypoint import",
    r"wrap_training_entrypoint",
]


def check_script_compliance(script_path: Path) -> Tuple[bool, str]:
    """Check if a training script uses RegalTrainingRunner.
    
    Args:
        script_path: Path to the training script
        
    Returns:
        Tuple of (is_compliant, reason)
    """
    script_name = script_path.name
    
    # Check allowlist first
    if script_name in TRAINING_SCRIPT_ALLOWLIST:
        return True, f"Allowlisted: {TRAINING_SCRIPT_ALLOWLIST[script_name]}"
    
    # Read script content
    try:
        content = script_path.read_text()
    except Exception as e:
        return False, f"Could not read script: {e}"
    
    # Check for regal patterns
    for pattern in REGAL_PATTERNS:
        if re.search(pattern, content):
            return True, f"Uses regality wrapper (matched: {pattern})"
    
    # No regal pattern found
    return False, "Does not use RegalTrainingRunner or @regal_training decorator"


def find_training_scripts(scripts_dir: Path) -> List[Path]:
    """Find all training scripts in the scripts directory."""
    return sorted(scripts_dir.glob("train_*.py"))


def run_compliance_check(
    scripts_dir: Path,
    verbose: bool = False,
    strict: bool = True,
) -> Tuple[bool, List[Tuple[Path, str]]]:
    """Run compliance check on all training scripts.
    
    Args:
        scripts_dir: Path to scripts directory
        verbose: Print detailed output
        strict: Fail on any non-compliant script
        
    Returns:
        Tuple of (all_passed, list of (script, reason) for failures)
    """
    training_scripts = find_training_scripts(scripts_dir)
    
    if verbose:
        print(f"Found {len(training_scripts)} training scripts")
        print(f"Allowlist: {len(TRAINING_SCRIPT_ALLOWLIST)} scripts")
        print()
    
    failures: List[Tuple[Path, str]] = []
    successes: List[Tuple[Path, str]] = []
    
    for script in training_scripts:
        is_compliant, reason = check_script_compliance(script)
        
        if is_compliant:
            successes.append((script, reason))
            if verbose:
                print(f"✓ {script.name}: {reason}")
        else:
            failures.append((script, reason))
            if verbose:
                print(f"✗ {script.name}: {reason}")
    
    if verbose:
        print()
        print(f"Summary: {len(successes)} compliant, {len(failures)} non-compliant")
    
    all_passed = len(failures) == 0
    return all_passed, failures


def main():
    parser = argparse.ArgumentParser(
        description="Check training script regality compliance"
    )
    parser.add_argument(
        "--scripts-dir", type=str, default="scripts",
        help="Path to scripts directory (default: scripts)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--strict", action="store_true", default=True,
        help="Exit with error on any non-compliant script (default: True)"
    )
    parser.add_argument(
        "--warn-only", action="store_true",
        help="Print warnings but exit with success (grace period mode)"
    )
    parser.add_argument(
        "--json-output", type=str,
        help="Write JSON report to file"
    )
    args = parser.parse_args()
    
    # Find repo root
    script_path = Path(__file__).parent
    repo_root = script_path.parent if script_path.name == "scripts" else script_path
    scripts_dir = repo_root / args.scripts_dir
    
    if not scripts_dir.exists():
        print(f"ERROR: Scripts directory not found: {scripts_dir}")
        sys.exit(1)
    
    all_passed, failures = run_compliance_check(
        scripts_dir, 
        verbose=args.verbose,
        strict=not args.warn_only,
    )
    
    # JSON output
    if args.json_output:
        import json
        report = {
            "all_passed": all_passed,
            "failures": [
                {"script": str(path.name), "reason": reason}
                for path, reason in failures
            ],
            "allowlist": TRAINING_SCRIPT_ALLOWLIST,
        }
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)
    
    if not all_passed:
        print()
        print("="*60)
        print("TRAINING REGALITY COMPLIANCE CHECK FAILED")
        print("="*60)
        print()
        print("The following training scripts do not use RegalTrainingRunner:")
        print()
        for path, reason in failures:
            print(f"  - {path.name}: {reason}")
        print()
        print("To fix this:")
        print("  1. Add @regal_training decorator to your training function, OR")
        print("  2. Use RegalTrainingRunner in your script, OR")
        print("  3. Add the script to TRAINING_SCRIPT_ALLOWLIST with a reason")
        print()
        
        if args.warn_only:
            print("[WARN MODE] Exiting with success (grace period)")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        if args.verbose:
            print()
            print("All training scripts are regality-compliant ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
