#!/usr/bin/env python3
"""Artifacts-only causal replay script.

Loads all artifacts from a run directory and recomputes:
- Regal evaluations at POST_AUDIT
- Deploy gate inputs + decision
- Verification report

Asserts that recomputed SHAs match manifest fields (or prints diffs).

Usage:
    python scripts/replay_run_from_artifacts.py --run-dir artifacts/stage6/<run_id>
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.contracts.schemas import (
    RunManifestV1,
    RegalGatesV1,
    RegalContextV1,
    SelectionManifestV1,
    OrchestratorStateV1,
    TrajectoryAuditV1,
    EconTensorV1,
)
from src.regal.regal_evaluator import evaluate_regals, RegalPhaseV1, write_ledger_regal
from src.valuation.valuation_verifier import verify_run
from src.utils.config_digest import sha256_file, sha256_json


def load_artifact(path: Path, model_class=None) -> Optional[Any]:
    """Load JSON artifact, optionally parse into pydantic model."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if model_class:
            return model_class.model_validate(data)
        return data
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None


def replay_run(run_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Replay a run from artifacts only.
    
    Args:
        run_dir: Path to run output directory
        verbose: Print detailed progress
        
    Returns:
        Dict with replay results and SHA comparisons
    """
    output_path = Path(run_dir)
    results = {
        "run_dir": str(output_path),
        "sha_comparisons": {},
        "all_match": True,
        "errors": [],
    }
    
    if verbose:
        print(f"\n=== Causal Replay: {output_path.name} ===\n")
    
    # 1. Load manifest
    manifest = load_artifact(output_path / "run_manifest.json", RunManifestV1)
    if not manifest:
        results["errors"].append("run_manifest.json not found or invalid")
        results["all_match"] = False
        return results
    
    if verbose:
        print(f"Loaded manifest: run_id={manifest.run_id}")
    
    # 2. Load artifacts for replay
    selection_manifest = load_artifact(output_path / "selection_manifest.json", SelectionManifestV1)
    orchestrator_state = load_artifact(output_path / "orchestrator_state.json", OrchestratorStateV1)
    trajectory_audit = load_artifact(output_path / "trajectory_audit.json", TrajectoryAuditV1)
    econ_tensor = load_artifact(output_path / "econ_tensor.json", EconTensorV1)
    regal_config = load_artifact(output_path / "regal_config.json", RegalGatesV1)
    regal_context = load_artifact(output_path / "regal_context.json", RegalContextV1)
    
    if verbose:
        print(f"  selection_manifest: {'✓' if selection_manifest else '✗'}")
        print(f"  orchestrator_state: {'✓' if orchestrator_state else '✗'}")
        print(f"  trajectory_audit: {'✓' if trajectory_audit else '✗'}")
        print(f"  econ_tensor: {'✓' if econ_tensor else '✗'}")
        print(f"  regal_config: {'✓' if regal_config else '✗'}")
    
    # 3. Recompute regals if config exists
    if regal_config:
        if verbose:
            print("\n--- Recomputing Regals at POST_AUDIT ---")
        
        try:
            recomputed_regal = evaluate_regals(
                config=regal_config,
                phase=RegalPhaseV1.POST_AUDIT,
                context=regal_context,
                trajectory_audit=trajectory_audit,
                econ_tensor=econ_tensor,
                selection_manifest=selection_manifest,
                orchestrator_state=orchestrator_state,
            )
            
            # Compare SHA
            if manifest.regal_report_sha:
                recomputed_sha = recomputed_regal.sha256()
                match = recomputed_sha == manifest.regal_report_sha
                results["sha_comparisons"]["regal_report_sha"] = {
                    "expected": manifest.regal_report_sha,
                    "actual": recomputed_sha,
                    "match": match,
                }
                if not match:
                    results["all_match"] = False
                if verbose:
                    print(f"  regal_report_sha: {'✓ MATCH' if match else '✗ MISMATCH'}")
                    if not match:
                        print(f"    expected: {manifest.regal_report_sha[:16]}...")
                        print(f"    actual:   {recomputed_sha[:16]}...")
        except Exception as e:
            results["errors"].append(f"Regal recomputation failed: {e}")
            if verbose:
                print(f"  [ERROR] Regal recomputation failed: {e}")
    
    # 4. Verify selection manifest SHA
    selection_path = output_path / "selection_manifest.json"
    if selection_path.exists() and manifest.selection_manifest_sha:
        computed_sha = sha256_file(str(selection_path))
        match = computed_sha == manifest.selection_manifest_sha
        results["sha_comparisons"]["selection_manifest_sha"] = {
            "expected": manifest.selection_manifest_sha,
            "actual": computed_sha,
            "match": match,
        }
        if not match:
            results["all_match"] = False
        if verbose:
            print(f"  selection_manifest_sha: {'✓ MATCH' if match else '✗ MISMATCH'}")
    
    # 5. Verify orchestrator state SHA
    orch_path = output_path / "orchestrator_state.json"
    if orch_path.exists() and manifest.orchestrator_state_sha:
        computed_sha = sha256_file(str(orch_path))
        match = computed_sha == manifest.orchestrator_state_sha
        results["sha_comparisons"]["orchestrator_state_sha"] = {
            "expected": manifest.orchestrator_state_sha,
            "actual": computed_sha,
            "match": match,
        }
        if not match:
            results["all_match"] = False
        if verbose:
            print(f"  orchestrator_state_sha: {'✓ MATCH' if match else '✗ MISMATCH'}")
    
    # 5b. Verify regality_thresholds_sha (Phase 10: policy as causal input)
    # Replay REFUSES runs missing thresholds if manifest has SHA
    thresholds_path = output_path / "regality_thresholds.json"
    if getattr(manifest, 'regality_thresholds_sha', None):
        if thresholds_path.exists():
            computed_sha = sha256_file(str(thresholds_path))
            match = computed_sha == manifest.regality_thresholds_sha
            results["sha_comparisons"]["regality_thresholds_sha"] = {
                "expected": manifest.regality_thresholds_sha,
                "actual": computed_sha,
                "match": match,
            }
            if not match:
                results["all_match"] = False
                results["errors"].append("regality_thresholds_sha mismatch - policy may have changed")
            if verbose:
                print(f"  regality_thresholds_sha: {'✓ MATCH' if match else '✗ MISMATCH'}")
        else:
            # HARD FAIL: manifest has SHA but file missing
            results["all_match"] = False
            results["errors"].append("regality_thresholds.json missing but SHA in manifest - replay refuses")
            if verbose:
                print(f"  regality_thresholds_sha: ✗ FILE MISSING (replay refuses)")
    
    # 6. Run full verification
    if verbose:
        print("\n--- Running Full Verification ---")
    
    verification_report = verify_run(str(output_path))
    results["verification_passed"] = verification_report.all_passed
    results["verification_checks"] = verification_report.check_count
    results["verification_failures"] = [
        {"check_id": c.check_id, "message": c.message}
        for c in verification_report.checks if not c.passed
    ]
    
    if verbose:
        print(f"  all_passed: {verification_report.all_passed}")
        print(f"  checks: {verification_report.passed_count}/{verification_report.check_count} passed")
        if not verification_report.all_passed:
            print("  failures:")
            for c in verification_report.checks:
                if not c.passed:
                    print(f"    - {c.check_id}: {c.message[:60]}...")
    
    # 7. Summary
    if verbose:
        print("\n=== Replay Summary ===")
        print(f"SHA comparisons: {len(results['sha_comparisons'])}")
        print(f"All SHAs match: {results['all_match']}")
        print(f"Verification passed: {verification_report.all_passed}")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for e in results['errors']:
                print(f"  - {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Replay run from artifacts only")
    parser.add_argument("--run-dir", required=True, help="Path to run output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    results = replay_run(args.run_dir, verbose=not args.json)
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with error if replay failed
    if not results["all_match"] or not results.get("verification_passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
