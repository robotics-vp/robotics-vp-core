"""Tamper tests for selection manifest and verifier enforcement.

These tests prove that mutation of selection_manifest.json (with everything else
constant) causes verify_run() to FAIL with an attributable check ID.
"""
import json
import pytest
import tempfile
from pathlib import Path

from src.valuation.valuation_verifier import verify_run
from src.utils.config_digest import sha256_json


def create_valid_run_artifacts(output_dir: Path) -> dict:
    """Create a valid set of run artifacts for testing.
    
    Returns the selection manifest dict for later tampering.
    """
    # Create run_manifest.json
    selection_manifest = {
        "schema_version": "v1",
        "manifest_id": "test_manifest",
        "created_at": "2026-01-12T12:00:00",
        "eligible_datapack_ids": ["dp_001", "dp_002", "dp_003", "dp_004"],
        "quarantine_datapack_ids": ["dp_bad_001"],
        "selected_datapack_ids": ["dp_001", "dp_002"],
        "rejected_datapacks": [
            {"id": "dp_003", "reason": "low_quality"},
            {"id": "dp_004", "reason": "duplicate"},
        ],
        "rng_seed": 42,
        "sampler_config_sha": "abc123",
    }
    
    # Write selection_manifest.json
    sel_path = output_dir / "selection_manifest.json"
    with open(sel_path, "w") as f:
        json.dump(selection_manifest, f, indent=2, sort_keys=True)
    
    # Use sha256_file (same as verifier) not sha256_json
    from src.utils.config_digest import sha256_file
    selection_sha = sha256_file(str(sel_path))
    
    # Create run_manifest.json referencing the selection manifest
    run_manifest = {
        "schema_version": "v1",
        "run_id": "tamper_test_run",
        "created_at": "2026-01-12T12:00:00",
        "plan_sha": "plan_sha_123",
        # Required fields
        "audit_suite_id": "test_audit",
        "audit_seed": 42,
        "audit_config_sha": "audit_config_sha_123",
        "datapack_manifest_sha": "datapack_manifest_sha_123",
        "seeds": {"main": 42, "env": 42},
        # Selection manifest (the one we're testing)
        "selection_manifest_sha": selection_sha,
        # Weight provenance = training run
        "baseline_weights_sha": "weights_before",
        "final_weights_sha": "weights_after",  # Different = training run
        # Optional fields
        "regal_config_sha": None,
        "regal_report_sha": None,
        "trajectory_audit_sha": None,
        "orchestrator_state_sha": None,
        "quarantine_manifest_sha": None,
        "deploy_gate_decision_sha": None,
        "probe_report_sha": None,
        "regal_context_sha": None,
    }
    
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
    
    # Create ledger.jsonl (minimal valid)
    ledger_entry = {
        "schema_version": "v1",
        "record_id": "record_001",
        "run_id": "tamper_test_run",
        "plan_sha": "plan_sha_123",
        "regal": None,
        "regal_degraded": False,
        "allow_deploy": True,
    }
    with open(output_dir / "ledger.jsonl", "w") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    
    return selection_manifest


class TestSelectionManifestTamper:
    """Test that tampering with selection_manifest.json causes verify_run() to fail."""
    
    def test_tamper_selection_manifest_sha_mismatch(self):
        """Mutate selection_manifest.json → verify_run() MUST fail with specific check ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original_manifest = create_valid_run_artifacts(output_dir)
            
            # Verify SHA check passes with original (ignore ledger failures)
            result_before = verify_run(str(output_dir))
            sha_check_before = next((c for c in result_before.checks if c.check_id == "selection_manifest_sha_match"), None)
            assert sha_check_before is not None, "SHA check should exist"
            assert sha_check_before.passed, f"Baseline SHA should match: {sha_check_before.message}"
            
            # Now tamper: change selected_datapack_ids
            tampered_manifest = original_manifest.copy()
            tampered_manifest["selected_datapack_ids"] = ["dp_001", "dp_003"]  # Changed dp_002 → dp_003
            
            sel_path = output_dir / "selection_manifest.json"
            with open(sel_path, "w") as f:
                json.dump(tampered_manifest, f, indent=2, sort_keys=True)
            
            # Verify MUST fail with SHA mismatch
            result_after = verify_run(str(output_dir))
            
            # Find the specific failing check
            sha_mismatch_check = None
            for check in result_after.checks:
                if check.check_id == "selection_manifest_sha_match" and not check.passed:
                    sha_mismatch_check = check
                    break
            
            assert sha_mismatch_check is not None, (
                f"Expected 'selection_manifest_sha_match' check to fail. "
                f"Failed checks: {[c.check_id for c in result_after.checks if not c.passed]}"
            )
            assert "mismatch" in sha_mismatch_check.message.lower()
    
    def test_tamper_quarantine_violation(self):
        """Select a quarantined datapack → verify_run() MUST fail with quarantine check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original_manifest = create_valid_run_artifacts(output_dir)
            
            # Tamper: add quarantined ID to selected
            tampered_manifest = original_manifest.copy()
            tampered_manifest["selected_datapack_ids"] = ["dp_001", "dp_bad_001"]  # dp_bad_001 is quarantined
            
            # Update selection file and run_manifest SHA to match
            sel_path = output_dir / "selection_manifest.json"
            with open(sel_path, "w") as f:
                json.dump(tampered_manifest, f, indent=2)
            
            # Update run_manifest with new SHA (so SHA check passes, but semantic check fails)
            tampered_sha = sha256_json(tampered_manifest)
            manifest_path = output_dir / "run_manifest.json"
            with open(manifest_path, "r") as f:
                run_manifest = json.load(f)
            run_manifest["selection_manifest_sha"] = tampered_sha
            with open(manifest_path, "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            # Verify MUST fail
            result = verify_run(str(output_dir))
            assert not result.all_passed, "Quarantine violation should cause failure"
            
            # Find the specific failing check
            quarantine_check = None
            for check in result.checks:
                if check.check_id == "selection_quarantine_disjoint" and not check.passed:
                    quarantine_check = check
                    break
            
            assert quarantine_check is not None, (
                f"Expected 'selection_quarantine_disjoint' check to fail. "
                f"Failed checks: {[c.check_id for c in result.checks if not c.passed]}"
            )
            assert "dp_bad_001" in quarantine_check.message or "quarantine" in quarantine_check.message.lower()
    
    def test_tamper_selected_not_subset_of_eligible(self):
        """Select a non-eligible datapack → verify_run() MUST fail with subset check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original_manifest = create_valid_run_artifacts(output_dir)
            
            # Tamper: add non-eligible ID to selected
            tampered_manifest = original_manifest.copy()
            tampered_manifest["selected_datapack_ids"] = ["dp_001", "dp_phantom"]  # dp_phantom not in eligible
            
            # Update with correct SHA so semantic check triggers
            sel_path = output_dir / "selection_manifest.json"
            with open(sel_path, "w") as f:
                json.dump(tampered_manifest, f, indent=2)
            
            tampered_sha = sha256_json(tampered_manifest)
            manifest_path = output_dir / "run_manifest.json"
            with open(manifest_path, "r") as f:
                run_manifest = json.load(f)
            run_manifest["selection_manifest_sha"] = tampered_sha
            with open(manifest_path, "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            # Verify MUST fail
            result = verify_run(str(output_dir))
            assert not result.all_passed, "Selected not subset of eligible should cause failure"
            
            # Find the specific failing check
            subset_check = None
            for check in result.checks:
                if check.check_id == "selection_selected_subset_eligible" and not check.passed:
                    subset_check = check
                    break
            
            assert subset_check is not None, (
                f"Expected 'selection_selected_subset_eligible' check to fail. "
                f"Failed checks: {[c.check_id for c in result.checks if not c.passed]}"
            )
            assert "dp_phantom" in subset_check.message


class TestOrchestratorStateTamper:
    """Test that tampering with orchestrator_state.json causes verify_run() to fail."""
    
    def test_tamper_orchestrator_state_sha_mismatch(self):
        """Mutate orchestrator_state.json → verify_run() MUST fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create orchestrator state
            orch_state = {
                "schema_version": "v1",
                "step": 100,
                "failure_counts": {"spec_guardian": 1},
                "patience_counters": {},
                "clamp_decisions": [],
                "noop_decisions": [],
                "cooldown_remaining": {},
                "backoff_multipliers": {},
                "applied_knob_deltas": [],
            }
            
            orch_path = output_dir / "orchestrator_state.json"
            with open(orch_path, "w") as f:
                json.dump(orch_state, f, indent=2)
            orch_sha = sha256_json(orch_state)
            
            # Create run manifest
            run_manifest = {
                "schema_version": "v1",
                "run_id": "orch_tamper_test",
                "created_at": "2026-01-12T12:00:00",
                "plan_sha": "plan_sha_123",
                # Required fields
                "audit_suite_id": "test_audit",
                "audit_seed": 42,
                "audit_config_sha": "audit_config_sha_123",
                "datapack_manifest_sha": "datapack_manifest_sha_123",
                "seeds": {"main": 42, "env": 42},
                # Orchestrator state (the one we're testing)
                "orchestrator_state_sha": orch_sha,
                "baseline_weights_sha": None,
                "final_weights_sha": None,
                "selection_manifest_sha": None,
                "regal_config_sha": None,
                "regal_report_sha": None,
                "trajectory_audit_sha": None,
                "quarantine_manifest_sha": None,
                "deploy_gate_decision_sha": None,
                "probe_report_sha": None,
                "regal_context_sha": None,
            }
            
            with open(output_dir / "run_manifest.json", "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            # Create ledger
            ledger_entry = {
                "schema_version": "v1",
                "record_id": "record_001",
                "run_id": "orch_tamper_test",
                "plan_sha": "plan_sha_123",
                "regal": None,
                "regal_degraded": False,
                "allow_deploy": True,
            }
            with open(output_dir / "ledger.jsonl", "w") as f:
                f.write(json.dumps(ledger_entry) + "\n")
            
            # Verify passes initially
            result_before = verify_run(str(output_dir))
            assert result_before.all_passed, f"Baseline should pass: {[c.message for c in result_before.checks if not c.passed]}"
            
            # Tamper: change failure_counts
            tampered_state = orch_state.copy()
            tampered_state["failure_counts"] = {"spec_guardian": 5, "reward_integrity": 2}
            
            with open(orch_path, "w") as f:
                json.dump(tampered_state, f, indent=2)
            
            # Verify MUST fail
            result_after = verify_run(str(output_dir))
            assert not result_after.all_passed, "Tampered orchestrator state should cause failure"
            
            # Find the specific failing check
            sha_mismatch = None
            for check in result_after.checks:
                if check.check_id == "orchestrator_state_sha_match" and not check.passed:
                    sha_mismatch = check
                    break
            
            assert sha_mismatch is not None, (
                f"Expected 'orchestrator_state_sha_match' check to fail. "
                f"Failed checks: {[c.check_id for c in result_after.checks if not c.passed]}"
            )
