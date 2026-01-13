"""Tamper tests for orchestrator state semantic enforcement.

These tests prove that:
1. SHA mismatch of orchestrator_state.json is detected
2. Semantic violations (negative counters) are caught even if SHA matches
3. Knob deltas cross-validation works
"""
import json
import pytest
import tempfile
from pathlib import Path

from src.valuation.valuation_verifier import verify_run
from src.utils.config_digest import sha256_file


def create_valid_orchestrator_run(output_dir: Path, is_training: bool = True) -> dict:
    """Create a valid run with orchestrator state for testing."""
    # Create orchestrator state
    orch_state = {
        "schema_version": "v1",
        "step": 100,
        "failure_counts": {"spec_guardian": 1, "reward_integrity": 0},
        "patience_counters": {"spec_guardian": 2},
        "clamp_decisions": [],
        "noop_decisions": [],
        "cooldown_remaining": {"spec_guardian": 0},
        "backoff_multipliers": {},
        "applied_knob_deltas": [
            {"knob_id": "task_weight_grasp", "delta": 0.1, "reason": "success_rate_low"}
        ],
    }
    
    orch_path = output_dir / "orchestrator_state.json"
    with open(orch_path, "w") as f:
        json.dump(orch_state, f, indent=2, sort_keys=True)
    orch_sha = sha256_file(str(orch_path))
    
    # Create applied knob deltas file (must match orchestrator)
    deltas_path = output_dir / "applied_knob_deltas.json"
    with open(deltas_path, "w") as f:
        json.dump(orch_state["applied_knob_deltas"], f, indent=2, sort_keys=True)
    
    # Create run manifest
    run_manifest = {
        "schema_version": "v1",
        "run_id": "orch_test_run",
        "created_at": "2026-01-12T12:00:00",
        "plan_sha": "plan_sha_123",
        # Required fields
        "audit_suite_id": "test_audit",
        "audit_seed": 42,
        "audit_config_sha": "audit_config_sha_123",
        "datapack_manifest_sha": "datapack_manifest_sha_123",
        "seeds": {"main": 42, "env": 42},
        # Orchestrator state
        "orchestrator_state_sha": orch_sha,
        # Training run markers
        "baseline_weights_sha": "weights_before" if is_training else None,
        "final_weights_sha": "weights_after" if is_training else None,
    }
    
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
    
    # Create minimal ledger
    ledger_entry = {
        "schema_version": "v1",
        "record_id": "record_001",
        "run_id": "orch_test_run",
        "plan_sha": "plan_sha_123",
    }
    with open(output_dir / "ledger.jsonl", "w") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    
    return orch_state


class TestOrchestratorStateTamper:
    """Test that tampering with orchestrator_state.json causes verify_run() to fail."""
    
    def test_tamper_sha_mismatch(self):
        """Mutate orchestrator_state.json → verify_run() MUST fail with SHA check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original_state = create_valid_orchestrator_run(output_dir)
            
            # Verify SHA check passes initially
            result_before = verify_run(str(output_dir))
            sha_check = next((c for c in result_before.checks if c.check_id == "orchestrator_state_sha_match"), None)
            assert sha_check is not None, "SHA check should exist"
            assert sha_check.passed, f"Baseline SHA should match: {sha_check.message}"
            
            # Tamper: change failure_counts
            tampered_state = original_state.copy()
            tampered_state["failure_counts"] = {"spec_guardian": 5, "reward_integrity": 2}
            
            orch_path = output_dir / "orchestrator_state.json"
            with open(orch_path, "w") as f:
                json.dump(tampered_state, f, indent=2, sort_keys=True)
            
            # Verify MUST fail with SHA mismatch
            result_after = verify_run(str(output_dir))
            sha_mismatch = next(
                (c for c in result_after.checks if c.check_id == "orchestrator_state_sha_match" and not c.passed),
                None
            )
            
            assert sha_mismatch is not None, (
                f"Expected 'orchestrator_state_sha_match' check to fail. "
                f"Failed checks: {[c.check_id for c in result_after.checks if not c.passed]}"
            )
    
    def test_semantic_violation_negative_counters(self):
        """Negative counter values → verify_run() MUST fail semantic check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original_state = create_valid_orchestrator_run(output_dir)
            
            # Tamper: introduce negative counter, but update SHA to match
            tampered_state = original_state.copy()
            tampered_state["failure_counts"] = {"spec_guardian": -5}  # Invalid!
            
            orch_path = output_dir / "orchestrator_state.json"
            with open(orch_path, "w") as f:
                json.dump(tampered_state, f, indent=2, sort_keys=True)
            tampered_sha = sha256_file(str(orch_path))
            
            # Update manifest with correct tampered SHA
            manifest_path = output_dir / "run_manifest.json"
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            manifest["orchestrator_state_sha"] = tampered_sha
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            
            # Verify MUST fail semantic check (not SHA check)
            result = verify_run(str(output_dir))
            
            semantic_check = next(
                (c for c in result.checks if c.check_id == "orchestrator_state_nonnegative_counters" and not c.passed),
                None
            )
            
            assert semantic_check is not None, (
                f"Expected 'orchestrator_state_nonnegative_counters' to fail. "
                f"Failed checks: {[c.check_id for c in result.checks if not c.passed]}"
            )
            assert "negative" in semantic_check.message.lower()
    
    def test_knob_deltas_mismatch(self):
        """Knob deltas in file differ from orchestrator → verify_run() MUST fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            create_valid_orchestrator_run(output_dir)
            
            # Tamper: change applied_knob_deltas.json only
            deltas_path = output_dir / "applied_knob_deltas.json"
            with open(deltas_path, "w") as f:
                json.dump([{"knob_id": "FAKE", "delta": 999, "reason": "tampered"}], f)
            
            # Verify MUST fail cross-validation
            result = verify_run(str(output_dir))
            
            deltas_check = next(
                (c for c in result.checks if c.check_id == "applied_knob_deltas_match_orchestrator_state" and not c.passed),
                None
            )
            
            assert deltas_check is not None, (
                f"Expected 'applied_knob_deltas_match_orchestrator_state' to fail. "
                f"Failed checks: {[c.check_id for c in result.checks if not c.passed]}"
            )


class TestLedgerRegalTamper:
    """Test that tampering with ledger_regal.json causes verify_run() to fail."""
    
    def test_tamper_ledger_regal_sha(self):
        """Mutate ledger_regal.json → verify_run() MUST fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create ledger_regal.json
            ledger_regal = {
                "schema_version": "v1",
                "phase": "POST_AUDIT",
                "regal_config_sha": "config_sha_123",
                "reports": [],
                "all_passed": True,
                "combined_inputs_sha": "inputs_sha_123",
            }
            
            regal_path = output_dir / "ledger_regal.json"
            with open(regal_path, "w") as f:
                json.dump(ledger_regal, f, indent=2, sort_keys=True)
            regal_sha = sha256_file(str(regal_path))
            
            # Create run manifest
            run_manifest = {
                "schema_version": "v1",
                "run_id": "regal_test",
                "created_at": "2026-01-12T12:00:00",
                "plan_sha": "plan_sha_123",
                "audit_suite_id": "test",
                "audit_seed": 42,
                "audit_config_sha": "audit_config_sha_123",
                "datapack_manifest_sha": "dp_sha_123",
                "seeds": {"main": 42},
                "ledger_regal_sha": regal_sha,
            }
            
            with open(output_dir / "run_manifest.json", "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            with open(output_dir / "ledger.jsonl", "w") as f:
                f.write(json.dumps({"schema_version": "v1", "record_id": "r1", "run_id": "regal_test", "plan_sha": "plan_sha_123"}) + "\n")
            
            # Verify passes initially
            result_before = verify_run(str(output_dir))
            sha_check = next((c for c in result_before.checks if c.check_id == "ledger_regal_sha_match"), None)
            assert sha_check is not None and sha_check.passed, f"Baseline should pass: {sha_check}"
            
            # Tamper
            tampered = ledger_regal.copy()
            tampered["all_passed"] = False
            with open(regal_path, "w") as f:
                json.dump(tampered, f, indent=2, sort_keys=True)
            
            # Verify MUST fail
            result_after = verify_run(str(output_dir))
            sha_mismatch = next(
                (c for c in result_after.checks if c.check_id == "ledger_regal_sha_match" and not c.passed),
                None
            )
            
            assert sha_mismatch is not None, (
                f"Expected 'ledger_regal_sha_match' to fail. "
                f"Failed checks: {[c.check_id for c in result_after.checks if not c.passed]}"
            )


class TestExposureManifestTamper:
    """Test exposure ↔ selection cross-validation."""
    
    def test_exposure_quarantine_violation(self):
        """Quarantined datapack in exposure → verify_run() MUST fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create selection manifest with quarantine
            selection = {
                "schema_version": "v1",
                "manifest_id": "sel1",
                "eligible_datapack_ids": ["dp_1", "dp_2"],
                "quarantine_datapack_ids": ["dp_bad"],
                "selected_datapack_ids": ["dp_1", "dp_2"],
                "rejected_datapacks": [],
                "rng_seed": 42,
            }
            with open(output_dir / "selection_manifest.json", "w") as f:
                json.dump(selection, f, indent=2, sort_keys=True)
            sel_sha = sha256_file(str(output_dir / "selection_manifest.json"))
            
            # Create exposure manifest with QUARANTINED datapack (violation!)
            exposure = {
                "schema_version": "v1",
                "datapack_ids": ["dp_1", "dp_bad"],  # dp_bad is quarantined!
            }
            with open(output_dir / "exposure_manifest.json", "w") as f:
                json.dump(exposure, f, indent=2)
            
            # Create manifest
            run_manifest = {
                "schema_version": "v1",
                "run_id": "exp_test",
                "created_at": "2026-01-12T12:00:00",
                "plan_sha": "plan_sha_123",
                "audit_suite_id": "test",
                "audit_seed": 42,
                "audit_config_sha": "audit_sha",
                "datapack_manifest_sha": "dp_sha",
                "seeds": {"main": 42},
                "selection_manifest_sha": sel_sha,
            }
            with open(output_dir / "run_manifest.json", "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            with open(output_dir / "ledger.jsonl", "w") as f:
                f.write(json.dumps({"schema_version": "v1", "record_id": "r1", "run_id": "exp_test", "plan_sha": "plan_sha_123"}) + "\n")
            
            # Verify MUST fail
            result = verify_run(str(output_dir))
            
            quarantine_check = next(
                (c for c in result.checks if c.check_id == "exposure_quarantine_exclusion" and not c.passed),
                None
            )
            
            assert quarantine_check is not None, (
                f"Expected 'exposure_quarantine_exclusion' to fail. "
                f"Failed checks: {[c.check_id for c in result.checks if not c.passed]}"
            )
            assert "dp_bad" in quarantine_check.message
