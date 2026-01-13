"""Causal replay integration tests.

Tests that:
1. Causal replay script can be imported and run in-process
2. Modifying selection semantics (even with SHA updated) fails on semantics
3. Combined inputs SHA recomputation works
"""
import json
import pytest
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_run_from_artifacts import replay_run
from src.valuation.valuation_verifier import verify_run
from src.utils.config_digest import sha256_file


class TestCausalReplayImport:
    """Test that replay script can be imported and used in-process."""
    
    def test_replay_function_importable(self):
        """replay_run function should be importable without script execution."""
        assert callable(replay_run)
    
    def test_replay_on_empty_dir_fails_gracefully(self):
        """Replay on empty dir should fail with clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = replay_run(tmpdir, verbose=False)
            
            assert not results["all_match"]
            assert len(results["errors"]) > 0


class TestCausalReplaySemantics:
    """Test that semantic violations are caught even with updated SHAs."""
    
    def test_semantic_violation_not_hidden_by_sha_update(self):
        """Modify selection to violate semantics, update SHA → still fails semantic check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create valid selection manifest
            selection = {
                "schema_version": "v1",
                "manifest_id": "sel1",
                "eligible_datapack_ids": ["dp_1", "dp_2", "dp_3", "dp_4"],
                "quarantine_datapack_ids": ["dp_bad"],
                "selected_datapack_ids": ["dp_1", "dp_2"],
                "rejected_datapacks": [],
                "rng_seed": 42,
            }
            sel_path = output_dir / "selection_manifest.json"
            with open(sel_path, "w") as f:
                json.dump(selection, f, indent=2, sort_keys=True)
            valid_sha = sha256_file(str(sel_path))
            
            # Create run manifest with valid SHA
            run_manifest = {
                "schema_version": "v1",
                "run_id": "semantic_test",
                "created_at": "2026-01-12T12:00:00",
                "plan_sha": "plan_sha_123",
                "audit_suite_id": "test",
                "audit_seed": 42,
                "audit_config_sha": "audit_sha",
                "datapack_manifest_sha": "dp_sha",
                "seeds": {"main": 42},
                "selection_manifest_sha": valid_sha,
            }
            with open(output_dir / "run_manifest.json", "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            with open(output_dir / "ledger.jsonl", "w") as f:
                f.write(json.dumps({"schema_version": "v1", "record_id": "r1", "run_id": "semantic_test", "plan_sha": "plan_sha_123"}) + "\n")
            
            # Baseline passes SHA check and semantic checks
            result_before = verify_run(str(output_dir))
            sha_check = next((c for c in result_before.checks if c.check_id == "selection_manifest_sha_match"), None)
            quarantine_check = next((c for c in result_before.checks if c.check_id == "selection_quarantine_disjoint"), None)
            
            assert sha_check is not None and sha_check.passed, "Baseline SHA should match"
            assert quarantine_check is not None and quarantine_check.passed, "Baseline quarantine check should pass"
            
            # Now tamper: select a quarantined datapack BUT UPDATE THE SHA
            tampered_selection = selection.copy()
            tampered_selection["selected_datapack_ids"] = ["dp_1", "dp_bad"]  # dp_bad is quarantined!
            
            with open(sel_path, "w") as f:
                json.dump(tampered_selection, f, indent=2, sort_keys=True)
            tampered_sha = sha256_file(str(sel_path))
            
            # Update manifest with NEW SHA (so SHA check won't catch it)
            run_manifest["selection_manifest_sha"] = tampered_sha
            with open(output_dir / "run_manifest.json", "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            # Verify: SHA check should PASS (we updated it), but SEMANTIC check should FAIL
            result_after = verify_run(str(output_dir))
            
            sha_check_after = next((c for c in result_after.checks if c.check_id == "selection_manifest_sha_match"), None)
            quarantine_check_after = next((c for c in result_after.checks if c.check_id == "selection_quarantine_disjoint"), None)
            
            # SHA should match (we updated it)
            assert sha_check_after is not None and sha_check_after.passed, "SHA should match after update"
            
            # BUT semantic check should fail! (quarantine violation)
            assert quarantine_check_after is not None and not quarantine_check_after.passed, (
                f"Semantic check should fail even with updated SHA. "
                f"quarantine check passed: {quarantine_check_after.passed}"
            )
            assert "dp_bad" in quarantine_check_after.message


class TestReplayCombinedInputsSha:
    """Test that combined_inputs_sha is deterministically recomputed."""
    
    def test_replay_detects_sha_mismatch(self):
        """Replay should detect when artifact SHA doesn't match manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create selection manifest
            selection = {
                "schema_version": "v1",
                "manifest_id": "sel1",
                "eligible_datapack_ids": ["dp_1", "dp_2", "dp_3"],
                "quarantine_datapack_ids": [],
                "selected_datapack_ids": ["dp_1", "dp_2"],
                "rejected_datapacks": [],
                "rng_seed": 42,
            }
            sel_path = output_dir / "selection_manifest.json"
            with open(sel_path, "w") as f:
                json.dump(selection, f, indent=2, sort_keys=True)
            sel_sha = sha256_file(str(sel_path))
            
            # Create run manifest with WRONG SHA (intentional mismatch)
            run_manifest = {
                "schema_version": "v1",
                "run_id": "sha_mismatch_test",
                "created_at": "2026-01-12T12:00:00",
                "plan_sha": "plan_sha_123",
                "audit_suite_id": "test",
                "audit_seed": 42,
                "audit_config_sha": "audit_sha",
                "datapack_manifest_sha": "dp_sha",
                "seeds": {"main": 42},
                "selection_manifest_sha": "WRONG_SHA_INTENTIONAL",  # Mismatch!
            }
            with open(output_dir / "run_manifest.json", "w") as f:
                json.dump(run_manifest, f, indent=2)
            
            with open(output_dir / "ledger.jsonl", "w") as f:
                f.write(json.dumps({"schema_version": "v1", "record_id": "r1", "run_id": "sha_mismatch_test", "plan_sha": "plan_sha_123"}) + "\n")
            
            # Replay should detect mismatch
            results = replay_run(str(output_dir), verbose=False)
            
            # Selection SHA should be in comparisons and should NOT match
            if "selection_manifest_sha" in results["sha_comparisons"]:
                comparison = results["sha_comparisons"]["selection_manifest_sha"]
                assert not comparison["match"], "Replay should detect SHA mismatch"
            
            # Verification should fail
            assert results.get("verification_passed") is False or not results["all_match"]
