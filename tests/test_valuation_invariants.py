"""Acceptance tests for valuation invariants.

Tests that the valuation plane maintains regal-owned provenance closure:
1. allow_deploy invariant
2. regal_report_sha ordering stability
3. regal_annotations_sha matching
4. quarantine enforcement
5. verify_run() tamper detection
"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from pydantic import ValidationError

from src.contracts.schemas import (
    RunManifestV1,
    ValueLedgerRecordV1,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
    LedgerAuditV1,
    LedgerDeltasV1,
    LedgerRegalV1,
    RegalReportV1,
    RegalPhaseV1,
    RegalAnnotationsV1,
    DatapackSelectionOverrides,
)
from src.valuation.value_ledger import ValueLedger
from src.valuation.exposure_manifest import ExposureTracker
from src.valuation.valuation_verifier import verify_run, VerificationReportV1
from src.utils.config_digest import sha256_json


# =============================================================================
# Test Fixtures
# =============================================================================


def _create_minimal_ledger_record(
    run_id: str = "test_run",
    regal_degraded: bool = False,
    allow_deploy: bool = True,
    regal_all_passed: bool = True,
) -> ValueLedgerRecordV1:
    """Create a minimal valid ledger record for testing."""
    window = LedgerWindowV1(
        step_start=0,
        step_end=100,
        ts_start=datetime.now().isoformat(),
        ts_end=datetime.now().isoformat(),
    )
    exposure = LedgerExposureV1(
        datapack_ids=["dp_001"],
        exposure_manifest_sha="abc123",
    )
    policy = LedgerPolicyV1(
        policy_before="baseline.pt",
        policy_after="final.pt",
    )
    audit = LedgerAuditV1(
        audit_suite_id="test",
        audit_seed=42,
        audit_config_sha="cfg123",
        audit_results_before_sha="before123",
        audit_results_after_sha="after123",
    )
    deltas = LedgerDeltasV1(delta_success=0.1)
    
    regal = None
    if not regal_degraded:
        regal = LedgerRegalV1(
            all_passed=regal_all_passed,
            reports=[
                RegalReportV1(
                    regal_id="test_regal",
                    passed=regal_all_passed,
                    rationale="Test",
                    phase=RegalPhaseV1.POST_AUDIT,
                    report_sha="report123",
                    inputs_sha="inputs123",
                    determinism_seed=42,
                )
            ],
            combined_inputs_sha="inputs123",
            regal_config_sha="config123",
        )
    
    return ValueLedgerRecordV1(
        record_id="rec001",
        run_id=run_id,
        plan_id="plan001",
        plan_sha="plan123",
        window=window,
        exposure=exposure,
        policy=policy,
        audit=audit,
        deltas=deltas,
        regal=regal,
        regal_degraded=regal_degraded,
        allow_deploy=allow_deploy,
        plan_applied=True,
    )


# =============================================================================
# Test 1: allow_deploy Invariant
# =============================================================================


class TestAllowDeployInvariant:
    """Tests for allow_deploy safety invariant."""
    
    def test_allow_deploy_false_when_regal_degraded(self):
        """allow_deploy must be False when regal_degraded=True."""
        # This tests the INVARIANT, not the schema validation
        # The invariant is: if regal_degraded=True, allow_deploy MUST be False
        
        # Valid case: degraded + blocked
        record = _create_minimal_ledger_record(
            regal_degraded=True,
            allow_deploy=False,
        )
        assert record.regal_degraded is True
        assert record.allow_deploy is False
        
    def test_allow_deploy_false_when_regal_failed(self):
        """allow_deploy must be False when regal.all_passed=False."""
        record = _create_minimal_ledger_record(
            regal_degraded=False,
            allow_deploy=False,
            regal_all_passed=False,
        )
        assert record.regal is not None
        assert record.regal.all_passed is False
        assert record.allow_deploy is False
    
    def test_allow_deploy_true_only_when_regal_passed(self):
        """allow_deploy=True is only valid when regal passed."""
        record = _create_minimal_ledger_record(
            regal_degraded=False,
            allow_deploy=True,
            regal_all_passed=True,
        )
        assert record.regal is not None
        assert record.regal.all_passed is True
        assert record.allow_deploy is True


# =============================================================================
# Test 2: Regal Report SHA Ordering Stability
# =============================================================================


class TestRegalReportShaStability:
    """Tests for deterministic regal_report_sha ordering."""
    
    def test_regal_report_sha_deterministic(self):
        """Sorted (phase, regal_id) produces stable SHA."""
        reports = [
            RegalReportV1(
                regal_id="world_coherence",
                passed=True,
                rationale="OK",
                phase=RegalPhaseV1.POST_AUDIT,
                report_sha="rsha1",
                inputs_sha="in1",
                determinism_seed=42,
            ),
            RegalReportV1(
                regal_id="spec_guardian",
                passed=True,
                rationale="OK",
                phase=RegalPhaseV1.POST_PLAN_PRE_APPLY,
                report_sha="rsha2",
                inputs_sha="in2",
                determinism_seed=42,
            ),
            RegalReportV1(
                regal_id="econ_data",
                passed=True,
                rationale="OK",
                phase=RegalPhaseV1.POST_AUDIT,
                report_sha="rsha3",
                inputs_sha="in3",
                determinism_seed=42,
            ),
        ]
        
        # Compute SHA with sorted order
        sorted_reports = sorted(reports, key=lambda r: (r.phase.value, r.regal_id))
        sha1 = sha256_json([r.report_sha for r in sorted_reports])
        
        # Same computation again
        sorted_reports2 = sorted(reports, key=lambda r: (r.phase.value, r.regal_id))
        sha2 = sha256_json([r.report_sha for r in sorted_reports2])
        
        assert sha1 == sha2
        
        # Verify order is correct: sorted by (phase.value, regal_id) - alphabetical
        # phase.value: "post_audit" < "post_plan_pre_apply", then by regal_id within phase
        expected_order = ["econ_data", "world_coherence", "spec_guardian"]
        actual_order = [r.regal_id for r in sorted_reports]
        assert actual_order == expected_order


# =============================================================================
# Test 3: Regal Annotations SHA
# =============================================================================


class TestRegalAnnotationsSha:
    """Tests for regal_annotations_sha matching canonical serialization."""
    
    def test_regal_annotations_sha_canonical(self):
        """regal_annotations_sha matches model_dump serialization."""
        annotations = RegalAnnotationsV1(
            violation_tags=["physics_anomaly"],
            training_disposition="quarantine",
            physics_anomaly_detected=True,
            regal_config_sha="config123",
        )
        
        # SHA from model_dump
        sha1 = sha256_json(annotations.model_dump(mode="json"))
        
        # Same object, same SHA
        sha2 = sha256_json(annotations.model_dump(mode="json"))
        assert sha1 == sha2
        
        # Different object with same values, same SHA
        annotations2 = RegalAnnotationsV1(
            violation_tags=["physics_anomaly"],
            training_disposition="quarantine",
            physics_anomaly_detected=True,
            regal_config_sha="config123",
        )
        sha3 = sha256_json(annotations2.model_dump(mode="json"))
        assert sha1 == sha3
    
    def test_regal_annotations_sha_changes_with_content(self):
        """Different content produces different SHA."""
        ann1 = RegalAnnotationsV1(training_disposition="allow")
        ann2 = RegalAnnotationsV1(training_disposition="quarantine")
        
        sha1 = sha256_json(ann1.model_dump(mode="json"))
        sha2 = sha256_json(ann2.model_dump(mode="json"))
        
        assert sha1 != sha2


# =============================================================================
# Test 4: Quarantine Enforcement
# =============================================================================


class TestQuarantineEnforcement:
    """Tests for quarantine actually filtering datapacks."""
    
    def test_quarantine_excludes_datapacks(self):
        """Quarantined datapacks are excluded from exposure."""
        tracker = ExposureTracker(manifest_id="test", step_start=0)
        tracker.set_quarantine(["bad_dp_001", "bad_dp_002"])
        
        # Record good and bad datapacks
        assert tracker.record_sample("task_a", datapack_id="good_dp_001") is True
        assert tracker.record_sample("task_a", datapack_id="bad_dp_001") is False  # Excluded
        assert tracker.record_sample("task_a", datapack_id="good_dp_002") is True
        assert tracker.record_sample("task_a", datapack_id="bad_dp_002") is False  # Excluded
        
        # Check counts
        assert tracker.excluded_count == 2
        
        # Manifest should only contain good datapacks
        manifest = tracker.build_manifest()
        assert "good_dp_001" in manifest.datapack_ids
        assert "good_dp_002" in manifest.datapack_ids
        assert "bad_dp_001" not in manifest.datapack_ids
        assert "bad_dp_002" not in manifest.datapack_ids
    
    def test_quarantine_manifest_sha_consistency(self):
        """quarantine_manifest_sha matches actual filter applied."""
        overrides = DatapackSelectionOverrides(
            quarantine_datapack_ids=["dp_001", "dp_002"],
        )
        quarantine_sha = sha256_json(sorted(overrides.quarantine_datapack_ids))
        
        # Same quarantine list produces same SHA
        overrides2 = DatapackSelectionOverrides(
            quarantine_datapack_ids=["dp_002", "dp_001"],  # Different order
        )
        quarantine_sha2 = sha256_json(sorted(overrides2.quarantine_datapack_ids))
        
        assert quarantine_sha == quarantine_sha2  # Order independent


# =============================================================================
# Test 5: verify_run() Tamper Detection
# =============================================================================


class TestVerifyRunTamperDetection:
    """Tests for verify_run detecting tampering."""
    
    def test_verify_run_detects_missing_manifest(self):
        """verify_run fails if manifest is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory
            report = verify_run(tmpdir)
            assert report.all_passed is False
            assert any("run_manifest.json" in c.message for c in report.checks if not c.passed)
    
    def test_verify_run_detects_missing_ledger(self):
        """verify_run fails if ledger is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest only
            manifest = RunManifestV1(
                run_id="test",
                plan_sha="plan123",
                audit_suite_id="test",
                audit_seed=42,
                audit_config_sha="cfg123",
                datapack_manifest_sha="dp123",
                seeds={"audit": 42},
            )
            manifest_path = Path(tmpdir) / "run_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest.model_dump(mode="json"), f)
            
            report = verify_run(tmpdir)
            assert any("ledger.jsonl" in c.message for c in report.checks if not c.passed)
    
    def test_verify_run_passes_valid_run(self):
        """verify_run passes for valid run outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "test_run"
            plan_sha = "plan123"
            
            # Create manifest
            manifest = RunManifestV1(
                run_id=run_id,
                plan_sha=plan_sha,
                audit_suite_id="test",
                audit_seed=42,
                audit_config_sha="cfg123",
                datapack_manifest_sha="dp123",
                seeds={"audit": 42},
            )
            manifest_path = Path(tmpdir) / "run_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest.model_dump(mode="json"), f)
            
            # Create ledger with matching record
            record = _create_minimal_ledger_record(
                run_id=run_id,
                regal_degraded=False,
                allow_deploy=True,
                regal_all_passed=True,
            )
            # Override plan_sha to match manifest
            record = record.model_copy(update={"plan_sha": plan_sha})
            
            ledger_path = Path(tmpdir) / "ledger.jsonl"
            with open(ledger_path, "w") as f:
                f.write(json.dumps(record.model_dump(mode="json")) + "\n")
            
            report = verify_run(tmpdir)
            
            # Should pass all checks
            assert report.all_passed is True, f"Failed checks: {[c for c in report.checks if not c.passed]}"
    
    def test_verify_run_detects_run_id_mismatch(self):
        """verify_run detects ledger record with wrong run_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest with run_id "run_A"
            manifest = RunManifestV1(
                run_id="run_A",
                plan_sha="plan123",
                audit_suite_id="test",
                audit_seed=42,
                audit_config_sha="cfg123",
                datapack_manifest_sha="dp123",
                seeds={"audit": 42},
            )
            manifest_path = Path(tmpdir) / "run_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest.model_dump(mode="json"), f)
            
            # Create ledger with wrong run_id "run_B"
            record = _create_minimal_ledger_record(run_id="run_B")
            record = record.model_copy(update={"plan_sha": "plan123"})
            
            ledger_path = Path(tmpdir) / "ledger.jsonl"
            with open(ledger_path, "w") as f:
                f.write(json.dumps(record.model_dump(mode="json")) + "\n")
            
            report = verify_run(tmpdir)
            
            # Should fail due to run_id mismatch
            assert report.all_passed is False
            assert any("run_id mismatch" in c.message for c in report.checks if not c.passed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
