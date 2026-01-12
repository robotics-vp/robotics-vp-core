"""Tests for econ SHA inclusion in manifests and ledger.

Verifies that run_manifest and ledger records include econ basis
and tensor SHAs when computed.
"""
import pytest
from typing import Optional

from src.contracts.schemas import (
    RunManifestV1,
    ValueLedgerRecordV1,
    LedgerEconV1,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
    LedgerAuditV1,
    LedgerDeltasV1,
    EconBasisSpecV1,
    EconTensorV1,
)
from src.economics.econ_basis_registry import get_default_basis
from src.economics.econ_tensor import econ_to_tensor, compute_tensor_summary
from src.valuation.run_manifest import create_run_manifest


class TestRunManifestEconFields:
    """Tests for econ fields in RunManifestV1."""

    def test_manifest_has_econ_basis_sha_field(self):
        """Test RunManifestV1 has econ_basis_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "econ_basis_sha")
        assert manifest.econ_basis_sha is None  # Default

    def test_manifest_has_econ_tensor_sha_field(self):
        """Test RunManifestV1 has econ_tensor_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "econ_tensor_sha")
        assert manifest.econ_tensor_sha is None  # Default

    def test_manifest_econ_fields_can_be_set(self):
        """Test econ fields can be populated."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=["dp1"],
            seeds={"main": 42},
            determinism_config={"enabled": True},
        )

        # Create econ tensor
        basis = get_default_basis()
        tensor = econ_to_tensor({"mpl_units_per_hour": 10.0})

        # Set econ fields
        manifest.econ_basis_sha = basis.sha256
        manifest.econ_tensor_sha = tensor.sha256()

        assert manifest.econ_basis_sha is not None
        assert len(manifest.econ_basis_sha) == 64
        assert manifest.econ_tensor_sha is not None
        assert len(manifest.econ_tensor_sha) == 64


class TestLedgerEconV1Schema:
    """Tests for LedgerEconV1 schema."""

    def test_ledger_econ_creation(self):
        """Test LedgerEconV1 can be created."""
        ledger_econ = LedgerEconV1(
            basis_sha="abc123" + "0" * 58,  # 64 chars
            econ_tensor_sha="def456" + "0" * 58,
        )

        assert ledger_econ.basis_sha.startswith("abc123")
        assert ledger_econ.econ_tensor_sha.startswith("def456")
        assert ledger_econ.econ_tensor_summary is None

    def test_ledger_econ_with_summary(self):
        """Test LedgerEconV1 with summary."""
        summary = {"norm": 10.5, "mpl_units_per_hour": 15.0}
        ledger_econ = LedgerEconV1(
            basis_sha="abc123",
            econ_tensor_sha="def456",
            econ_tensor_summary=summary,
        )

        assert ledger_econ.econ_tensor_summary == summary
        assert ledger_econ.econ_tensor_summary["norm"] == 10.5


class TestValueLedgerRecordEconField:
    """Tests for econ field in ValueLedgerRecordV1."""

    def test_ledger_record_has_econ_field(self):
        """Test ValueLedgerRecordV1 has econ field."""
        record = ValueLedgerRecordV1(
            record_id="rec_001",
            run_id="run_001",
            plan_id="plan_001",
            plan_sha="plan_sha",
            audit=LedgerAuditV1(
                audit_suite_id="test_suite",
                audit_seed=42,
                audit_config_sha="config_sha",
                audit_results_before_sha="before_sha",
                audit_results_after_sha="after_sha",
            ),
            deltas=LedgerDeltasV1(),
            window=LedgerWindowV1(
                step_start=0,
                step_end=1000,
                ts_start="2024-01-01T00:00:00",
                ts_end="2024-01-01T01:00:00",
            ),
            exposure=LedgerExposureV1(
                datapack_ids=["dp1"],
                slice_ids=["s1"],
                exposure_manifest_sha="manifest_sha",
            ),
            policy=LedgerPolicyV1(
                policy_before="ckpt_a",
                policy_after="ckpt_b",
            ),
        )

        assert hasattr(record, "econ")
        assert record.econ is None  # Default

    def test_ledger_record_with_econ(self):
        """Test ValueLedgerRecordV1 can include econ."""
        # Create econ tensor and summary
        tensor = econ_to_tensor({"mpl_units_per_hour": 10.0})
        basis = get_default_basis()
        summary = compute_tensor_summary(tensor)

        ledger_econ = LedgerEconV1(
            basis_sha=basis.sha256,
            econ_tensor_sha=tensor.sha256(),
            econ_tensor_summary=summary,
        )

        record = ValueLedgerRecordV1(
            record_id="rec_001",
            run_id="run_001",
            plan_id="plan_001",
            plan_sha="plan_sha",
            audit=LedgerAuditV1(
                audit_suite_id="test_suite",
                audit_seed=42,
                audit_config_sha="config_sha",
                audit_results_before_sha="before_sha",
                audit_results_after_sha="after_sha",
            ),
            deltas=LedgerDeltasV1(),
            window=LedgerWindowV1(
                step_start=0,
                step_end=1000,
                ts_start="2024-01-01T00:00:00",
                ts_end="2024-01-01T01:00:00",
            ),
            exposure=LedgerExposureV1(
                datapack_ids=["dp1"],
                slice_ids=["s1"],
                exposure_manifest_sha="manifest_sha",
            ),
            policy=LedgerPolicyV1(
                policy_before="ckpt_a",
                policy_after="ckpt_b",
            ),
            econ=ledger_econ,
        )

        assert record.econ is not None
        assert record.econ.basis_sha == basis.sha256
        assert record.econ.econ_tensor_sha == tensor.sha256()


class TestEconShaRecomputation:
    """Tests that SHA values match recomputation."""

    def test_basis_sha_recomputable(self):
        """Test basis SHA can be recomputed."""
        basis = get_default_basis()
        sha1 = basis.sha256
        sha2 = basis.spec.sha256()

        assert sha1 == sha2

    def test_tensor_sha_recomputable(self):
        """Test tensor SHA can be recomputed from same data."""
        econ_data = {
            "mpl_units_per_hour": 10.0,
            "wage_parity": 1.0,
            "success_rate": 0.9,
        }

        tensor1 = econ_to_tensor(econ_data)
        tensor2 = econ_to_tensor(econ_data)

        assert tensor1.sha256() == tensor2.sha256()

    def test_manifest_sha_matches_tensor_sha(self):
        """Test manifest econ_tensor_sha matches actual tensor SHA."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        tensor = econ_to_tensor({"mpl_units_per_hour": 10.0})
        manifest.econ_tensor_sha = tensor.sha256()

        # Verify it matches
        assert manifest.econ_tensor_sha == tensor.sha256()


class TestEconBasisSpecSha:
    """Tests for EconBasisSpecV1 SHA computation."""

    def test_basis_spec_sha_deterministic(self):
        """Test basis spec SHA is deterministic."""
        spec1 = EconBasisSpecV1(
            basis_id="test_basis",
            axes=["a", "b", "c"],
            units={"a": "unit_a"},
            scales={"a": 1.0},
            missing_policy="zero_fill",
        )
        spec2 = EconBasisSpecV1(
            basis_id="test_basis",
            axes=["a", "b", "c"],
            units={"a": "unit_a"},
            scales={"a": 1.0},
            missing_policy="zero_fill",
        )

        assert spec1.sha256() == spec2.sha256()

    def test_basis_spec_sha_changes_with_axes(self):
        """Test SHA changes when axes change."""
        spec1 = EconBasisSpecV1(basis_id="test", axes=["a", "b"])
        spec2 = EconBasisSpecV1(basis_id="test", axes=["a", "c"])

        assert spec1.sha256() != spec2.sha256()


class TestEconTensorSha:
    """Tests for EconTensorV1 SHA computation."""

    def test_tensor_sha_deterministic(self):
        """Test tensor SHA is deterministic."""
        tensor1 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0, 3.0],
            source="synthetic",
        )
        tensor2 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0, 3.0],
            source="synthetic",
        )

        assert tensor1.sha256() == tensor2.sha256()

    def test_tensor_sha_changes_with_values(self):
        """Test SHA changes when values change."""
        tensor1 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0, 3.0],
            source="synthetic",
        )
        tensor2 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0, 4.0],  # Different value
            source="synthetic",
        )

        assert tensor1.sha256() != tensor2.sha256()

    def test_tensor_sha_length(self):
        """Test tensor SHA is proper length."""
        tensor = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0],
            source="synthetic",
        )

        sha = tensor.sha256()
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)
