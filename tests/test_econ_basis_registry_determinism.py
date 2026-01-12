"""Tests for econ basis registry determinism.

Verifies that basis definitions are immutable and produce stable SHAs.
"""
import pytest

from src.contracts.schemas import EconBasisSpecV1
from src.economics.econ_basis_registry import (
    EconBasisDefinition,
    register_basis,
    get_basis,
    get_basis_sha,
    list_bases,
    get_default_basis,
    ECON_BASIS_V1,
    ECON_BASIS_V1_AXES,
)


class TestEconBasisRegistry:
    """Tests for basis registry functionality."""

    def test_default_basis_exists(self):
        """Test that default basis is registered on module load."""
        basis = get_default_basis()
        assert basis is not None
        assert basis.basis_id == "econ_basis_v1"

    def test_default_basis_sha_stable(self):
        """Test that default basis SHA is deterministic."""
        basis1 = get_default_basis()
        basis2 = get_default_basis()
        assert basis1.sha256 == basis2.sha256
        assert len(basis1.sha256) == 64  # SHA-256 hex

    def test_basis_id_returns_same_definition(self):
        """Test that same basis_id always returns same definition."""
        basis1 = get_basis("econ_basis_v1")
        basis2 = get_basis("econ_basis_v1")
        assert basis1 is basis2
        assert basis1.sha256 == basis2.sha256

    def test_basis_sha_matches_get_basis_sha(self):
        """Test get_basis_sha returns correct SHA."""
        basis = get_basis("econ_basis_v1")
        sha = get_basis_sha("econ_basis_v1")
        assert basis.sha256 == sha

    def test_unknown_basis_returns_none(self):
        """Test that unknown basis_id returns None."""
        assert get_basis("nonexistent_basis") is None
        assert get_basis_sha("nonexistent_basis") is None


class TestEconBasisAxesOrder:
    """Tests for axis ordering stability."""

    def test_axes_order_stable(self):
        """Test that axes order is canonical and stable."""
        basis = get_default_basis()
        expected_axes = [
            "mpl_units_per_hour",
            "wage_parity",
            "energy_cost",
            "damage_cost",
            "novelty_delta",
            "reward_scalar_sum",
            "mobility_penalty",
            "throughput",
            "error_rate",
            "success_rate",
        ]
        assert basis.spec.axes == expected_axes

    def test_axes_constant_matches_spec(self):
        """Test ECON_BASIS_V1_AXES constant matches spec."""
        basis = get_default_basis()
        assert basis.spec.axes == ECON_BASIS_V1_AXES

    def test_axes_order_affects_sha(self):
        """Test that different axis order produces different SHA."""
        # Create spec with reversed axes
        reversed_spec = EconBasisSpecV1(
            basis_id="test_reversed",
            axes=list(reversed(ECON_BASIS_V1_AXES)),
            missing_policy="zero_fill",
        )

        # Original spec
        original_spec = ECON_BASIS_V1

        assert original_spec.sha256() != reversed_spec.sha256()


class TestEconBasisImmutability:
    """Tests for basis immutability."""

    def test_frozen_definition(self):
        """Test EconBasisDefinition is frozen."""
        basis = get_default_basis()
        assert isinstance(basis, EconBasisDefinition)

        # frozen=True should prevent modification
        with pytest.raises((AttributeError, TypeError)):
            basis.basis_id = "modified"

    def test_reregistration_with_same_spec_ok(self):
        """Test re-registering same spec doesn't raise."""
        # This should not raise
        register_basis(ECON_BASIS_V1)

        # Should still be the same
        basis = get_basis("econ_basis_v1")
        assert basis.sha256 == ECON_BASIS_V1.sha256()

    def test_reregistration_with_different_spec_raises(self):
        """Test re-registering different spec raises ValueError."""
        # Try to register a different spec with same basis_id
        different_spec = EconBasisSpecV1(
            basis_id="econ_basis_v1",  # Same ID
            axes=["different_axis"],  # Different axes
            missing_policy="mask",
        )

        with pytest.raises(ValueError, match="already registered"):
            register_basis(different_spec)


class TestEconBasisSpecV1:
    """Tests for EconBasisSpecV1 schema."""

    def test_spec_creation(self):
        """Test spec can be created."""
        spec = EconBasisSpecV1(
            basis_id="test_spec",
            axes=["axis1", "axis2"],
        )
        assert spec.basis_id == "test_spec"
        assert spec.axes == ["axis1", "axis2"]
        assert spec.missing_policy == "zero_fill"  # Default

    def test_spec_sha_deterministic(self):
        """Test spec SHA is deterministic."""
        spec1 = EconBasisSpecV1(
            basis_id="test",
            axes=["a", "b"],
            units={"a": "unit_a"},
            scales={"a": 1.0},
        )
        spec2 = EconBasisSpecV1(
            basis_id="test",
            axes=["a", "b"],
            units={"a": "unit_a"},
            scales={"a": 1.0},
        )
        assert spec1.sha256() == spec2.sha256()

    def test_spec_sha_changes_with_content(self):
        """Test SHA changes when content changes."""
        spec1 = EconBasisSpecV1(basis_id="test", axes=["a", "b"])
        spec2 = EconBasisSpecV1(basis_id="test", axes=["a", "c"])
        assert spec1.sha256() != spec2.sha256()

    def test_spec_missing_policy_options(self):
        """Test missing_policy accepts valid options."""
        spec_zero = EconBasisSpecV1(
            basis_id="test_zero",
            axes=["a"],
            missing_policy="zero_fill",
        )
        spec_mask = EconBasisSpecV1(
            basis_id="test_mask",
            axes=["a"],
            missing_policy="mask",
        )
        assert spec_zero.missing_policy == "zero_fill"
        assert spec_mask.missing_policy == "mask"


class TestListBases:
    """Tests for listing registered bases."""

    def test_list_bases_includes_default(self):
        """Test list_bases includes the default basis."""
        bases = list_bases()
        assert "econ_basis_v1" in bases
        assert bases["econ_basis_v1"] == get_basis_sha("econ_basis_v1")
