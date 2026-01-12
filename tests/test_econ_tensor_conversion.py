"""Tests for econ tensor conversion utilities.

Verifies deterministic conversion between econ dicts and EconTensorV1.
"""
import pytest
from typing import Dict

from src.contracts.schemas import EconBasisSpecV1, EconTensorV1, RegimeFeaturesV1
from src.economics.econ_basis_registry import get_default_basis, ECON_BASIS_V1
from src.economics.econ_tensor import (
    econ_to_tensor,
    tensor_to_econ_dict,
    hash_econ_tensor,
    create_synthetic_econ_tensor,
    extract_key_econ_values,
    compute_tensor_summary,
)


class TestEconToTensor:
    """Tests for econ_to_tensor conversion."""

    def test_basic_conversion(self):
        """Test basic dict to tensor conversion."""
        econ_data = {
            "mpl_units_per_hour": 10.0,
            "wage_parity": 1.0,
            "energy_cost": 2.5,
        }
        tensor = econ_to_tensor(econ_data)

        assert tensor.basis_id == "econ_basis_v1"
        assert len(tensor.x) == 10  # Number of axes
        assert tensor.x[0] == 10.0  # mpl_units_per_hour
        assert tensor.x[1] == 1.0   # wage_parity
        assert tensor.x[2] == 2.5   # energy_cost

    def test_missing_axes_zero_fill(self):
        """Test zero_fill policy for missing axes."""
        econ_data = {"mpl_units_per_hour": 5.0}  # Only one axis

        tensor = econ_to_tensor(econ_data)

        assert tensor.x[0] == 5.0  # Present axis
        assert tensor.x[1] == 0.0  # Missing axis (zero-filled)
        assert tensor.mask is None  # No mask for zero_fill

    def test_missing_axes_mask_policy(self):
        """Test mask policy for missing axes."""
        basis = EconBasisSpecV1(
            basis_id="test_mask_basis",
            axes=["axis1", "axis2", "axis3"],
            missing_policy="mask",
        )
        econ_data = {"axis1": 1.0, "axis3": 3.0}  # axis2 missing

        tensor = econ_to_tensor(econ_data, basis=basis)

        assert tensor.x == [1.0, 0.0, 3.0]
        assert tensor.mask == [True, False, True]

    def test_source_tracking(self):
        """Test source field is set correctly."""
        tensor1 = econ_to_tensor({"mpl_units_per_hour": 1.0}, source="episode_metrics")
        tensor2 = econ_to_tensor({"mpl_units_per_hour": 1.0}, source="synthetic")

        assert tensor1.source == "episode_metrics"
        assert tensor2.source == "synthetic"

    def test_regime_features_linked(self):
        """Test regime features SHA is linked when provided."""
        features = RegimeFeaturesV1(audit_delta_success=0.1)
        tensor = econ_to_tensor(
            {"mpl_units_per_hour": 1.0},
            regime_features=features,
        )

        assert tensor.regime_features_sha == features.sha256()

    def test_stats_computed(self):
        """Test stats are computed correctly."""
        econ_data = {
            "mpl_units_per_hour": 3.0,
            "wage_parity": 4.0,  # 3^2 + 4^2 = 25, sqrt = 5 for these two
        }
        tensor = econ_to_tensor(econ_data)

        assert tensor.stats is not None
        assert "norm" in tensor.stats
        assert "min" in tensor.stats
        assert "max" in tensor.stats
        assert "nnz" in tensor.stats

    def test_nan_inf_handling(self):
        """Test NaN and Inf values are replaced with 0.0."""
        import math
        econ_data = {
            "mpl_units_per_hour": float("nan"),
            "wage_parity": float("inf"),
            "energy_cost": 1.0,
        }
        tensor = econ_to_tensor(econ_data)

        assert tensor.x[0] == 0.0  # NaN -> 0.0
        assert tensor.x[1] == 0.0  # Inf -> 0.0
        assert tensor.x[2] == 1.0  # Normal value preserved


class TestTensorToEconDict:
    """Tests for tensor_to_econ_dict conversion."""

    def test_roundtrip_conversion(self):
        """Test dict -> tensor -> dict roundtrip."""
        original = {
            "mpl_units_per_hour": 10.0,
            "wage_parity": 1.0,
            "energy_cost": 2.5,
            "damage_cost": 0.5,
            "novelty_delta": 0.1,
            "reward_scalar_sum": 1.5,
            "mobility_penalty": 0.0,
            "throughput": 20.0,
            "error_rate": 0.05,
            "success_rate": 0.95,
        }
        tensor = econ_to_tensor(original)
        recovered = tensor_to_econ_dict(tensor)

        for key, value in original.items():
            assert key in recovered
            assert abs(recovered[key] - value) < 1e-9

    def test_mask_respected(self):
        """Test mask is respected during conversion."""
        basis = EconBasisSpecV1(
            basis_id="test_mask",
            axes=["a", "b", "c"],
            missing_policy="mask",
        )
        tensor = EconTensorV1(
            basis_id="test_mask",
            basis_sha=basis.sha256(),
            x=[1.0, 2.0, 3.0],
            mask=[True, False, True],  # b is masked
            source="synthetic",
        )

        # Need to register basis for lookup
        from src.economics.econ_basis_registry import register_basis
        register_basis(basis)

        result = tensor_to_econ_dict(tensor, basis)

        assert "a" in result
        assert "b" not in result  # Masked out
        assert "c" in result

    def test_unknown_basis_raises(self):
        """Test unknown basis raises ValueError."""
        tensor = EconTensorV1(
            basis_id="unknown_basis",
            basis_sha="fake_sha",
            x=[1.0],
            source="synthetic",
        )

        with pytest.raises(ValueError, match="Unknown basis_id"):
            tensor_to_econ_dict(tensor)


class TestHashEconTensor:
    """Tests for tensor hashing."""

    def test_hash_deterministic(self):
        """Test hash is deterministic for same tensor."""
        econ_data = {"mpl_units_per_hour": 10.0, "success_rate": 0.9}

        tensor1 = econ_to_tensor(econ_data)
        tensor2 = econ_to_tensor(econ_data)

        assert hash_econ_tensor(tensor1) == hash_econ_tensor(tensor2)

    def test_hash_changes_with_values(self):
        """Test hash changes when values change."""
        tensor1 = econ_to_tensor({"mpl_units_per_hour": 10.0})
        tensor2 = econ_to_tensor({"mpl_units_per_hour": 20.0})

        assert hash_econ_tensor(tensor1) != hash_econ_tensor(tensor2)

    def test_hash_is_sha256(self):
        """Test hash is valid SHA-256 hex string."""
        tensor = econ_to_tensor({"mpl_units_per_hour": 1.0})
        h = hash_econ_tensor(tensor)

        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestSyntheticTensor:
    """Tests for synthetic tensor generation."""

    def test_synthetic_deterministic(self):
        """Test synthetic tensor is deterministic with same seed."""
        tensor1 = create_synthetic_econ_tensor(seed=42)
        tensor2 = create_synthetic_econ_tensor(seed=42)

        assert tensor1.x == tensor2.x
        assert tensor1.sha256() == tensor2.sha256()

    def test_synthetic_different_seeds(self):
        """Test different seeds produce different tensors."""
        tensor1 = create_synthetic_econ_tensor(seed=42)
        tensor2 = create_synthetic_econ_tensor(seed=43)

        assert tensor1.x != tensor2.x

    def test_synthetic_source_is_synthetic(self):
        """Test synthetic tensor has correct source."""
        tensor = create_synthetic_econ_tensor()
        assert tensor.source == "synthetic"


class TestTensorSummary:
    """Tests for tensor summary utilities."""

    def test_extract_key_values(self):
        """Test extracting key values from tensor."""
        econ_data = {
            "mpl_units_per_hour": 15.0,
            "success_rate": 0.85,
            "energy_cost": 3.0,
            "damage_cost": 1.0,
        }
        tensor = econ_to_tensor(econ_data)

        key_values = extract_key_econ_values(tensor)

        assert "mpl_units_per_hour" in key_values
        assert "success_rate" in key_values
        assert key_values["mpl_units_per_hour"] == 15.0
        assert key_values["success_rate"] == 0.85

    def test_compute_summary(self):
        """Test computing tensor summary."""
        econ_data = {
            "mpl_units_per_hour": 10.0,
            "success_rate": 0.9,
        }
        tensor = econ_to_tensor(econ_data)

        summary = compute_tensor_summary(tensor)

        assert "norm" in summary
        assert summary["norm"] > 0


class TestEconTensorV1Schema:
    """Tests for EconTensorV1 schema."""

    def test_tensor_creation(self):
        """Test tensor can be created directly."""
        tensor = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0, 3.0],
            source="synthetic",
        )

        assert tensor.basis_id == "test"
        assert tensor.x == [1.0, 2.0, 3.0]

    def test_tensor_sha_deterministic(self):
        """Test tensor SHA is deterministic."""
        tensor1 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0],
            source="synthetic",
        )
        tensor2 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0],
            source="synthetic",
        )

        assert tensor1.sha256() == tensor2.sha256()

    def test_tensor_sha_changes(self):
        """Test tensor SHA changes with different values."""
        tensor1 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 2.0],
            source="synthetic",
        )
        tensor2 = EconTensorV1(
            basis_id="test",
            basis_sha="abc123",
            x=[1.0, 3.0],  # Different value
            source="synthetic",
        )

        assert tensor1.sha256() != tensor2.sha256()
