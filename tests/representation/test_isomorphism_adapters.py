"""Tests for representation isomorphism adapters."""
import numpy as np
import pytest

from src.representation.space import RepresentationPayload, InvariantReport
from src.representation.isomorphisms import LinearAlign, AlignmentReport


class TestRepresentationPayload:
    """Tests for RepresentationPayload."""

    def test_pooled_1d(self):
        """Test pooled() on 1D features."""
        payload = RepresentationPayload(
            features=np.array([1.0, 2.0, 3.0]),
            dim=3,
        )
        pooled = payload.pooled()
        np.testing.assert_array_equal(pooled, np.array([1.0, 2.0, 3.0]))

    def test_pooled_2d(self):
        """Test pooled() on 2D features (sequence)."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        payload = RepresentationPayload(features=features, dim=2)
        pooled = payload.pooled()
        np.testing.assert_array_almost_equal(pooled, np.array([3.0, 4.0]))


class TestLinearAlign:
    """Tests for LinearAlign adapter."""

    def test_fit_identity(self):
        """Test fitting on identical source/target."""
        np.random.seed(42)
        dim = 8
        n_samples = 10

        features = np.random.randn(n_samples, dim).astype(np.float32)
        source_payloads = [RepresentationPayload(f, dim) for f in features]
        target_payloads = [RepresentationPayload(f, dim) for f in features]

        adapter = LinearAlign(whiten=False)
        adapter.fit(source_payloads, target_payloads)

        # Should be close to identity
        for source, target in zip(source_payloads, target_payloads):
            transformed = adapter.transform(source)
            np.testing.assert_array_almost_equal(
                transformed.pooled(), target.pooled(), decimal=5
            )

    def test_fit_rotation(self):
        """Test fitting on rotated source/target."""
        np.random.seed(42)
        dim = 4
        n_samples = 50  # More samples for stable fit

        # Create orthogonal rotation matrix
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))

        source_features = np.random.randn(n_samples, dim).astype(np.float32)
        target_features = source_features @ Q

        source_payloads = [RepresentationPayload(f, dim) for f in source_features]
        target_payloads = [RepresentationPayload(f, dim) for f in target_features]

        adapter = LinearAlign(whiten=False)
        adapter.fit(source_payloads, target_payloads)

        # Transform should recover target with low error
        errors = []
        for source, target in zip(source_payloads, target_payloads):
            transformed = adapter.transform(source)
            error = np.linalg.norm(transformed.pooled() - target.pooled())
            errors.append(error)

        mean_error = np.mean(errors)
        assert mean_error < 0.5, f"Mean alignment error too large: {mean_error}"

    def test_cycle_consistency(self):
        """Test A->B->A cycle consistency error."""
        np.random.seed(42)
        dim = 8
        n_samples = 15

        source_features = np.random.randn(n_samples, dim).astype(np.float32)
        # Add noise and slight rotation
        noise = np.random.randn(n_samples, dim).astype(np.float32) * 0.1
        target_features = source_features + noise

        source_payloads = [RepresentationPayload(f, dim) for f in source_features]
        target_payloads = [RepresentationPayload(f, dim) for f in target_features]

        adapter = LinearAlign()
        adapter.fit(source_payloads, target_payloads)

        # Cycle error should be small (rotation is invertible)
        cycle_error = adapter.cycle_error(source_payloads)
        assert cycle_error < 0.1, f"Cycle error too large: {cycle_error}"

    def test_alignment_report(self):
        """Test alignment report generation."""
        np.random.seed(42)
        dim = 4
        n_samples = 10

        source_features = np.random.randn(n_samples, dim).astype(np.float32)
        target_features = source_features * 1.1 + 0.1  # Slight scale and shift

        source_payloads = [RepresentationPayload(f, dim) for f in source_features]
        target_payloads = [RepresentationPayload(f, dim) for f in target_features]

        adapter = LinearAlign()
        adapter.fit(source_payloads, target_payloads)

        report = adapter.alignment_report(source_payloads, target_payloads)

        assert isinstance(report, AlignmentReport)
        assert report.alignment_error >= 0
        assert report.cycle_error >= 0
        assert len(report.per_sample_errors) == n_samples
        assert report.metadata["dim"] == dim

    def test_export_import(self):
        """Test export and import of adapter."""
        np.random.seed(42)
        dim = 4
        n_samples = 10

        source_features = np.random.randn(n_samples, dim).astype(np.float32)
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        target_features = source_features @ Q

        source_payloads = [RepresentationPayload(f, dim) for f in source_features]
        target_payloads = [RepresentationPayload(f, dim) for f in target_features]

        # Fit original adapter
        adapter1 = LinearAlign(source_name="rgb", target_name="bev")
        adapter1.fit(source_payloads, target_payloads)

        # Export and import
        export_data = adapter1.export()
        adapter2 = LinearAlign.from_export(export_data)

        # Both should produce same results
        for source in source_payloads:
            result1 = adapter1.transform(source).pooled()
            result2 = adapter2.transform(source).pooled()
            np.testing.assert_array_almost_equal(result1, result2, decimal=6)

    def test_whiten_effect(self):
        """Test that whitening handles different variances."""
        np.random.seed(42)
        dim = 4
        n_samples = 20

        # Source with high variance in one dim
        source_features = np.random.randn(n_samples, dim).astype(np.float32)
        source_features[:, 0] *= 10  # High variance in dim 0

        # Target with uniform variance
        target_features = np.random.randn(n_samples, dim).astype(np.float32)

        source_payloads = [RepresentationPayload(f, dim) for f in source_features]
        target_payloads = [RepresentationPayload(f, dim) for f in target_features]

        # Without whitening
        adapter_no_whiten = LinearAlign(whiten=False)
        adapter_no_whiten.fit(source_payloads, target_payloads)

        # With whitening
        adapter_whiten = LinearAlign(whiten=True)
        adapter_whiten.fit(source_payloads, target_payloads)

        # Both should produce valid transforms (no NaN)
        for source in source_payloads:
            result_no = adapter_no_whiten.transform(source).pooled()
            result_yes = adapter_whiten.transform(source).pooled()
            assert not np.any(np.isnan(result_no))
            assert not np.any(np.isnan(result_yes))


class TestInvariantReport:
    """Tests for InvariantReport."""

    def test_default_values(self):
        """Test default initialization."""
        report = InvariantReport()
        assert report.norm_mean == 0.0
        assert report.stability_score == 1.0

    def test_metadata(self):
        """Test metadata storage."""
        report = InvariantReport(metadata={"dim": 64, "samples": 100})
        assert report.metadata["dim"] == 64
        assert report.metadata["samples"] == 100
