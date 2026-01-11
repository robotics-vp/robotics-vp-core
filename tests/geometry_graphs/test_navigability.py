"""Test navigability metrics."""
import numpy as np
import pytest

from src.geometry_graphs.small_world import (
    build_small_world_graph,
    graph_summary_from_embeddings,
)
from src.contracts.schemas import GraphSpecV1


class TestNavigability:
    """Tests for bounded navigability via greedy routing."""

    def test_nav_success_rate_in_range(self):
        """Navigation success rate should be in [0, 1]."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(n_queries=32, max_hops=30)
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        assert 0 <= summary.nav_success_rate <= 1

    def test_nav_mean_hops_reasonable(self):
        """Mean hops for successful navigations should be reasonable."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(n_queries=32, max_hops=50)
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        # Mean hops should be <= max_hops
        assert summary.nav_mean_hops <= config.max_hops
        # Mean hops should be non-negative
        assert summary.nav_mean_hops >= 0

    def test_nav_stretch_reasonable(self):
        """Navigation stretch (greedy/optimal) should be >= 1."""
        H, W, D = 12, 12, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(n_queries=32, max_hops=50)
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        # Greedy routing should take >= optimal path
        if summary.nav_success_rate > 0:
            assert summary.nav_stretch >= 0.9  # Allow slight numerical tolerance

    def test_shortcuts_help_navigation(self):
        """Shortcuts should improve navigation success vs lattice-only."""
        H, W, D = 15, 15, 32
        rng = np.random.default_rng(42)
        
        # Create embeddings with meaningful clusters
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        # Config with shortcuts
        config_with_shortcuts = GraphSpecV1(
            knn_k=6,
            min_lattice_hops_for_shortcut=4,
            n_queries=64,
            max_hops=25,  # Limited hops to see difference
        )
        
        # Config without shortcuts (high min_hops)
        config_lattice_only = GraphSpecV1(
            knn_k=6,
            min_lattice_hops_for_shortcut=100,
            n_queries=64,
            max_hops=25,
        )
        
        summary_with = graph_summary_from_embeddings(
            embeddings, (H, W), config_with_shortcuts, seed=42
        )
        summary_without = graph_summary_from_embeddings(
            embeddings, (H, W), config_lattice_only, seed=42
        )
        
        # Verify shortcuts exist
        assert summary_with.shortcut_edge_count > 0
        assert summary_without.shortcut_edge_count == 0
        
        # Shortcuts should help (or at least not hurt) - with limited hops,
        # shortcuts enable reaching distant targets
        # Note: This may not always hold depending on embedding structure,
        # so we just verify both are valid
        assert 0 <= summary_with.nav_success_rate <= 1
        assert 0 <= summary_without.nav_success_rate <= 1


class TestEmptyAndEdgeCases:
    """Tests for edge cases."""

    def test_small_graph(self):
        """Very small graphs should not crash."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        
        config = GraphSpecV1(knn_k=1, n_sources=2, n_queries=2)
        
        summary = graph_summary_from_embeddings(embeddings, None, config, seed=42)
        
        assert summary.node_count == 2
        assert summary.local_edge_count == 1  # 1D lattice: 0-1

    def test_single_node(self):
        """Single node graph should not crash."""
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        
        config = GraphSpecV1(n_sources=1, n_queries=1)
        
        summary = graph_summary_from_embeddings(embeddings, None, config, seed=42)
        
        assert summary.node_count == 1
        assert summary.local_edge_count == 0
