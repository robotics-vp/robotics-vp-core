"""Test determinism of small-world graph construction and metrics."""
import numpy as np
import pytest

from src.geometry_graphs.small_world import (
    build_small_world_graph,
    graph_summary_from_embeddings,
)
from src.contracts.schemas import GraphSpecV1


class TestGraphDeterminism:
    """Tests that graph construction and metrics are deterministic."""

    def test_same_seed_same_sha(self):
        """Same embeddings + seed → identical graph_summary_sha."""
        # Create synthetic grid embeddings
        rng = np.random.default_rng(42)
        H, W, D = 10, 10, 16
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(
            spec_id="test_spec",
            local_connectivity=4,
            knn_k=4,
            min_lattice_hops_for_shortcut=3,
            n_sources=8,
            n_queries=16,
            max_hops=20,
            seed=42,
        )
        
        # Run twice with same seed
        summary1 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        summary2 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        assert summary1.summary_sha == summary2.summary_sha
        assert summary1.sigma == summary2.sigma
        assert summary1.nav_success_rate == summary2.nav_success_rate

    def test_different_seed_different_sha(self):
        """Different seeds → different graph_summary_sha (due to sampling)."""
        rng = np.random.default_rng(42)
        H, W, D = 10, 10, 16
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(
            spec_id="test_spec",
            n_sources=8,
            n_queries=16,
        )
        
        summary1 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        summary2 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=123)
        
        # Metrics may differ due to random sampling in path length estimation
        # and navigability queries
        assert summary1.graph_spec_sha == summary2.graph_spec_sha  # Spec is same
        # SHA will differ due to different sampled metrics
        assert summary1.summary_sha != summary2.summary_sha

    def test_graph_structure_deterministic(self):
        """Graph structure (adjacency) is deterministic for same inputs."""
        rng = np.random.default_rng(123)
        H, W, D = 8, 8, 8
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1(knn_k=4, min_lattice_hops_for_shortcut=3)

        adj1, counts1, hops1, _, scores1 = build_small_world_graph(embeddings, (H, W), config, seed=42)
        adj2, counts2, hops2, _, scores2 = build_small_world_graph(embeddings, (H, W), config, seed=42)

        assert adj1 == adj2
        assert counts1.local_edges == counts2.local_edges
        assert counts1.shortcut_edges == counts2.shortcut_edges
        assert hops1 == hops2
        assert scores1 == scores2


class TestTokenModeFallback:
    """Tests for 1D token mode (when no grid available)."""

    def test_token_mode_1d_lattice(self):
        """Token mode uses 1D lattice (adjacent tokens as local edges)."""
        rng = np.random.default_rng(42)
        N, D = 20, 8
        embeddings = rng.standard_normal((N, D)).astype(np.float32)

        config = GraphSpecV1(knn_k=2, min_lattice_hops_for_shortcut=5)

        adj, counts, _, _, _ = build_small_world_graph(embeddings, None, config, seed=42)

        # Should have N-1 local edges in a 1D chain (undirected)
        assert counts.local_edges == N - 1

        # Check adjacency structure
        assert 1 in adj[0]  # Node 0 connected to 1
        assert 0 in adj[1]  # Node 1 connected to 0
        assert 2 in adj[1]  # Node 1 connected to 2

    def test_token_mode_produces_valid_summary(self):
        """Token mode produces valid GraphSummaryV1."""
        rng = np.random.default_rng(42)
        N, D = 30, 16
        embeddings = rng.standard_normal((N, D)).astype(np.float32)
        
        config = GraphSpecV1(n_sources=8, n_queries=16)
        
        summary = graph_summary_from_embeddings(embeddings, None, config, seed=42)
        
        assert summary.node_mode == "tokens"
        assert summary.node_count == N
        assert summary.sigma >= 0
        assert 0 <= summary.nav_success_rate <= 1
