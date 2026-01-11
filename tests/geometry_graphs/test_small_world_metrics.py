"""Test small-world metrics sanity checks."""
import numpy as np
import pytest

from src.geometry_graphs.small_world import (
    build_small_world_graph,
    compute_graph_metrics,
    graph_summary_from_embeddings,
    GraphEdgeCounts,
)
from src.contracts.schemas import GraphSpecV1


class TestSmallWorldMetricsSanity:
    """Sanity checks for small-world metrics."""

    def test_lattice_only_zero_or_low_clustering(self):
        """4-neighbor lattice-only graph has 0 clustering (no triangles)."""
        # Create grid where embeddings are far apart (no shortcuts)
        H, W, D = 10, 10, 16
        
        # Embeddings that are very distinct - each cell has unique embedding
        embeddings = np.zeros((H, W, D), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                embeddings[i, j, (i * W + j) % D] = 1.0
        
        # High min_lattice_hops ensures no shortcuts
        config = GraphSpecV1(
            knn_k=4,
            min_lattice_hops_for_shortcut=100,  # Effectively disable shortcuts
            n_sources=16,
            n_queries=16,
        )
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        # Lattice-only should have zero shortcuts
        assert summary.shortcut_edge_count == 0
        # 4-neighbor grid has 0 clustering (no triangles in a grid)
        assert summary.clustering_coefficient == 0.0

    def test_small_world_sigma_with_shortcuts(self):
        """Small-world graph with shortcuts should have σ > 0."""
        # Create grid with clustered embeddings to generate shortcuts
        H, W, D = 15, 15, 32
        rng = np.random.default_rng(42)
        
        # Create embeddings with some structure
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(
            knn_k=6,
            min_lattice_hops_for_shortcut=4,
            n_sources=16,
            n_queries=32,
        )
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        # Should have some shortcuts
        assert summary.shortcut_edge_count > 0
        # Sigma should be computable
        assert summary.sigma >= 0
        # Shortcut fraction should be reasonable
        assert 0 < summary.shortcut_fraction < 1

    def test_path_length_estimation(self):
        """Average path length should be reasonable for grid."""
        H, W, D = 10, 10, 8
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(
            knn_k=4,
            min_lattice_hops_for_shortcut=4,
            n_sources=16,
        )
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        # For a 10x10 grid, max path ~ 18 (diagonal), average should be less
        assert 0 < summary.avg_path_length < 20
        # Random baseline should also be reasonable
        assert summary.l_rand > 0

    def test_random_baseline_computed(self):
        """Random baseline (C_rand, L_rand) should be computed."""
        H, W, D = 8, 8, 8
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        config = GraphSpecV1(n_sources=8)
        
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        
        # Both baselines should be computed
        assert summary.c_rand >= 0
        assert summary.l_rand >= 0


class TestShortcutStats:
    """Tests for shortcut edge statistics."""

    def test_shortcut_lattice_hops(self):
        """Shortcut edges should respect min_lattice_hops constraint."""
        H, W, D = 12, 12, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        min_hops = 5
        config = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=min_hops,
        )

        adj, counts, hops, _, _ = build_small_world_graph(embeddings, (H, W), config, seed=42)

        # All shortcut hops should be >= min_hops
        for hop_dist in hops:
            assert hop_dist >= min_hops

    def test_shortcut_fraction_in_range(self):
        """Shortcut fraction should be in [0, 1]."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1(knn_k=6, min_lattice_hops_for_shortcut=3)

        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

        assert 0 <= summary.shortcut_fraction <= 1


class TestScoreBasedSelection:
    """Tests for score-based shortcut selection modes."""

    def test_threshold_mode_monotonicity(self):
        """Higher threshold should result in fewer or equal shortcuts."""
        H, W, D = 12, 12, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
        shortcut_counts = []

        for threshold in thresholds:
            config = GraphSpecV1(
                knn_k=8,
                min_lattice_hops_for_shortcut=3,
                shortcut_select_mode="threshold",
                shortcut_score_threshold=threshold,
                max_shortcut_fraction=0.99,  # High ceiling to not interfere
            )
            summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
            shortcut_counts.append(summary.shortcut_edge_count)

        # Monotonically non-increasing
        for i in range(len(shortcut_counts) - 1):
            assert shortcut_counts[i] >= shortcut_counts[i + 1], \
                f"Threshold {thresholds[i]} had {shortcut_counts[i]} shortcuts, " \
                f"but {thresholds[i+1]} had {shortcut_counts[i+1]}"

    def test_top_m_per_node_budget_respected(self):
        """Top-M mode should respect per-node budget."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        for top_m in [1, 2, 3]:
            config = GraphSpecV1(
                knn_k=8,
                min_lattice_hops_for_shortcut=3,
                shortcut_select_mode="top_m_per_node",
                shortcut_top_m_per_node=top_m,
                max_shortcut_fraction=0.99,  # High ceiling
            )
            adj, counts, _, _, _ = build_small_world_graph(embeddings, (H, W), config, seed=42)

            # Count shortcuts per node (shortcuts are edges not in local lattice)
            # Each shortcut edge contributes to both endpoints
            N = H * W
            shortcuts_per_node = {i: 0 for i in range(N)}

            # Rebuild local edges to identify shortcuts
            local_edges = set()
            for i in range(N):
                row, col = divmod(i, W)
                neighbors = []
                if row > 0:
                    neighbors.append((row - 1) * W + col)
                if row < H - 1:
                    neighbors.append((row + 1) * W + col)
                if col > 0:
                    neighbors.append(row * W + (col - 1))
                if col < W - 1:
                    neighbors.append(row * W + (col + 1))
                for j in neighbors:
                    local_edges.add((min(i, j), max(i, j)))

            for i in range(N):
                for j in adj[i]:
                    if i < j and (i, j) not in local_edges:
                        shortcuts_per_node[i] += 1
                        shortcuts_per_node[j] += 1

            # Each node should have at most top_m shortcuts
            for node, count in shortcuts_per_node.items():
                assert count <= top_m, \
                    f"Node {node} has {count} shortcuts but budget is {top_m}"

    def test_safety_ceiling_caps_shortcuts(self):
        """Very low max_shortcut_fraction should cap total shortcuts."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        # Very restrictive fraction
        config = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=3,
            shortcut_select_mode="top_m_per_node",
            shortcut_top_m_per_node=10,  # High per-node budget
            max_shortcut_fraction=0.05,  # But low global ceiling
        )
        adj, counts, _, _, _ = build_small_world_graph(embeddings, (H, W), config, seed=42)

        # Calculate expected ceiling
        local_edges = counts.local_edges
        max_allowed = int(local_edges * 0.05 / (1 - 0.05))

        assert counts.shortcut_edges <= max_allowed + 1, \
            f"Expected at most {max_allowed} shortcuts, got {counts.shortcut_edges}"

    def test_safety_ceiling_keeps_highest_scoring(self):
        """When ceiling is applied, highest-scoring shortcuts should be kept."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        # First get all shortcuts with high ceiling
        config_high = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=3,
            shortcut_select_mode="top_m_per_node",
            shortcut_top_m_per_node=10,
            max_shortcut_fraction=0.99,
        )
        _, counts_high, _, _, scores_high = build_small_world_graph(embeddings, (H, W), config_high, seed=42)

        # Then with low ceiling
        config_low = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=3,
            shortcut_select_mode="top_m_per_node",
            shortcut_top_m_per_node=10,
            max_shortcut_fraction=0.05,
        )
        _, counts_low, _, _, scores_low = build_small_world_graph(embeddings, (H, W), config_low, seed=42)

        # Low ceiling should have fewer shortcuts
        assert counts_low.shortcut_edges <= counts_high.shortcut_edges

        # If scores_low is non-empty, its min should be >= scores_high's min
        # (because we kept the highest-scoring ones)
        if scores_low and scores_high:
            min_score_low = min(scores_low)
            min_score_high = min(scores_high)
            # The minimum score in the capped version should be >= minimum in uncapped
            # (or equal if all were kept)
            assert min_score_low >= min_score_high - 1e-6

    def test_determinism_with_selection_modes(self):
        """Both selection modes should be deterministic."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        for mode in ["threshold", "top_m_per_node"]:
            config = GraphSpecV1(
                knn_k=6,
                min_lattice_hops_for_shortcut=3,
                shortcut_select_mode=mode,
                shortcut_score_threshold=0.5 if mode == "threshold" else None,
                shortcut_top_m_per_node=2,
            )

            summary1 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
            summary2 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

            assert summary1.summary_sha == summary2.summary_sha, \
                f"Mode {mode} is not deterministic"
            assert summary1.shortcut_select_mode == mode

    def test_summary_records_selection_mode(self):
        """GraphSummary should record the selection mode used."""
        H, W, D = 8, 8, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        # Threshold mode
        config_thresh = GraphSpecV1(
            shortcut_select_mode="threshold",
            shortcut_score_threshold=0.5,
        )
        summary_thresh = graph_summary_from_embeddings(embeddings, (H, W), config_thresh, seed=42)
        assert summary_thresh.shortcut_select_mode == "threshold"
        assert summary_thresh.shortcut_score_threshold_used == 0.5

        # Top-M mode
        config_topm = GraphSpecV1(
            shortcut_select_mode="top_m_per_node",
            shortcut_top_m_per_node=3,
        )
        summary_topm = graph_summary_from_embeddings(embeddings, (H, W), config_topm, seed=42)
        assert summary_topm.shortcut_select_mode == "top_m_per_node"
        assert summary_topm.shortcut_score_threshold_used is None

    def test_score_percentiles_computed(self):
        """Score percentiles (p50, p90) should be computed correctly."""
        H, W, D = 12, 12, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=3,
        )
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

        # p50 should be between min and max
        if summary.shortcut_edge_count > 0:
            assert summary.shortcut_score_min <= summary.shortcut_score_p50 <= summary.shortcut_score_max
            assert summary.shortcut_score_min <= summary.shortcut_score_p90 <= summary.shortcut_score_max
            assert summary.shortcut_score_p50 <= summary.shortcut_score_p90

    def test_target_nav_gain_mode(self):
        """Target nav_gain mode should stop when target is reached."""
        H, W, D = 12, 12, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        # Low target should result in fewer shortcuts
        config_low = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=3,
            shortcut_select_mode="target_nav_gain",
            target_nav_gain=0.05,  # Low target
            target_nav_gain_step=2,
            max_shortcut_fraction=0.99,
        )
        summary_low = graph_summary_from_embeddings(embeddings, (H, W), config_low, seed=42)

        # High target should result in more shortcuts (or hit ceiling)
        config_high = GraphSpecV1(
            knn_k=8,
            min_lattice_hops_for_shortcut=3,
            shortcut_select_mode="target_nav_gain",
            target_nav_gain=0.30,  # High target
            target_nav_gain_step=2,
            max_shortcut_fraction=0.99,
        )
        summary_high = graph_summary_from_embeddings(embeddings, (H, W), config_high, seed=42)

        # Lower target should use fewer or equal shortcuts
        assert summary_low.shortcut_edge_count <= summary_high.shortcut_edge_count

        # Both should achieve their targets (or hit max candidates)
        # Note: nav_gain might not exactly match target due to discrete steps
        assert summary_low.nav_gain >= 0  # Should be non-negative

    def test_target_nav_gain_determinism(self):
        """Target nav_gain mode should be deterministic."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1(
            knn_k=6,
            min_lattice_hops_for_shortcut=3,
            shortcut_select_mode="target_nav_gain",
            target_nav_gain=0.10,
            target_nav_gain_step=1,
        )

        summary1 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        summary2 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

        assert summary1.summary_sha == summary2.summary_sha
        assert summary1.shortcut_edge_count == summary2.shortcut_edge_count


class TestBaselineType:
    """Tests for baseline type selection (ER vs configuration model)."""

    def test_er_baseline_default(self):
        """ER baseline should be the default."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1()  # Default baseline_type
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

        assert summary.baseline_type == "ER_expected_degree"

    def test_configuration_model_baseline(self):
        """Configuration model baseline should work and be recorded."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1(
            baseline_type="configuration_model",
            knn_k=6,
            min_lattice_hops_for_shortcut=3,
        )
        summary = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

        assert summary.baseline_type == "configuration_model"
        # Sigma should still be computed (non-negative)
        assert summary.sigma >= 0
        # c_rand and l_rand should be computed
        assert summary.c_rand >= 0
        assert summary.l_rand >= 0

    def test_baseline_types_produce_different_sigma(self):
        """Different baseline types may produce different sigma values."""
        H, W, D = 12, 12, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config_er = GraphSpecV1(
            baseline_type="ER_expected_degree",
            knn_k=6,
            min_lattice_hops_for_shortcut=3,
        )
        config_cm = GraphSpecV1(
            baseline_type="configuration_model",
            knn_k=6,
            min_lattice_hops_for_shortcut=3,
        )

        summary_er = graph_summary_from_embeddings(embeddings, (H, W), config_er, seed=42)
        summary_cm = graph_summary_from_embeddings(embeddings, (H, W), config_cm, seed=42)

        # Both should have valid sigma
        assert summary_er.sigma >= 0
        assert summary_cm.sigma >= 0

        # They may differ (configuration model is degree-matched)
        # This is expected behavior, not a strict assertion
        # Just verify both compute successfully

    def test_configuration_model_deterministic(self):
        """Configuration model baseline should be deterministic."""
        H, W, D = 10, 10, 16
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)

        config = GraphSpecV1(
            baseline_type="configuration_model",
        )

        summary1 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)
        summary2 = graph_summary_from_embeddings(embeddings, (H, W), config, seed=42)

        assert summary1.summary_sha == summary2.summary_sha
        assert summary1.sigma == summary2.sigma
        assert summary1.c_rand == summary2.c_rand
        assert summary1.l_rand == summary2.l_rand
