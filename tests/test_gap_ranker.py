"""Tests for gap_ranker.py (Phase 2)."""
import unittest


from src.world_model.gap_ranker import (
    GapFeatureExtractor,
    EDGE_TYPES,
    NODE_TYPES,
)
from src.world_model.semantic_coverage_graph import (
    CoverageEdge,
    CoverageNode,
    SemanticCoverageGraph,
)


def _make_graph():
    nodes = [
        CoverageNode(node_id="task:open_drawer", node_type="task", label="open_drawer"),
        CoverageNode(node_id="skill:grasp", node_type="skill", label="grasp"),
        CoverageNode(node_id="prim:force_control", node_type="env_primitive", label="force_control"),
    ]
    edges = [
        CoverageEdge(
            source_id="task:open_drawer", target_id="skill:grasp",
            edge_type="covers", evidence_count=0,
            economic_priority=0.8, trust_priority=0.5, promotion_readiness=0.7,
        ),
        CoverageEdge(
            source_id="skill:grasp", target_id="prim:force_control",
            edge_type="requires", evidence_count=3,
            economic_priority=0.3, trust_priority=0.9, promotion_readiness=0.6,
        ),
    ]
    return SemanticCoverageGraph(nodes=nodes, edges=edges)


class TestGapFeatureExtractor(unittest.TestCase):

    def test_feature_dim(self):
        ext = GapFeatureExtractor()
        self.assertEqual(ext.FEATURE_DIM, 7 + len(EDGE_TYPES) + 2 * len(NODE_TYPES))

    def test_single_edge_extraction(self):
        ext = GapFeatureExtractor()
        graph = _make_graph()
        edge = graph.edges[0]
        fv = ext(edge, graph)
        self.assertEqual(fv.dim, ext.FEATURE_DIM)
        # economic_priority should be at index 2
        self.assertAlmostEqual(fv.raw[2], 0.8)

    def test_batch_extraction_shape(self):
        ext = GapFeatureExtractor()
        graph = _make_graph()
        batch = ext.extract_batch(graph.edges, graph)
        self.assertEqual(batch.shape, (2, ext.FEATURE_DIM))

    def test_edge_type_one_hot(self):
        ext = GapFeatureExtractor()
        graph = _make_graph()
        edge = graph.edges[0]  # edge_type="covers"
        fv = ext(edge, graph)
        offset = 7
        covers_idx = EDGE_TYPES.index("covers")
        self.assertEqual(fv.raw[offset + covers_idx], 1.0)
        # Other edge types should be 0
        for i in range(len(EDGE_TYPES)):
            if i != covers_idx:
                self.assertEqual(fv.raw[offset + i], 0.0)

    def test_node_type_one_hot(self):
        ext = GapFeatureExtractor()
        graph = _make_graph()
        edge = graph.edges[0]  # task -> skill
        fv = ext(edge, graph)
        src_offset = 7 + len(EDGE_TYPES)
        task_idx = NODE_TYPES.index("task")
        self.assertEqual(fv.raw[src_offset + task_idx], 1.0)

    def test_from_outcome_record(self):
        """Test extraction from a mock outcome record."""
        class MockRecord:
            gap_features = {"economic_priority": 0.7, "trust_priority": 0.3, "readiness": 0.5}
            pre_evidence_count = 2

        fv = GapFeatureExtractor.from_outcome_record(MockRecord())
        self.assertEqual(fv.dim, GapFeatureExtractor.FEATURE_DIM)
        self.assertAlmostEqual(fv.raw[2], 0.7)  # economic_priority
        self.assertAlmostEqual(fv.raw[3], 0.3)  # trust_priority

    def test_null_graph_safe(self):
        """Extraction should work even with graph=None."""
        ext = GapFeatureExtractor()
        edge = CoverageEdge(
            source_id="a", target_id="b", edge_type="covers",
            evidence_count=0, economic_priority=0.5,
        )
        fv = ext(edge, None)
        self.assertEqual(fv.dim, ext.FEATURE_DIM)


class TestRankGapsWithRanker(unittest.TestCase):

    def test_heuristic_fallback(self):
        """rank_gaps with gap_ranker=None should use heuristic as before."""
        graph = _make_graph()
        gaps = graph.rank_gaps()
        self.assertEqual(len(gaps), 1)  # only 1 missing edge

    def test_rank_gaps_with_mock_ranker(self):
        """rank_gaps should use a mock ranker when provided."""
        class MockRanker:
            def rank_edges(self, edges, graph):
                return [(e, float(i)) for i, e in enumerate(edges)]

        graph = _make_graph()
        gaps = graph.rank_gaps(gap_ranker=MockRanker())
        self.assertEqual(len(gaps), 1)

    def test_ranker_exception_falls_back(self):
        """If ranker raises, should fall back to heuristic."""
        class BrokenRanker:
            def rank_edges(self, edges, graph):
                raise RuntimeError("broken")

        graph = _make_graph()
        gaps = graph.rank_gaps(gap_ranker=BrokenRanker())
        self.assertEqual(len(gaps), 1)


if __name__ == "__main__":
    unittest.main()
