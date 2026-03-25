"""Tests for fill_outcome_store.py (Phase 1)."""
import os
import tempfile
import unittest

from src.world_model.fill_outcome_store import FillOutcomeRecord, FillOutcomeStore


def _make_record(edge_key="A -> B", method="diffusion", delta=0.1, quality=0.8, wall=1.0):
    return FillOutcomeRecord(
        edge_key=edge_key,
        fill_method=method,
        gap_features={"economic_priority": 0.5, "trust_priority": 0.6},
        pre_evidence_count=0,
        post_evidence_count=3,
        coverage_delta=delta,
        wall_time_s=wall,
        quality_score=quality,
        timestamp="2026-01-01T00:00:00Z",
    )


class TestFillOutcomeStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "outcomes.jsonl")
        self.store = FillOutcomeStore(self.store_path)

    def test_append_and_load(self):
        r = _make_record()
        self.store.append(r)
        loaded = self.store.load_all()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].edge_key, "A -> B")

    def test_append_batch(self):
        records = [_make_record(edge_key=f"E{i} -> F{i}") for i in range(5)]
        self.store.append_batch(records)
        self.assertEqual(self.store.record_count(), 5)

    def test_load_for_edge(self):
        self.store.append(_make_record(edge_key="X -> Y"))
        self.store.append(_make_record(edge_key="Z -> W"))
        self.assertEqual(len(self.store.load_for_edge("X -> Y")), 1)

    def test_load_for_method(self):
        self.store.append(_make_record(method="real_sim"))
        self.store.append(_make_record(method="diffusion"))
        self.store.append(_make_record(method="diffusion"))
        self.assertEqual(len(self.store.load_for_method("diffusion")), 2)

    def test_summary(self):
        self.store.append(_make_record(method="diffusion", delta=0.2, quality=1.0))
        self.store.append(_make_record(method="real_sim", delta=0.3, quality=0.9))
        summary = self.store.summary()
        self.assertEqual(summary["total_records"], 2)
        self.assertIn("diffusion", summary["methods"])
        self.assertIn("real_sim", summary["methods"])

    def test_empty_store(self):
        summary = self.store.summary()
        self.assertEqual(summary["total_records"], 0)
        self.assertEqual(self.store.load_all(), [])

    def test_marginal_value(self):
        r = _make_record(delta=0.5, quality=2.0, wall=1.0)
        self.assertAlmostEqual(r.marginal_value, 1.0)

    def test_marginal_value_min_cost(self):
        r = _make_record(delta=0.5, quality=2.0, wall=0.01)
        # wall_time clipped to 0.1 minimum
        self.assertAlmostEqual(r.marginal_value, 0.5 * 2.0 / 0.1, places=4)

    def test_serialization_round_trip(self):
        r = _make_record()
        d = r.to_dict()
        r2 = FillOutcomeRecord.from_dict(d)
        self.assertEqual(r.edge_key, r2.edge_key)
        self.assertEqual(r.fill_method, r2.fill_method)
        self.assertAlmostEqual(r.coverage_delta, r2.coverage_delta)

    def test_auto_timestamp(self):
        r = FillOutcomeRecord(
            edge_key="A -> B", fill_method="diffusion",
            gap_features={}, pre_evidence_count=0, post_evidence_count=1,
            coverage_delta=0.1, wall_time_s=1.0, quality_score=0.5,
        )
        self.assertEqual(r.timestamp, "")
        self.store.append(r)
        loaded = self.store.load_all()
        self.assertNotEqual(loaded[0].timestamp, "")


if __name__ == "__main__":
    unittest.main()
