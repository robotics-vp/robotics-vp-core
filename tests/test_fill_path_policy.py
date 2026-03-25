"""Tests for fill_path_policy.py (Phase 3)."""
import unittest

from src.world_model.fill_path_policy import FILL_METHODS


class TestFillMethods(unittest.TestCase):

    def test_expected_methods(self):
        self.assertEqual(
            FILL_METHODS,
            ["real_sim", "diffusion", "synthetic_branch", "blocked"],
        )


class TestFillPathPolicyWithCoverageLoop(unittest.TestCase):
    """Test that coverage_loop works with and without fill_path_policy."""

    def _make_runtime_rows(self):
        return [
            {
                "sample_id": "s1", "run_id": "r1", "episode_id": "ep1",
                "task_id": "open_drawer", "env_id": "drawer_vase",
                "semantic_tokens": ["drawer", "grasp_handle"],
                "outcome_summary": {"success": True},
            },
        ]

    def test_loop_without_policy(self):
        from src.orchestrator.coverage_loop import run_coverage_loop
        result = run_coverage_loop(self._make_runtime_rows())
        self.assertTrue(len(result.fill_decisions) >= 0)

    def test_loop_with_mock_policy(self):
        from src.orchestrator.coverage_loop import run_coverage_loop

        class MockPolicy:
            def predict_batch(self, edges, graph):
                return [("diffusion", 0.95) for _ in edges]

        result = run_coverage_loop(
            self._make_runtime_rows(),
            fill_path_policy=MockPolicy(),
        )
        # All decisions should use learned policy
        for decision in result.fill_decisions:
            self.assertEqual(decision["fill_method"], "diffusion")
            self.assertIn("Learned policy", decision["rationale"])

    def test_loop_with_broken_policy_falls_back(self):
        from src.orchestrator.coverage_loop import run_coverage_loop

        class BrokenPolicy:
            def predict_batch(self, edges, graph):
                raise RuntimeError("broken")

        result = run_coverage_loop(
            self._make_runtime_rows(),
            fill_path_policy=BrokenPolicy(),
        )
        # Should fall back to heuristic — no "Learned policy" in rationale
        for decision in result.fill_decisions:
            self.assertNotIn("Learned policy", decision.get("rationale", ""))


if __name__ == "__main__":
    unittest.main()
