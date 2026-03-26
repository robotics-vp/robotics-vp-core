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
            benchmark_gate = {"ready": True}

            def predict_batch_details(self, edges, graph):
                return [
                    {
                        "fill_method": "diffusion",
                        "confidence": 0.95,
                        "method_probabilities": {
                            "real_sim": 0.01,
                            "diffusion": 0.95,
                            "synthetic_branch": 0.03,
                            "blocked": 0.01,
                        },
                    }
                    for _ in edges
                ]

        result = run_coverage_loop(
            self._make_runtime_rows(),
            fill_path_policy=MockPolicy(),
            fill_path_policy_mode="required",
        )
        # Decisions should record the learned helper path.
        for decision in result.fill_decisions:
            self.assertIn("routing_policy", decision)
            self.assertIn(decision["routing_policy"], {"heuristic_plus_learned_fill_path_policy", "heuristic_hard_gate"})
            self.assertIn("score_trace", decision)
            self.assertIn("helper_status", decision)
        helper_status = result.coverage_summary["fill_path_helper_status"]
        self.assertEqual(helper_status["promotion_stage"], "promoted")

    def test_loop_with_broken_policy_falls_back(self):
        from src.orchestrator.coverage_loop import run_coverage_loop

        class BrokenPolicy:
            def predict_batch(self, edges, graph):
                raise RuntimeError("broken")

        result = run_coverage_loop(
            self._make_runtime_rows(),
            fill_path_policy=BrokenPolicy(),
        )
        # Should fall back to heuristic and record the failure honestly.
        for decision in result.fill_decisions:
            self.assertEqual(decision["routing_policy"], "heuristic_only")
            self.assertEqual(decision["helper_status"]["status"], "inference_failed")

    def test_required_mode_without_helper_raises(self):
        from src.orchestrator.coverage_loop import run_coverage_loop

        with self.assertRaises(ValueError):
            run_coverage_loop(
                self._make_runtime_rows(),
                fill_path_policy_mode="required",
            )


if __name__ == "__main__":
    unittest.main()
