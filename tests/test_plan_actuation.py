"""Integration test for plan actuation.

Tests that applying a SemanticUpdatePlanV1 actually shifts the sampling distribution.
"""
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from src.contracts.schemas import (
    SemanticUpdatePlanV1,
    TaskGraphOp,
    PlanOpType,
    TaskSamplerOverrides,
)
from src.orchestrator.plan_applier import PlanApplier


def sample_with_weights(weights: dict, n: int, seed: int) -> list:
    """Simulate weighted sampling."""
    import random
    rng = random.Random(seed)
    families = list(weights.keys())
    probs = [weights[f] / sum(weights.values()) for f in families]
    samples = []
    for _ in range(n):
        r = rng.random()
        cumsum = 0.0
        for f, p in zip(families, probs):
            cumsum += p
            if r <= cumsum:
                samples.append(f)
                break
    return samples


class TestPlanActuation:
    """Tests for plan actuation shifting sampling distribution."""

    def test_plan_load_validates_schema(self, tmp_path):
        """Test that plan loading validates strict schema."""
        plan = SemanticUpdatePlanV1(
            plan_id="test_plan",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=0.7),
            ],
        )
        plan_path = tmp_path / "plan.json"
        with open(plan_path, "w") as f:
            json.dump(plan.model_dump(mode="json"), f)

        applier = PlanApplier(plan_path=str(plan_path))
        result = applier.load()

        assert result.applied
        assert result.plan_id == "test_plan"
        assert applier.task_overrides.weights["manipulation"] == 0.7

    def test_plan_apply_shifts_distribution(self, tmp_path):
        """Test that applying a plan shifts sampling distribution."""
        # Baseline plan: equal weights
        baseline = SemanticUpdatePlanV1(
            plan_id="baseline",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="A", weight=0.5),
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="B", weight=0.5),
            ],
        )
        baseline_path = tmp_path / "baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline.model_dump(mode="json"), f)

        # Updated plan: biased weights
        updated = SemanticUpdatePlanV1(
            plan_id="updated",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="A", weight=0.9),
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="B", weight=0.1),
            ],
        )
        updated_path = tmp_path / "updated.json"
        with open(updated_path, "w") as f:
            json.dump(updated.model_dump(mode="json"), f)

        # Apply baseline
        applier = PlanApplier(plan_path=str(baseline_path))
        applier.load()
        baseline_samples = sample_with_weights(applier.task_overrides.weights, 1000, seed=42)
        baseline_counter = Counter(baseline_samples)

        # Apply updated (use new applier to force fresh load)
        applier2 = PlanApplier(plan_path=str(updated_path))
        applier2.load()
        updated_samples = sample_with_weights(applier2.task_overrides.weights, 1000, seed=42)
        updated_counter = Counter(updated_samples)

        # Verify shift
        baseline_ratio = baseline_counter["A"] / baseline_counter["B"]
        updated_ratio = updated_counter["A"] / updated_counter["B"]

        # Baseline should be ~1.0, updated should be ~9.0
        assert 0.7 < baseline_ratio < 1.3, f"Baseline ratio should be ~1, got {baseline_ratio}"
        assert updated_ratio > 5.0, f"Updated ratio should be >5, got {updated_ratio}"

    def test_plan_sha_changes_on_update(self, tmp_path):
        """Test that plan SHA changes when content changes."""
        plan1 = SemanticUpdatePlanV1(
            plan_id="plan1",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="X", weight=0.5),
            ],
        )
        plan2 = SemanticUpdatePlanV1(
            plan_id="plan2",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="X", weight=0.8),
            ],
        )

        sha1 = plan1.sha256()
        sha2 = plan2.sha256()

        assert sha1 != sha2, "Different plans should have different SHAs"

    def test_plan_applier_detects_unchanged(self, tmp_path):
        """Test that applier detects unchanged plan."""
        plan = SemanticUpdatePlanV1(
            plan_id="static",
            task_graph_changes=[],
        )
        plan_path = tmp_path / "plan.json"
        with open(plan_path, "w") as f:
            json.dump(plan.model_dump(mode="json"), f)

        applier = PlanApplier(plan_path=str(plan_path))
        result1 = applier.load()
        result2 = applier.load()

        assert result1.applied
        assert not result2.applied
        assert "unchanged" in (result2.error or "").lower()

    def test_disabled_task_family(self, tmp_path):
        """Test that disabled task families are tracked."""
        plan = SemanticUpdatePlanV1(
            plan_id="disable_test",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="A", weight=1.0),
                TaskGraphOp(op=PlanOpType.DISABLE, task_family="B"),
            ],
        )
        plan_path = tmp_path / "plan.json"
        with open(plan_path, "w") as f:
            json.dump(plan.model_dump(mode="json"), f)

        applier = PlanApplier(plan_path=str(plan_path))
        applier.load()

        assert "B" in applier.task_overrides.disabled
        assert applier.task_overrides.weights["A"] == 1.0
