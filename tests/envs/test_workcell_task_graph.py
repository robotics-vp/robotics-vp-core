"""Tests for workcell task graph specifications."""
from __future__ import annotations

import pytest

from src.envs.workcell_env.tasks.task_base import (
    ActionStepSpec,
    TaskGraphSpec,
    TaskSpec,
)
from src.envs.workcell_env.compiler import WorkcellTaskCompiler


class TestTaskSpec:
    """Tests for TaskSpec dataclass."""

    def test_create_task_spec(self) -> None:
        """Test creating a task specification."""
        spec = TaskSpec(
            task_id="kitting_001",
            task_type="ASSEMBLY",
            success_metric="all_placed",
            reward_type="DENSE",
            time_limit=100,
            parameters={"num_items": 5},
        )

        assert spec.task_id == "kitting_001"
        assert spec.task_type == "ASSEMBLY"
        assert spec.parameters["num_items"] == 5

    def test_task_spec_serialization(self) -> None:
        """Test TaskSpec serialization roundtrip."""
        spec = TaskSpec(
            task_id="test",
            task_type="SORTING",
            success_metric="sorted",
            reward_type="SPARSE",
            time_limit=50,
        )

        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)

        assert spec.task_id == restored.task_id
        assert spec.task_type == restored.task_type


class TestActionStepSpec:
    """Tests for ActionStepSpec dataclass."""

    def test_create_action_step(self) -> None:
        """Test creating an action step."""
        step = ActionStepSpec(
            step_id="pick_0",
            action_type="PICK",
            target_object_id="part_0",
            constraints={"max_force": 10.0},
        )

        assert step.step_id == "pick_0"
        assert step.action_type == "PICK"
        assert step.constraints["max_force"] == 10.0

    def test_action_step_serialization(self) -> None:
        """Test ActionStepSpec serialization."""
        step = ActionStepSpec(
            step_id="insert_1",
            action_type="INSERT",
            target_object_id="hole_0",
            constraints={"tolerance_mm": 1.5},
        )

        d = step.to_dict()
        assert d["step_id"] == "insert_1"
        assert d["action_type"] == "INSERT"

        restored = ActionStepSpec.from_dict(d)
        assert step.step_id == restored.step_id


class TestTaskGraphSpec:
    """Tests for TaskGraphSpec dataclass."""

    def test_create_empty_graph(self) -> None:
        """Test creating an empty task graph."""
        graph = TaskGraphSpec(
            nodes=[],
            edges=[],
            entry_node="",
            exit_nodes=[],
        )

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_create_linear_graph(self) -> None:
        """Test creating a linear task graph."""
        nodes = [
            ActionStepSpec(step_id="pick", action_type="PICK", target_object_id="part"),
            ActionStepSpec(step_id="place", action_type="PLACE", target_object_id="tray"),
        ]
        edges = [("pick", "place")]

        graph = TaskGraphSpec(
            nodes=nodes,
            edges=edges,
            entry_node="pick",
            exit_nodes=["place"],
        )

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.entry_node == "pick"

    def test_task_graph_serialization(self) -> None:
        """Test TaskGraphSpec serialization roundtrip."""
        nodes = [
            ActionStepSpec(step_id="a", action_type="PICK", target_object_id="p1"),
            ActionStepSpec(step_id="b", action_type="PLACE", target_object_id="t1"),
            ActionStepSpec(step_id="c", action_type="INSPECT", target_object_id="p1"),
        ]
        edges = [("a", "b"), ("b", "c")]

        graph = TaskGraphSpec(
            nodes=nodes,
            edges=edges,
            entry_node="a",
            exit_nodes=["c"],
        )

        d = graph.to_dict()
        restored = TaskGraphSpec.from_dict(d)

        assert len(restored.nodes) == 3
        assert len(restored.edges) == 2
        assert restored.entry_node == "a"


class TestWorkcellTaskCompiler:
    """Tests for WorkcellTaskCompiler."""

    @pytest.fixture
    def compiler(self) -> WorkcellTaskCompiler:
        """Create a compiler instance."""
        return WorkcellTaskCompiler(default_seed=42)

    def test_infer_kitting_task(self, compiler: WorkcellTaskCompiler) -> None:
        """Test inferring kitting task from prompt."""
        result = compiler.compile_from_prompt("Pack 6 items into a box")
        assert result.inferred_task_type == "kitting"
        assert len(result.task_graph.nodes) > 0

    def test_infer_sorting_task(self, compiler: WorkcellTaskCompiler) -> None:
        """Test inferring sorting task from prompt."""
        result = compiler.compile_from_prompt("Sort widgets by color into bins")
        assert result.inferred_task_type == "sorting"
        assert result.config.conveyor_enabled is True

    def test_infer_peg_in_hole_task(self, compiler: WorkcellTaskCompiler) -> None:
        """Test inferring peg-in-hole task from prompt."""
        result = compiler.compile_from_prompt("Insert peg into hole with 1mm tolerance")
        assert result.inferred_task_type == "peg_in_hole"
        assert result.config.tolerance_mm == 1.0

    def test_infer_assembly_task(self, compiler: WorkcellTaskCompiler) -> None:
        """Test inferring assembly task from prompt."""
        result = compiler.compile_from_prompt("Assemble bracket with 2 screws")
        assert result.inferred_task_type == "assembly"

    def test_extract_num_items(self, compiler: WorkcellTaskCompiler) -> None:
        """Test extracting number of items from prompt."""
        result = compiler.compile_from_prompt("Pack 12 parts into tray")
        assert result.config.num_parts == 12

    def test_extract_tolerance(self, compiler: WorkcellTaskCompiler) -> None:
        """Test extracting tolerance from prompt."""
        result = compiler.compile_from_prompt("Insert with 0.5mm tolerance")
        assert result.config.tolerance_mm == 0.5

    def test_extract_num_bins(self, compiler: WorkcellTaskCompiler) -> None:
        """Test extracting number of bins from prompt."""
        result = compiler.compile_from_prompt("Sort into 5 bins")
        assert result.config.num_bins == 5

    def test_compile_from_dict(self, compiler: WorkcellTaskCompiler) -> None:
        """Test compiling from structured dict."""
        spec_dict = {
            "task_type": "kitting",
            "config": {"num_parts": 8},
            "parameters": {"num_items": 8},
        }
        result = compiler.compile_from_dict(spec_dict)

        assert result.inferred_task_type == "kitting"
        assert result.config.num_parts == 8

    def test_deterministic_compilation(self, compiler: WorkcellTaskCompiler) -> None:
        """Test that compilation is deterministic with same seed."""
        result1 = compiler.compile_from_prompt("Pack 5 items", seed=42)
        result2 = compiler.compile_from_prompt("Pack 5 items", seed=42)

        assert result1.scene_spec.workcell_id == result2.scene_spec.workcell_id
        assert len(result1.task_graph.nodes) == len(result2.task_graph.nodes)

    def test_task_graph_structure(self, compiler: WorkcellTaskCompiler) -> None:
        """Test that compiled task graph has valid structure."""
        result = compiler.compile_from_prompt("Pack 3 items into tray")

        graph = result.task_graph
        node_ids = {n.step_id for n in graph.nodes}

        # Entry node should exist
        assert graph.entry_node in node_ids or graph.entry_node == ""

        # Exit nodes should exist
        for exit_node in graph.exit_nodes:
            assert exit_node in node_ids

        # Edge endpoints should exist
        for src, dst in graph.edges:
            assert src in node_ids
            assert dst in node_ids
