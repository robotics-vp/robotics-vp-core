"""
Task compiler for workcell environments.

Converts natural language prompts or structured dicts into
WorkcellEnvConfig + WorkcellSceneSpec + TaskGraphSpec.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.envs.workcell_env.config import WorkcellEnvConfig, PRESETS
from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec
from src.envs.workcell_env.scene.generators import WorkcellSceneGenerator
from src.envs.workcell_env.tasks.task_base import TaskSpec, TaskGraphSpec, ActionStepSpec


@dataclass
class CompilationResult:
    """Result of compiling a prompt or spec into workcell configuration."""
    config: WorkcellEnvConfig
    scene_spec: WorkcellSceneSpec
    task_graph: TaskGraphSpec
    inferred_task_type: str
    raw_prompt: Optional[str] = None


class WorkcellTaskCompiler:
    """
    Compiles natural language prompts or structured specs into workcell configurations.

    Supports prompts like:
    - "Pack 6 items from conveyor into box; reject damaged items"
    - "Assemble bracket A into base B; fasten with 2 screws; inspect alignment"
    - "Sort 20 widgets into 3 bins by color"
    - "Insert peg into hole with 1mm tolerance"
    """

    # Patterns for extracting parameters from prompts
    PATTERNS = {
        "num_items": re.compile(r"(\d+)\s*(?:items?|parts?|widgets?|pieces?|objects?)", re.I),
        "tolerance": re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm|millimeter)", re.I),
        "num_bins": re.compile(r"(\d+)\s*(?:bins?|boxes?|containers?|categories?)", re.I),
        "num_screws": re.compile(r"(\d+)\s*(?:screws?|fasteners?|bolts?)", re.I),
        "num_stations": re.compile(r"(\d+)\s*(?:stations?|benches?|cells?)", re.I),
    }

    TASK_KEYWORDS = {
        "kitting": ["kit", "pack", "box", "tray", "arrange", "package"],
        "sorting": ["sort", "classify", "separate", "categorize", "color", "type"],
        "assembly": ["assemble", "fasten", "attach", "mount", "screw", "bolt"],
        "inspection": ["inspect", "check", "verify", "examine", "detect", "quality"],
        "peg_in_hole": ["peg", "hole", "insert", "tolerance", "press-fit"],
        "bin_picking": ["pick from bin", "bin picking", "cluttered"],
        "conveyor": ["conveyor", "belt", "line", "moving"],
    }

    PRESET_MAP = {
        "kitting": "assembly_bench_simple",
        "sorting": "conveyor_sorting",
        "inspection": "inspection_simple",
        "assembly": "assembly_bench_simple",
        "peg_in_hole": "assembly_bench_simple",
        "bin_picking": "assembly_bench_simple",
        "conveyor": "conveyor_sorting",
    }

    def __init__(self, default_seed: int = 42):
        self.default_seed = default_seed
        self.scene_generator = WorkcellSceneGenerator()

    def compile_from_prompt(
        self, prompt: str, seed: Optional[int] = None
    ) -> CompilationResult:
        """
        Compile a natural language prompt into workcell configuration.

        Args:
            prompt: Natural language task description
            seed: Random seed for scene generation

        Returns:
            CompilationResult with config, scene_spec, task_graph
        """
        seed = seed if seed is not None else self.default_seed
        task_type = self._infer_task_type(prompt)
        params = self._extract_parameters(prompt)
        config = self._build_config(task_type, params)
        scene_spec = self.scene_generator.generate(config, seed)
        task_graph = self._build_task_graph(task_type, params, scene_spec)

        return CompilationResult(
            config=config,
            scene_spec=scene_spec,
            task_graph=task_graph,
            inferred_task_type=task_type,
            raw_prompt=prompt,
        )

    def compile_from_dict(
        self, spec_dict: Dict[str, Any], seed: Optional[int] = None
    ) -> CompilationResult:
        """
        Compile a structured specification dict into workcell configuration.

        Args:
            spec_dict: Dict with keys: task_type, config, parameters
            seed: Random seed for scene generation

        Returns:
            CompilationResult with config, scene_spec, task_graph
        """
        seed = seed if seed is not None else self.default_seed
        task_type = spec_dict.get("task_type", "kitting")

        config_data = spec_dict.get("config", {})
        if config_data:
            config = WorkcellEnvConfig.from_dict(config_data)
        else:
            preset_name = self.PRESET_MAP.get(task_type, "assembly_bench_simple")
            config = PRESETS.get(preset_name, WorkcellEnvConfig())

        scene_spec = self.scene_generator.generate(config, seed)
        task_graph = self._build_task_graph(
            task_type, spec_dict.get("parameters", {}), scene_spec
        )

        return CompilationResult(
            config=config,
            scene_spec=scene_spec,
            task_graph=task_graph,
            inferred_task_type=task_type,
        )

    def _infer_task_type(self, prompt: str) -> str:
        """Infer task type from prompt keywords."""
        prompt_lower = prompt.lower()

        # Check for specific task keywords
        for task_type, keywords in self.TASK_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                return task_type

        return "kitting"  # default fallback

    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract numeric parameters from prompt using regex patterns."""
        params: Dict[str, Any] = {}

        for name, pattern in self.PATTERNS.items():
            match = pattern.search(prompt)
            if match:
                value_str = match.group(1)
                if "." in value_str:
                    params[name] = float(value_str)
                else:
                    params[name] = int(value_str)

        return params

    def _build_config(
        self, task_type: str, params: Dict[str, Any]
    ) -> WorkcellEnvConfig:
        """Build WorkcellEnvConfig from task type and extracted parameters."""
        # Start from preset
        preset_name = self.PRESET_MAP.get(task_type, "assembly_bench_simple")
        base = PRESETS.get(preset_name, WorkcellEnvConfig())

        # Build overrides from params
        overrides: Dict[str, Any] = {}

        if "num_items" in params:
            overrides["num_parts"] = params["num_items"]
        if "num_bins" in params:
            overrides["num_bins"] = params["num_bins"]
        if "tolerance" in params:
            overrides["tolerance_mm"] = params["tolerance"]
        if "num_stations" in params:
            overrides["num_stations"] = params["num_stations"]
        if "num_screws" in params:
            overrides["tool_changes_required"] = 1  # screwdriver needed

        # Task-specific overrides
        if task_type in ("sorting", "conveyor"):
            overrides["conveyor_enabled"] = True
            overrides["topology_type"] = "CONVEYOR_LINE"
        elif task_type == "inspection":
            overrides["topology_type"] = "INSPECTION_STATION"
        elif task_type in ("assembly", "peg_in_hole"):
            overrides["topology_type"] = "ASSEMBLY_BENCH"

        # Merge and return
        merged = base.to_dict()
        merged.update(overrides)
        return WorkcellEnvConfig.from_dict(merged)

    def _build_task_graph(
        self,
        task_type: str,
        params: Dict[str, Any],
        scene_spec: WorkcellSceneSpec,
    ) -> TaskGraphSpec:
        """Build task graph based on task type and scene."""
        if task_type == "kitting":
            return self._build_kitting_graph(params, scene_spec)
        elif task_type == "peg_in_hole":
            return self._build_peg_in_hole_graph(params, scene_spec)
        elif task_type == "sorting":
            return self._build_sorting_graph(params, scene_spec)
        elif task_type == "assembly":
            return self._build_assembly_graph(params, scene_spec)
        elif task_type == "inspection":
            return self._build_inspection_graph(params, scene_spec)
        else:
            return self._build_generic_graph(params, scene_spec)

    def _build_kitting_graph(
        self, params: Dict[str, Any], scene_spec: WorkcellSceneSpec
    ) -> TaskGraphSpec:
        """Build kitting task graph: pick items, place in tray."""
        num_items = params.get("num_items", len(scene_spec.parts))
        num_items = min(num_items, len(scene_spec.parts)) if scene_spec.parts else num_items

        nodes: List[ActionStepSpec] = []
        edges: List[tuple[str, str]] = []

        for i in range(num_items):
            pick_id = f"pick_{i}"
            place_id = f"place_{i}"

            part_id = scene_spec.parts[i].id if i < len(scene_spec.parts) else f"part_{i}"
            target_id = scene_spec.containers[0].id if scene_spec.containers else "tray_0"

            nodes.append(ActionStepSpec(
                step_id=pick_id,
                action_type="PICK",
                target_object_id=part_id,
            ))
            nodes.append(ActionStepSpec(
                step_id=place_id,
                action_type="PLACE",
                target_object_id=target_id,
            ))

            edges.append((pick_id, place_id))
            if i > 0:
                edges.append((f"place_{i-1}", pick_id))

        entry = "pick_0" if nodes else ""
        exit_nodes = [f"place_{num_items-1}"] if nodes else []

        return TaskGraphSpec(
            nodes=nodes, edges=edges, entry_node=entry, exit_nodes=exit_nodes
        )

    def _build_peg_in_hole_graph(
        self, params: Dict[str, Any], scene_spec: WorkcellSceneSpec
    ) -> TaskGraphSpec:
        """Build peg-in-hole task graph: pick peg, insert into hole."""
        tolerance = params.get("tolerance", 2.0)

        nodes = [
            ActionStepSpec(
                step_id="pick_peg",
                action_type="PICK",
                target_object_id="peg_0",
            ),
            ActionStepSpec(
                step_id="insert_peg",
                action_type="INSERT",
                target_object_id="hole_0",
                constraints={"tolerance_mm": tolerance},
            ),
        ]
        edges = [("pick_peg", "insert_peg")]

        return TaskGraphSpec(
            nodes=nodes,
            edges=edges,
            entry_node="pick_peg",
            exit_nodes=["insert_peg"],
        )

    def _build_sorting_graph(
        self, params: Dict[str, Any], scene_spec: WorkcellSceneSpec
    ) -> TaskGraphSpec:
        """Build sorting task graph: pick from conveyor, place in correct bin."""
        num_items = params.get("num_items", 10)

        nodes: List[ActionStepSpec] = []
        edges: List[tuple[str, str]] = []

        for i in range(num_items):
            pick_id = f"pick_{i}"
            inspect_id = f"inspect_{i}"
            place_id = f"place_{i}"

            nodes.extend([
                ActionStepSpec(step_id=pick_id, action_type="PICK", target_object_id=f"item_{i}"),
                ActionStepSpec(step_id=inspect_id, action_type="INSPECT", target_object_id=f"item_{i}"),
                ActionStepSpec(step_id=place_id, action_type="PLACE", target_object_id="bin_classified"),
            ])

            edges.extend([
                (pick_id, inspect_id),
                (inspect_id, place_id),
            ])
            if i > 0:
                edges.append((f"place_{i-1}", pick_id))

        entry = "pick_0" if nodes else ""
        exit_nodes = [f"place_{num_items-1}"] if nodes else []

        return TaskGraphSpec(
            nodes=nodes, edges=edges, entry_node=entry, exit_nodes=exit_nodes
        )

    def _build_assembly_graph(
        self, params: Dict[str, Any], scene_spec: WorkcellSceneSpec
    ) -> TaskGraphSpec:
        """Build assembly task graph with optional fastening."""
        num_screws = params.get("num_screws", 0)

        nodes = [
            ActionStepSpec(step_id="pick_part_a", action_type="PICK", target_object_id="part_a"),
            ActionStepSpec(step_id="place_part_a", action_type="PLACE", target_object_id="fixture_0"),
            ActionStepSpec(step_id="pick_part_b", action_type="PICK", target_object_id="part_b"),
            ActionStepSpec(step_id="insert_part_b", action_type="INSERT", target_object_id="part_a"),
        ]
        edges = [
            ("pick_part_a", "place_part_a"),
            ("place_part_a", "pick_part_b"),
            ("pick_part_b", "insert_part_b"),
        ]

        last_step = "insert_part_b"
        for i in range(num_screws):
            fasten_id = f"fasten_{i}"
            nodes.append(ActionStepSpec(
                step_id=fasten_id,
                action_type="FASTEN",
                target_object_id=f"screw_{i}",
            ))
            edges.append((last_step, fasten_id))
            last_step = fasten_id

        return TaskGraphSpec(
            nodes=nodes,
            edges=edges,
            entry_node="pick_part_a",
            exit_nodes=[last_step],
        )

    def _build_inspection_graph(
        self, params: Dict[str, Any], scene_spec: WorkcellSceneSpec
    ) -> TaskGraphSpec:
        """Build inspection task graph."""
        num_items = params.get("num_items", 5)

        nodes: List[ActionStepSpec] = []
        edges: List[tuple[str, str]] = []

        for i in range(num_items):
            pick_id = f"pick_{i}"
            inspect_id = f"inspect_{i}"
            place_id = f"place_{i}"

            nodes.extend([
                ActionStepSpec(step_id=pick_id, action_type="PICK", target_object_id=f"part_{i}"),
                ActionStepSpec(step_id=inspect_id, action_type="INSPECT", target_object_id=f"part_{i}"),
                ActionStepSpec(step_id=place_id, action_type="PLACE", target_object_id="pass_bin"),
            ])

            edges.extend([
                (pick_id, inspect_id),
                (inspect_id, place_id),
            ])
            if i > 0:
                edges.append((f"place_{i-1}", pick_id))

        entry = "pick_0" if nodes else ""
        exit_nodes = [f"place_{num_items-1}"] if nodes else []

        return TaskGraphSpec(
            nodes=nodes, edges=edges, entry_node=entry, exit_nodes=exit_nodes
        )

    def _build_generic_graph(
        self, params: Dict[str, Any], scene_spec: WorkcellSceneSpec
    ) -> TaskGraphSpec:
        """Build a generic single-step task graph."""
        nodes = [
            ActionStepSpec(
                step_id="step_0",
                action_type="PICK",
                target_object_id="part_0",
            )
        ]
        return TaskGraphSpec(
            nodes=nodes,
            edges=[],
            entry_node="step_0",
            exit_nodes=["step_0"],
        )


def compile_workcell_task(
    prompt_or_spec: str | Dict[str, Any],
    seed: int = 42,
) -> CompilationResult:
    """
    Convenience function to compile a workcell task.

    Args:
        prompt_or_spec: Either a natural language prompt (str) or structured spec (dict)
        seed: Random seed for scene generation

    Returns:
        CompilationResult with config, scene_spec, task_graph
    """
    compiler = WorkcellTaskCompiler(default_seed=seed)

    if isinstance(prompt_or_spec, str):
        return compiler.compile_from_prompt(prompt_or_spec, seed)
    else:
        return compiler.compile_from_dict(prompt_or_spec, seed)
