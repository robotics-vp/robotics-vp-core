"""
Workcell orchestration adapter.

Wires WorkcellTaskCompiler into the orchestrator system for
promptable workcell environment generation and episode execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.envs.workcell_env.compiler import (
    CompilationResult,
    WorkcellTaskCompiler,
    compile_workcell_task,
)
from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec
from src.envs.workcell_env.tasks.task_base import TaskGraphSpec
from src.motor_backend.rollout_capture import RolloutBundle

logger = logging.getLogger(__name__)


@dataclass
class WorkcellEpisodeResult:
    """Result of running a workcell episode."""
    success: bool
    total_reward: float
    steps_taken: int
    task_graph_progress: Dict[str, bool]
    metrics: Dict[str, float]
    rollout_bundle: Optional[RolloutBundle] = None


class WorkcellOrchestrationAdapter:
    """
    Adapter that wires WorkcellTaskCompiler into the orchestrator.

    Provides:
    - request_task(): Compile a prompt or spec into workcell configuration
    - run_episode(): Execute an episode using the motor backend
    - run_batch(): Run multiple episodes with different seeds
    """

    def __init__(
        self,
        default_seed: int = 42,
        rollout_base_dir: Optional[Path] = None,
    ):
        self.compiler = WorkcellTaskCompiler(default_seed=default_seed)
        self.default_seed = default_seed
        self.rollout_base_dir = rollout_base_dir or Path("rollouts/workcell")

    def request_task(
        self,
        prompt_or_spec: Union[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> CompilationResult:
        """
        Compile a prompt or spec into workcell configuration.

        Args:
            prompt_or_spec: Natural language prompt or structured spec dict
            seed: Random seed for scene generation

        Returns:
            CompilationResult with config, scene_spec, task_graph
        """
        seed = seed if seed is not None else self.default_seed

        if isinstance(prompt_or_spec, str):
            result = self.compiler.compile_from_prompt(prompt_or_spec, seed)
            logger.info(
                "Compiled prompt to task_type=%s with %d graph nodes",
                result.inferred_task_type,
                len(result.task_graph.nodes),
            )
        else:
            result = self.compiler.compile_from_dict(prompt_or_spec, seed)
            logger.info(
                "Compiled spec to task_type=%s with %d graph nodes",
                result.inferred_task_type,
                len(result.task_graph.nodes),
            )

        return result

    def run_episode(
        self,
        config: WorkcellEnvConfig,
        scene_spec: WorkcellSceneSpec,
        task_graph: TaskGraphSpec,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        capture_rollout: bool = True,
        scenario_id: Optional[str] = None,
    ) -> WorkcellEpisodeResult:
        """
        Run a single workcell episode.

        Args:
            config: Environment configuration
            scene_spec: Scene specification
            task_graph: Task graph specification
            max_steps: Override max steps (uses config default if None)
            seed: Episode seed
            capture_rollout: Whether to capture rollout data
            scenario_id: Scenario ID for rollout capture

        Returns:
            WorkcellEpisodeResult with success, reward, metrics
        """
        from src.envs.workcell_env.env import WorkcellEnv

        seed = seed if seed is not None else self.default_seed
        max_steps = max_steps if max_steps is not None else config.max_steps

        # Create environment
        env = WorkcellEnv(config=config, scene_spec=scene_spec)

        # Run episode
        obs = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # For now, use random actions (policy integration comes later)
            action = env.action_space.sample() if hasattr(env, "action_space") else {}
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        # Extract results
        task_progress = info.get("task_graph_progress", {})
        success = info.get("success", False)

        metrics = {
            "total_reward": total_reward,
            "steps": float(steps),
            "success": float(success),
            "completion_rate": sum(task_progress.values()) / max(len(task_progress), 1),
        }

        # Capture rollout if requested
        rollout_bundle = None
        if capture_rollout and hasattr(env, "get_episode_log"):
            episode_log = env.get_episode_log()
            # Would integrate with rollout_capture here
            logger.debug("Episode log captured with %d steps", steps)

        env.close()

        return WorkcellEpisodeResult(
            success=success,
            total_reward=total_reward,
            steps_taken=steps,
            task_graph_progress=task_progress,
            metrics=metrics,
            rollout_bundle=rollout_bundle,
        )

    def run_from_prompt(
        self,
        prompt: str,
        num_episodes: int = 1,
        seed: Optional[int] = None,
    ) -> list[WorkcellEpisodeResult]:
        """
        Compile a prompt and run episodes.

        Args:
            prompt: Natural language task description
            num_episodes: Number of episodes to run
            seed: Base seed (incremented per episode)

        Returns:
            List of WorkcellEpisodeResult
        """
        seed = seed if seed is not None else self.default_seed

        # Compile once
        compilation = self.request_task(prompt, seed)

        # Run episodes
        results = []
        for i in range(num_episodes):
            episode_seed = seed + i
            result = self.run_episode(
                config=compilation.config,
                scene_spec=compilation.scene_spec,
                task_graph=compilation.task_graph,
                seed=episode_seed,
            )
            results.append(result)

        return results

    def validate_task_graph(self, task_graph: TaskGraphSpec) -> Dict[str, Any]:
        """
        Validate a task graph for structural issues.

        Returns:
            Dict with 'valid' bool and 'issues' list
        """
        issues = []

        # Check for empty graph
        if not task_graph.nodes:
            issues.append("Task graph has no nodes")

        # Check entry node exists
        node_ids = {n.step_id for n in task_graph.nodes}
        if task_graph.entry_node and task_graph.entry_node not in node_ids:
            issues.append(f"Entry node '{task_graph.entry_node}' not in graph")

        # Check exit nodes exist
        for exit_node in task_graph.exit_nodes:
            if exit_node not in node_ids:
                issues.append(f"Exit node '{exit_node}' not in graph")

        # Check edges reference valid nodes
        for src, dst in task_graph.edges:
            if src not in node_ids:
                issues.append(f"Edge source '{src}' not in graph")
            if dst not in node_ids:
                issues.append(f"Edge destination '{dst}' not in graph")

        # Check for cycles (simple DFS)
        if self._has_cycle(task_graph):
            issues.append("Task graph contains a cycle")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def _has_cycle(self, task_graph: TaskGraphSpec) -> bool:
        """Check if task graph has a cycle using DFS."""
        adjacency: Dict[str, list[str]] = {n.step_id: [] for n in task_graph.nodes}
        for src, dst in task_graph.edges:
            if src in adjacency:
                adjacency[src].append(dst)

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node_id in adjacency:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False


def run_workcell_simulation(
    prompt_or_spec: Union[str, Dict[str, Any]],
    num_episodes: int = 1,
    seed: int = 42,
    rollout_base_dir: Optional[Path] = None,
) -> list[WorkcellEpisodeResult]:
    """
    Convenience function to run workcell simulation from prompt or spec.

    Args:
        prompt_or_spec: Natural language prompt or structured spec
        num_episodes: Number of episodes to run
        seed: Random seed
        rollout_base_dir: Directory for rollout capture

    Returns:
        List of episode results
    """
    adapter = WorkcellOrchestrationAdapter(
        default_seed=seed,
        rollout_base_dir=rollout_base_dir,
    )

    compilation = adapter.request_task(prompt_or_spec, seed)

    results = []
    for i in range(num_episodes):
        result = adapter.run_episode(
            config=compilation.config,
            scene_spec=compilation.scene_spec,
            task_graph=compilation.task_graph,
            seed=seed + i,
        )
        results.append(result)

    return results
