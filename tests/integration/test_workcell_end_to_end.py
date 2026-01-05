"""
End-to-end integration test for workcell environment suite.

Validates the core pipeline components work together:
  config -> env -> rollout -> analytics

This test prevents "wire drift" where individual components pass but the
full integration is broken.
"""
from __future__ import annotations

import math
import pytest
from typing import Any, Dict


class TestWorkcellEnvIntegration:
    """Integration tests for workcell env core pipeline."""

    def test_config_to_env_to_rollout(self) -> None:
        """Test: config -> env -> rollout produces valid episodes."""
        from src.envs.workcell_env import WorkcellEnv
        from src.envs.workcell_env.config import WorkcellEnvConfig

        config = WorkcellEnvConfig(
            num_parts=6,
            max_steps=50,
            tolerance_mm=2.0,
        )

        env = WorkcellEnv(config=config, seed=42)
        obs = env.reset(seed=42)

        assert obs is not None
        assert "object_ids" in obs

        # Run episode
        rewards = []
        for _ in range(10):
            action = {"target_position": [0.5, 0.5, 0.1], "gripper_action": 0}
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            assert math.isfinite(reward)
            if terminated or truncated:
                break

        # Get episode log
        log = env.get_episode_log()
        assert log is not None
        assert log.metadata.episode_id is not None
        assert len(log.trajectory) > 0

    def test_scene_generation_deterministic(self) -> None:
        """Test: same seed produces same scene."""
        from src.envs.workcell_env.scene.generators import WorkcellSceneGenerator
        from src.envs.workcell_env.config import WorkcellEnvConfig

        config = WorkcellEnvConfig(num_parts=8, num_bins=3)
        generator = WorkcellSceneGenerator()

        scene1 = generator.generate(config, seed=123)
        scene2 = generator.generate(config, seed=123)

        assert scene1.workcell_id == scene2.workcell_id
        assert len(scene1.parts) == len(scene2.parts)
        for p1, p2 in zip(scene1.parts, scene2.parts):
            assert p1.id == p2.id
            assert p1.position == p2.position

    def test_task_compiler_to_config(self) -> None:
        """Test: NL prompt -> task compiler -> config."""
        from src.envs.workcell_env.compiler import WorkcellTaskCompiler

        compiler = WorkcellTaskCompiler(default_seed=42)

        # Test kitting prompt
        result = compiler.compile_from_prompt("Pack 6 bolts into a tray with 1mm tolerance")
        assert result.inferred_task_type == "kitting"
        assert result.config is not None
        assert result.scene_spec is not None
        assert result.task_graph is not None
        assert len(result.task_graph.nodes) > 0

        # Test peg-in-hole prompt
        result2 = compiler.compile_from_prompt("Insert peg into hole with 0.5mm tolerance")
        assert result2.inferred_task_type == "peg_in_hole"

    def test_difficulty_features_integration(self) -> None:
        """Test: config -> difficulty features -> composite score."""
        from src.envs.workcell_env.config import WorkcellEnvConfig, PRESETS
        from src.envs.workcell_env.difficulty import compute_difficulty_features

        # Test all presets produce valid difficulty scores
        for preset_name, config in PRESETS.items():
            features = compute_difficulty_features(config)
            composite = features.composite_difficulty()

            assert 0.0 <= composite <= 1.0, f"Preset {preset_name} out of range"
            assert math.isfinite(composite)

    def test_analytics_from_episode_data(self) -> None:
        """Test: episode data -> analytics metrics."""
        from src.analytics.workcell_analytics import compute_episode_metrics

        episode_data = {
            "episode_id": "test_ep_001",
            "task_type": "kitting",
            "success": True,
            "total_reward": 15.5,
            "steps": 45,
            "time_s": 45.0,
            "items_completed": 5,
            "items_total": 6,
            "errors": 1,
            "tolerance_violations": 0,
        }

        metrics = compute_episode_metrics(
            episode_id=episode_data["episode_id"],
            task_type=episode_data["task_type"],
            episode_data=episode_data,
        )

        assert metrics.episode_id == "test_ep_001"
        assert metrics.task_type == "kitting"
        assert metrics.success is True
        assert math.isfinite(metrics.total_reward)
        assert math.isfinite(metrics.quality_score)
        assert 0.0 <= metrics.quality_score <= 1.0

    def test_scene_reconstruction_adapter(self) -> None:
        """Test: SceneTracks -> reconstruction -> scene spec."""
        import numpy as np
        from src.envs.workcell_env.reconstruction.scene_tracks_adapter import (
            SceneTracksAdapter,
        )

        # Create mock SceneTracks data
        mock_tracks = {
            "positions": np.array([
                [[0.5, 0.5, 0.1]],  # Track 0
                [[1.0, 0.5, 0.1]],  # Track 1
                [[0.5, 1.0, 0.1]],  # Track 2
            ]),
            "orientations": np.array([
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
            ]),
            "track_ids": np.array([0, 1, 2]),
        }

        adapter = SceneTracksAdapter()
        result = adapter.reconstruct_from_tracks(mock_tracks)

        assert result.scene_spec is not None
        assert len(result.scene_spec.parts) == 3
        assert result.confidence_score > 0.0
        assert "part_0" in result.track_mapping.values()

    def test_orchestrator_adapter_integration(self) -> None:
        """Test: orchestrator adapter request -> compilation result."""
        from src.orchestrator.workcell_adapter import WorkcellOrchestrationAdapter

        adapter = WorkcellOrchestrationAdapter(default_seed=42)

        # Request task via NL
        result = adapter.request_task("Sort 10 widgets by color into 3 bins")

        assert result.inferred_task_type == "sorting"
        assert result.config is not None
        assert result.scene_spec is not None
        assert result.task_graph is not None

        # Validate task graph structure (returns dict with 'valid' key)
        validation = adapter.validate_task_graph(result.task_graph)
        assert validation.get("valid", False) is True


class TestWorkcellFactoryIntegration:
    """Tests for motor backend factory integration."""

    def test_factory_creates_workcell_backend(self) -> None:
        """Test: factory creates workcell backend with proper deps."""
        from src.economics.econ_meter import EconomicMeter
        from src.motor_backend.factory import make_motor_backend
        from src.ontology.models import Task
        from src.ontology.store import OntologyStore

        task = Task(
            task_id="workcell_test",
            name="Workcell Test Task",
            description="Test workcell task",
        )

        econ_meter = EconomicMeter(task)
        store = OntologyStore()

        backend = make_motor_backend(
            name="workcell_env",
            econ_meter=econ_meter,
            store=store,
        )

        assert backend is not None
        # Verify it's the right type
        from src.motor_backend.workcell_env_backend import WorkcellEnvBackend
        assert isinstance(backend, WorkcellEnvBackend)

    def test_factory_with_config(self) -> None:
        """Test: factory accepts backend_config for workcell."""
        from src.economics.econ_meter import EconomicMeter
        from src.motor_backend.factory import make_motor_backend
        from src.ontology.models import Task
        from src.ontology.store import OntologyStore

        task = Task(
            task_id="workcell_config_test",
            name="Workcell Config Test",
            description="Test workcell with config",
        )

        econ_meter = EconomicMeter(task)
        store = OntologyStore()

        backend = make_motor_backend(
            name="workcell_env",
            econ_meter=econ_meter,
            store=store,
            backend_config={"num_parts": 10, "topology_type": "CONVEYOR_LINE"},
        )

        assert backend is not None


class TestWorkcellMetricsFinite:
    """Verify all metrics are finite (no NaN/Inf)."""

    def test_all_env_outputs_finite(self) -> None:
        """All env outputs should be finite."""
        from src.envs.workcell_env import WorkcellEnv
        from src.envs.workcell_env.config import WorkcellEnvConfig

        config = WorkcellEnvConfig(num_parts=4, max_steps=20)
        env = WorkcellEnv(config=config, seed=42)

        obs = env.reset(seed=42)

        for _ in range(15):
            action = {"target_position": [0.5, 0.5, 0.1], "gripper_action": 0}
            obs, reward, terminated, truncated, info = env.step(action)

            assert math.isfinite(reward), "Reward not finite"

            # Check info dict values
            for key, val in info.items():
                if isinstance(val, (int, float)):
                    assert math.isfinite(val), f"Info[{key}] not finite"

            if terminated or truncated:
                break


class TestWorkcellDeterminism:
    """Verify deterministic behavior with fixed seeds."""

    def test_same_seed_same_trajectory(self) -> None:
        """Same seed should produce same trajectory."""
        from src.envs.workcell_env import WorkcellEnv
        from src.envs.workcell_env.config import WorkcellEnvConfig

        config = WorkcellEnvConfig(num_parts=4, max_steps=20)

        # First run
        env1 = WorkcellEnv(config=config, seed=42)
        env1.reset(seed=42)
        rewards1 = []
        for i in range(5):
            action = {"target_position": [0.5 + i * 0.05, 0.5, 0.1], "gripper_action": i % 2}
            _, reward, _, _, _ = env1.step(action)
            rewards1.append(reward)

        # Second run with same seed
        env2 = WorkcellEnv(config=config, seed=42)
        env2.reset(seed=42)
        rewards2 = []
        for i in range(5):
            action = {"target_position": [0.5 + i * 0.05, 0.5, 0.1], "gripper_action": i % 2}
            _, reward, _, _, _ = env2.step(action)
            rewards2.append(reward)

        assert rewards1 == rewards2, "Trajectories differ with same seed"
