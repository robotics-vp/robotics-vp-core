"""
Integration tests for LSD Vector Scene orchestrator, MPL logging, and training harnesses.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


class TestLSDVectorSceneConfig:
    """Tests for LSDVectorSceneConfig."""

    def test_default_config(self):
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        config = LSDVectorSceneConfig()
        assert config.topology_type == "WAREHOUSE_AISLES"
        assert config.num_nodes == 10
        assert config.num_humans == 2
        assert config.tilt == -1.0

    def test_from_dict(self):
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        data = {
            "topology_type": "KITCHEN_LAYOUT",
            "num_nodes": 5,
            "density": 0.8,
            "tilt": 0.5,
        }
        config = LSDVectorSceneConfig.from_dict(data)
        assert config.topology_type == "KITCHEN_LAYOUT"
        assert config.num_nodes == 5
        assert config.density == 0.8
        assert config.tilt == 0.5

    def test_to_dict(self):
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        config = LSDVectorSceneConfig(num_nodes=15, density=0.6)
        data = config.to_dict()
        assert data["num_nodes"] == 15
        assert data["density"] == 0.6
        assert "topology_type" in data

    def test_presets(self):
        from src.config.lsd_vector_scene_config import PRESETS, load_lsd_vector_scene_config

        assert "warehouse_easy" in PRESETS
        assert "warehouse_medium" in PRESETS
        assert "warehouse_hard" in PRESETS

        config = load_lsd_vector_scene_config(preset="warehouse_easy")
        assert config.density == 0.3
        assert config.num_humans == 1

    def test_json_serialization(self):
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        config = LSDVectorSceneConfig(num_nodes=8)
        json_str = config.to_json()
        loaded = LSDVectorSceneConfig.from_json(json_str)
        assert loaded.num_nodes == 8


class TestLSDVectorSceneBackend:
    """Tests for LSDVectorSceneBackend motor backend."""

    @pytest.fixture
    def mock_econ_meter(self):
        from src.economics.econ_meter import EconomicMeter
        from src.ontology.models import Task

        task = Task(
            task_id="test_task",
            name="Test Task",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.0001,
        )
        return EconomicMeter(task=task)

    def test_backend_creation(self, mock_econ_meter):
        from src.motor_backend.lsd_vector_scene_backend import LSDVectorSceneBackend
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        config = LSDVectorSceneConfig(num_nodes=5, max_steps=10)
        backend = LSDVectorSceneBackend(
            econ_meter=mock_econ_meter,
            default_config=config,
        )
        assert backend is not None

    @pytest.mark.slow
    def test_train_policy(self, mock_econ_meter):
        from src.motor_backend.lsd_vector_scene_backend import LSDVectorSceneBackend
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig
        from src.objectives.economic_objective import EconomicObjectiveSpec

        config = LSDVectorSceneConfig(num_nodes=3, max_steps=5)
        backend = LSDVectorSceneBackend(
            econ_meter=mock_econ_meter,
            default_config=config,
        )

        objective = EconomicObjectiveSpec(mpl_weight=1.0, error_weight=0.5)
        result = backend.train_policy(
            task_id="test_task",
            objective=objective,
            datapack_ids=[],
            num_envs=1,
            max_steps=10,
            seed=42,
            lsd_config=config,
        )

        assert result.policy_id.startswith("lsd_vector_scene_")
        assert "mpl_units_per_hour" in result.econ_metrics
        assert result.econ_metrics["mpl_units_per_hour"] >= 0

    @pytest.mark.slow
    def test_evaluate_policy(self, mock_econ_meter):
        from src.motor_backend.lsd_vector_scene_backend import LSDVectorSceneBackend
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig
        from src.objectives.economic_objective import EconomicObjectiveSpec

        config = LSDVectorSceneConfig(num_nodes=3, max_steps=5)
        backend = LSDVectorSceneBackend(
            econ_meter=mock_econ_meter,
            default_config=config,
        )

        objective = EconomicObjectiveSpec(mpl_weight=1.0)
        result = backend.evaluate_policy(
            policy_id="test_policy",
            task_id="test_task",
            objective=objective,
            num_episodes=2,
            seed=42,
            lsd_config=config,
        )

        assert result.policy_id == "test_policy"
        assert "mpl_units_per_hour" in result.econ_metrics
        assert result.raw_metrics.get("num_episodes") == 2.0


class TestMotorBackendFactory:
    """Tests for motor backend factory integration."""

    def test_factory_creates_lsd_backend(self):
        from src.economics.econ_meter import EconomicMeter
        from src.motor_backend.factory import make_motor_backend
        from src.ontology.models import Task
        from src.ontology.store import OntologyStore

        task = Task(
            task_id="test",
            name="Test",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.0001,
        )
        econ_meter = EconomicMeter(task=task)
        store = OntologyStore()

        backend = make_motor_backend(
            name="lsd_vector_scene",
            econ_meter=econ_meter,
            store=store,
        )

        assert backend is not None
        assert hasattr(backend, "train_policy")
        assert hasattr(backend, "evaluate_policy")

    def test_factory_with_config(self):
        from src.economics.econ_meter import EconomicMeter
        from src.motor_backend.factory import make_motor_backend
        from src.ontology.models import Task
        from src.ontology.store import OntologyStore

        task = Task(
            task_id="test",
            name="Test",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.0001,
        )
        econ_meter = EconomicMeter(task=task)
        store = OntologyStore()

        backend = make_motor_backend(
            name="lsd_vector_scene",
            econ_meter=econ_meter,
            store=store,
            backend_config={"num_nodes": 5, "topology_type": "KITCHEN_LAYOUT"},
        )

        assert backend is not None


class TestDifficultyFeatures:
    """Tests for difficulty feature computation."""

    @pytest.mark.slow
    def test_get_difficulty_features(self):
        from src.envs.lsd_vector_scene_env import (
            LSDVectorSceneEnv,
            LSDVectorSceneEnvConfig,
            SceneGraphConfig,
            VisualStyleConfig,
            BehaviourConfig,
        )

        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                density=0.7,
                route_length=40.0,
                num_nodes=5,
            ),
            behaviour_config=BehaviourConfig(
                num_humans=3,
                num_forklifts=1,
                tilt=0.5,
                use_simple_policy=True,
            ),
            max_steps=5,
        )
        env = LSDVectorSceneEnv(config)
        env.reset()
        _, info, _ = env.step(np.array([0.5, 0.5]))

        features = info["difficulty_features"]
        assert "graph_density" in features
        assert "route_length" in features
        assert "num_dynamic_agents" in features
        assert "tilt" in features

        assert features["graph_density"] == 0.7
        assert features["tilt"] == 0.5


class TestEconReportsDifficulty:
    """Tests for difficulty-based econ reports."""

    def test_compute_difficulty_mpl_analysis(self):
        from src.analytics.econ_reports import compute_difficulty_mpl_analysis

        # Create synthetic episode data
        episodes = [
            {
                "mpl_units_per_hour": 50.0,
                "difficulty_features": {"graph_density": 0.3, "tilt": -1.0},
            },
            {
                "mpl_units_per_hour": 40.0,
                "difficulty_features": {"graph_density": 0.5, "tilt": 0.0},
            },
            {
                "mpl_units_per_hour": 30.0,
                "difficulty_features": {"graph_density": 0.7, "tilt": 0.5},
            },
            {
                "mpl_units_per_hour": 20.0,
                "difficulty_features": {"graph_density": 0.9, "tilt": 1.0},
            },
        ]

        analysis = compute_difficulty_mpl_analysis(episodes)

        assert "features" in analysis
        assert "graph_density" in analysis["features"]
        assert "tilt" in analysis["features"]

        # Higher density should correlate negatively with MPL
        density_result = analysis["features"]["graph_density"]
        assert "correlation" in density_result
        assert density_result["correlation"] < 0  # Negative correlation

    def test_analysis_with_insufficient_data(self):
        from src.analytics.econ_reports import compute_difficulty_mpl_analysis

        # Only 2 episodes - insufficient for analysis
        episodes = [
            {"mpl_units_per_hour": 50.0, "difficulty_features": {"tilt": 0.0}},
            {"mpl_units_per_hour": 40.0, "difficulty_features": {"tilt": 0.5}},
        ]

        analysis = compute_difficulty_mpl_analysis(episodes)

        # Should handle gracefully
        assert "features" in analysis


class TestDatasetExport:
    """Tests for dataset export pipeline."""

    def test_export_tiny_dataset(self):
        """Test export with minimal configuration."""
        from scripts.export_lsd_vector_scene_dataset import (
            ExportConfig,
            export_dataset,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                num_scenes=1,
                episodes_per_scene=1,
                max_steps=5,
                output_path=tmpdir,
                shard_size=10,
                seed=42,
            )

            result = export_dataset(config, verbose=False)

            # Check output files exist
            assert Path(tmpdir, "index.json").exists()
            assert Path(tmpdir, "scenes.json").exists()

            # Check index content
            with open(Path(tmpdir, "index.json")) as f:
                index = json.load(f)

            assert index["total_episodes"] >= 1
            assert index["total_scenes"] >= 1
            assert len(index["shards"]) >= 1


class TestVisualization:
    """Tests for visualization utilities."""

    def test_viz_headless(self):
        """Test visualization in headless mode."""
        from scripts.debug_lsd_vector_scene_viz import create_debug_visualization

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_debug_visualization(
                config_path=None,
                output_dir=tmpdir,
                show=False,
                seed=42,
            )

            assert "scene_id" in result
            assert "files" in result

            # Should create at least some files (even if matplotlib not available)
            output_path = Path(tmpdir)
            assert output_path.exists()


class TestBehaviourTraining:
    """Tests for behaviour model training harness."""

    def test_trajectory_dataset_creation(self):
        """Test TrajectoryDataset with synthetic data."""
        from scripts.train_behaviour_model import TrajectoryDataset
        from scripts.export_lsd_vector_scene_dataset import (
            ExportConfig,
            export_dataset,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # First export a tiny dataset
            export_config = ExportConfig(
                num_scenes=1,
                episodes_per_scene=2,
                max_steps=10,
                output_path=tmpdir,
                shard_size=10,
                seed=42,
            )
            export_dataset(export_config, verbose=False)

            # Then load it
            dataset = TrajectoryDataset(
                dataset_path=tmpdir,
                max_seq_len=50,
                num_action_bins=32,
            )

            assert len(dataset) >= 1

            # Get an item
            item = dataset[0]
            assert "action_tokens" in item
            assert "rewards" in item
            assert "condition" in item
            assert "total_return" in item


class TestGGDSTraining:
    """Tests for GGDS training harness."""

    def test_dummy_ldm(self):
        """Test dummy LDM creation."""
        from scripts.train_ggds_on_lsd_vector_scenes import create_dummy_ldm

        ldm = create_dummy_ldm()
        assert hasattr(ldm, "encode")
        assert hasattr(ldm, "decode")
        assert hasattr(ldm, "compute_sds_gradient")

        # Test SDS gradient
        images = np.random.rand(4, 3, 256, 256).astype(np.float32)
        grad = ldm.compute_sds_gradient(images, "test prompt")
        assert grad.shape == images.shape

    @pytest.mark.slow
    def test_ggds_smoke(self):
        """Smoke test for GGDS training."""
        from scripts.train_ggds_on_lsd_vector_scenes import (
            GGDSTrainingConfig,
            train_ggds_on_scenes,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GGDSTrainingConfig(
                num_scenes=1,
                num_iterations=2,
                output_dir=tmpdir,
                seed=42,
            )

            result = train_ggds_on_scenes(config, verbose=False)

            assert "total_scenes" in result
            assert Path(tmpdir, "summary.json").exists()
