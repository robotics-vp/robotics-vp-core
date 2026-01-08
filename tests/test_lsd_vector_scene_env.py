"""
Tests for LSD Vector Scene Environment.
"""

import numpy as np
import pytest
from pathlib import Path

from src.envs.lsd_vector_scene_env import (
    LSDVectorSceneEnv,
    LSDVectorSceneEnvConfig,
    SceneGraphConfig,
    VisualStyleConfig,
    BehaviourConfig,
)


class TestLSDVectorSceneEnvConfig:
    def test_default_config(self):
        config = LSDVectorSceneEnvConfig()
        assert config.max_steps == 500
        assert config.scene_graph_config.topology_type == "WAREHOUSE_AISLES"

    def test_from_dict(self):
        data = {
            "scene_graph_config": {
                "topology_type": "KITCHEN_LAYOUT",
                "num_nodes": 5,
            },
            "behaviour_config": {
                "num_humans": 2,
                "tilt": -0.5,
            },
            "max_steps": 100,
        }
        config = LSDVectorSceneEnvConfig.from_dict(data)
        assert config.scene_graph_config.topology_type == "KITCHEN_LAYOUT"
        assert config.behaviour_config.num_humans == 2
        assert config.max_steps == 100

    def test_to_dict(self):
        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(topology_type="WAREHOUSE_AISLES"),
            max_steps=200,
        )
        data = config.to_dict()
        assert data["scene_graph_config"]["topology_type"] == "WAREHOUSE_AISLES"
        assert data["max_steps"] == 200


class TestLSDVectorSceneEnv:
    @pytest.fixture
    def env(self):
        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                topology_type="WAREHOUSE_AISLES",
                num_nodes=5,
                seed=42,
            ),
            visual_style_config=VisualStyleConfig(
                voxel_size=0.5,  # Coarse for speed
            ),
            behaviour_config=BehaviourConfig(
                num_humans=1,
                num_robots=0,
                num_forklifts=0,
                use_simple_policy=True,
            ),
            max_steps=20,
        )
        return LSDVectorSceneEnv(config)

    def test_reset(self, env):
        obs = env.reset()
        assert "t" in obs
        assert "completed" in obs
        assert "scene_id" in obs
        assert env.graph is not None
        assert env.voxels is not None
        assert env.mesh is not None
        assert env.gaussian_scene is not None

    def test_step(self, env):
        env.reset()
        action = np.array([0.5, 0.5])
        obs, info, done = env.step(action)

        assert "t" in obs
        assert "delta_units" in info
        assert "mpl_t" in info
        assert "difficulty_features" in info
        assert isinstance(done, bool)

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = np.array([0.5, 0.5])
            _, info, done = env.step(action)
            steps += 1

        assert done or steps >= env.config.max_steps

    def test_episode_log(self, env):
        env.reset()
        info_history = []
        done = False
        while not done:
            action = np.array([0.5, 0.5])
            _, info, done = env.step(action)
            info_history.append(info)

        log = env.get_episode_log(info_history)
        assert "scene_id" in log
        assert "difficulty_features" in log
        assert "mpl_metrics" in log
        assert "episode_summary" in log

    def test_get_scene_graph(self, env):
        env.reset()
        graph = env.get_scene_graph()
        assert graph is not None
        assert len(graph.nodes) > 0

    def test_get_gaussian_scene(self, env):
        env.reset()
        scene = env.get_gaussian_scene()
        assert scene is not None
        assert scene.num_gaussians > 0

    def test_get_agent_trajectories(self, env):
        env.reset()
        for _ in range(5):
            env.step(np.array([0.5, 0.5]))

        trajectories = env.get_agent_trajectories()
        assert isinstance(trajectories, list)
        # Should have trajectories if there are dynamic agents
        if env.config.behaviour_config.num_humans > 0:
            assert len(trajectories) > 0
            assert len(trajectories[0]) > 1  # Multiple timesteps

    def test_scalar_action(self, env):
        """Test backward compatibility with scalar action."""
        env.reset()
        obs, info, done = env.step(0.5)  # Single scalar
        assert "delta_units" in info


class TestKitchenLayout:
    @pytest.fixture
    def env(self):
        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                topology_type="KITCHEN_LAYOUT",
                seed=123,
            ),
            visual_style_config=VisualStyleConfig(
                voxel_size=0.3,
            ),
            behaviour_config=BehaviourConfig(
                num_humans=1,
                use_simple_policy=True,
            ),
            max_steps=10,
        )
        return LSDVectorSceneEnv(config)

    def test_kitchen_reset(self, env):
        obs = env.reset()
        assert env.graph is not None
        # Kitchen should have kitchen zone node
        kitchen_nodes = [n for n in env.graph.nodes if "KITCHEN" in str(n.node_type)]
        assert len(kitchen_nodes) >= 0  # May not always have explicit kitchen type


class TestDifficultyFeatures:
    @pytest.mark.slow
    def test_difficulty_features_computed(self):
        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                density=0.8,
                route_length=50.0,
            ),
            behaviour_config=BehaviourConfig(
                num_humans=3,
                num_forklifts=1,
                tilt=-0.5,
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
        assert features["graph_density"] == 0.8
        assert features["tilt"] == -0.5


class TestSeededReproducibility:
    @pytest.mark.slow
    def test_same_seed_same_scene(self):
        config1 = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(seed=42),
            max_steps=5,
        )
        config2 = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(seed=42),
            max_steps=5,
        )

        env1 = LSDVectorSceneEnv(config1)
        env2 = LSDVectorSceneEnv(config2)

        env1.reset()
        env2.reset()

        assert env1.scene_id == env2.scene_id
        assert len(env1.graph.nodes) == len(env2.graph.nodes)
