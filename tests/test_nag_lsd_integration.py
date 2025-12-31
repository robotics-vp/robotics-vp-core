"""
Tests for NAG ↔ LSD Vector Scene integration.

Tests the pipeline from LSD backend rollouts to NAGScene construction
and counterfactual generation.
"""

import numpy as np
import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class TestNAGFromLSDConfig:
    """Tests for NAGFromLSDConfig dataclass."""

    def test_default_config(self):
        from src.vision.nag.integration_lsd_backend import NAGFromLSDConfig

        config = NAGFromLSDConfig()
        assert config.atlas_size == (256, 256)
        assert config.max_nodes == 8
        assert config.max_iters == 200
        assert config.fov_deg == 60.0
        assert config.image_size == (256, 256)

    def test_custom_config(self):
        from src.vision.nag.integration_lsd_backend import NAGFromLSDConfig

        config = NAGFromLSDConfig(
            atlas_size=(512, 512),
            max_nodes=16,
            max_iters=500,
            fov_deg=90.0,
        )
        assert config.atlas_size == (512, 512)
        assert config.max_nodes == 16
        assert config.max_iters == 500
        assert config.fov_deg == 90.0


class TestNAGEditPolicyConfig:
    """Tests for NAGEditPolicyConfig dataclass."""

    def test_default_policy_config(self):
        from src.vision.nag.integration_lsd_backend import NAGEditPolicyConfig

        config = NAGEditPolicyConfig()
        assert config.num_counterfactuals == 3
        assert config.prob_remove == 0.15
        assert config.prob_duplicate == 0.2
        assert config.prob_pose_shift == 0.35
        assert config.prob_color_shift == 0.2

    def test_probabilities_sum(self):
        from src.vision.nag.integration_lsd_backend import NAGEditPolicyConfig

        config = NAGEditPolicyConfig()
        total_prob = (
            config.prob_remove +
            config.prob_duplicate +
            config.prob_pose_shift +
            config.prob_color_shift
        )
        # Should sum to less than or equal to 1.0
        assert total_prob <= 1.0


class TestNAGDatapack:
    """Tests for NAGDatapack dataclass."""

    def test_create_datapack(self):
        from src.vision.nag.integration_lsd_backend import NAGDatapack

        frames = np.random.rand(10, 3, 64, 64).astype(np.float32)
        datapack = NAGDatapack(
            base_episode_id="ep_001",
            counterfactual_id="ep_001_cf0",
            frames=frames,
            nag_edit_vector=[{"edit_type": "pose", "node_id": "obj_0"}],
        )

        assert datapack.frames.shape == (10, 3, 64, 64)
        assert len(datapack.nag_edit_vector) == 1
        assert datapack.base_episode_id == "ep_001"
        assert datapack.counterfactual_id == "ep_001_cf0"
        assert datapack.difficulty_features == {}

    def test_datapack_with_metadata(self):
        from src.vision.nag.integration_lsd_backend import NAGDatapack

        frames = np.random.rand(5, 3, 32, 32).astype(np.float32)
        lsd_metadata = {"task": "dishwashing", "baseline_mpl": 54.0}

        datapack = NAGDatapack(
            base_episode_id="ep_002",
            counterfactual_id="ep_002_cf1",
            frames=frames,
            nag_edit_vector=[],
            lsd_metadata=lsd_metadata,
        )

        assert datapack.lsd_metadata["task"] == "dishwashing"
        assert datapack.lsd_metadata["baseline_mpl"] == 54.0

    def test_datapack_to_dict(self):
        from src.vision.nag.integration_lsd_backend import NAGDatapack

        frames = np.random.rand(5, 3, 32, 32).astype(np.float32)
        datapack = NAGDatapack(
            base_episode_id="ep_003",
            counterfactual_id="ep_003_cf0",
            frames=frames,
            nag_edit_vector=[{"edit_type": "pose"}, {"edit_type": "duplicate"}],
        )

        d = datapack.to_dict()
        assert d["base_episode_id"] == "ep_003"
        assert d["counterfactual_id"] == "ep_003_cf0"
        assert d["frames_shape"] == [5, 3, 32, 32]
        assert d["num_edits"] == 2
        assert d["edit_types"] == ["pose", "duplicate"]


class TestCreateCameraFromLSDConfig:
    """Tests for create_camera_from_lsd_config helper."""

    def test_create_camera(self):
        from src.vision.nag.integration_lsd_backend import (
            NAGFromLSDConfig,
            create_camera_from_lsd_config,
        )

        config = NAGFromLSDConfig(
            fov_deg=60.0,
            image_size=(128, 128),
        )
        camera = create_camera_from_lsd_config(config)

        # Use approximate comparison for FOV due to floating point
        assert abs(camera.fov_deg - 60.0) < 0.01
        assert camera.height == 128
        assert camera.width == 128
        assert camera.world_from_cam.shape == (1, 4, 4)

    def test_camera_rays(self):
        from src.vision.nag.integration_lsd_backend import (
            NAGFromLSDConfig,
            create_camera_from_lsd_config,
        )

        config = NAGFromLSDConfig(image_size=(64, 64))
        camera = create_camera_from_lsd_config(config)

        origins, dirs = camera.get_rays(t=0)
        assert origins.shape == (64, 64, 3)
        assert dirs.shape == (64, 64, 3)

        # Directions should be normalized
        norms = np.linalg.norm(dirs, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestRenderLSDEpisodeFrames:
    """Tests for render_lsd_episode_frames stub."""

    def test_stub_returns_colored_frames(self):
        from src.vision.nag.integration_lsd_backend import (
            render_lsd_episode_frames,
            create_camera_from_lsd_config,
            NAGFromLSDConfig,
        )

        # Create CameraParams via the helper function
        config = NAGFromLSDConfig(image_size=(64, 64))
        camera = create_camera_from_lsd_config(config)

        # No scene - should return random frames
        frames = render_lsd_episode_frames(None, camera, num_frames=10)

        # Stub should return frames with correct shape
        assert frames.shape == (10, 3, 64, 64)
        assert frames.dtype == torch.float32

        # Check values are in valid range
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0


class TestBuildNAGSceneFromLSD:
    """Tests for build_nag_scene_from_lsd_rollout."""

    def test_build_with_mock_data(self):
        from src.vision.nag.integration_lsd_backend import (
            NAGFromLSDConfig,
            build_nag_scene_from_lsd_rollout,
            create_camera_from_lsd_config,
        )

        config = NAGFromLSDConfig(
            atlas_size=(64, 64),
            max_nodes=4,
            max_iters=10,  # Small for testing
            image_size=(32, 32),
        )

        camera = create_camera_from_lsd_config(config)

        # Mock backend episode with no scene_graph
        backend_episode = {
            "episode_id": "test_ep_001",
            "gaussian_scene": None,
            "scene_graph": None,
            "num_frames": 5,
        }

        # Build NAG scene from mock rollout
        nag_scene = build_nag_scene_from_lsd_rollout(
            backend_episode=backend_episode,
            camera=camera,
            config=config,
        )

        # Should return a valid NAGScene
        assert nag_scene is not None
        # With no scene_graph, only background should exist
        assert nag_scene.num_nodes() == 1
        assert nag_scene.background_node_id is not None


class TestGenerateNAGCounterfactuals:
    """Tests for generate_nag_counterfactuals_for_lsd_episode."""

    def test_generate_counterfactuals(self):
        from src.vision.nag.integration_lsd_backend import (
            NAGFromLSDConfig,
            NAGEditPolicyConfig,
            generate_nag_counterfactuals_for_lsd_episode,
            create_camera_from_lsd_config,
        )

        nag_config = NAGFromLSDConfig(
            atlas_size=(32, 32),
            max_iters=5,
            image_size=(32, 32),
        )

        policy_config = NAGEditPolicyConfig(
            num_counterfactuals=2,
        )

        camera = create_camera_from_lsd_config(nag_config)

        backend_episode = {
            "episode_id": "test_ep_001",
            "gaussian_scene": None,
            "scene_graph": None,
            "num_frames": 5,
        }

        datapacks = generate_nag_counterfactuals_for_lsd_episode(
            backend_episode=backend_episode,
            camera=camera,
            nag_config=nag_config,
            edit_config=policy_config,
        )

        # Should return requested number of datapacks
        assert len(datapacks) == 2

        for i, dp in enumerate(datapacks):
            # Check frames shape
            assert dp.frames.shape == (5, 3, 32, 32)
            assert dp.frames.dtype == np.float32

            # Check base episode ID
            assert dp.base_episode_id == "test_ep_001"

            # Check counterfactual ID
            assert dp.counterfactual_id == f"test_ep_001_cf{i}"

            # Edit vectors may be empty if scene has no foreground nodes
            assert isinstance(dp.nag_edit_vector, list)

    def test_counterfactual_frames_valid_range(self):
        from src.vision.nag.integration_lsd_backend import (
            NAGFromLSDConfig,
            NAGEditPolicyConfig,
            generate_nag_counterfactuals_for_lsd_episode,
            create_camera_from_lsd_config,
        )

        nag_config = NAGFromLSDConfig(
            atlas_size=(32, 32),
            max_iters=5,
            image_size=(32, 32),
        )

        policy_config = NAGEditPolicyConfig(num_counterfactuals=1)

        camera = create_camera_from_lsd_config(nag_config)

        backend_episode = {
            "episode_id": "ep_valid_range",
            "gaussian_scene": None,
            "scene_graph": None,
            "num_frames": 3,
        }

        datapacks = generate_nag_counterfactuals_for_lsd_episode(
            backend_episode=backend_episode,
            camera=camera,
            nag_config=nag_config,
            edit_config=policy_config,
        )

        # RGB values should be in [0, 1]
        for dp in datapacks:
            assert dp.frames.min() >= 0.0
            assert dp.frames.max() <= 1.0


class TestIntegrationWithLSDConfig:
    """Tests for NAG integration with LSDVectorSceneConfig."""

    def test_config_has_nag_settings(self):
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        config = LSDVectorSceneConfig()

        # Check NAG settings exist with defaults
        assert hasattr(config, "enable_nag_overlay")
        assert hasattr(config, "nag_atlas_size")
        assert hasattr(config, "nag_max_nodes")
        assert hasattr(config, "nag_fit_iters")
        assert hasattr(config, "nag_num_counterfactuals")

        # Check default values
        assert config.enable_nag_overlay is False
        assert config.nag_atlas_size == (256, 256)
        assert config.nag_max_nodes == 8
        assert config.nag_fit_iters == 200
        assert config.nag_num_counterfactuals == 3

    def test_config_serialization(self):
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        config = LSDVectorSceneConfig(
            enable_nag_overlay=True,
            nag_atlas_size=(512, 512),
            nag_max_nodes=16,
        )

        # Convert to dict and back
        config_dict = config.to_dict()
        assert config_dict["enable_nag_overlay"] is True
        # JSON serialization may convert tuples to lists
        assert tuple(config_dict["nag_atlas_size"]) == (512, 512)
        assert config_dict["nag_max_nodes"] == 16

        # Reconstruct from dict
        config2 = LSDVectorSceneConfig.from_dict(config_dict)
        assert config2.enable_nag_overlay is True
        assert config2.nag_atlas_size == (512, 512)
        assert config2.nag_max_nodes == 16


class TestNAGEditVectorStructure:
    """Tests for NAGEditVector metadata in datapacks."""

    def test_edit_vector_structure(self):
        from src.vision.nag.types import NAGEditVector, make_node_id

        edit = NAGEditVector(
            node_id=make_node_id("obj_0"),
            edit_type="pose",
            parameters={
                "delta_translation": [0.1, 0.2, 0.3],
                "delta_rotation_euler": [0.0, 0.0, 0.1],
            },
            timestamp=0.5,
        )

        assert str(edit.node_id) == "obj_0"
        assert edit.edit_type == "pose"
        assert edit.parameters["delta_translation"] == [0.1, 0.2, 0.3]
        assert edit.timestamp == 0.5

    def test_edit_vector_to_dict(self):
        from src.vision.nag.types import NAGEditVector, make_node_id

        edit = NAGEditVector(
            node_id=make_node_id("obj_1"),
            edit_type="texture",
            parameters={"blend_weight": 0.8, "num_pixels": 1024},
        )

        edit_dict = edit.to_dict()
        assert edit_dict["node_id"] == "obj_1"
        assert edit_dict["edit_type"] == "texture"
        assert edit_dict["parameters"]["blend_weight"] == 0.8


class TestEconReportsNAGIntegration:
    """Tests for NAG analysis functions in econ_reports."""

    def test_nag_edit_surface_summary(self):
        from src.analytics.econ_reports import compute_nag_edit_surface_summary
        from src.vision.nag.types import NAGEditVector, make_node_id

        # The function expects List[Dict], not NAGDatapack objects
        datapacks = [
            {
                "base_episode_id": "ep_001",
                "counterfactual_id": "ep_001_cf0",
                "nag_edit_vector": [
                    NAGEditVector(
                        node_id=make_node_id("obj_0"),
                        edit_type="pose",
                        parameters={},
                    ).to_dict(),
                    NAGEditVector(
                        node_id=make_node_id("obj_1"),
                        edit_type="duplicate",
                        parameters={},
                    ).to_dict(),
                ],
                "difficulty_features": {"wage_parity": 0.85},
            },
            {
                "base_episode_id": "ep_001",
                "counterfactual_id": "ep_001_cf1",
                "nag_edit_vector": [
                    NAGEditVector(
                        node_id=make_node_id("obj_0"),
                        edit_type="pose",
                        parameters={},
                    ).to_dict(),
                ],
                "difficulty_features": {"wage_parity": 0.92},
            },
        ]

        summary = compute_nag_edit_surface_summary(datapacks)

        # Check the actual return structure
        assert "edit_type_distribution" in summary
        assert summary["edit_type_distribution"]["pose"] == 2
        assert summary["edit_type_distribution"]["duplicate"] == 1
        assert summary["counterfactual_impact"]["total_counterfactuals"] == 2
        assert summary["counterfactual_impact"]["total_edits"] == 3

    def test_nag_counterfactual_mpl_analysis(self):
        from src.analytics.econ_reports import compute_nag_counterfactual_mpl_analysis

        # The actual function signature expects episode dicts, not single metric dicts
        base_episodes = [
            {"episode_id": "ep_001", "mpl_units_per_hour": 54.0},
        ]

        counterfactual_episodes = [
            {
                "base_episode_id": "ep_001",
                "nag_edit_vector": [{"edit_type": "pose"}],
                "difficulty_features": {"nag_objects_removed": 1, "nag_objects_added": 0, "nag_pose_perturbations": 1},
            },
            {
                "base_episode_id": "ep_001",
                "nag_edit_vector": [{"edit_type": "duplicate"}],
                "difficulty_features": {"nag_objects_removed": 0, "nag_objects_added": 1, "nag_pose_perturbations": 0},
            },
        ]

        analysis = compute_nag_counterfactual_mpl_analysis(
            base_episodes, counterfactual_episodes
        )

        # Check the actual return structure
        assert "num_counterfactuals_analyzed" in analysis
        assert analysis["num_counterfactuals_analyzed"] == 2
        assert "avg_mpl_change" in analysis
        assert "edit_type_impact" in analysis


class TestEndToEndMicroPipeline:
    """End-to-end test of a minimal NAG pipeline."""

    def test_micro_pipeline(self):
        """Run a minimal NAG pipeline from mock LSD data to counterfactuals."""
        from src.vision.nag.integration_lsd_backend import (
            NAGFromLSDConfig,
            NAGEditPolicyConfig,
            build_nag_scene_from_lsd_rollout,
            generate_nag_counterfactuals_for_lsd_episode,
            create_camera_from_lsd_config,
        )

        # 1. Configure NAG with minimal settings for fast testing
        nag_config = NAGFromLSDConfig(
            atlas_size=(16, 16),
            max_nodes=2,
            max_iters=2,
            image_size=(16, 16),
        )

        policy_config = NAGEditPolicyConfig(
            num_counterfactuals=1,
        )

        camera = create_camera_from_lsd_config(nag_config)

        # 2. Create mock backend episode
        backend_episode = {
            "episode_id": "micro_test",
            "gaussian_scene": None,
            "scene_graph": None,
            "num_frames": 3,
        }

        # 3. Build NAG scene
        nag_scene = build_nag_scene_from_lsd_rollout(
            backend_episode=backend_episode,
            camera=camera,
            config=nag_config,
        )
        assert nag_scene is not None
        assert nag_scene.num_nodes() >= 1  # At least background

        # 4. Generate counterfactuals
        datapacks = generate_nag_counterfactuals_for_lsd_episode(
            backend_episode=backend_episode,
            camera=camera,
            nag_config=nag_config,
            edit_config=policy_config,
        )

        assert len(datapacks) == 1
        dp = datapacks[0]

        # 5. Verify datapack structure
        assert dp.frames.shape == (3, 3, 16, 16)
        assert dp.base_episode_id == "micro_test"
        assert dp.counterfactual_id == "micro_test_cf0"
        assert isinstance(dp.nag_edit_vector, list)

        # 6. RGB values valid
        assert dp.frames.min() >= 0.0
        assert dp.frames.max() <= 1.0
