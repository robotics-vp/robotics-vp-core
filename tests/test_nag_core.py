"""
Core tests for Neural Atlas-Graph (NAG) components.

Tests renderer sanity, atlas behavior, and basic scene operations.
"""

import numpy as np
import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class TestCameraParams:
    """Tests for CameraParams type."""

    def test_create_from_fov(self):
        from src.vision.nag.types import CameraParams

        world_from_cam = np.eye(4, dtype=np.float32)
        world_from_cam[2, 3] = 5.0  # Camera at z=5

        camera = CameraParams.from_fov(
            fov_deg=60.0,
            height=256,
            width=256,
            world_from_cam=world_from_cam,
        )

        assert camera.height == 256
        assert camera.width == 256
        assert camera.num_frames == 1
        assert abs(camera.fov_deg - 60.0) < 5.0  # Approximate

    def test_create_from_camera_rig(self):
        from src.vision.nag.types import CameraParams

        positions = np.array([[0, 0, 5]], dtype=np.float32)
        look_at = np.array([[0, 0, 0]], dtype=np.float32)
        up = np.array([[0, 1, 0]], dtype=np.float32)

        camera = CameraParams.from_camera_rig(
            positions=positions,
            look_at=look_at,
            up=up,
            fov=60.0,
            width=128,
            height=128,
        )

        assert camera.num_frames == 1
        assert camera.width == 128

    def test_get_rays(self):
        from src.vision.nag.types import CameraParams

        world_from_cam = np.eye(4, dtype=np.float32)
        camera = CameraParams.from_fov(60.0, 64, 64, world_from_cam)

        origins, directions = camera.get_rays(t=0)

        assert origins.shape == (64, 64, 3)
        assert directions.shape == (64, 64, 3)

        # Directions should be normalized
        norms = np.linalg.norm(directions, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestPlaneParams:
    """Tests for PlaneParams type."""

    def test_create_frontal(self):
        from src.vision.nag.types import PlaneParams

        plane = PlaneParams.create_frontal(
            center=(0, 0, 0),
            extent=(2.0, 2.0),
            normal=(0, 0, 1),
        )

        assert plane.extent[0] == 2.0
        assert np.allclose(plane.origin, [0, 0, 0])
        assert np.allclose(plane.normal, [0, 0, 1], atol=1e-5)

    def test_world_point_to_uv(self):
        from src.vision.nag.types import PlaneParams

        plane = PlaneParams.create_frontal(
            center=(0, 0, 0),
            extent=(2.0, 2.0),
            normal=(0, 0, 1),
        )

        # Point at center should map to (0.5, 0.5)
        center_point = np.array([0, 0, 0], dtype=np.float32)
        uv = plane.world_point_to_uv(center_point)
        assert np.allclose(uv, [0.5, 0.5], atol=1e-5)

    def test_uv_to_world_roundtrip(self):
        from src.vision.nag.types import PlaneParams

        plane = PlaneParams.create_frontal(
            center=(1, 2, 3),
            extent=(4.0, 3.0),
            normal=(0, 0, 1),
        )

        # Random UV points
        uv = np.array([[0.2, 0.3], [0.8, 0.7]], dtype=np.float32)
        world_pts = plane.uv_to_world_point(uv)
        uv_back = plane.world_point_to_uv(world_pts)

        assert np.allclose(uv, uv_back, atol=1e-4)


class TestPoseSplineParams:
    """Tests for PoseSplineParams type."""

    def test_static_spline(self):
        from src.vision.nag.types import PoseSplineParams

        spline = PoseSplineParams.create_static(
            translation=(1, 2, 3),
            euler=(0, 0, 0),
            t_range=(0, 1),
        )

        # Should return same pose at any time
        pose_0 = spline.pose_at(0.0)
        pose_mid = spline.pose_at(0.5)
        pose_1 = spline.pose_at(1.0)

        assert np.allclose(pose_0[:3, 3], [1, 2, 3])
        assert np.allclose(pose_mid[:3, 3], [1, 2, 3])
        assert np.allclose(pose_1[:3, 3], [1, 2, 3])

    def test_linear_spline(self):
        from src.vision.nag.types import PoseSplineParams

        spline = PoseSplineParams.create_linear(
            start_trans=(0, 0, 0),
            end_trans=(10, 0, 0),
            t_range=(0, 1),
        )

        pose_mid = spline.pose_at(0.5)
        assert np.allclose(pose_mid[:3, 3], [5, 0, 0], atol=1e-4)

    def test_apply_offset(self):
        from src.vision.nag.types import PoseSplineParams

        spline = PoseSplineParams.create_static((0, 0, 0))
        offset_spline = spline.apply_offset(
            delta_translation=np.array([1, 2, 3]),
            delta_euler=np.array([0, 0, 0]),
        )

        pose = offset_spline.pose_at(0.5)
        assert np.allclose(pose[:3, 3], [1, 2, 3])


class TestNAGPlaneNode:
    """Tests for NAGPlaneNode."""

    def test_create_node(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(
            node_id=make_node_id("test"),
            plane_params=plane,
            pose_spline=spline,
            atlas_size=(64, 64),
        )

        assert node.atlas_size == (64, 64)
        assert node.base_texture.shape == (3, 64, 64)

    def test_pose_at(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_linear(
            start_trans=(0, 0, 0),
            end_trans=(5, 0, 0),
        )
        node = NAGPlaneNode(make_node_id("test"), plane, spline)

        pose = node.pose_at(torch.tensor(0.5))
        assert pose.shape == (4, 4)
        assert np.allclose(pose[:3, 3].detach().numpy(), [2.5, 0, 0], atol=0.5)

    def test_sample_atlas(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("test"), plane, spline, atlas_size=(32, 32))

        uv = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        t = torch.tensor(0.5)
        view_dir = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32)

        rgb, alpha = node.sample_atlas(uv, t, view_dir)

        assert rgb.shape == (2, 3)
        assert alpha.shape == (2,)
        assert (rgb >= 0).all() and (rgb <= 1).all()

    def test_initialize_from_image(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("test"), plane, spline, atlas_size=(32, 32))

        # Create test image (red square)
        test_image = torch.zeros(3, 64, 64)
        test_image[0] = 1.0  # Red channel

        node.initialize_from_image(test_image)

        # Base texture should be reddish
        assert node.base_texture[0].mean() > 0.5

    def test_clone(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("test"), plane, spline)

        cloned = node.clone()

        # Modify original
        with torch.no_grad():
            node.base_texture.data.fill_(0.1)

        # Clone should be independent
        assert cloned.base_texture.mean() != node.base_texture.mean()


class TestNAGScene:
    """Tests for NAGScene container."""

    def test_create_empty_scene(self):
        from src.vision.nag.scene import NAGScene

        scene = NAGScene()
        assert scene.num_nodes() == 0
        assert scene.list_nodes() == []

    def test_add_and_get_node(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene

        scene = NAGScene()

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("obj_0"), plane, spline)

        scene.add_node(make_node_id("obj_0"), node)

        assert scene.num_nodes() == 1
        retrieved = scene.get_node(make_node_id("obj_0"))
        assert retrieved is node

    def test_clone_node(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene

        scene = NAGScene()

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("obj_0"), plane, spline)
        scene.add_node(make_node_id("obj_0"), node)

        # Clone with offset
        scene.clone_node(
            make_node_id("obj_0"),
            make_node_id("obj_1"),
            pose_offset={"translation": np.array([1, 0, 0])},
        )

        assert scene.num_nodes() == 2
        assert scene.has_node(make_node_id("obj_1"))

    def test_scene_clone(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene

        scene = NAGScene()
        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("obj_0"), plane, spline)
        scene.add_node(make_node_id("obj_0"), node)

        cloned_scene = scene.clone()

        assert cloned_scene.num_nodes() == 1
        scene.remove_node(make_node_id("obj_0"))
        assert scene.num_nodes() == 0
        assert cloned_scene.num_nodes() == 1  # Independent


class TestRenderer:
    """Tests for NAG renderer."""

    def test_single_frontal_plane(self):
        """Single front-parallel plane renders as expected color."""
        from src.vision.nag.types import CameraParams, PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene
        from src.vision.nag.renderer import render_scene

        # Camera looking at origin from z=5
        world_from_cam = np.eye(4, dtype=np.float32)
        world_from_cam[2, 3] = 5.0
        camera = CameraParams.from_fov(60.0, 64, 64, world_from_cam)

        # Plane at origin facing camera
        plane = PlaneParams.create_frontal((0, 0, 0), (4, 4), normal=(0, 0, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("plane"), plane, spline, atlas_size=(32, 32))

        # Set constant red texture
        with torch.no_grad():
            node.base_texture.fill_(0)
            node.base_texture[0].fill_(1.0)  # Red
            node.base_alpha.fill_(1.0)

        scene = NAGScene()
        scene.add_node(make_node_id("plane"), node)

        t = torch.tensor(0.0)
        result = render_scene(scene, camera, t)

        assert "rgb" in result
        assert result["rgb"].shape == (3, 64, 64)

        # Center region should have red contribution
        center_rgb = result["rgb"][:, 28:36, 28:36].mean(dim=(1, 2))
        # Due to alpha compositing, should have some red
        assert center_rgb[0] > 0.3  # Red channel present

    def test_two_planes_depth_ordering(self):
        """Closer plane occludes farther plane."""
        from src.vision.nag.types import CameraParams, PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene
        from src.vision.nag.renderer import render_scene

        world_from_cam = np.eye(4, dtype=np.float32)
        world_from_cam[2, 3] = 10.0
        camera = CameraParams.from_fov(60.0, 64, 64, world_from_cam)

        # Far plane (blue) at z=0
        far_plane = PlaneParams.create_frontal((0, 0, 0), (6, 6), normal=(0, 0, 1))
        far_spline = PoseSplineParams.create_static((0, 0, 0))
        far_node = NAGPlaneNode(make_node_id("far"), far_plane, far_spline, atlas_size=(32, 32))
        with torch.no_grad():
            far_node.base_texture.fill_(0)
            far_node.base_texture[2].fill_(1.0)  # Blue
            far_node.base_alpha.fill_(1.0)

        # Near plane (green) at z=2
        near_plane = PlaneParams.create_frontal((0, 0, 2), (3, 3), normal=(0, 0, 1))
        near_spline = PoseSplineParams.create_static((0, 0, 2))
        near_node = NAGPlaneNode(make_node_id("near"), near_plane, near_spline, atlas_size=(32, 32))
        with torch.no_grad():
            near_node.base_texture.fill_(0)
            near_node.base_texture[1].fill_(1.0)  # Green
            near_node.base_alpha.fill_(1.0)

        scene = NAGScene()
        scene.add_node(make_node_id("far"), far_node)
        scene.add_node(make_node_id("near"), near_node)

        t = torch.tensor(0.0)
        result = render_scene(scene, camera, t)

        # Near plane is closer, so center should be more green
        center_rgb = result["rgb"][:, 28:36, 28:36].mean(dim=(1, 2))
        # Green should dominate in center (near plane)
        assert center_rgb[1] > center_rgb[2] * 0.5  # Green > Blue/2


class TestAtlasSampling:
    """Tests for atlas sampling behavior."""

    def test_identity_flow(self):
        """With zero flow, sampled atlas equals base texture."""
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode

        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("test"), plane, spline, atlas_size=(32, 32))

        # Set distinctive pattern
        with torch.no_grad():
            # Gradient texture
            for i in range(32):
                node.base_texture[:, i, :] = i / 31.0
            node.base_alpha.fill_(1.0)
            # Zero out flow MLP bias to get near-zero flow
            for param in node.flow_mlp.parameters():
                param.data.fill_(0)

        # Sample at center
        uv = torch.tensor([[0.5, 0.5]])
        t = torch.tensor(0.0)
        view_dir = torch.tensor([[0, 0, 1]], dtype=torch.float32)

        rgb, _ = node.sample_atlas(uv, t, view_dir)

        # Should be close to mid-gray (0.5)
        assert abs(rgb[0, 0].item() - 0.5) < 0.3


class TestEditorAPI:
    """Tests for editor operations."""

    def test_edit_pose(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene
        from src.vision.nag.editor import edit_pose

        scene = NAGScene()
        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("obj"), plane, spline)
        scene.add_node(make_node_id("obj"), node)

        # Apply pose edit
        edit = edit_pose(
            scene,
            make_node_id("obj"),
            delta_translation=torch.tensor([1.0, 2.0, 0.0]),
            delta_rotation_euler=torch.tensor([0.0, 0.0, 0.1]),
        )

        assert edit.edit_type == "pose"
        assert edit.node_id == make_node_id("obj")

        # Check pose changed
        updated_node = scene.get_node(make_node_id("obj"))
        pose = updated_node.pose_at(torch.tensor(0.0))
        assert pose[0, 3].item() > 0.5  # X translation applied

    def test_duplicate_node(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene
        from src.vision.nag.editor import duplicate_node

        scene = NAGScene()
        plane = PlaneParams.create_frontal((0, 0, 0), (1, 1))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("obj"), plane, spline)
        scene.add_node(make_node_id("obj"), node)

        # Duplicate
        edit = duplicate_node(
            scene,
            make_node_id("obj"),
            make_node_id("obj_copy"),
            {"translation": torch.tensor([2.0, 0.0, 0.0])},
        )

        assert edit.edit_type == "duplicate"
        assert scene.num_nodes() == 2
        assert scene.has_node(make_node_id("obj_copy"))

    def test_render_clip(self):
        from src.vision.nag.types import CameraParams, PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene
        from src.vision.nag.editor import render_clip

        world_from_cam = np.eye(4, dtype=np.float32)
        world_from_cam[2, 3] = 5.0
        camera = CameraParams.from_fov(60.0, 32, 32, world_from_cam)

        scene = NAGScene()
        plane = PlaneParams.create_frontal((0, 0, 0), (2, 2))
        spline = PoseSplineParams.create_static((0, 0, 0))
        node = NAGPlaneNode(make_node_id("obj"), plane, spline, atlas_size=(16, 16))
        scene.add_node(make_node_id("obj"), node)

        times = torch.linspace(0, 1, 5)
        clip = render_clip(scene, camera, times)

        assert clip.shape == (5, 3, 32, 32)

    def test_apply_random_edits(self):
        from src.vision.nag.types import PlaneParams, PoseSplineParams, make_node_id
        from src.vision.nag.plane_node import NAGPlaneNode
        from src.vision.nag.scene import NAGScene
        from src.vision.nag.editor import NAGEditPolicy, apply_random_edits

        scene = NAGScene()
        for i in range(3):
            plane = PlaneParams.create_frontal((i, 0, 0), (1, 1))
            spline = PoseSplineParams.create_static((i, 0, 0))
            node = NAGPlaneNode(make_node_id(f"obj_{i}"), plane, spline, atlas_size=(16, 16))
            scene.add_node(make_node_id(f"obj_{i}"), node)

        policy = NAGEditPolicy(
            prob_remove=0.2,
            prob_duplicate=0.2,
            prob_pose_shift=0.3,
            prob_color_shift=0.2,
        )

        rng = np.random.default_rng(42)
        edits = apply_random_edits(scene, policy, rng, max_edits=2)

        assert len(edits) <= 2
        for edit in edits:
            assert edit.edit_type in ["remove", "duplicate", "pose", "color_shift"]
