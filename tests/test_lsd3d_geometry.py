"""
Tests for LSD-3D style geometry (voxels, mesh, gaussians).
"""

import numpy as np
import pytest

from src.envs.lsd3d_env.proxy_geometry import (
    Mesh,
    VoxelGrid,
    scene_graph_to_voxels,
    voxels_to_mesh,
)
from src.envs.lsd3d_env.gaussian_scene import (
    GaussianScene,
    combine_scenes,
    mesh_to_gaussians,
)
from src.scene.vector_scene.graph import SceneGraph


class TestVoxelGrid:
    def test_voxel_grid_creation(self):
        data = np.zeros((10, 10, 5), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(0.0, 0.0, 0.0), voxel_size=0.1)
        assert voxels.shape == (10, 10, 5)
        assert voxels.voxel_size == 0.1

    def test_world_to_voxel(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(0.0, 0.0, 0.0), voxel_size=0.5)
        ix, iy, iz = voxels.world_to_voxel(1.0, 2.0, 3.0)
        assert ix == 2
        assert iy == 4
        assert iz == 6

    def test_voxel_to_world(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(1.0, 2.0, 0.0), voxel_size=0.5)
        x, y, z = voxels.voxel_to_world(2, 4, 6)
        assert abs(x - 2.25) < 1e-6  # 1.0 + 2.5 * 0.5
        assert abs(y - 4.25) < 1e-6  # 2.0 + 4.5 * 0.5
        assert abs(z - 3.25) < 1e-6  # 0.0 + 6.5 * 0.5

    def test_set_voxel(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(0.0, 0.0, 0.0), voxel_size=0.1)
        voxels.set_voxel(5, 5, 5, 1.0)
        assert voxels.data[5, 5, 5] == 1.0
        assert voxels.get_occupied_count() == 1

    def test_is_valid_index(self):
        data = np.zeros((10, 10, 5), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(0.0, 0.0, 0.0), voxel_size=0.1)
        assert voxels.is_valid_index(0, 0, 0)
        assert voxels.is_valid_index(9, 9, 4)
        assert not voxels.is_valid_index(10, 0, 0)
        assert not voxels.is_valid_index(-1, 0, 0)

    def test_bounding_box(self):
        data = np.zeros((10, 20, 5), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(1.0, 2.0, 0.0), voxel_size=0.5)
        min_corner, max_corner = voxels.get_bounding_box()
        assert min_corner == (1.0, 2.0, 0.0)
        assert max_corner == (6.0, 12.0, 2.5)


class TestMesh:
    def test_mesh_creation(self):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        assert len(mesh.vertices) == 3
        assert len(mesh.faces) == 1
        assert mesh.face_normals is not None

    def test_face_normals(self):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        # Normal should point in +z direction
        assert abs(mesh.face_normals[0, 2] - 1.0) < 1e-6

    def test_face_centroids(self):
        vertices = np.array([
            [0, 0, 0],
            [3, 0, 0],
            [0, 3, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        centroids = mesh.get_face_centroids()
        assert len(centroids) == 1
        assert np.allclose(centroids[0], [1, 1, 0])

    def test_face_areas(self):
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        areas = mesh.get_face_areas()
        # Area of right triangle with legs 2 and 2 is 2
        assert abs(areas[0] - 2.0) < 1e-6

    def test_bounding_box(self):
        vertices = np.array([
            [-1, -2, -3],
            [4, 5, 6],
            [0, 0, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        min_corner, max_corner = mesh.get_bounding_box()
        assert np.allclose(min_corner, [-1, -2, -3])
        assert np.allclose(max_corner, [4, 5, 6])


class TestSceneGraphToVoxels:
    def test_simple_warehouse(self):
        graph = SceneGraph.create_simple_warehouse(num_aisles=3, aisle_length=10.0)
        voxels = scene_graph_to_voxels(graph, voxel_size=0.5)
        assert voxels.get_occupied_count() > 0
        assert voxels.shape[0] > 0
        assert voxels.shape[1] > 0
        assert voxels.shape[2] > 0

    def test_empty_graph(self):
        graph = SceneGraph()
        voxels = scene_graph_to_voxels(graph, voxel_size=0.5, margin=1.0)
        # Even empty graph should create a grid
        assert voxels.shape[0] >= 1


class TestVoxelsToMesh:
    def test_simple_voxels(self):
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[2, 2, 2] = 1.0  # Single occupied voxel
        voxels = VoxelGrid(data=data, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        mesh = voxels_to_mesh(voxels)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        # A cube has 6 faces * 2 triangles = 12 triangles
        assert len(mesh.faces) == 12

    def test_empty_voxels(self):
        data = np.zeros((5, 5, 5), dtype=np.float32)
        voxels = VoxelGrid(data=data, origin=(0.0, 0.0, 0.0), voxel_size=1.0)
        mesh = voxels_to_mesh(voxels)
        assert len(mesh.vertices) == 0
        assert len(mesh.faces) == 0

    def test_from_scene_graph(self):
        graph = SceneGraph.create_simple_warehouse(num_aisles=2, aisle_length=5.0)
        voxels = scene_graph_to_voxels(graph, voxel_size=0.5)
        mesh = voxels_to_mesh(voxels)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


class TestGaussianScene:
    def test_creation(self):
        means = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        covs = np.array([[1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1]], dtype=np.float32)
        colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        opacities = np.array([0.8, 0.9], dtype=np.float32)
        normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        scene = GaussianScene(
            means=means,
            covs=covs,
            colors=colors,
            opacities=opacities,
            normals=normals,
        )
        assert scene.num_gaussians == 2

    def test_bounding_box(self):
        means = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.float32)
        scene = GaussianScene(
            means=means,
            covs=np.zeros((2, 6)),
            colors=np.zeros((2, 3)),
            opacities=np.zeros(2),
            normals=np.zeros((2, 3)),
        )
        min_corner, max_corner = scene.get_bounding_box()
        assert np.allclose(min_corner, [-1, -2, -3])
        assert np.allclose(max_corner, [4, 5, 6])

    def test_filter_by_opacity(self):
        means = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        opacities = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        scene = GaussianScene(
            means=means,
            covs=np.zeros((3, 6)),
            colors=np.zeros((3, 3)),
            opacities=opacities,
            normals=np.zeros((3, 3)),
        )
        filtered = scene.filter_by_opacity(threshold=0.3)
        assert filtered.num_gaussians == 2

    def test_subsample(self):
        n = 100
        scene = GaussianScene(
            means=np.random.randn(n, 3).astype(np.float32),
            covs=np.random.rand(n, 6).astype(np.float32),
            colors=np.random.rand(n, 3).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            normals=np.random.randn(n, 3).astype(np.float32),
        )
        subsampled = scene.subsample(10, seed=42)
        assert subsampled.num_gaussians == 10

    def test_to_dict_and_back(self):
        scene = GaussianScene(
            means=np.array([[0, 0, 0]], dtype=np.float32),
            covs=np.array([[1, 0, 0, 1, 0, 1]], dtype=np.float32),
            colors=np.array([[1, 0, 0]], dtype=np.float32),
            opacities=np.array([0.8], dtype=np.float32),
            normals=np.array([[0, 0, 1]], dtype=np.float32),
        )
        data = scene.to_dict()
        restored = GaussianScene.from_dict(data)
        assert np.allclose(restored.means, scene.means)
        assert np.allclose(restored.opacities, scene.opacities)


class TestMeshToGaussians:
    def test_simple_mesh(self):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        scene = mesh_to_gaussians(mesh, gaussians_per_face=1)
        assert scene.num_gaussians == 1
        # Gaussian should be at centroid
        centroid = mesh.get_face_centroids()[0]
        assert np.allclose(scene.means[0], centroid)

    def test_empty_mesh(self):
        mesh = Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
        )
        scene = mesh_to_gaussians(mesh)
        assert scene.num_gaussians == 0

    def test_multiple_gaussians_per_face(self):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        scene = mesh_to_gaussians(mesh, gaussians_per_face=5)
        assert scene.num_gaussians == 5


class TestCombineScenes:
    def test_combine_two_scenes(self):
        scene1 = GaussianScene(
            means=np.array([[0, 0, 0]], dtype=np.float32),
            covs=np.zeros((1, 6), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.float32),
            opacities=np.ones(1, dtype=np.float32),
            normals=np.zeros((1, 3), dtype=np.float32),
        )
        scene2 = GaussianScene(
            means=np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
            covs=np.zeros((2, 6), dtype=np.float32),
            colors=np.zeros((2, 3), dtype=np.float32),
            opacities=np.ones(2, dtype=np.float32),
            normals=np.zeros((2, 3), dtype=np.float32),
        )
        combined = combine_scenes([scene1, scene2])
        assert combined.num_gaussians == 3

    def test_combine_empty_list(self):
        combined = combine_scenes([])
        assert combined.num_gaussians == 0
