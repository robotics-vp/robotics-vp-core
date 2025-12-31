"""
LSD-3D style environment with geometry and Gaussian scene generation.

This package provides:
- Proxy geometry generation from scene graphs (voxels, meshes)
- 3D Gaussian scene representation
- GGDS (Geometry-Grounded Distillation Sampling) hooks for texturing
"""

from src.envs.lsd3d_env.proxy_geometry import (
    VoxelGrid,
    Mesh,
    scene_graph_to_voxels,
    voxels_to_mesh,
)
from src.envs.lsd3d_env.gaussian_scene import (
    GaussianScene,
    mesh_to_gaussians,
)
from src.envs.lsd3d_env.ggds import (
    GGDSConfig,
    GGDSOptimizer,
)

__all__ = [
    "VoxelGrid",
    "Mesh",
    "scene_graph_to_voxels",
    "voxels_to_mesh",
    "GaussianScene",
    "mesh_to_gaussians",
    "GGDSConfig",
    "GGDSOptimizer",
]
