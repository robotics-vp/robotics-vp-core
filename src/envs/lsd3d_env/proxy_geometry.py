"""
Proxy geometry generation from scene graphs.

Converts SceneGraph to voxel grids and meshes for LSD-3D style rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.scene.vector_scene.graph import (
    DEFAULT_NODE_WIDTHS,
    NodeType,
    SceneGraph,
    SceneNode,
    SceneObject,
)


@dataclass
class VoxelGrid:
    """
    3D voxel grid representation.

    Attributes:
        data: 3D array of voxel values (1 = solid, 0 = empty)
        origin: World coordinates of the grid origin (min corner)
        voxel_size: Size of each voxel in world units
        shape: Grid dimensions (nx, ny, nz)
    """
    data: np.ndarray  # (nx, ny, nz) uint8 or float32
    origin: Tuple[float, float, float]
    voxel_size: float
    shape: Tuple[int, int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.shape = tuple(self.data.shape)  # type: ignore

    def world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        ix = int((x - self.origin[0]) / self.voxel_size)
        iy = int((y - self.origin[1]) / self.voxel_size)
        iz = int((z - self.origin[2]) / self.voxel_size)
        return (ix, iy, iz)

    def voxel_to_world(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Convert voxel indices to world coordinates (center of voxel)."""
        x = self.origin[0] + (ix + 0.5) * self.voxel_size
        y = self.origin[1] + (iy + 0.5) * self.voxel_size
        z = self.origin[2] + (iz + 0.5) * self.voxel_size
        return (x, y, z)

    def is_valid_index(self, ix: int, iy: int, iz: int) -> bool:
        """Check if voxel indices are within bounds."""
        return (0 <= ix < self.shape[0] and
                0 <= iy < self.shape[1] and
                0 <= iz < self.shape[2])

    def set_voxel(self, ix: int, iy: int, iz: int, value: float = 1.0) -> None:
        """Set a voxel value if indices are valid."""
        if self.is_valid_index(ix, iy, iz):
            self.data[ix, iy, iz] = value

    def get_occupied_count(self) -> int:
        """Return number of occupied voxels."""
        return int(np.sum(self.data > 0))

    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return world-space bounding box of the grid."""
        min_corner = self.origin
        max_corner = (
            self.origin[0] + self.shape[0] * self.voxel_size,
            self.origin[1] + self.shape[1] * self.voxel_size,
            self.origin[2] + self.shape[2] * self.voxel_size,
        )
        return (min_corner, max_corner)


@dataclass
class Mesh:
    """
    Triangle mesh representation.

    Attributes:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
        normals: (N, 3) array of vertex normals (optional)
        colors: (N, 3) or (N, 4) array of vertex colors (optional)
        face_normals: (M, 3) array of face normals (computed if not provided)
    """
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32
    normals: Optional[np.ndarray] = None  # (N, 3) float32
    colors: Optional[np.ndarray] = None  # (N, 3) or (N, 4) float32
    face_normals: Optional[np.ndarray] = None  # (M, 3) float32

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float32)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.float32)
        if self.face_normals is None and len(self.faces) > 0:
            self.face_normals = self._compute_face_normals()

    def _compute_face_normals(self) -> np.ndarray:
        """Compute face normals from vertices and faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, where=norms > 1e-8, out=normals)
        return normals.astype(np.float32)

    def compute_vertex_normals(self) -> None:
        """Compute vertex normals by averaging adjacent face normals."""
        if self.face_normals is None:
            self.face_normals = self._compute_face_normals()

        vertex_normals = np.zeros_like(self.vertices)
        counts = np.zeros(len(self.vertices))

        for i, face in enumerate(self.faces):
            for vi in face:
                vertex_normals[vi] += self.face_normals[i]
                counts[vi] += 1

        counts = np.maximum(counts, 1)[:, np.newaxis]
        vertex_normals = vertex_normals / counts
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        self.normals = np.divide(vertex_normals, norms, where=norms > 1e-8, out=vertex_normals)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return min and max corners of the mesh bounding box."""
        if len(self.vertices) == 0:
            return (np.zeros(3), np.zeros(3))
        return (self.vertices.min(axis=0), self.vertices.max(axis=0))

    def get_face_centroids(self) -> np.ndarray:
        """Return centroids of all faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return (v0 + v1 + v2) / 3.0

    def get_face_areas(self) -> np.ndarray:
        """Return areas of all faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        return 0.5 * np.linalg.norm(cross, axis=1)


def _rasterize_polyline_corridor(
    voxels: VoxelGrid,
    polyline: np.ndarray,
    width: float,
    height: float,
    z_base: float = 0.0,
) -> None:
    """
    Rasterize a corridor/aisle along a polyline into the voxel grid.

    The corridor is extruded perpendicular to the polyline direction.
    """
    if len(polyline) < 2:
        return

    half_width = width / 2.0

    for i in range(len(polyline) - 1):
        p0 = polyline[i]
        p1 = polyline[i + 1]

        # Direction and perpendicular
        direction = p1[:2] - p0[:2]
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length
        perp = np.array([-direction[1], direction[0]])

        # Sample along the segment
        num_samples = max(2, int(length / voxels.voxel_size) + 1)
        for t in np.linspace(0, 1, num_samples):
            center = p0[:2] + t * (p1[:2] - p0[:2])

            # Sample across the width
            num_width_samples = max(2, int(width / voxels.voxel_size) + 1)
            for w in np.linspace(-half_width, half_width, num_width_samples):
                x = center[0] + w * perp[0]
                y = center[1] + w * perp[1]

                # Rasterize the height column
                num_height_samples = max(1, int(height / voxels.voxel_size) + 1)
                for h in np.linspace(z_base, z_base + height, num_height_samples):
                    ix, iy, iz = voxels.world_to_voxel(x, y, h)
                    voxels.set_voxel(ix, iy, iz, 1.0)


def _rasterize_box(
    voxels: VoxelGrid,
    x: float,
    y: float,
    z: float,
    length: float,
    width: float,
    height: float,
    heading: float = 0.0,
) -> None:
    """
    Rasterize an axis-aligned box into the voxel grid.

    For simplicity, heading rotation is ignored (objects treated as axis-aligned).
    """
    half_l = length / 2.0
    half_w = width / 2.0

    # Sample the box volume
    x_samples = max(2, int(length / voxels.voxel_size) + 1)
    y_samples = max(2, int(width / voxels.voxel_size) + 1)
    z_samples = max(2, int(height / voxels.voxel_size) + 1)

    for dx in np.linspace(-half_l, half_l, x_samples):
        for dy in np.linspace(-half_w, half_w, y_samples):
            for dz in np.linspace(0, height, z_samples):
                ix, iy, iz = voxels.world_to_voxel(x + dx, y + dy, z + dz)
                voxels.set_voxel(ix, iy, iz, 1.0)


def scene_graph_to_voxels(
    graph: SceneGraph,
    voxel_size: float = 0.1,
    margin: float = 2.0,
    default_height: float = 3.0,
) -> VoxelGrid:
    """
    Convert a SceneGraph to a voxel grid.

    Args:
        graph: Input SceneGraph
        voxel_size: Size of each voxel in world units (meters)
        margin: Margin to add around the scene bounding box
        default_height: Default height for corridors/rooms

    Returns:
        VoxelGrid containing rasterized scene geometry
    """
    # Compute scene bounding box
    bbox = graph.bounding_box()
    min_x, min_y, max_x, max_y = bbox

    # Add margin
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    # Determine grid dimensions
    nx = max(1, int(np.ceil((max_x - min_x) / voxel_size)))
    ny = max(1, int(np.ceil((max_y - min_y) / voxel_size)))
    nz = max(1, int(np.ceil(default_height / voxel_size)))

    # Create empty voxel grid
    data = np.zeros((nx, ny, nz), dtype=np.float32)
    voxels = VoxelGrid(
        data=data,
        origin=(min_x, min_y, 0.0),
        voxel_size=voxel_size,
    )

    # Rasterize nodes (corridors, rooms, etc.)
    for node in graph.nodes:
        width = node.width if node.width is not None else DEFAULT_NODE_WIDTHS.get(node.node_type, 2.0)
        height = node.height if node.height is not None else default_height

        _rasterize_polyline_corridor(
            voxels,
            node.polyline,
            width=width,
            height=height,
            z_base=0.0,
        )

    # Rasterize objects as solid boxes
    for obj in graph.objects:
        _rasterize_box(
            voxels,
            obj.x,
            obj.y,
            obj.z,
            obj.length,
            obj.width,
            obj.height,
            obj.heading,
        )

    return voxels


def voxels_to_mesh(
    voxels: VoxelGrid,
    threshold: float = 0.5,
) -> Mesh:
    """
    Convert a voxel grid to a triangle mesh using naive surface extraction.

    This uses a simple approach where each exposed voxel face becomes two triangles.
    For production use, consider marching cubes for smoother surfaces.

    Args:
        voxels: Input VoxelGrid
        threshold: Threshold for considering a voxel occupied

    Returns:
        Mesh with vertices, faces, and computed normals
    """
    vertices = []
    faces = []

    occupied = voxels.data > threshold
    nx, ny, nz = voxels.shape
    vs = voxels.voxel_size

    # For each occupied voxel, add faces for exposed sides
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not occupied[ix, iy, iz]:
                    continue

                # Get world position of voxel center
                cx, cy, cz = voxels.voxel_to_world(ix, iy, iz)
                half = vs / 2.0

                # Check each of 6 faces
                # -X face
                if ix == 0 or not occupied[ix - 1, iy, iz]:
                    v_base = len(vertices)
                    vertices.extend([
                        [cx - half, cy - half, cz - half],
                        [cx - half, cy + half, cz - half],
                        [cx - half, cy + half, cz + half],
                        [cx - half, cy - half, cz + half],
                    ])
                    faces.append([v_base, v_base + 2, v_base + 1])
                    faces.append([v_base, v_base + 3, v_base + 2])

                # +X face
                if ix == nx - 1 or not occupied[ix + 1, iy, iz]:
                    v_base = len(vertices)
                    vertices.extend([
                        [cx + half, cy - half, cz - half],
                        [cx + half, cy + half, cz - half],
                        [cx + half, cy + half, cz + half],
                        [cx + half, cy - half, cz + half],
                    ])
                    faces.append([v_base, v_base + 1, v_base + 2])
                    faces.append([v_base, v_base + 2, v_base + 3])

                # -Y face
                if iy == 0 or not occupied[ix, iy - 1, iz]:
                    v_base = len(vertices)
                    vertices.extend([
                        [cx - half, cy - half, cz - half],
                        [cx + half, cy - half, cz - half],
                        [cx + half, cy - half, cz + half],
                        [cx - half, cy - half, cz + half],
                    ])
                    faces.append([v_base, v_base + 1, v_base + 2])
                    faces.append([v_base, v_base + 2, v_base + 3])

                # +Y face
                if iy == ny - 1 or not occupied[ix, iy + 1, iz]:
                    v_base = len(vertices)
                    vertices.extend([
                        [cx - half, cy + half, cz - half],
                        [cx + half, cy + half, cz - half],
                        [cx + half, cy + half, cz + half],
                        [cx - half, cy + half, cz + half],
                    ])
                    faces.append([v_base, v_base + 2, v_base + 1])
                    faces.append([v_base, v_base + 3, v_base + 2])

                # -Z face (floor)
                if iz == 0 or not occupied[ix, iy, iz - 1]:
                    v_base = len(vertices)
                    vertices.extend([
                        [cx - half, cy - half, cz - half],
                        [cx + half, cy - half, cz - half],
                        [cx + half, cy + half, cz - half],
                        [cx - half, cy + half, cz - half],
                    ])
                    faces.append([v_base, v_base + 2, v_base + 1])
                    faces.append([v_base, v_base + 3, v_base + 2])

                # +Z face (ceiling)
                if iz == nz - 1 or not occupied[ix, iy, iz + 1]:
                    v_base = len(vertices)
                    vertices.extend([
                        [cx - half, cy - half, cz + half],
                        [cx + half, cy - half, cz + half],
                        [cx + half, cy + half, cz + half],
                        [cx - half, cy + half, cz + half],
                    ])
                    faces.append([v_base, v_base + 1, v_base + 2])
                    faces.append([v_base, v_base + 2, v_base + 3])

    if not vertices:
        # Return empty mesh
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
        )

    mesh = Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )
    mesh.compute_vertex_normals()

    return mesh
