"""
3D Gaussian scene representation for LSD-3D style rendering.

Provides GaussianScene class for representing scenes as collections of
3D Gaussians, suitable for differentiable rendering and GGDS optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.envs.lsd3d_env.proxy_geometry import Mesh


@dataclass
class GaussianScene:
    """
    3D Gaussian scene representation.

    Each Gaussian is defined by:
    - Position (mean): 3D location
    - Covariance: 3x3 symmetric positive-definite matrix (stored as 6 params)
    - Color: RGB values
    - Opacity: Alpha value
    - Normal: Surface normal (for geometry-aware rendering)

    Attributes:
        means: (N, 3) array of Gaussian centers
        covs: (N, 6) array of covariance parameters (upper triangle)
        colors: (N, 3) array of RGB colors in [0, 1]
        opacities: (N,) array of opacity values in [0, 1]
        normals: (N, 3) array of surface normals
        scales: (N, 3) array of scale factors (alternative to covariance)
        rotations: (N, 4) array of quaternion rotations
        sh_coeffs: (N, C, 3) array of spherical harmonic coefficients (optional)
    """
    means: np.ndarray  # (N, 3) float32
    covs: np.ndarray  # (N, 6) float32 - upper triangle of 3x3 covariance
    colors: np.ndarray  # (N, 3) float32
    opacities: np.ndarray  # (N,) float32
    normals: np.ndarray  # (N, 3) float32
    scales: Optional[np.ndarray] = None  # (N, 3) float32
    rotations: Optional[np.ndarray] = None  # (N, 4) float32 quaternion
    sh_coeffs: Optional[np.ndarray] = None  # (N, C, 3) float32
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.covs = np.asarray(self.covs, dtype=np.float32)
        self.colors = np.asarray(self.colors, dtype=np.float32)
        self.opacities = np.asarray(self.opacities, dtype=np.float32)
        self.normals = np.asarray(self.normals, dtype=np.float32)

        if self.scales is not None:
            self.scales = np.asarray(self.scales, dtype=np.float32)
        if self.rotations is not None:
            self.rotations = np.asarray(self.rotations, dtype=np.float32)
        if self.sh_coeffs is not None:
            self.sh_coeffs = np.asarray(self.sh_coeffs, dtype=np.float32)

    @property
    def num_gaussians(self) -> int:
        """Return number of Gaussians in the scene."""
        return len(self.means)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return min and max corners of the scene bounding box."""
        if len(self.means) == 0:
            return (np.zeros(3), np.zeros(3))
        return (self.means.min(axis=0), self.means.max(axis=0))

    def filter_by_opacity(self, threshold: float = 0.1) -> "GaussianScene":
        """Return a new scene with only Gaussians above opacity threshold."""
        mask = self.opacities > threshold
        return GaussianScene(
            means=self.means[mask],
            covs=self.covs[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            normals=self.normals[mask],
            scales=self.scales[mask] if self.scales is not None else None,
            rotations=self.rotations[mask] if self.rotations is not None else None,
            sh_coeffs=self.sh_coeffs[mask] if self.sh_coeffs is not None else None,
            metadata=self.metadata.copy(),
        )

    def subsample(self, n: int, seed: Optional[int] = None) -> "GaussianScene":
        """Return a random subsample of n Gaussians."""
        if n >= self.num_gaussians:
            return self

        rng = np.random.default_rng(seed)
        indices = rng.choice(self.num_gaussians, size=n, replace=False)

        return GaussianScene(
            means=self.means[indices],
            covs=self.covs[indices],
            colors=self.colors[indices],
            opacities=self.opacities[indices],
            normals=self.normals[indices],
            scales=self.scales[indices] if self.scales is not None else None,
            rotations=self.rotations[indices] if self.rotations is not None else None,
            sh_coeffs=self.sh_coeffs[indices] if self.sh_coeffs is not None else None,
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "means": self.means.tolist(),
            "covs": self.covs.tolist(),
            "colors": self.colors.tolist(),
            "opacities": self.opacities.tolist(),
            "normals": self.normals.tolist(),
            "metadata": self.metadata,
        }
        if self.scales is not None:
            result["scales"] = self.scales.tolist()
        if self.rotations is not None:
            result["rotations"] = self.rotations.tolist()
        if self.sh_coeffs is not None:
            result["sh_coeffs"] = self.sh_coeffs.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianScene":
        """Create from dictionary."""
        return cls(
            means=np.array(data["means"]),
            covs=np.array(data["covs"]),
            colors=np.array(data["colors"]),
            opacities=np.array(data["opacities"]),
            normals=np.array(data["normals"]),
            scales=np.array(data["scales"]) if "scales" in data else None,
            rotations=np.array(data["rotations"]) if "rotations" in data else None,
            sh_coeffs=np.array(data["sh_coeffs"]) if "sh_coeffs" in data else None,
            metadata=data.get("metadata", {}),
        )

    def clone(self) -> "GaussianScene":
        """Return a deep copy."""
        return GaussianScene(
            means=self.means.copy(),
            covs=self.covs.copy(),
            colors=self.colors.copy(),
            opacities=self.opacities.copy(),
            normals=self.normals.copy(),
            scales=self.scales.copy() if self.scales is not None else None,
            rotations=self.rotations.copy() if self.rotations is not None else None,
            sh_coeffs=self.sh_coeffs.copy() if self.sh_coeffs is not None else None,
            metadata=self.metadata.copy(),
        )


def _covariance_from_scale_and_rotation(
    scale: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute covariance matrix from scale and rotation.

    Args:
        scale: (3,) scale factors
        rotation: (4,) quaternion [w, x, y, z] or None for identity

    Returns:
        (6,) upper triangle of 3x3 covariance matrix
    """
    # Scale matrix
    S = np.diag(scale ** 2)

    if rotation is not None:
        # Convert quaternion to rotation matrix
        w, x, y, z = rotation
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        cov = R @ S @ R.T
    else:
        cov = S

    # Extract upper triangle: [cov[0,0], cov[0,1], cov[0,2], cov[1,1], cov[1,2], cov[2,2]]
    return np.array([
        cov[0, 0], cov[0, 1], cov[0, 2],
        cov[1, 1], cov[1, 2], cov[2, 2],
    ], dtype=np.float32)


def mesh_to_gaussians(
    mesh: Mesh,
    gaussians_per_face: int = 1,
    base_scale: float = 0.1,
    default_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    default_opacity: float = 0.8,
) -> GaussianScene:
    """
    Initialize 3D Gaussians from a mesh.

    Places Gaussians at triangle centroids with scales derived from face areas.

    Args:
        mesh: Input triangle mesh
        gaussians_per_face: Number of Gaussians per face (1 = centroid only)
        base_scale: Base scale factor for Gaussians
        default_color: Default RGB color for Gaussians
        default_opacity: Default opacity

    Returns:
        GaussianScene initialized from the mesh
    """
    if len(mesh.faces) == 0:
        # Return empty scene
        return GaussianScene(
            means=np.zeros((0, 3), dtype=np.float32),
            covs=np.zeros((0, 6), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros(0, dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
        )

    centroids = mesh.get_face_centroids()
    areas = mesh.get_face_areas()
    face_normals = mesh.face_normals if mesh.face_normals is not None else np.zeros_like(centroids)

    # Compute scales from areas (approximate as sqrt(area) for characteristic length)
    face_scales = np.sqrt(areas + 1e-8) * base_scale

    all_means = []
    all_covs = []
    all_colors = []
    all_opacities = []
    all_normals = []
    all_scales = []
    all_rotations = []

    for i, (centroid, area, normal, face_scale) in enumerate(zip(centroids, areas, face_normals, face_scales)):
        if gaussians_per_face == 1:
            # Single Gaussian at centroid
            positions = [centroid]
        else:
            # Multiple Gaussians sampled within the face
            v0 = mesh.vertices[mesh.faces[i, 0]]
            v1 = mesh.vertices[mesh.faces[i, 1]]
            v2 = mesh.vertices[mesh.faces[i, 2]]

            positions = []
            for _ in range(gaussians_per_face):
                # Random barycentric coordinates
                r1, r2 = np.random.random(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                r3 = 1 - r1 - r2
                pos = r1 * v0 + r2 * v1 + r3 * v2
                positions.append(pos)

        for pos in positions:
            all_means.append(pos)

            # Isotropic scale for simplicity
            scale = np.array([face_scale, face_scale, face_scale], dtype=np.float32)
            all_scales.append(scale)

            # Identity rotation
            rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            all_rotations.append(rotation)

            # Compute covariance
            cov = _covariance_from_scale_and_rotation(scale, rotation)
            all_covs.append(cov)

            all_colors.append(np.array(default_color, dtype=np.float32))
            all_opacities.append(default_opacity)
            all_normals.append(normal)

    return GaussianScene(
        means=np.array(all_means, dtype=np.float32),
        covs=np.array(all_covs, dtype=np.float32),
        colors=np.array(all_colors, dtype=np.float32),
        opacities=np.array(all_opacities, dtype=np.float32),
        normals=np.array(all_normals, dtype=np.float32),
        scales=np.array(all_scales, dtype=np.float32),
        rotations=np.array(all_rotations, dtype=np.float32),
        metadata={"source": "mesh", "gaussians_per_face": gaussians_per_face},
    )


def combine_scenes(scenes: List[GaussianScene]) -> GaussianScene:
    """Combine multiple GaussianScenes into one."""
    if not scenes:
        return GaussianScene(
            means=np.zeros((0, 3), dtype=np.float32),
            covs=np.zeros((0, 6), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros(0, dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
        )

    return GaussianScene(
        means=np.concatenate([s.means for s in scenes]),
        covs=np.concatenate([s.covs for s in scenes]),
        colors=np.concatenate([s.colors for s in scenes]),
        opacities=np.concatenate([s.opacities for s in scenes]),
        normals=np.concatenate([s.normals for s in scenes]),
        scales=np.concatenate([s.scales for s in scenes if s.scales is not None]) if any(s.scales is not None for s in scenes) else None,
        rotations=np.concatenate([s.rotations for s in scenes if s.rotations is not None]) if any(s.rotations is not None for s in scenes) else None,
        metadata={"source": "combined", "num_sources": len(scenes)},
    )
