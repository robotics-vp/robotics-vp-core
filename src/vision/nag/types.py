"""
Core types for Neural Atlas-Graph (NAG) representation.

Provides small, explicit types used throughout the NAG module:
- CameraParams: intrinsics + extrinsics per frame
- PlaneParams: world_from_plane transform + extent
- PoseSplineParams: SE(3) pose trajectory via cubic spline
- NAGNodeId: lightweight identifier for NAG nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    TORCH_AVAILABLE = False


# Lightweight wrapper for node identifiers
NAGNodeId = NewType("NAGNodeId", str)


def make_node_id(value: Union[str, int]) -> NAGNodeId:
    """Create a NAGNodeId from string or int."""
    return NAGNodeId(str(value))


@dataclass
class CameraParams:
    """
    Camera parameters for a frame sequence.

    Stores intrinsics and per-frame extrinsics for rendering.
    Follows the convention: world_from_cam transforms points from
    camera space to world space.

    Attributes:
        fx: Focal length x (pixels)
        fy: Focal length y (pixels)
        cx: Principal point x (pixels)
        cy: Principal point y (pixels)
        height: Image height in pixels
        width: Image width in pixels
        world_from_cam: (T, 4, 4) or (4, 4) extrinsics transforms
        near: Near clipping plane
        far: Far clipping plane
    """
    fx: float
    fy: float
    cx: float
    cy: float
    height: int
    width: int
    world_from_cam: np.ndarray  # (T, 4, 4) or (4, 4)
    near: float = 0.01
    far: float = 100.0

    def __post_init__(self) -> None:
        self.world_from_cam = np.asarray(self.world_from_cam, dtype=np.float32)
        if self.world_from_cam.ndim == 2:
            # Single frame - add time dimension
            self.world_from_cam = self.world_from_cam[np.newaxis, ...]

    @property
    def num_frames(self) -> int:
        """Number of frames in the sequence."""
        return self.world_from_cam.shape[0]

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float32)

    @property
    def K_inv(self) -> np.ndarray:
        """Inverse of intrinsic matrix."""
        return np.linalg.inv(self.K)

    @property
    def fov_deg(self) -> float:
        """Approximate vertical field of view in degrees."""
        return 2.0 * np.degrees(np.arctan(self.height / (2.0 * self.fy)))

    def cam_from_world(self, t: int = 0) -> np.ndarray:
        """Get camera_from_world transform at time t."""
        return np.linalg.inv(self.world_from_cam[t])

    def get_rays(self, t: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ray origins and directions for all pixels at time t.

        Returns:
            origins: (H, W, 3) ray origins in world space
            directions: (H, W, 3) normalized ray directions in world space
        """
        H, W = self.height, self.width

        # Pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.astype(np.float32) + 0.5  # pixel centers
        v = v.astype(np.float32) + 0.5

        # Unproject to camera space (z=1)
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        z_cam = np.ones_like(x_cam)

        # Stack into (H, W, 3) directions in camera space
        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world space
        w2c = self.world_from_cam[t]
        R = w2c[:3, :3]
        t_vec = w2c[:3, 3]

        # Origins are camera position
        origins = np.broadcast_to(t_vec, (H, W, 3)).copy()

        # Directions: rotate from camera to world
        dirs_world = np.einsum("ij,hwj->hwi", R, dirs_cam)
        dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8)

        return origins, dirs_world

    @classmethod
    def from_fov(
        cls,
        fov_deg: float,
        height: int,
        width: int,
        world_from_cam: np.ndarray,
        near: float = 0.01,
        far: float = 100.0,
    ) -> "CameraParams":
        """Create CameraParams from field of view and resolution."""
        fy = height / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        fx = fy  # Assume square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(
            fx=fx, fy=fy, cx=cx, cy=cy,
            height=height, width=width,
            world_from_cam=world_from_cam,
            near=near, far=far,
        )

    @classmethod
    def from_camera_rig(
        cls,
        positions: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        fov: float,
        width: int,
        height: int,
    ) -> "CameraParams":
        """
        Create CameraParams from eye/target/up vectors (compatible with CameraRig).

        Args:
            positions: (N, 3) camera positions
            look_at: (N, 3) look-at points
            up: (N, 3) up vectors
            fov: Field of view in degrees
            width: Image width
            height: Image height
        """
        positions = np.asarray(positions, dtype=np.float32)
        look_at = np.asarray(look_at, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        if positions.ndim == 1:
            positions = positions[np.newaxis, :]
            look_at = look_at[np.newaxis, :]
            up = up[np.newaxis, :]

        N = positions.shape[0]
        world_from_cam = np.zeros((N, 4, 4), dtype=np.float32)

        for i in range(N):
            # Build rotation matrix (camera looks down -Z in camera space)
            forward = look_at[i] - positions[i]
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            right = np.cross(forward, up[i])
            right = right / (np.linalg.norm(right) + 1e-8)

            up_corrected = np.cross(right, forward)

            # world_from_cam: columns are right, up, -forward (camera convention)
            R = np.stack([right, up_corrected, -forward], axis=1)

            world_from_cam[i, :3, :3] = R
            world_from_cam[i, :3, 3] = positions[i]
            world_from_cam[i, 3, 3] = 1.0

        return cls.from_fov(fov, height, width, world_from_cam)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "height": self.height,
            "width": self.width,
            "world_from_cam": self.world_from_cam.tolist(),
            "near": self.near,
            "far": self.far,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraParams":
        """Create from dictionary."""
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            height=data["height"],
            width=data["width"],
            world_from_cam=np.array(data["world_from_cam"]),
            near=data.get("near", 0.01),
            far=data.get("far", 100.0),
        )


@dataclass
class PlaneParams:
    """
    Parameters for a 3D plane in world space.

    The plane is defined by a 4x4 transform (world_from_plane) and an extent.
    Points on the plane in local coordinates have z=0.

    Attributes:
        world_from_plane: 4x4 transform from plane-local to world coordinates
        extent: (width, height) of the plane in world units
    """
    world_from_plane: np.ndarray  # (4, 4)
    extent: np.ndarray  # (2,) width, height in world units

    def __post_init__(self) -> None:
        self.world_from_plane = np.asarray(self.world_from_plane, dtype=np.float32)
        self.extent = np.asarray(self.extent, dtype=np.float32)

    @property
    def origin(self) -> np.ndarray:
        """Origin of the plane in world coordinates."""
        return self.world_from_plane[:3, 3]

    @property
    def normal(self) -> np.ndarray:
        """Normal vector of the plane (Z-axis in plane space)."""
        return self.world_from_plane[:3, 2]

    @property
    def right(self) -> np.ndarray:
        """Right vector of the plane (X-axis in plane space)."""
        return self.world_from_plane[:3, 0]

    @property
    def up(self) -> np.ndarray:
        """Up vector of the plane (Y-axis in plane space)."""
        return self.world_from_plane[:3, 1]

    def plane_from_world(self) -> np.ndarray:
        """Get world_to_plane transform."""
        return np.linalg.inv(self.world_from_plane)

    def world_point_to_uv(self, points: np.ndarray) -> np.ndarray:
        """
        Convert world points to UV coordinates on the plane.

        Args:
            points: (..., 3) world coordinates

        Returns:
            uv: (..., 2) in [0, 1] if on plane
        """
        shape = points.shape[:-1]
        points_flat = points.reshape(-1, 3)

        # Transform to plane space
        p_from_w = self.plane_from_world()
        ones = np.ones((points_flat.shape[0], 1), dtype=np.float32)
        homogeneous = np.concatenate([points_flat, ones], axis=1)
        local = (p_from_w @ homogeneous.T).T[:, :3]

        # Convert to UV (assuming plane center is at origin)
        u = local[:, 0] / self.extent[0] + 0.5
        v = local[:, 1] / self.extent[1] + 0.5

        uv = np.stack([u, v], axis=-1)
        return uv.reshape(*shape, 2)

    def uv_to_world_point(self, uv: np.ndarray) -> np.ndarray:
        """
        Convert UV coordinates to world points on the plane.

        Args:
            uv: (..., 2) in [0, 1]

        Returns:
            points: (..., 3) world coordinates
        """
        shape = uv.shape[:-1]
        uv_flat = uv.reshape(-1, 2)

        # UV to plane-local coordinates
        local_x = (uv_flat[:, 0] - 0.5) * self.extent[0]
        local_y = (uv_flat[:, 1] - 0.5) * self.extent[1]
        local_z = np.zeros_like(local_x)
        ones = np.ones_like(local_x)

        local = np.stack([local_x, local_y, local_z, ones], axis=1)

        # Transform to world
        world = (self.world_from_plane @ local.T).T[:, :3]
        return world.reshape(*shape, 3)

    @classmethod
    def create_frontal(
        cls,
        center: Tuple[float, float, float],
        extent: Tuple[float, float],
        normal: Tuple[float, float, float] = (0, 0, 1),
    ) -> "PlaneParams":
        """
        Create a plane facing a given direction.

        Args:
            center: Center of the plane in world coordinates
            extent: (width, height) of the plane
            normal: Normal direction of the plane
        """
        center = np.asarray(center, dtype=np.float32)
        normal = np.asarray(normal, dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        # Build orthonormal basis
        up_hint = np.array([0, 1, 0], dtype=np.float32)
        if abs(np.dot(normal, up_hint)) > 0.99:
            up_hint = np.array([1, 0, 0], dtype=np.float32)

        right = np.cross(up_hint, normal)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(normal, right)

        # Build transform
        world_from_plane = np.eye(4, dtype=np.float32)
        world_from_plane[:3, 0] = right
        world_from_plane[:3, 1] = up
        world_from_plane[:3, 2] = normal
        world_from_plane[:3, 3] = center

        return cls(
            world_from_plane=world_from_plane,
            extent=np.array(extent, dtype=np.float32),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "world_from_plane": self.world_from_plane.tolist(),
            "extent": self.extent.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaneParams":
        """Create from dictionary."""
        return cls(
            world_from_plane=np.array(data["world_from_plane"]),
            extent=np.array(data["extent"]),
        )


@dataclass
class PoseSplineParams:
    """
    SE(3) pose trajectory represented as a cubic spline.

    Uses separate splines for translation and Euler angles (roll, pitch, yaw).
    This is a simple but effective representation that supports:
    - Evaluation at arbitrary times
    - Smooth interpolation
    - Differentiable through standard spline operations

    Attributes:
        knot_times: (K,) monotone times for spline knots
        translations: (K, 3) position at each knot
        euler_angles: (K, 3) Euler angles (roll, pitch, yaw) at each knot
    """
    knot_times: np.ndarray  # (K,)
    translations: np.ndarray  # (K, 3)
    euler_angles: np.ndarray  # (K, 3) roll, pitch, yaw

    def __post_init__(self) -> None:
        self.knot_times = np.asarray(self.knot_times, dtype=np.float32)
        self.translations = np.asarray(self.translations, dtype=np.float32)
        self.euler_angles = np.asarray(self.euler_angles, dtype=np.float32)

    @property
    def num_knots(self) -> int:
        """Number of spline knots."""
        return len(self.knot_times)

    @property
    def t_min(self) -> float:
        """Minimum time in the spline."""
        return float(self.knot_times[0])

    @property
    def t_max(self) -> float:
        """Maximum time in the spline."""
        return float(self.knot_times[-1])

    def pose_at(self, t: float) -> np.ndarray:
        """
        Evaluate the pose at time t.

        Args:
            t: Time to evaluate (will be clamped to [t_min, t_max])

        Returns:
            4x4 world_from_plane transform
        """
        t = np.clip(t, self.t_min, self.t_max)

        # Find the segment
        idx = np.searchsorted(self.knot_times, t, side="right") - 1
        idx = np.clip(idx, 0, self.num_knots - 2)

        # Linear interpolation (can upgrade to cubic if needed)
        t0 = self.knot_times[idx]
        t1 = self.knot_times[idx + 1]
        alpha = (t - t0) / (t1 - t0 + 1e-8)
        alpha = np.clip(alpha, 0, 1)

        # Interpolate translation
        trans = (1 - alpha) * self.translations[idx] + alpha * self.translations[idx + 1]

        # Interpolate Euler angles (simple lerp - can use SLERP for quaternions)
        euler = (1 - alpha) * self.euler_angles[idx] + alpha * self.euler_angles[idx + 1]

        return self._euler_to_matrix(trans, euler)

    def _euler_to_matrix(self, trans: np.ndarray, euler: np.ndarray) -> np.ndarray:
        """Convert translation + Euler angles to 4x4 matrix."""
        roll, pitch, yaw = euler

        # Rotation matrices for each axis
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Combined rotation (ZYX order)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr],
        ], dtype=np.float32)

        # Build 4x4 transform
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = trans

        return T

    def apply_offset(
        self,
        delta_translation: np.ndarray,
        delta_euler: np.ndarray,
    ) -> "PoseSplineParams":
        """
        Apply a global offset to the spline.

        Args:
            delta_translation: (3,) translation offset
            delta_euler: (3,) Euler angle offset

        Returns:
            New PoseSplineParams with offset applied
        """
        return PoseSplineParams(
            knot_times=self.knot_times.copy(),
            translations=self.translations + np.asarray(delta_translation),
            euler_angles=self.euler_angles + np.asarray(delta_euler),
        )

    @classmethod
    def create_static(
        cls,
        translation: Tuple[float, float, float],
        euler: Tuple[float, float, float] = (0, 0, 0),
        t_range: Tuple[float, float] = (0, 1),
    ) -> "PoseSplineParams":
        """Create a static (non-moving) pose spline."""
        return cls(
            knot_times=np.array([t_range[0], t_range[1]], dtype=np.float32),
            translations=np.tile(np.array(translation, dtype=np.float32), (2, 1)),
            euler_angles=np.tile(np.array(euler, dtype=np.float32), (2, 1)),
        )

    @classmethod
    def create_linear(
        cls,
        start_trans: Tuple[float, float, float],
        end_trans: Tuple[float, float, float],
        start_euler: Tuple[float, float, float] = (0, 0, 0),
        end_euler: Tuple[float, float, float] = (0, 0, 0),
        t_range: Tuple[float, float] = (0, 1),
    ) -> "PoseSplineParams":
        """Create a linear motion spline between two poses."""
        return cls(
            knot_times=np.array([t_range[0], t_range[1]], dtype=np.float32),
            translations=np.array([start_trans, end_trans], dtype=np.float32),
            euler_angles=np.array([start_euler, end_euler], dtype=np.float32),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "knot_times": self.knot_times.tolist(),
            "translations": self.translations.tolist(),
            "euler_angles": self.euler_angles.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseSplineParams":
        """Create from dictionary."""
        return cls(
            knot_times=np.array(data["knot_times"]),
            translations=np.array(data["translations"]),
            euler_angles=np.array(data["euler_angles"]),
        )


@dataclass
class NAGEditVector:
    """
    Structured representation of edits applied to a NAG scene.

    Used for logging and analysis in the economics pipeline.
    """
    node_id: NAGNodeId
    edit_type: str  # "texture", "pose", "duplicate", "remove"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": str(self.node_id),
            "edit_type": self.edit_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NAGEditVector":
        """Create from dictionary."""
        return cls(
            node_id=make_node_id(data["node_id"]),
            edit_type=data["edit_type"],
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", 0.0),
        )
